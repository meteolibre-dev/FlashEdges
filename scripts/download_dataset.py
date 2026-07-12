#!/usr/bin/env python3
"""
Robust resumable downloader for the `meteolibre-dev/global_sat_metar` dataset.

The dataset is ~3 TB across ~45k parquet files. A plain `hf download` saturates
the cluster egress quota and stalls every ~20 GB. This script instead downloads
files one at a time (optionally with low parallelism), retries each file with
exponential backoff when the quota kicks in, and never re-downloads a file that
is already present with the right size — so you can restart it as many times as
you want and it picks up exactly where it stopped.

Typical usage (runs for ~1-2 days, safe to interrupt & restart at any time):

    uv run python scripts/download_dataset.py

Options:

    uv run python scripts/download_dataset.py --workers 4      # mild parallelism
    uv run python scripts/download_dataset.py --max-backoff 1200
    uv run python scripts/download_dataset.py --refresh-list   # re-pull file list

Requires: huggingface_hub (already in pyproject.toml).
Set HF_TOKEN env var if the repo ever becomes gated.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:  # pragma: no cover
    sys.exit(
        "huggingface_hub is required. Install with:  uv sync   "
        "(it is already listed in pyproject.toml)"
    )

REPO_ID = "meteolibre-dev/global_sat_metar"
REPO_TYPE = "dataset"
DEFAULT_OUT = "data"
STATE_DIR = ".download_state"

# ---------------------------------------------------------------------------
# File listing (cached)
# ---------------------------------------------------------------------------

def list_repo_files(force_refresh: bool) -> list[dict]:
    """Return [{path, size}], cached to disk so restarts are instant & offline."""
    state_dir = Path(STATE_DIR)
    state_dir.mkdir(exist_ok=True)
    cache = state_dir / "file_list.json"

    if cache.exists() and not force_refresh:
        files = json.loads(cache.read_text())
        print(f"Loaded cached file list: {len(files):,} files ({cache})")
        return files

    print(f"Querying HuggingFace for full file list of {REPO_ID} ...")
    api = HfApi()
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=True)
    files = []
    for s in info.siblings:
        # Skip HF-internal pointer files; keep README for completeness.
        if s.rfilename == ".gitattributes":
            continue
        files.append({"path": s.rfilename, "size": int(s.size or 0)})

    files.sort(key=lambda f: f["path"])
    cache.write_text(json.dumps(files, indent=2))
    total_gb = sum(f["size"] for f in files) / 1e9
    print(f"  -> {len(files):,} files, total {total_gb:.1f} GB. Cached to {cache}")
    return files


# ---------------------------------------------------------------------------
# Resume / completion bookkeeping
# ---------------------------------------------------------------------------

class Progress:
    """Thread-safe counters persisted to disk for safe restarts."""

    def __init__(self, state_dir: Path):
        self.lock = threading.Lock()
        self.done_bytes = 0
        self.done_files = 0
        self.fail_files = 0
        self.start = time.time()
        self.path = state_dir / "progress.json"
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                self.done_bytes = d.get("done_bytes", 0)
                self.done_files = d.get("done_files", 0)
                self.fail_files = d.get("fail_files", 0)
            except Exception:
                pass

    def add(self, nbytes: int, ok: bool):
        with self.lock:
            if ok:
                self.done_files += 1
                self.done_bytes += nbytes
            else:
                self.fail_files += 1
            # Persist (cheap, only on completion).
            self.path.write_text(json.dumps({
                "done_bytes": self.done_bytes,
                "done_files": self.done_files,
                "fail_files": self.fail_files,
            }))

    def snapshot(self):
        with self.lock:
            return self.done_bytes, self.done_files, self.fail_files


def local_target(out_dir: Path, repo_path: str) -> Path:
    return out_dir / repo_path


def is_complete(out_dir: Path, file: dict) -> bool:
    """A file is complete iff it exists locally with the expected byte size."""
    if not file["size"]:
        return False
    p = local_target(out_dir, file["path"])
    return p.exists() and p.stat().st_size == file["size"]


# ---------------------------------------------------------------------------
# Per-file download with exponential backoff
# ---------------------------------------------------------------------------

# Network/transient errors that justify a retry.
def _is_transient(err: Exception) -> bool:
    s = str(err).lower()
    if isinstance(err, HfHubHTTPError):
        code = getattr(err, "response", None)
        code = getattr(code, "status_code", None) if code else None
        if code in (429, 500, 502, 503, 504):
            return True
    hints = (
        "429", "too many requests", "rate limit", "quota",
        "503", "502", "500", "service unavailable", "temporar",
        "timeout", "timed out", "connection reset", "connection aborted",
        "connection refused", "broken pipe", "chunked", "read timed out",
        "max retries", "retries exceeded", "eof occurred", "ssl",
        "incomplete read", "partial content",
    )
    return any(h in s for h in hints)


def download_one(
    file: dict,
    out_dir: Path,
    max_retries: int,
    base_backoff: float,
    max_backoff: float,
    delay: float,
    stop_event: threading.Event,
) -> tuple[bool, str]:
    """Download a single file. Returns (ok, message)."""
    repo_path = file["path"]
    size = file["size"]

    if is_complete(out_dir, file):
        return True, "skipped (already present)"

    attempt = 0
    while not stop_event.is_set():
        attempt += 1
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=repo_path,
                local_dir=str(out_dir),
                # Force a fresh etag check only every 30 days; existing complete
                # files are verified by size above so this stays cheap.
                etag_timeout=30,
            )
            # Verify size matches expectation.
            lp = local_target(out_dir, repo_path)
            if lp.exists() and (size == 0 or lp.stat().st_size == size):
                if delay:
                    time.sleep(delay)
                return True, "downloaded"
            # Size mismatch — treat as transient, retry.
            raise IOError(
                f"size mismatch: expected {size}, got "
                f"{lp.stat().st_size if lp.exists() else 'missing'}"
            )
        except KeyboardInterrupt:
            stop_event.set()
            raise
        except Exception as e:  # noqa: BLE001
            if attempt >= max_retries or not _is_transient(e):
                return False, f"FAILED after {attempt} attempts: {e}"
            # Exponential backoff with full jitter.
            backoff = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
            backoff = random.uniform(0.5 * backoff, backoff)
            print(
                f"    [{repo_path}] transient error (attempt {attempt}/"
                f"{max_retries}): {e.__class__.__name__}: {e}\n"
                f"    -> backing off {backoff:.0f}s and retrying...",
                flush=True,
            )
            # Sleep in small increments so Ctrl-C / stop stays responsive.
            slept = 0.0
            while slept < backoff:
                if stop_event.is_set():
                    break
                time.sleep(min(1.0, backoff - slept))
                slept += 1.0
    return False, "interrupted"


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def fmt_duration(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel download threads (default: 1 — safest against "
                         "the ~20 GB cluster quota; raise to 4-8 for speed).")
    ap.add_argument("--max-retries", type=int, default=1000,
                    help="Max retries per file before giving up (default: 1000). "
                         "Quota resets eventually, so high is fine.")
    ap.add_argument("--base-backoff", type=float, default=5.0,
                    help="Initial backoff seconds (default: 5).")
    ap.add_argument("--max-backoff", type=float, default=1800.0,
                    help="Max backoff seconds per retry (default: 1800 = 30 min). "
                         "Tuned for the ~20 GB cluster stall.")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="Pause seconds after each file to self-throttle (default: 0).")
    ap.add_argument("--refresh-list", action="store_true",
                    help="Re-query HuggingFace for the file list instead of using cache.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Download at most N files (debugging). 0 = all.")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir = Path(STATE_DIR)
    state_dir.mkdir(exist_ok=True)

    files = list_repo_files(force_refresh=args.refresh_list)
    total_bytes = sum(f["size"] for f in files)
    total_n = len(files)
    if args.limit:
        files = files[: args.limit]
        total_bytes = sum(f["size"] for f in files)

    # Partition into pending vs already-complete (no network needed).
    pending, done_n, done_bytes = [], 0, 0
    for f in files:
        if is_complete(out_dir, f):
            done_n += 1
            done_bytes += f["size"]
        else:
            pending.append(f)

    print()
    print("=" * 68)
    print(f"  Repo      : {REPO_ID}")
    print(f"  Output    : {out_dir}")
    print(f"  Files     : {total_n:,} total "
          f"({done_n:,} done, {len(pending):,} pending)")
    print(f"  Size      : {fmt_bytes(total_bytes)} total "
          f"({fmt_bytes(done_bytes)} present, "
          f"{fmt_bytes(total_bytes - done_bytes)} to go)")
    print(f"  Workers   : {args.workers}")
    print("=" * 68)
    print("  Safe to Ctrl-C at any time — re-run to resume.\n")

    if not pending:
        print("Everything is already downloaded. 🎉")
        return 0

    progress = Progress(state_dir)
    # Don't double-count files already on disk before this run.
    # (Progress persists cumulative stats; we only add newly-downloaded bytes.)
    stop_event = threading.Event()
    failures: list[str] = []
    last_report = time.time()

    def report(force: bool = False):
        nonlocal last_report
        now = time.time()
        if not force and now - last_report < 30:
            return
        last_report = now
        b, nf, nfail = progress.snapshot()
        session_elapsed = now - progress.start
        session_bytes = max(0, b - done_bytes)
        rate = session_bytes / session_elapsed if session_elapsed > 0 else 0
        remaining_bytes = (total_bytes - done_bytes) - session_bytes
        eta = remaining_bytes / rate if rate > 1 else float("inf")
        print(
            f"  [{nf + done_n:,}/{total_n:,}] "
            f"{fmt_bytes(done_bytes + session_bytes)} / {fmt_bytes(total_bytes)} "
            f"| {fmt_bytes(rate)}/s "
            f"| ETA {fmt_duration(eta) if eta != float('inf') else '?'} "
            f"| fails={nfail}",
            flush=True,
        )

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    download_one, f, out_dir,
                    args.max_retries, args.base_backoff,
                    args.max_backoff, args.delay, stop_event,
                ): f
                for f in pending
            }
            for fut in as_completed(futures):
                if stop_event.is_set():
                    break
                f = futures[fut]
                try:
                    ok, msg = fut.result()
                except KeyboardInterrupt:
                    stop_event.set()
                    break
                except Exception as e:  # noqa: BLE001
                    ok, msg = False, f"crashed: {e}"
                progress.add(f["size"] if ok else 0, ok=ok)
                if not ok:
                    failures.append(f["path"])
                    print(f"  ✗ {f['path']}: {msg}", flush=True)
                report()
    except KeyboardInterrupt:
        print("\nInterrupted — stopping workers. State saved; re-run to resume.")
        stop_event.set()
        return 130
    finally:
        report(force=True)
        fail_log = state_dir / "failures.txt"
        fail_log.write_text("\n".join(failures))

    print()
    b, nf, nfail = progress.snapshot()
    if failures:
        print(f"Completed with {len(failures)} file(s) still failing.")
        print(f"  See {fail_log}. Re-run this script to retry them.")
        return 1

    # Final integrity sweep — verify every expected file is present & sized.
    print("Running final integrity check ...")
    missing = [f["path"] for f in files if not is_complete(out_dir, f)]
    if missing:
        print(f"  {len(missing):,} file(s) missing/mis-sized. Re-run to fetch them.")
        (state_dir / "failures.txt").write_text("\n".join(missing))
        return 1

    print("All files present and verified. ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
