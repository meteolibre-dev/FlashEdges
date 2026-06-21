"""
Local throughput benchmark for ``FlashEdgesStreamingDataset`` over local parquet.

It measures samples/sec of the real ``torch.utils.data.DataLoader`` pipeline
(parquet decode -> ``preprocess_record`` -> optional shuffle buffer -> optional
prefetch) on the local filesystem, so the numbers reflect CPU/preprocess cost
and worker parallelism -- not network.

Two things this benchmark is designed to surface:

1. **Multi-worker scaling.**  The dataset shards files across DataLoader
   workers itself (``_shard``) and disables ``datasets``' built-in torch-worker
   sharding (``_no_torch_worker_sharding``) so every worker streams its own
   files in full.  If that were broken, throughput would *not* scale with
   ``num_workers`` (only worker 0 would produce).  A near-linear speedup here is
   the end-to-end proof the sharding works.

2. **Full-coverage invariant.**  The total number of samples produced by a
   multi-worker sweep over a fixed file set equals what a single worker
   produces over the same set -- i.e. no row is dropped or double-counted.

Run standalone::

    uv run python -m tests.benchmark_dataloader
    uv run python -m tests.benchmark_dataloader --samples 1024 --workers 1 2 4 8
    uv run python -m tests.benchmark_dataloader --data_glob 'data/2021-07-14_*.parquet'

Run under pytest (loose perf/correctness smoke checks; install pytest first)::

    uv run pip install pytest
    uv run pytest tests/benchmark_dataloader.py -s
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from meteolibre_model.dataset.dataset_global_satellite_streaming import (
    FlashEdgesStreamingDataset,
)

# Default local parquet glob.  The repo ships ``data/`` as a symlink to the
# generator's output; ``data/2021-07-14_*.parquet`` matches many files so every
# worker gets a non-trivial slice even at high num_workers.
DEFAULT_DATA_GLOB = "data/2021-07-14_*.parquet"


def _resolve_glob(data_glob: str) -> str:
    """Return ``data_glob`` if it matches files, else fall back to a sane default."""
    if glob.glob(data_glob):
        return data_glob
    # Try a couple of common local layouts so the benchmark is portable.
    for cand in (DEFAULT_DATA_GLOB, "data/*.parquet", "*.parquet"):
        if glob.glob(cand):
            print(f"[benchmark] '{data_glob}' matched nothing; using '{cand}'")
            return cand
    raise FileNotFoundError(
        f"No local .parquet found for '{data_glob}' (or the fallback globs). "
        "Point --data_glob at a local parquet directory."
    )


def bench_config(
    data_glob: str,
    num_workers: int,
    batch_size: int = 32,
    prefetch_rows: int = 8,
    shuffle_buffer: int = 1000,
    n_samples: int = 256,
    warmup: int = 32,
) -> dict:
    """Time consuming ``n_samples`` from a real DataLoader over local parquet.

    A ``warmup`` worth of samples is consumed (uncached, fills the prefetch
    queue) before the timed loop so the steady-state rate is measured.

    Returns a dict with throughput and run details. The dataloader is created
    fresh per call (``persistent_workers=False``) so configs are independent.
    """
    ds = FlashEdgesStreamingDataset(
        data_files=data_glob,
        shuffle_buffer=shuffle_buffer,
        prefetch_rows=prefetch_rows,
        nb_temporal=7,
        precip_to_dbz=True,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,  # no accelerator on CI / CPU boxes
        persistent_workers=False,
    )

    it = iter(dl)
    n_warm = 0
    for batch in it:
        n_warm += len(batch["sat_patch_data"])
        if n_warm >= warmup:
            break

    t0 = time.perf_counter()
    n = 0
    for batch in it:
        n += len(batch["sat_patch_data"])
        if n >= n_samples:
            break
    dt = time.perf_counter() - t0

    samples_per_sec = n / dt if dt > 0 else float("inf")
    return {
        "num_workers": num_workers,
        "batch_size": batch_size,
        "prefetch_rows": prefetch_rows,
        "shuffle_buffer": shuffle_buffer,
        "n_samples": n,
        "dt_s": dt,
        "samples_per_sec": samples_per_sec,
        "batches_per_sec": (n / batch_size) / dt if dt > 0 else float("inf"),
        "ms_per_sample": dt / n * 1000 if n > 0 else float("inf"),
    }



def _print_row(r: dict, baseline: float | None = None) -> None:
    speedup = (r["samples_per_sec"] / baseline) if baseline else float("nan")
    print(
        f"  W={r['num_workers']:<2d} B={r['batch_size']:<3d} "
        f"pf={r['prefetch_rows']:<3d} buf={r['shuffle_buffer']:<5d} | "
        f"{r['samples_per_sec']:7.1f} samp/s  "
        f"{r['batches_per_sec']:6.2f} batch/s  "
        f"{r['ms_per_sample']:6.2f} ms/samp  "
        f"(n={r['n_samples']}, {r['dt_s']:5.2f}s)"
        + (f"  {speedup:4.2f}x" if baseline else "")
    )


def run_sweep(
    data_glob: str,
    workers: list[int],
    batch_size: int,
    prefetch_rows: int,
    shuffle_buffer: int,
    n_samples: int,
) -> list[dict]:
    """Run the multi-worker scaling sweep and print a table. Returns results."""
    print(
        f"\n=== DataLoader throughput (local parquet: '{data_glob}', "
        f"{len(glob.glob(data_glob))} files, n_samples={n_samples}) ==="
    )
    print(
        "  config                                           | throughput"
    )
    print("  " + "-" * 92)

    results: list[dict] = []
    baseline = None
    for nw in workers:
        r = bench_config(
            data_glob,
            num_workers=nw,
            batch_size=batch_size,
            prefetch_rows=prefetch_rows,
            shuffle_buffer=shuffle_buffer,
            n_samples=n_samples,
        )
        if baseline is None:
            baseline = r["samples_per_sec"]
        _print_row(r, baseline)
        results.append(r)
    return results


def run_prefetch_sweep(
    data_glob: str, num_workers: int, batch_size: int, n_samples: int
) -> None:
    print(
        f"\n=== Prefetch effect (workers={num_workers}, batch={batch_size}, "
        f"n_samples={n_samples}) ==="
    )
    for pf in (0, 4, 8):
        r = bench_config(
            data_glob,
            num_workers=num_workers,
            batch_size=batch_size,
            prefetch_rows=pf,
            shuffle_buffer=1,  # isolate prefetch from shuffle cost
            n_samples=n_samples,
        )
        _print_row(r)


def run_shuffle_sweep(
    data_glob: str, num_workers: int, batch_size: int, n_samples: int
) -> None:
    print(
        f"\n=== Shuffle-buffer effect (workers={num_workers}, batch={batch_size}, "
        f"n_samples={n_samples}) ==="
    )
    for buf in (1, 100):
        r = bench_config(
            data_glob,
            num_workers=num_workers,
            batch_size=batch_size,
            prefetch_rows=8,
            shuffle_buffer=buf,
            n_samples=n_samples,
        )
        _print_row(r)


def run_full_coverage_check(data_glob: str, batch_size: int = 16) -> None:
    print(f"\n=== Full-coverage invariant (local parquet: '{data_glob}') ===")
    # Use a bounded slice of files so this can't accidentally drain a huge
    # directory and OOM the box. 8 files is enough to exercise multi-worker
    # file sharding (every worker gets >=1 file) while staying cheap.
    sample_files = sorted(glob.glob(data_glob))[:8]
    if len(sample_files) < 2:
        print("  skipped: need >=2 files to test sharding")
        return
    t0 = time.perf_counter()
    single = count_total_bounded(sample_files, num_workers=1, batch_size=batch_size)
    t1 = time.perf_counter()
    multi = count_total_bounded(sample_files, num_workers=2, batch_size=batch_size)
    t2 = time.perf_counter()
    status = "OK" if single == multi else "MISMATCH"
    print(
        f"  files={len(sample_files)}  single-worker total = {single}  ({t1 - t0:5.2f}s)\n"
        f"  files={len(sample_files)}  multi-worker  total = {multi}  ({t2 - t1:5.2f}s)\n"
        f"  -> {status}"
        + ("" if single == multi else "  (rows dropped / double-counted!)")
    )


def count_total_bounded(files: list[str], num_workers: int, batch_size: int = 16) -> int:
    """Drain a bounded explicit file list and return total sample count.

    Accepts a literal list of paths (instead of a glob) so the coverage check
    can run on a small, fixed slice of files regardless of how big the dataset
    directory is -- keeping it safe on small machines. Symlinks the chosen
    files into a temp dir and globs that.
    """
    import tempfile
    import shutil
    import os as _os

    tmp = tempfile.mkdtemp(prefix="flashedges_bench_")
    try:
        for f in files:
            _os.symlink(_os.path.abspath(f), _os.path.join(tmp, _os.path.basename(f)))
        ds = FlashEdgesStreamingDataset(
            data_files=_os.path.join(tmp, "*.parquet"),
            shuffle_buffer=1,
            prefetch_rows=0,
            nb_temporal=7,
            precip_to_dbz=True,
        )
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        n = 0
        for batch in dl:
            n += len(batch["sat_patch_data"])
        return n
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local throughput benchmark for FlashEdgesStreamingDataset."
    )
    parser.add_argument(
        "--data_glob",
        type=str,
        default=DEFAULT_DATA_GLOB,
        help=f"Local parquet glob (recursive ** supported). Default '{DEFAULT_DATA_GLOB}'.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=None,
        help="num_workers values for the scaling sweep. Default: 1,2 (conservative; "
        "tune up on a big box).",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--prefetch_rows", type=int, default=4)
    parser.add_argument("--shuffle_buffer", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256, help="Samples per config.")
    parser.add_argument(
        "--skip",
        choices=["scaling", "prefetch", "shuffle", "coverage"],
        nargs="*",
        default=[],
        help="Sections to skip.",
    )
    args = parser.parse_args(argv)

    torch.set_num_threads(1)  # deterministic: 1 BLAS thread per process

    # Cap total concurrency conservatively: num_workers forked processes, each
    # running the datasets parquet builder + prefetch thread + preprocess.  We
    # keep it tiny by default so the benchmark can't accidentally OOM/hang a
    # small box.  Override with --workers.
    data_glob = _resolve_glob(args.data_glob)
    cpu = os.cpu_count() or 1
    workers = args.workers or [w for w in (1, 2) if w <= cpu]
    if not workers:
        workers = [1]
    if max(workers) > cpu:
        print(f"[benchmark] warning: max workers {max(workers)} > cpu {cpu}; "
              f"this may thrash.")

    print(f"cpu_count={cpu} | torch threads/process={torch.get_num_threads()}")

    if "scaling" not in args.skip:
        run_sweep(
            data_glob, workers, args.batch_size, args.prefetch_rows,
            args.shuffle_buffer, args.samples,
        )
    if "prefetch" not in args.skip:
        run_prefetch_sweep(data_glob, workers[0], args.batch_size, args.samples)
    if "shuffle" not in args.skip:
        run_shuffle_sweep(data_glob, workers[0], args.batch_size, args.samples)
    if "coverage" not in args.skip:
        run_full_coverage_check(data_glob, args.batch_size)

    return 0


# --------------------------------------------------------------------------- #
# pytest entry points (loose perf + correctness smoke checks)
# --------------------------------------------------------------------------- #
def _has_local_parquet() -> bool:
    for cand in (DEFAULT_DATA_GLOB, "data/*.parquet", "*.parquet"):
        if glob.glob(cand):
            return True
    return False


def test_full_coverage_single_vs_multi():
    """Multi-worker drain == single-worker drain for the same files.

    The core correctness invariant for the manual file sharding: no row is
    dropped (a worker assigned no shard) or double-counted.
    """
    if not _has_local_parquet():
        import pytest

        pytest.skip("no local parquet found for the coverage check")
    g = _resolve_glob(DEFAULT_DATA_GLOB)
    files = sorted(glob.glob(g))[:8]
    single = count_total_bounded(files, num_workers=1)
    multi = count_total_bounded(files, num_workers=2)
    assert single == multi, f"full-coverage mismatch: single={single} multi={multi}"


def test_multiworker_scales():
    """All workers contribute, and throughput doesn't collapse vs single worker.

    The strong correctness signal that file sharding works is the full-coverage
    invariant (see ``test_full_coverage_single_vs_multi``): if only worker 0
    were producing, the multi-worker total would be a fraction of the
    single-worker total. Here we additionally check that steady-state
    multi-worker throughput is not catastrophically worse than single-worker
    (a >2x slowdown would indicate workers are starved/broken).

    Note: on a bandwidth-bound box (small local files, single worker already
    near memory/disk BW) extra workers may NOT speed things up -- that's fine
    and expected. The real scaling benefit shows on remote streaming where each
    worker has independent network I/O.
    """
    if not _has_local_parquet():
        import pytest

        pytest.skip("no local parquet found for the scaling check")
    g = _resolve_glob(DEFAULT_DATA_GLOB)
    r1 = bench_config(g, num_workers=1, n_samples=192)
    r2 = bench_config(g, num_workers=2, n_samples=192)
    ratio = r2["samples_per_sec"] / r1["samples_per_sec"]
    print(f"\n  single: {r1['samples_per_sec']:.1f} samp/s")
    print(f"  2-wkr : {r2['samples_per_sec']:.1f} samp/s ({ratio:.2f}x)")
    # Not a hard speedup gate (bandwidth-bound locally): just guard against a
    # collapse that would indicate broken/starved workers.
    assert ratio > 0.5, (
        "2-worker throughput collapsed below 0.5x single-worker; "
        "file sharding across workers may be broken (workers idle/starved?)."
    )


def test_single_worker_smoke():
    """Single-worker DataLoader produces well-formed batches above a floor rate."""
    if not _has_local_parquet():
        import pytest

        pytest.skip("no local parquet found for the smoke check")
    g = _resolve_glob(DEFAULT_DATA_GLOB)
    r = bench_config(g, num_workers=1, batch_size=16, n_samples=128)
    assert r["n_samples"] >= 128
    assert r["samples_per_sec"] > 0
    print(f"\n  single-worker smoke: {r['samples_per_sec']:.1f} samp/s")


if __name__ == "__main__":
    raise SystemExit(main())
