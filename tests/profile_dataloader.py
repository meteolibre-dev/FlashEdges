"""
Stage-by-stage profiling for FlashEdgesStreamingDataset.

Breaks the local streaming pipeline into measurable stages so we can see where
the per-sample time actually goes:

  S1  raw_row   : datasets parquet streaming iteration (HTTP/disk read + arrow
                  decode into a python dict), NO preprocessing.
  S2  preprocess: preprocess_record() on an already-fetched raw row (np.frombuffer,
                  astype, dBZ, where, concatenate, suncalc x2).
  S3  full      : the real single-worker loop = S1 + S2.

All runs are single-process (no DataLoader workers) and bounded by N samples so
the machine can't be overwhelmed. Set TORCH_NUM_THREADS=1 implicitly.

Run:
    uv run python -m tests.profile_dataloader
    uv run python -m tests.profile_dataloader --samples 200 --data_glob 'data/2021-07-14_*.parquet'
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from meteolibre_model.dataset.dataset_global_satellite_metar import (
    preprocess_record,
    resolve_date,
)
from meteolibre_model.dataset.dataset_global_satellite_streaming import (
    FlashEdgesStreamingDataset,
    _no_torch_worker_sharding,
)

DEFAULT_DATA_GLOB = "data/2021-07-14_*.parquet"


def _resolve_glob(data_glob: str) -> str:
    if glob.glob(data_glob):
        return data_glob
    for cand in (DEFAULT_DATA_GLOB, "data/*.parquet", "*.parquet"):
        if glob.glob(cand):
            return cand
    raise FileNotFoundError(f"No local parquet for '{data_glob}'.")


def _iter_raw_records(data_glob: str, n: int):
    """Yield up to n raw rows from datasets parquet streaming (no preprocess)."""
    from datasets import load_dataset

    files = sorted(glob.glob(data_glob))
    with _no_torch_worker_sharding():
        ds = load_dataset(
            "parquet", data_files={"train": files}, split="train", streaming=True
        )
        for i, rec in enumerate(ds):
            if i >= n:
                break
            yield rec


def time_stage_raw(data_glob: str, n: int) -> float:
    """S1: time fetching n raw rows (parquet decode -> dict), no preprocess."""
    t0 = time.perf_counter()
    c = 0
    for rec in _iter_raw_records(data_glob, n):
        c += 1
    return time.perf_counter() - t0


def time_stage_preprocess(data_glob: str, n: int) -> float:
    """S2: time preprocess_record over n rows (rows pre-fetched, decode cost only)."""
    records = list(_iter_raw_records(data_glob, n))
    t0 = time.perf_counter()
    for rec in records:
        date = resolve_date(rec)
        preprocess_record(date, rec, nb_temporal=7, precip_to_dbz=True)
    return time.perf_counter() - t0


def time_stage_preprocess_components(data_glob: str, n: int) -> dict:
    """S2 breakdown: where inside preprocess_record does the time go?"""
    records = list(_iter_raw_records(data_glob, n))
    timings = {
        "frombuffer_decode": 0.0,
        "astype_copy": 0.0,
        "dbz": 0.0,
        "where_mask": 0.0,
        "concat": 0.0,
        "suncalc": 0.0,
    }
    from meteolibre_model.dataset.dataset_global_satellite_metar import (
        mmh_to_dbz, METAR_PRECIP_IDX, ELEVATION_FLOOR,
    )
    from suncalc import get_position

    for rec in records:
        # --- frombuffer + reshape ---
        t = time.perf_counter()
        sat_patch = np.frombuffer(rec["sat_data"], dtype=rec["sat_dtype"]).reshape(rec["sat_shape"])
        elev = np.frombuffer(rec["elevation_data"], dtype=rec["elevation_dtype"]).reshape(rec["elevation_shape"])
        metar_patch = np.frombuffer(rec["metar_data"], dtype=rec["metar_dtype"]).reshape(rec["metar_shape"])
        timings["frombuffer_decode"] += time.perf_counter() - t

        # --- astype float32 (copies) ---
        t = time.perf_counter()
        sat_f = sat_patch.astype(np.float32)
        elev_f = elev.astype(np.float32)
        metar_f = metar_patch.astype(np.float32)
        timings["astype_copy"] += time.perf_counter() - t

        # --- dBZ conversion ---
        t = time.perf_counter()
        metar_dbz = metar_f.copy()
        metar_dbz[:, METAR_PRECIP_IDX] = mmh_to_dbz(metar_f[:, METAR_PRECIP_IDX])
        timings["dbz"] += time.perf_counter() - t

        # --- mask + where (the -10000 sentinel fill) ---
        t = time.perf_counter()
        mask = (~np.isnan(metar_dbz)).astype(np.float32)
        out = np.where(np.isnan(metar_dbz), -10000.0, metar_dbz)
        timings["where_mask"] += time.perf_counter() - t

        # --- concatenate (sat + elevation stacked) ---
        t = time.perf_counter()
        elev_b = np.where(elev_f < 0, ELEVATION_FLOOR, elev_f)
        elev_b = elev_b[None, None, :, :].repeat(sat_f.shape[0], axis=0)
        cat = np.concatenate([sat_f, elev_b], axis=1)
        timings["concat"] += time.perf_counter() - t

        # --- suncalc (2 calls) ---
        t = time.perf_counter()
        date = resolve_date(rec)
        lon, lat = float(rec["lon"]), float(rec["lat"])
        _ = get_position(date, lon, lat)
        date_noon = date.replace(hour=12, minute=0, second=0, microsecond=0)
        _ = get_position(date_noon, lon, lat)
        timings["suncalc"] += time.perf_counter() - t
    return timings


def time_stage_full(data_glob: str, n: int, prefetch: int) -> float:
    """S3: full single-process loop over the dataset, prefetch on/off."""
    ds = FlashEdgesStreamingDataset(
        data_files=data_glob, shuffle_buffer=1, prefetch_rows=prefetch,
        nb_temporal=7, precip_to_dbz=True,
    )
    t0 = time.perf_counter()
    c = 0
    for _ in ds:
        c += 1
        if c >= n:
            break
    return time.perf_counter() - t0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage profiler for FlashEdgesStreamingDataset.")
    p.add_argument("--data_glob", type=str, default=DEFAULT_DATA_GLOB)
    p.add_argument("--samples", type=int, default=150, help="Samples per stage.")
    args = p.parse_args(argv)

    n = args.samples
    data_glob = _resolve_glob(args.data_glob)
    nfiles = len(glob.glob(data_glob))
    print(f"\nFlashEdgesStreamingDataset stage profile")
    print(f"  data_glob = '{data_glob}' ({nfiles} files), n_samples = {n}, single process\n")

    # sample tensor size (for bandwidth calc)
    ds0 = FlashEdgesStreamingDataset(data_files=data_glob, shuffle_buffer=1, prefetch_rows=0)
    s = next(iter(ds0))
    bytes_per_sample = sum(
        v.element_size() * v.nelement() for v in s.values() if hasattr(v, "element_size")
    )
    print(f"  decoded sample size = {bytes_per_sample/1e6:.2f} MB\n")

    t_raw = time_stage_raw(data_glob, n)
    t_pre = time_stage_preprocess(data_glob, n)
    t_full_pf0 = time_stage_full(data_glob, n, prefetch=0)
    t_full_pf8 = time_stage_full(data_glob, n, prefetch=8)

    def line(stage, t):
        ms = t / n * 1000
        sps = n / t
        bw = bytes_per_sample * n / t / 1e6
        print(f"  {stage:32s} {ms:6.2f} ms/samp  {sps:7.1f} samp/s  {bw:7.1f} MB/s")
        return ms

    print("  " + "-" * 70)
    m_raw = line("S1  raw row (parquet decode)", t_raw)
    m_pre = line("S2  preprocess_record only", t_pre)
    print("  " + "-" * 70)
    m_full0 = line("S3  full loop  (prefetch=0)", t_full_pf0)
    m_full8 = line("S3  full loop  (prefetch=8)", t_full_pf8)
    print("  " + "-" * 70)

    print(f"\n  bottleneck: preprocess is {m_pre/(m_raw+m_pre)*100:.0f}% of (raw+preprocess); "
          f"raw decode is {m_raw/(m_raw+m_pre)*100:.0f}%.")
    print(f"  prefetch speedup: {m_full0/m_full8:.2f}x (hides I/O behind decode).")

    print("\n  preprocess_record internals breakdown:")
    comp = time_stage_preprocess_components(data_glob, min(n, 100))
    total_comp = sum(comp.values())
    for k, v in comp.items():
        print(f"    {k:18s} {v/min(n,100)*1000:6.2f} ms/samp  "
              f"({v/total_comp*100:4.1f}% of preprocess)")

    # training-keep-up projection
    print("\n  training projection (batch_size=32):")
    for ms in (m_full8,):
        ms_per_batch = ms * 32
        print(f"    ~{ms_per_batch:6.0f} ms/batch on dataloader (1 worker, local)")
        print(f"    -> keeps up with a GPU step > {ms_per_batch:.0f} ms/batch; "
              f"bottleneck below that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
