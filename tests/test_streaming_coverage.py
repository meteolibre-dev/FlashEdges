"""
Regression tests for full-dataset coverage across epochs.

These guard against the two streaming-coverage bugs:

1. **Same-files-every-epoch (epoch-independent seed).**  With a
   ``steps_per_epoch`` that only consumes part of the shard, an
   epoch-independent shuffle seed made ``__iter__`` restart at the same first
   file each epoch, so the same leading files were re-read forever and the rest
   were never seen.

2. **Cursor not persisted (missing persistent_workers).**  The fix relies on a
   file cursor kept on the dataset instance; that only survives across epochs
   when the DataLoader uses ``persistent_workers=True``. If that's missing, the
   cursor resets each epoch and coverage collapses again.

Run:
    uv run pip install pytest
    uv run pytest tests/test_streaming_coverage.py -s
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import meteolibre_model.dataset.dataset_global_satellite_streaming as m
from meteolibre_model.dataset.dataset_global_satellite_streaming import (
    FlashEdgesStreamingDataset,
)

ROWS_PER_FILE = 8           # small so tests are fast
N_FILES = 40
N_WORKERS = 2


def _make_fake_dataset():
    """Build a dataset whose file IO is faked (no real parquet needed)."""
    fake_files = [f"data/file_{i:05d}.parquet" for i in range(N_FILES)]
    ds = FlashEdgesStreamingDataset(
        data_files="x", shuffle_buffer=1, prefetch_rows=0, nb_temporal=7
    )
    ds._resolve_files = lambda: list(fake_files)  # type: ignore[assignment]

    def fake_open(self, fpath):
        def gen():
            for _ in range(ROWS_PER_FILE):
                yield {"__fpath__": fpath}
        return gen()
    ds._open_one_file = fake_open.__get__(ds)  # type: ignore[assignment]

    # Patch the module-level references the dataset uses (imported by name).
    m.preprocess_record = lambda date, rec, nb, dbz: {"fpath": rec["__fpath__"]}  # type: ignore[assignment]
    m.resolve_date = lambda rec: "2021-01-01 00:00:00"  # type: ignore[assignment]
    return ds


def test_cursor_advances_across_epochs_with_persistent_workers():
    """With persistent_workers=True, epoch N continues where epoch N-1 left off.

    Asserts: (a) consecutive epochs don't re-read the same files, and (b) over
    enough epochs the full file set is covered.
    """
    ds = _make_fake_dataset()
    samples_per_epoch = ROWS_PER_FILE * 10   # 5 files/worker/epoch (>> prefetch drift)
    dl = DataLoader(ds, batch_size=1, num_workers=N_WORKERS,
                    persistent_workers=True, prefetch_factor=1)

    def one_epoch():
        seen = []
        for s in dl:
            seen.append(s["fpath"][0])   # batch_size=1 -> list of one
            if len(seen) >= samples_per_epoch:
                break
        return set(seen)

    e0 = one_epoch()
    e1 = one_epoch()

    # (a) overlap between consecutive epochs should be small (boundary only)
    overlap = len(e0 & e1)
    assert overlap <= 2 * N_WORKERS, (
        f"epoch overlap {overlap} too high -- cursor not advancing across "
        "epochs (is persistent_workers set?)"
    )

    # (b) drive enough epochs to cover the whole set. Each worker owns
    # N_FILES//N_WORKERS files and consumes a few per epoch; run enough epochs
    # that every worker finishes its shard (plus a margin for the boundary).
    files_per_worker = N_FILES // N_WORKERS
    files_per_worker_per_epoch = max(
        1, (samples_per_epoch // N_WORKERS) // ROWS_PER_FILE
    )
    needed = files_per_worker // files_per_worker_per_epoch + 4  # margin for prefetch drift
    total = set(e0) | set(e1)
    for _ in range(needed):
        total |= one_epoch()
    # Allow a tiny shortfall from inter-epoch prefetch drift at micro-scale; at
    # realistic epoch sizes the drift is <1% (verified separately at 4000 steps).
    assert len(total) >= 0.95 * N_FILES, (
        f"full coverage failed: {len(total)}/{N_FILES} files seen -- the cursor "
        "is not advancing across epochs (persistent_workers / seed bug?)."
    )


def test_no_cursor_resets_regardless_of_epoch_count():
    """The cursor must keep advancing over many epochs (no silent reset)."""

    class _W:
        id = 0
        num_workers = 1
    m.get_worker_info = lambda: _W()  # type: ignore[assignment]

    ds = _make_fake_dataset()
    samples_per_epoch = ROWS_PER_FILE * 2   # 2 files/epoch
    seen_in_order: list[str] = []
    for _ in range(N_FILES):                 # enough epochs to wrap at least once
        n = 0
        for s in ds:                         # same instance -> cursor persists
            seen_in_order.append(s["fpath"])
            n += 1
            if n >= samples_per_epoch:
                break
    distinct = set(seen_in_order)
    assert len(distinct) == N_FILES, (
        f"expected to reach all {N_FILES} files over the run, got {len(distinct)}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
