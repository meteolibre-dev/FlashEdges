"""
Streaming Torch IterableDataset for the FlashEdges global satellite + METAR
parquet dataset hosted on Hugging Face.

Use this for remote training (Jean Zay, Vast.ai, ...) where you do not want to
pre-download the full dataset locally.  It streams parquet row-groups directly
from the Hub via ``datasets.load_dataset(..., streaming=True)`` and applies the
exact same per-row preprocessing (``preprocess_record``) as the map-style
``FlashEdgesGlobalDataset`` — so the two are interchangeable from the model's
perspective.

Hub folder layout
-----------------
The dataset generator writes parquet into multiple subfolders (``data/``,
``data_2022_02/``, ``data_2022_05/``, ...) because of HF's 10k-files-per-folder
limit.  This loader does NOT rely on HF's path-based split inference (which is
unreliable for non-``train``-named folders).  Instead it enumerates every
``.parquet`` file in the repo via ``HfApi().list_repo_files`` and feeds them as
explicit ``hf://datasets/...`` URLs to the parquet builder, so all subfolders
are combined into one ``train`` stream regardless of their names.  Pass
``data_dir="data_2022_02"`` to train on a single subfolder subset.

Prefetching
-----------
Two layers compose to hide network latency behind GPU compute:

1. **In-dataset row prefetch** (``prefetch_rows``, default 8): a background
   thread inside each worker pulls upcoming raw rows from the HF stream into a
   bounded queue.  This is where the actual network I/O lives (one HTTP range
   request per parquet row-group); the DataLoader's own prefetch can't see
   inside the streaming iterator, so this layer is essential.  Mirrors
   ``torchdata.IterableWrapper.prefetch`` but is dependency-free (torchdata is
   archived).
2. **DataLoader batch prefetch** (``prefetch_factor``, default 2 when
   ``num_workers>0``): each worker process prepares batches ahead.  This hides
   the row→batch collation and transfer latency.  No code needed — it's the
   PyTorch default.

Key differences vs the map-style dataset
----------------------------------------
* **No random access / no ``__len__``.**  Use ``steps_per_epoch`` in the
  training loop instead of ``len(dataloader)``.
* **Shuffle buffer, not true shuffle.**  Rows arrive in parquet file order
  (roughly chronological because the generator names files by timestamp).  A
  buffer of ``shuffle_buffer`` rows is maintained and sampled randomly; larger
  buffers decorrelate better but cost RAM.  Each row is ~4 MB, so a 1000-row
  buffer is ~4 GB — fine on H100 nodes, watch it on small instances.
* **Worker sharding.**  ``datasets`` IterableDataset natively shards across
  DataLoader workers and distributed ranks when ``num_workers>0`` / distributed
  is initialised, so every worker/rank sees a disjoint slice of the stream.
* **Local cache.**  ``datasets`` caches downloaded parquet row-groups under
  ``~/.cache/huggingface``; the second epoch reads mostly from local disk.

Usage
-----
    from meteolibre_model.dataset.dataset_global_satellite_streaming import (
        FlashEdgesStreamingDataset,
    )
    ds = FlashEdgesStreamingDataset(
        hf_dataset_repo="meteolibre-dev/<your-flashedges-dataset>",
        split="train",
        shuffle_buffer=1000,
        prefetch_rows=8,
        precip_to_dbz=True,
        nb_temporal=7,
    )
    dl = DataLoader(ds, batch_size=32, num_workers=4, pin_memory=True)

For local testing without hitting the network, point ``data_files`` at local
parquet files (recursive glob so dated subfolders are included):

    ds = FlashEdgesStreamingDataset(
        hf_dataset_repo=None,
        data_files="data/**/*.parquet",
        ...
    )
"""

import contextlib
import random
import threading
from collections import deque

import torch
import torch.distributed as dist
import torch.utils.data
from torch.utils.data import IterableDataset, get_worker_info

from meteolibre_model.dataset.dataset_global_satellite_metar import (
    preprocess_record,
    resolve_date,
)


def _list_hub_parquet_files(hf_dataset_repo: str, data_dir) -> list:
    """Enumerate every ``.parquet`` file in a HF dataset repo.

    Robust to the multi-folder layout the generator produces (``data/``,
    ``data_2022_02/``, ``data_2022_05/``, ...) because it does NOT rely on HF's
    split-folder inference.  ``data_dir`` optionally restricts to one subfolder
    (e.g. ``"data_2022_02"``) so you can train on a subset.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = f"{data_dir}/" if data_dir else ""
    paths = api.list_repo_files(hf_dataset_repo, repo_type="dataset")
    parquet = sorted(
        p
        for p in paths
        if p.endswith(".parquet") and (not prefix or p.startswith(prefix))
    )
    if not parquet:
        raise FileNotFoundError(
            f"No .parquet files found in '{hf_dataset_repo}'"
            + (f" under '{data_dir}'" if data_dir else "")
        )
    return parquet


class _PrefetchIter:
    """Background-thread prefetcher over any iterator.

    Mirrors ``torchdata.IterableWrapper.prefetch(n)`` but is dependency-free
    (torchdata is archived).  A daemon thread pulls items from ``source`` into a
    bounded queue; ``__next__`` drains the queue.  This overlaps the network
    I/O of upcoming rows with the consumer's GPU compute, hiding the per-row
    latency of HF streaming (one HTTP range request per parquet row-group).

    Exceptions raised in the producer thread are re-raised on the consumer side
    (StopIteration is treated as end-of-stream, not an error).
    """

    def __init__(self, source, n: int):
        import queue

        self.q = queue.Queue(maxsize=max(n, 1))
        self._sentinel = object()
        self._exc = None
        self._thread = threading.Thread(
            target=self._produce, args=(source,), daemon=True
        )
        self._thread.start()

    def _produce(self, source):
        try:
            for item in source:
                self.q.put(item)
        except StopIteration:
            pass  # normal end of stream
        except Exception as e:  # noqa: BLE001 - propagate to consumer
            self._exc = e
        finally:
            self.q.put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        if self._exc is not None:
            e, self._exc = self._exc, None
            raise e
        item = self.q.get()
        if item is self._sentinel:
            raise StopIteration
        return item


@contextlib.contextmanager
def _no_torch_worker_sharding():
    """Make HuggingFace ``datasets`` skip its torch-DataLoader worker sharding.

    We shard files across torch DataLoader workers / distributed ranks ourselves
    (``FlashEdgesStreamingDataset._shard``), so when we iterate a *single-file*
    ``datasets.IterableDataset`` inside a torch worker we want every row of that
    file -- not 1/num_workers of them.

    ``datasets`` decides whether to shard by calling
    ``torch.utils.data.get_worker_info()``: if it returns a worker in a
    multi-worker DataLoader, ``_iter_pytorch`` splits the file's shards across
    all torch workers, which for a 1-file dataset means only worker slot 0 gets
    data and the other workers stream nothing (this also produces the
    ``"Too many dataloader workers ... max is dataset.num_shards=1"`` warning).
    Returning ``None`` makes ``datasets`` take the plain single-process path and
    yield the whole file. The override is scoped to the file read and restored
    on exit; only the prefetch thread (or the synchronous iterator) touches
    ``datasets`` in this process, so this is race-free.
    """
    orig = torch.utils.data.get_worker_info
    torch.utils.data.get_worker_info = lambda: None
    try:
        yield
    finally:
        torch.utils.data.get_worker_info = orig


class FlashEdgesStreamingDataset(IterableDataset):
    """IterableDataset wrapping the HF streaming parquet stream.

    Args:
        hf_dataset_repo (str | None): HF dataset repo id (e.g.
            ``"meteolibre-dev/flashedges_global_v1"``).  If None, ``data_files``
            must be set (local parquet glob, useful for tests).
        split (str): HF split name. Default "train".
        data_files (str | None): Recursive glob of local parquet files for
            offline mode (e.g. ``"data/**/*.parquet"``).  Recursive so dated
            subfolders (``data_2022_02/``) are included.
        data_dir (str | None): For Hub mode, restrict to one subfolder
            (e.g. ``"data_2022_02"``).  None = all subfolders combined.
        shuffle_buffer (int): Number of rows held in the shuffle buffer.  Set to
            0/1 to disable shuffling.  Larger => better decorrelation but more
            RAM (~4 MB per row).  Default 1000.
        precip_to_dbz (bool): Convert p01m mm/h -> dBZ (Marshall-Palmer).
        nb_temporal (int): Number of temporal frames to emit per row.
        seed (int): Base seed for the shuffle RNG (per-worker).
        max_retries (int): Max retries per file on transient read errors
            (HTTP 429 throttling returns a truncated buffer that pyarrow
            rejects as "Parquet magic bytes not found"); also the threshold of
            consecutive unpreprocessible rows before the stream aborts.
        prefetch_rows (int): Number of upcoming rows to fetch in a background
            thread, overlapping network I/O with GPU compute (mirrors
            ``torchdata.IterableWrapper.prefetch`` but dependency-free, since
            torchdata is archived). 0 disables. Default 8.  This layers beneath
            the DataLoader's own ``prefetch_factor`` (batch-level, default 2).
    """

    def __init__(
        self,
        hf_dataset_repo=None,
        split: str = "train",
        data_files=None,
        data_dir=None,
        shuffle_buffer: int = 100,
        precip_to_dbz: bool = True,
        nb_temporal: int = 7,
        seed: int = 42,
        max_retries: int = 8,
        prefetch_rows: int = 8,
    ):
        super().__init__()
        self.hf_dataset_repo = hf_dataset_repo
        self.split = split
        self.data_files = data_files
        self.data_dir = data_dir
        self.shuffle_buffer = shuffle_buffer
        self.precip_to_dbz = precip_to_dbz
        self.nb_temporal = nb_temporal
        self.seed = seed
        self.max_retries = max_retries
        self.prefetch_rows = prefetch_rows

    def _worker_seed(self) -> int:
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        rank = (
            dist.get_rank()
            if (dist.is_available() and dist.is_initialized())
            else 0
        )
        return self.seed + wid + rank * 100003

    def _resolve_files(self) -> list:
        """Full parquet file list (Hub ``hf://`` URLs or local paths)."""
        if self.hf_dataset_repo:
            paths = _list_hub_parquet_files(self.hf_dataset_repo, self.data_dir)
            return [f"hf://datasets/{self.hf_dataset_repo}/{p}" for p in paths]
        import glob

        files = sorted(glob.glob(self.data_files, recursive=True))
        if not files:
            raise FileNotFoundError(f"No parquet files matched: {self.data_files}")
        return files

    def _shard(self, files: list) -> list:
        """Disjoint slice of ``files`` for this DataLoader worker / rank.

        Replicates the worker/rank sharding ``datasets`` would normally do for
        an IterableDataset, but at file granularity so a per-file retry/skip
        only affects this worker's own files. Each (rank, worker) pair takes a
        strided slice ``files[idx::stride]``.
        """
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        n_workers = worker.num_workers if worker is not None else 1
        rank, world = 0, 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
        idx = rank * n_workers + wid
        stride = world * n_workers
        return files[idx::stride]

    def _open_one_file(self, fpath: str):
        """Open a single parquet file as a streaming record iterator."""
        from datasets import load_dataset

        return iter(
            load_dataset(
                "parquet",
                data_files={self.split: [fpath]},
                split=self.split,
                streaming=True,
            )
        )

    def _iter_file_records(self, fpath: str):
        """Yield raw records from one parquet file, with retry/backoff.

        HF rate-limiting (HTTP 429) makes the parquet builder receive a
        truncated buffer, which pyarrow rejects as
        ``ArrowInvalid: Parquet magic bytes not found in footer``. That error
        can surface either when opening the file or mid-iteration, so the whole
        read is wrapped: on any exception we sleep with exponential backoff and
        reopen, resuming *after* the last record we successfully yielded (so no
        record is duplicated or skipped). After ``max_retries`` the file is
        logged and abandoned, and the caller moves on to the next file -- the
        epoch keeps going instead of dying on the first 429.
        """
        import time

        yielded = 0
        attempt = 0
        while True:
            try:
                seen = 0
                # Disable datasets' torch-worker sharding for this per-file
                # read: we've already assigned whole files to this worker via
                # _shard, so we want the full file, not a 1/num_workers slice.
                with _no_torch_worker_sharding():
                    for record in self._open_one_file(fpath):
                        if seen < yielded:  # resume: skip already-emitted records
                            seen += 1
                            continue
                        seen += 1
                        yield record
                        yielded += 1
                return  # file fully consumed
            except Exception as e:  # noqa: BLE001 - any read failure is retryable
                attempt += 1
                if attempt > self.max_retries:
                    print(
                        f"[FlashEdgesStreamingDataset] giving up on file after "
                        f"{self.max_retries} retries, skipping: {fpath} "
                        f"(last error: {e})",
                        flush=True,
                    )
                    return
                backoff = min(30.0, 0.5 * (2 ** (attempt - 1)))
                print(
                    f"[FlashEdgesStreamingDataset] file read failed "
                    f"(attempt {attempt}/{self.max_retries}), retrying in "
                    f"{backoff:.1f}s: {fpath}: {e}",
                    flush=True,
                )
                time.sleep(backoff)

    def __iter__(self):
        rng = random.Random(self._worker_seed())

        # Resolve the full file list, shard across workers/ranks (each gets a
        # disjoint slice), and shuffle this worker's order for extra
        # decorrelation on top of the row-level shuffle buffer.
        files = self._shard(self._resolve_files())
        rng.shuffle(files)

        # Resilient raw-row stream: walks files one-by-one, retrying transient
        # read failures (HF throttling -> truncated buffer -> ArrowInvalid)
        # with exponential backoff, and skipping a file only after max_retries.
        # This keeps a single throttled/missing file from killing the epoch --
        # the old all-files-in-one-load_dataset approach died on the first 429.
        def raw_rows():
            for fpath in files:
                for record in self._iter_file_records(fpath):
                    yield record

        # Preprocess raw rows, skipping individual bad rows but aborting if too
        # many fail in a row (indicates a systemic schema problem).
        def preprocessed_rows():
            consecutive_bad = 0
            for record in raw_rows():
                try:
                    date = resolve_date(record)
                    yield preprocess_record(
                        date, record, self.nb_temporal, self.precip_to_dbz
                    )
                    consecutive_bad = 0
                except Exception as e:
                    consecutive_bad += 1
                    if consecutive_bad >= self.max_retries:
                        raise RuntimeError(
                            f"FlashEdgesStreamingDataset: {consecutive_bad} "
                            f"consecutive bad rows; last error: {e}"
                        ) from e
                    print(
                        f"[FlashEdgesStreamingDataset] skipping bad row: {e}",
                        flush=True,
                    )

        it = preprocessed_rows()

        # Background prefetch over the resilient stream: overlap the network
        # I/O of upcoming rows with the consumer's GPU compute. Mirrors
        # torchdata.IterableWrapper.prefetch but is dependency-free (torchdata
        # is archived). Layers beneath the DataLoader's own prefetch_factor.
        if self.prefetch_rows > 0:
            it = _PrefetchIter(it, self.prefetch_rows)

        buffer = deque(maxlen=self.shuffle_buffer) if self.shuffle_buffer > 1 else None

        def fill_buffer():
            while buffer is not None and len(buffer) < self.shuffle_buffer:
                try:
                    buffer.append(next(it))
                except StopIteration:
                    break

        if buffer is not None:
            fill_buffer()

        while True:
            # --- pick the next preprocessed sample ---
            if buffer is not None:
                if not buffer:
                    fill_buffer()
                if not buffer:
                    break  # stream exhausted
                idx = rng.randrange(len(buffer))
                # swap-pop for O(1) removal
                sample = buffer[idx]
                buffer[idx] = buffer[-1]
                buffer.pop()
                # refill lazily
                try:
                    buffer.append(next(it))
                except StopIteration:
                    pass
                yield sample
            else:
                try:
                    yield next(it)
                except StopIteration:
                    break
