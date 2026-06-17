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

import random
from collections import deque

import torch
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


def _load_streaming_dataset(
    hf_dataset_repo, split, data_files, data_dir, streaming=True
):
    """Build the HF IterableDataset, supporting both Hub and local parquet.

    Hub mode explicitly enumerates parquet files via ``HfApi`` and passes them
    as ``hf://datasets/...`` URLs to the parquet builder.  This sidesteps HF's
    path-based split inference, which is unreliable for the generator's
    multi-folder layout (``data/``, ``data_2022_02/``, ...) and would otherwise
    silently drop folders or mis-assign splits.
    """
    from datasets import load_dataset

    if hf_dataset_repo is not None:
        paths = _list_hub_parquet_files(hf_dataset_repo, data_dir)
        urls = [f"hf://datasets/{hf_dataset_repo}/{p}" for p in paths]
        return load_dataset(
            "parquet", data_files={split: urls}, split=split, streaming=streaming
        )

    if data_files is None:
        raise ValueError(
            "Either hf_dataset_repo or data_files must be provided."
        )

    # Recursive glob so dated subfolders (data_2022_02/, ...) are included.
    import glob

    files = sorted(glob.glob(data_files, recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {data_files}")
    return load_dataset(
        "parquet", data_files={"train": files}, split="train", streaming=streaming
    )


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
        max_retries (int): How many times to skip a bad row before giving up.
    """

    def __init__(
        self,
        hf_dataset_repo=None,
        split: str = "train",
        data_files=None,
        data_dir=None,
        shuffle_buffer: int = 1000,
        precip_to_dbz: bool = True,
        nb_temporal: int = 7,
        seed: int = 42,
        max_retries: int = 8,
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

    def _worker_seed(self) -> int:
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        import torch.distributed as dist

        rank = (
            dist.get_rank()
            if (dist.is_available() and dist.is_initialized())
            else 0
        )
        return self.seed + wid + rank * 100003

    def __iter__(self):
        rng = random.Random(self._worker_seed())

        ds = _load_streaming_dataset(
            self.hf_dataset_repo,
            self.split,
            self.data_files,
            self.data_dir,
            streaming=True,
        )

        # `datasets` IterableDataset already shards across workers/ranks when
        # used inside a DataLoader with num_workers>0, so we just iterate.
        it = iter(ds)

        buffer = deque(maxlen=self.shuffle_buffer) if self.shuffle_buffer > 1 else None

        def fill_buffer():
            nonlocal it
            while buffer is not None and len(buffer) < self.shuffle_buffer:
                try:
                    buffer.append(next(it))
                except StopIteration:
                    break

        if buffer is not None:
            fill_buffer()

        consecutive_bad = 0
        while True:
            # --- pick the next raw row ---
            if buffer is not None:
                if not buffer:
                    fill_buffer()
                if not buffer:
                    break  # stream exhausted
                idx = rng.randrange(len(buffer))
                # swap-pop for O(1) removal
                record = buffer[idx]
                buffer[idx] = buffer[-1]
                buffer.pop()
                # refill lazily
                try:
                    buffer.append(next(it))
                except StopIteration:
                    pass
            else:
                try:
                    record = next(it)
                except StopIteration:
                    break

            # --- preprocess, skipping bad rows ---
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
                        f"FlashEdgesStreamingDataset: {consecutive_bad} consecutive "
                        f"bad rows; last error: {e}"
                    ) from e
                # log and continue
                print(
                    f"[FlashEdgesStreamingDataset] skipping bad row: {e}",
                    flush=True,
                )
                continue
