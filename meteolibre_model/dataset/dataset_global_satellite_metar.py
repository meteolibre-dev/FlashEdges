"""
Torch Dataset for the FlashEdges global satellite + METAR parquet dataset.

This mirrors the FlashNet ``MeteoLibreMapDataset`` (LRU-cached parquet reads,
per-worker file shuffling, bisect index lookup, suncalc spatial features) but
reads the global patches produced by
``meteolibre_datasetgen.src.generate.generate_satellite_metar_dataset_v1``.

Each parquet row is one 128x128 spatial patch carrying the full temporal
context window of ``back_step_hours + 1 + forward_step_hours`` hourly frames
(default 7: 5h past + reference + 1h future):

  sat_data        : (T, 4, 128, 128)   float16  — GMGSI channels
                                                (LW IR, VIS, WV, SW IR)
  metar_data      : (T, 7, 128, 128)   float32  — [tmpc, dwpc, mslp,
                                                    cloud_cover, p01m,
                                                    wind_u, wind_v]
                                                NaN where no station reported
  elevation_data  : (128, 128)         float32  — DEM on the GMGSI grid

The dataset returns, per item:

  sat_patch_data    : (T, 5, H, W)   float32  — GMGSI(4) + elevation(1)
  metar_patch_data  : (T, 7, H, W)   float32  — METAR, NaN -> -10000 sentinel
  metar_mask        : (T, 7, H, W)   float32  — 1.0 where a station reported,
                                                 0.0 elsewhere (use to mask
                                                 the loss)
  spatial_position  : (4,)           float32  — [sun_azimuth, sun_altitude,
                                                 noon_sun_altitude, lat/10]

The reference timestamp is taken from the row's ``date`` column
(``YYYY-MM-DD HH:MM:SS``) rather than parsed from the filename, so the dataset
is robust to filename formatting changes in the generator.
"""

from datetime import datetime
import glob
import os
from collections import OrderedDict
import bisect

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch

from suncalc import get_position


# METAR channel order, kept in sync with the dataset generator.
METAR_FEATURES = [
    "tmpc",
    "dwpc",
    "mslp",
    "cloud_cover",
    "p01m",
    "wind_u",
    "wind_v",
]

# Sentinel used to fill NaN METAR pixels (matches the radar convention in
# FlashNet's dataset). The ``metar_mask`` output lets training code recover the
# valid-station locations exactly.
METAR_NAN_SENTINEL = -10000.0

# Elevation floor for below-sea-level / nodata pixels (FlashNet convention).
ELEVATION_FLOOR = -100.0

# Index of the precipitation channel (p01m) in METAR_FEATURES.
METAR_PRECIP_IDX = METAR_FEATURES.index("p01m")

# dBZ value assigned to perfectly dry reports (R = 0 mm/h).
#
# Marshall-Palmer maps R>0 to dBZ = 23.01 + 16*log10(R), so trace rain
# (R ~ 0.01..0.1 mm/h) lands at roughly -9..+7 dBZ. With the natural choice
# dry -> 0 dBZ, the "definitely no rain" signal sits *inside* the light-rain
# cluster, so the model cannot cleanly separate dry from drizzle. Pushing dry
# below that cluster (here -5 dBZ) gives the model a distinct "I am reporting no
# precipitation" marker separate from "a little". Tune as needed; only affects
# the dBZ-converted channel (precip_to_dbz=True). NOTE: changing this shifts
# the p01m distribution, so METAR_MEAN/STD and METAR_LOSS_WEIGHT must be
# recomputed (see scripts/compute_mean_std.py and compute_loss_weights.py).
DRY_DBZ = -5.0


def mmh_to_dbz(rate_mmh: np.ndarray) -> np.ndarray:
    """Convert rainfall rate (mm/h) to radar reflectivity (dBZ).

    Uses the Marshall-Palmer Z-R relation  Z = 200 * R^1.6,  dBZ = 10*log10(Z),
    i.e. dBZ = 23.0103 + 16*log10(R). Logarithmic reflectivity is far better
    behaved for ML losses than raw mm/h (heavy rain otherwise dominates the
    gradient).

      R = 0   -> DRY_DBZ (-5 by default)   (no echo, valid station -- kept
                 distinct from trace rain, which Marshall-Palmer places near
                 0 dBZ)
      R = 1   -> 23 dBZ
      R = 25  -> 45.7 dBZ
      NaN     -> NaN     (no station; left for the sentinel fill)
    """
    out = np.full_like(rate_mmh, np.nan, dtype=np.float32)
    finite = np.isfinite(rate_mmh)
    dry = finite & (rate_mmh <= 0.0)
    wet = finite & (rate_mmh > 0.0)
    out[dry] = DRY_DBZ
    out[wet] = 10.0 * np.log10(200.0) + 16.0 * np.log10(rate_mmh[wet])
    return out


def resolve_date(record) -> datetime:
    """Reference timestamp of a row, from the ``date`` column.

    Works with both a pandas Series and a plain dict (HF streaming rows are
    dicts), and tolerates the column being a string, a datetime, or a
    pandas Timestamp.
    """
    get = (record.get if hasattr(record, "get") else lambda k, d=None: record[k])
    date_val = get("date")
    if date_val is None or (isinstance(date_val, float) and np.isnan(date_val)):
        raise ValueError("record has no usable 'date' field")
    if isinstance(date_val, datetime):
        return date_val
    if hasattr(date_val, "to_pydatetime"):  # pandas.Timestamp
        return date_val.to_pydatetime()
    return datetime.strptime(str(date_val), "%Y-%m-%d %H:%M:%S")


def preprocess_record(date: datetime, record, nb_temporal: int, precip_to_dbz: bool) -> dict:
    """Turn one raw parquet row (dict or Series) into the model's input tensors.

    Shared by the map-style ``FlashEdgesGlobalDataset`` and the streaming
    ``FlashEdgesStreamingDataset`` so the two stay byte-identical.
    """
    # --- satellite (T, 4, H, W) float16 -> float32 ---
    sat_patch = (
        np.frombuffer(record["sat_data"], dtype=record["sat_dtype"])
        .reshape(record["sat_shape"])
        .astype(np.float32)
        .copy()
    )

    # --- elevation (H, W) -> (T, 1, H, W), floor negatives/nodata ---
    if record.get("elevation_data") is not None:
        elev = (
            np.frombuffer(record["elevation_data"], dtype=record["elevation_dtype"])
            .reshape(record["elevation_shape"])
            .astype(np.float32)
            .copy()
        )
    else:
        # Parquet file generated without elevation: zeros keep channel count.
        t_, _, h, w = sat_patch.shape
        elev = np.zeros((h, w), dtype=np.float32)

    elev = np.where(elev < 0, ELEVATION_FLOOR, elev)
    elev = elev[None, None, :, :].repeat(sat_patch.shape[0], axis=0)

    # dense conditioning field: GMGSI + elevation
    sat_patch_data = np.concatenate([sat_patch, elev], axis=1)

    # --- METAR (T, 7, H, W): optional dBZ, then validity mask + sentinel ---
    metar_patch = (
        np.frombuffer(record["metar_data"], dtype=record["metar_dtype"])
        .reshape(record["metar_shape"])
        .astype(np.float32)
        .copy()
    )

    if precip_to_dbz:
        metar_patch[:, METAR_PRECIP_IDX] = mmh_to_dbz(
            metar_patch[:, METAR_PRECIP_IDX]
        )

    metar_mask = (~np.isnan(metar_patch)).astype(np.float32)
    metar_patch_data = np.where(
        np.isnan(metar_patch), METAR_NAN_SENTINEL, metar_patch
    )

    # --- crop temporal dim if the row carries more frames than requested ---
    if sat_patch_data.shape[0] > nb_temporal:
        sat_patch_data = sat_patch_data[:nb_temporal]
    if metar_patch_data.shape[0] > nb_temporal:
        metar_patch_data = metar_patch_data[:nb_temporal]
        metar_mask = metar_mask[:nb_temporal]

    # --- sun position features from patch centre + reference time ---
    lon = float(record["lon"])
    lat = float(record["lat"])

    sun_pos = get_position(date, lon, lat)
    date_noon = date.replace(hour=12, minute=0, second=0, microsecond=0)
    sun_pos_noon = get_position(date_noon, lon, lat)

    spatial_position = torch.tensor(
        [
            float(sun_pos["azimuth"]),
            float(sun_pos["altitude"]),
            float(sun_pos_noon["altitude"]),
            lat / 25.0,
        ],
        dtype=torch.float32,
    )

    return {
        "sat_patch_data": torch.from_numpy(sat_patch_data),
        "metar_patch_data": torch.from_numpy(metar_patch_data),
        "metar_mask": torch.from_numpy(metar_mask),
        "spatial_position": spatial_position,
    }


class FlashEdgesGlobalDataset(torch.utils.data.Dataset):
    """
    Map-style dataset over the FlashEdges global GMGSI + METAR parquet patches.

    Args:
        localrepo (str): Root of the local dataset clone. Parquet files are
            read from *every* ``{localrepo}/data*/`` subdirectory
            (e.g. ``data/``, ``data_v1/``, ``data_2022_02/``, ...), so new
            time-bucketed ``data_YYYY_MM`` folders are picked up
            automatically.
        cache_size (int): Number of parquet DataFrames kept in the per-worker
            LRU cache.
        seed (int): Base seed for the one-time file-order shuffle performed
            in ``__init__``. The shuffled order is shared across all workers
            via fork, so every worker resolves a given index to the same
            physical row -- this is what guarantees full per-epoch coverage
            (the old per-worker shuffle silently dropped ~35% of rows).
            Re-seeding with a different value changes the file order run-to-run.
        nb_temporal (int): Number of temporal frames to return. If a parquet
            row carries more frames than this, the series is cropped to the
            first ``nb_temporal`` frames. Default 7 matches the generator's
            default 5-back / 1-forward window.
        precip_to_dbz (bool): If True (default), convert the METAR p01m
            precipitation channel (mm/h) to radar reflectivity (dBZ) via the
            Marshall-Palmer relation, so the model trains on a log-scaled
            target. Set False to keep raw mm/h.
    """

    def __init__(
        self,
        localrepo: str,
        cache_size: int = 8,
        seed: int = 42,
        nb_temporal: int = 7,
        precip_to_dbz: bool = True,
    ):
        super().__init__()
        self.localrepo = localrepo
        self.cache_size = cache_size
        self.seed = seed
        self.nb_temporal = nb_temporal
        self.precip_to_dbz = precip_to_dbz

        # Discover parquet files from *every* ``data*/`` subdirectory under
        # the repo root (e.g. data/, data_v1/, data_2022_02/, data_2024_12/,
        # ...). This is robust to the generator's time-bucketed folder layout:
        # new ``data_YYYY_MM`` buckets are picked up automatically. Sorted for a
        # stable base order; each worker reshuffles this list in __getitem__.
        data_dirs = sorted(
            d
            for d in glob.glob(os.path.join(self.localrepo, "data*"))
            if os.path.isdir(d)
        )
        candidates = sorted(
            pq_file
            for d in data_dirs
            for pq_file in glob.glob(os.path.join(d, "*.parquet"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No Parquet files found under any 'data*/' subdirectory of "
                f"'{self.localrepo}'. Found dirs: {data_dirs or '<none>'}"
            )
        # Shuffle the file order ONCE here in __init__ (main process, before
        # any worker fork) so every worker inherits the SAME permutation.
        # This replaces the old per-worker shuffle in __getitem__, which gave
        # each worker a DIFFERENT index->row map and silently dropped ~35% of
        # rows per epoch (two workers could resolve the same index to
        # different physical rows, and some rows were never reached by any
        # worker's permutation). With a shared deterministic map +
        # SequentialSampler (shuffle=False) every index 0..R-1 maps to a
        # unique physical row -> 100% coverage, and because each worker
        # receives strided contiguous index blocks it still reads files
        # near-sequentially (LRU cache stays hot, ~97% same-file reads),
        # preserving parquet read locality.
        g = torch.Generator()
        g.manual_seed(self.seed)
        perm = torch.randperm(len(candidates), generator=g).tolist()
        self.base_file_paths = [candidates[i] for i in perm]
        self.file_paths = list(self.base_file_paths)

        self.cache: "OrderedDict[int, pd.DataFrame]" = OrderedDict()
        
        print("number of files: ", len(self.file_paths))

        # Record counts per file -> cumulative offsets for bisect lookup.
        self.records_per_file_list = [
            sum(p.count_rows() for p in pq.ParquetDataset(fp).fragments)
            for fp in self.base_file_paths
        ]
        self.total_records = sum(self.records_per_file_list)

        print("total raws for training: ", self.total_records)

        if self.total_records == 0:
            raise ValueError(
                f"Parquet files under '{self.localrepo}' contain 0 rows."
            )
        self.cumulative_records = np.cumsum(
            [0] + self.records_per_file_list[:-1]
        ).tolist()

    def __len__(self) -> int:
        return self.total_records

    def _get_dataframe(self, file_index: int) -> pd.DataFrame:
        if file_index in self.cache:
            self.cache.move_to_end(file_index)
            return self.cache[file_index]

        file_path = self.file_paths[file_index]
        data_df = pd.read_parquet(file_path)
        self.cache[file_index] = data_df
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return data_df

    def _preprocess(self, date: datetime, record) -> dict:
        return preprocess_record(
            date, record, nb_temporal=self.nb_temporal, precip_to_dbz=self.precip_to_dbz
        )

    def _resolve_date(self, record) -> datetime:
        return resolve_date(record)

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self.total_records:
            raise IndexError(
                f"Index {index} out of range for dataset with size "
                f"{self.total_records}"
            )

        file_index = bisect.bisect_right(self.cumulative_records, index) - 1
        row_index_in_file = index - self.cumulative_records[file_index]

        data_df = self._get_dataframe(file_index)
        record = data_df.iloc[row_index_in_file]

        try:
            date = self._resolve_date(record)
            return self._preprocess(date, record)
        except Exception as e:
            # Skip a corrupt/unreadable row by wrapping to a neighbour. Logged
            # rather than raised so a single bad patch never kills a training
            # step.
            print(
                f"[FlashEdgesGlobalDataset] bad row index={index} "
                f"file_index={file_index} row={row_index_in_file}: {e}"
            )
            return self.__getitem__((index + 1) % self.total_records)
