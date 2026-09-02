"""
Torch Dataset for the FlashEdges global satellite + METAR/SYNOP + RADAR parquet
dataset (generator ``generate_satellite_metar_dataset_v2``, HF repo
``meteolibre-dev/global_sat_metar_v2``).

This mirrors ``dataset_global_satellite_metar.py`` (v1) — same LRU-cached
parquet reads, per-file bisect index lookup, suncalc spatial features — and
adds the radar channel:

  sat_data        : (T, 4, 128, 128)   float16  — GMGSI channels
                                                (LW IR, VIS, WV, SW IR)
  metar_data      : (T, 7, 128, 128)   float32  — merged METAR + SYNOP
                                                [tmpc, dwpc, mslp, cloud_cover,
                                                 p01m, wind_u, wind_v]
                                                NaN where no station reported
  radar_data      : (T, 1, 128, 128)   float32  — OPERA + MRMS DBZH (dBZ)
  elevation_data  : (128, 128)         float32  — DEM on the GMGSI grid

The dataset returns, per item:

  sat_patch_data    : (T, 6, H, W)   float32  — GMGSI(4) + radar(1) + elevation(1)
                                                (sat_patch_data = concat([sat, radar, elev], axis=1))
  metar_patch_data  : (T, 7, H, W)   float32  — METAR/SYNOP, NaN -> -10000 sentinel
  metar_mask        : (T, 7, H, W)   float32  — 1.0 where a station reported,
                                                0.0 elsewhere (use to mask the loss)
  spatial_position  : (4,)           float32  — [sun_azimuth, sun_altitude,
                                                noon_sun_altitude, lat/10]

Radar coverage handling
-----------------------
In the v2 parquet files there is NO distinction between "no radar data" and a
real 0 dBZ value: the upstream ``radar_v1`` GeoTIFFs encode both no-coverage
and no-echo/missing frames as NaN or 0, and bilinear reprojection further
blurs the two. The static coverage mask ``data_info/radar_cov_test.npz``
(sampled OPERA+MRMS coverage, packed bits, one mask per hour-of-day bin plus a
union) is therefore applied at load time:

  * pixels OUTSIDE the radar network coverage -> forced to NaN (whatever the
    stored value was), so the trainer's ``~torch.isnan`` masking treats them
    as invalid input exactly like the satellite NaN masking
    (``sat_patch = np.where(bad_lwir, np.nan, sat_patch)``);
  * pixels INSIDE coverage keep their stored value, then everything below
    0.1 dBZ (noise / ground clutter / trace) is snapped to DRY_DBZ (-5 dBZ),
    the same "no rain" marker as the METAR p01m dBZ conversion — giving a
    sharp rain/no-rain cut (dry = -5, rain > 0). Covered-but-NaN pixels
    (e.g. a fully missing radar frame) stay NaN and are simply excluded
    from the loss.

The mask is indexed by patch centre (lon/lat) on the same 1800x3600 0.1-degree
GMGSI grid the generator used. For simplicity a SINGLE static array is used,
time-independent: the any-hour UNION of the per-hour-of-day coverage samples
(the 0h/12h UTC bins differ by ~1e-6 of the globe, so the union costs nothing
and removes all time bookkeeping).

NOTE: the radar channel is stored in raw dBZ (roughly [-5, 70] over coverage).
Normalization statistics (scripts/compute_mean_std.py) and loss weights must be
recomputed for the new 6-channel input layout before training.
"""

from datetime import datetime
from pathlib import Path
import glob
import os
from collections import OrderedDict
import bisect

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch

from suncalc import get_position

from meteolibre_model.dataset.dataset_global_satellite_metar import (
    DRY_DBZ,
    METAR_FEATURES,
    METAR_NAN_SENTINEL,
    METAR_PRECIP_IDX,
    ELEVATION_FLOOR,
    mmh_to_dbz,
    resolve_date,
)


# ---------------------------------------------------------------------------
# Radar coverage mask
# ---------------------------------------------------------------------------

# Candidate locations for the packed coverage NPZ when no explicit path is
# given: CWD first, then the FlashEdges repo root (package parents).
_DEFAULT_COV_CANDIDATES = [
    Path("data_info/radar_cov_test.npz"),
    Path(__file__).resolve().parents[2] / "data_info" / "radar_cov_test.npz",
]

# Module-level cache so DataLoader forked workers share one unpacked copy
# (fork inherits it; unpacking ~20 MB of bools once per process is cheap but
# pointless to repeat per dataset instance).
_RADAR_COV_CACHE: "OrderedDict[str, 'RadarCoverageMask']" = OrderedDict()
_RADAR_COV_CACHE_MAX = 4


class RadarCoverageMask:
    """Static radar coverage mask (OPERA + MRMS), packed-bit NPZ format.

    Layout mirrors ``meteolibre_datasetgen.src.radar.coverage_mask.RadarCoverage``:
    a 1800x3600 0.1-degree global grid, origin upper-left (row 0 = lat_max).
    Only the any-hour UNION of the per-hour-of-day samples is kept — the mask
    is deliberately static / time-independent.
    """

    def __init__(self, path):
        with np.load(str(path), allow_pickle=False) as z:
            self.width = int(z["width"])
            self.height = int(z["height"])
            self.resolution = float(z["resolution"])
            self.lon_min = float(z["lon_min"])
            self.lat_max = float(z["lat_max"])
            n = self.width * self.height
            self.union = (
                np.unpackbits(z["union"])[:n]
                .reshape(self.height, self.width)
                .astype(bool)
            )

    def patch_mask(
        self,
        lon: float,
        lat: float,
        patch_size: int,
    ) -> np.ndarray:
        """(patch_size, patch_size) bool coverage for one patch (static).

        The patch centre (lon, lat) sits at local pixel (patch_size/2,
        patch_size/2) of the same 0.1-degree grid the generator sliced from,
        so the patch's top-left corner maps to grid index
        ((lat_max - lat)/res - patch/2, (lon - lon_min)/res - patch/2).
        Out-of-grid pixels (should not happen with the generator's offsets)
        are marked uncovered.
        """
        col0 = int(round((lon - self.lon_min) / self.resolution)) - patch_size // 2
        row0 = int(round((self.lat_max - lat) / self.resolution)) - patch_size // 2

        mask = self.union

        out = np.zeros((patch_size, patch_size), dtype=bool)
        r0 = max(row0, 0)
        c0 = max(col0, 0)
        r1 = min(row0 + patch_size, self.height)
        c1 = min(col0 + patch_size, self.width)
        if r1 > r0 and c1 > c0:
            out[
                r0 - row0 : r1 - row0, c0 - col0 : c1 - col0
            ] = mask[r0:r1, c0:c1]
        return out


def load_radar_coverage(path=None) -> RadarCoverageMask:
    """Load (and cache) the packed radar coverage NPZ."""
    if path is None:
        for cand in _DEFAULT_COV_CANDIDATES:
            if cand.exists():
                path = cand
                break
        if path is None:
            raise FileNotFoundError(
                "No radar coverage NPZ found; pass radar_cov_path explicitly "
                f"(searched: {[str(c) for c in _DEFAULT_COV_CANDIDATES]})"
            )
    key = str(Path(path).resolve())
    if key in _RADAR_COV_CACHE:
        _RADAR_COV_CACHE.move_to_end(key)
        return _RADAR_COV_CACHE[key]
    cov = RadarCoverageMask(key)
    while len(_RADAR_COV_CACHE) >= _RADAR_COV_CACHE_MAX:
        _RADAR_COV_CACHE.popitem(last=False)
    _RADAR_COV_CACHE[key] = cov
    return cov


# ---------------------------------------------------------------------------
# Per-record preprocessing
# ---------------------------------------------------------------------------


def preprocess_record(
    date: datetime,
    record,
    nb_temporal: int,
    precip_to_dbz: bool,
    coverage: RadarCoverageMask,
) -> dict:
    """Turn one raw v2 parquet row (dict or Series) into the model's tensors.

    Same contract as ``dataset_global_satellite_metar.preprocess_record`` but
    concatenates the radar channel into ``sat_patch_data``:
    ``sat_patch_data = np.concatenate([sat_patch, radar, elev], axis=1)``
    giving (T, 6, H, W). Radar pixels outside the static coverage mask are
    forced to NaN (see module docstring).
    """
    # --- satellite (T, 4, H, W) float16 -> float32 ---
    sat_patch = (
        np.frombuffer(record["sat_data"], dtype=record["sat_dtype"])
        .reshape(record["sat_shape"])
        .astype(np.float32)
        .copy()
    )

    # --- Sanitize GMGSI fill/saturation artifacts via LWIR -> NaN (all 4ch) ---
    # (identical to v1: detect from LWIR channel, propagate to all GMGSI
    # channels; see dataset_global_satellite_metar.py for the thresholds'
    # justification)
    lwir = sat_patch[:, 0]
    bad_lwir = (lwir < 10.0) | (lwir >= 255.0)            # (T, H, W) bool
    bad_lwir = np.broadcast_to(bad_lwir[:, None, :, :], sat_patch.shape)  # (T,4,H,W)
    sat_patch = np.where(bad_lwir, np.nan, sat_patch)

    # --- radar (T, 1, H, W) -> float32, coverage-masked to NaN ---
    if record.get("radar_data") is not None:
        radar = (
            np.frombuffer(record["radar_data"], dtype=record["radar_dtype"])
            .reshape(record["radar_shape"])
            .astype(np.float32)
            .copy()
        )
    else:
        # Row generated without radar (e.g. a v1 file picked up from another
        # data*/ dir): all-NaN keeps the channel count consistent.
        t_, _, h, w = sat_patch.shape
        radar = np.full((t_, 1, h, w), np.nan, dtype=np.float32)

    # Single static (time-independent) coverage mask for the patch, applied
    # to every frame.
    covered = coverage.patch_mask(
        float(record["lon"]), float(record["lat"]), radar.shape[-1]
    )

    # Outside coverage: force NaN (stored 0 dBZ there is a no-data artefact,
    # not a dry measurement). Inside coverage the stored value is meaningful;
    # covered-but-NaN (missing frame / no echo encoding) stays NaN and is
    # excluded from input + loss by the trainer's ~torch.isnan masking.
    radar = np.where(covered[None, None, :, :], radar, np.nan)

    # --- Dry/trace binarization: dBZ < 0.1 -> DRY_DBZ (-5), same convention
    # as the METAR p01m channel (mmh_to_dbz in dataset_global_satellite_metar).
    # Radar reflectivities just above 0 dBZ are noise/clutter ground mist, not
    # meaningful rain: without this, the model sees a continuous fog of tiny
    # dBZ values and cannot draw a clean rain/no-rain boundary. Snapping
    # everything below 0.1 dBZ (R <~ 0.03 mm/h, i.e. below Marshall-Palmer's
    # validity floor) to the same DRY_DBZ marker used for dry METAR reports
    # gives a sharp cut: DRY_DBZ (-5) = no rain, > 0 dBZ = rain, matching the
    # precip channel. NaN (uncovered / missing frame) is untouched: NaN < 0.1
    # is False so those pixels stay NaN and keep being masked out.
    radar = np.where(radar < 0.1, DRY_DBZ, radar)

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
        _, _, h, w = sat_patch.shape
        elev = np.zeros((h, w), dtype=np.float32)

    elev = np.where(elev < 0, ELEVATION_FLOOR, elev)
    elev = elev[None, None, :, :].repeat(sat_patch.shape[0], axis=0)

    # dense conditioning field: GMGSI + radar + elevation  -> (T, 6, H, W)
    sat_patch_data = np.concatenate([sat_patch, radar, elev], axis=1)

    # --- METAR/SYNOP (T, 7, H, W): optional dBZ, then validity mask + sentinel ---
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


# ---------------------------------------------------------------------------
# Map-style dataset
# ---------------------------------------------------------------------------


class FlashEdgesGlobalDatasetV2(torch.utils.data.Dataset):
    """
    Map-style dataset over the FlashEdges global GMGSI + METAR/SYNOP + radar
    parquet patches (generator v2).

    Args:
        localrepo (str): Root of the local dataset clone. Parquet files are
            read from *every* ``{localrepo}/data*/`` subdirectory, so new
            time-bucketed ``data_{year}`` folders (the v2 HF layout) are
            picked up automatically.
        cache_size (int): Number of parquet DataFrames kept in the per-worker
            LRU cache.
        seed (int): Base seed for the one-time file-order shuffle performed in
            ``__init__`` and shared across workers via fork (guarantees full
            per-epoch coverage; see v1 class for details).
        nb_temporal (int): Number of temporal frames to return. Rows carrying
            more frames are cropped to the first ``nb_temporal``. Default 7
            matches the generator's 5-back / ref / 1-forward window.
        precip_to_dbz (bool): If True (default), convert the METAR/SYNOP p01m
            channel (mm/h) to dBZ via Marshall-Palmer (same as v1).
        radar_cov_path (str | Path | None): Path to the packed radar coverage
            NPZ (``radar_cov_test.npz``). If None, resolved from
            ``data_info/`` under the CWD or the repo root. The mask is applied
            time-independent (any-hour union).
    """

    def __init__(
        self,
        localrepo: str,
        cache_size: int = 8,
        seed: int = 42,
        nb_temporal: int = 7,
        precip_to_dbz: bool = True,
        radar_cov_path=None,
    ):
        super().__init__()
        self.localrepo = localrepo
        self.cache_size = cache_size
        self.seed = seed
        self.nb_temporal = nb_temporal
        self.precip_to_dbz = precip_to_dbz

        self.coverage = load_radar_coverage(radar_cov_path)

        # Discover parquet files from every data*/ subdirectory (sorted for a
        # stable base order), shuffle ONCE here so forked workers share the
        # same index -> row map (100% per-epoch coverage).
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
        g = torch.Generator()
        g.manual_seed(self.seed)
        perm = torch.randperm(len(candidates), generator=g).tolist()
        self.base_file_paths = [candidates[i] for i in perm]
        self.file_paths = list(self.base_file_paths)

        self.cache: "OrderedDict[int, pd.DataFrame]" = OrderedDict()

        print("number of files: ", len(self.file_paths))

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
            date,
            record,
            nb_temporal=self.nb_temporal,
            precip_to_dbz=self.precip_to_dbz,
            coverage=self.coverage,
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
            # Skip a corrupt/unreadable row by wrapping to a neighbour.
            print(
                f"[FlashEdgesGlobalDatasetV2] bad row index={index} "
                f"file_index={file_index} row={row_index_in_file}: {e}"
            )
            return self.__getitem__((index + 1) % self.total_records)
