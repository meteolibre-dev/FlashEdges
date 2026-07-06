#!/usr/bin/env python3
"""
Convert FlashEdges forecast GeoTIFFs into videos.

FlashEdges inference (``backend/main.py``) writes, per forecast timestep:

    forecast_{YYYYMMDDHHMM}_sat.tif    — 2 bands:
        [gmgsi_lwir (IR), gmgsi_vis (VIS)]
    forecast_{YYYYMMDDHHMM}_metar.tif  — 7 bands:
        [tmpc, dwpc, mslp, cloud_cover, p01m(dBZ), wind_u, wind_v]

on the global 1800x3600 grid (0.1 deg, EPSG:4326, origin at (-180, 90)).
This script turns those into:

  - A combined 4-panel video (default):  LWIR · Temperature · Precip · Cloud
  - One video per band (with --per-channel)

Global coastlines are drawn if ``geopandas`` is installed (gracefully skipped
otherwise). Inspired by flashnet's ``tiff_to_video.py`` but adapted to the
FlashEdges global satellite+METAR layout.

Usage
-----
    # default: combined 4-panel video in forecasts/
    python scripts/tiff_to_video.py forecasts/

    # all per-band videos too, 6 fps
    python scripts/tiff_to_video.py forecasts/ --per-channel --fps 6

    # zoom on a region (preset or custom lon_min,lat_min,lon_max,lat_max)
    python scripts/tiff_to_video.py forecasts/ --crop europe
    python scripts/tiff_to_video.py forecasts/ --crop 2,43,8,46

Dependencies:  rasterio, numpy, opencv-python, matplotlib (+ geopandas optional)
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Natural Earth coastlines (optional, global)
# ---------------------------------------------------------------------------
_borders_gdf = None
NE_COUNTRIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_110m_admin_0_countries.geojson"
)


def get_world_borders():
    """Load and cache global country borders (110m). Returns None if unavailable."""
    global _borders_gdf
    if _borders_gdf is not None:
        return _borders_gdf
    try:
        import geopandas as gpd
    except ImportError:
        print("  [info] geopandas not installed -> skipping coastlines "
              "(pip install geopandas to enable).")
        return None
    print("  [info] downloading Natural Earth 110m country borders ...")
    _borders_gdf = gpd.read_file(NE_COUNTRIES_URL)
    return _borders_gdf


# ---------------------------------------------------------------------------
# Region crop presets (lon_min, lat_min, lon_max, lat_max)
# ---------------------------------------------------------------------------
CROP_PRESETS = {
    "world":  (-180, -90, 180, 90),
    "europe": (-25, 34, 45, 72),
    "france": (-5.5, 41.0, 10.0, 51.5),
    "usa":    (-125, 24, -66, 50),
    "conus":  (-125, 24, -66, 50),
    "asia":   (60, 5, 150, 55),
    "africa": (-20, -35, 55, 38),
    "tropics":(-180, -30, 180, 30),
}


def parse_crop(crop_str):
    s = crop_str.strip().lower()
    if s in CROP_PRESETS:
        return CROP_PRESETS[s]
    parts = s.split(",")
    if len(parts) == 4:
        try:
            return tuple(float(p) for p in parts)
        except ValueError:
            pass
    print(f"Error: invalid --crop '{crop_str}'. Use a preset "
          f"({', '.join(CROP_PRESETS)}) or lon_min,lat_min,lon_max,lat_max.",
          file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Colormap LUTs (256 x 4 uint8 RGBA)
# ---------------------------------------------------------------------------
def _mpl_lut(name):
    cmap = matplotlib.colormaps[name]
    return (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)


def _custom_lut(stops):
    cmap = LinearSegmentedColormap.from_list("c", stops)
    return (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)


# Satellite / physical colormaps
LUT_LWIR = _custom_lut([  # thermal IR: deep blue -> cyan -> green -> yellow -> red
    (0.0, (0.03, 0.0, 0.18)), (0.2, (0.0, 0.2, 0.55)),
    (0.4, (0.0, 0.55, 0.55)), (0.55, (0.1, 0.75, 0.2)),
    (0.72, (0.95, 0.85, 0.05)), (0.88, (0.95, 0.35, 0.0)),
    (1.0, (0.85, 0.0, 0.05)),
])
LUT_VIS = _custom_lut([(0, (0, 0, 0)), (0.55, (0.35, 0.35, 0.35)),
                       (0.85, (0.85, 0.85, 0.85)), (1, (1, 1, 1))])
LUT_WV = _custom_lut([   # water vapour: brown -> green -> cyan -> white
    (0.0, (0.20, 0.10, 0.05)), (0.35, (0.45, 0.30, 0.10)),
    (0.6, (0.20, 0.55, 0.30)), (0.8, (0.15, 0.6, 0.7)),
    (1.0, (0.95, 0.95, 0.95)),
])
LUT_SWIR = _mpl_lut("inferno")
LUT_TERRAIN = _mpl_lut("terrain")
LUT_TEMP = _custom_lut([   # cold blue -> white -> hot red
    (0.0, (0.12, 0.20, 0.65)), (0.5, (0.97, 0.97, 0.97)),
    (1.0, (0.75, 0.05, 0.05)),
])
LUT_DEWP = _mpl_lut("YlGnBu")
LUT_PRESS = _custom_lut([  # low pressure blue, high red
    (0.0, (0.2, 0.35, 0.75)), (0.5, (0.9, 0.9, 0.9)),
    (1.0, (0.7, 0.12, 0.12)),
])
LUT_CLOUD = _custom_lut([(0, (0, 0, 0)), (1, (1, 1, 1))])
LUT_WIND = _mpl_lut("coolwarm")      # negative blue, positive red
LUT_RADAR = _custom_lut([  # dBZ: light blue -> green -> yellow -> orange -> red -> magenta
    (0.0, (0.0, 0.93, 0.93)), (0.17, (0.0, 0.65, 0.0)),
    (0.34, (0.0, 1.0, 0.0)), (0.50, (1.0, 1.0, 0.0)),
    (0.64, (1.0, 0.65, 0.0)), (0.80, (1.0, 0.0, 0.0)),
    (1.0, (0.7, 0.0, 0.7)),
])


# ---------------------------------------------------------------------------
# Band catalogue
# ---------------------------------------------------------------------------
# key -> dict(file=, band=, display=, unit=, vmin=, vmax=, lut=, percentile=)
BANDS = {
    # ---- satellite (_sat.tif, 2 bands: IR + VIS) ----
    "gmgsi_lwir": dict(file="sat", band=1, display="GMGSI LWIR (IR)",   unit="",
                       vmin=None, vmax=None, lut=LUT_LWIR, percentile=True),
    "gmgsi_vis":  dict(file="sat", band=2, display="GMGSI VIS",         unit="",
                       vmin=None, vmax=None, lut=LUT_VIS,  percentile=True),
    # ---- METAR (_metar.tif, 7 bands) ----
    "tmpc":        dict(file="metar", band=1, display="2m Temperature", unit="C",
                        vmin=-40, vmax=50,   lut=LUT_TEMP,  percentile=False),
    "dwpc":        dict(file="metar", band=2, display="2m Dewpoint",    unit="C",
                        vmin=-40, vmax=35,   lut=LUT_DEWP,  percentile=False),
    "mslp":        dict(file="metar", band=3, display="MSLP",           unit="hPa",
                        vmin=970, vmax=1050, lut=LUT_PRESS, percentile=False),
    "cloud_cover": dict(file="metar", band=4, display="Cloud Cover",    unit="",
                        vmin=0,    vmax=1,    lut=LUT_CLOUD, percentile=False),
    "p01m":        dict(file="metar", band=5, display="Precipitation",  unit="dBZ",
                        vmin=0,    vmax=65,   lut=LUT_RADAR, percentile=False),
    "wind_u":      dict(file="metar", band=6, display="Wind U",         unit="m/s",
                        vmin=-30,  vmax=30,   lut=LUT_WIND,  percentile=False),
    "wind_v":      dict(file="metar", band=7, display="Wind V",         unit="m/s",
                        vmin=-30,  vmax=30,   lut=LUT_WIND,  percentile=False),
}

# Default combined-panel layout
DEFAULT_LAYOUT = ["gmgsi_lwir", "tmpc", "p01m", "cloud_cover"]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
TS_PATTERN = "%Y%m%d%H%M"   # forecast_{YYYYMMDDHHMM}_<sat|metar>.tif


def collect_files(tiff_dir):
    """Return {timestamp_str: {"sat": path, "metar": path}}."""
    import re
    pat = re.compile(r"forecast_(\d{12})_(sat|metar)\.tif[f]?$")
    out = defaultdict(dict)
    for f in sorted(tiff_dir.iterdir()):
        m = pat.match(f.name)
        if m:
            out[m.group(1)][m.group(2)] = f
    return dict(out)


def format_ts(ts):
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}:{ts[10:12]} UTC"


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------
def lonlat_to_pixel(lon, lat, transform):
    col, row = (~transform) * (lon, lat)
    return int(round(col)), int(round(row))


def crop_array(arr, transform, bounds):
    lon_min, lat_min, lon_max, lat_max = bounds
    col_start, row_start = (~transform) * (lon_min, lat_max)
    col_end, row_end = (~transform) * (lon_max, lat_min)
    col_start = max(0, int(round(col_start)))
    row_start = max(0, int(round(row_start)))
    col_end = min(arr.shape[1], int(round(col_end)))
    row_end = min(arr.shape[0], int(round(row_end)))
    if col_end <= col_start or row_end <= row_start:
        return arr, transform
    cropped = arr[row_start:row_end, col_start:col_end]
    new_t = rasterio.transform.Affine(
        transform.a, transform.b, transform.c + col_start * transform.a,
        transform.d, transform.e, transform.f + row_start * transform.e,
    )
    return cropped, new_t


# ---------------------------------------------------------------------------
# Normalisation & rendering
# ---------------------------------------------------------------------------
def to_uint8(arr, vmin, vmax):
    arr = np.nan_to_num(arr, nan=0.0)
    if vmax > vmin:
        norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    else:
        norm = np.zeros_like(arr)
    return (norm * 255).astype(np.uint8)


def render_band(arr, lut, vmin, vmax):
    """arr (H,W) float -> BGR (H,W,3) uint8 on a dark background."""
    idx = to_uint8(arr, vmin, vmax)              # (H,W)
    rgba = lut[idx]                              # (H,W,4)
    alpha = (rgba[..., 3:4].astype(np.float32) / 255.0)
    bg = np.array([10, 12, 24], dtype=np.float32)
    rgb = rgba[..., :3].astype(np.float32) * alpha + bg * (1 - alpha)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_coastlines(frame, transform, gdf, color=(190, 190, 190), thickness=1):
    if gdf is None:
        return frame
    h, w = frame.shape[:2]
    out = frame.copy()
    for geom in gdf.geometry:
        rings = []
        if geom.geom_type == "Polygon":
            rings = [geom.exterior]
        elif geom.geom_type == "MultiPolygon":
            rings = [p.exterior for p in geom.geoms]
        else:
            continue
        for ring in rings:
            pts = []
            for lon, lat in np.array(ring.coords):
                c, r = lonlat_to_pixel(lon, lat, transform)
                if -1000 <= c < w + 1000 and -1000 <= r < h + 1000:
                    pts.append((c, r))
            if len(pts) >= 2:
                cv2.polylines(out, [np.array(pts, np.int32)], False, color, thickness)
    return out


def draw_overlay(frame, title, ts, sub=None):
    """Semi-transparent info box top-left: title + timestamp (+ optional sub)."""
    h, w = frame.shape[:2]
    s = max(w / 2000.0, 0.25)
    pad = max(6, round(12 * s))
    th = max(1, round(1 * s))
    out = frame.copy()

    # title larger than the timestamp line
    title_fs, time_fs = 0.7 * s, 0.5 * s
    lines = [(title, title_fs), (format_ts(ts), time_fs)]
    if sub:
        lines.append((sub, time_fs))

    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0]
             for t, fs in lines]
    box_w = max(tw for tw, _ in sizes) + pad * 2
    box_h = sum(h_ + pad for _, h_ in sizes) + pad

    ov = out.copy()
    cv2.rectangle(ov, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, out, 0.4, 0, out)

    y = pad + sizes[0][1] + pad // 2
    for (t, fs), (_, h_) in zip(lines, sizes):
        cv2.putText(out, t, (pad + pad // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
        y += h_ + pad
    return out


def load_band(path, band):
    with rasterio.open(path) as src:
        return src.read(band).astype(np.float32), src.transform


def compute_percentiles(channels_files, band_key, cfg):
    """2nd/98th percentile across all frames (subsampled) for percentile bands."""
    samples = []
    for ts, files in channels_files.items():
        path = files.get(cfg["file"])
        if path is None:
            continue
        arr, _ = load_band(path, cfg["band"])
        arr = arr[::4, ::4].ravel()           # subsample for speed
        arr = arr[np.isfinite(arr)]
        samples.append(arr)
    if not samples:
        return 0.0, 1.0
    s = np.concatenate(samples)
    if s.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(s, [2, 98])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------
def write_video(frames, out_path, fps):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), True)
    for f in frames:
        writer.write(f)
    writer.release()


def resize_to(frame, target_w):
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    new_h = max(1, round(h * target_w / w))
    return cv2.resize(frame, (target_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Build a single channel's frame sequence
# ---------------------------------------------------------------------------
def build_channel_video(channels_files, band_key, cfg, out_dir, fps,
                        borders_gdf, crop_bounds, panel_w, title_suffix=""):
    entries = sorted(channels_files.items())
    vmin, vmax = cfg["vmin"], cfg["vmax"]
    if cfg["percentile"]:
        vmin, vmax = compute_percentiles(channels_files, band_key, cfg)
        print(f"    [{band_key}] percentile range {vmin:.2f} .. {vmax:.2f}")

    title = cfg["display"] + (f" ({cfg['unit']})" if cfg["unit"] else "")
    if title_suffix:
        title += title_suffix

    frames = []
    for i, (ts, files) in enumerate(entries):
        path = files.get(cfg["file"])
        if path is None:
            continue
        arr, transform = load_band(path, cfg["band"])
        if crop_bounds:
            arr, transform = crop_array(arr, transform, crop_bounds)
        bgr = render_band(arr, cfg["lut"], vmin, vmax)
        bgr = draw_coastlines(bgr, transform, borders_gdf)
        bgr = resize_to(bgr, panel_w)
        bgr = draw_overlay(bgr, title, ts)
        frames.append(bgr)
        if (i + 1) % 10 == 0:
            print(f"    [{band_key}] {i + 1}/{len(entries)} frames")

    tag = ("_" + "_".join(f"{b:g}" for b in crop_bounds)) if crop_bounds else ""
    out_path = out_dir / f"forecast_{band_key}{tag}.mp4"
    write_video(frames, out_path, fps)
    print(f"    -> {out_path}  ({len(frames)} frames, {fps} fps)")
    return out_path


def build_combined_video(channels_files, layout, out_dir, fps,
                         borders_gdf, crop_bounds, panel_w):
    entries = sorted(channels_files.items())
    # layout: list of band keys or None (pad to 4 slots)
    slots = (layout + [None] * 4)[:4]
    cfgs = {k: BANDS[k] for k in slots if k}

    # Precompute ranges (percentile bands first)
    ranges = {}
    for k in [k for k in slots if k]:
        cfg = cfgs[k]
        if cfg["percentile"]:
            ranges[k] = compute_percentiles(channels_files, k, cfg)
            print(f"    [{k}] percentile range {ranges[k][0]:.2f} .. {ranges[k][1]:.2f}")
        else:
            ranges[k] = (cfg["vmin"], cfg["vmax"])

    frames = []
    gap = 6
    title_h = 70

    for i, (ts, files) in enumerate(entries):
        panels = []
        for k in slots:
            cfg = cfgs.get(k)
            path = files.get(cfg["file"]) if cfg else None
            if cfg is None or path is None:
                ph = max(panel_w // 2, 1)
                placeholder = np.full((ph, panel_w, 3), 24, np.uint8)
                panels.append(draw_overlay(placeholder, "-", ts))
                continue
            arr, transform = load_band(path, cfg["band"])
            if crop_bounds:
                arr, transform = crop_array(arr, transform, crop_bounds)
            bgr = render_band(arr, cfg["lut"], *ranges[k])
            bgr = draw_coastlines(bgr, transform, borders_gdf)
            bgr = resize_to(bgr, panel_w)
            bgr = draw_overlay(bgr, cfg["display"], ts)
            panels.append(bgr)

        ph, pw = panels[0].shape[:2]
        canvas = np.full((title_h + 2 * ph + gap, 2 * pw + gap, 3), 14, np.uint8)

        # title bar
        cv2.putText(canvas, f"FlashEdges Forecast  -  {format_ts(ts)}",
                    (20, title_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        positions = [(0, title_h), (pw + gap, title_h),
                     (0, title_h + ph + gap), (pw + gap, title_h + ph + gap)]
        for (x, y), pan in zip(positions, panels):
            canvas[y:y + pan.shape[0], x:x + pan.shape[1]] = pan

        frames.append(canvas)
        if (i + 1) % 10 == 0:
            print(f"    [combined] {i + 1}/{len(entries)} timestamps")

    tag = ("_" + "_".join(f"{b:g}" for b in crop_bounds)) if crop_bounds else ""
    out_path = out_dir / f"forecast_combined{tag}.mp4"
    write_video(frames, out_path, fps)
    print(f"    -> {out_path}  ({len(frames)} frames, {fps} fps)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Render FlashEdges forecast GeoTIFFs into videos.")
    ap.add_argument("tiff_dir", type=Path, help="Directory with forecast_*.tif files")
    ap.add_argument("--fps", type=int, default=4, help="Frames per second (default 4)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: same as tiff_dir)")
    ap.add_argument("--panel-width", type=int, default=1280,
                    help="Width of each panel in pixels (default 1280)")
    ap.add_argument("--per-channel", action="store_true",
                    help="Also render one video per band")
    ap.add_argument("--no-combined", action="store_true",
                    help="Skip the combined 4-panel video")
    ap.add_argument("--layout", type=str, default=",".join(DEFAULT_LAYOUT),
                    help="Comma-separated band keys for the combined panels "
                         "(default: " + ",".join(DEFAULT_LAYOUT) + ")")
    ap.add_argument("--channels", type=str, default=None,
                    help="Comma-separated band keys for --per-channel "
                         "(default: all)")
    ap.add_argument("--crop", type=str, default=None,
                    help="Region preset (world/europe/france/usa/asia/africa/"
                         "tropics) or lon_min,lat_min,lon_max,lat_max")
    ap.add_argument("--no-coastlines", action="store_true",
                    help="Do not draw coastlines")
    args = ap.parse_args()

    tiff_dir = args.tiff_dir.resolve()
    if not tiff_dir.is_dir():
        print(f"Error: {tiff_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    out_dir = (args.out_dir or tiff_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    channels_files = collect_files(tiff_dir)
    if not channels_files:
        print("No forecast_*.tif files found.", file=sys.stderr)
        sys.exit(1)

    n_ts = len(channels_files)
    print(f"Found {n_ts} forecast timesteps in {tiff_dir}")
    crop_bounds = parse_crop(args.crop) if args.crop else None
    if crop_bounds:
        print(f"Crop region: lon[{crop_bounds[0]},{crop_bounds[2]}] "
              f"lat[{crop_bounds[1]},{crop_bounds[3]}]")

    layout = [c.strip() for c in args.layout.split(",") if c.strip()]
    for c in layout:
        if c not in BANDS:
            print(f"Error: unknown band '{c}'. Valid: {', '.join(BANDS)}",
                  file=sys.stderr)
            sys.exit(1)

    borders = None if args.no_coastlines else get_world_borders()

    if not args.no_combined:
        if len(layout) != 4:
            print(f"[note] layout has {len(layout)} panels; combined grid is "
                  f"designed for 4 (continuing anyway)")
        print("\n[combined] building multi-panel video ...")
        build_combined_video(channels_files, layout, out_dir, args.fps,
                             borders, crop_bounds, args.panel_width)

    if args.per_channel:
        sel = ([c.strip() for c in args.channels.split(",")]
               if args.channels else list(BANDS.keys()))
        for c in sel:
            if c not in BANDS:
                print(f"  [skip] unknown band '{c}'")
                continue
            print(f"\n[{c}] rendering per-channel video ...")
            build_channel_video(channels_files, c, BANDS[c], out_dir, args.fps,
                                borders, crop_bounds, args.panel_width)

    print("\nDone.")


if __name__ == "__main__":
    main()
