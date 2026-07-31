#!/usr/bin/env python3
"""
Render FlashEdges forecast GeoTIFFs into a web-friendly visualization.

Instead of writing videos (see ``tiff_to_video.py``), this script renders each
selected band of each forecast timestep to a small transparent WebP/PNG image
and emits a ``manifest.js`` that ``webviz/index.html`` consumes to display the
forecast as an animated overlay on a MapTiler / MapLibre GL JS map.

It reuses the colormaps (LUTs), band catalogue and helpers from
``tiff_to_video.py`` so the colours match the videos exactly.

The forecast grid is the global 1800x3600 (0.1 deg, EPSG:4326, origin at
(-180, 90)) grid written by ``backend/inference_engine.py``. Each rendered
frame carries the same geographic bounds, so the web viewer just overlays it
as a MapLibre ``image`` source with 4-corner coordinates.

Usage
-----
    # default: render the 7 most useful bands into ./webviz and emit manifest.js
    python3 scripts/tiff_to_web.py forecasts/

    # only a few bands, full resolution PNG, custom output dir
    python3 scripts/tiff_to_web.py forecasts/ --bands tmpc,p01m,cloud_cover \
        --format png --downsample 1 --out-dir webviz

    # zoom on Europe (frames + bounds are cropped accordingly)
    python3 scripts/tiff_to_web.py forecasts/ --crop europe

Dependencies:  rasterio, numpy, opencv-python (or Pillow), matplotlib
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse the colormaps / band catalogue / helpers from the video script so the
# web colours match the videos exactly. ``tiff_to_video.py`` lives next to us.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from tiff_to_video import (  # noqa: E402
    BANDS,
    collect_files,
    compute_percentiles,
    crop_array,
    format_ts,
    load_band,
    parse_crop,
)

# ---------------------------------------------------------------------------
# Web-specific config
# ---------------------------------------------------------------------------
# Bands rendered by default. wind_u/wind_v are vector fields and don't render
# well as a single scalar image, so they're excluded by default (still
# selectable via --bands).
DEFAULT_WEB_BANDS = [
    "gmgsi_lwir", "gmgsi_vis",
    "tmpc", "dwpc", "mslp", "cloud_cover", "p01m",
]

# Per-band alpha mode.
#   "mask"      -> fully opaque where the value is finite, transparent on NaN.
#   "value"     -> alpha scales with the normalized value so empty / zero areas
#                  are transparent and let the base map show through.
#   "threshold" -> transparent below ALPHA_THRESHOLD, opaque above it.
#                  Used for precipitation: no rain = see-through, rain = solid.
ALPHA_MODE = {
    "cloud_cover": "value",
    "p01m":        "threshold",
}
ALPHA_THRESHOLD = {
    "p01m": 0.0,          # dBZ <= 0 means no rain -> transparent
}
DEFAULT_ALPHA_MODE = "mask"

# Global grid bounds (matches the GeoTIFF transform when uncropped).
WORLD_BOUNDS = (-180.0, -90.0, 180.0, 90.0)

# Web Mercator latitude limit — MapLibre image sources cannot render at
# latitude +-90 (projects to infinity / degenerate geometry). Clamp the
# image-source corner coordinates to this safe limit.
MERCATOR_MAX_LAT = 85.05


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_rgba(arr, lut, vmin, vmax, alpha_mode=DEFAULT_ALPHA_MODE,
                 threshold=0.0):
    """Render a (H, W) float array to an (H, W, 4) uint8 RGBA array.

    NaN / inf pixels become fully transparent. Alpha modes:
      ``mask``      fully opaque where finite.
      ``value``     alpha scales with the normalized value (gamma 0.6).
      ``threshold`` transparent where value <= threshold, opaque above.
    """
    finite = np.isfinite(arr)
    a = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    idx = (a * 255).astype(np.uint8)
    rgba = lut[idx].copy()                       # (H, W, 4) uint8

    if alpha_mode == "value":
        av = np.clip(a, 0.0, 1.0) ** 0.6          # gamma so faint values still show
        out_a = (av * 255.0).astype(np.uint8)
    elif alpha_mode == "threshold":
        out_a = np.where(arr > threshold, 255, 0).astype(np.uint8)
    else:
        out_a = np.full(arr.shape, 255, np.uint8)
    out_a[~finite] = 0
    rgba[..., 3] = out_a
    return rgba


def reproject_to_mercator(rgba, lat_min, lat_max):
    """Reproject an equirectangular RGBA image to Web Mercator (EPSG:3857)
    y-spacing.

    MapLibre image sources assume the image is in Web Mercator projection:
    the 4-corner coordinates define a quad in Mercator space, and the image
    is texture-mapped onto it. Our forecast data is equirectangular (equal
    latitude spacing), so without reprojection the polar regions get
    stretched ~11x toward the poles (Mercator distortion).

    This resamples the rows so the pixel distribution matches Mercator
    y = ln(tan(pi/4 + lat/2)).
    """
    h, w = rgba.shape[:2]
    if h <= 1:
        return rgba

    # Mercator y for each latitude (top = lat_max, bottom = lat_min)
    y_top = np.log(np.tan(np.pi / 4 + np.radians(lat_max) / 2))
    y_bot = np.log(np.tan(np.pi / 4 + np.radians(lat_min) / 2))

    # For each output row, compute the Mercator y, convert back to latitude,
    # and find the source row in the equirectangular image.
    out_rows = np.arange(h)
    y_merc = y_top + (y_bot - y_top) * out_rows / (h - 1)  # top to bottom
    lats = np.degrees(2 * np.arctan(np.exp(y_merc)) - np.pi / 2)

    # Source row: lat_max at row 0, decreasing by pixel_size per row
    src_rows = (lat_max - lats) / ((lat_max - lat_min) / (h - 1))
    src_rows = np.clip(src_rows, 0, h - 1)

    # Bilinear interpolation between adjacent source rows
    row_low = np.floor(src_rows).astype(np.int32)
    row_high = np.minimum(row_low + 1, h - 1)
    frac = (src_rows - row_low).astype(np.float32)[:, None, None]  # (H,1,1)

    result = (rgba[row_low] * (1 - frac) + rgba[row_high] * frac).astype(np.uint8)
    return result


def downsample(rgba, factor):
    if factor <= 1:
        return rgba
    h, w = rgba.shape[:2]
    new_w = max(1, w // factor)
    new_h = max(1, h // factor)
    import cv2
    # INTER_AREA is best for downscaling; cv2 handles 4-channel arrays fine.
    return cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)


def lut_gradient(lut, n=12):
    """Sample an (256,4) LUT into a compact [(pos, 'rgb(r,g,b)'), ...] list
    used by the web viewer to draw a legend colorbar."""
    stops = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b = lut[int(round(t * 255))][:3]
        stops.append([round(t, 3), f"rgb({r},{g},{b})"])
    return stops


def band_meta_for(keys, ranges):
    meta = []
    for k in keys:
        cfg = BANDS[k]
        vmin, vmax = ranges[k]
        meta.append({
            "key": k,
            "display": cfg["display"],
            "unit": cfg["unit"],
            "vmin": vmin,
            "vmax": vmax,
            "alpha_mode": ALPHA_MODE.get(k, DEFAULT_ALPHA_MODE),
            "gradient": lut_gradient(cfg["lut"]),
        })
    return meta


def save_image(rgba, path: Path, fmt: str, quality: int):
    rgba = np.ascontiguousarray(rgba)
    try:
        import cv2
        # cv2 expects BGRA order for 4-channel images.
        bgra = rgba[..., [2, 1, 0, 3]]
        if fmt == "webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, max(1, quality)]
        elif fmt == "png":
            params = []                          # lossless; quality ignored
        else:
            raise ValueError(f"unsupported format {fmt!r}")
        ok = cv2.imwrite(str(path), bgra, params)
        if not ok:
            raise RuntimeError("cv2.imwrite returned False")
    except Exception:
        # Fallback to Pillow (RGBA order).
        from PIL import Image
        img = Image.fromarray(rgba, mode="RGBA")
        if fmt == "webp":
            img.save(path, format="WEBP", quality=quality, lossless=False)
        else:
            img.save(path, format="PNG")


# ---------------------------------------------------------------------------
# Bounds / coordinates
# ---------------------------------------------------------------------------
def bounds_from_transform(transform, width, height):
    """Return (lon_min, lat_min, lon_max, lat_max) covered by the raster."""
    lon_min = transform.c
    lat_max = transform.f
    lon_max = lon_min + transform.a * width
    lat_min = lat_max + transform.e * height       # transform.e is negative
    return (lon_min, lat_min, lon_max, lat_max)


def image_coordinates(bounds, clamp_lat=True):
    """4-corner [lng, lat] order MapLibre expects: TL, TR, BR, BL.

    MapLibre image sources cannot render at latitude +-90 (Mercator projects
    to infinity -> degenerate geometry, overlay invisible). Clamp to the Web
    Mercator limit (85.05) unless ``clamp_lat=False``.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    if clamp_lat:
        lat_max = min(lat_max, MERCATOR_MAX_LAT)
        lat_min = max(lat_min, -MERCATOR_MAX_LAT)
    return [
        [lon_min, lat_max],
        [lon_max, lat_max],
        [lon_max, lat_min],
        [lon_min, lat_min],
    ]


def iso_from_ts(ts: str) -> str:
    return (
        f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:00Z"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Render FlashEdges forecast GeoTIFFs into a web overlay "
                    "(frames + manifest.js for webviz/index.html).")
    ap.add_argument("tiff_dir", type=Path,
                    help="Directory with forecast_*.tif files")
    ap.add_argument("--out-dir", type=Path, default=Path("webviz"),
                    help="Output directory (default: ./webviz)")
    ap.add_argument("--bands", type=str, default=",".join(DEFAULT_WEB_BANDS),
                    help="Comma-separated band keys to render "
                         f"(default: {','.join(DEFAULT_WEB_BANDS)})")
    ap.add_argument("--format", choices=["webp", "png"], default="ext",
                    help="Image format (default: webp)")
    ap.add_argument("--quality", type=int, default=80,
                    help="WebP quality 1-100 (default 80, ignored for PNG)")
    ap.add_argument("--downsample", type=int, default=2,
                    help="Downsample factor (default 2 -> 1800x900 from "
                         "3600x1800). Use 1 for full resolution.")
    ap.add_argument("--crop", type=str, default=None,
                    help="Region preset (world/europe/france/usa/asia/africa/"
                         "tropics) or lon_min,lat_min,lon_max,lat_max")
    ap.add_argument("--default-band", type=str, default=None,
                    help="Band selected by default in the viewer "
                         f"(default: first of --bands = {DEFAULT_WEB_BANDS[0]})")
    args = ap.parse_args()

    # `--format ext` lets us keep the default visible as "webp" in --help while
    # still accepting the literal default value.
    fmt = "webp" if args.format == "ext" else args.format

    tiff_dir = args.tiff_dir.resolve()
    if not tiff_dir.is_dir():
        print(f"Error: {tiff_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    bands = [c.strip() for c in args.bands.split(",") if c.strip()]
    for c in bands:
        if c not in BANDS:
            print(f"Error: unknown band '{c}'. Valid: {', '.join(BANDS)}",
                  file=sys.stderr)
            sys.exit(1)
    default_band = args.default_band or bands[0]
    if default_band not in bands:
        print(f"Error: --default-band '{default_band}' not in --bands",
              file=sys.stderr)
        sys.exit(1)

    channels_files = collect_files(tiff_dir)
    if not channels_files:
        print("No forecast_*.tif files found.", file=sys.stderr)
        sys.exit(1)
    entries = sorted(channels_files.items())
    n_ts = len(entries)
    print(f"Found {n_ts} forecast timesteps in {tiff_dir}")

    crop_bounds = parse_crop(args.crop) if args.crop else None
    if crop_bounds:
        print(f"Crop region: lon[{crop_bounds[0]},{crop_bounds[2]}] "
              f"lat[{crop_bounds[1]},{crop_bounds[3]}]")

    # MapLibre image sources cannot render at latitude +-90 (Mercator
    # projects to infinity). Always crop the image data to the Web Mercator
    # limit so the pixels and the coordinate bounds match exactly — otherwise
    # the full -90..90 image gets squeezed into +-85.05 and everything shifts
    # toward the equator.
    effective_crop = crop_bounds
    if effective_crop is None:
        effective_crop = (-180.0, -MERCATOR_MAX_LAT, 180.0, MERCATOR_MAX_LAT)
    else:
        lon_min, lat_min, lon_max, lat_max = effective_crop
        lat_min = max(lat_min, -MERCATOR_MAX_LAT)
        lat_max = min(lat_max, MERCATOR_MAX_LAT)
        effective_crop = (lon_min, lat_min, lon_max, lat_max)
    if effective_crop != crop_bounds:
        print(f"Clamping latitude to Web Mercator limit: "
              f"lat[{effective_crop[1]:.2f},{effective_crop[3]:.2f}]")

    out_dir = args.out_dir.resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute display ranges (percentile bands need a 2-pass scan).
    ranges = {}
    for k in bands:
        cfg = BANDS[k]
        if cfg["percentile"]:
            print(f"  [{k}] computing percentile range ...")
            ranges[k] = compute_percentiles(channels_files, k, cfg)
            print(f"      -> {ranges[k][0]:.2f} .. {ranges[k][1]:.2f}")
        else:
            ranges[k] = (cfg["vmin"], cfg["vmax"])

    band_meta = band_meta_for(bands, ranges)

    frames_manifest = []
    rendered = 0
    frame_bounds = None   # captured on first band of first timestep

    for i, (ts, files) in enumerate(entries):
        frame_images = {}

        for k in bands:
            cfg = BANDS[k]
            path = files.get(cfg["file"])
            if path is None:
                continue
            arr, transform = load_band(path, cfg["band"])
            # Always crop to effective_crop (which is at least clamped to
            # +-85.05 lat) so image pixels match the coordinate bounds.
            arr, transform = crop_array(arr, transform, effective_crop)
            if frame_bounds is None:
                frame_bounds = bounds_from_transform(
                    transform, arr.shape[1], arr.shape[0])

            rgba = render_rgba(arr, cfg["lut"], *ranges[k],
                               alpha_mode=ALPHA_MODE.get(k, DEFAULT_ALPHA_MODE),
                               threshold=ALPHA_THRESHOLD.get(k, 0.0))
            # Reproject equirectangular -> Web Mercator so the pixel
            # distribution matches what MapLibre expects (image sources are
            # treated as Mercator, so an equirectangular image gets stretched
            # toward the poles without this step).
            b = bounds_from_transform(transform, arr.shape[1], arr.shape[0])
            rgba = reproject_to_mercator(rgba, b[1], b[3])
            rgba = downsample(rgba, args.downsample)
            fname = f"{k}_{ts}.{fmt}"
            save_image(rgba, frames_dir / fname, fmt, args.quality)
            frame_images[k] = f"frames/{fname}"
            rendered += 1

        frames_manifest.append({
            "ts": ts,
            "label": format_ts(ts),
            "iso": iso_from_ts(ts),
            "images": frame_images,
        })
        print(f"  [{i + 1}/{n_ts}] {ts}  ({len(frame_images)} bands)")

    if frame_bounds is None:
        frame_bounds = effective_crop

    manifest = {
        "bounds": list(frame_bounds),
        "coordinates": image_coordinates(frame_bounds),
        "bands": band_meta,
        "default_band": default_band,
        "frame_interval_hours": 1,
        "frame_count": len(frames_manifest),
        "frames": frames_manifest,
        "format": fmt,
    }

    # manifest.js assigns a global so the page works from file:// too (no
    # fetch/CORS needed). Also write manifest.json for parity / external use.
    js_path = out_dir / "manifest.js"
    js_path.write_text(
        "/* Auto-generated by scripts/tiff_to_web.py. Do not edit. */\n"
        "window.__FE_MANIFEST = " + json.dumps(manifest, indent=2) + ";\n",
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {rendered} frame images to {frames_dir}")
    print(f"Manifest: {js_path}")
    print(f"\nNext: serve the repo root and open the viewer, e.g.\n"
          f"  python3 -m http.server 8000\n"
          f"  -> http://localhost:8000/{out_dir.name}/")
    print("You'll need a free MapTiler API key: https://cloud.maptiler.com/")


if __name__ == "__main__":
    main()
