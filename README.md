# FlashEdges
Global ML model using only sensor data (satellite + ground based station) for fast inference and edge on global market

## Visualizing forecasts on an interactive map (web)

`scripts/tiff_to_web.py` renders the `forecasts/*.tif` files into a web-based,
interactive MapTiler map (instead of the `.mp4` from `scripts/tiff_to_video.py`).
It reuses the same colormaps and band definitions so the web colors match the
videos.

### Quick start

```bash
# 1. Render frames + manifest from your forecast GeoTIFFs
python3 scripts/tiff_to_web.py forecasts/

# 2. Serve the repo root and open the viewer
node scripts/serve_webviz.js            # default port 8000
node scripts/serve_webviz.js 9000       # optional custom port
#  -> http://localhost:8000/webviz/
```

The `serve_webviz.js` server is zero-dependency Node (no `npm install` needed).
If you prefer, `python3 -m http.server 8000` works just as well.

You need a **free MapTiler API key**: get one at
https://cloud.maptiler.com/account/keys/ and paste it into the panel, or pass
it via the URL: `http://localhost:8000/webviz/?key=YOUR_KEY`.

> Open via `http://`, not `file://` — image overlays require an HTTP origin.

### Options

```bash
# only a few bands, full-resolution PNG, custom output dir
python3 scripts/tiff_to_web.py forecasts/ \
  --bands tmpc,p01m,cloud_cover --format png --downsample 1 --out-dir webviz

# zoom on Europe (frames + bounds are cropped accordingly)
python3 scripts/tiff_to_web.py forecasts/ --crop europe
```

| Flag | Default | Description |
|------|---------|-------------|
| `--bands` | `gmgsi_lwir,gmgsi_vis,tmpc,dwpc,mslp,cloud_cover,p01m` | Comma-separated band keys to render |
| `--format` | `webp` | `webp` or `png` |
| `--quality` | `80` | WebP quality (1-100, ignored for PNG) |
| `--downsample` | `2` | Downsample factor (2 -> 1800x900 from 3600x1800; 1 = full res) |
| `--crop` | world | Region preset (`world`/`europe`/`france`/`usa`/`asia`/`africa`/`tropics`) or `lon_min,lat_min,lon_max,lat_max` |
| `--out-dir` | `webviz` | Output directory |
| `--default-band` | first of `--bands` | Band selected by default in the viewer |

### Viewer features (`webviz/index.html`)

- Animated forecast overlay on a pan/zoomable map.
- Band switcher with live legend colorbar.
- Time animation bar (play/pause, speed, keyboard: Space / ← / →).
- Opacity slider.
- Optional MapTiler live weather layers (Radar / Precipitation / Temperature)
  for side-by-side comparison — uses `animateByFactor(3600)` like the
  [official examples](https://docs.maptiler.com/sdk-js/examples/weather-radar/). 
