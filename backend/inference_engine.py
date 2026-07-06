"""
Inference engine for FlashEdges global satellite + METAR tiled diffusion.

Mirrors flashnet's ``backend/inference_engine.py`` but adapted for the
FlashEdges model:

  - Dual branches: satellite (GMGSI 4 + elevation 1 = 5ch) + METAR (7ch)
    instead of satellite + lightning + radar.
  - 1-hour temporal cadence (GMGSI is hourly, not 10-min like MTG FCI).
  - No residual targets (config ``residual: false``).
  - METAR is sparse: NaN where no station, sentinel -10000 -> 0 after
    normalize.  The valid-station mask is preserved to blank non-station
    pixels in the output.
  - ``metar_ref`` skip path (JiT3D_Modern) is fed the metar input channels.
  - Global 1800x3600 grid at 0.1° (EPSG:4326).

The H5 input format is produced by
``meteolibre_datasetgen.src.backend_flashedges.services``:

  sat_data        : (4, 4, 1800, 3600)   float16  — GMGSI
  metar_data      : (4, 7, 1800, 3600)   float32  — METAR, NaN where no station
  elevation_data  : (1800, 3600)         float32  — DEM

Output: GeoTIFF forecast files per timestep, optionally converted to COG.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable, Generator, List, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import torch
import torch.nn.functional as F
import h5py
import rasterio
from suncalc import get_position
from tqdm.auto import tqdm

# COG conversion (optional)
try:
    from rio_cogeo.cogeo import cog_translate
    from rio_cogeo.profiles import cog_profiles
    COG_AVAILABLE = True
except ImportError:
    COG_AVAILABLE = False
    print("Warning: rio-cogeo not available. COG conversion will be skipped.")

# Add project root to sys.path so ``meteolibre_model`` resolves
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from meteolibre_model.models.jit3d_dual_v2 import DualJiT3D
from meteolibre_model.diffusion.rectified_flow_satellite_metar_v1 import (
    normalize,
    denormalize,
    CLIP_MIN,
)
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


# --- FlashEdges constants (mirror dataset_global_satellite_metar.py) ---------

# Sentinel used to fill NaN METAR pixels in the dataset loader.
METAR_NAN_SENTINEL = -10000.0

# Elevation floor for below-sea-level / nodata pixels.
ELEVATION_FLOOR = -100.0

# Number of METAR channels (tmpc, dwpc, mslp, cloud_cover, p01m, wind_u, wind_v)
NUM_METAR_CHANNELS = 7

# Number of satellite channels (GMGSI 4 + elevation 1)
NUM_SAT_CHANNELS = 5


def convert_to_cog(input_path: str, delete_original: bool = True) -> str:
    """Convert a TIFF to Cloud Optimized GeoTIFF format in-place."""
    if not COG_AVAILABLE:
        return input_path
    if not os.path.exists(input_path):
        logger.warning(f"Input file not found: {input_path}")
        return input_path

    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_cog = os.path.join(temp_dir, f"temp_cog_{os.path.basename(input_path)}")

    try:
        logger.info(f"Converting to COG: {input_path}")
        with rasterio.open(input_path) as src:
            height = src.height
            width = src.width

        def get_optimal_block_size(dim):
            for size in [512, 256, 128, 64]:
                if dim % size == 0:
                    return size
            return 256

        block_size = min(get_optimal_block_size(height), get_optimal_block_size(width), 512)
        logger.info(f"Using block size {block_size}x{block_size} for {width}x{height} image")

        output_profile = cog_profiles.get("deflate")
        output_profile.update({
            "blockxsize": block_size,
            "blockysize": block_size,
        })

        config = {
            "GDAL_NUM_THREADS": "ALL_CPUS",
            "GDAL_TIFF_INTERNAL_MASK": "YES",
            "COMPRESS_OVERVIEW": "DEFLATE",
        }

        cog_translate(input_path, temp_cog, output_profile, config=config,
                       resampling="bilinear", quiet=True)

        if os.path.exists(temp_cog):
            with rasterio.open(temp_cog) as cog_src:
                if not cog_src.profile.get('tiled', False):
                    logger.warning(f"COG may not be properly tiled: {cog_src.profile}")

        if delete_original:
            os.remove(input_path)
            os.rename(temp_cog, input_path)
            logger.info(f"COG created: {input_path}")
        else:
            os.rename(temp_cog, input_path.replace('.tiff', '_cog.tiff'))
            logger.info(f"COG created: {input_path.replace('.tiff', '_cog.tiff')}")
        return input_path
    except Exception as e:
        logger.error(f"Error converting to COG: {e}")
        if os.path.exists(temp_cog):
            os.remove(temp_cog)
        return input_path


class InferenceStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class InferenceResult:
    status: InferenceStatus
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class FlashEdgesInferenceEngine:
    """Engine for running FlashEdges tiled diffusion inference."""

    def __init__(
        self,
        model_path: str,
        config_name: str = "model_v1_global_satellite_metar",
        patch_size: int = 128,
        denoising_steps: int = 32,
        batch_size: int = 64,
        context_frames: int = 4,
        interpolation: str = "linear",
        use_residual: bool = False,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to the .safetensors model weights.
            config_name: Config key in meteolibre_model/config/configs.yml.
            patch_size: Spatial patch size for tiled inference (default 128,
                matching training).
            denoising_steps: Number of Euler steps for the RF ODE.
            batch_size: Number of patches processed per model forward pass.
            context_frames: Number of past frames fed as context (default 4).
            interpolation: 'linear' or 'polynomial' RF schedule.
            use_residual: Whether the model was trained with residual targets.
            device: 'cuda' or 'cpu' (auto-detected if None).
        """
        self.model_path = model_path
        self.config_name = config_name
        self.patch_size = patch_size
        self.denoising_steps = denoising_steps
        self.batch_size = batch_size
        self.context_frames = context_frames
        self.interpolation = interpolation
        self.use_residual = use_residual

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[torch.nn.Module] = None
        self.params: Dict[str, Any] = {}

        self._load_config()
        self._load_model()

    def _load_config(self) -> None:
        """Load model configuration from configs.yml."""
        import yaml
        config_path = os.path.join(project_root, "meteolibre_model", "config", "configs.yml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        if self.config_name not in config:
            raise KeyError(f"Config '{self.config_name}' not found in {config_path}")
        self.params = config[self.config_name]

    def _load_model(self) -> None:
        """Load model weights from safetensors."""
        logger.info(f"Loading model from {self.model_path}")
        torch.set_float32_matmul_precision('medium')

        model_params = self.params["model"]

        self.model = DualJiT3D(**model_params)

        if os.path.exists(self.model_path):
            loaded_state_dict = load_file(self.model_path)
            self.model.load_state_dict(loaded_state_dict)
            logger.info(f"Loaded model weights from {self.model_path}")
        else:
            logger.warning(
                f"Model weights not found at {self.model_path}. "
                "Using randomly initialized model."
            )

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")

    # ------------------------------------------------------------------ #
    # Tiled inference helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_patch(image: torch.Tensor, x: int, y: int, ps: int) -> torch.Tensor:
        return image[..., y: y + ps, x: x + ps]

    @staticmethod
    def _get_gaussian_weights(patch_size: int, device: str, sigma_scale: float = 0.3) -> torch.Tensor:
        """2D Gaussian weight mask for smooth patch blending."""
        x = torch.linspace(-(patch_size - 1) / 2, (patch_size - 1) / 2, patch_size, device=device)
        sigma = sigma_scale * patch_size
        w_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        w_2d = w_1d.unsqueeze(1) * w_1d.unsqueeze(0)
        w_2d = w_2d / w_2d.max()
        return w_2d

    def _build_patch_coords(self, H_big: int, W_big: int) -> List[Tuple[int, int]]:
        """Build overlapping patch coordinates (two shifted grids + border bands)."""
        ps = self.patch_size
        shift = ps // 2

        def get_starts(total, size, start_offset):
            starts = list(range(start_offset, total - size + 1, size))
            if not starts or starts[-1] != total - size:
                if total >= size:
                    starts.append(total - size)
            return starts

        y0 = get_starts(H_big, ps, 0)
        x0 = get_starts(W_big, ps, 0)
        yS = get_starts(H_big, ps, shift)
        xS = get_starts(W_big, ps, shift)

        coords1 = [(x, y) for y in y0 for x in x0]
        coords2 = [(x, y) for y in yS for x in xS]
        extra_top = [(x, y0[0]) for x in xS]
        extra_left = [(x0[0], y) for y in yS]
        extra_bottom = [(x, y0[-1]) for x in xS]
        extra_right = [(x0[-1], y) for y in yS]

        return list(set(coords1 + coords2 + extra_top + extra_left + extra_bottom + extra_right))

    @torch.no_grad()
    def tiled_inference(
        self,
        initial_context: torch.Tensor,
        forecast_steps: int = 3,
        nb_forecast: int = 3,
        date: Optional[datetime] = None,
        c_sat: int = NUM_SAT_CHANNELS,
        c_metar: int = NUM_METAR_CHANNELS,
        sat_nodata_mask: Optional[torch.Tensor] = None,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Run tiled diffusion inference with autoregressive rollout.

        Args:
            initial_context: (1, C, T_ctx, H, W) normalized context tensor.
                Channel order: [sat(5), metar(7)].
            forecast_steps: Total number of forecast frames to produce.
            nb_forecast: Frames generated per model call (autoregressive batch).
            date: Datetime of the first context frame.
            c_sat: Number of satellite channels (GMGSI + elevation = 5).
            c_metar: Number of METAR channels (7).
            sat_nodata_mask: (1, c_sat, T_ctx, H, W) boolean, True where sat
                is no-data. Used to blank those pixels in the output.

        Yields:
            (sat_batch_cpu, metar_batch_cpu) after each autoregressive step,
            each of shape (1, C, nb, H, W) in physical units.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self.model.eval()
        self.model.to(self.device)

        C = c_sat + c_metar
        _, _, T_ctx, H_big, W_big = initial_context.shape

        patch_weights = self._get_gaussian_weights(self.patch_size, self.device)
        patch_weights = patch_weights.view(1, 1, 1, self.patch_size, self.patch_size)
        patch_coords = self._build_patch_coords(H_big, W_big)

        # Geo transform from the global GMGSI grid (pixel -> lon/lat)
        # EPSG:4326, transform = [0.1, 0, -180, 0, -0.1, 90]
        # No CRS conversion needed.
        geo_transform = [0.1, 0.0, -180.0, 0.0, -0.1, 90.0]

        x_t_full_res = torch.randn(1, C, nb_forecast, H_big, W_big, device=self.device)
        current_context = initial_context

        current_step = 0
        while current_step < forecast_steps:
            remaining = forecast_steps - current_step
            this_nb = min(nb_forecast, remaining)

            if date:
                # Each forecast frame is 1 hour after the last context frame
                prediction_date = date + timedelta(hours=current_step + 1)
            else:
                prediction_date = datetime.utcnow()

            logger.info(f"Generating forecast frames {current_step + 1}-{current_step + this_nb}/{forecast_steps}")

            # Generate this_nb forecast frames at once. We run the full
            # denoising loop on the nb_forecast-shaped noise, but only keep
            # this_nb frames at the end.
            x_t = torch.randn(1, C, nb_forecast, H_big, W_big, device=self.device)

            for i in tqdm(range(self.denoising_steps), desc="Denoising"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                aggregated_velocity = torch.zeros(
                    1, C, nb_forecast, H_big, W_big, device=self.device, dtype=torch.bfloat16
                )
                weights_sum = torch.zeros(
                    1, 1, nb_forecast, H_big, W_big, device=self.device, dtype=torch.bfloat16
                )

                t_val = 1.0 - i / self.denoising_steps
                dt = 1.0 / self.denoising_steps
                t_batch_val = torch.full((1,), t_val, device=self.device)
                d_batch_val = torch.full((1,), 0.0, device=self.device)

                for i_batch in range(0, len(patch_coords), self.batch_size):
                    coords_batch = patch_coords[i_batch: i_batch + self.batch_size]

                    pixel_xs = [x + self.patch_size // 2 for x, y in coords_batch]
                    pixel_ys = [y + self.patch_size // 2 for x, y in coords_batch]

                    lons, lats = [], []
                    for j in range(len(coords_batch)):
                        px = pixel_xs[j]
                        py = pixel_ys[j]
                        lon = geo_transform[0] * px + geo_transform[2]
                        lat = geo_transform[4] * py + geo_transform[5]
                        lons.append(lon)
                        lats.append(lat)

                    patch_x_t_batch, patch_context_batch, context_global_batch = [], [], []

                    for j, (x_start, y_start) in enumerate(coords_batch):
                        patch_x_t = self._extract_patch(x_t, x_start, y_start, self.patch_size)
                        patch_context = self._extract_patch(
                            current_context, x_start, y_start, self.patch_size
                        )

                        result = get_position(prediction_date, lons[j], lats[j])
                        date_noon = prediction_date.replace(hour=12, minute=0, second=0, microsecond=0)
                        result_noon = get_position(date_noon, lons[j], lats[j])

                        spatial_position = torch.tensor(
                            [result["azimuth"], result["altitude"],
                             result_noon["altitude"], lats[j] / 25.0],
                            device=self.device,
                        )

                        # context_global = [az, alt, noon_alt, lat/10, d, t]
                        # t LAST (JiT3D_Modern reads time_val = t[:, -1])
                        context_global = torch.cat([
                            spatial_position.unsqueeze(0),
                            d_batch_val.unsqueeze(-1),
                            t_batch_val.unsqueeze(-1),
                        ], dim=1)

                        patch_x_t_batch.append(patch_x_t)
                        patch_context_batch.append(patch_context)
                        context_global_batch.append(context_global)

                    model_input = torch.cat([
                        torch.cat(patch_context_batch, dim=0),
                        torch.cat(patch_x_t_batch, dim=0),
                    ], dim=2)

                    # DualJiT3D.forward(sat_input, kpi_input, context, metar_ref)
                    # sat_input = sat channels, kpi_input = metar channels
                    model_input_sat = model_input[:, :c_sat]
                    model_input_metar = model_input[:, c_sat:]

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device == "cuda" else torch.amp.autocast('cpu', enabled=False):
                        sat_pred_batch, metar_pred_batch = self.model(
                            model_input_sat.float(),
                            model_input_metar.float(),
                            torch.cat(context_global_batch, dim=0).float(),
                            metar_ref=model_input_metar.float(),
                        )

                    del model_input, model_input_sat, model_input_metar, context_global_batch
                    del patch_x_t_batch, patch_context_batch

                    # x_pred: predicted target frames (skip context frames)
                    x_pred_batch = torch.cat([sat_pred_batch, metar_pred_batch], dim=1)[
                        :, :, self.context_frames:, :, :
                    ].to(torch.bfloat16)
                    del sat_pred_batch, metar_pred_batch

                    pw = patch_weights.to(torch.bfloat16)

                    for j, (x_start, y_start) in enumerate(coords_batch):
                        x_t_patch = x_t[
                            ..., y_start: y_start + self.patch_size,
                            x_start: x_start + self.patch_size
                        ].to(torch.bfloat16)

                        # Velocity field s_theta = (x_t - x_pred) / t
                        if self.interpolation == "polynomial":
                            v_patch = (x_t_patch - x_pred_batch[j: j + 1]) / (2 * t_val + 1e-8)
                        else:
                            v_patch = (x_t_patch - x_pred_batch[j: j + 1]) / t_val

                        aggregated_velocity[
                            ..., y_start: y_start + self.patch_size,
                            x_start: x_start + self.patch_size
                        ] += v_patch * pw
                        weights_sum[
                            ..., y_start: y_start + self.patch_size,
                            x_start: x_start + self.patch_size
                        ] += pw

                    del x_pred_batch, pw

                weights_sum[weights_sum == 0] = 1.0
                aggregated_velocity.div_(weights_sum)
                averaged_velocity = aggregated_velocity.float()
                del aggregated_velocity, weights_sum

                # Euler step: x_t = x_t - v * dt
                x_t.sub_(averaged_velocity, alpha=dt)
                x_t.clamp_(-7, 8)
                del averaged_velocity

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Truncate to the number of frames we actually need this step
            x_t = x_t[:, :, :this_nb, :, :]

            if self.use_residual:
                last_ctx = current_context[:, :, -1:, :, :]
                x_t = x_t + last_ctx.expand_as(x_t)

            # Denormalize to physical units
            sat_frame = x_t[:, :c_sat, :, :, :]
            metar_frame = x_t[:, c_sat:, :, :, :]
            sat_denorm, metar_denorm = denormalize(sat_frame, metar_frame, self.device)

            # Blank sat no-data pixels (GMGSI off-disk / polar gaps)
            if sat_nodata_mask is not None:
                nodata_last = sat_nodata_mask[:, :, -1:, :, :].expand_as(sat_denorm)
                sat_denorm = torch.where(nodata_last, torch.zeros_like(sat_denorm), sat_denorm)

            yield sat_denorm.cpu(), metar_denorm.cpu()

            # Update context: drop oldest frames, append generated frames
            if this_nb >= T_ctx:
                new_context = x_t[:, :, -T_ctx:, :, :].clone()
            else:
                tail = current_context[:, :, this_nb:, :, :]
                new_context = torch.cat([tail, x_t[:, :, :this_nb, :, :]], dim=2)
            del current_context
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current_context = new_context
            current_step += this_nb

    # ------------------------------------------------------------------ #
    # Output saving
    # ------------------------------------------------------------------ #

    def _save_timestep_files(
        self,
        sat_frame: torch.Tensor,
        metar_frame: torch.Tensor,
        output_dir: str,
        pred_date: datetime,
        geo_transform,
        crs: str,
        upload_fn: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        """Write GeoTIFF files for one forecast timestep.

        Saves:
          - One multi-band TIFF for satellite (2 channels: IR/LWIR + VIS)
          - One multi-band TIFF for METAR (7 channels)
        """
        base_filename = f"forecast_{pred_date.strftime('%Y%m%d%H%M')}"
        saved = []

        sat_np = sat_frame.squeeze(0).numpy().astype(np.float32)  # (C, nb, H, W)
        metar_np = metar_frame.squeeze(0).numpy().astype(np.float32)

        # sat: squeeze time dim (nb=1 per call), write as multi-band
        if sat_np.ndim == 4:
            sat_np = sat_np[:, 0, :, :]  # (C_sat, H, W)
        if metar_np.ndim == 4:
            metar_np = metar_np[:, 0, :, :]  # (C_metar, H, W)

        # Keep only the IR (LWIR, idx 0) and VIS (idx 1) satellite channels.
        # Full sat layout is [gmgsi_lwir, gmgsi_vis, gmgsi_wv, gmgsi_sw, elevation].
        sat_np = sat_np[:2]  # (2, H, W): [gmgsi_lwir, gmgsi_vis]

        common_kwargs = dict(
            driver='GTiff', crs=crs, transform=geo_transform,
            compress='deflate', tiled=True, blockxsize=512, blockysize=512,
        )

        h, w = sat_np.shape[1], sat_np.shape[2]

        # Satellite GeoTIFF (2 bands: IR + VIS)
        sat_path = os.path.join(output_dir, f"{base_filename}_sat.tif")
        with rasterio.open(
            sat_path, 'w', height=h, width=w,
            count=sat_np.shape[0], dtype='float32',
            nodata=np.nan, **common_kwargs,
        ) as dst:
            for ch in range(sat_np.shape[0]):
                dst.write(sat_np[ch], ch + 1)
        convert_to_cog(sat_path)
        if upload_fn:
            upload_fn(sat_path)
        saved.append(sat_path)

        # METAR GeoTIFF (7 bands)
        metar_path = os.path.join(output_dir, f"{base_filename}_metar.tif")
        with rasterio.open(
            metar_path, 'w', height=h, width=w,
            count=metar_np.shape[0], dtype='float32',
            nodata=np.nan, **common_kwargs,
        ) as dst:
            for ch in range(metar_np.shape[0]):
                dst.write(metar_np[ch], ch + 1)
        convert_to_cog(metar_path)
        if upload_fn:
            upload_fn(metar_path)
        saved.append(metar_path)

        logger.info(f"Saved forecast: {base_filename}_sat.tif + {base_filename}_metar.tif")
        return saved

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #

    def run_inference(
        self,
        data_path: str,
        output_dir: str,
        forecast_steps: int = 8,
        nb_forecast: int = 3,
        upload_fn: Optional[Callable[[str], None]] = None,
    ) -> InferenceResult:
        """Run full inference from an H5 input file.

        Args:
            data_path: Path to the FlashEdges H5 file (from backend_flashedges).
            output_dir: Directory to save GeoTIFF outputs.
            forecast_steps: Total number of forecast hours to produce.
            nb_forecast: Frames generated per autoregressive step.
            upload_fn: Optional callback called with each saved file path.

        Returns:
            InferenceResult with status, output path, and metrics.
        """
        start_time = datetime.now()
        result = InferenceResult(status=InferenceStatus.RUNNING, created_at=start_time)

        try:
            if not os.path.exists(data_path):
                raise ValueError(f"Data file {data_path} not found")

            os.makedirs(output_dir, exist_ok=True)

            # --- Load data from H5 ---
            with h5py.File(data_path, "r") as hf:
                sat_data = hf["sat_data"][:]         # (T, 4, H, W) float16
                metar_data = hf["metar_data"][:]     # (T, 7, H, W) float32
                elevation_data = hf["elevation_data"][:]  # (H, W) float32

                num_frames = int(hf.attrs["num_frames"])
                transform = list(hf.attrs["transform"])
                epsg = int(hf.attrs["epsg"])
                frame_timestamps = list(hf.attrs.get("frame_timestamps", []))

            c_sat = NUM_SAT_CHANNELS   # 5 (GMGSI 4 + elevation 1)
            c_metar = NUM_METAR_CHANNELS  # 7

            if num_frames < self.context_frames:
                raise ValueError(
                    f"Not enough frames: need {self.context_frames}, found {num_frames}"
                )

            # --- Parse reference date from attrs or filename ---
            if frame_timestamps:
                date_str = str(frame_timestamps[-1])  # last context frame = T
                initial_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
            else:
                # Fallback: parse from series_date attr or filename
                series_date = str(hf.attrs.get("series_date", ""))
                if series_date:
                    initial_date = datetime.strptime(series_date, "%Y-%m-%d %H:%M:%S")
                else:
                    raise ValueError("Cannot determine reference date from H5 attrs")

            logger.info(f"Input H5: {num_frames} frames, ref date {initial_date}")
            logger.info(f"  sat_data: {sat_data.shape}, metar_data: {metar_data.shape}")
            logger.info(f"  elevation: {elevation_data.shape}")

            # --- Build initial context tensor ---
            # sat_mask: True where GMGSI data is valid (not NaN)
            # metar_mask: True where a station reported
            initial_frames = []
            sat_nodata_masks = []

            for i in range(self.context_frames):
                sat_frame = sat_data[i].astype(np.float32)  # (4, H, W)
                metar_frame = metar_data[i].astype(np.float32)  # (7, H, W)
                elev_frame = elevation_data[None, :, :]  # (1, H, W)

                # Elevation floor (FlashNet/dataset convention)
                elev_frame = np.where(elev_frame < 0, ELEVATION_FLOOR, elev_frame)

                # Capture sat no-data mask BEFORE filling NaN
                sat_valid = ~np.isnan(sat_frame)
                sat_frame = np.where(np.isnan(sat_frame), 0.0, sat_frame)

                # Capture metar mask, then replace NaN with sentinel
                metar_valid = ~np.isnan(metar_frame)
                metar_frame = np.where(np.isnan(metar_frame), METAR_NAN_SENTINEL, metar_frame)

                # Concatenate: [sat(4) + elev(1) | metar(7)] = 12 channels
                sat_elev = np.concatenate([sat_frame, elev_frame], axis=0)  # (5, H, W)
                frame = np.concatenate([sat_elev, metar_frame], axis=0)[None, ...]  # (1, 12, H, W)
                initial_frames.append(frame)
                sat_nodata_masks.append((~sat_valid)[None, ...])  # (1, 4, H, W)

            current_context = np.stack(initial_frames, axis=2)  # (1, 12, T_ctx, H, W)
            current_context = torch.from_numpy(current_context).float().to(self.device)

            sat_nodata_mask = np.stack(sat_nodata_masks, axis=2)  # (1, 4, T_ctx, H, W)
            sat_nodata_mask = torch.from_numpy(sat_nodata_mask).to(self.device)
            # Expand to include elevation channel (always valid)
            elev_valid = torch.zeros_like(sat_nodata_mask[:, :1])  # all False = valid
            sat_nodata_mask = torch.cat([sat_nodata_mask, elev_valid], dim=1)  # (1, 5, T_ctx, H, W)

            # --- Normalize ---
            sat_tensor = current_context[:, :c_sat]
            metar_tensor = current_context[:, c_sat:]

            # METAR: replace sentinel with 0 AFTER capturing mask, BEFORE normalize
            metar_valid = (metar_tensor != METAR_NAN_SENTINEL)
            metar_tensor = torch.where(
                metar_valid, metar_tensor, torch.zeros_like(metar_tensor)
            )

            sat_tensor, metar_tensor = normalize(sat_tensor, metar_tensor, self.device)
            # Zero out no-data pixels after normalize (sentinel = neutral mean 0)
            sat_tensor = torch.where(
                ~sat_nodata_mask, sat_tensor, torch.zeros_like(sat_tensor)
            )
            metar_tensor = torch.where(
                metar_valid, metar_tensor, torch.zeros_like(metar_tensor)
            )

            current_context = torch.cat([sat_tensor, metar_tensor], dim=1)

            # --- Geo transform for output TIFFs ---
            # Global GMGSI grid: 0.1° resolution, origin at (-180, 90)
            from rasterio.transform import Affine
            geo_transform = Affine(transform[0], 0, transform[2], 0, transform[4], transform[5])
            crs = f"EPSG:{epsg}"

            # --- Run tiled inference ---
            output_files = []
            pending_futures: List[Future] = []
            global_step = 0

            with ThreadPoolExecutor(max_workers=2) as executor:
                for sat_batch, metar_batch in self.tiled_inference(
                    initial_context=current_context,
                    forecast_steps=forecast_steps,
                    nb_forecast=nb_forecast,
                    date=initial_date,
                    c_sat=c_sat,
                    c_metar=c_metar,
                    sat_nodata_mask=sat_nodata_mask,
                ):
                    batch_nb = sat_batch.shape[2]
                    for k in range(batch_nb):
                        pred_date = initial_date + timedelta(hours=global_step + k + 1)
                        sat_frame = sat_batch[:, :, k, :, :]
                        metar_frame = metar_batch[:, :, k, :, :]

                        fut = executor.submit(
                            self._save_timestep_files,
                            sat_frame, metar_frame,
                            output_dir, pred_date, geo_transform, crs, upload_fn,
                        )
                        pending_futures.append(fut)

                    global_step += batch_nb

                for fut in pending_futures:
                    output_files.extend(fut.result())

            result.status = InferenceStatus.COMPLETED
            result.output_path = output_dir
            result.completed_at = datetime.now()
            result.metrics = {
                "output_files": len(output_files),
                "duration_seconds": (result.completed_at - start_time).total_seconds(),
            }
            logger.info(
                f"Inference completed: {len(output_files)} files in "
                f"{result.metrics['duration_seconds']:.1f}s"
            )

        except Exception as e:
            logger.exception("Inference failed")
            result.status = InferenceStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()

        return result

    def cleanup(self) -> None:
        if self.model:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
