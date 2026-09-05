"""
Backend configuration for the FlashEdges inference service.

Mirrors flashnet/backend/config.py but tuned for the FlashEdges global
satellite + METAR model:
  - source bucket / prefix : gs://eumetsat_mtg_preprocess/inference_h5_global
  - dest bucket / prefix   : gs://inference_result_flashedges_forecast/forecasts

All values are overridable through environment variables.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from dotenv import load_dotenv

# Allow a local .env next to this file for development.
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class GCPConfig(BaseModel):
    """Google Cloud Platform configuration."""
    source_bucket: str = "eumetsat_mtg_preprocess"
    source_prefix: str = "inference_h5_global"
    dest_bucket: str = "inference_result_flashedges_forecast"
    dest_prefix: str = "forecasts"
    credentials_path: Optional[str] = None
    project_id: Optional[str] = None


class ModelConfig(BaseModel):
    """FlashEdges model configuration."""
    model_path: str = "/tmp/flashedges_cache/model.safetensors"
    model_gcs_path: str = "gs://eumetsat_mtg_preprocess/assets/flashedges_v1.safetensors"
    config_name: str = "model_v4_global_satellite_metar"
    patch_size: int = 129
    denoising_steps: int = 32
    batch_size: int = 64
    forecast_steps: int = 24
    nb_forecast: int = 3
    context_frames: int = 4
    interpolation: str = "linear"
    sampler: str = "sde"
    sde_eps: float = 0.1
    sde_eps_schedule: str = "t2"
    inference_seed: Optional[int] = 128
    # Fraction of NON-station METAR pixels kept (predicted values) in the AR
    # feedback, in addition to real station positions (0.0 = strict
    # re-sparsification; ~0.05 stabilizes station-sparse patches).
    metar_keep_ratio: float = 0.0
    # Path to the packed radar coverage NPZ used to mask the radar channel
    # (v2 6-channel configs only). None auto-resolves data_info/radar_cov_test.npz
    # from the CWD or the repo root (same resolution as the v2 training dataset).
    radar_cov_path: Optional[str] = None


class CacheConfig(BaseModel):
    """Cache / scratch configuration."""
    cache_dir: str = "/tmp/flashedges_cache"
    max_cache_size_gb: float = 20.0
    cache_expiry_hours: int = 24


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None


class Config(BaseModel):
    """Main configuration class."""
    gcp: GCPConfig = GCPConfig()
    model: ModelConfig = ModelConfig()
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            gcp=GCPConfig(
                source_bucket=os.getenv("GCP_SOURCE_BUCKET", "eumetsat_mtg_preprocess"),
                source_prefix=os.getenv("GCP_SOURCE_PREFIX", "inference_h5_global"),
                dest_bucket=os.getenv("GCP_DEST_BUCKET", "inference_result_flashedges_forecast"),
                dest_prefix=os.getenv("GCP_DEST_PREFIX", "forecasts"),
                credentials_path=os.getenv("GCP_CREDENTIALS_PATH"),
                project_id=os.getenv("GCP_PROJECT_ID"),
            ),
            model=ModelConfig(
                model_path=os.getenv("MODEL_PATH", "/tmp/flashedges_cache/model.safetensors"),
                model_gcs_path=os.getenv("MODEL_GCS_PATH",
                                         "gs://eumetsat_mtg_preprocess/assets/flashedges_v1.safetensors"),
                config_name=os.getenv("CONFIG_NAME", "model_v4_global_satellite_metar"),
                patch_size=int(os.getenv("PATCH_SIZE", 129)),
                denoising_steps=int(os.getenv("DENOISING_STEPS", 32)),
                batch_size=int(os.getenv("BATCH_SIZE", 64)),
                forecast_steps=int(os.getenv("FORECAST_STEPS", 24)),
                nb_forecast=int(os.getenv("NB_FORECAST", 3)),
                context_frames=int(os.getenv("CONTEXT_FRAMES", 4)),
                interpolation=os.getenv("INTERPOLATION", "linear"),
                sampler=os.getenv("SAMPLER", "sde"),
                sde_eps=float(os.getenv("SDE_EPS", 0.1)),
                sde_eps_schedule=os.getenv("SDE_EPS_SCHEDULE", "t2"),
                inference_seed=(int(os.getenv("INFERENCE_SEED", "128"))
                                if os.getenv("INFERENCE_SEED") not in (None, "") else None),
                metar_keep_ratio=float(os.getenv("METAR_KEEP_RATIO", "0.0")),
                radar_cov_path=os.getenv("RADAR_COV_PATH") or None,
            ),
            cache=CacheConfig(
                cache_dir=os.getenv("CACHE_DIR", "/tmp/flashedges_cache"),
                max_cache_size_gb=float(os.getenv("MAX_CACHE_SIZE_GB", 20.0)),
                cache_expiry_hours=int(os.getenv("CACHE_EXPIRY_HOURS", 24)),
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                log_file=os.getenv("LOG_FILE"),
            ),
        )


_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance (lazy singleton)."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None
