"""
CLI entrypoint for FlashEdges tiled diffusion inference.

Two modes:

  - ``local``  : run inference on a local H5 file (--data_path) and write
                 GeoTIFFs to a local output dir. This is the original
                 development / SLURM workflow.
  - ``cloud``  : GCP Cloud Run GPU job workflow:
                 1. List ``gs://eumetsat_mtg_preprocess/inference_h5_global``
                    and pick the H5 with the latest date in its filename
                    (``global_live_YYYYMMDD_HHMM.h5``).
                 2. Download it (and the model weights if not cached) to a
                    local scratch dir.
                 3. Run tiled diffusion inference, uploading each GeoTIFF to
                    ``gs://inference_result_flashedges_forecast/forecasts/{DATE}/
                    {RUN_YYYYMMDD_HHMM}/<tif>`` as soon as it is written, where
                    {RUN_YYYYMMDD_HHMM} is the time the run started (not the
                    forecast target time).

Examples
--------
# Local dev
python -m backend.main --mode local --model_path models/checkpoint.safetensors \
    --data_path data_h5/global_live_20260816_0900.h5 --output_dir forecasts/

# Cloud job (uses env vars / config defaults for buckets & model)
python -m backend.main --mode cloud
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Ensure project root is on sys.path so meteolibre_model resolves
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# HDF5 file locking off (mirrors flashnet — H5 may be read concurrently)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("HDF5_PLUGIN_PATH", "")

from backend.inference_engine import FlashEdgesInferenceEngine


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cloud pipeline
# ---------------------------------------------------------------------------

def _make_upload_fn(gcs_client, input_date_folder: str):
    """Build the per-file upload callback for the inference engine.

    Output layout:
        gs://<dest_bucket>/forecasts/{input_date}/{RUN_YYYYMMDD_HHMM}/<filename>

    where ``RUN_YYYYMMDD_HHMM`` is the time the inference RUN started
    (captured once when this callback is built), not the forecast target
    time embedded in the TIFF filename. All files of a single run therefore
    land in the same folder even if generation spans a minute boundary.
    """
    from backend.config import get_config

    config = get_config()
    dest_prefix = config.gcp.dest_prefix  # "forecasts"

    # Run time captured ONCE, so every file of the run shares one folder.
    run_folder = datetime.now().strftime("%Y%m%d_%H%M")

    def upload_fn(filepath: str):
        try:
            filepath = Path(filepath)
            name = filepath.name

            dest_blob = f"{dest_prefix}/{input_date_folder}/{run_folder}/{name}"
            gcs_client.upload_file(str(filepath), dest_blob,
                                   content_type="image/tiff")
        except Exception:
            logger.exception(f"Failed to upload {filepath}")

    return upload_fn


def download_latest_h5(gcs_client) -> str:
    """Download the latest-dated H5 input from the source bucket."""
    from backend.config import get_config

    config = get_config()
    cache_dir = Path(config.cache.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    latest = gcs_client.get_latest_file()
    if not latest:
        raise FileNotFoundError(
            f"No .h5 files found in gs://{config.gcp.source_bucket}/"
            f"{config.gcp.source_prefix}/"
        )

    local_path = cache_dir / latest.name
    gcs_client.download_file(latest.gcs_path, str(local_path))
    logger.info(f"Downloaded {latest.name} to {local_path}")
    return str(local_path)


def run_cloud_pipeline(target_date: Optional[datetime] = None) -> dict:
    """Run the full download -> inference -> upload pipeline."""
    from backend.config import get_config
    from backend.gcp_client import GCPStorageClient, parse_h5_date

    start_time = datetime.now()
    config = get_config()
    logger.info(f"Starting FlashEdges cloud pipeline at {start_time.isoformat()}")

    gcs_client = GCPStorageClient(config.gcp)

    try:
        # 1. Latest H5 input
        data_path = download_latest_h5(gcs_client)
        h5_name = os.path.basename(data_path)
        input_date_folder = parse_h5_date(h5_name)  # YYYYMMDD
        logger.info(f"Input H5: {h5_name} (run date folder: {input_date_folder})")

        # 2. Inference engine (downloads model from MODEL_GCS_PATH if needed)
        cache_dir = Path(config.cache.cache_dir)
        output_dir = cache_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        engine = FlashEdgesInferenceEngine(
            model_path=config.model.model_path,
            config_name=config.model.config_name,
            patch_size=config.model.patch_size,
            denoising_steps=config.model.denoising_steps,
            batch_size=config.model.batch_size,
            context_frames=config.model.context_frames,
            interpolation=config.model.interpolation,
            sampler=config.model.sampler,
            sde_eps=config.model.sde_eps,
            sde_eps_schedule=config.model.sde_eps_schedule,
            inference_seed=config.model.inference_seed,
            metar_keep_ratio=config.model.metar_keep_ratio,
        )

        upload_fn = _make_upload_fn(gcs_client, input_date_folder)

        result = engine.run_inference(
            data_path=data_path,
            output_dir=str(output_dir),
            forecast_steps=config.model.forecast_steps,
            nb_forecast=config.model.nb_forecast,
            upload_fn=upload_fn,
        )

        if result.status.value != "completed":
            raise RuntimeError(f"Inference failed: {result.error_message}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Cloud pipeline completed in {duration:.1f}s "
                    f"({result.metrics['output_files']} files uploaded)")

        return {
            "status": "success",
            "input_file": h5_name,
            "run_date_folder": input_date_folder,
            "output_files": result.metrics["output_files"],
            "duration_seconds": duration,
            "completed_at": end_time.isoformat(),
        }

    except Exception as e:
        logger.exception("Cloud pipeline failed")
        return {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat(),
        }
    finally:
        gcs_client.close()


# ---------------------------------------------------------------------------
# Local pipeline
# ---------------------------------------------------------------------------

def run_local(args):
    """Original local-H5 inference workflow."""
    engine = FlashEdgesInferenceEngine(
        model_path=args.model_path,
        config_name=args.config_name,
        patch_size=args.patch_size,
        denoising_steps=args.denoising_steps,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        interpolation=args.interpolation,
        sampler=args.sampler,
        sde_eps=args.sde_eps,
        sde_eps_schedule=args.sde_eps_schedule,
        inference_seed=args.inference_seed,
        mask_all_metar=args.mask_all_metar,
        metar_keep_ratio=args.metar_keep_ratio,
        device=args.device,
    )

    result = engine.run_inference(
        data_path=args.data_path,
        output_dir=args.output_dir,
        forecast_steps=args.forecast_steps,
        nb_forecast=args.nb_forecast,
    )

    if result.status.value == "completed":
        print(f"\n✅ Inference completed successfully!")
        print(f"   Output: {result.output_path}")
        print(f"   Files:  {result.metrics['output_files']}")
        print(f"   Time:   {result.metrics['duration_seconds']:.1f}s")
    else:
        print(f"\n❌ Inference failed: {result.error_message}")
        sys.exit(1)

    engine.cleanup()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FlashEdges global tiled diffusion inference.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "cloud"],
        help="local: run on a local H5 (--data_path). cloud: GCP job — "
             "download latest H5 from the source bucket, run inference, "
             "upload GeoTIFFs to the dest bucket (default: local).",
    )

    # --- local-mode args ---
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the .safetensors model weights (local mode).")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to the input H5 file (local mode).")
    parser.add_argument("--output_dir", type=str, default="forecasts",
                        help="Directory to save GeoTIFF outputs (local mode).")
    parser.add_argument("--config_name", type=str,
                        default="model_v4_global_satellite_metar",
                        help="Config key in meteolibre_model/config/configs.yml.")
    parser.add_argument("--forecast_steps", type=int, default=24)
    parser.add_argument("--nb_forecast", type=int, default=3)
    parser.add_argument("--denoising_steps", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--interpolation", type=str, default="linear",
                        choices=["linear", "polynomial"])
    parser.add_argument("--sampler", type=str, default="sde",
                        choices=["sde", "ode"])
    parser.add_argument("--sde_eps", type=float, default=0.1)
    parser.add_argument("--sde_eps_schedule", type=str, default="t2",
                        choices=["const", "t", "t2"])
    parser.add_argument("--inference_seed", type=int, default=None)
    parser.add_argument("--mask_all_metar", action="store_true")
    parser.add_argument(
        "--metar_keep_ratio", type=float, default=0.0,
        help="Fraction in [0,1] of NON-station METAR pixels whose predicted "
             "values are kept in the autoregressive feedback instead of "
             "re-masking them to 0 (e.g. 0.05 = keep a random 5%% of "
             "non-station pixels as 'virtual stations' in addition to real "
             "station positions). Stabilizes patches with very few METAR "
             "stations. 0.0 = strict re-sparsification (default).",
    )
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (auto-detected if not specified).")

    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.mode == "cloud":
        result = run_cloud_pipeline()
        if result["status"] == "success":
            print(f"\n✅ Cloud pipeline completed!")
            print(f"   Input:      {result['input_file']}")
            print(f"   Run folder: forecasts/{result['run_date_folder']}/")
            print(f"   Files:      {result['output_files']}")
            print(f"   Time:       {result['duration_seconds']:.1f}s")
        else:
            print(f"\n❌ Cloud pipeline failed: {result['error']}")
            sys.exit(1)
        return

    # local mode
    if not args.model_path or not args.data_path:
        parser.error("local mode requires --model_path and --data_path")
    run_local(args)


if __name__ == "__main__":
    main()
