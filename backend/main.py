"""
CLI entrypoint for FlashEdges tiled diffusion inference.

Reads a global H5 input file (produced by
``meteolibre_datasetgen.src.backend_flashedges``) and runs the FlashEdges model
over the full 1800x3600 grid using tiled diffusion with autoregressive rollout.

Mirrors ``flashnet/backend/main.py`` but emits a CLI script (no FastAPI server).

Examples
--------
# Basic inference (defaults from configs.yml)
python -m backend.main --model_path models/flashedges_v1.safetensors --data_path /path/to/input.h5

# Custom forecast horizon and output dir
python -m backend.main \
    --model_path models/flashedges_v1.safetensors \
    --data_path /path/to/input.h5 \
    --forecast_steps 16 \
    --output_dir forecasts/

# With specific device and patch size
python -m backend.main \
    --model_path models/flashedges_v1.safetensors \
    --data_path /path/to/input.h5 \
    --device cuda \
    --patch_size 128
"""

import argparse
import os
import sys

# Ensure project root is on sys.path so meteolibre_model resolves
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.inference_engine import FlashEdgesInferenceEngine


def main():
    parser = argparse.ArgumentParser(
        description="FlashEdges global tiled diffusion inference."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the .safetensors model weights.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input H5 file (from backend_flashedges).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="forecasts",
        help="Directory to save GeoTIFF forecast outputs (default: forecasts/).",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="model_v1_global_satellite_metar",
        help="Config key in meteolibre_model/config/configs.yml.",
    )
    parser.add_argument(
        "--forecast_steps",
        type=int,
        default=18,
        help="Total number of forecast hours to produce (default: 8).",
    )
    parser.add_argument(
        "--nb_forecast",
        type=int,
        default=3,
        help="Frames generated per autoregressive step (default: 3).",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=128,
        help="Number of Euler denoising steps for the RF ODE (default: 128).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Spatial patch size for tiled inference (default: 128).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of patches per model forward pass (default: 64).",
    )
    parser.add_argument(
        "--context_frames",
        type=int,
        default=4,
        help="Number of context frames from the input (default: 4).",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="linear",
        choices=["linear", "polynomial"],
        help="RF interpolation schedule (default: linear).",
    )
    parser.add_argument(
        "--use_residual",
        action="store_true",
        help="Enable residual targets (if the model was trained that way).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda or cpu (auto-detected if not specified).",
    )

    args = parser.parse_args()

    engine = FlashEdgesInferenceEngine(
        model_path=args.model_path,
        config_name=args.config_name,
        patch_size=args.patch_size,
        denoising_steps=args.denoising_steps,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        interpolation=args.interpolation,
        use_residual=args.use_residual,
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


if __name__ == "__main__":
    main()
