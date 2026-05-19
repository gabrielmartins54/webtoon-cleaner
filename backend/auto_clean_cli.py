import argparse
from pathlib import Path

from main import process_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR/inpainting auto-clean on an image folder.")
    parser.add_argument("--input", required=True, help="Folder containing input images.")
    parser.add_argument("--output", required=True, help="Folder where cleaned PSD outputs are written.")
    parser.add_argument("--high-quality-redraw", action="store_true", help="Enable LaMa + SDXL/FLUX refinement.")
    parser.add_argument("--redraw-engine", default="sdxl", choices=["sdxl", "flux"], help="Generative redraw backend.")
    parser.add_argument("--redraw-prompt", default="clean webtoon panel, preserve original composition and lighting, consistent manga line art, remove text only")
    parser.add_argument("--redraw-denoise", type=float, default=0.35)
    parser.add_argument("--redraw-steps", type=int, default=28)
    parser.add_argument("--redraw-guidance", type=float, default=6.5)
    parser.add_argument("--redraw-lora-path", default=None)
    parser.add_argument("--redraw-lora-scale", type=float, default=0.8)
    parser.add_argument("--post-lama-boundary-cleanup", action="store_true", default=False)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    process_images(
        input_dir,
        output_dir,
        high_quality_redraw=args.high_quality_redraw,
        redraw_engine=args.redraw_engine,
        redraw_prompt=args.redraw_prompt,
        redraw_denoise=args.redraw_denoise,
        redraw_steps=args.redraw_steps,
        redraw_guidance=args.redraw_guidance,
        redraw_lora_path=args.redraw_lora_path,
        redraw_lora_scale=args.redraw_lora_scale,
        post_lama_boundary_cleanup=args.post_lama_boundary_cleanup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
