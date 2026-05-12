import os
from pathlib import Path

import modal

APP_NAME = "webtoon-cleaner"
BASE_DIR = Path(__file__).resolve().parent
DATA_VOLUME_NAME = os.getenv("MODAL_DATA_VOLUME_NAME", "webtoon-cleaner-data-v2")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
HF_SECRET_NAME = os.getenv("MODAL_HF_SECRET_NAME", "hf-secret")
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])


def _resolve_modal_gpu() -> str:
    requested = os.getenv("MODAL_GPU", "L40S").strip()
    # Modal does not currently expose an explicit RTX 4090 SKU.
    # L40S is a stable Ada option and closest practical equivalent here.
    if requested.upper() in {"RTX4090", "RTX-4090", "4090"}:
        return "L40S"
    if requested.upper() == "RTX-PRO-6000":
        # Blackwell SKU may require newer torch/CUDA kernels than this stack pins.
        return "L40S"
    return requested


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(BASE_DIR / "requirements.txt"))
    .env({"BUILD_REV": "2026-05-12-oom-guard-1"})
    .add_local_dir(
        str(BASE_DIR),
        remote_path="/root/app",
        ignore=[
            "__pycache__",
            "*.pyc",
            "build",
            "dist",
            "scratch",
            "uploads",
            "processed",
            ".venv",
        ],
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=[_resolve_modal_gpu(), "A10"],
    timeout=60 * 60,
    scaledown_window=2,
    volumes={"/data": data_volume},
    secrets=[hf_secret],
)
@modal.asgi_app()
def fastapi_app():
    import sys

    os.environ.setdefault("APP_DATA_DIR", "/data")
    os.environ.setdefault("MODAL_VOLUME_NAME", DATA_VOLUME_NAME)
    os.environ.setdefault("TEXT_MASK_MODEL_PATH", "/data/models/text_mask_detector.onnx")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("FLUX_LOW_VRAM_MODE", "1")
    os.environ.setdefault("FLUX_MIN_VRAM_GB", "30")

    if "/root/app" not in sys.path:
        sys.path.insert(0, "/root/app")

    from main import app as web_app

    return web_app
