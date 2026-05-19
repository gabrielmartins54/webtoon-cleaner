"""
modal_backend.py — Modal GPU cloud deployment for webtoon-cleaner.

Deploy:
    py -3.12 -m modal deploy modal_backend.py

Serve locally (for testing):
    py -3.12 -m modal serve modal_backend.py

Environment variables (set via Modal secrets or .env):
    MODAL_GPU            GPU SKU to use (default: L40S). Options: L40S, A100, A10G, T4
    MODAL_HF_SECRET      Name of the Modal HF secret (default: hf-secret, optional)
    MODAL_DATA_VOLUME_NAME  Name of the persistent data volume (default: webtoon-cleaner-data-v2)
    HF_TOKEN             HuggingFace token (used inside the container if HF models are needed)
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "webtoon-cleaner"
BASE_DIR = Path(__file__).resolve().parent

# Persistent volume — stores uploaded files, processed outputs, and model caches.
DATA_VOLUME_NAME = os.getenv("MODAL_DATA_VOLUME_NAME", "webtoon-cleaner-data-v2")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

# HuggingFace secret name — create this in your Modal dashboard if you use
# gated HuggingFace models (Flux, etc.).
# Dashboard: https://modal.com/secrets
HF_SECRET_NAME = os.getenv("MODAL_HF_SECRET_NAME", "hf-secret")
# Set USE_HF_SECRET=1 to include the HF secret in the deployment.
_USE_HF_SECRET = os.getenv("USE_HF_SECRET", "0").strip() == "1"
_SECRETS: list[modal.Secret] = (
    [modal.Secret.from_name(HF_SECRET_NAME)] if _USE_HF_SECRET else []
)

# GPU selection — falls back gracefully so the app still runs on alternatives.
GPU_SKU = os.getenv("MODAL_GPU", "L40S").strip().upper()
_GPU_ALIASES: dict[str, str] = {
    "RTX4090": "L40S",
    "RTX-4090": "L40S",
    "4090": "L40S",
    "RTX-PRO-6000": "L40S",
    "BLACKWELL": "L40S",
}
_RESOLVED_GPU = _GPU_ALIASES.get(GPU_SKU, GPU_SKU)

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    # System libraries required by OpenCV, EasyOCR, and Pillow.
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libsm6",
        "libxext6",
        "libxrender-dev",
    )
    # Install Python dependencies from the backend requirements file.
    # NOTE: onnxruntime-gpu is included — Modal's CUDA environment supports it.
    .pip_install_from_requirements(str(BASE_DIR / "requirements.txt"))
    # Cache-bust marker — bump this string to force a full image rebuild.
    .env({"BUILD_REV": "2026-05-16-modal-gpu-v2"})
    # Copy the entire backend directory into the container.
    .add_local_dir(
        str(BASE_DIR),
        remote_path="/root/app",
        ignore=[
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "build",
            "dist",
            "scratch",
            "uploads",
            "processed",
            ".venv",
            "venv",
            "*.egg-info",
        ],
    )
)

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)




# ---------------------------------------------------------------------------
# ASGI app (FastAPI served on Modal GPU)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    # Primary GPU + automatic fallback to A10G if L40S is unavailable.
    gpu=[_RESOLVED_GPU, "A10G"],
    # 60-minute timeout per request (large images can take a while).
    timeout=60 * 60,
    # Keep the container warm for 10 seconds after last request to avoid
    # cold-start penalty on back-to-back uploads.
    scaledown_window=10,
    # Persistent volume mounted at /data for uploads, outputs, and models.
    volumes={"/data": data_volume},
    # Secrets: empty by default; set USE_HF_SECRET=1 env var to include hf-secret.
    secrets=_SECRETS,
)
@modal.concurrent(max_inputs=3)
@modal.asgi_app()
def fastapi_app():
    """Entry point: configure environment then hand off to the FastAPI app."""
    import sys

    # --- Data directories on the persistent volume ---
    os.environ.setdefault("APP_DATA_DIR", "/data")
    os.environ.setdefault("MODAL_VOLUME_NAME", DATA_VOLUME_NAME)

    # --- Model paths on the persistent volume ---
    os.environ.setdefault("TEXT_MASK_MODEL_PATH", "/data/models/text_mask_detector.onnx")

    # --- GPU / CUDA tuning ---
    # Prevent OOM by allowing PyTorch to release memory to the OS.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Low-VRAM mode for diffusion models (reduces peak VRAM usage).
    os.environ.setdefault("FLUX_LOW_VRAM_MODE", "1")
    os.environ.setdefault("FLUX_MIN_VRAM_GB", "30")

    # onnxruntime: prefer CUDA execution provider inside the Modal container.
    os.environ.setdefault("ORT_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")

    # --- Python path ---
    if "/root/app" not in sys.path:
        sys.path.insert(0, "/root/app")

    # Import the FastAPI application from the backend.
    from main import app as web_app  # noqa: PLC0415

    return web_app
