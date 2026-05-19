from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import warnings
import re
from typing import Optional, Protocol

# Suppress EasyOCR/Torch FutureWarnings about weights_only
warnings.filterwarnings("ignore", category=FutureWarning, module="easyocr")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import SAM
from huggingface_hub import hf_hub_download
from simple_lama_inpainting import SimpleLama
from googleapiclient.discovery import build
from google.oauth2 import service_account

from transformers import AutoImageProcessor, AutoModelForObjectDetection
import pytoshop
from pytoshop.user import nested_layers
from pytoshop.enums import ColorMode, Compression
import easyocr
from advanced_redrawer import AdvancedRedrawer
from onnx_lama_inpainter import OnnxLamaInpainter
from text_mask_detector import TextMaskDetector

try:
    import transformers
    import diffusers

    print(f"STARTUP: transformers={transformers.__version__} diffusers={diffusers.__version__}")
except Exception as _ver_err:
    print(f"STARTUP: Version probe failed: {_ver_err}")

DEFAULT_RTDETR_MODEL_ID = os.getenv("RTDETR_MODEL_ID", "ko_webtoon_v2").strip()
FALLBACK_RTDETR_MODEL_ID = os.getenv("RTDETR_FALLBACK_MODEL_ID", "ogkalu/comic-text-and-bubble-detector").strip()


def _resolve_rtdetr_model_ref(model_id: str) -> str:
    raw = (model_id or "").strip()
    if not raw:
        return raw
    p = Path(raw).expanduser()
    if p.exists():
        return str(p)
    current_dir = Path(__file__).resolve().parent
    for candidate in (current_dir / "models" / raw, current_dir / raw):
        if candidate.exists():
            return str(candidate)
    return raw


def _load_rtdetr_with_fallback() -> tuple[Optional[AutoImageProcessor], Optional[AutoModelForObjectDetection], str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tried: list[str] = []
    for model_id in [DEFAULT_RTDETR_MODEL_ID, FALLBACK_RTDETR_MODEL_ID]:
        if not model_id or model_id in tried:
            continue
        tried.append(model_id)
        resolved = _resolve_rtdetr_model_ref(model_id)
        print(f"STARTUP: Loading RT-DETR ({resolved})...")
        try:
            proc = AutoImageProcessor.from_pretrained(
                resolved,
                trust_remote_code=True,
            )
            model = AutoModelForObjectDetection.from_pretrained(
                resolved,
                trust_remote_code=True,
            ).to(device)
            print(f"STARTUP: RT-DETR loaded ({resolved}) on {device}")
            return proc, model, resolved
        except Exception as exc:
            print(f"WARN: RT-DETR load failed for {resolved}: {exc}")
    return None, None, ""


processor, rt_model, RTDETR_MODEL_REF = _load_rtdetr_with_fallback()
if rt_model is None:
    print("!!! STARTUP ERROR: RT-DETR unavailable after fallback attempts !!!")

class Inpainter(Protocol):
    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image: ...


def _assert_cuda_ready() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but unavailable. Install NVIDIA driver + CUDA-enabled PyTorch in backend/.venv."
        )
    if torch.version.cuda is None:
        raise RuntimeError(
            "Current PyTorch build has no CUDA runtime. Reinstall CUDA-enabled torch/torchvision/torchaudio."
        )


# 2. Load LaMa
print("STARTUP: Loading LaMa...")
lama_cache: dict[str, Optional[Inpainter]] = {}
try:
    _assert_cuda_ready()
    lama_device = torch.device("cuda")
    lama = SimpleLama(device=lama_device)
    print(f"STARTUP: LaMa loaded on {lama_device}")
    lama_cache["default"] = lama
except Exception as e:
    print("!!! STARTUP ERROR: LaMa load fail !!!", e)
    lama_device = torch.device("cpu")
    lama = None
    lama_cache["default"] = None

# Legacy EasyOCR path is kept only for backward compatibility in helper functions.
ocr_reader = None
EASYOCR_RECOG_NETWORK = os.getenv("EASYOCR_RECOG_NETWORK", "ko_webtoon_v2").strip()
EASYOCR_LANG_LIST = [lang.strip() for lang in os.getenv("EASYOCR_LANG_LIST", "ko").split(",") if lang.strip()]


app = FastAPI()

frontend_origins_env = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
)
frontend_origins = [origin.strip() for origin in frontend_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DATA_DIR = Path(os.getenv("APP_DATA_DIR", ".")).resolve()
UPLOAD_DIR = BASE_DATA_DIR / "uploads"
PROCESSED_DIR = BASE_DATA_DIR / "processed"
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)


def _resolve_easyocr_user_network_dir() -> Path:
    explicit = os.getenv("EASYOCR_USER_NETWORK_DIR", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
    current_dir = Path(__file__).resolve().parent
    candidates = [current_dir / "user_network", BASE_DATA_DIR / "user_network"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Return default candidate even if missing so EasyOCR can still try default behavior.
    return current_dir / "user_network"


def _load_easyocr_reader() -> Optional[easyocr.Reader]:
    user_net_dir = _resolve_easyocr_user_network_dir()
    print(
        "STARTUP: Loading EasyOCR "
        f"(langs={EASYOCR_LANG_LIST}, recog_network={EASYOCR_RECOG_NETWORK}, user_network={user_net_dir})..."
    )
    try:
        reader = easyocr.Reader(
            EASYOCR_LANG_LIST or ["ko"],
            recog_network=EASYOCR_RECOG_NETWORK or "ko_webtoon_v2",
            user_network_directory=str(user_net_dir),
            model_storage_directory=str(user_net_dir),
        )
        print("STARTUP: EasyOCR loaded")
        return reader
    except Exception as exc:
        print(f"WARN: EasyOCR unavailable: {exc}")
        return None


ocr_reader = _load_easyocr_reader()

DEFAULT_REDRAW_PROMPT = (
    "clean Korean webtoon panel redraw, remove dialogue text only, preserve original composition, panel borders, "
    "speech bubbles and tails, line weight, screentones, hatching, texture and lighting continuity, "
    "no new objects, no new text, no SFX, no watermark, seamless inpaint matching surrounding art style"
)
DEFAULT_MASK_ENGINE = os.getenv("MASK_ENGINE_DEFAULT", "onnx_textmask").strip().lower()
DEFAULT_REDRAW_STEPS = int(os.getenv("REDRAW_STEPS_DEFAULT", "16"))
DEFAULT_INPAINT_MODEL = os.getenv("INPAINT_MODEL_DEFAULT", "default").strip()
INPAINT_MODEL_ALIASES = {
    "lama": "default",
    "simple_lama": "default",
    "builtin": "default",
    "sdxl": "sdxl_fast_inpaint",
    "sdxl_turbo": "sdxl_fast_inpaint",
    "sdxl_inpaint": "sdxl_fast_inpaint",
    "flux": "flux_fill_inpaint",
    "flux_fill": "flux_fill_inpaint",
    "sd15": "sd15_webtoon_inpaint",
    "sd1.5": "sd15_webtoon_inpaint",
    "sd15_inpaint": "sd15_webtoon_inpaint",
}
DIFFUSION_INPAINT_PRESETS: dict[str, dict[str, float | int | str]] = {
    "sdxl_fast_inpaint": {
        "engine": "sdxl_turbo",
        "denoise": 0.55,
        "steps": 3,
        "guidance": 0.0,
    },
    "flux_fill_inpaint": {
        "engine": "flux",
        "denoise": 0.38,
        "steps": 20,
        "guidance": 20.0,
    },
    "sd15_webtoon_inpaint": {
        "engine": "sd15",
        "denoise": 0.45,
        "steps": 14,
        "guidance": 6.0,
    },
}
advanced_redrawer: Optional[AdvancedRedrawer] = None
sam2_model: Optional[SAM] = None
text_mask_detector: Optional[TextMaskDetector] = None
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
NON_ALNUM_KO_RE = re.compile(r"[^0-9A-Za-z\uac00-\ud7a3]+")
SFX_REPEAT_RE = re.compile(r"^([\uac00-\ud7a3]{1,2})\1{1,}$")
SFX_STRETCH_RE = re.compile(r"([\uac00-\ud7a3])\1{2,}")
SFX_KEYWORDS = {
    "쿵", "쿵쿵", "탕", "탕탕", "탁", "탁탁", "쾅", "퍽", "팍", "철컥", "덜컥",
    "슥", "스윽", "촤악", "촥", "드륵", "끼익", "쪼르르", "주륵", "뚝뚝", "또각",
    "딸깍", "후두둑", "첨벙", "철퍼덕", "찰박", "사각", "부웅", "휘익", "후욱",
}

# Override mojibake-prone literals with safe unicode-escaped values.
SFX_KEYWORDS = {
    "\ucfe1", "\ud0d5", "\ud0c1", "\ucffc", "\ud37d", "\ud379", "\ucca0\ucee5", "\ub35c\ucee5",
    "\uc2a5", "\uc2a4\uc735", "\ucd24\uc545", "\ucd25", "\ub4dc\ub975", "\ub07c\uc775", "\ucaa9\ub974\ub974",
    "\uc8fc\ub975", "\ub69d\ub69d", "\ub610\uac01", "\ub51c\uae4d", "\ud6c4\ub450\ub451", "\ucca8\ubc99",
    "\ucca0\ud37c\ub355", "\ucc30\ubc15", "\uc0ac\uac01", "\ubd80\uc6c5", "\ud719\uc775", "\ud6c4\uc6b1",
}
SFX_PUNCT_RE = re.compile(r"[~!?.…·,;:\"'`^*_+=<>|\\/()\[\]{}\-]")

@app.get("/")
def read_root():
    return {"status": "Backend running. Caveman mode."}


def _load_sam2_model() -> Optional[SAM]:
    # Prefer SAM2.1 first, then SAM2 fallback.
    for model_name in ("sam2.1_b.pt", "sam2_b.pt"):
        try:
            model = SAM(model_name)
            print(f"STARTUP: SAM2 loaded: {model_name}")
            return model
        except Exception as exc:
            print(f"WARN: SAM2 load failed for {model_name}: {exc}")
    return None


def _resolve_text_mask_model_path() -> Optional[Path]:
    explicit = os.getenv("TEXT_MASK_MODEL_PATH", "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    current_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            current_dir / "models" / "text_mask_detector.onnx",
            current_dir / "text_mask_detector.onnx",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _normalize_inpaint_model(inpaint_model: Optional[str]) -> str:
    model = (inpaint_model or "default").strip()
    if not model:
        return "default"
    lowered = model.lower()
    if lowered in INPAINT_MODEL_ALIASES:
        return INPAINT_MODEL_ALIASES[lowered]
    if lowered == "default":
        return "default"
    return model


def _get_lama_inpainter(inpaint_model: str) -> Optional[Inpainter]:
    normalized = _normalize_inpaint_model(inpaint_model)
    if normalized in lama_cache:
        return lama_cache[normalized]
    if normalized == "default":
        return lama_cache.get("default")

    model_path = Path(normalized).expanduser()
    if not model_path.is_file():
        print(f"WARN: Inpaint model not found: {model_path}. Falling back to default LaMa.")
        lama_cache[normalized] = lama_cache.get("default")
        return lama_cache[normalized]

    if model_path.suffix.lower() == ".onnx":
        try:
            onnx_inpainter = OnnxLamaInpainter(model_path=str(model_path))
            print(f"STARTUP: ONNX inpaint model loaded on {onnx_inpainter.device} from {model_path}")
            lama_cache[normalized] = onnx_inpainter
        except Exception as exc:
            print(f"WARN: Failed to load ONNX inpaint model {model_path}: {exc}")
            lama_cache[normalized] = lama_cache.get("default")
        return lama_cache[normalized]

    # simple-lama-inpainting expects a TorchScript .pt model via LAMA_MODEL.
    previous_lama_model = os.environ.get("LAMA_MODEL")
    try:
        os.environ["LAMA_MODEL"] = str(model_path)
        custom_lama = SimpleLama(device=lama_device)
        print(f"STARTUP: Custom LaMa model loaded from {model_path}")
        lama_cache[normalized] = custom_lama
    except Exception as exc:
        print(f"WARN: Failed to load custom inpaint model {model_path}: {exc}")
        lama_cache[normalized] = lama_cache.get("default")
    finally:
        if previous_lama_model is None:
            os.environ.pop("LAMA_MODEL", None)
        else:
            os.environ["LAMA_MODEL"] = previous_lama_model

    return lama_cache[normalized]


def _resolve_inpaint_runtime(
    inpaint_model: str,
    high_quality_redraw: bool,
    redraw_engine: str,
    redraw_denoise: float,
    redraw_steps: int,
    redraw_guidance: float,
) -> tuple[str, bool, str, float, int, float]:
    normalized_model = _normalize_inpaint_model(inpaint_model)
    preset = DIFFUSION_INPAINT_PRESETS.get(normalized_model)
    engine = (redraw_engine or "sdxl").strip().lower()
    if engine not in {"sdxl", "flux", "sdxl_turbo", "sd15"}:
        engine = "sdxl"
    denoise = float(redraw_denoise)
    steps = int(redraw_steps)
    guidance = float(redraw_guidance)

    if preset is not None:
        engine = str(preset["engine"])
        denoise = float(preset["denoise"])
        steps = int(preset["steps"])
        guidance = float(preset["guidance"])
        high_quality_redraw = True
        # Diffusion presets still keep LaMa as base pass for safer fallback/output continuity.
        normalized_model = "default"
        print(
            "DEBUG: Inpaint preset selected -> "
            f"{inpaint_model} (engine={engine}, steps={steps}, guidance={guidance}, denoise={denoise})"
        )

    return normalized_model, high_quality_redraw, engine, denoise, steps, guidance


def _normalize_craft_detect_output(horizontal_list, free_list):
    # easyocr.detect often returns nested lists for batch compatibility.
    hlist = horizontal_list[0] if isinstance(horizontal_list, list) and horizontal_list and isinstance(horizontal_list[0], list) else horizontal_list
    flist = free_list[0] if isinstance(free_list, list) and free_list and isinstance(free_list[0], list) else free_list
    return hlist or [], flist or []


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def _is_mostly_korean(text: str) -> bool:
    core = NON_ALNUM_KO_RE.sub("", text or "")
    if not core:
        return False
    hangul_count = len(HANGUL_RE.findall(core))
    latin_count = len(re.findall(r"[A-Za-z]", core))
    total = max(1, hangul_count + latin_count)
    return hangul_count >= 1 and (hangul_count / total) >= 0.7


def _is_likely_korean_sfx(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    stripped = re.sub(r"[~!?.…·,;:\"'`^*_+=<>|\\/()\[\]{}\-]", "", t)
    if not stripped:
        return False
    if stripped in SFX_KEYWORDS:
        return True
    if SFX_REPEAT_RE.fullmatch(stripped):
        return True
    if SFX_STRETCH_RE.search(stripped):
        return True
    if re.fullmatch(r"[\u314b\u314e\u3160\u315c]{2,}", stripped):
        return True
    if len(stripped) <= 4 and any(ch in stripped for ch in "쿵탕탁쾅퍽팍슥촥철컥덜컥"):
        return True
    return False


def _is_likely_korean_sfx(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    stripped = SFX_PUNCT_RE.sub("", t)
    if not stripped:
        return False
    if stripped in SFX_KEYWORDS:
        return True
    if SFX_REPEAT_RE.fullmatch(stripped):
        return True
    if SFX_STRETCH_RE.search(stripped):
        return True
    if re.fullmatch(r"[\u314b\u314e\u3160\u315c]{2,}", stripped):
        return True
    if len(stripped) <= 4 and any(ch in stripped for ch in "\ucfe1\ud0d5\ud0c1\ucffc\ud37d\ud379\uc2a5\ucd25\ucca0\ucee5\ub35c\ucee5"):
        return True
    return False


def _is_text_inside_speech_bubble(tile_rgb: np.ndarray, box: list[int]) -> bool:
    x1, y1, x2, y2 = box
    h, w = tile_rgb.shape[:2]
    pad = 28
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    roi = tile_rgb[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, bright = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )

    cx1, cy1 = x1 - rx1, y1 - ry1
    cx2, cy2 = x2 - rx1, y2 - ry1
    text_region = bright[max(0, cy1):max(0, cy2), max(0, cx1):max(0, cx2)]
    if text_region.size == 0:
        return False
    if float(np.count_nonzero(text_region)) / float(text_region.size) < 0.55:
        return False

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    text_cx = (cx1 + cx2) // 2
    text_cy = (cy1 + cy2) // 2
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, (float(text_cx), float(text_cy)), False) < 0:
            continue
        area = cv2.contourArea(cnt)
        if area < 600:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity >= 0.12:
            return True
    return False


def _recognize_korean_text_in_box(tile_rgb: np.ndarray, box: list[int]) -> str:
    if ocr_reader is None:
        return ""
    x1, y1, x2, y2 = box
    roi = tile_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return ""
    try:
        preds = ocr_reader.readtext(
            roi,
            detail=1,
            paragraph=False,
            decoder="greedy",
            batch_size=1,
            width_ths=0.6,
        )
    except Exception:
        return ""
    if not preds:
        return ""
    best_text = ""
    best_prob = -1.0
    for _, txt, prob in preds:
        if prob > best_prob:
            best_prob = float(prob)
            best_text = txt
    best_text = _normalize_text(best_text)
    if not _is_mostly_korean(best_text):
        return ""
    return best_text


def _boxes_from_easyocr(tile_rgb: np.ndarray, w: int, h: int, pad: int = 8) -> list[list[int]]:
    if ocr_reader is None:
        return []
    try:
        preds = ocr_reader.readtext(
            tile_rgb,
            detail=1,
            paragraph=False,
            decoder="greedy",
            batch_size=1,
            width_ths=0.6,
        )
    except Exception as exc:
        print(f"WARN: EasyOCR readtext failed: {exc}")
        return []

    boxes: list[list[int]] = []
    for pred in preds:
        if not isinstance(pred, (list, tuple)) or len(pred) < 3:
            continue
        bbox, text, prob = pred
        if prob is not None and float(prob) < 0.28:
            continue
        arr = np.array(bbox, dtype=np.int32).reshape(-1, 2)
        x1 = max(0, int(arr[:, 0].min()) - pad)
        y1 = max(0, int(arr[:, 1].min()) - pad)
        x2 = min(w, int(arr[:, 0].max()) + pad)
        y2 = min(h, int(arr[:, 1].max()) + pad)
        if x2 - x1 <= 2 or y2 - y1 <= 2:
            continue
        norm_text = _normalize_text(str(text))
        if not _is_mostly_korean(norm_text):
            continue
        if _is_likely_korean_sfx(norm_text):
            continue
        boxes.append([x1, y1, x2, y2])

    return boxes


def _boxes_from_easyocr(tile_rgb: np.ndarray, w: int, h: int, pad: int = 8) -> list[list[int]]:
    if ocr_reader is None:
        return []
    try:
        preds = ocr_reader.readtext(
            tile_rgb,
            detail=1,
            paragraph=False,
            decoder="greedy",
            batch_size=1,
            width_ths=0.6,
        )
    except Exception as exc:
        print(f"WARN: EasyOCR readtext failed: {exc}")
        return []

    boxes: list[list[int]] = []
    for pred in preds:
        if not isinstance(pred, (list, tuple)) or len(pred) < 3:
            continue
        bbox, text, prob = pred
        if prob is not None and float(prob) < 0.28:
            continue
        arr = np.array(bbox, dtype=np.int32).reshape(-1, 2)
        x1 = max(0, int(arr[:, 0].min()) - pad)
        y1 = max(0, int(arr[:, 1].min()) - pad)
        x2 = min(w, int(arr[:, 0].max()) + pad)
        y2 = min(h, int(arr[:, 1].max()) + pad)
        if x2 - x1 <= 2 or y2 - y1 <= 2:
            continue

        norm_text = _normalize_text(str(text))
        if not _is_mostly_korean(norm_text):
            continue

        is_sfx = _is_likely_korean_sfx(norm_text)
        if is_sfx and not _is_text_inside_speech_bubble(tile_rgb, [x1, y1, x2, y2]):
            continue

        boxes.append([x1, y1, x2, y2])

    return boxes


def _boxes_from_rtdetr(tile_rgb: np.ndarray, device: str) -> list[list[int]]:
    if rt_model is None:
        return []
    inputs = processor(images=tile_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = rt_model(**inputs)
    target_sizes = torch.tensor([tile_rgb.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]
    boxes: list[list[int]] = []
    for label, box in zip(results["labels"], results["boxes"]):
        class_name = rt_model.config.id2label[label.item()]
        if "text" not in class_name.lower():
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        pad = 15
        boxes.append([
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(tile_rgb.shape[1], x2 + pad),
            min(tile_rgb.shape[0], y2 + pad),
        ])
    return boxes


def _filter_boxes_to_korean_non_sfx(tile_rgb: np.ndarray, boxes: list[list[int]]) -> list[list[int]]:
    filtered: list[list[int]] = []
    for box in boxes:
        text = _recognize_korean_text_in_box(tile_rgb, box)
        if not text:
            continue
        if _is_likely_korean_sfx(text):
            continue
        filtered.append(box)
    return filtered


def _refine_mask_with_text_effects(tile_bgr: np.ndarray, boxes: list[list[int]], base_mask: np.ndarray) -> np.ndarray:
    h, w = tile_bgr.shape[:2]
    refined = base_mask.copy()
    for x1, y1, x2, y2 in boxes:
        grow = 18
        rx1, ry1 = max(0, x1 - grow), max(0, y1 - grow)
        rx2, ry2 = min(w, x2 + grow), min(h, y2 + grow)
        roi = tile_bgr[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        g_thr = max(12, int(np.percentile(grad, 72)))
        edge_like = (grad > g_thr).astype(np.uint8) * 255

        # Colored/bright halo text effects.
        glow_like = ((sat > 45) & (val > 80)).astype(np.uint8) * 255
        effect = cv2.bitwise_or(edge_like, glow_like)

        # Track duplicated motion blur by directional expansion.
        effect_h = cv2.dilate(effect, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)), iterations=1)
        effect_v = cv2.dilate(effect, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11)), iterations=1)
        effect = cv2.bitwise_or(effect_h, effect_v)

        base_roi = base_mask[ry1:ry2, rx1:rx2]
        anchor = cv2.dilate(base_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
        effect = cv2.bitwise_and(effect, anchor)

        effect = cv2.morphologyEx(effect, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        refined[ry1:ry2, rx1:rx2] = cv2.bitwise_or(refined[ry1:ry2, rx1:rx2], effect)

    return refined


def _mask_from_boxes_sam2(tile_bgr: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    tile_mask = np.zeros(tile_bgr.shape[:2], dtype=np.uint8)
    if sam2_model is None or not boxes:
        return tile_mask
    try:
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        result = sam2_model(tile_rgb, bboxes=boxes, verbose=False)
        if not result or result[0].masks is None:
            return tile_mask
        data = result[0].masks.data
        if data is None:
            return tile_mask
        masks_np = data.detach().cpu().numpy()
        for m in masks_np:
            tile_mask[m > 0.5] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        tile_mask = _refine_mask_with_text_effects(tile_bgr, boxes, tile_mask)
    except Exception as exc:
        print(f"WARN: SAM2 mask generation failed: {exc}")
    return tile_mask


def _boxes_from_binary_mask(mask: np.ndarray, pad: int = 10) -> list[list[int]]:
    h, w = mask.shape[:2]
    total_area = float(h * w)
    min_area = max(36.0, total_area * 0.00008)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[int]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 6 or bh < 6:
            continue
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        boxes.append([x1, y1, x2, y2])
    return boxes


def _refine_onnx_mask_with_sam2(tile_bgr: np.ndarray, onnx_mask: np.ndarray) -> np.ndarray:
    if sam2_model is None:
        return onnx_mask
    if onnx_mask is None or not np.any(onnx_mask > 0):
        return onnx_mask

    boxes = _boxes_from_binary_mask(onnx_mask)
    if not boxes:
        return onnx_mask

    sam_mask = _mask_from_boxes_sam2(tile_bgr, boxes)
    if sam_mask is None or not np.any(sam_mask > 0):
        return onnx_mask

    # Constrain SAM2 expansion to neighborhood of ONNX detections to avoid face bleed.
    onnx_anchor = cv2.dilate(
        onnx_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        iterations=1,
    )
    constrained = cv2.bitwise_and(sam_mask, onnx_anchor)
    if not np.any(constrained > 0):
        return onnx_mask

    # Recover glow/motion-blur style effects close to the constrained text region.
    refined = _refine_mask_with_text_effects(tile_bgr, boxes, constrained)
    refined = cv2.bitwise_and(refined, onnx_anchor)
    refined = cv2.bitwise_or(refined, constrained)
    return refined


def _fallback_stroke_mask(tile_bgr: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    tile_mask = np.zeros(tile_bgr.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        box_roi = tile_bgr[y1:y2, x1:x2]
        if box_roi.size == 0:
            continue
        gray_roi = cv2.cvtColor(box_roi, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        grad = cv2.morphologyEx(gray_roi, cv2.MORPH_GRADIENT, kernel)
        otsu_thresh_val, _ = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        lower_thresh_val = max(10, otsu_thresh_val * 0.5)
        _, thresh = cv2.threshold(grad, lower_thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(thresh, contours, -1, 255, cv2.FILLED)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 7))
        roi_mask = cv2.dilate(thresh, dilate_kernel, iterations=2)
        tile_mask[y1:y2, x1:x2] = cv2.bitwise_or(tile_mask[y1:y2, x1:x2], roi_mask)
    return tile_mask


print("STARTUP: Loading SAM2 for text mask refinement...")
sam2_model = _load_sam2_model()
if sam2_model is None:
    print("WARN: SAM2 unavailable. Falling back to non-SAM masking.")

print("STARTUP: Loading ONNX TextMaskDetector...")
try:
    model_path = _resolve_text_mask_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "Missing ONNX model. Set TEXT_MASK_MODEL_PATH or place text_mask_detector.onnx in backend/models."
        )
    text_mask_detector = TextMaskDetector(model_path=str(model_path))
    print(f"STARTUP: TextMaskDetector loaded on {text_mask_detector.device} from {model_path}")
except Exception as exc:
    print(f"WARN: TextMaskDetector unavailable. Falling back to RT-DETR masking: {exc}")
    text_mask_detector = None

def _build_boundary_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    _, boundary = cv2.threshold(boundary, 10, 255, cv2.THRESH_BINARY)
    return boundary


def _apply_redraw_on_mask_roi(
    base_image: Image.Image,
    full_mask: Image.Image,
    redrawer: AdvancedRedrawer,
    prompt: str,
    denoise: float,
    lora_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
) -> Image.Image:
    mask_np = np.array(full_mask.convert("L"), dtype=np.uint8)
    ys, xs = np.where(mask_np > 0)
    if ys.size == 0 or xs.size == 0:
        return base_image

    # Crop redraw to mask bounds (+context) to avoid full-tile diffusion cost.
    pad = 48
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(mask_np.shape[1], int(xs.max()) + 1 + pad)
    y2 = min(mask_np.shape[0], int(ys.max()) + 1 + pad)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return base_image

    crop_img = base_image.crop((x1, y1, x2, y2))
    crop_mask = full_mask.crop((x1, y1, x2, y2))
    redraw_crop = redrawer.redraw(
        image=crop_img,
        mask=crop_mask,
        prompt=prompt,
        denoise=denoise,
        lora_scale=lora_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    out = base_image.copy()
    # Preserve untouched pixels by compositing redraw only on the masked region.
    precise_mask = crop_mask.convert("L").point(lambda p: 255 if p > 0 else 0)
    out.paste(redraw_crop, (x1, y1), precise_mask)
    return out


def _get_advanced_redrawer(
    high_quality_redraw: bool,
    redraw_engine: str,
    redraw_lora_path: Optional[str],
) -> Optional[AdvancedRedrawer]:
    global advanced_redrawer
    if not high_quality_redraw:
        return None
    if not torch.cuda.is_available():
        print("WARN: High quality redraw requested but CUDA is unavailable. Using LaMa only.")
        return None
    desired_lora = (redraw_lora_path or "").strip() or None
    if (
        advanced_redrawer is not None
        and advanced_redrawer.config.engine == redraw_engine.lower()
        and (advanced_redrawer.config.lora_path or None) == desired_lora
    ):
        return advanced_redrawer
    try:
        advanced_redrawer = AdvancedRedrawer(
            engine=redraw_engine,
            lora_path=desired_lora,
            device="cuda",
        )
        if advanced_redrawer.ensure_loaded():
            return advanced_redrawer
    except Exception as exc:
        print(f"WARN: Advanced redrawer init failed. Falling back to LaMa only: {exc}")
    return None


def process_images(
    extract_dir: Path,
    output_dir: Path,
    mask_engine: str = DEFAULT_MASK_ENGINE,
    inpaint_model: str = DEFAULT_INPAINT_MODEL,
    high_quality_redraw: bool = False,
    redraw_engine: str = "sdxl",
    redraw_prompt: str = DEFAULT_REDRAW_PROMPT,
    redraw_denoise: float = 0.35,
    redraw_steps: int = DEFAULT_REDRAW_STEPS,
    redraw_guidance: float = 6.5,
    redraw_lora_path: Optional[str] = None,
    redraw_lora_scale: float = 0.8,
    post_lama_boundary_cleanup: bool = True,
) -> dict[str, int | str]:
    """
    AI Pipeline: YOLO detect -> Mask -> LaMa Inpaint
    Handles long Webtoon strips by tiling.
    """
    _assert_cuda_ready()

    extensions = ["*.jpg", "*.jpeg", "*.png"]
    images = []
    for ext in extensions:
        images.extend(extract_dir.rglob(ext))

    if not images:
        return {"found": 0, "processed": 0, "skipped": 0, "reason": "no_images_found"}
        
    print(f"DEBUG: Found {len(images)} images")
    mask_engine = (mask_engine or DEFAULT_MASK_ENGINE).strip().lower()
    if mask_engine == "easyocr_sam2":
        mask_engine = "easyocr"
    if mask_engine not in {"onnx_textmask", "rtdetr", "easyocr"}:
        mask_engine = DEFAULT_MASK_ENGINE
    if mask_engine == "easyocr" and ocr_reader is None:
        print("WARN: EasyOCR requested but unavailable; falling back to onnx_textmask.")
        mask_engine = "onnx_textmask"
    print(f"DEBUG: Mask engine = {mask_engine}")
    normalized_inpaint_model, high_quality_redraw, redraw_engine, redraw_denoise, redraw_steps, redraw_guidance = _resolve_inpaint_runtime(
        inpaint_model=inpaint_model,
        high_quality_redraw=high_quality_redraw,
        redraw_engine=redraw_engine,
        redraw_denoise=redraw_denoise,
        redraw_steps=redraw_steps,
        redraw_guidance=redraw_guidance,
    )
    print(f"DEBUG: Inpaint model = {inpaint_model} (resolved={normalized_inpaint_model})")
    lama_inpainter = _get_lama_inpainter(normalized_inpaint_model)
    if lama_inpainter is None:
        return {
            "found": len(images),
            "processed": 0,
            "skipped": len(images),
            "reason": "lama_unavailable",
        }
    redrawer = _get_advanced_redrawer(
        high_quality_redraw=high_quality_redraw,
        redraw_engine=redraw_engine,
        redraw_lora_path=redraw_lora_path,
    )
    processed_count = 0
    skipped_count = 0
        
    for img_path in images:
        rel_path = img_path.relative_to(extract_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None or lama_inpainter is None:
            print(f"DEBUG: Skipping {img_path.name} (load fail or models missing)")
            skipped_count += 1
            continue

        h, w = img_bgr.shape[:2]
        print(f"DEBUG: Processing {img_path.name} ({w}x{h})")

        # Webtoon Tiling Logic
        tile_size = w
        overlap = 100
        step = tile_size - overlap
        
        # Result canvas
        final_img = img_bgr.copy()
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for y in range(0, h, step):
            y_end = min(y + tile_size, h)
            y_start = max(0, y_end - tile_size)
            
            tile = img_bgr[y_start:y_end, 0:w]
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            
            if mask_engine == "rtdetr":
                boxes = _boxes_from_rtdetr(tile_rgb, device=device)
                if not boxes:
                    continue
                print(f"DEBUG: Tile at y={y_start} -> {len(boxes)} text regions")
                tile_mask = _fallback_stroke_mask(tile, boxes)
            elif mask_engine == "easyocr":
                boxes = _boxes_from_easyocr(tile_rgb, w=tile_rgb.shape[1], h=tile_rgb.shape[0], pad=10)
                if not boxes:
                    continue
                if sam2_model is not None:
                    tile_mask = _mask_from_boxes_sam2(tile, boxes)
                    if not np.any(tile_mask > 0):
                        tile_mask = _fallback_stroke_mask(tile, boxes)
                else:
                    tile_mask = _fallback_stroke_mask(tile, boxes)
            else:
                if text_mask_detector is not None:
                    onnx_mask = text_mask_detector.detect(tile_rgb)
                    tile_mask = _refine_onnx_mask_with_sam2(tile, onnx_mask)
                    if sam2_model is not None:
                        print("DEBUG: ONNX+SAM2 mask refinement active")
                else:
                    boxes = _boxes_from_rtdetr(tile_rgb, device=device)
                    if not boxes:
                        continue
                    print(f"DEBUG: TextMaskDetector unavailable; RT-DETR fallback regions={len(boxes)}")
                    tile_mask = _fallback_stroke_mask(tile, boxes)
                
            # Inpaint tile (MUST BE OUTSIDE THE BOX LOOP)
            tile_pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(tile_mask).convert('L')
            final_mask[y_start:y_end, 0:w] = cv2.bitwise_or(final_mask[y_start:y_end, 0:w], tile_mask)
            
            try:
                clean_tile_pil = lama_inpainter(tile_pil, mask_pil)
            except Exception as e:
                print(f"DEBUG: Tile LaMa inpaint fail: {e}")
                continue

            if redrawer is not None and np.any(tile_mask > 0):
                try:
                    clean_tile_pil = _apply_redraw_on_mask_roi(
                        base_image=clean_tile_pil,
                        full_mask=mask_pil,
                        redrawer=redrawer,
                        prompt=redraw_prompt,
                        denoise=redraw_denoise,
                        lora_scale=redraw_lora_scale,
                        guidance_scale=redraw_guidance,
                        num_inference_steps=redraw_steps,
                    )

                    if post_lama_boundary_cleanup:
                        boundary_mask = _build_boundary_mask(tile_mask)
                        if np.any(boundary_mask > 0):
                            boundary_mask_pil = Image.fromarray(boundary_mask).convert("L")
                            clean_tile_pil = lama_inpainter(clean_tile_pil, boundary_mask_pil)
                except Exception as e:
                    print(f"DEBUG: Tile redraw fail; keeping LaMa result: {e}")

            try:
                clean_tile_bgr = cv2.cvtColor(np.array(clean_tile_pil), cv2.COLOR_RGB2BGR)
                # Paste back to final
                final_img[y_start:y_end, 0:w] = clean_tile_bgr
            except Exception as e:
                print(f"DEBUG: Tile compose fail: {e}")

        # Save as layered PSD using pytoshop
        # Load original image and convert both to RGBA
        orig_bgr = cv2.imread(str(img_path))
        orig_rgba = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGBA)
        clean_rgba = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGBA)
        
        # Pytoshop channels dict: 0=R, 1=G, 2=B, -1=A
        layer_bg = nested_layers.Image(
            name='Background', visible=True, top=0, left=0, bottom=orig_rgba.shape[0], right=orig_rgba.shape[1],
            channels={0: orig_rgba[:,:,0], 1: orig_rgba[:,:,1], 2: orig_rgba[:,:,2], -1: orig_rgba[:,:,3]}
        )
        
        layer_clean = nested_layers.Image(
            name='Cleaned', visible=True, top=0, left=0, bottom=clean_rgba.shape[0], right=clean_rgba.shape[1],
            channels={0: clean_rgba[:,:,0], 1: clean_rgba[:,:,1], 2: clean_rgba[:,:,2], -1: clean_rgba[:,:,3]}
        )
        
        # Create PSD with RAW compression for maximum compatibility
        # Layer order is bottom-to-top: [Background, Cleaned]
        psd = nested_layers.nested_layers_to_psd([layer_clean , layer_bg], color_mode=ColorMode.rgb, compression=Compression.raw)
        
        psd_path = output_dir / f"{img_path.stem}.psd"
        with open(psd_path, 'wb') as f:
            psd.write(f)

        print(f"DEBUG: {img_path.name} finished. Created PSD.")
        processed_count += 1

    return {
        "found": len(images),
        "processed": processed_count,
        "skipped": skipped_count,
        "reason": "ok" if processed_count > 0 else "all_skipped",
    }


async def _commit_modal_volume_if_configured() -> None:
    volume_name = os.getenv("MODAL_VOLUME_NAME", "").strip()
    if not volume_name:
        return
    try:
        import modal

        await modal.Volume.from_name(volume_name).commit.aio()
        print(f"DEBUG: Committed Modal volume '{volume_name}'")
    except Exception as exc:
        print(f"WARN: Failed to commit Modal volume '{volume_name}': {exc}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    safe_name = Path(filename).name
    file_path = PROCESSED_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), media_type="application/zip", filename=safe_name)

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    mask_engine: str = Form(DEFAULT_MASK_ENGINE),
    inpaint_model: str = Form(DEFAULT_INPAINT_MODEL),
    high_quality_redraw: bool = Form(False),
    redraw_engine: str = Form("sdxl"),
    redraw_prompt: str = Form(DEFAULT_REDRAW_PROMPT),
    redraw_denoise: float = Form(0.35),
    redraw_steps: int = Form(DEFAULT_REDRAW_STEPS),
    redraw_guidance: float = Form(6.5),
    redraw_lora_path: Optional[str] = Form(None),
    redraw_lora_scale: float = Form(0.8),
    post_lama_boundary_cleanup: bool = Form(True),
):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    if not file.filename.endswith('.zip') and not file.filename.endswith('.rar'):
        return {"error": "Only zip/rar supported."}
        
    # Unzip
    job_id = file.filename.rsplit('.', 1)[0]
    import re
    # Sanitize job_id just in case
    job_id = re.sub(r'[<>:"/\\|?*]', '_', job_id).strip(' .')
    
    extract_dir = UPLOAD_DIR / job_id
    extract_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            if member.is_dir():
                continue
            
            # Very aggressive sanitization for Windows
            # Replace backslashes with forward slashes
            filename = member.filename.replace('\\', '/')
            
            # Keep only safe characters (alphanumerics, spaces, and basic punctuation)
            import string
            valid_chars = f"-_.() /{string.ascii_letters}{string.digits}"
            clean_name = ''.join(c for c in filename if c in valid_chars)
            
            # Remove trailing spaces/dots from each part of the path
            clean_name = '/'.join(p.strip(' .\n\r\t') for p in clean_name.split('/'))
            
            if not clean_name:
                continue
                
            target_path = extract_dir / clean_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)
        
    # Process
    output_dir = PROCESSED_DIR / job_id
    output_dir.mkdir(exist_ok=True)
    process_summary = process_images(
        extract_dir,
        output_dir,
        mask_engine=mask_engine,
        inpaint_model=inpaint_model,
        high_quality_redraw=high_quality_redraw,
        redraw_engine=redraw_engine,
        redraw_prompt=redraw_prompt,
        redraw_denoise=redraw_denoise,
        redraw_steps=redraw_steps,
        redraw_guidance=redraw_guidance,
        redraw_lora_path=redraw_lora_path,
        redraw_lora_scale=redraw_lora_scale,
        post_lama_boundary_cleanup=post_lama_boundary_cleanup,
    )
    if int(process_summary.get("processed", 0)) <= 0:
        reason = str(process_summary.get("reason", "unknown"))
        raise HTTPException(
            status_code=500,
            detail=(
                "No images were processed. "
                f"found={process_summary.get('found', 0)} skipped={process_summary.get('skipped', 0)} reason={reason}. "
                "Check LaMa/ONNX model startup logs and GPU/CPU backend configuration."
            ),
        )
    
    # Zip
    zip_base_name = PROCESSED_DIR / f"{job_id}_clean"
    shutil.make_archive(str(zip_base_name), 'zip', output_dir)
    result_zip_path = Path(f"{zip_base_name}.zip")
    
    print(f"DEBUG: Created zip at {result_zip_path}")
    await _commit_modal_volume_if_configured()
    _, resolved_hq, resolved_engine, _, _, _ = _resolve_inpaint_runtime(
        inpaint_model=inpaint_model,
        high_quality_redraw=high_quality_redraw,
        redraw_engine=redraw_engine,
        redraw_denoise=redraw_denoise,
        redraw_steps=redraw_steps,
        redraw_guidance=redraw_guidance,
    )
    
    return {
        "filename": result_zip_path.name,
        "message": "Cleaned and zipped locally.",
        "mask_engine": mask_engine,
        "inpaint_model": _normalize_inpaint_model(inpaint_model),
        "high_quality_redraw": resolved_hq,
        "redraw_engine": resolved_engine,
        "download_url": f"/download/{result_zip_path.name}",
        "storage_root": str(BASE_DATA_DIR),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
