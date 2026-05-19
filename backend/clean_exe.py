import os
import sys
import warnings

# Suppress EasyOCR/Torch FutureWarnings about weights_only
warnings.filterwarnings("ignore", category=FutureWarning, module="easyocr")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from simple_lama_inpainting import SimpleLama
import pytoshop
from pytoshop.user import nested_layers
from pytoshop.enums import ColorMode, Compression, Version
import easyocr

# --- CONFIGURATION FOR EXECUTABLE ---
def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# --- MODEL LOADING ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"STARTUP: Using device: {DEVICE}")

# 1. Load RT-DETR
print("STARTUP: Loading RT-DETR (ogkalu/comic-text-and-bubble-detector)...")
try:
    processor = AutoImageProcessor.from_pretrained("ogkalu/comic-text-and-bubble-detector")
    rt_model = AutoModelForObjectDetection.from_pretrained("ogkalu/comic-text-and-bubble-detector").to(DEVICE)
except Exception as e:
    print("!!! ERROR: RT-DETR load fail !!!", e)
    rt_model = None

# 2. Load LaMa
print("STARTUP: Loading LaMa...")
try:
    lama = SimpleLama(device=torch.device(DEVICE))
except Exception as e:
    print("!!! ERROR: LaMa load fail !!!", e)
    lama = None

# 3. Load EasyOCR (Korean)
print("STARTUP: Loading Korean OCR...")
try:
    user_net_dir = get_resource_path('user_network')
    ocr_reader = easyocr.Reader(['ko'], 
                                 recog_network='ko_webtoon_v2', 
                                 user_network_directory=user_net_dir,
                                 model_storage_directory=user_net_dir)
    print("STARTUP: OCR Loaded")
except Exception as e:
    print("!!! OCR LOAD FAIL !!!", e)
    ocr_reader = None

def process_image(img_path: Path):
    """ Processes a single image and saves PSD/PSB in the same folder """
    if rt_model is None or lama is None:
        print(f"Skipping {img_path.name}: Models not loaded.")
        return

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Skipping {img_path.name}: Load fail.")
        return

    h, w = img_bgr.shape[:2]
    print(f"Processing: {img_path.name} ({w}x{h})")

    # Webtoon Tiling Logic
    tile_size = w
    overlap = 100
    step = tile_size - overlap
    final_img = img_bgr.copy()
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(0, h, step):
        y_end = min(y + tile_size, h)
        y_start = max(0, y_end - tile_size)
        
        tile = img_bgr[y_start:y_end, 0:w]
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        
        # RT-DETR Predict
        inputs = processor(images=tile_rgb, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = rt_model(**inputs)
            
        target_sizes = torch.tensor([tile_rgb.shape[:2]]).to(DEVICE)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]
        
        if len(results["boxes"]) == 0:
            continue
            
        # Create Mask for tile
        tile_mask = np.zeros(tile.shape[:2], dtype=np.uint8)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_name = rt_model.config.id2label[label.item()]
            if "text" not in class_name.lower():
                continue # Skip bubbles
                
            x1, y1, x2, y2 = map(int, box.tolist())
            pad = 15
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(tile.shape[1], x2 + pad); y2 = min(tile.shape[0], y2 + pad)
            
            box_roi = tile[y1:y2, x1:x2]
            if box_roi.size == 0: continue

            # OCR detection
            if ocr_reader:
                ocr_res = ocr_reader.readtext(box_roi)
                for (bbox, text, prob) in ocr_res:
                    print(f"  [OCR] {text} ({prob:.2f})")

            # Masking logic
            gray_roi = cv2.cvtColor(box_roi, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            grad = cv2.morphologyEx(gray_roi, cv2.MORPH_GRADIENT, kernel)
            otsu_val, _ = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            _, thresh = cv2.threshold(grad, max(10, otsu_val * 0.5), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(thresh, contours, -1, 255, cv2.FILLED)
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 7))
            roi_mask = cv2.dilate(thresh, dilate_kernel, iterations=3)
            
            tile_mask[y1:y2, x1:x2] = cv2.bitwise_or(tile_mask[y1:y2, x1:x2], roi_mask)
            
        # Inpaint tile
        tile_pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(tile_mask).convert('L')
        final_mask[y_start:y_end, 0:w] = cv2.bitwise_or(final_mask[y_start:y_end, 0:w], tile_mask)
        
        try:
            clean_tile_pil = lama(tile_pil, mask_pil)
            clean_tile_bgr = cv2.cvtColor(np.array(clean_tile_pil), cv2.COLOR_RGB2BGR)
            final_img[y_start:y_end, 0:w] = clean_tile_bgr
        except Exception as e:
            print(f"  !!! Inpaint fail: {e}")

    # Save Layered PSD
    orig_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
    clean_rgba = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGBA)
    
    layer_bg = nested_layers.Image(
        name='Original', visible=True, top=0, left=0, bottom=h, right=w,
        channels={0: orig_rgba[:,:,0], 1: orig_rgba[:,:,1], 2: orig_rgba[:,:,2], -1: orig_rgba[:,:,3]}
    )
    layer_clean = nested_layers.Image(
        name='Cleaned', visible=True, top=0, left=0, bottom=h, right=w,
        channels={0: clean_rgba[:,:,0], 1: clean_rgba[:,:,1], 2: clean_rgba[:,:,2], -1: clean_rgba[:,:,3]}
    )
    
    psd_version = Version.psb if max(h, w) > 30000 else Version.version_1
    psd = nested_layers.nested_layers_to_psd([layer_clean, layer_bg], color_mode=ColorMode.rgb, compression=Compression.rle, version=psd_version)
    
    ext = "psb" if psd_version == Version.psb else "psd"
    output_path = img_path.with_suffix(f".{ext}")
    with open(output_path, 'wb') as f:
        psd.write(f)

        
    print(f"DONE: {output_path.name} saved next to original.")

def main():
    if len(sys.argv) < 2:
        print("Usage: clean_exe.exe <folder_path_or_file_path>")
        input_path = input("Drag and drop folder/file here and press Enter: ").strip('"\' ')
    else:
        input_path = sys.argv[1]

    path = Path(input_path)
    if not path.exists():
        print(f"Error: Path {path} does not exist.")
        return

    if path.is_file():
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            process_image(path)
        else:
            print("Unsupported file format.")
    elif path.is_dir():
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        images = []
        for ext in extensions:
            images.extend(path.rglob(ext))
        
        print(f"Found {len(images)} images in {path}")
        for img_path in images:
            # Avoid re-processing if output exists? (optional)
            process_image(img_path)

    print("\n--- ALL FINISHED ---")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
