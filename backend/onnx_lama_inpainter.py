from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image


class OnnxLamaInpainter:
    """
    ONNX runtime wrapper for LaMa-style inpainting models.
    Input:
      - RGB image (PIL or ndarray HxWx3)
      - L mask (PIL or ndarray HxW), non-zero pixels are inpainted
    Output:
      - PIL RGB image
    """

    def __init__(
        self,
        model_path: str,
        pad_multiple: int = 8,
        providers: Optional[Iterable[str]] = None,
    ) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX inpaint model not found: {model_path}")

        self.model_path = model_path
        self.pad_multiple = max(1, int(pad_multiple))

        sess_opt = ort.SessionOptions()
        sess_opt.enable_mem_pattern = False
        sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if providers is None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opt,
            providers=list(providers),
        )

        inputs = self.session.get_inputs()
        if len(inputs) < 2:
            raise ValueError("ONNX inpaint model must have at least 2 inputs: image + mask")

        self.image_input_name = ""
        self.mask_input_name = ""
        for node in inputs:
            lowered = node.name.lower()
            if "mask" in lowered and not self.mask_input_name:
                self.mask_input_name = node.name
            elif not self.image_input_name:
                self.image_input_name = node.name

        if not self.image_input_name:
            self.image_input_name = inputs[0].name
        if not self.mask_input_name:
            for node in inputs:
                if node.name != self.image_input_name:
                    self.mask_input_name = node.name
                    break
        if not self.mask_input_name:
            self.mask_input_name = inputs[1].name

        self.output_name = self.session.get_outputs()[0].name

    @property
    def active_providers(self) -> list[str]:
        return self.session.get_providers()

    @property
    def device(self) -> str:
        return "GPU" if "CUDAExecutionProvider" in self.active_providers else "CPU"

    def __call__(self, image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray) -> Image.Image:
        img_rgb = self._to_rgb_uint8(image)
        mask_gray = self._to_gray_uint8(mask)
        if not np.any(mask_gray > 0):
            return Image.fromarray(img_rgb, mode="RGB")

        orig_h, orig_w = img_rgb.shape[:2]
        img_chw = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        mask_chw = (mask_gray[np.newaxis, :, :] > 0).astype(np.float32)

        img_chw = self._pad_to_multiple(img_chw, self.pad_multiple)
        mask_chw = self._pad_to_multiple(mask_chw, self.pad_multiple)

        image_batch = np.expand_dims(img_chw, axis=0)
        mask_batch = np.expand_dims(mask_chw, axis=0)

        out = self.session.run(
            [self.output_name],
            {
                self.image_input_name: image_batch.astype(np.float32),
                self.mask_input_name: mask_batch.astype(np.float32),
            },
        )[0]

        if out.ndim != 4 or out.shape[0] < 1 or out.shape[1] < 3:
            raise ValueError(f"Unexpected ONNX output shape: {out.shape}")

        out_chw = out[0, :3, :, :].astype(np.float32)
        # Some LaMa exports return [-1,1], others [0,1].
        if float(out_chw.min()) < -0.01:
            out_chw = (out_chw + 1.0) * 0.5

        out_hwc = np.transpose(out_chw, (1, 2, 0))
        out_hwc = np.clip(out_hwc, 0.0, 1.0)
        out_u8 = (out_hwc * 255.0).round().astype(np.uint8)
        out_u8 = out_u8[:orig_h, :orig_w, :]
        return Image.fromarray(out_u8, mode="RGB")

    @staticmethod
    def _to_rgb_uint8(image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"), dtype=np.uint8)
            return arr
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("image must be RGB HxWx3")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _to_gray_uint8(mask: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(mask, Image.Image):
            arr = np.array(mask.convert("L"), dtype=np.uint8)
            return arr
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError("mask must be grayscale HxW")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _ceil_modulo(x: int, mod: int) -> int:
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod

    def _pad_to_multiple(self, chw: np.ndarray, mod: int) -> np.ndarray:
        c, h, w = chw.shape
        oh = self._ceil_modulo(h, mod)
        ow = self._ceil_modulo(w, mod)
        if oh == h and ow == w:
            return chw
        return np.pad(chw, ((0, 0), (0, oh - h), (0, ow - w)), mode="symmetric")
