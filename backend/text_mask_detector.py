import os
from typing import Iterable, Optional

import cv2
import numpy as np
import onnxruntime as ort


class TextMaskDetector:
    """
    Reusable ONNX text detector + mask expander.
    Input: RGB image as HxWx3 uint8.
    Output: binary uint8 mask (0 or 255), same HxW.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.3,
        dilation_ratio: float = 0.04,
        min_kernel_size: int = 11,
        dilation_iterations: int = 2,
        pad_multiple: int = 32,
        providers: Optional[Iterable] = None,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.threshold = float(threshold)
        self.dilation_ratio = float(dilation_ratio)
        self.min_kernel_size = int(min_kernel_size)
        self.dilation_iterations = int(dilation_iterations)
        self.pad_multiple = int(pad_multiple)

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
        self.input_name = self.session.get_inputs()[0].name

    @property
    def active_providers(self):
        return self.session.get_providers()

    @property
    def device(self) -> str:
        return "GPU" if "CUDAExecutionProvider" in self.active_providers else "CPU"

    def detect(self, rgb_image: np.ndarray) -> np.ndarray:
        if rgb_image is None:
            raise ValueError("rgb_image is None")
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("rgb_image must be HxWx3 RGB")

        h, w = rgb_image.shape[:2]
        ph = ((h + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        pw = ((w + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
        pad_h, pad_w = ph - h, pw - w

        padded = cv2.copyMakeBorder(
            rgb_image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        inp = padded.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis, :]

        heatmap = self.session.run(None, {self.input_name: inp})[0][0][0]
        heatmap = heatmap[:h, :w]

        mask = (heatmap > self.threshold).astype(np.uint8) * 255

        k_size = max(self.min_kernel_size, int(w * self.dilation_ratio))
        if k_size % 2 == 0:
            k_size += 1

        if self.dilation_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            mask = cv2.dilate(mask, kernel, iterations=self.dilation_iterations)

        return mask

