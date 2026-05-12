from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gc
import numpy as np
import torch
from PIL import Image, ImageFilter
import time
import os


@dataclass
class RedrawConfig:
    engine: str = "sdxl"
    sd15_model_id: str = "runwayml/stable-diffusion-inpainting"
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    controlnet_id: str = "xinsir/controlnet-union-sdxl-1.0"
    turbo_model_id: str = "stabilityai/sdxl-turbo"
    flux_model_id: str = "black-forest-labs/FLUX.1-Fill-dev"
    lora_path: Optional[str] = None
    prompt: str = (
        "clean Korean webtoon panel redraw, remove dialogue text only, preserve composition, "
        "speech bubbles/tails, line weight, screentones, hatching, texture continuity, no new text, no SFX"
    )
    negative_prompt: str = "new text, SFX, watermark, logo, blurry, distorted anatomy, oversharpened, color shift"
    denoise: float = 0.35
    steps: int = 28
    guidance_scale: float = 6.5
    lora_scale: float = 0.8
    controlnet_conditioning_scale: float = 0.9


class AdvancedRedrawer:
    """
    Two engine modes:
    - SDXL ControlNet inpaint (default)
    - FLUX Fill (optional, very heavy)
    """

    def __init__(
        self,
        engine: str = "sdxl",
        sd15_model_id: str = "runwayml/stable-diffusion-inpainting",
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet_id: str = "xinsir/controlnet-union-sdxl-1.0",
        turbo_model_id: str = "stabilityai/sdxl-turbo",
        flux_model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
        lora_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = RedrawConfig(
            engine=engine.lower(),
            sd15_model_id=sd15_model_id,
            model_id=model_id,
            controlnet_id=controlnet_id,
            turbo_model_id=turbo_model_id,
            flux_model_id=flux_model_id,
            lora_path=lora_path,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self._loaded = False
        self._load_attempted = False
        self._disabled_due_to_oom = False

    def ensure_loaded(self) -> bool:
        if self._disabled_due_to_oom:
            return False
        if self._loaded:
            return True
        if self._load_attempted and not self._loaded:
            return False
        self._load_attempted = True
        try:
            if self.config.engine == "flux" and self.device == "cuda":
                min_vram_gb = float(os.getenv("FLUX_MIN_VRAM_GB", "30"))
                total_vram_gb = self._cuda_total_vram_gb()
                if 0 < total_vram_gb < min_vram_gb:
                    print(
                        f"WARN: FLUX requested but GPU VRAM is {total_vram_gb:.2f} GiB "
                        f"(< FLUX_MIN_VRAM_GB={min_vram_gb}). Falling back to SDXL."
                    )
                    self.config.engine = "sdxl"
            if self.config.engine == "flux":
                self._load_flux()
            elif self.config.engine == "sd15":
                self._load_sd15()
            elif self.config.engine == "sdxl_turbo":
                self._load_sdxl_turbo()
            else:
                self._load_sdxl()
            self._loaded = self.pipe is not None
            return self._loaded
        except Exception as exc:
            print(f"ERROR: AdvancedRedrawer load failed ({self.config.engine}): {exc}")
            if self._is_cuda_oom(exc):
                self._disabled_due_to_oom = True
            self._cleanup_cuda_state()
            self.pipe = None
            self._loaded = False
            return False

    def redraw(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: Optional[str] = None,
        denoise: float = 0.35,
        lora_scale: float = 0.8,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Image.Image:
        if not self.ensure_loaded() or self.pipe is None:
            return image

        mask_l = mask.convert("L")
        mask_np = np.array(mask_l, dtype=np.uint8)
        if not np.any(mask_np > 0):
            return image

        width, height = self._fit_size_for_pipeline(image.size)
        base_image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        base_mask = mask_l.resize((width, height), Image.Resampling.NEAREST)

        prompt = prompt or self.config.prompt
        denoise = float(np.clip(denoise, 0.2, 0.55))
        steps = int(num_inference_steps or self.config.steps)
        guidance = float(guidance_scale if guidance_scale is not None else self.config.guidance_scale)

        try:
            if self.config.engine == "flux":
                out = self.pipe(
                    prompt=prompt,
                    image=base_image,
                    mask_image=base_mask,
                    width=width,
                    height=height,
                    strength=denoise,
                    guidance_scale=max(1.0, guidance),
                    num_inference_steps=max(20, steps),
                    max_sequence_length=512,
                ).images[0]
            elif self.config.engine == "sd15":
                out = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.config.negative_prompt,
                    image=base_image,
                    mask_image=base_mask,
                    strength=denoise,
                    num_inference_steps=max(8, steps),
                    guidance_scale=max(1.0, guidance),
                    cross_attention_kwargs={"scale": float(np.clip(lora_scale, 0.0, 1.5))},
                ).images[0]
            elif self.config.engine == "sdxl_turbo":
                turbo_steps = int(np.clip(steps, 2, 4))
                turbo_strength = float(np.clip(denoise, 0.35, 0.8))
                turbo_img = self.pipe(
                    prompt=prompt,
                    image=base_image,
                    strength=turbo_strength,
                    guidance_scale=0.0,
                    num_inference_steps=turbo_steps,
                ).images[0]
                out = self._composite_masked(base_image, turbo_img, base_mask)
            else:
                control_image = self._prepare_control_image(base_image, base_mask)
                out = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.config.negative_prompt,
                    image=base_image,
                    mask_image=base_mask,
                    control_image=control_image,
                    controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
                    strength=denoise,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    cross_attention_kwargs={"scale": float(np.clip(lora_scale, 0.0, 1.5))},
                ).images[0]
        except Exception as exc:
            if self._is_cuda_oom(exc):
                print(
                    f"WARN: Redraw CUDA OOM ({self.config.engine}). "
                    "Disabling redrawer for this request and keeping LaMa output."
                )
                self._disabled_due_to_oom = True
                self._loaded = False
                self.pipe = None
                self._cleanup_cuda_state()
                return image
            raise

        if out.size != image.size:
            out = out.resize(image.size, Image.Resampling.LANCZOS)
        return out

    def _load_sdxl(self) -> None:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"LOADING: SDXL redrawer on {self.device}...")

        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        load_kwargs = {
            "controlnet": controlnet,
            "torch_dtype": dtype,
            "use_safetensors": True,
        }
        if dtype == torch.float16:
            load_kwargs["variant"] = "fp16"

        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            self.config.model_id,
            **load_kwargs,
        )
        self._disable_safety_checker(pipe)
        try:
            from diffusers import EDMSolverMultistepScheduler

            pipe.scheduler = EDMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if self.device == "cuda":
            pipe = pipe.to(self.device)
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        else:
            pipe.enable_model_cpu_offload()

        if self.config.lora_path:
            print(f"LOADING: Style LoRA {self.config.lora_path}")
            pipe.load_lora_weights(self.config.lora_path)

        self.pipe = pipe

    def _load_sd15(self) -> None:
        from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"LOADING: SD1.5 inpaint redrawer on {self.device}...")
        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
        }
        if dtype == torch.float16:
            # HF repo commonly stores fp16 under this variant name for SD1.5 inpaint.
            load_kwargs["variant"] = "fp16"

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.config.sd15_model_id,
            **load_kwargs,
        )
        self._disable_safety_checker(pipe)
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            pass

        if self.device == "cuda":
            pipe = pipe.to(self.device)
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        else:
            pipe.enable_model_cpu_offload()

        if self.config.lora_path:
            print(f"LOADING: Style LoRA {self.config.lora_path}")
            pipe.load_lora_weights(self.config.lora_path)

        self.pipe = pipe

    def _load_sdxl_turbo(self) -> None:
        from diffusers import AutoPipelineForImage2Image, EulerAncestralDiscreteScheduler

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"LOADING: SDXL-Turbo redrawer on {self.device}...")
        load_kwargs = {"torch_dtype": dtype}
        if dtype == torch.float16:
            load_kwargs["variant"] = "fp16"
        pipe = AutoPipelineForImage2Image.from_pretrained(
            self.config.turbo_model_id,
            **load_kwargs,
        )
        self._disable_safety_checker(pipe)
        try:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        except Exception:
            pass
        if self.device == "cuda":
            pipe = pipe.to(self.device)
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        else:
            pipe.enable_model_cpu_offload()
        self.pipe = pipe

    def _load_flux(self) -> None:
        from diffusers import FluxFillPipeline

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        print(f"LOADING: FLUX redrawer on {self.device}...")
        pipe = FluxFillPipeline.from_pretrained(self.config.flux_model_id, torch_dtype=dtype)
        if self.device == "cuda":
            low_vram_mode = os.getenv("FLUX_LOW_VRAM_MODE", "1").strip().lower() not in {"0", "false", "no"}
            if low_vram_mode:
                pipe.enable_sequential_cpu_offload()
                try:
                    pipe.vae.enable_slicing()
                    pipe.vae.enable_tiling()
                except Exception:
                    pass
            else:
                pipe = pipe.to(self.device)
        else:
            pipe.enable_model_cpu_offload()
        self.pipe = pipe

    @staticmethod
    def _disable_safety_checker(pipe) -> None:
        # Webtoon panels frequently false-trigger SD safety filters, yielding black images.
        if hasattr(pipe, "safety_checker"):
            try:
                pipe.safety_checker = None
            except Exception:
                pass
        if hasattr(pipe, "requires_safety_checker"):
            try:
                pipe.requires_safety_checker = False
            except Exception:
                pass

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        message = str(exc).lower()
        return "out of memory" in message and "cuda" in message

    @staticmethod
    def _cuda_total_vram_gb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.get_device_properties(0).total_memory) / (1024.0 ** 3)

    @staticmethod
    def _cleanup_cuda_state() -> None:
        gc.collect()
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def _fit_size_for_pipeline(size: tuple[int, int]) -> tuple[int, int]:
        w, h = size
        # Inpaint pipelines are happiest with dimensions divisible by 16.
        w = max(64, int(round(w / 16.0) * 16))
        h = max(64, int(round(h / 16.0) * 16))
        return w, h

    @staticmethod
    def _prepare_control_image(image: Image.Image, mask: Image.Image) -> Image.Image:
        image_np = np.array(image.convert("RGB"), dtype=np.uint8)
        mask_np = (np.array(mask.convert("L"), dtype=np.uint8) > 127).astype(np.uint8)
        masked = image_np * (1 - mask_np[..., np.newaxis])
        return Image.fromarray(masked.astype(np.uint8), mode="RGB")

    @staticmethod
    def _composite_masked(base_image: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
        base = base_image.convert("RGB")
        gen = generated.convert("RGB")
        m = mask.convert("L")
        # Keep precise boundaries by default; optional feather can be enabled via env.
        feather_px = int(os.getenv("REDRAW_MASK_FEATHER_PX", "0"))
        if feather_px > 0:
            m = m.filter(ImageFilter.GaussianBlur(radius=feather_px))
        return Image.composite(gen, base, m)


if __name__ == "__main__":
    # Tiny smoke test for loader structure without downloading models.
    test_img = Image.new("RGB", (256, 256), (240, 240, 240))
    test_mask = Image.new("L", (256, 256), 0)
    test_mask_np = np.array(test_mask)
    test_mask_np[96:160, 96:160] = 255
    test_mask = Image.fromarray(test_mask_np)
    redrawer = AdvancedRedrawer()
    print("AdvancedRedrawer constructed:", redrawer is not None)
    print("Mask pixels:", int(np.count_nonzero(np.array(test_mask))))
    # Optional real load/inference benchmark:
    # ADVANCED_REDRAWER_LOAD_TEST=1 python backend/advanced_redrawer.py
    if os.getenv("ADVANCED_REDRAWER_LOAD_TEST") == "1":
        t0 = time.time()
        loaded = redrawer.ensure_loaded()
        print("Loaded:", loaded, "in", round(time.time() - t0, 2), "s")
