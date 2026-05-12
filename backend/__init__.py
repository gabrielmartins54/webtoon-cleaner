try:
    from .text_mask_detector import TextMaskDetector
except ImportError:
    from text_mask_detector import TextMaskDetector

__all__ = ["TextMaskDetector"]
