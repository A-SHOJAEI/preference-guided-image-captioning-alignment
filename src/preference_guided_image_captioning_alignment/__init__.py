"""
Preference-Guided Image Captioning Alignment

A novel multimodal alignment system that combines image-caption contrastive learning
with human preference optimization for generating accurate and human-aligned captions.
"""

__version__ = "0.1.0"
__author__ = "A-SHOJAEI"

# Lazy imports to avoid triggering heavy dependency loading (torch, transformers,
# pycocotools, etc.) at package import time. This prevents circular import issues
# and speeds up CLI/script startup.


def __getattr__(name: str):
    """Lazy import of package-level classes and functions."""
    _lazy_imports = {
        "PreferenceGuidedCaptioningModel": ".models.model",
        "ConceptualCaptionsDataset": ".data.loader",
        "UltraFeedbackDataset": ".data.loader",
        "PreferenceGuidedTrainer": ".training.trainer",
        "CaptioningMetrics": ".evaluation.metrics",
        "Config": ".utils.config",
    }

    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __name__)
        value = getattr(module, name)
        # Cache on the module so subsequent access doesn't re-import
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PreferenceGuidedCaptioningModel",
    "ConceptualCaptionsDataset",
    "UltraFeedbackDataset",
    "PreferenceGuidedTrainer",
    "CaptioningMetrics",
    "Config",
]