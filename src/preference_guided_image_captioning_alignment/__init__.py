"""
Preference-Guided Image Captioning Alignment

A novel multimodal alignment system that combines image-caption contrastive learning
with human preference optimization for generating accurate and human-aligned captions.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .models.model import PreferenceGuidedCaptioningModel
from .data.loader import ConceptualCaptionsDataset, UltraFeedbackDataset
from .training.trainer import PreferenceGuidedTrainer
from .evaluation.metrics import CaptioningMetrics
from .utils.config import Config

__all__ = [
    "PreferenceGuidedCaptioningModel",
    "ConceptualCaptionsDataset",
    "UltraFeedbackDataset",
    "PreferenceGuidedTrainer",
    "CaptioningMetrics",
    "Config",
]