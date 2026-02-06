"""Test fixtures and configuration for pytest."""

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
from PIL import Image

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preference_guided_image_captioning_alignment.data.preprocessing import (
    ImageProcessor,
    TextProcessor,
)
from preference_guided_image_captioning_alignment.models.model import (
    PreferenceGuidedCaptioningModel,
)
from preference_guided_image_captioning_alignment.utils.config import Config


@pytest.fixture
def temp_config_file():
    """Create temporary configuration file."""
    config_content = """
# Test configuration
data:
  image_size: 224
  max_caption_length: 64
  num_workers: 0
  pin_memory: false

model:
  vision_model: "openai/clip-vit-base-patch32"
  text_model: "microsoft/DialoGPT-medium"
  projection_dim: 256
  temperature: 0.07
  dropout: 0.1
  freeze_vision_backbone: false
  freeze_text_backbone: false

training:
  stage1:
    batch_size: 4
    learning_rate: 1e-4
    weight_decay: 0.01
    num_epochs: 1
    warmup_steps: 10
    gradient_accumulation_steps: 1
    max_grad_norm: 1.0

  stage2:
    batch_size: 2
    learning_rate: 1e-5
    weight_decay: 0.01
    num_epochs: 1
    warmup_steps: 5
    gradient_accumulation_steps: 2
    max_grad_norm: 1.0
    dpo_beta: 0.1

  seed: 42

evaluation:
  metrics: ["bleu", "rouge"]
  generate_config:
    max_length: 50
    num_beams: 2

targets:
  cider_score: 1.0
  preference_win_rate: 0.6

logging:
  level: "WARNING"

paths:
  output_dir: "./test_outputs"
  cache_dir: "./test_cache"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        yield f.name

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def config(temp_config_file):
    """Create test configuration object."""
    return Config(temp_config_file)


@pytest.fixture
def image_processor():
    """Create test image processor."""
    return ImageProcessor(
        image_size=224,
        augment=False,  # Disable augmentation for reproducible tests
    )


@pytest.fixture
def text_processor():
    """Create test text processor."""
    return TextProcessor(
        model_name="microsoft/DialoGPT-medium",
        max_length=64,
        padding="max_length",
        truncation=True,
    )


@pytest.fixture
def dummy_model(config):
    """Create test model with minimal configuration."""
    model_config = config.get_model_config()
    return PreferenceGuidedCaptioningModel(
        vision_model=model_config["vision_model"],
        text_model=model_config["text_model"],
        projection_dim=model_config["projection_dim"],
        temperature=model_config["temperature"],
        dropout=model_config["dropout"],
    )


@pytest.fixture
def sample_image():
    """Create sample PIL image."""
    image = Image.new("RGB", (224, 224), color="red")
    return image


@pytest.fixture
def sample_images():
    """Create batch of sample PIL images."""
    colors = ["red", "green", "blue", "yellow"]
    images = []
    for color in colors:
        image = Image.new("RGB", (224, 224), color=color)
        images.append(image)
    return images


@pytest.fixture
def sample_captions():
    """Create sample captions."""
    return [
        "A red car on the street",
        "A green tree in the park",
        "A blue sky with clouds",
        "A yellow flower in the garden",
    ]


@pytest.fixture
def sample_image_tensor():
    """Create sample image tensor."""
    return torch.randn(3, 224, 224)


@pytest.fixture
def sample_image_batch():
    """Create batch of sample image tensors."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def processed_caption_batch(text_processor, sample_captions):
    """Create batch of processed captions."""
    encoding = text_processor.encode_batch(sample_captions)
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "raw_captions": sample_captions,
    }


@pytest.fixture
def preference_data():
    """Create sample preference data."""
    return {
        "preferred_captions": [
            "A beautiful red sports car driving on a sunny road",
            "A magnificent green oak tree standing in a peaceful park",
        ],
        "rejected_captions": [
            "A car",
            "A tree",
        ],
        "preference_scores": [0.8, 0.9],
    }


@pytest.fixture
def temp_dataset_dir():
    """Create temporary dataset directory with sample data."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create sample dataset files
    sample_data = [
        {"image_path": "image1.jpg", "caption": "A red car on the street"},
        {"image_path": "image2.jpg", "caption": "A green tree in the park"},
        {"image_path": "image3.jpg", "caption": "A blue sky with clouds"},
    ]

    # Save as JSON
    import json
    with open(temp_dir / "annotations.json", "w") as f:
        json.dump(sample_data, f)

    # Create dummy image files
    for item in sample_data:
        image_path = temp_dir / item["image_path"]
        image = Image.new("RGB", (224, 224), color="white")
        image.save(image_path)

    yield temp_dir

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_preference_dataset():
    """Create temporary preference dataset."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create preference data
    preference_data = [
        {
            "image_path": "pref_image1.jpg",
            "preferred_caption": "A beautiful red sports car on a winding mountain road",
            "rejected_caption": "A red car",
            "preference_score": 0.85,
        },
        {
            "image_path": "pref_image2.jpg",
            "preferred_caption": "A majestic green oak tree in a serene park setting",
            "rejected_caption": "A tree",
            "preference_score": 0.92,
        },
    ]

    # Save preference data
    import json
    with open(temp_dir / "preferences.json", "w") as f:
        json.dump(preference_data, f)

    # Create dummy images
    for item in preference_data:
        image_path = temp_dir / item["image_path"]
        image = Image.new("RGB", (224, 224), color="blue")
        image.save(image_path)

    yield temp_dir

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def device():
    """Get test device (CPU for consistent testing)."""
    return torch.device("cpu")


@pytest.fixture
def small_vocab_tokenizer():
    """Create tokenizer with small vocabulary for faster testing."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        return tokenizer
    except Exception:
        # Fallback to default if distilgpt2 is not available
        return AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    # Disable warnings for cleaner test output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set deterministic behavior
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Test markers
def pytest_collection_modifyitems(config, items):
    """Add markers to tests."""
    for item in items:
        # Mark slow tests
        if "model" in item.nodeid or "training" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark GPU tests
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)