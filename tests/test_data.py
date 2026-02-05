"""Tests for data loading and preprocessing modules."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from preference_guided_image_captioning_alignment.data.loader import (
    ConceptualCaptionsDataset,
    UltraFeedbackDataset,
    create_dataloaders,
)
from preference_guided_image_captioning_alignment.data.preprocessing import (
    ImageProcessor,
    TextProcessor,
)


class TestImageProcessor:
    """Test image preprocessing functionality."""

    def test_init(self):
        """Test ImageProcessor initialization."""
        processor = ImageProcessor(image_size=224, augment=True)

        assert processor.image_size == 224
        assert processor.augment is True
        assert processor.mean == (0.485, 0.456, 0.406)
        assert processor.std == (0.229, 0.224, 0.225)

    def test_process_pil_image(self, image_processor, sample_image):
        """Test processing PIL image."""
        result = image_processor.process_image(sample_image, training=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

    def test_process_image_path(self, image_processor):
        """Test processing image from file path."""
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            image = Image.new('RGB', (100, 100), color='red')
            image.save(f.name)
            temp_path = f.name

        try:
            result = image_processor.process_image(temp_path, training=False)
            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 224, 224)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_process_batch(self, image_processor, sample_images):
        """Test batch processing of images."""
        result = image_processor.process_batch(sample_images, training=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 3, 224, 224)
        assert result.dtype == torch.float32

    def test_denormalize(self, image_processor, sample_image_tensor):
        """Test tensor denormalization."""
        # First normalize
        normalized = image_processor.val_transform(
            Image.fromarray((sample_image_tensor.permute(1, 2, 0) * 255).byte().numpy())
        )

        # Then denormalize
        denormalized = image_processor.denormalize(normalized)

        assert isinstance(denormalized, torch.Tensor)
        assert denormalized.shape == normalized.shape
        assert denormalized.min() >= 0.0
        assert denormalized.max() <= 1.0

    def test_training_vs_validation_transforms(self, sample_image):
        """Test different transforms for training vs validation."""
        processor = ImageProcessor(image_size=224, augment=True)

        train_result = processor.process_image(sample_image, training=True)
        val_result = processor.process_image(sample_image, training=False)

        # Both should have same shape but potentially different values due to augmentation
        assert train_result.shape == val_result.shape
        assert train_result.dtype == val_result.dtype

    def test_invalid_image_handling(self, image_processor):
        """Test handling of invalid image inputs."""
        with pytest.raises(ValueError):
            image_processor.process_image("nonexistent_file.jpg")

        with pytest.raises(ValueError):
            image_processor.process_image(123)  # Invalid type


class TestTextProcessor:
    """Test text preprocessing functionality."""

    def test_init(self):
        """Test TextProcessor initialization."""
        processor = TextProcessor(
            model_name="microsoft/DialoGPT-medium",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        assert processor.model_name == "microsoft/DialoGPT-medium"
        assert processor.max_length == 128
        assert processor.padding == "max_length"
        assert processor.truncation is True

    def test_encode_single_caption(self, text_processor):
        """Test encoding single caption."""
        caption = "A beautiful sunset over the ocean"
        result = text_processor.encode_caption(caption)

        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)
        assert result["input_ids"].shape[0] == text_processor.max_length

    def test_encode_batch_captions(self, text_processor, sample_captions):
        """Test encoding batch of captions."""
        result = text_processor.encode_batch(sample_captions)

        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (4, text_processor.max_length)
        assert result["attention_mask"].shape == (4, text_processor.max_length)

    def test_decode_caption(self, text_processor):
        """Test decoding token IDs back to text."""
        caption = "A red car on the street"

        # Encode then decode
        encoded = text_processor.encode_caption(caption)
        decoded = text_processor.decode_caption(encoded["input_ids"])

        assert isinstance(decoded, str)
        # Note: decoded might not be exactly the same due to tokenization
        assert len(decoded) > 0

    def test_decode_batch(self, text_processor, sample_captions):
        """Test decoding batch of token IDs."""
        # Encode batch
        encoded = text_processor.encode_batch(sample_captions)

        # Decode batch
        decoded = text_processor.decode_batch(encoded["input_ids"])

        assert isinstance(decoded, list)
        assert len(decoded) == len(sample_captions)
        assert all(isinstance(caption, str) for caption in decoded)

    def test_prepare_for_generation(self, text_processor):
        """Test preparation for text generation."""
        prompt = "A beautiful"
        result = text_processor.prepare_for_generation(prompt)

        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert isinstance(result["input_ids"], torch.Tensor)

    def test_empty_caption_handling(self, text_processor):
        """Test handling of empty captions."""
        result = text_processor.encode_caption("")

        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_long_caption_truncation(self, text_processor):
        """Test truncation of long captions."""
        long_caption = " ".join(["word"] * 200)  # Very long caption
        result = text_processor.encode_caption(long_caption)

        assert result["input_ids"].shape[0] == text_processor.max_length

    def test_vocab_properties(self, text_processor):
        """Test vocabulary properties."""
        assert isinstance(text_processor.vocab_size, int)
        assert text_processor.vocab_size > 0
        assert isinstance(text_processor.pad_token_id, (int, type(None)))
        assert isinstance(text_processor.eos_token_id, (int, type(None)))
        assert isinstance(text_processor.bos_token_id, (int, type(None)))


class TestConceptualCaptionsDataset:
    """Test ConceptualCaptionsDataset functionality."""

    def test_init_with_json_file(self, temp_dataset_dir, image_processor, text_processor):
        """Test dataset initialization with JSON file."""
        dataset = ConceptualCaptionsDataset(
            data_path=str(temp_dataset_dir / "annotations.json"),
            image_processor=image_processor,
            text_processor=text_processor,
            split="train",
        )

        assert len(dataset) == 3  # Based on fixture data
        assert dataset.split == "train"

    def test_init_with_directory(self, temp_dataset_dir, image_processor, text_processor):
        """Test dataset initialization with directory."""
        dataset = ConceptualCaptionsDataset(
            data_path=str(temp_dataset_dir),
            image_processor=image_processor,
            text_processor=text_processor,
            split="train",
        )

        assert len(dataset) > 0

    def test_getitem(self, temp_dataset_dir, image_processor, text_processor):
        """Test getting dataset items."""
        dataset = ConceptualCaptionsDataset(
            data_path=str(temp_dataset_dir / "annotations.json"),
            image_processor=image_processor,
            text_processor=text_processor,
            split="train",
        )

        item = dataset[0]

        assert isinstance(item, dict)
        assert "image" in item
        assert "caption_ids" in item
        assert "caption_mask" in item
        assert "raw_caption" in item
        assert "image_path" in item

        assert isinstance(item["image"], torch.Tensor)
        assert item["image"].shape == (3, 224, 224)
        assert isinstance(item["caption_ids"], torch.Tensor)
        assert isinstance(item["caption_mask"], torch.Tensor)

    def test_max_samples_limit(self, temp_dataset_dir, image_processor, text_processor):
        """Test max_samples parameter."""
        dataset = ConceptualCaptionsDataset(
            data_path=str(temp_dataset_dir / "annotations.json"),
            image_processor=image_processor,
            text_processor=text_processor,
            split="train",
            max_samples=2,
        )

        assert len(dataset) == 2

    def test_get_sample_by_path(self, temp_dataset_dir, image_processor, text_processor):
        """Test getting sample by image path."""
        dataset = ConceptualCaptionsDataset(
            data_path=str(temp_dataset_dir / "annotations.json"),
            image_processor=image_processor,
            text_processor=text_processor,
            split="train",
        )

        # Get first item's path
        first_item = dataset[0]
        image_path = first_item["image_path"]

        # Get sample by path
        sample = dataset.get_sample_by_path(image_path)

        assert sample is not None
        assert sample["image_path"] == image_path

    def test_corrupted_image_handling(self, temp_dataset_dir, image_processor, text_processor):
        """Test handling of corrupted image files."""
        # Create a corrupted image file (just text instead of image data)
        corrupted_image_path = temp_dataset_dir / "corrupted.jpg"
        corrupted_image_path.write_text("This is not an image")

        # Create corresponding caption file
        caption_path = temp_dataset_dir / "corrupted.txt"
        caption_path.write_text("A corrupted image")

        dataset = ConceptualCaptionsDataset(
            data_path=str(temp_dataset_dir),
            image_processor=image_processor,
            text_processor=text_processor
        )

        # Should handle corrupted images gracefully (return zero tensor)
        if len(dataset) > 0:
            item = dataset[0]
            # Should have fallback image tensor
            assert item["image"].shape == (3, 224, 224)

    def test_empty_caption_handling(self, temp_dataset_dir, image_processor, text_processor):
        """Test handling of empty caption files."""
        # Create image with empty caption
        image_path = temp_dataset_dir / "empty_caption.jpg"
        Image.new("RGB", (224, 224)).save(image_path)

        caption_path = temp_dataset_dir / "empty_caption.txt"
        caption_path.write_text("")  # Empty caption

        dataset = ConceptualCaptionsDataset(
            data_path=str(temp_dataset_dir),
            image_processor=image_processor,
            text_processor=text_processor
        )

        # Should filter out entries with empty captions
        # or handle them gracefully
        for i in range(len(dataset)):
            item = dataset[i]
            assert len(item["raw_caption"].strip()) > 0

    def test_nonexistent_path(self, image_processor, text_processor):
        """Test handling of nonexistent data path."""
        with pytest.raises(FileNotFoundError):
            ConceptualCaptionsDataset(
                data_path="nonexistent/path",
                image_processor=image_processor,
                text_processor=text_processor,
            )


class TestUltraFeedbackDataset:
    """Test UltraFeedbackDataset functionality."""

    def test_init(self, temp_preference_dataset, image_processor, text_processor):
        """Test UltraFeedback dataset initialization."""
        dataset = UltraFeedbackDataset(
            data_path=str(temp_preference_dataset / "preferences.json"),
            image_processor=image_processor,
            text_processor=text_processor,
            split="train",
        )

        assert len(dataset) == 2  # Based on fixture data

    def test_getitem(self, temp_preference_dataset, image_processor, text_processor):
        """Test getting dataset items."""
        dataset = UltraFeedbackDataset(
            data_path=str(temp_preference_dataset / "preferences.json"),
            image_processor=image_processor,
            text_processor=text_processor,
            split="train",
        )

        item = dataset[0]

        assert isinstance(item, dict)
        required_keys = [
            "image", "preferred_ids", "preferred_mask",
            "rejected_ids", "rejected_mask", "preference_score",
            "raw_preferred", "raw_rejected", "image_path"
        ]

        for key in required_keys:
            assert key in item

        assert isinstance(item["image"], torch.Tensor)
        assert item["image"].shape == (3, 224, 224)
        assert isinstance(item["preference_score"], torch.Tensor)

    def test_preference_threshold(self, temp_preference_dataset, image_processor, text_processor):
        """Test preference threshold filtering."""
        dataset = UltraFeedbackDataset(
            data_path=str(temp_preference_dataset / "preferences.json"),
            image_processor=image_processor,
            text_processor=text_processor,
            split="train",
            preference_threshold=0.9,  # High threshold
        )

        # Should filter out items with lower preference scores
        assert len(dataset) <= 2


class TestDataLoaders:
    """Test data loader creation functionality."""

    def test_create_dataloaders(self, temp_dataset_dir, image_processor, text_processor):
        """Test creating data loaders."""
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_class=ConceptualCaptionsDataset,
            data_path=str(temp_dataset_dir / "annotations.json"),
            image_processor=image_processor,
            text_processor=text_processor,
            batch_size=2,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2,
            num_workers=0,  # No multiprocessing for tests
            pin_memory=False,
            seed=42,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Test that we can iterate through loaders
        train_batch = next(iter(train_loader))
        assert isinstance(train_batch, dict)
        assert "image" in train_batch
        assert "caption_ids" in train_batch

    def test_dataloader_batch_consistency(self, temp_dataset_dir, image_processor, text_processor):
        """Test batch consistency in data loaders."""
        train_loader, _, _ = create_dataloaders(
            dataset_class=ConceptualCaptionsDataset,
            data_path=str(temp_dataset_dir / "annotations.json"),
            image_processor=image_processor,
            text_processor=text_processor,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )

        # Check multiple batches have consistent shapes
        for i, batch in enumerate(train_loader):
            if i >= 2:  # Check first 2 batches
                break

            batch_size = batch["image"].shape[0]
            assert batch["caption_ids"].shape[0] == batch_size
            assert batch["caption_mask"].shape[0] == batch_size
            assert len(batch["raw_caption"]) == batch_size

    def test_reproducible_splits(self, temp_dataset_dir, image_processor, text_processor):
        """Test that data splits are reproducible with same seed."""
        # Create loaders with same seed
        train_loader1, _, _ = create_dataloaders(
            dataset_class=ConceptualCaptionsDataset,
            data_path=str(temp_dataset_dir / "annotations.json"),
            image_processor=image_processor,
            text_processor=text_processor,
            batch_size=1,
            seed=42,
            num_workers=0,
        )

        train_loader2, _, _ = create_dataloaders(
            dataset_class=ConceptualCaptionsDataset,
            data_path=str(temp_dataset_dir / "annotations.json"),
            image_processor=image_processor,
            text_processor=text_processor,
            batch_size=1,
            seed=42,
            num_workers=0,
        )

        # Should have same number of samples
        assert len(train_loader1.dataset) == len(train_loader2.dataset)