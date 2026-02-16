"""Tests for model architecture and components."""

import pytest
import torch
import torch.nn as nn

from preference_guided_image_captioning_alignment.models.model import (
    VisionEncoder,
    TextEncoder,
    CaptionDecoder,
    PreferenceGuidedCaptioningModel,
    ContrastiveLoss,
    PreferenceLoss,
)


class TestVisionEncoder:
    """Test vision encoder functionality."""

    @pytest.fixture
    def vision_encoder(self):
        """Create test vision encoder."""
        return VisionEncoder(
            model_name="openai/clip-vit-base-patch32",
            projection_dim=256,
            freeze_backbone=False,
            dropout=0.1,
        )

    def test_init(self, vision_encoder):
        """Test vision encoder initialization."""
        assert vision_encoder.projection_dim == 256
        assert vision_encoder.freeze_backbone is False
        assert hasattr(vision_encoder, 'vision_model')
        assert hasattr(vision_encoder, 'projection')

    def test_forward_pass(self, vision_encoder):
        """Test vision encoder forward pass."""
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        outputs = vision_encoder(pixel_values)

        assert isinstance(outputs, dict)
        assert "features" in outputs
        assert "embeddings" in outputs
        assert "pooled_output" in outputs

        # Check output shapes
        assert outputs["embeddings"].shape == (batch_size, 256)
        assert len(outputs["features"].shape) == 3  # (batch, seq_len, hidden_size)
        assert outputs["pooled_output"].shape[0] == batch_size

    def test_input_validation(self, vision_encoder):
        """Test vision encoder input validation."""
        # Test wrong number of dimensions
        with pytest.raises(ValueError, match="Expected pixel_values to be 4D tensor"):
            vision_encoder(torch.randn(3, 224, 224))  # Missing batch dimension

        # Test wrong number of channels
        with pytest.raises(ValueError, match="Expected 3 channels"):
            vision_encoder(torch.randn(2, 4, 224, 224))  # 4 channels instead of 3

    def test_different_batch_sizes(self, vision_encoder):
        """Test vision encoder with different batch sizes."""
        for batch_size in [1, 4, 8]:
            pixel_values = torch.randn(batch_size, 3, 224, 224)
            outputs = vision_encoder(pixel_values)

            assert outputs["embeddings"].shape[0] == batch_size
            assert outputs["features"].shape[0] == batch_size
            assert outputs["pooled_output"].shape[0] == batch_size

    def test_freeze_backbone(self):
        """Test backbone freezing functionality."""
        encoder = VisionEncoder(
            model_name="openai/clip-vit-base-patch32",
            projection_dim=256,
            freeze_backbone=True,
        )

        # Check that vision model parameters are frozen
        for param in encoder.vision_model.parameters():
            assert not param.requires_grad

        # Check that projection parameters are not frozen
        for param in encoder.projection.parameters():
            assert param.requires_grad

    def test_different_batch_sizes(self, vision_encoder):
        """Test with different batch sizes."""
        for batch_size in [1, 4, 8]:
            pixel_values = torch.randn(batch_size, 3, 224, 224)
            outputs = vision_encoder(pixel_values)

            assert outputs["embeddings"].shape == (batch_size, 256)


class TestTextEncoder:
    """Test text encoder functionality."""

    @pytest.fixture
    def text_encoder(self):
        """Create test text encoder."""
        return TextEncoder(
            model_name="microsoft/DialoGPT-medium",
            projection_dim=256,
            freeze_backbone=False,
            dropout=0.1,
        )

    def test_init(self, text_encoder):
        """Test text encoder initialization."""
        assert text_encoder.projection_dim == 256
        assert text_encoder.freeze_backbone is False
        assert hasattr(text_encoder, 'text_model')
        assert hasattr(text_encoder, 'tokenizer')
        assert hasattr(text_encoder, 'projection')

    def test_forward_pass(self, text_encoder):
        """Test text encoder forward pass."""
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = text_encoder(input_ids, attention_mask)

        assert isinstance(outputs, dict)
        assert "features" in outputs
        assert "embeddings" in outputs
        assert "pooled_output" in outputs

        # Check output shapes
        assert outputs["embeddings"].shape == (batch_size, 256)
        assert outputs["features"].shape == (batch_size, seq_len, text_encoder.feature_dim)
        assert outputs["pooled_output"].shape == (batch_size, text_encoder.feature_dim)

    def test_return_hidden_states(self, text_encoder):
        """Test returning hidden states."""
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = text_encoder(input_ids, attention_mask, return_hidden_states=True)

        assert "hidden_states" in outputs
        assert isinstance(outputs["hidden_states"], tuple)

    def test_input_validation(self, text_encoder):
        """Test text encoder input validation."""
        # Test wrong number of dimensions for input_ids
        with pytest.raises(ValueError, match="Expected input_ids to be 2D tensor"):
            text_encoder(
                torch.randint(0, 1000, (32,)),  # 1D instead of 2D
                torch.ones(2, 32)
            )

        # Test wrong number of dimensions for attention_mask
        with pytest.raises(ValueError, match="Expected attention_mask to be 2D tensor"):
            text_encoder(
                torch.randint(0, 1000, (2, 32)),
                torch.ones(32,)  # 1D instead of 2D
            )

        # Test mismatched shapes
        with pytest.raises(ValueError, match="input_ids shape .* doesn't match attention_mask shape"):
            text_encoder(
                torch.randint(0, 1000, (2, 32)),
                torch.ones(2, 16)  # Different sequence length
            )

    def test_attention_mask_handling(self, text_encoder):
        """Test proper attention mask handling in pooling."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Create attention mask with padding
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 5:] = 0  # Pad second half of first sequence
        attention_mask[1, 8:] = 0  # Pad last two tokens of second sequence

        outputs = text_encoder(input_ids, attention_mask)

        # Should not raise errors and produce valid embeddings
        assert outputs["embeddings"].shape == (batch_size, text_encoder.projection_dim)
        assert not torch.isnan(outputs["embeddings"]).any()
        assert not torch.isinf(outputs["embeddings"]).any()

    def test_freeze_backbone(self):
        """Test backbone freezing functionality."""
        encoder = TextEncoder(
            model_name="microsoft/DialoGPT-medium",
            projection_dim=256,
            freeze_backbone=True,
        )

        # Check that text model parameters are frozen
        for param in encoder.text_model.parameters():
            assert not param.requires_grad

        # Check that projection parameters are not frozen
        for param in encoder.projection.parameters():
            assert param.requires_grad


class TestCaptionDecoder:
    """Test caption decoder functionality."""

    @pytest.fixture
    def caption_decoder(self):
        """Create test caption decoder."""
        return CaptionDecoder(
            model_name="microsoft/DialoGPT-medium",
            vision_feature_dim=256,
            dropout=0.1,
        )

    def test_init(self, caption_decoder):
        """Test caption decoder initialization."""
        assert caption_decoder.vision_feature_dim == 256
        assert hasattr(caption_decoder, 'lm_model')
        assert hasattr(caption_decoder, 'tokenizer')
        assert hasattr(caption_decoder, 'vision_projection')
        assert hasattr(caption_decoder, 'cross_attention')

    def test_forward_pass(self, caption_decoder):
        """Test caption decoder forward pass."""
        batch_size = 2
        seq_len = 20
        vision_features = torch.randn(batch_size, 256)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = caption_decoder(
            vision_features=vision_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert hasattr(outputs, 'logits')
        assert outputs.logits.shape == (batch_size, seq_len, caption_decoder.vocab_size)

    def test_generation_mode(self, caption_decoder):
        """Test generation mode without input_ids."""
        batch_size = 2
        vision_features = torch.randn(batch_size, 256)

        outputs = caption_decoder(vision_features=vision_features)

        assert hasattr(outputs, 'logits')
        assert outputs.logits.shape[0] == batch_size

    def test_generate_method(self, caption_decoder):
        """Test caption generation method."""
        batch_size = 2
        vision_features = torch.randn(batch_size, 256)

        # Test generation
        generated_ids = caption_decoder.generate(
            vision_features=vision_features,
            max_length=20,
            num_beams=2,
            do_sample=False,
        )

        assert isinstance(generated_ids, torch.Tensor)
        assert generated_ids.shape[0] == batch_size
        assert generated_ids.shape[1] <= 20


class TestPreferenceGuidedCaptioningModel:
    """Test the main preference-guided captioning model."""

    def test_init(self, dummy_model):
        """Test model initialization."""
        assert hasattr(dummy_model, 'vision_encoder')
        assert hasattr(dummy_model, 'text_encoder')
        assert hasattr(dummy_model, 'caption_decoder')
        assert dummy_model.projection_dim == 256

    def test_contrastive_forward(self, dummy_model):
        """Test forward pass in contrastive mode."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        caption_ids = torch.randint(0, 1000, (batch_size, 32))
        caption_mask = torch.ones(batch_size, 32)

        outputs = dummy_model(
            images=images,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
            mode="contrastive",
        )

        assert "image_embeddings" in outputs
        assert "text_embeddings" in outputs
        assert "vision_features" in outputs
        assert "text_features" in outputs

        assert outputs["image_embeddings"].shape == (batch_size, 256)
        assert outputs["text_embeddings"].shape == (batch_size, 256)

    def test_generation_forward(self, dummy_model):
        """Test forward pass in generation mode."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        caption_ids = torch.randint(0, 1000, (batch_size, 32))
        caption_mask = torch.ones(batch_size, 32)

        outputs = dummy_model(
            images=images,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
            mode="generation",
        )

        assert "logits" in outputs
        assert "generation_loss" in outputs

        expected_vocab_size = dummy_model.caption_decoder.vocab_size
        assert outputs["logits"].shape == (batch_size, 32, expected_vocab_size)

    def test_dual_mode(self, dummy_model):
        """Test forward pass in dual mode."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        caption_ids = torch.randint(0, 1000, (batch_size, 32))
        caption_mask = torch.ones(batch_size, 32)

        outputs = dummy_model(
            images=images,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
            mode="dual",
        )

        # Should have both contrastive and generation outputs
        assert "image_embeddings" in outputs
        assert "text_embeddings" in outputs
        assert "logits" in outputs
        assert "generation_loss" in outputs

    def test_generate_captions(self, dummy_model):
        """Test caption generation method."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        captions = dummy_model.generate_captions(
            images=images,
            max_length=20,
            num_beams=2,
            do_sample=False,
        )

        assert isinstance(captions, list)
        assert len(captions) == batch_size
        assert all(isinstance(caption, str) for caption in captions)

    def test_compute_similarity(self, dummy_model):
        """Test similarity computation."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        captions = torch.randint(0, 1000, (batch_size, 32))
        caption_mask = torch.ones(batch_size, 32)

        similarity = dummy_model.compute_similarity(images, captions, caption_mask)

        assert isinstance(similarity, torch.Tensor)
        assert similarity.shape == (batch_size, batch_size)

    def test_model_device_consistency(self, dummy_model):
        """Test that model components are on same device."""
        device = next(dummy_model.parameters()).device

        # Check that all submodules are on the same device
        assert next(dummy_model.vision_encoder.parameters()).device == device
        assert next(dummy_model.text_encoder.parameters()).device == device
        assert next(dummy_model.caption_decoder.parameters()).device == device


class TestLossFunctions:
    """Test loss function implementations."""

    def test_contrastive_loss(self):
        """Test contrastive loss computation."""
        loss_fn = ContrastiveLoss(temperature=0.07)

        batch_size = 4
        embedding_dim = 256

        # Create normalized embeddings
        image_embeddings = torch.randn(batch_size, embedding_dim)
        text_embeddings = torch.randn(batch_size, embedding_dim)

        # Normalize
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)

        loss = loss_fn(image_embeddings, text_embeddings)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0.0

    def test_preference_loss(self):
        """Test preference loss computation."""
        loss_fn = PreferenceLoss(beta=0.1)

        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        preferred_logits = torch.randn(batch_size, seq_len, vocab_size)
        rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
        preferred_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        preferred_mask = torch.ones(batch_size, seq_len)
        rejected_mask = torch.ones(batch_size, seq_len)

        loss = loss_fn(
            preferred_logits=preferred_logits,
            rejected_logits=rejected_logits,
            preferred_labels=preferred_labels,
            rejected_labels=rejected_labels,
            preferred_mask=preferred_mask,
            rejected_mask=rejected_mask,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0.0

    def test_preference_loss_log_probs(self):
        """Test internal log probability computation."""
        loss_fn = PreferenceLoss(beta=0.1)

        batch_size = 2
        seq_len = 10
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)

        log_probs = loss_fn._compute_log_probs(logits, labels, mask)

        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.shape == (batch_size,)

    def test_contrastive_loss_temperature_effect(self):
        """Test effect of temperature on contrastive loss."""
        batch_size = 4
        embedding_dim = 256

        # Create identical embeddings (should have low loss)
        embeddings = torch.randn(batch_size, embedding_dim)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        # Test different temperatures
        loss_low_temp = ContrastiveLoss(temperature=0.01)(embeddings, embeddings)
        loss_high_temp = ContrastiveLoss(temperature=1.0)(embeddings, embeddings)

        # Lower temperature should give lower loss for perfectly aligned embeddings
        assert loss_low_temp.item() < loss_high_temp.item()

    def test_loss_backward_compatibility(self):
        """Test that losses work with gradient computation."""
        # Test contrastive loss
        contrastive_loss = ContrastiveLoss()
        image_emb = torch.randn(2, 256, requires_grad=True)
        text_emb = torch.randn(2, 256, requires_grad=True)

        loss = contrastive_loss(
            torch.nn.functional.normalize(image_emb, p=2, dim=-1),
            torch.nn.functional.normalize(text_emb, p=2, dim=-1)
        )
        loss.backward()

        assert image_emb.grad is not None
        assert text_emb.grad is not None

        # Test preference loss
        preference_loss = PreferenceLoss()
        preferred_logits = torch.randn(2, 10, 100, requires_grad=True)
        rejected_logits = torch.randn(2, 10, 100, requires_grad=True)
        preferred_labels = torch.randint(0, 100, (2, 10))
        rejected_labels = torch.randint(0, 100, (2, 10))
        mask = torch.ones(2, 10)

        loss = preference_loss(
            preferred_logits, rejected_logits,
            preferred_labels, rejected_labels,
            mask, mask
        )
        loss.backward()

        assert preferred_logits.grad is not None


class TestModelIntegration:
    """Test complete model integration and end-to-end functionality."""

    @pytest.fixture
    def complete_model(self):
        """Create complete preference-guided model for testing."""
        return PreferenceGuidedCaptioningModel(
            vision_model="openai/clip-vit-base-patch32",
            text_model="microsoft/DialoGPT-medium",
            projection_dim=256,
            freeze_backbone=False
        )

    def test_end_to_end_contrastive_mode(self, complete_model):
        """Test complete model in contrastive mode."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        caption_ids = torch.randint(0, 1000, (batch_size, 32))
        caption_mask = torch.ones(batch_size, 32)

        outputs = complete_model(
            images=images,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
            mode="contrastive"
        )

        assert "image_embeddings" in outputs
        assert "text_embeddings" in outputs
        assert outputs["image_embeddings"].shape == (batch_size, 256)
        assert outputs["text_embeddings"].shape == (batch_size, 256)

    def test_caption_generation(self, complete_model):
        """Test caption generation functionality."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        generated = complete_model.generate_captions(
            images=images,
            max_length=20,
            num_beams=2,
            temperature=0.8,
            do_sample=True
        )

        assert isinstance(generated, list)
        assert len(generated) == batch_size
        assert all(isinstance(caption, str) for caption in generated)

    def test_similarity_computation(self, complete_model):
        """Test image-caption similarity computation."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        captions = ["A cat sitting on a table", "A dog playing in the park"]

        similarities = complete_model.compute_similarity(images, captions)

        assert similarities.shape == (batch_size, len(captions))
        assert not torch.isnan(similarities).any()
        assert not torch.isinf(similarities).any()

    def test_model_device_consistency(self, complete_model):
        """Test that model components are on the same device."""
        device = next(complete_model.parameters()).device

        # Check all major components are on the same device
        assert next(complete_model.vision_encoder.parameters()).device == device
        assert next(complete_model.text_encoder.parameters()).device == device
        assert next(complete_model.caption_decoder.parameters()).device == device

    def test_model_training_mode_switching(self, complete_model):
        """Test switching between training and evaluation modes."""
        # Test training mode
        complete_model.train()
        assert complete_model.training
        assert complete_model.vision_encoder.training
        assert complete_model.text_encoder.training
        assert complete_model.caption_decoder.training

        # Test evaluation mode
        complete_model.eval()
        assert not complete_model.training
        assert not complete_model.vision_encoder.training
        assert not complete_model.text_encoder.training
        assert not complete_model.caption_decoder.training

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_variable_batch_sizes(self, complete_model, batch_size):
        """Test model with different batch sizes."""
        images = torch.randn(batch_size, 3, 224, 224)
        caption_ids = torch.randint(0, 1000, (batch_size, 16))
        caption_mask = torch.ones(batch_size, 16)

        outputs = complete_model(
            images=images,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
            mode="contrastive"
        )

        assert outputs["image_embeddings"].shape[0] == batch_size
        assert outputs["text_embeddings"].shape[0] == batch_size
        assert rejected_logits.grad is not None


class TestModelIntegration:
    """Integration tests for model components."""

    def test_end_to_end_contrastive(self, dummy_model):
        """Test end-to-end contrastive learning."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        caption_ids = torch.randint(0, 1000, (batch_size, 32))
        caption_mask = torch.ones(batch_size, 32)

        # Forward pass
        outputs = dummy_model(
            images=images,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
            mode="contrastive",
        )

        # Compute loss
        contrastive_loss = ContrastiveLoss()
        loss = contrastive_loss(
            outputs["image_embeddings"],
            outputs["text_embeddings"],
        )

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        assert any(p.grad is not None for p in dummy_model.parameters() if p.requires_grad)

    def test_end_to_end_generation(self, dummy_model):
        """Test end-to-end generation."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        caption_ids = torch.randint(0, 1000, (batch_size, 32))
        caption_mask = torch.ones(batch_size, 32)

        # Forward pass
        outputs = dummy_model(
            images=images,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
            labels=caption_ids,
            mode="generation",
        )

        # Check generation loss
        if outputs["generation_loss"] is not None:
            loss = outputs["generation_loss"]
            loss.backward()

            # Check that gradients are computed
            assert any(p.grad is not None for p in dummy_model.parameters() if p.requires_grad)

    def test_model_eval_mode(self, dummy_model):
        """Test model behavior in evaluation mode."""
        dummy_model.eval()

        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            captions = dummy_model.generate_captions(images, max_length=10)

        assert len(captions) == batch_size
        assert all(isinstance(caption, str) for caption in captions)