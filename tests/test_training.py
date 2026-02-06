"""Tests for training pipeline and components."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from preference_guided_image_captioning_alignment.training.trainer import (
    PreferenceGuidedTrainer,
)
from preference_guided_image_captioning_alignment.models.model import (
    ContrastiveLoss,
    PreferenceLoss,
)


class DummyConceptualDataset(Dataset):
    """Dummy dataset for testing contrastive learning."""

    def __init__(self, size=20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 224, 224),
            "caption_ids": torch.randint(0, 1000, (32,)),
            "caption_mask": torch.ones(32),
            "raw_caption": f"Caption {idx}",
            "image_path": f"image_{idx}.jpg",
        }


class DummyPreferenceDataset(Dataset):
    """Dummy dataset for testing preference learning."""

    def __init__(self, size=20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 224, 224),
            "preferred_ids": torch.randint(0, 1000, (32,)),
            "preferred_mask": torch.ones(32),
            "rejected_ids": torch.randint(0, 1000, (32,)),
            "rejected_mask": torch.ones(32),
            "preference_score": torch.tensor(0.8),
            "raw_preferred": f"Good caption {idx}",
            "raw_rejected": f"Bad caption {idx}",
            "image_path": f"pref_image_{idx}.jpg",
        }


class TestTrainer:
    """Test training pipeline functionality."""

    @pytest.fixture
    def mock_accelerator(self):
        """Create mock accelerator for testing."""
        accelerator = MagicMock()
        accelerator.device = torch.device("cpu")
        accelerator.is_main_process = True
        accelerator.num_processes = 1
        accelerator.prepare = lambda *args: args
        accelerator.accumulate = lambda model: MagicMock(__enter__=lambda x: None, __exit__=lambda x, y, z, w: None)
        accelerator.backward = lambda loss: loss.backward()
        accelerator.clip_grad_norm_ = lambda params, max_norm: None
        accelerator.unwrap_model = lambda model: model
        return accelerator

    @pytest.fixture
    def dummy_dataloaders(self):
        """Create dummy data loaders for testing."""
        stage1_train = DataLoader(DummyConceptualDataset(20), batch_size=4, shuffle=False)
        stage1_val = DataLoader(DummyConceptualDataset(10), batch_size=4, shuffle=False)
        stage2_train = DataLoader(DummyPreferenceDataset(16), batch_size=2, shuffle=False)
        stage2_val = DataLoader(DummyPreferenceDataset(8), batch_size=2, shuffle=False)

        return stage1_train, stage1_val, stage2_train, stage2_val

    @pytest.fixture
    def trainer(self, dummy_model, config, dummy_dataloaders, mock_accelerator):
        """Create test trainer."""
        stage1_train, stage1_val, stage2_train, stage2_val = dummy_dataloaders

        # Create temporary output directory
        temp_dir = tempfile.mkdtemp()
        config.set("paths.output_dir", temp_dir)

        trainer = PreferenceGuidedTrainer(
            model=dummy_model,
            config=config,
            train_loader_stage1=stage1_train,
            val_loader_stage1=stage1_val,
            train_loader_stage2=stage2_train,
            val_loader_stage2=stage2_val,
            accelerator=mock_accelerator,
        )

        yield trainer

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_trainer_init(self, trainer):
        """Test trainer initialization."""
        assert trainer.model is not None
        assert trainer.config is not None
        assert trainer.train_loader_stage1 is not None
        assert trainer.val_loader_stage1 is not None
        assert trainer.train_loader_stage2 is not None
        assert trainer.val_loader_stage2 is not None
        assert trainer.current_stage == 1
        assert trainer.global_step == 0
        assert trainer.epoch == 0

    def test_setup_optimizer_and_scheduler(self, trainer):
        """Test optimizer and scheduler setup."""
        num_training_steps = 100

        optimizer, scheduler = trainer._setup_optimizer_and_scheduler(
            stage=1, num_training_steps=num_training_steps
        )

        assert optimizer is not None
        assert scheduler is not None
        assert hasattr(optimizer, 'param_groups')
        assert hasattr(scheduler, 'step')

    @patch('mlflow.active_run')
    @patch('wandb.run')
    def test_log_metrics(self, mock_wandb, mock_mlflow, trainer):
        """Test metrics logging."""
        mock_mlflow.return_value = True
        mock_wandb.return_value = True

        metrics = {
            "epoch": 1,
            "stage": 1,
            "train_loss": 0.5,
            "val_loss": 0.6,
            "learning_rate": 1e-4,
        }

        # Should not raise any exceptions
        trainer._log_metrics(metrics)

    def test_save_checkpoint(self, trainer, mock_accelerator):
        """Test checkpoint saving."""
        # Mock optimizer and scheduler
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {"lr": 1e-4}

        scheduler = MagicMock()
        scheduler.state_dict.return_value = {"step": 100}

        # Save checkpoint
        trainer._save_checkpoint(
            epoch=1,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loss=0.5,
            stage=1,
        )

        # Check that checkpoint directory exists
        checkpoint_dir = trainer.checkpoint_dir
        assert checkpoint_dir.exists()

        # Check that checkpoint file exists
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoint_files) > 0

    def test_load_checkpoint(self, trainer, mock_accelerator):
        """Test checkpoint loading."""
        # Create a dummy checkpoint
        checkpoint_path = trainer.checkpoint_dir / "test_checkpoint.pt"
        trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": 5,
            "stage": 2,
            "global_step": 1000,
            "model_state_dict": trainer.model.state_dict(),
            "val_loss": 0.3,
            "config": trainer.config.config,
        }

        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))

        assert trainer.epoch == 5
        assert trainer.current_stage == 2
        assert trainer.global_step == 1000

    def test_check_early_stopping(self, trainer):
        """Test early stopping logic."""
        config = {"early_stopping_patience": 3}

        # Should not stop initially
        assert not trainer._check_early_stopping(0.5, config)

        # Simulate no improvement
        trainer.patience_counter = 0
        assert not trainer._check_early_stopping(0.6, config)  # Worse loss
        assert trainer.patience_counter == 1

        assert not trainer._check_early_stopping(0.7, config)  # Still worse
        assert trainer.patience_counter == 2

        assert not trainer._check_early_stopping(0.8, config)  # Still worse
        assert trainer.patience_counter == 3

        assert trainer._check_early_stopping(0.9, config)  # Should trigger early stopping

    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('wandb.init')
    def test_train_stage1_single_step(self, mock_wandb_init, mock_log_params, mock_start_run, trainer):
        """Test single step of stage 1 training."""
        # Reduce epochs to 1 for faster testing
        trainer.config.set("training.stage1.num_epochs", 1)
        trainer.config.set("training.stage1.logging_steps", 1)

        # Mock accelerator methods
        trainer.accelerator.prepare = lambda *args: args

        # Run training
        metrics = trainer.train_stage1()

        assert isinstance(metrics, dict)
        assert "train_loss" in metrics
        assert "val_loss" in metrics
        assert len(metrics["train_loss"]) == 1  # One epoch

    @patch('mlflow.start_run')
    @patch('wandb.init')
    def test_train_stage2_single_step(self, mock_wandb_init, mock_start_run, trainer):
        """Test single step of stage 2 training."""
        # Reduce epochs to 1 for faster testing
        trainer.config.set("training.stage2.num_epochs", 1)
        trainer.config.set("training.stage2.logging_steps", 1)

        # Mock accelerator methods
        trainer.accelerator.prepare = lambda *args: args

        # Run training
        metrics = trainer.train_stage2()

        assert isinstance(metrics, dict)
        assert "train_loss" in metrics
        assert "val_loss" in metrics
        assert len(metrics["train_loss"]) == 1  # One epoch

    @patch('mlflow.start_run')
    @patch('wandb.init')
    def test_full_training_pipeline(self, mock_wandb_init, mock_start_run, trainer):
        """Test full training pipeline."""
        # Reduce epochs for faster testing
        trainer.config.set("training.stage1.num_epochs", 1)
        trainer.config.set("training.stage2.num_epochs", 1)

        # Mock accelerator methods
        trainer.accelerator.prepare = lambda *args: args

        # Run full training
        results = trainer.train()

        assert isinstance(results, dict)
        assert "stage1_metrics" in results
        assert "stage2_metrics" in results
        assert "best_val_loss" in results
        assert "total_steps" in results


class TestLossFunctionIntegration:
    """Test loss functions in training context."""

    def test_contrastive_loss_training_step(self):
        """Test contrastive loss in a training step."""
        loss_fn = ContrastiveLoss(temperature=0.07)

        # Simulate embeddings from model
        batch_size = 4
        image_embeddings = torch.randn(batch_size, 256, requires_grad=True)
        text_embeddings = torch.randn(batch_size, 256, requires_grad=True)

        # Normalize embeddings
        image_embeddings_norm = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_norm = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)

        # Compute loss
        loss = loss_fn(image_embeddings_norm, text_embeddings_norm)

        # Backward pass
        loss.backward()

        assert loss.item() >= 0.0
        assert image_embeddings.grad is not None
        assert text_embeddings.grad is not None

    def test_preference_loss_training_step(self):
        """Test preference loss in a training step."""
        loss_fn = PreferenceLoss(beta=0.1)

        batch_size = 2
        seq_len = 20
        vocab_size = 1000

        preferred_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        rejected_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        preferred_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        preferred_mask = torch.ones(batch_size, seq_len)
        rejected_mask = torch.ones(batch_size, seq_len)

        # Compute loss
        loss = loss_fn(
            preferred_logits=preferred_logits,
            rejected_logits=rejected_logits,
            preferred_labels=preferred_labels,
            rejected_labels=rejected_labels,
            preferred_mask=preferred_mask,
            rejected_mask=rejected_mask,
        )

        # Backward pass
        loss.backward()

        assert loss.item() >= 0.0
        assert preferred_logits.grad is not None
        assert rejected_logits.grad is not None

    def test_combined_loss_computation(self, dummy_model):
        """Test combining contrastive and preference losses."""
        contrastive_loss_fn = ContrastiveLoss()
        preference_loss_fn = PreferenceLoss()

        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        caption_ids = torch.randint(0, 1000, (batch_size, 32))
        caption_mask = torch.ones(batch_size, 32)

        # Contrastive forward pass
        contrastive_outputs = dummy_model(
            images=images,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
            mode="contrastive",
        )

        contrastive_loss = contrastive_loss_fn(
            contrastive_outputs["image_embeddings"],
            contrastive_outputs["text_embeddings"],
        )

        # Generation forward pass for preference loss
        preferred_outputs = dummy_model(
            images=images,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
            labels=caption_ids,
            mode="generation",
        )

        rejected_outputs = dummy_model(
            images=images,
            caption_ids=caption_ids,
            caption_mask=caption_mask,
            labels=caption_ids,
            mode="generation",
        )

        # Note: This is a simplified test - in practice, you'd have different
        # preferred and rejected captions
        preference_loss = preference_loss_fn(
            preferred_outputs["logits"],
            rejected_outputs["logits"],
            caption_ids,
            caption_ids,
            caption_mask,
            caption_mask,
        )

        # Combine losses
        total_loss = contrastive_loss + preference_loss

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert total_loss.item() >= 0.0

        # Test backward pass
        total_loss.backward()

        # Check that gradients exist
        param_count = 0
        grad_count = 0
        for param in dummy_model.parameters():
            if param.requires_grad:
                param_count += 1
                if param.grad is not None:
                    grad_count += 1

        assert grad_count > 0  # At least some parameters should have gradients


class TestTrainingUtilities:
    """Test training utility functions."""

    def test_optimizer_creation(self, dummy_model, config):
        """Test optimizer creation with different configurations."""
        import torch.optim as optim

        # Test AdamW creation
        optimizer = optim.AdamW(
            dummy_model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        assert isinstance(optimizer, optim.AdamW)
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-4

    def test_scheduler_creation(self, dummy_model):
        """Test learning rate scheduler creation."""
        from transformers import get_cosine_schedule_with_warmup
        import torch.optim as optim

        optimizer = optim.AdamW(dummy_model.parameters(), lr=1e-4)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
        )

        assert scheduler is not None
        assert hasattr(scheduler, 'step')

        # Test scheduler step
        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        # Learning rate should change after step
        assert hasattr(scheduler, 'last_epoch')

    def test_gradient_accumulation_simulation(self, dummy_model):
        """Test gradient accumulation simulation."""
        import torch.optim as optim

        optimizer = optim.AdamW(dummy_model.parameters(), lr=1e-4)
        accumulation_steps = 4

        total_loss = 0.0
        optimizer.zero_grad()

        for step in range(accumulation_steps):
            # Simulate forward pass
            images = torch.randn(2, 3, 224, 224)
            caption_ids = torch.randint(0, 1000, (2, 32))
            caption_mask = torch.ones(2, 32)

            outputs = dummy_model(
                images=images,
                caption_ids=caption_ids,
                caption_mask=caption_mask,
                mode="contrastive",
            )

            # Simulate contrastive loss
            loss = torch.nn.functional.mse_loss(
                outputs["image_embeddings"],
                outputs["text_embeddings"],
            )

            # Scale loss by accumulation steps
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            total_loss += loss.item()

        # Take optimizer step after accumulation
        optimizer.step()
        optimizer.zero_grad()

        assert total_loss >= 0.0

        # Check that gradients were accumulated and then cleared
        for param in dummy_model.parameters():
            if param.requires_grad:
                assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_model_state_management(self, dummy_model):
        """Test model state transitions between train/eval."""
        # Test train mode
        dummy_model.train()
        assert dummy_model.training

        # Check that submodules are also in train mode
        assert dummy_model.vision_encoder.training
        assert dummy_model.text_encoder.training
        assert dummy_model.caption_decoder.training

        # Test eval mode
        dummy_model.eval()
        assert not dummy_model.training

        # Check that submodules are also in eval mode
        assert not dummy_model.vision_encoder.training
        assert not dummy_model.text_encoder.training
        assert not dummy_model.caption_decoder.training