"""Training pipeline for preference-guided image captioning alignment.

This module implements a comprehensive two-stage training pipeline for preference-guided
image captioning models that combines contrastive learning and human preference optimization.

The training process consists of:
1. **Stage 1 - Contrastive Learning**: Learn multimodal alignment between images and
   captions using contrastive loss on large-scale image-caption datasets.
2. **Stage 2 - Preference Optimization**: Fine-tune for human preference alignment
   using preference pairs and DPO-style optimization.

Key Features:
- **Distributed Training**: Multi-GPU support via Hugging Face Accelerate
- **Mixed Precision**: FP16 training for memory efficiency and speed
- **Gradient Accumulation**: Support for large effective batch sizes
- **Flexible Scheduling**: Linear and cosine learning rate schedules with warmup
- **Comprehensive Logging**: MLflow and Weights & Biases integration
- **Checkpoint Management**: Automatic saving and loading of model states
- **Early Stopping**: Patience-based stopping to prevent overfitting

Classes:
    PreferenceGuidedTrainer: Main training orchestrator for two-stage pipeline.

Example:
    ```python
    from preference_guided_image_captioning_alignment.training.trainer import (
        PreferenceGuidedTrainer
    )

    # Initialize trainer
    trainer = PreferenceGuidedTrainer(
        model=model,
        config=config,
        train_loader_stage1=contrastive_loader,
        val_loader_stage1=val_contrastive_loader,
        train_loader_stage2=preference_loader,
        val_loader_stage2=val_preference_loader
    )

    # Execute two-stage training
    trainer.train_stage1(num_epochs=10)  # Contrastive learning
    trainer.train_stage2(num_epochs=5)   # Preference optimization

    # Or train both stages sequentially
    trainer.train_full_pipeline()
    ```

Note:
    This module requires proper configuration of data loaders, model, and
    accelerate for distributed training. See configs/default.yaml for
    complete training configuration examples.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import wandb

from ..models.model import (
    ContrastiveLoss,
    PreferenceLoss,
    PreferenceGuidedCaptioningModel,
)
from ..utils.config import Config


class PreferenceGuidedTrainer:
    """Two-stage training pipeline for preference-guided captioning models.

    This class orchestrates the complete training process for preference-guided image
    captioning models, implementing a carefully designed two-stage approach that first
    establishes strong multimodal alignment through contrastive learning, then optimizes
    for human preference alignment using preference optimization techniques.

    Training Stages:
        **Stage 1 - Contrastive Learning:**
        - Trains image and text encoders on large-scale image-caption pairs
        - Uses contrastive loss to align visual and textual representations
        - Establishes shared multimodal embedding space
        - Typically requires 10-20 epochs on datasets like Conceptual Captions

        **Stage 2 - Preference Optimization:**
        - Fine-tunes caption decoder using human preference data
        - Applies DPO (Direct Preference Optimization) or similar techniques
        - Optimizes for human-judged caption quality and relevance
        - Typically requires 3-8 epochs on preference datasets

    Key Features:
        - **Accelerated Training**: Built on Hugging Face Accelerate for multi-GPU support
        - **Memory Optimization**: Gradient accumulation and mixed precision (FP16/BF16)
        - **Flexible Scheduling**: Support for linear, cosine, and polynomial LR schedules
        - **Robust Logging**: Integration with MLflow and Weights & Biases
        - **Checkpointing**: Automatic model state saving and recovery
        - **Early Stopping**: Validation-based early stopping with configurable patience
        - **Stage Management**: Independent or sequential stage execution

    Attributes:
        model (PreferenceGuidedCaptioningModel): The model being trained.
        config (Config): Training configuration containing hyperparameters.
        train_loader_stage1 (DataLoader): Contrastive learning training data.
        val_loader_stage1 (DataLoader): Contrastive learning validation data.
        train_loader_stage2 (DataLoader): Preference optimization training data.
        val_loader_stage2 (DataLoader): Preference optimization validation data.
        accelerator (Accelerator): Distributed training accelerator.
        logger: Logger for training progress and debugging.

    Example:
        ```python
        # Configure training
        config = Config.from_yaml("configs/training.yaml")

        # Setup data loaders
        train_loader_contrastive = create_contrastive_dataloader(...)
        train_loader_preference = create_preference_dataloader(...)

        # Initialize trainer
        trainer = PreferenceGuidedTrainer(
            model=model,
            config=config,
            train_loader_stage1=train_loader_contrastive,
            val_loader_stage1=val_loader_contrastive,
            train_loader_stage2=train_loader_preference,
            val_loader_stage2=val_loader_preference
        )

        # Execute training pipeline
        trainer.train_full_pipeline()

        # Or train stages independently
        stage1_metrics = trainer.train_stage1(num_epochs=15)
        stage2_metrics = trainer.train_stage2(num_epochs=5)
        ```

    Note:
        Proper initialization requires careful attention to:
        - Data loader configuration for different training stages
        - Learning rate scheduling appropriate for each stage
        - Memory management for large batch sizes
        - Distributed training setup via accelerate config
    """

    def __init__(
        self,
        model: PreferenceGuidedCaptioningModel,
        config: Config,
        train_loader_stage1: DataLoader,
        val_loader_stage1: DataLoader,
        train_loader_stage2: Optional[DataLoader] = None,
        val_loader_stage2: Optional[DataLoader] = None,
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Preference-guided captioning model.
            config: Training configuration.
            train_loader_stage1: Training data loader for stage 1.
            val_loader_stage1: Validation data loader for stage 1.
            train_loader_stage2: Training data loader for stage 2.
            val_loader_stage2: Validation data loader for stage 2.
            accelerator: Accelerate accelerator for distributed training.
        """
        self.model = model
        self.config = config
        self.train_loader_stage1 = train_loader_stage1
        self.val_loader_stage1 = val_loader_stage1
        self.train_loader_stage2 = train_loader_stage2
        self.val_loader_stage2 = val_loader_stage2

        # Initialize accelerator
        if accelerator is None:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.get("training.gradient_accumulation_steps", 1),
                mixed_precision=config.get("hardware.mixed_precision", "fp16"),
                log_with="wandb" if config.get("logging.wandb_project") else None,
            )
        else:
            self.accelerator = accelerator

        self.device = self.accelerator.device
        self.logger = logging.getLogger(__name__)

        # Move model to device and prepare for distributed training
        self.model = self.accelerator.prepare(self.model)

        # Initialize loss functions
        self.contrastive_loss = ContrastiveLoss(
            temperature=config.get("model.temperature", 0.07)
        )
        self.preference_loss = PreferenceLoss(
            beta=config.get("training.stage2.dpo_beta", 0.1)
        )

        # Training state
        self.current_stage = 1
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Initialize logging
        self._setup_logging()
        self._setup_output_directories()

        self.logger.info("Initialized PreferenceGuidedTrainer")

    def _setup_logging(self) -> None:
        """Setup experiment logging with MLflow and Wandb."""
        # MLflow setup
        if self.config.get("logging.mlflow_experiment"):
            mlflow.set_experiment(self.config.get("logging.mlflow_experiment"))
            mlflow.start_run()
            mlflow.log_params(self.config.config)

        # Wandb setup
        if self.config.get("logging.wandb_project") and self.accelerator.is_main_process:
            wandb.init(
                project=self.config.get("logging.wandb_project"),
                config=self.config.config,
                name=f"preference-captioning-{int(time.time())}",
            )

    def _setup_output_directories(self) -> None:
        """Setup output directories for checkpoints and logs."""
        self.output_dir = Path(self.config.get("paths.output_dir", "./outputs"))
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"

        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _setup_optimizer_and_scheduler(
        self,
        stage: int,
        num_training_steps: int,
    ) -> Tuple[optim.Optimizer, LRScheduler]:
        """Setup optimizer and learning rate scheduler for given stage.

        Args:
            stage: Training stage (1 or 2).
            num_training_steps: Total number of training steps.

        Returns:
            Tuple of (optimizer, scheduler).
        """
        stage_config = self.config.get(f"training.stage{stage}")

        # Create optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=stage_config["learning_rate"],
            weight_decay=stage_config.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Create scheduler
        warmup_steps = stage_config.get("warmup_steps", 0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare with accelerator
        optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)

        return optimizer, scheduler

    def train_stage1(self) -> Dict[str, float]:
        """Train stage 1: Contrastive learning on image-caption pairs.

        Returns:
            Dictionary of training metrics.
        """
        self.logger.info("Starting Stage 1: Contrastive Learning")
        start_time = time.time()
        self.current_stage = 1

        # Log system information
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU Memory Available: {gpu_memory:.1f}GB")

        # Get stage 1 config
        stage1_config = self.config.get_stage1_config()
        num_epochs = stage1_config["num_epochs"]

        # Calculate and log training overview
        steps_per_epoch = len(self.train_loader_stage1) // stage1_config.get("gradient_accumulation_steps", 1)
        total_steps = steps_per_epoch * num_epochs
        self.logger.info(
            f"Stage 1 Training Plan: {num_epochs} epochs, {steps_per_epoch} steps/epoch, "
            f"{total_steps} total steps, batch size: {stage1_config.get('batch_size', 32)}"
        )
        num_training_steps = steps_per_epoch * num_epochs

        # Setup optimizer and scheduler
        optimizer, scheduler = self._setup_optimizer_and_scheduler(1, num_training_steps)

        # Prepare data loaders
        train_loader, val_loader = self.accelerator.prepare(
            self.train_loader_stage1, self.val_loader_stage1
        )

        # Training metrics
        metrics = {"train_loss": [], "val_loss": [], "learning_rates": []}

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Training phase
            train_loss = self._train_epoch_stage1(train_loader, optimizer, scheduler, stage1_config)
            metrics["train_loss"].append(train_loss)

            # Validation phase
            val_loss = self._validate_epoch_stage1(val_loader)
            metrics["val_loss"].append(val_loss)

            # Log metrics
            self._log_metrics({
                "epoch": epoch,
                "stage": 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

            # Save checkpoint
            if self.accelerator.is_main_process:
                self._save_checkpoint(epoch, optimizer, scheduler, val_loss, stage=1)

            # Early stopping check
            if self._check_early_stopping(val_loss, stage1_config):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.logger.info("Completed Stage 1 training")
        return metrics

    def train_stage2(self) -> Dict[str, float]:
        """Train stage 2: Preference optimization.

        Returns:
            Dictionary of training metrics.
        """
        if self.train_loader_stage2 is None:
            self.logger.warning("No stage 2 data loader provided, skipping stage 2")
            return {}

        self.logger.info("Starting Stage 2: Preference Optimization")
        self.current_stage = 2

        # Get stage 2 config
        stage2_config = self.config.get_stage2_config()
        num_epochs = stage2_config["num_epochs"]

        # Calculate training steps
        steps_per_epoch = len(self.train_loader_stage2) // stage2_config.get("gradient_accumulation_steps", 1)
        num_training_steps = steps_per_epoch * num_epochs

        # Setup optimizer and scheduler
        optimizer, scheduler = self._setup_optimizer_and_scheduler(2, num_training_steps)

        # Prepare data loaders
        train_loader, val_loader = self.accelerator.prepare(
            self.train_loader_stage2, self.val_loader_stage2 or self.train_loader_stage2
        )

        # Training metrics
        metrics = {"train_loss": [], "val_loss": [], "learning_rates": []}

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Training phase
            train_loss = self._train_epoch_stage2(train_loader, optimizer, scheduler, stage2_config)
            metrics["train_loss"].append(train_loss)

            # Validation phase
            val_loss = self._validate_epoch_stage2(val_loader)
            metrics["val_loss"].append(val_loss)

            # Log metrics
            self._log_metrics({
                "epoch": epoch,
                "stage": 2,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

            # Save checkpoint
            if self.accelerator.is_main_process:
                self._save_checkpoint(epoch, optimizer, scheduler, val_loss, stage=2)

            # Early stopping check
            if self._check_early_stopping(val_loss, stage2_config):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.logger.info("Completed Stage 2 training")
        return metrics

    def _train_epoch_stage1(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        config: Dict[str, Any],
    ) -> float:
        """Train one epoch for stage 1.

        Args:
            train_loader: Training data loader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            config: Stage 1 configuration.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Stage 1 Epoch {self.epoch}",
            disable=not self.accelerator.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(
                    images=batch["image"],
                    caption_ids=batch["caption_ids"],
                    caption_mask=batch["caption_mask"],
                    mode="contrastive",
                )

                # Compute contrastive loss
                loss = self.contrastive_loss(
                    outputs["image_embeddings"],
                    outputs["text_embeddings"],
                )

                # Backward pass
                self.accelerator.backward(loss)

                # Gradient clipping
                if config.get("max_grad_norm"):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        config["max_grad_norm"],
                    )

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

                # Log step metrics
                if self.global_step % config.get("logging_steps", 100) == 0:
                    self._log_metrics({
                        "step_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step,
                    })

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _train_epoch_stage2(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        config: Dict[str, Any],
    ) -> float:
        """Train one epoch for stage 2.

        Args:
            train_loader: Training data loader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            config: Stage 2 configuration.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Stage 2 Epoch {self.epoch}",
            disable=not self.accelerator.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Forward pass for preferred captions
                preferred_outputs = self.model(
                    images=batch["image"],
                    caption_ids=batch["preferred_ids"],
                    caption_mask=batch["preferred_mask"],
                    labels=batch["preferred_ids"],
                    mode="generation",
                )

                # Forward pass for rejected captions
                rejected_outputs = self.model(
                    images=batch["image"],
                    caption_ids=batch["rejected_ids"],
                    caption_mask=batch["rejected_mask"],
                    labels=batch["rejected_ids"],
                    mode="generation",
                )

                # Compute preference loss
                loss = self.preference_loss(
                    preferred_outputs["logits"],
                    rejected_outputs["logits"],
                    batch["preferred_ids"],
                    batch["rejected_ids"],
                    batch["preferred_mask"],
                    batch["rejected_mask"],
                )

                # Backward pass
                self.accelerator.backward(loss)

                # Gradient clipping
                if config.get("max_grad_norm"):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        config["max_grad_norm"],
                    )

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

                # Log step metrics
                if self.global_step % config.get("logging_steps", 100) == 0:
                    self._log_metrics({
                        "step_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step,
                    })

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch_stage1(self, val_loader: DataLoader) -> float:
        """Validate one epoch for stage 1.

        Args:
            val_loader: Validation data loader.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc="Validation",
                disable=not self.accelerator.is_main_process,
            ):
                # Forward pass
                outputs = self.model(
                    images=batch["image"],
                    caption_ids=batch["caption_ids"],
                    caption_mask=batch["caption_mask"],
                    mode="contrastive",
                )

                # Compute contrastive loss
                loss = self.contrastive_loss(
                    outputs["image_embeddings"],
                    outputs["text_embeddings"],
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch_stage2(self, val_loader: DataLoader) -> float:
        """Validate one epoch for stage 2.

        Args:
            val_loader: Validation data loader.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc="Validation",
                disable=not self.accelerator.is_main_process,
            ):
                # Forward pass for preferred captions
                preferred_outputs = self.model(
                    images=batch["image"],
                    caption_ids=batch["preferred_ids"],
                    caption_mask=batch["preferred_mask"],
                    labels=batch["preferred_ids"],
                    mode="generation",
                )

                # Forward pass for rejected captions
                rejected_outputs = self.model(
                    images=batch["image"],
                    caption_ids=batch["rejected_ids"],
                    caption_mask=batch["rejected_mask"],
                    labels=batch["rejected_ids"],
                    mode="generation",
                )

                # Compute preference loss
                loss = self.preference_loss(
                    preferred_outputs["logits"],
                    rejected_outputs["logits"],
                    batch["preferred_ids"],
                    batch["rejected_ids"],
                    batch["preferred_mask"],
                    batch["rejected_mask"],
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to MLflow and Wandb.

        Args:
            metrics: Dictionary of metrics to log.
        """
        if self.accelerator.is_main_process:
            # MLflow logging
            if mlflow.active_run():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value, step=self.global_step)

            # Wandb logging
            if wandb.run:
                wandb.log(metrics, step=self.global_step)

            # Console logging
            if "epoch" in metrics:
                self.logger.info(
                    f"Epoch {metrics['epoch']}, Stage {metrics.get('stage', self.current_stage)}: "
                    f"Train Loss: {metrics.get('train_loss', 'N/A'):.4f}, "
                    f"Val Loss: {metrics.get('val_loss', 'N/A'):.4f}, "
                    f"LR: {metrics.get('learning_rate', 'N/A'):.2e}"
                )

    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: optim.Optimizer,
        scheduler: LRScheduler,
        val_loss: float,
        stage: int,
    ) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch.
            optimizer: Optimizer state.
            scheduler: Scheduler state.
            val_loss: Validation loss.
            stage: Training stage.
        """
        if not self.accelerator.is_main_process:
            return

        # Create checkpoint
        checkpoint = {
            "epoch": epoch,
            "stage": stage,
            "global_step": self.global_step,
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config.config,
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_stage{stage}_epoch{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_checkpoint_path = self.checkpoint_dir / f"best_model_stage{stage}.pt"
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f"Saved best model with val_loss: {val_loss:.4f}")

        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _check_early_stopping(self, val_loss: float, config: Dict[str, Any]) -> bool:
        """Check early stopping criteria.

        Args:
            val_loss: Current validation loss.
            config: Stage configuration.

        Returns:
            True if should stop early, False otherwise.
        """
        patience = config.get("early_stopping_patience", 3)

        if val_loss < self.best_val_loss:
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                return True
            return False

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load training state
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.current_stage = checkpoint["stage"]
        self.best_val_loss = checkpoint.get("val_loss", float("inf"))

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def train(self) -> Dict[str, Any]:
        """Run full training pipeline.

        Returns:
            Dictionary containing training results and metrics.
        """
        try:
            # Stage 1: Contrastive learning
            stage1_metrics = self.train_stage1()

            # Stage 2: Preference optimization
            stage2_metrics = self.train_stage2()

            # Combine metrics
            training_results = {
                "stage1_metrics": stage1_metrics,
                "stage2_metrics": stage2_metrics,
                "best_val_loss": self.best_val_loss,
                "total_steps": self.global_step,
            }

            # Log final results
            if self.accelerator.is_main_process:
                self._log_metrics({
                    "final_best_val_loss": self.best_val_loss,
                    "total_training_steps": self.global_step,
                })

            self.logger.info("Training completed successfully")
            return training_results

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

        finally:
            # Cleanup
            if mlflow.active_run():
                mlflow.end_run()
            if wandb.run:
                wandb.finish()

    def __del__(self) -> None:
        """Cleanup when trainer is destroyed."""
        try:
            if mlflow.active_run():
                mlflow.end_run()
            if wandb.run:
                wandb.finish()
        except Exception:
            pass