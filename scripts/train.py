#!/usr/bin/env python3
"""Training script for preference-guided image captioning alignment."""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preference_guided_image_captioning_alignment.data.loader import (
    ConceptualCaptionsDataset,
    UltraFeedbackDataset,
    create_dataloaders,
)
from preference_guided_image_captioning_alignment.data.preprocessing import (
    ImageProcessor,
    TextProcessor,
)
from preference_guided_image_captioning_alignment.models.model import (
    PreferenceGuidedCaptioningModel,
)
from preference_guided_image_captioning_alignment.training.trainer import (
    PreferenceGuidedTrainer,
)
from preference_guided_image_captioning_alignment.utils.config import Config


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log"),
        ],
    )


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config: Config) -> PreferenceGuidedCaptioningModel:
    """Create and initialize the model.

    Args:
        config: Configuration object.

    Returns:
        Initialized model.
    """
    model_config = config.get_model_config()

    model = PreferenceGuidedCaptioningModel(
        vision_model=model_config["vision_model"],
        text_model=model_config["text_model"],
        projection_dim=model_config["projection_dim"],
        temperature=model_config.get("temperature", 0.07),
        dropout=model_config.get("dropout", 0.1),
        freeze_vision_backbone=model_config.get("freeze_vision_backbone", False),
        freeze_text_backbone=model_config.get("freeze_text_backbone", False),
        lora_config=model_config.get("lora_config"),
    )

    return model


def create_data_processors(config: Config) -> tuple:
    """Create image and text processors.

    Args:
        config: Configuration object.

    Returns:
        Tuple of (image_processor, text_processor).
    """
    data_config = config.get_data_config()
    model_config = config.get_model_config()

    image_processor = ImageProcessor(
        image_size=data_config["image_size"],
        augment=True,  # Enable augmentation for training
    )

    text_processor = TextProcessor(
        model_name=model_config["text_model"],
        max_length=data_config["max_caption_length"],
        padding="max_length",
        truncation=True,
    )

    return image_processor, text_processor


def create_datasets_and_loaders(
    config: Config,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
) -> tuple:
    """Create datasets and data loaders.

    Args:
        config: Configuration object.
        image_processor: Image preprocessing pipeline.
        text_processor: Text preprocessing pipeline.

    Returns:
        Tuple of data loaders for stage 1 and stage 2.
    """
    data_config = config.get_data_config()
    training_config = config.get_training_config()

    # Stage 1: Conceptual Captions data loaders
    if not Path(data_config["conceptual_captions_path"]).exists():
        logging.warning(
            f"Conceptual Captions path not found: {data_config['conceptual_captions_path']}. "
            "Creating dummy data loaders."
        )
        # Create dummy datasets for demonstration
        train_loader_stage1 = create_dummy_conceptual_dataloader(
            image_processor, text_processor, training_config["stage1"]["batch_size"]
        )
        val_loader_stage1 = create_dummy_conceptual_dataloader(
            image_processor, text_processor, training_config["stage1"]["batch_size"]
        )
    else:
        train_loader_stage1, val_loader_stage1, _ = create_dataloaders(
            dataset_class=ConceptualCaptionsDataset,
            data_path=data_config["conceptual_captions_path"],
            image_processor=image_processor,
            text_processor=text_processor,
            batch_size=training_config["stage1"]["batch_size"],
            train_split=data_config["train_split"],
            val_split=data_config["val_split"],
            test_split=data_config["test_split"],
            num_workers=data_config["num_workers"],
            pin_memory=data_config.get("pin_memory", True),
            seed=training_config["seed"],
        )

    # Stage 2: UltraFeedback data loaders
    train_loader_stage2 = None
    val_loader_stage2 = None

    if Path(data_config["ultrafeedback_path"]).exists():
        train_loader_stage2, val_loader_stage2, _ = create_dataloaders(
            dataset_class=UltraFeedbackDataset,
            data_path=data_config["ultrafeedback_path"],
            image_processor=image_processor,
            text_processor=text_processor,
            batch_size=training_config["stage2"]["batch_size"],
            train_split=data_config["train_split"],
            val_split=data_config["val_split"],
            test_split=data_config["test_split"],
            num_workers=data_config["num_workers"],
            pin_memory=data_config.get("pin_memory", True),
            seed=training_config["seed"],
        )
    else:
        logging.warning(
            f"UltraFeedback path not found: {data_config['ultrafeedback_path']}. "
            "Skipping stage 2 training."
        )

    return train_loader_stage1, val_loader_stage1, train_loader_stage2, val_loader_stage2


def create_dummy_conceptual_dataloader(
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Create a dummy data loader for demonstration purposes.

    Args:
        image_processor: Image processor.
        text_processor: Text processor.
        batch_size: Batch size.

    Returns:
        Dummy data loader.
    """
    from torch.utils.data import Dataset, DataLoader

    class DummyDataset(Dataset):
        def __init__(self, size: int = 100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Create dummy image
            image = torch.randn(3, 224, 224)

            # Create dummy caption
            captions = [
                "A cat sitting on a chair",
                "A dog running in the park",
                "A beautiful sunset over the ocean",
                "People walking on a busy street",
                "A red car parked outside",
            ]
            caption = captions[idx % len(captions)]

            # Process caption
            caption_encoding = text_processor.encode_caption(caption)

            return {
                "image": image,
                "caption_ids": caption_encoding["input_ids"],
                "caption_mask": caption_encoding["attention_mask"],
                "raw_caption": caption,
                "image_path": f"dummy_image_{idx}.jpg",
            }

    dataset = DummyDataset()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing for dummy data
        pin_memory=False,
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train preference-guided captioning model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="Specific stage to run (1 or 2). If not specified, runs both stages."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without actual training"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting preference-guided image captioning training")
    logger.info(f"Configuration file: {args.config}")

    try:
        # Load configuration
        config = Config(args.config)

        # Override output directory if specified
        if args.output_dir:
            config.set("paths.output_dir", args.output_dir)

        # Set random seeds
        seed = config.get("training.seed", 42)
        set_random_seeds(seed)
        logger.info(f"Set random seed to {seed}")

        # Initialize accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=config.get("training.gradient_accumulation_steps", 1),
            mixed_precision=config.get("hardware.mixed_precision", "fp16"),
            log_with="wandb" if config.get("logging.wandb_project") else None,
        )

        logger.info(f"Using device: {accelerator.device}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"Distributed training: {accelerator.num_processes > 1}")

        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Create data processors
        logger.info("Creating data processors...")
        image_processor, text_processor = create_data_processors(config)

        # Create datasets and data loaders
        logger.info("Creating datasets and data loaders...")
        train_loader_stage1, val_loader_stage1, train_loader_stage2, val_loader_stage2 = (
            create_datasets_and_loaders(config, image_processor, text_processor)
        )

        logger.info(f"Stage 1 - Train batches: {len(train_loader_stage1)}, Val batches: {len(val_loader_stage1)}")
        if train_loader_stage2:
            logger.info(f"Stage 2 - Train batches: {len(train_loader_stage2)}, Val batches: {len(val_loader_stage2)}")

        # Dry run check
        if args.dry_run:
            logger.info("Dry run completed successfully. Model and data loaders created.")
            return

        # Create trainer
        logger.info("Creating trainer...")
        trainer = PreferenceGuidedTrainer(
            model=model,
            config=config,
            train_loader_stage1=train_loader_stage1,
            val_loader_stage1=val_loader_stage1,
            train_loader_stage2=train_loader_stage2,
            val_loader_stage2=val_loader_stage2,
            accelerator=accelerator,
        )

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Run training
        if args.stage == 1:
            logger.info("Running Stage 1 training only")
            results = {"stage1_metrics": trainer.train_stage1()}
        elif args.stage == 2:
            logger.info("Running Stage 2 training only")
            if train_loader_stage2 is None:
                logger.error("Stage 2 data not available. Cannot run stage 2 training.")
                return
            results = {"stage2_metrics": trainer.train_stage2()}
        else:
            logger.info("Running full training pipeline (Stage 1 + Stage 2)")
            results = trainer.train()

        # Log final results
        logger.info("Training completed!")
        logger.info("Final Results:")
        for key, value in results.items():
            if isinstance(value, dict) and "train_loss" in value:
                final_train_loss = value["train_loss"][-1] if value["train_loss"] else "N/A"
                final_val_loss = value["val_loss"][-1] if value["val_loss"] else "N/A"
                logger.info(f"  {key}: Train Loss = {final_train_loss}, Val Loss = {final_val_loss}")
            else:
                logger.info(f"  {key}: {value}")

        # Save final configuration
        output_dir = Path(config.get("paths.output_dir", "./outputs"))
        config.save(str(output_dir / "final_config.yaml"))

        logger.info(f"Training artifacts saved to: {output_dir}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("Training script completed")


if __name__ == "__main__":
    main()