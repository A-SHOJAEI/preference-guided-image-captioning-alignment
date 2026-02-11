#!/usr/bin/env python3
"""Standalone evaluation script for preference-guided image captioning model.

This script loads a trained model and evaluates it on test data using
comprehensive metrics including BLEU, ROUGE, CIDEr, METEOR, BERTScore, and CLIP-Score.

Usage:
    # Evaluate model on test set
    python scripts/evaluate.py --model-path outputs/best_model.pt --split test

    # Evaluate with custom config
    python scripts/evaluate.py --model-path outputs/best_model.pt --config configs/default.yaml

    # Save detailed results
    python scripts/evaluate.py --model-path outputs/best_model.pt --output results/evaluation.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preference_guided_image_captioning_alignment.data.loader import (
    create_dataloaders,
)
from preference_guided_image_captioning_alignment.evaluation.metrics import (
    CaptionMetrics,
)
from preference_guided_image_captioning_alignment.models.model import (
    PreferenceGuidedCaptioningModel,
)
from preference_guided_image_captioning_alignment.utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for preference-guided captioning models."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: Optional[str] = None,
    ):
        """Initialize evaluator.

        Args:
            model_path: Path to saved model checkpoint
            config_path: Path to configuration file
            device: Device to use (cuda/cpu, auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load configuration
        config_obj = Config(config_path)
        self.config = config_obj.config

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Initialize metrics
        self.metrics = CaptionMetrics(device=self.device)

        logger.info("Evaluator initialized successfully")

    def _load_model(self, model_path: str) -> PreferenceGuidedCaptioningModel:
        """Load model from checkpoint.

        Args:
            model_path: Path to model checkpoint

        Returns:
            Loaded model instance
        """
        # Initialize model
        model = PreferenceGuidedCaptioningModel(
            vision_model=self.config["model"]["vision_model"],
            text_model=self.config["model"]["text_model"],
            projection_dim=self.config["model"]["projection_dim"],
            temperature=self.config["model"]["temperature"],
            dropout=self.config["model"]["dropout"],
            freeze_vision_backbone=self.config["model"]["freeze_vision_backbone"],
            freeze_text_backbone=self.config["model"]["freeze_text_backbone"],
        )

        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
        else:
            logger.warning(f"Model path {model_path} not found, using untrained model")

        return model

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate model on dataset.

        Args:
            dataloader: DataLoader for evaluation data
            max_samples: Maximum number of samples to evaluate (None = all)

        Returns:
            Dictionary of metric scores
        """
        all_predictions = []
        all_references = []
        all_images = []

        logger.info("Generating captions for evaluation...")

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                if max_samples and i * dataloader.batch_size >= max_samples:
                    break

                # Move batch to device
                images = batch["image"].to(self.device)
                captions = batch.get("caption", None)

                # Generate captions
                generated_ids = self.model.generate_captions(
                    images,
                    max_length=self.config["evaluation"]["generate_config"]["max_length"],
                    num_beams=self.config["evaluation"]["generate_config"]["num_beams"],
                    temperature=self.config["evaluation"]["generate_config"]["temperature"],
                    do_sample=self.config["evaluation"]["generate_config"]["do_sample"],
                    top_p=self.config["evaluation"]["generate_config"]["top_p"],
                )

                # Decode generated captions
                for j, gen_ids in enumerate(generated_ids):
                    pred_caption = self.model.text_encoder.tokenizer.decode(
                        gen_ids, skip_special_tokens=True
                    )
                    all_predictions.append(pred_caption)

                    if captions is not None:
                        if isinstance(captions, list):
                            ref_caption = captions[j]
                        else:
                            ref_caption = self.model.text_encoder.tokenizer.decode(
                                captions[j], skip_special_tokens=True
                            )
                        all_references.append([ref_caption])

                    all_images.append(images[j].cpu())

        # Compute metrics
        logger.info("Computing evaluation metrics...")
        results = {}

        if all_references:
            # Compute caption quality metrics
            caption_metrics = self.metrics.compute_caption_metrics(
                predictions=all_predictions,
                references=all_references,
            )
            results.update(caption_metrics)

        # Compute semantic similarity metrics if images available
        if all_images and len(all_images) > 0:
            try:
                semantic_metrics = self.metrics.compute_semantic_metrics(
                    predictions=all_predictions,
                    images=torch.stack(all_images[:len(all_predictions)]),
                )
                results.update(semantic_metrics)
            except Exception as e:
                logger.warning(f"Failed to compute semantic metrics: {e}")

        # Add summary statistics
        results["num_samples"] = len(all_predictions)
        results["avg_caption_length"] = sum(len(p.split()) for p in all_predictions) / len(all_predictions)

        return results

    def evaluate_dataset(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate on a specific dataset split.

        Args:
            split: Dataset split to evaluate ('train', 'val', or 'test')
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Creating {split} dataloader...")

        # Create dataloaders
        dataloaders = create_dataloaders(self.config)

        # Select appropriate dataloader
        if split == "train":
            dataloader = dataloaders["stage1_train"]
        elif split == "val":
            dataloader = dataloaders["stage1_val"]
        elif split == "test":
            # Use validation data as test if no separate test set
            dataloader = dataloaders["stage1_val"]
            logger.warning("Using validation set as test set (no separate test split)")
        else:
            raise ValueError(f"Invalid split: {split}")

        # Run evaluation
        results = self.evaluate(dataloader, max_samples=max_samples)

        return results


def main():
    """Main evaluation script entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate preference-guided captioning model"
    )

    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )

    # Data arguments
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (default: results/evaluation_{split}.json)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to file",
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        config_path=args.config,
    )

    # Run evaluation
    logger.info(f"Evaluating on {args.split} split...")
    results = evaluator.evaluate_dataset(
        split=args.split,
        max_samples=args.max_samples,
    )

    # Print results
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS ({args.split} split)")
    print("=" * 80)
    for metric, value in sorted(results.items()):
        if isinstance(value, (int, float)):
            print(f"{metric:25s}: {value:.4f}")
        else:
            print(f"{metric:25s}: {value}")
    print("=" * 80)

    # Save results to file
    if args.output is None:
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / f"evaluation_{args.split}.json"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
