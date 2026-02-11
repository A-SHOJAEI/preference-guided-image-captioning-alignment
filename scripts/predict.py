#!/usr/bin/env python3
"""Prediction script for preference-guided image captioning model.

This script loads a trained model and generates captions for input images.
Supports both single image and batch prediction modes.

Usage:
    # Single image prediction
    python scripts/predict.py --image path/to/image.jpg --model-path outputs/best_model.pt

    # Batch prediction from directory
    python scripts/predict.py --image-dir path/to/images/ --model-path outputs/best_model.pt

    # Use default demo mode (generates sample predictions)
    python scripts/predict.py --demo
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preference_guided_image_captioning_alignment.data.preprocessing import (
    ImageProcessor,
    TextProcessor,
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


class CaptionPredictor:
    """Caption prediction interface for trained models."""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize predictor with trained model.

        Args:
            model_path: Path to saved model checkpoint (.pt or .pth file)
            config_path: Path to config file (defaults to configs/default.yaml)
            device: Device to run inference on (cuda/cpu, auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        config_obj = Config(str(config_path))
        self.config = config_obj.config

        # Initialize processors
        self.image_processor = ImageProcessor(
            image_size=self.config["data"]["image_size"]
        )
        self.text_processor = TextProcessor(
            model_name=self.config["model"]["text_model"],
            max_length=self.config["data"]["max_caption_length"],
        )

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def _load_model(self, model_path: str) -> PreferenceGuidedCaptioningModel:
        """Load trained model from checkpoint.

        Args:
            model_path: Path to model checkpoint

        Returns:
            Loaded model instance
        """
        # Initialize model architecture
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

    def predict_single(
        self,
        image_path: str,
        num_beams: int = 4,
        max_length: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> Dict[str, any]:
        """Generate caption for a single image.

        Args:
            image_path: Path to input image
            num_beams: Number of beams for beam search
            max_length: Maximum caption length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Dictionary with caption and confidence scores
        """
        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return {"error": str(e)}

        image_tensor = self.image_processor.process_image(image).unsqueeze(0).to(self.device)

        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate_captions(
                image_tensor,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
            )

            # Decode generated caption
            caption = self.text_processor.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            # Get confidence score (average log probability)
            outputs = self.model(image_tensor, mode="inference")
            confidence = torch.softmax(outputs["logits"], dim=-1).max().item()

        return {
            "image_path": image_path,
            "caption": caption,
            "confidence": confidence,
            "model_path": self.model.__class__.__name__,
        }

    def predict_batch(
        self,
        image_dir: str,
        extensions: List[str] = None,
        **generation_kwargs,
    ) -> List[Dict[str, any]]:
        """Generate captions for all images in a directory.

        Args:
            image_dir: Directory containing images
            extensions: List of file extensions to process (default: jpg, jpeg, png)
            **generation_kwargs: Additional arguments for caption generation

        Returns:
            List of prediction results
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        image_dir = Path(image_dir)
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in extensions
        ]

        logger.info(f"Found {len(image_files)} images in {image_dir}")

        results = []
        for image_file in image_files:
            logger.info(f"Processing {image_file.name}")
            result = self.predict_single(str(image_file), **generation_kwargs)
            results.append(result)

        return results


def main():
    """Main prediction script entry point."""
    parser = argparse.ArgumentParser(
        description="Generate captions for images using trained model"
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
        default=None,
        help="Path to config file (default: configs/default.yaml)",
    )

    # Input arguments
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image file",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing images for batch prediction",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode with sample predictions",
    )

    # Generation arguments
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for beam search",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum caption length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for batch predictions",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.demo and args.image is None and args.image_dir is None:
        logger.error("Must specify --image, --image-dir, or --demo")
        parser.print_help()
        return 1

    # Initialize predictor
    predictor = CaptionPredictor(
        model_path=args.model_path,
        config_path=args.config,
    )

    # Generation kwargs
    gen_kwargs = {
        "num_beams": args.num_beams,
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    # Run prediction
    if args.demo:
        logger.info("Running in demo mode (model architecture loaded)")
        print("\n" + "=" * 80)
        print("DEMO MODE - Model Architecture Ready")
        print("=" * 80)
        print(f"Model: {predictor.model.__class__.__name__}")
        print(f"Device: {predictor.device}")
        print(f"Vision Encoder: {predictor.config['model']['vision_model']}")
        print(f"Text Encoder: {predictor.config['model']['text_model']}")
        print(f"Projection Dim: {predictor.config['model']['projection_dim']}")
        print("\nTo generate captions, provide --image or --image-dir")
        print("=" * 80)

    elif args.image:
        result = predictor.predict_single(args.image, **gen_kwargs)
        print("\n" + "=" * 80)
        print("PREDICTION RESULT")
        print("=" * 80)
        print(f"Image: {result.get('image_path', 'N/A')}")
        print(f"Caption: {result.get('caption', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0.0):.4f}")
        print("=" * 80)

    elif args.image_dir:
        results = predictor.predict_batch(args.image_dir, **gen_kwargs)

        # Print summary
        print("\n" + "=" * 80)
        print(f"BATCH PREDICTION RESULTS ({len(results)} images)")
        print("=" * 80)
        for result in results:
            print(f"\n{Path(result['image_path']).name}")
            print(f"  Caption: {result.get('caption', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 0.0):.4f}")
        print("=" * 80)

        # Save to file if specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
