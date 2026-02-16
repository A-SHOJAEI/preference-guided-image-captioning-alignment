"""Data loading utilities for Conceptual Captions and UltraFeedback datasets."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

from .preprocessing import ImageProcessor, TextProcessor


class ConceptualCaptionsDataset(Dataset):
    """Dataset for Conceptual Captions 3M with image-caption pairs.

    Loads and processes image-caption pairs for contrastive learning phase.
    Supports various data formats including CSV, TSV, and JSON.
    """

    def __init__(
        self,
        data_path: str,
        image_processor: ImageProcessor,
        text_processor: TextProcessor,
        split: str = "train",
        max_samples: Optional[int] = None,
        cache_images: bool = False,
    ) -> None:
        """Initialize Conceptual Captions dataset.

        Args:
            data_path: Path to dataset directory or file.
            image_processor: Image preprocessing pipeline.
            text_processor: Text preprocessing pipeline.
            split: Dataset split ('train', 'val', 'test').
            max_samples: Maximum number of samples to load (for debugging).
            cache_images: Whether to cache processed images in memory.
        """
        self.data_path = Path(data_path)
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.split = split
        self.max_samples = max_samples
        self.cache_images = cache_images
        self.logger = logging.getLogger(__name__)

        self.image_cache: Dict[str, torch.Tensor] = {}
        self.data = self._load_data()

        self.logger.info(
            f"Loaded {len(self.data)} samples for {split} split from {data_path}"
        )

    def _load_data(self) -> List[Dict[str, str]]:
        """Load dataset from file.

        Returns:
            List of dictionaries containing image_path and caption.

        Raises:
            FileNotFoundError: If data file doesn't exist.
            ValueError: If data format is unsupported.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        if self.data_path.is_file():
            return self._load_data_file()
        else:
            return self._load_data_directory()

    def _load_data_file(self) -> List[Dict[str, str]]:
        """Load data from a single file."""
        file_ext = self.data_path.suffix.lower()

        try:
            if file_ext in [".csv", ".tsv"]:
                delimiter = "\t" if file_ext == ".tsv" else ","
                df = pd.read_csv(self.data_path, delimiter=delimiter)

                # Expect columns: image_path, caption
                if "image_path" not in df.columns or "caption" not in df.columns:
                    # Try alternative column names
                    image_col = None
                    caption_col = None

                    for col in df.columns:
                        if col.lower() in ["image", "image_path", "image_url", "url"]:
                            image_col = col
                        elif col.lower() in ["caption", "text", "description"]:
                            caption_col = col

                    if image_col is None or caption_col is None:
                        raise ValueError(
                            f"Could not find image and caption columns in {df.columns}"
                        )

                    df = df.rename(columns={image_col: "image_path", caption_col: "caption"})

                data = df[["image_path", "caption"]].to_dict("records")

            elif file_ext == ".json":
                with open(self.data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Ensure consistent format
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]

                # Normalize keys
                normalized_data = []
                for item in data:
                    normalized_item = {}
                    for key, value in item.items():
                        if key.lower() in ["image", "image_path", "image_url", "url"]:
                            normalized_item["image_path"] = value
                        elif key.lower() in ["caption", "text", "description"]:
                            normalized_item["caption"] = value

                    if "image_path" in normalized_item and "caption" in normalized_item:
                        normalized_data.append(normalized_item)

                data = normalized_data

            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            # Filter out invalid entries
            valid_data = []
            for item in data:
                if (
                    isinstance(item.get("image_path"), str) and
                    isinstance(item.get("caption"), str) and
                    len(item["caption"].strip()) > 0
                ):
                    # Convert relative paths to absolute
                    image_path = item["image_path"]
                    if not os.path.isabs(image_path):
                        image_path = os.path.join(self.data_path.parent, image_path)

                    item["image_path"] = image_path
                    valid_data.append(item)

            if self.max_samples:
                valid_data = valid_data[:self.max_samples]

            return valid_data

        except Exception as e:
            self.logger.error(f"Error loading data from {self.data_path}: {e}")
            raise

    def _load_data_directory(self) -> List[Dict[str, str]]:
        """Load data from directory structure."""
        # Look for annotation files
        annotation_files = list(self.data_path.glob("*.json")) + \
                          list(self.data_path.glob("*.csv")) + \
                          list(self.data_path.glob("*.tsv"))

        if annotation_files:
            # Use the first annotation file found
            annotation_file = annotation_files[0]
            self.logger.info(f"Using annotation file: {annotation_file}")
            temp_path = self.data_path
            self.data_path = annotation_file
            data = self._load_data_file()
            self.data_path = temp_path
            return data

        # If no annotation files, try to pair images with text files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        text_extensions = {".txt", ".caption"}

        data = []
        for image_file in self.data_path.iterdir():
            if image_file.suffix.lower() in image_extensions:
                # Look for corresponding text file
                text_file = None
                for ext in text_extensions:
                    candidate = image_file.with_suffix(ext)
                    if candidate.exists():
                        text_file = candidate
                        break

                if text_file:
                    try:
                        with open(text_file, "r", encoding="utf-8") as f:
                            caption = f.read().strip()

                        if caption:
                            data.append({
                                "image_path": str(image_file),
                                "caption": caption
                            })
                    except Exception as e:
                        self.logger.warning(f"Error reading {text_file}: {e}")

        if not data:
            raise ValueError(f"No valid image-caption pairs found in {self.data_path}")

        if self.max_samples:
            data = data[:self.max_samples]

        return data

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing processed image and caption tensors.
        """
        item = self.data[idx]
        image_path = item["image_path"]
        caption = item["caption"]

        # Load and process image
        if self.cache_images and image_path in self.image_cache:
            image_tensor = self.image_cache[image_path]
        else:
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.image_processor.process_image(
                    image, training=(self.split == "train")
                )

                if self.cache_images:
                    self.image_cache[image_path] = image_tensor

            except Exception as e:
                self.logger.warning(f"Error loading image {image_path}: {e}")
                # Return zero tensor as fallback
                image_tensor = torch.zeros(
                    3, self.image_processor.image_size, self.image_processor.image_size
                )

        # Process caption
        caption_encoding = self.text_processor.encode_caption(caption)

        return {
            "image": image_tensor,
            "caption_ids": caption_encoding["input_ids"],
            "caption_mask": caption_encoding["attention_mask"],
            "raw_caption": caption,
            "image_path": image_path,
        }

    def get_sample_by_path(self, image_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get sample by image path.

        Args:
            image_path: Path to image file.

        Returns:
            Sample dictionary if found, None otherwise.
        """
        for idx, item in enumerate(self.data):
            if item["image_path"] == image_path:
                return self[idx]
        return None


class UltraFeedbackDataset(Dataset):
    """Dataset for UltraFeedback preference learning.

    Loads preference pairs for training preference-aligned caption generation.
    Each sample contains an image, preferred caption, and rejected caption.
    """

    def __init__(
        self,
        data_path: str,
        image_processor: ImageProcessor,
        text_processor: TextProcessor,
        split: str = "train",
        max_samples: Optional[int] = None,
        preference_threshold: float = 0.6,
    ) -> None:
        """Initialize UltraFeedback dataset.

        Args:
            data_path: Path to preference dataset.
            image_processor: Image preprocessing pipeline.
            text_processor: Text preprocessing pipeline.
            split: Dataset split ('train', 'val', 'test').
            max_samples: Maximum number of samples to load.
            preference_threshold: Minimum preference score difference for inclusion.
        """
        self.data_path = Path(data_path)
        self.image_processor = image_processor
        self.text_processor = text_processor
        self.split = split
        self.max_samples = max_samples
        self.preference_threshold = preference_threshold
        self.logger = logging.getLogger(__name__)

        self.data = self._load_preference_data()

        self.logger.info(
            f"Loaded {len(self.data)} preference pairs for {split} split"
        )

    def _load_preference_data(self) -> List[Dict[str, Any]]:
        """Load preference data from file.

        Returns:
            List of preference pairs with image, preferred, and rejected captions.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        try:
            if self.data_path.suffix.lower() == ".json":
                with open(self.data_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            else:
                # Assume CSV format
                df = pd.read_csv(self.data_path)
                raw_data = df.to_dict("records")

            # Process preference data
            preference_pairs = []

            for item in raw_data:
                # Handle different data formats
                if "conversations" in item:
                    # UltraFeedback format
                    preference_pairs.extend(self._process_ultrafeedback_item(item))
                elif all(k in item for k in ["image_path", "preferred_caption", "rejected_caption"]):
                    # Direct preference format
                    preference_pairs.append({
                        "image_path": item["image_path"],
                        "preferred_caption": item["preferred_caption"],
                        "rejected_caption": item["rejected_caption"],
                        "preference_score": item.get("preference_score", 1.0),
                    })
                elif all(k in item for k in ["image_path", "captions", "scores"]):
                    # Multiple captions with scores
                    preference_pairs.extend(self._process_scored_captions(item))

            # Resolve relative image paths (relative to current working directory)
            for pair in preference_pairs:
                if "image_path" in pair and not os.path.isabs(pair["image_path"]):
                    pair["image_path"] = str(Path(pair["image_path"]).resolve())

            # Filter by preference threshold
            filtered_pairs = [
                pair for pair in preference_pairs
                if pair.get("preference_score", 1.0) >= self.preference_threshold
            ]

            if self.max_samples:
                filtered_pairs = filtered_pairs[:self.max_samples]

            return filtered_pairs

        except Exception as e:
            self.logger.error(f"Error loading preference data: {e}")
            raise

    def _process_ultrafeedback_item(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process UltraFeedback format item.

        Args:
            item: Raw UltraFeedback item.

        Returns:
            List of preference pairs extracted from the item.
        """
        pairs = []

        if "image_path" not in item:
            return pairs

        conversations = item.get("conversations", [])

        # Extract caption pairs with different quality scores
        captions_with_scores = []
        for conv in conversations:
            if "response" in conv and "score" in conv:
                captions_with_scores.append({
                    "caption": conv["response"],
                    "score": conv["score"],
                })

        # Create preference pairs from score differences
        captions_with_scores.sort(key=lambda x: x["score"], reverse=True)

        for i in range(len(captions_with_scores) - 1):
            preferred = captions_with_scores[i]
            rejected = captions_with_scores[i + 1]

            score_diff = preferred["score"] - rejected["score"]
            if score_diff >= self.preference_threshold:
                pairs.append({
                    "image_path": item["image_path"],
                    "preferred_caption": preferred["caption"],
                    "rejected_caption": rejected["caption"],
                    "preference_score": score_diff,
                })

        return pairs

    def _process_scored_captions(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process item with multiple captions and scores.

        Args:
            item: Item with captions and scores lists.

        Returns:
            List of preference pairs.
        """
        pairs = []

        captions = item["captions"]
        scores = item["scores"]

        if len(captions) != len(scores):
            self.logger.warning("Mismatch between captions and scores length")
            return pairs

        caption_score_pairs = list(zip(captions, scores))
        caption_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Create preference pairs
        for i in range(len(caption_score_pairs) - 1):
            preferred_caption, preferred_score = caption_score_pairs[i]
            rejected_caption, rejected_score = caption_score_pairs[i + 1]

            score_diff = preferred_score - rejected_score
            if score_diff >= self.preference_threshold:
                pairs.append({
                    "image_path": item["image_path"],
                    "preferred_caption": preferred_caption,
                    "rejected_caption": rejected_caption,
                    "preference_score": score_diff,
                })

        return pairs

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing image and preference pair tensors.
        """
        item = self.data[idx]
        image_path = item["image_path"]
        preferred_caption = item["preferred_caption"]
        rejected_caption = item["rejected_caption"]

        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_processor.process_image(
                image, training=(self.split == "train")
            )
        except Exception as e:
            self.logger.warning(f"Error loading image {image_path}: {e}")
            image_tensor = torch.zeros(
                3, self.image_processor.image_size, self.image_processor.image_size
            )

        # Process captions
        preferred_encoding = self.text_processor.encode_caption(preferred_caption)
        rejected_encoding = self.text_processor.encode_caption(rejected_caption)

        return {
            "image": image_tensor,
            "preferred_ids": preferred_encoding["input_ids"],
            "preferred_mask": preferred_encoding["attention_mask"],
            "rejected_ids": rejected_encoding["input_ids"],
            "rejected_mask": rejected_encoding["attention_mask"],
            "preference_score": torch.tensor(item["preference_score"], dtype=torch.float32),
            "raw_preferred": preferred_caption,
            "raw_rejected": rejected_caption,
            "image_path": image_path,
        }


def create_dataloaders(
    dataset_class: type,
    data_path: str,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.

    Args:
        dataset_class: Dataset class to instantiate.
        data_path: Path to dataset.
        image_processor: Image preprocessing pipeline.
        text_processor: Text preprocessing pipeline.
        batch_size: Batch size for data loaders.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
        seed: Random seed for reproducible splits.
        **dataset_kwargs: Additional arguments for dataset initialization.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Set random seed for reproducible splits
    random.seed(seed)
    torch.manual_seed(seed)

    # Create full dataset
    full_dataset = dataset_class(
        data_path=data_path,
        image_processor=image_processor,
        text_processor=text_processor,
        split="full",
        **dataset_kwargs,
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Update split information for data augmentation
    train_dataset.dataset.split = "train"
    val_dataset.dataset.split = "val"
    test_dataset.dataset.split = "test"

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logging.getLogger(__name__).info(
        f"Created data loaders: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader