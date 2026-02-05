"""Image and text preprocessing utilities."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms


class ImageProcessor:
    """Image preprocessing for vision-language models.

    Handles image loading, resizing, normalization, and augmentation
    for both training and inference phases.
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        augment: bool = True,
    ) -> None:
        """Initialize image processor.

        Args:
            image_size: Target image size for resizing.
            mean: Normalization mean values for RGB channels.
            std: Normalization standard deviation values for RGB channels.
            augment: Whether to apply data augmentation during training.
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.logger = logging.getLogger(__name__)

        self._setup_transforms()

    def _setup_transforms(self) -> None:
        """Setup image transformations for training and validation."""
        # Base transforms for both train and val
        base_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]

        # Training transforms with augmentation
        train_transforms = []
        if self.augment:
            train_transforms.extend([
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomRotation(degrees=5),
            ])
        else:
            train_transforms.append(transforms.Resize((self.image_size, self.image_size)))

        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.train_transform = transforms.Compose(train_transforms)
        self.val_transform = transforms.Compose(base_transforms)

    def process_image(
        self,
        image: Union[Image.Image, str],
        training: bool = True
    ) -> torch.Tensor:
        """Process a single image.

        Args:
            image: PIL Image or path to image file.
            training: Whether to apply training transforms.

        Returns:
            Processed image tensor of shape (C, H, W).

        Raises:
            ValueError: If image cannot be loaded or processed.
        """
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Expected PIL Image or string path, got {type(image)}")

            transform = self.train_transform if training else self.val_transform
            return transform(image)

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            raise ValueError(f"Failed to process image: {e}")

    def process_batch(
        self,
        images: List[Union[Image.Image, str]],
        training: bool = True
    ) -> torch.Tensor:
        """Process a batch of images.

        Args:
            images: List of PIL Images or image paths.
            training: Whether to apply training transforms.

        Returns:
            Batch of processed images tensor of shape (B, C, H, W).
        """
        processed_images = []
        for image in images:
            processed_images.append(self.process_image(image, training))

        return torch.stack(processed_images)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor for visualization.

        Args:
            tensor: Normalized image tensor.

        Returns:
            Denormalized tensor with values in [0, 1].
        """
        mean = torch.tensor(self.mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(self.std).view(-1, 1, 1).to(tensor.device)

        denorm = tensor * std + mean
        return torch.clamp(denorm, 0, 1)


class TextProcessor:
    """Text preprocessing for caption generation and preference learning.

    Handles tokenization, encoding, and special token management
    for both input captions and generated text.
    """

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> None:
        """Initialize text processor.

        Args:
            model_name: Name of the pre-trained tokenizer model.
            max_length: Maximum sequence length.
            padding: Padding strategy ('max_length' or 'longest').
            truncation: Whether to truncate sequences exceeding max_length.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.logger = logging.getLogger(__name__)

        self._setup_tokenizer()

    def _setup_tokenizer(self) -> None:
        """Setup tokenizer with special tokens."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Add special tokens if not present
            special_tokens = {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "sep_token": "[SEP]",
            }

            # Only add tokens that don't exist
            tokens_to_add = {}
            for token_type, token in special_tokens.items():
                if getattr(self.tokenizer, token_type) is None:
                    tokens_to_add[token_type] = token

            if tokens_to_add:
                self.tokenizer.add_special_tokens(tokens_to_add)

            self.logger.info(f"Initialized tokenizer for {self.model_name}")
            self.logger.info(f"Vocabulary size: {len(self.tokenizer)}")

        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            raise

    def encode_caption(
        self,
        caption: str,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Encode a single caption.

        Args:
            caption: Input caption text.
            add_special_tokens: Whether to add BOS/EOS tokens.
            return_attention_mask: Whether to return attention mask.

        Returns:
            Dictionary containing input_ids and optionally attention_mask.
        """
        try:
            encoding = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                add_special_tokens=add_special_tokens,
                return_tensors="pt",
                return_attention_mask=return_attention_mask,
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding.get("attention_mask", torch.ones_like(
                    encoding["input_ids"]
                )).squeeze(0),
            }

        except Exception as e:
            self.logger.error(f"Error encoding caption '{caption}': {e}")
            raise ValueError(f"Failed to encode caption: {e}")

    def encode_batch(
        self,
        captions: List[str],
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of captions.

        Args:
            captions: List of caption texts.
            add_special_tokens: Whether to add BOS/EOS tokens.
            return_attention_mask: Whether to return attention mask.

        Returns:
            Dictionary containing batched input_ids and attention_mask.
        """
        try:
            encoding = self.tokenizer(
                captions,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                add_special_tokens=add_special_tokens,
                return_tensors="pt",
                return_attention_mask=return_attention_mask,
            )

            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding.get("attention_mask", torch.ones_like(
                    encoding["input_ids"]
                )),
            }

        except Exception as e:
            self.logger.error(f"Error encoding batch of {len(captions)} captions: {e}")
            raise ValueError(f"Failed to encode caption batch: {e}")

    def decode_caption(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: Token IDs tensor.
            skip_special_tokens: Whether to skip special tokens in output.
            clean_up_tokenization_spaces: Whether to clean up tokenization artifacts.

        Returns:
            Decoded caption text.
        """
        try:
            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze()

            caption = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

            return caption.strip()

        except Exception as e:
            self.logger.error(f"Error decoding tokens: {e}")
            return ""

    def decode_batch(
        self,
        token_ids_batch: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """Decode a batch of token IDs back to text.

        Args:
            token_ids_batch: Batch of token IDs tensor.
            skip_special_tokens: Whether to skip special tokens in output.
            clean_up_tokenization_spaces: Whether to clean up tokenization artifacts.

        Returns:
            List of decoded caption texts.
        """
        captions = []
        for token_ids in token_ids_batch:
            caption = self.decode_caption(
                token_ids, skip_special_tokens, clean_up_tokenization_spaces
            )
            captions.append(caption)

        return captions

    def prepare_for_generation(
        self,
        prompt: str = "",
        max_new_tokens: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """Prepare input for text generation.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Dictionary containing input_ids and attention_mask for generation.
        """
        if prompt:
            encoding = self.encode_caption(prompt, add_special_tokens=False)
        else:
            # Start with BOS token for unconditional generation
            bos_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
            encoding = {
                "input_ids": torch.tensor([bos_id]),
                "attention_mask": torch.tensor([1]),
            }

        return encoding

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id