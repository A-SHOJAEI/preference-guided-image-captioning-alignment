"""Preference-guided multimodal captioning model implementation.

This module contains the core model components for preference-guided image captioning
alignment including vision encoder, text encoder, caption decoder, and complete
multimodal model with contrastive and preference losses.

Classes:
    VisionEncoder: CLIP-based image encoder for visual feature extraction.
    TextEncoder: GPT-based text encoder for caption embedding.
    CaptionDecoder: Autoregressive decoder for caption generation.
    PreferenceGuidedCaptioningModel: Main model combining all components.
    ContrastiveLoss: Contrastive loss for image-caption alignment.
    PreferenceLoss: DPO-style preference optimization loss.

Example:
    Basic usage of the preference-guided model:

    ```python
    from preference_guided_image_captioning_alignment.models.model import (
        PreferenceGuidedCaptioningModel
    )

    # Initialize model
    model = PreferenceGuidedCaptioningModel(
        vision_model="openai/clip-vit-base-patch32",
        text_model="microsoft/DialoGPT-medium"
    )

    # Forward pass for contrastive training
    outputs = model(images, captions, mode="contrastive")

    # Generate captions
    generated = model.generate_captions(images, max_length=128)
    ```
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


class VisionEncoder(nn.Module):
    """Vision encoder based on CLIP vision model for image feature extraction.

    This class implements a vision encoder that leverages pre-trained CLIP vision
    models to extract visual features from images and project them into a shared
    multimodal embedding space for alignment with text representations.

    The encoder consists of:
    - Pre-trained CLIP vision backbone (ViT or ResNet)
    - Multi-layer projection head with ReLU activation and LayerNorm
    - Optional backbone freezing for transfer learning scenarios

    Attributes:
        model_name (str): Name of the pre-trained CLIP model.
        projection_dim (int): Dimension of the output projection space.
        freeze_backbone (bool): Whether backbone parameters are frozen.
        feature_dim (int): Dimension of CLIP vision features.
        clip_model: Pre-trained CLIP model instance.
        vision_model: CLIP vision component.
        projection: Multi-layer projection head.
        logger: Logger instance for debugging and monitoring.

    Example:
        ```python
        # Initialize vision encoder
        encoder = VisionEncoder(
            model_name="openai/clip-vit-base-patch32",
            projection_dim=512,
            freeze_backbone=True
        )

        # Extract features from images
        images = torch.randn(32, 3, 224, 224)  # Batch of images
        outputs = encoder(images)
        features = outputs["features"]          # Raw CLIP features
        embeddings = outputs["embeddings"]      # Projected embeddings
        ```
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        projection_dim: int = 512,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ) -> None:
        """Initialize vision encoder.

        Args:
            model_name: Pre-trained CLIP model name.
            projection_dim: Dimension of projection layer.
            freeze_backbone: Whether to freeze backbone parameters.
            dropout: Dropout probability.
        """
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.freeze_backbone = freeze_backbone
        self.logger = logging.getLogger(__name__)

        # Load CLIP vision model
        try:
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.vision_model = self.clip_model.vision_model
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model {model_name}: {e}")
            raise

        # Get feature dimension from CLIP
        self.feature_dim = self.vision_model.config.hidden_size

        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

        self.logger.info(f"Initialized vision encoder with {model_name}")

    def _freeze_backbone(self) -> None:
        """Freeze vision model backbone parameters for transfer learning.

        This method sets requires_grad=False for all parameters in the CLIP vision
        backbone, preventing them from being updated during training. This is useful
        for transfer learning scenarios where only the projection layers should be
        fine-tuned.

        Note:
            This method is automatically called during initialization if
            freeze_backbone=True is specified.
        """
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.logger.info("Frozen vision model backbone")

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through vision encoder for image feature extraction.

        Processes a batch of images through the CLIP vision backbone and projection
        layers to generate visual representations in the shared multimodal space.

        Args:
            pixel_values (torch.Tensor): Batch of preprocessed image pixel values
                with shape (batch_size, channels, height, width). Expected to be
                normalized to [0, 1] range and resized to model's expected input size.
            attention_mask (Optional[torch.Tensor], optional): Attention mask for
                images. Currently unused but provided for API compatibility.
                Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing multiple representations:
                - "features" (torch.Tensor): Raw sequence-level vision features from
                  CLIP backbone with shape (batch_size, sequence_length, hidden_size).
                  Contains spatial feature map flattened into sequence.
                - "embeddings" (torch.Tensor): Projected image embeddings suitable for
                  contrastive learning with shape (batch_size, projection_dim).
                  These are L2-normalized and aligned with text embedding space.
                - "pooled_output" (torch.Tensor): Global pooled features from CLIP
                  with shape (batch_size, hidden_size). Represents whole-image features.

        Example:
            ```python
            batch_size, channels, height, width = 32, 3, 224, 224
            images = torch.randn(batch_size, channels, height, width)

            outputs = vision_encoder(images)
            embeddings = outputs["embeddings"]  # Shape: (32, 512)
            features = outputs["features"]      # Shape: (32, 197, 768)
            ```

        Note:
            The input images should be preprocessed using CLIP's standard preprocessing
            pipeline including resizing, center cropping, and normalization.
        """
        # Input validation
        if pixel_values.dim() != 4:
            raise ValueError(
                f"Expected pixel_values to be 4D tensor (B, C, H, W), got {pixel_values.dim()}D"
            )

        if pixel_values.size(1) != 3:
            raise ValueError(
                f"Expected 3 channels (RGB), got {pixel_values.size(1)} channels"
            )

        try:
            # Get CLIP vision features
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract features
            features = vision_outputs.last_hidden_state  # (B, seq_len, hidden_size)
            pooled_output = vision_outputs.pooler_output  # (B, hidden_size)

        except Exception as e:
            self.logger.error(f"Error in vision model forward pass: {e}")
            raise RuntimeError(f"Vision encoding failed: {e}") from e

        # Project to shared embedding space
        embeddings = self.projection(pooled_output)  # (B, projection_dim)

        return {
            "features": features,
            "embeddings": embeddings,
            "pooled_output": pooled_output,
        }


class TextEncoder(nn.Module):
    """Text encoder for caption processing and multimodal alignment.

    This class implements a text encoder that processes caption text using pre-trained
    language models (GPT-2/DialoGPT) and projects them into a shared multimodal embedding
    space for alignment with visual features. Supports parameter-efficient fine-tuning
    through LoRA (Low-Rank Adaptation).

    The encoder consists of:
    - Pre-trained transformer backbone (GPT-2, DialoGPT, etc.)
    - Optional LoRA layers for parameter-efficient fine-tuning
    - Mean pooling over token representations with attention masking
    - Multi-layer projection head with normalization

    Attributes:
        model_name (str): Name of the pre-trained text model.
        projection_dim (int): Dimension of the output projection space.
        freeze_backbone (bool): Whether backbone parameters are frozen.
        text_model: Pre-trained transformer model instance.
        tokenizer: Corresponding tokenizer for text preprocessing.
        projection: Multi-layer projection head.
        logger: Logger instance for debugging and monitoring.

    Example:
        ```python
        # Initialize text encoder with LoRA
        encoder = TextEncoder(
            model_name="microsoft/DialoGPT-medium",
            projection_dim=512,
            lora_config={"r": 16, "alpha": 32}
        )

        # Encode captions
        input_ids = torch.randint(0, 1000, (32, 128))  # Tokenized captions
        attention_mask = torch.ones_like(input_ids)
        outputs = encoder(input_ids, attention_mask)
        embeddings = outputs["embeddings"]  # Shape: (32, 512)
        ```
    """

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        projection_dim: int = 512,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize text encoder.

        Args:
            model_name: Pre-trained text model name.
            projection_dim: Dimension of projection layer.
            freeze_backbone: Whether to freeze backbone parameters.
            dropout: Dropout probability.
            lora_config: LoRA configuration for parameter-efficient fine-tuning.
        """
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.freeze_backbone = freeze_backbone
        self.logger = logging.getLogger(__name__)

        # Load text model
        try:
            self.text_model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Add same special tokens as TextProcessor to keep vocab in sync
            special_tokens = {}
            if self.tokenizer.pad_token is None:
                special_tokens["pad_token"] = "[PAD]"
            if self.tokenizer.bos_token is None:
                special_tokens["bos_token"] = "[BOS]"
            if self.tokenizer.sep_token is None:
                special_tokens["sep_token"] = "[SEP]"
            if special_tokens:
                self.tokenizer.add_special_tokens(special_tokens)
                self.text_model.resize_token_embeddings(len(self.tokenizer))

        except Exception as e:
            self.logger.error(f"Failed to load text model {model_name}: {e}")
            raise

        # Get feature dimension
        self.feature_dim = self.text_model.config.hidden_size

        # Apply LoRA if specified
        if lora_config:
            self._setup_lora(lora_config)

        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

        self.logger.info(f"Initialized text encoder with {model_name}")

    def _setup_lora(self, lora_config: Dict[str, Any]) -> None:
        """Setup LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

        Applies LoRA to the text model, allowing efficient fine-tuning by training
        only low-rank decomposition matrices instead of full model parameters.
        This significantly reduces memory usage and training time.

        Args:
            lora_config (Dict[str, Any]): LoRA configuration dictionary containing:
                - r (int): Rank of adaptation. Lower values mean fewer parameters.
                  Default: 16. Typical range: 4-64.
                - lora_alpha (int): LoRA scaling parameter. Default: 32.
                  Controls the magnitude of adaptation.
                - target_modules (List[str]): List of module names to apply LoRA to.
                  Default: ["q_proj", "v_proj"]. Common choices include attention
                  projection layers.
                - lora_dropout (float): Dropout probability for LoRA layers.
                  Default: 0.1.

        Raises:
            Exception: If LoRA setup fails due to invalid configuration or
                incompatible model architecture.

        Note:
            After LoRA setup, only LoRA parameters will be trainable unless
            freeze_backbone=False. This dramatically reduces the number of
            trainable parameters from millions to thousands.
        """
        try:
            config = LoraConfig(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 32),
                target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=lora_config.get("lora_dropout", 0.1),
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )

            self.text_model = get_peft_model(self.text_model, config)
            self.logger.info("Applied LoRA to text model")

        except Exception as e:
            self.logger.warning(f"Failed to apply LoRA: {e}")

    def _freeze_backbone(self) -> None:
        """Freeze text model backbone parameters."""
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.logger.info("Frozen text model backbone")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through text encoder.

        Args:
            input_ids: Input token IDs of shape (B, seq_len).
            attention_mask: Attention mask of shape (B, seq_len).
            return_hidden_states: Whether to return all hidden states.

        Returns:
            Dictionary containing:
                - features: Hidden states (B, seq_len, hidden_size)
                - embeddings: Projected embeddings (B, projection_dim)
                - pooled_output: Pooled features (B, hidden_size)
        """
        # Input validation
        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected input_ids to be 2D tensor (B, seq_len), got {input_ids.dim()}D"
            )

        if attention_mask.dim() != 2:
            raise ValueError(
                f"Expected attention_mask to be 2D tensor (B, seq_len), got {attention_mask.dim()}D"
            )

        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"input_ids shape {input_ids.shape} doesn't match attention_mask shape {attention_mask.shape}"
            )

        try:
            # Get text features
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=return_hidden_states,
                return_dict=True,
            )

            # Extract features
            features = outputs.last_hidden_state  # (B, seq_len, hidden_size)

            # Pool features (mean pooling with attention mask)
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(features)
            masked_features = features * attention_mask_expanded

            # Avoid division by zero
            attention_sums = attention_mask.sum(dim=1, keepdim=True)
            attention_sums = torch.clamp(attention_sums, min=1)  # Ensure no zeros
            pooled_output = masked_features.sum(dim=1) / attention_sums

        except Exception as e:
            self.logger.error(f"Error in text model forward pass: {e}")
            raise RuntimeError(f"Text encoding failed: {e}") from e

        # Project to shared embedding space (cast to float32 for LoRA compatibility)
        embeddings = self.projection(pooled_output.float())

        result = {
            "features": features.float(),
            "embeddings": embeddings,
            "pooled_output": pooled_output.float(),
        }

        if return_hidden_states:
            result["hidden_states"] = outputs.hidden_states

        return result


class CaptionDecoder(nn.Module):
    """Caption decoder for generating text descriptions.

    Generates captions conditioned on image features using an autoregressive model.
    """

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        vision_feature_dim: int = 512,
        dropout: float = 0.1,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize caption decoder.

        Args:
            model_name: Pre-trained language model name.
            vision_feature_dim: Dimension of input vision features.
            dropout: Dropout probability.
            lora_config: LoRA configuration for parameter-efficient fine-tuning.
        """
        super().__init__()
        self.model_name = model_name
        self.vision_feature_dim = vision_feature_dim
        self.logger = logging.getLogger(__name__)

        # Load language model
        try:
            self.lm_model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

            # Add special tokens
            special_tokens = {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
            self.tokenizer.add_special_tokens(special_tokens)
            self.lm_model.resize_token_embeddings(len(self.tokenizer))

        except Exception as e:
            self.logger.error(f"Failed to load language model {model_name}: {e}")
            raise

        self.hidden_size = self.lm_model.config.n_embd
        self.vocab_size = len(self.tokenizer)

        # Vision feature projection
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_feature_dim, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # Cross-attention for vision-language fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_norm = nn.LayerNorm(self.hidden_size)

        # Apply LoRA if specified
        if lora_config:
            self._setup_lora(lora_config)

        self.logger.info(f"Initialized caption decoder with {model_name}")

    def _setup_lora(self, lora_config: Dict[str, Any]) -> None:
        """Setup LoRA for parameter-efficient fine-tuning."""
        try:
            config = LoraConfig(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 32),
                target_modules=lora_config.get("target_modules", ["c_attn", "c_proj"]),
                lora_dropout=lora_config.get("lora_dropout", 0.1),
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.lm_model = get_peft_model(self.lm_model, config)
            self.logger.info("Applied LoRA to language model")

        except Exception as e:
            self.logger.warning(f"Failed to apply LoRA: {e}")

    def forward(
        self,
        vision_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through caption decoder.

        Args:
            vision_features: Vision features tensor (B, vision_dim).
            input_ids: Input token IDs (B, seq_len).
            attention_mask: Attention mask (B, seq_len).
            labels: Target labels for loss computation (B, seq_len).
            use_cache: Whether to use caching for generation.
            return_dict: Whether to return dictionary format.

        Returns:
            Dictionary containing logits and loss.
        """
        batch_size = vision_features.size(0)

        # Project vision features (ensure float32 for LoRA compatibility)
        projected_vision = self.vision_projection(vision_features.float())  # (B, hidden_size)
        projected_vision = projected_vision.unsqueeze(1)  # (B, 1, hidden_size)

        if input_ids is not None:
            # Get text embeddings (cast to float32 for cross-attention compatibility)
            text_embeddings = self.lm_model.transformer.wte(input_ids).float()  # (B, seq_len, hidden_size)

            # Apply cross-attention
            attended_text, _ = self.cross_attention(
                query=text_embeddings,
                key=projected_vision,
                value=projected_vision,
            )

            # Residual connection and normalization
            text_embeddings = self.attention_norm(text_embeddings + attended_text)

            # Forward through language model with custom embeddings
            outputs = self.lm_model(
                inputs_embeds=text_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                return_dict=return_dict,
            )
        else:
            # Generation mode - start with vision features
            outputs = self.lm_model(
                inputs_embeds=projected_vision,
                use_cache=use_cache,
                return_dict=return_dict,
            )

        return outputs

    def generate(
        self,
        vision_features: torch.Tensor,
        max_length: int = 50,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate captions using beam search or sampling.

        Args:
            vision_features: Vision features tensor (B, vision_dim).
            max_length: Maximum generation length.
            num_beams: Number of beams for beam search.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            top_p: Top-p sampling parameter.
            repetition_penalty: Repetition penalty.
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID.
            **kwargs: Additional generation parameters.

        Returns:
            Generated token IDs tensor (B, generated_length).
        """
        batch_size = vision_features.size(0)

        # Project vision features as initial input
        projected_vision = self.vision_projection(vision_features)
        initial_embeddings = projected_vision.unsqueeze(1)  # (B, 1, hidden_size)

        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # Generate with language model
        with torch.no_grad():
            outputs = self.lm_model.generate(
                inputs_embeds=initial_embeddings,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs,
            )

        return outputs


class PreferenceGuidedCaptioningModel(nn.Module):
    """Preference-guided multimodal captioning model for human-aligned caption generation.

    This is the main model class that combines vision encoding, text encoding, and
    caption generation capabilities with preference-based alignment. The model supports
    two-stage training: contrastive learning for multimodal alignment followed by
    preference optimization for human preference alignment.

    The model architecture consists of:
    - VisionEncoder: CLIP-based image feature extraction
    - TextEncoder: GPT-based text embedding for contrastive learning
    - CaptionDecoder: Autoregressive decoder for caption generation
    - Shared embedding space for image-text alignment

    Training modes:
    - "contrastive": Image-caption contrastive learning for alignment
    - "generation": Caption generation with cross-entropy loss
    - "preference": Human preference optimization (DPO-style)

    Attributes:
        vision_encoder (VisionEncoder): Image feature extraction component.
        text_encoder (TextEncoder): Text embedding component for contrastive learning.
        caption_decoder (CaptionDecoder): Caption generation component.
        projection_dim (int): Dimension of shared embedding space.
        logger: Logger instance for debugging and monitoring.

    Example:
        ```python
        # Initialize complete model
        model = PreferenceGuidedCaptioningModel(
            vision_model="openai/clip-vit-base-patch32",
            text_model="microsoft/DialoGPT-medium",
            projection_dim=512
        )

        # Contrastive training mode
        batch_size = 32
        images = torch.randn(batch_size, 3, 224, 224)
        captions = torch.randint(0, 1000, (batch_size, 128))
        attention_mask = torch.ones_like(captions)

        outputs = model(images, captions, attention_mask, mode="contrastive")
        contrastive_loss = compute_contrastive_loss(outputs)

        # Caption generation
        generated = model.generate_captions(
            images,
            max_length=128,
            num_beams=4,
            temperature=0.7
        )
        ```

    Note:
        This model requires careful two-stage training for optimal performance.
        Stage 1 focuses on image-text alignment through contrastive learning,
        while Stage 2 optimizes for human preferences using preference data.
    """

    def __init__(
        self,
        vision_model: str = "openai/clip-vit-base-patch32",
        text_model: str = "microsoft/DialoGPT-medium",
        projection_dim: int = 512,
        temperature: float = 0.07,
        dropout: float = 0.1,
        freeze_vision_backbone: bool = False,
        freeze_text_backbone: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize preference-guided captioning model.

        Args:
            vision_model: Pre-trained vision model name.
            text_model: Pre-trained text model name.
            projection_dim: Shared embedding dimension.
            temperature: Temperature for contrastive learning.
            dropout: Dropout probability.
            freeze_vision_backbone: Whether to freeze vision backbone.
            freeze_text_backbone: Whether to freeze text backbone.
            lora_config: LoRA configuration.
        """
        super().__init__()
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

        # Initialize encoders
        self.vision_encoder = VisionEncoder(
            model_name=vision_model,
            projection_dim=projection_dim,
            freeze_backbone=freeze_vision_backbone,
            dropout=dropout,
        )

        self.text_encoder = TextEncoder(
            model_name=text_model,
            projection_dim=projection_dim,
            freeze_backbone=freeze_text_backbone,
            dropout=dropout,
            lora_config=lora_config,
        )

        # Initialize decoder for caption generation
        self.caption_decoder = CaptionDecoder(
            model_name=text_model,
            vision_feature_dim=projection_dim,
            dropout=dropout,
            lora_config=lora_config,
        )

        self.logger.info("Initialized PreferenceGuidedCaptioningModel")

    def forward(
        self,
        images: torch.Tensor,
        caption_ids: Optional[torch.Tensor] = None,
        caption_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mode: str = "contrastive",
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            images: Input images tensor (B, C, H, W).
            caption_ids: Caption token IDs (B, seq_len).
            caption_mask: Caption attention mask (B, seq_len).
            labels: Target labels for generation loss (B, seq_len).
            mode: Forward mode ('contrastive', 'generation', 'dual').

        Returns:
            Dictionary containing model outputs and losses.
        """
        batch_size = images.size(0)
        outputs = {}

        # Encode images
        vision_outputs = self.vision_encoder(images)
        image_embeddings = vision_outputs["embeddings"]  # (B, projection_dim)

        if mode in ["contrastive", "dual"] and caption_ids is not None:
            # Encode captions for contrastive learning
            text_outputs = self.text_encoder(caption_ids, caption_mask)
            text_embeddings = text_outputs["embeddings"]  # (B, projection_dim)

            # L2 normalize embeddings for contrastive learning
            # F.normalize uses clamp(norm, min=eps) internally which is gradient-safe
            image_embeddings_norm = F.normalize(image_embeddings, p=2, dim=-1)
            text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)

            outputs.update({
                "image_embeddings": image_embeddings_norm,
                "text_embeddings": text_embeddings_norm,
                "vision_features": vision_outputs["features"],
                "text_features": text_outputs["features"],
            })

        if mode in ["generation", "dual"]:
            # Generate captions
            decoder_outputs = self.caption_decoder(
                vision_features=image_embeddings,
                input_ids=caption_ids,
                attention_mask=caption_mask,
                labels=labels,
                return_dict=True,
            )

            outputs.update({
                "logits": decoder_outputs.logits,
                "generation_loss": decoder_outputs.loss if decoder_outputs.loss is not None else torch.tensor(0.0, device=images.device),
            })

        return outputs

    def generate_captions(
        self,
        images: torch.Tensor,
        max_length: int = 50,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        """Generate captions for input images.

        Args:
            images: Input images tensor (B, C, H, W).
            max_length: Maximum caption length.
            num_beams: Number of beams for beam search.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            top_p: Top-p sampling parameter.
            **kwargs: Additional generation parameters.

        Returns:
            List of generated captions.
        """
        start_time = time.time()
        batch_size = images.shape[0]
        self.logger.debug(f"Starting caption generation for batch of {batch_size} images")

        self.eval()
        with torch.no_grad():
            # Encode images
            encode_start = time.time()
            vision_outputs = self.vision_encoder(images)
            image_embeddings = vision_outputs["embeddings"]
            encode_time = time.time() - encode_start
            self.logger.debug(f"Vision encoding completed in {encode_time:.3f}s")

            # Generate captions
            generation_start = time.time()
            generated_ids = self.caption_decoder.generate(
                vision_features=image_embeddings,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                **kwargs,
            )
            generation_time = time.time() - generation_start
            self.logger.debug(f"Caption generation completed in {generation_time:.3f}s")

            # Decode to text
            decode_start = time.time()
            captions = []
            for ids in generated_ids:
                caption = self.caption_decoder.tokenizer.decode(
                    ids, skip_special_tokens=True
                ).strip()
                captions.append(caption)
            decode_time = time.time() - decode_start

            total_time = time.time() - start_time
            self.logger.info(
                f"Generated {len(captions)} captions in {total_time:.3f}s "
                f"(encode: {encode_time:.3f}s, generate: {generation_time:.3f}s, "
                f"decode: {decode_time:.3f}s) - {total_time/len(captions):.3f}s per caption"
            )

            return captions

    def compute_similarity(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute image-caption similarity scores.

        Args:
            images: Input images tensor (B, C, H, W).
            captions: Caption token IDs (B, seq_len).
            caption_mask: Caption attention mask (B, seq_len).

        Returns:
            Similarity scores tensor (B, B).
        """
        outputs = self(
            images=images,
            caption_ids=captions,
            caption_mask=caption_mask,
            mode="contrastive",
        )

        image_embeddings = outputs["image_embeddings"]
        text_embeddings = outputs["text_embeddings"]

        # Compute cosine similarity
        similarity = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature

        return similarity


class ContrastiveLoss(nn.Module):
    """Contrastive loss for image-caption alignment."""

    def __init__(self, temperature: float = 0.07) -> None:
        """Initialize contrastive loss.

        Args:
            temperature: Temperature parameter for scaling similarities.
        """
        super().__init__()
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            image_embeddings: Normalized image embeddings (B, D).
            text_embeddings: Normalized text embeddings (B, D).

        Returns:
            Contrastive loss value.
        """
        batch_size = image_embeddings.size(0)
        device = image_embeddings.device

        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature

        # Create labels (positive pairs are on the diagonal)
        labels = torch.arange(batch_size, device=device)

        # Compute losses for both directions
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)

        # Average the two losses
        contrastive_loss = (loss_i2t + loss_t2i) / 2

        return contrastive_loss


class PreferenceLoss(nn.Module):
    """Preference loss for human preference alignment using DPO-style optimization."""

    def __init__(self, beta: float = 0.1) -> None:
        """Initialize preference loss.

        Args:
            beta: Regularization parameter for preference optimization.
        """
        super().__init__()
        self.beta = beta
        self.logger = logging.getLogger(__name__)

    def forward(
        self,
        preferred_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        preferred_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
        preferred_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute preference loss.

        Args:
            preferred_logits: Logits for preferred captions (B, seq_len, vocab_size).
            rejected_logits: Logits for rejected captions (B, seq_len, vocab_size).
            preferred_labels: Target labels for preferred captions (B, seq_len).
            rejected_labels: Target labels for rejected captions (B, seq_len).
            preferred_mask: Attention mask for preferred captions (B, seq_len).
            rejected_mask: Attention mask for rejected captions (B, seq_len).

        Returns:
            Preference loss value.
        """
        # Compute log probabilities
        preferred_log_probs = self._compute_log_probs(
            preferred_logits, preferred_labels, preferred_mask
        )
        rejected_log_probs = self._compute_log_probs(
            rejected_logits, rejected_labels, rejected_mask
        )

        # Compute preference loss using DPO formulation
        log_ratio = preferred_log_probs - rejected_log_probs
        preference_loss = -F.logsigmoid(self.beta * log_ratio).mean()

        return preference_loss

    def _compute_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for given logits and labels.

        Args:
            logits: Model logits (B, seq_len, vocab_size).
            labels: Target labels (B, seq_len).
            mask: Attention mask (B, seq_len).

        Returns:
            Log probabilities tensor (B,).
        """
        # Shift labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather probabilities for target tokens
        target_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Apply mask and compute average log probability per sequence
        masked_log_probs = target_log_probs * shift_mask
        sequence_log_probs = masked_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1)

        return sequence_log_probs