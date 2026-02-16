"""Custom model components for preference-guided image captioning.

This module contains reusable components including custom loss functions,
attention mechanisms, and training utilities for the preference-guided
captioning system.

Components:
    ContrastiveLoss: NT-Xent contrastive loss for vision-text alignment
    DPOPreferenceLoss: Direct Preference Optimization loss for caption alignment
    TemperatureScaledSimilarity: Temperature-scaled cosine similarity
    NaNSafeGradientNorm: Gradient clipping with NaN detection
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemperatureScaledSimilarity(nn.Module):
    """Temperature-scaled cosine similarity for contrastive learning.

    This component computes cosine similarity between vision and text embeddings
    with learnable or fixed temperature scaling. Temperature controls the
    sharpness of the similarity distribution and gradient magnitude.

    Args:
        temperature: Initial temperature value (default: 0.5)
        learnable: Whether temperature is a learnable parameter
        min_temp: Minimum allowed temperature (prevents division by zero)
        max_temp: Maximum allowed temperature (prevents gradient explosion)

    Example:
        >>> similarity = TemperatureScaledSimilarity(temperature=0.5)
        >>> vision_embeds = torch.randn(32, 512)
        >>> text_embeds = torch.randn(32, 512)
        >>> sim_matrix = similarity(vision_embeds, text_embeds)
        >>> print(sim_matrix.shape)  # (32, 32)
    """

    def __init__(
        self,
        temperature: float = 0.5,
        learnable: bool = False,
        min_temp: float = 0.1,
        max_temp: float = 2.0,
    ):
        super().__init__()
        self.min_temp = min_temp
        self.max_temp = max_temp

        if learnable:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))

    def forward(
        self, vision_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Compute temperature-scaled similarity matrix.

        Args:
            vision_embeds: Vision embeddings [batch_size, embed_dim]
            text_embeds: Text embeddings [batch_size, embed_dim]

        Returns:
            Similarity matrix [batch_size, batch_size]
        """
        # Normalize embeddings
        vision_embeds = F.normalize(vision_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        # Clamp temperature to valid range
        temp = torch.clamp(self.temperature, self.min_temp, self.max_temp)

        # Compute scaled similarity
        similarity = torch.matmul(vision_embeds, text_embeds.T) / temp

        return similarity


class ContrastiveLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.

    Implements the InfoNCE contrastive loss used in CLIP and SimCLR for learning
    aligned vision-text representations. Treats image-caption pairs as positive
    examples and all other combinations in the batch as negatives.

    The loss encourages:
    - High similarity between matching image-caption pairs
    - Low similarity between mismatched pairs

    Args:
        temperature: Temperature parameter for similarity scaling (default: 0.5)
        reduction: Loss reduction method ('mean' or 'sum')

    Mathematical formulation:
        For a batch of N image-caption pairs (v_i, t_i):
        L = -1/N * sum_i [ log( exp(sim(v_i, t_i)/tau) / sum_j exp(sim(v_i, t_j)/tau) ) ]

    Example:
        >>> loss_fn = ContrastiveLoss(temperature=0.5)
        >>> vision_embeds = torch.randn(32, 512)
        >>> text_embeds = torch.randn(32, 512)
        >>> loss = loss_fn(vision_embeds, text_embeds)
    """

    def __init__(self, temperature: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.similarity = TemperatureScaledSimilarity(temperature=temperature)
        self.reduction = reduction

    def forward(
        self, vision_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            vision_embeds: Vision embeddings [batch_size, embed_dim]
            text_embeds: Text embeddings [batch_size, embed_dim]

        Returns:
            Contrastive loss scalar
        """
        batch_size = vision_embeds.size(0)

        # Compute similarity matrices
        sim_v2t = self.similarity(vision_embeds, text_embeds)  # [B, B]
        sim_t2v = sim_v2t.T  # [B, B]

        # Create labels (diagonal indices are positives)
        labels = torch.arange(batch_size, device=vision_embeds.device)

        # Compute cross-entropy loss in both directions
        loss_v2t = F.cross_entropy(sim_v2t, labels, reduction=self.reduction)
        loss_t2v = F.cross_entropy(sim_t2v, labels, reduction=self.reduction)

        # Average bidirectional losses
        loss = (loss_v2t + loss_t2v) / 2.0

        return loss


class DPOPreferenceLoss(nn.Module):
    """Direct Preference Optimization (DPO) loss for preference learning.

    Implements the DPO loss function for aligning model outputs with human
    preferences. Given a reference model and pairs of chosen/rejected outputs,
    DPO optimizes the policy to increase the likelihood ratio of chosen over
    rejected outputs.

    This is the core novelty component that enables preference-guided alignment
    without requiring reward model training or RL optimization.

    Args:
        beta: Temperature parameter controlling KL regularization strength
        reference_free: If True, omits reference model (assumes uniform prior)
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Mathematical formulation:
        L_DPO = -log(sigmoid(beta * log(pi(y_w|x)/pi_ref(y_w|x)) -
                                   beta * log(pi(y_l|x)/pi_ref(y_l|x))))

        where y_w = chosen output, y_l = rejected output, pi = policy,
        pi_ref = reference policy, beta = KL regularization strength

    Example:
        >>> dpo_loss = DPOPreferenceLoss(beta=0.1)
        >>> chosen_logprobs = torch.randn(16)  # Log probs for chosen captions
        >>> rejected_logprobs = torch.randn(16)  # Log probs for rejected captions
        >>> ref_chosen_logprobs = torch.randn(16)  # Reference log probs (chosen)
        >>> ref_rejected_logprobs = torch.randn(16)  # Reference log probs (rejected)
        >>> loss = dpo_loss(chosen_logprobs, rejected_logprobs,
        ...                 ref_chosen_logprobs, ref_rejected_logprobs)
    """

    def __init__(
        self,
        beta: float = 0.1,
        reference_free: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.reference_free = reference_free
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logprobs: torch.Tensor,
        policy_rejected_logprobs: torch.Tensor,
        reference_chosen_logprobs: Optional[torch.Tensor] = None,
        reference_rejected_logprobs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute DPO loss.

        Args:
            policy_chosen_logprobs: Log probabilities for chosen outputs [batch_size]
            policy_rejected_logprobs: Log probabilities for rejected outputs [batch_size]
            reference_chosen_logprobs: Reference log probs for chosen [batch_size]
            reference_rejected_logprobs: Reference log probs for rejected [batch_size]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute log ratios for policy
        policy_logratios = policy_chosen_logprobs - policy_rejected_logprobs

        # Compute log ratios for reference (if provided)
        if self.reference_free or reference_chosen_logprobs is None:
            reference_logratios = torch.zeros_like(policy_logratios)
        else:
            reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs

        # Compute DPO loss: -log(sigmoid(beta * (policy_logratios - ref_logratios)))
        logits = self.beta * (policy_logratios - reference_logratios)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Smooth labels: (1 - smoothing) * 1 + smoothing * 0.5
            smoothed_labels = (1.0 - self.label_smoothing) * torch.ones_like(logits)
            loss = F.binary_cross_entropy_with_logits(
                logits, smoothed_labels, reduction="mean"
            )
        else:
            # Standard DPO loss
            loss = -F.logsigmoid(logits).mean()

        # Compute metrics
        with torch.no_grad():
            # Reward margin (higher is better)
            reward_margin = (policy_logratios - reference_logratios).mean()

            # Implicit reward accuracy (chosen > rejected)
            reward_accuracy = (policy_logratios > reference_logratios).float().mean()

        metrics = {
            "dpo_loss": loss.item(),
            "reward_margin": reward_margin.item(),
            "reward_accuracy": reward_accuracy.item(),
            "policy_chosen_logprob": policy_chosen_logprobs.mean().item(),
            "policy_rejected_logprob": policy_rejected_logprobs.mean().item(),
        }

        return loss, metrics


class NaNSafeGradientNorm(nn.Module):
    """Gradient clipping with NaN/Inf detection and handling.

    This component provides safe gradient clipping that detects NaN or Inf
    gradients and returns diagnostic information. Used to prevent training
    instability from numerical issues.

    Args:
        max_norm: Maximum gradient norm for clipping
        norm_type: Type of norm (2.0 for L2 norm)
        error_if_nonfinite: If True, raises error on NaN/Inf gradients

    Example:
        >>> grad_clipper = NaNSafeGradientNorm(max_norm=1.0)
        >>> parameters = model.parameters()
        >>> total_norm, is_finite = grad_clipper(parameters)
        >>> if not is_finite:
        ...     print("NaN or Inf gradients detected!")
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
    ):
        super().__init__()
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite

    def forward(self, parameters) -> Tuple[torch.Tensor, bool]:
        """Clip gradients and check for NaN/Inf.

        Args:
            parameters: Iterable of parameters (typically model.parameters())

        Returns:
            Tuple of (total_norm, is_finite)
                total_norm: Total gradient norm before clipping
                is_finite: True if all gradients are finite
        """
        # Filter parameters with gradients
        parameters = [p for p in parameters if p.grad is not None]

        if len(parameters) == 0:
            return torch.tensor(0.0), True

        # Compute total gradient norm
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), self.norm_type) for p in parameters]),
            self.norm_type,
        )

        # Check if norm is finite
        is_finite = torch.isfinite(total_norm).item()

        if not is_finite:
            if self.error_if_nonfinite:
                raise RuntimeError("Non-finite gradient norm detected")
            logger.warning(f"Non-finite gradient norm detected: {total_norm}")
            return total_norm, False

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)

        return total_norm, is_finite


def compute_sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-sequence log probabilities from model logits.

    This utility function computes the log probability of each sequence given
    the model's logits. Used in DPO loss computation for chosen/rejected captions.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        attention_mask: Mask for valid tokens [batch_size, seq_len]

    Returns:
        Per-sequence log probabilities [batch_size]
    """
    # Shift logits and labels for autoregressive modeling
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous()
    else:
        shift_mask = torch.ones_like(shift_labels)

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for actual tokens
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask padding tokens
    token_log_probs = token_log_probs * shift_mask

    # Sum over sequence length
    sequence_log_probs = token_log_probs.sum(dim=1)

    return sequence_log_probs
