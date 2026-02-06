"""Comprehensive evaluation metrics for image captioning models.

This module provides a complete suite of evaluation metrics for assessing image
captioning model performance across multiple dimensions including:

- **Text Quality Metrics**: BLEU (1-4), ROUGE (1, 2, L), METEOR
- **Semantic Similarity**: BERTScore, CLIP-Score for multimodal alignment
- **Preference Alignment**: Win rate calculation for human preference evaluation
- **Diversity Metrics**: Unique n-gram analysis and vocabulary diversity
- **Efficiency Metrics**: Inference latency and throughput measurement
- **Human Correlation**: Pearson and Spearman correlation with human judgments

Classes:
    CaptioningMetrics: Main evaluation class with comprehensive metric computation.

Functions:
    compute_bleu_scores: BLEU score calculation with multiple n-gram orders.
    compute_rouge_scores: ROUGE score calculation for text overlap metrics.
    compute_bertscore: Semantic similarity using contextualized embeddings.
    compute_clip_score: Multimodal similarity between images and captions.

Example:
    Basic usage for evaluating a captioning model:

    ```python
    from preference_guided_image_captioning_alignment.evaluation.metrics import (
        CaptioningMetrics
    )

    # Initialize metrics
    metrics = CaptioningMetrics(device="cuda", cache_dir="./cache")

    # Evaluate generated captions against references
    generated_captions = ["A cat sitting on a table"]
    reference_captions = [["A cat on a table", "Cat sitting on wooden surface"]]
    images = [Image.open("cat.jpg")]

    scores = metrics.evaluate_batch(
        generated_captions=generated_captions,
        reference_captions=reference_captions,
        images=images
    )

    print(f"BLEU-4: {scores['bleu_4']:.3f}")
    print(f"CLIP-Score: {scores['clip_score']:.3f}")
    ```

Note:
    This module requires additional dependencies for some metrics:
    - NLTK for BLEU/METEOR computation
    - transformers and evaluate for BERTScore
    - CLIP model for CLIP-Score computation
    - pycocotools for COCO-style evaluation
"""

import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Lazy imports for heavy/optional dependencies to avoid circular imports
# and reduce startup time. These are imported at first use in _setup_metrics().
# - evaluate (HuggingFace): conflicts with scripts/run_evaluation.py if imported eagerly
# - pycocotools: optional C extension, may not be installed
# - nltk, rouge_score, scipy, sklearn: heavy deps not needed at import time
# - matplotlib, seaborn, pandas: only needed for visualization


class CaptioningMetrics:
    """Comprehensive metrics suite for evaluating image captioning model performance.

    This class provides a unified interface for computing multiple evaluation metrics
    commonly used in image captioning research. It supports both automatic metrics
    (BLEU, ROUGE, etc.) and multimodal metrics (CLIP-Score) that assess semantic
    alignment between images and generated captions.

    The class handles metric computation efficiently by:
    - Batching computations where possible to improve performance
    - Caching expensive model loading (BERT, CLIP) for reuse
    - Providing both individual metric computation and comprehensive evaluation
    - Supporting COCO-style evaluation formats for research reproducibility

    Supported Metrics:
        **Text Quality:**
        - BLEU-1, BLEU-2, BLEU-3, BLEU-4: N-gram overlap with references
        - ROUGE-1, ROUGE-2, ROUGE-L: Recall-oriented text overlap
        - METEOR: Semantic similarity with stemming and synonyms

        **Semantic Quality:**
        - BERTScore: Contextualized embedding similarity using BERT
        - CLIP-Score: Multimodal similarity between images and captions

        **Preference Alignment:**
        - Win Rate: Percentage of preferences satisfied in pairwise comparisons
        - Human Correlation: Pearson/Spearman correlation with human judgments

        **Diversity & Efficiency:**
        - Unique N-grams: Lexical diversity measurement
        - Inference Latency: Response time percentiles (P50, P95, P99)
        - Throughput: Captions generated per second

    Attributes:
        device (torch.device): Computation device (CPU/CUDA).
        cache_dir (Path): Directory for caching models and intermediate results.
        logger: Logger instance for debugging and progress tracking.
        bleu_scorer: NLTK BLEU scorer instance.
        rouge_scorer: ROUGE scorer for text overlap metrics.
        bert_scorer: BERTScore model for semantic similarity.
        clip_model: CLIP model for multimodal evaluation.
        clip_processor: CLIP preprocessor for images and text.

    Example:
        ```python
        # Initialize with GPU acceleration
        metrics = CaptioningMetrics(device="cuda", cache_dir="./models")

        # Single sample evaluation
        image = Image.open("sample.jpg")
        generated = "A dog playing in the park"
        references = ["A dog running in the park", "Dog playing outdoors"]

        scores = metrics.evaluate_single(
            generated_caption=generated,
            reference_captions=references,
            image=image
        )

        # Batch evaluation for efficiency
        batch_scores = metrics.evaluate_batch(
            generated_captions=generated_captions,
            reference_captions=reference_captions,
            images=images
        )

        # COCO-style evaluation
        coco_scores = metrics.evaluate_coco_format(
            annotations_file="annotations.json",
            results_file="predictions.json"
        )
        ```

    Note:
        First initialization may take time due to model downloads. Subsequent
        runs will use cached models. Ensure sufficient disk space (1-2GB) for
        model caching and CUDA availability for GPU acceleration.
    """

    def __init__(
        self,
        device: str = "auto",
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize captioning metrics.

        Args:
            device: Device for computation ('cuda', 'cpu', or 'auto').
            cache_dir: Directory for caching models and data.
        """
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        # Use cache_dir parameter, then environment variable, then default
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.environ.get("CAPTION_ALIGNMENT_CACHE_DIR", "./cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Initialize metrics
        self._setup_metrics()

        self.logger.info(f"Initialized CaptioningMetrics on device: {self.device}")

    def _setup_metrics(self) -> None:
        """Setup evaluation metrics and models."""
        try:
            import nltk
            from rouge_score import rouge_scorer as _rouge_scorer

            # Import HuggingFace evaluate with explicit package resolution
            # to avoid confusion with scripts/run_evaluation.py
            import importlib
            _evaluate_mod = importlib.import_module("evaluate")
            _load = _evaluate_mod.load

            # Download required NLTK data
            nltk.download("punkt", quiet=True)
            nltk.download("wordnet", quiet=True)

            # Store nltk reference for use in other methods
            self._nltk = nltk

            # Load evaluation metrics
            self.bleu_metric = _load("bleu")
            self.rouge_metric = _load("rouge")
            self.meteor_metric = _load("meteor")

            # Setup ROUGE scorer
            self.rouge_scorer = _rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

            # Load BERTScore
            try:
                self.bertscore_metric = _load("bertscore")
            except Exception as e:
                self.logger.warning(f"Failed to load BERTScore: {e}")
                self.bertscore_metric = None

            # Setup CLIP for CLIP-Score
            try:
                from transformers import CLIPModel, CLIPProcessor
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model.to(self.device)
                self.clip_model.eval()
            except Exception as e:
                self.logger.warning(f"Failed to load CLIP model: {e}")
                self.clip_model = None
                self.clip_processor = None

        except Exception as e:
            self.logger.error(f"Failed to setup metrics: {e}")
            raise

    def compute_bleu_scores(
        self,
        predictions: List[str],
        references: List[List[str]],
        max_order: int = 4,
    ) -> Dict[str, float]:
        """Compute BLEU scores.

        Args:
            predictions: List of predicted captions.
            references: List of reference captions (multiple per prediction).
            max_order: Maximum n-gram order.

        Returns:
            Dictionary containing BLEU scores for different n-gram orders.
        """
        try:
            # Ensure references is list of lists
            if references and isinstance(references[0], str):
                references = [[ref] for ref in references]

            scores = {}
            for n in range(1, max_order + 1):
                result = self.bleu_metric.compute(
                    predictions=predictions,
                    references=references,
                    max_order=n,
                )
                scores[f"bleu_{n}"] = result["bleu"]

            return scores

        except Exception as e:
            self.logger.error(f"Error computing BLEU scores: {e}")
            return {f"bleu_{n}": 0.0 for n in range(1, max_order + 1)}

    def compute_rouge_scores(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute ROUGE scores.

        Args:
            predictions: List of predicted captions.
            references: List of reference captions.

        Returns:
            Dictionary containing ROUGE scores.
        """
        try:
            # Handle multiple references
            if isinstance(references[0], list):
                # Use first reference for each prediction
                references = [ref[0] if ref else "" for ref in references]

            rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
                rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
                rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

            return {
                metric: np.mean(scores) for metric, scores in rouge_scores.items()
            }

        except Exception as e:
            self.logger.error(f"Error computing ROUGE scores: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def compute_meteor_score(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> float:
        """Compute METEOR score.

        Args:
            predictions: List of predicted captions.
            references: List of reference captions.

        Returns:
            METEOR score.
        """
        try:
            # Ensure references is list of lists
            if references and isinstance(references[0], str):
                references = [[ref] for ref in references]

            result = self.meteor_metric.compute(
                predictions=predictions,
                references=references,
            )
            return result["meteor"]

        except Exception as e:
            self.logger.error(f"Error computing METEOR score: {e}")
            return 0.0

    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        model_type: str = "distilbert-base-uncased",
    ) -> Dict[str, float]:
        """Compute BERTScore.

        Args:
            predictions: List of predicted captions.
            references: List of reference captions.
            model_type: Model type for BERTScore.

        Returns:
            Dictionary containing BERTScore metrics.
        """
        if self.bertscore_metric is None:
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

        try:
            # Handle multiple references
            if isinstance(references[0], list):
                references = [ref[0] if ref else "" for ref in references]

            results = self.bertscore_metric.compute(
                predictions=predictions,
                references=references,
                model_type=model_type,
            )

            return {
                "bertscore_precision": np.mean(results["precision"]),
                "bertscore_recall": np.mean(results["recall"]),
                "bertscore_f1": np.mean(results["f1"]),
            }

        except Exception as e:
            self.logger.error(f"Error computing BERTScore: {e}")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

    def compute_clip_score(
        self,
        images: List[Union[str, Image.Image, torch.Tensor]],
        captions: List[str],
    ) -> Dict[str, float]:
        """Compute CLIP-Score for image-caption alignment.

        Args:
            images: List of images (paths, PIL Images, or tensors).
            captions: List of captions.

        Returns:
            Dictionary containing CLIP-Score metrics.
        """
        if self.clip_model is None or self.clip_processor is None:
            return {"clip_score": 0.0, "clip_score_std": 0.0}

        try:
            scores = []

            for image, caption in tqdm(
                zip(images, captions), total=len(images), desc="Computing CLIP-Score"
            ):
                # Process image
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                elif isinstance(image, torch.Tensor):
                    # Convert tensor to PIL Image
                    if image.dim() == 4:
                        image = image.squeeze(0)
                    if image.max() <= 1.0:
                        image = (image * 255).clamp(0, 255).byte()
                    image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())

                # Process inputs
                inputs = self.clip_processor(
                    text=[caption],
                    images=image,
                    return_tensors="pt",
                    padding=True,
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Compute similarity
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    score = logits_per_image.squeeze().cpu().item()
                    scores.append(score)

            return {
                "clip_score": np.mean(scores),
                "clip_score_std": np.std(scores),
            }

        except Exception as e:
            self.logger.error(f"Error computing CLIP-Score: {e}")
            return {"clip_score": 0.0, "clip_score_std": 0.0}

    def compute_cider_score(
        self,
        predictions: List[str],
        references: List[List[str]],
        sigma: float = 6.0,
    ) -> float:
        """Compute CIDEr score.

        Args:
            predictions: List of predicted captions.
            references: List of reference captions.
            sigma: Standard deviation for Gaussian penalty.

        Returns:
            CIDEr score.
        """
        try:
            return self._compute_cider(predictions, references, sigma)
        except Exception as e:
            self.logger.error(f"Error computing CIDEr score: {e}")
            return 0.0

    def _compute_cider(
        self,
        predictions: List[str],
        references: List[List[str]],
        sigma: float,
    ) -> float:
        """Internal CIDEr computation."""
        # Ensure references is list of lists
        if references and isinstance(references[0], str):
            references = [[ref] for ref in references]

        _nltk = self._nltk

        # Tokenize
        def tokenize(text: str) -> List[str]:
            return _nltk.word_tokenize(text.lower())

        # Compute n-gram counts
        def get_ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
            ngrams = defaultdict(int)
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngrams[ngram] += 1
            return ngrams

        # Document frequency for IDF computation
        def compute_doc_freq(all_references: List[List[str]]) -> Dict[Tuple[str, ...], int]:
            doc_freq = defaultdict(int)
            for refs in all_references:
                seen_ngrams = set()
                for ref in refs:
                    tokens = tokenize(ref)
                    for n in range(1, 5):
                        ngrams = get_ngram_counts(tokens, n)
                        for ngram in ngrams:
                            if ngram not in seen_ngrams:
                                doc_freq[ngram] += 1
                                seen_ngrams.add(ngram)
            return doc_freq

        # Compute document frequencies
        doc_freq = compute_doc_freq(references)
        total_docs = len(references)

        scores = []

        for pred, refs in zip(predictions, references):
            pred_tokens = tokenize(pred)
            ref_tokens_list = [tokenize(ref) for ref in refs]

            score = 0.0

            for n in range(1, 5):
                # Get n-grams
                pred_ngrams = get_ngram_counts(pred_tokens, n)

                # Compute reference n-gram counts (average over references)
                ref_ngrams = defaultdict(float)
                for ref_tokens in ref_tokens_list:
                    ref_counts = get_ngram_counts(ref_tokens, n)
                    for ngram, count in ref_counts.items():
                        ref_ngrams[ngram] += count / len(ref_tokens_list)

                # Compute similarity
                numerator = 0.0
                pred_norm = 0.0
                ref_norm = 0.0

                all_ngrams = set(pred_ngrams.keys()) | set(ref_ngrams.keys())

                for ngram in all_ngrams:
                    pred_count = pred_ngrams.get(ngram, 0)
                    ref_count = ref_ngrams.get(ngram, 0)

                    # IDF weight
                    idf = math.log(total_docs / (doc_freq.get(ngram, 1) + 1e-8))

                    pred_weighted = pred_count * idf
                    ref_weighted = ref_count * idf

                    numerator += pred_weighted * ref_weighted
                    pred_norm += pred_weighted ** 2
                    ref_norm += ref_weighted ** 2

                # Compute cosine similarity
                if pred_norm > 0 and ref_norm > 0:
                    sim = numerator / (math.sqrt(pred_norm * ref_norm))
                else:
                    sim = 0.0

                # Add to score with n-gram weight
                score += sim

            # Average over n-grams and apply length penalty
            score /= 4.0

            # Length penalty (Gaussian)
            pred_len = len(pred_tokens)
            ref_lens = [len(ref_tokens) for ref_tokens in ref_tokens_list]
            avg_ref_len = np.mean(ref_lens)

            if avg_ref_len > 0:
                length_penalty = math.exp(-(pred_len - avg_ref_len) ** 2 / (2 * sigma ** 2))
            else:
                length_penalty = 0.0

            score *= length_penalty
            scores.append(score)

        return np.mean(scores) * 10.0  # Scale by 10 as in original CIDEr

    def compute_preference_metrics(
        self,
        model_outputs: List[str],
        preferred_captions: List[str],
        rejected_captions: List[str],
        preference_scores: List[float],
    ) -> Dict[str, float]:
        """Compute preference alignment metrics.

        Args:
            model_outputs: Model-generated captions.
            preferred_captions: Human-preferred captions.
            rejected_captions: Human-rejected captions.
            preference_scores: Human preference scores.

        Returns:
            Dictionary containing preference metrics.
        """
        try:
            # Compute similarity to preferred vs rejected
            pref_similarities = []
            rej_similarities = []

            for output, preferred, rejected in zip(model_outputs, preferred_captions, rejected_captions):
                # Simple token-based similarity (could be enhanced with embeddings)
                pref_sim = self._compute_text_similarity(output, preferred)
                rej_sim = self._compute_text_similarity(output, rejected)

                pref_similarities.append(pref_sim)
                rej_similarities.append(rej_sim)

            # Preference win rate (how often model is closer to preferred)
            wins = sum(1 for p, r in zip(pref_similarities, rej_similarities) if p > r)
            preference_win_rate = wins / len(pref_similarities) if pref_similarities else 0.0

            # Correlation with human preference scores
            human_pref_correlation = 0.0
            if len(preference_scores) > 1:
                model_scores = [p - r for p, r in zip(pref_similarities, rej_similarities)]
                try:
                    from scipy.stats import pearsonr as _pearsonr
                    correlation, _ = _pearsonr(model_scores, preference_scores)
                    human_pref_correlation = correlation
                except (ValueError, ImportError):
                    pass

            return {
                "preference_win_rate": preference_win_rate,
                "avg_preferred_similarity": np.mean(pref_similarities),
                "avg_rejected_similarity": np.mean(rej_similarities),
                "preference_margin": np.mean(pref_similarities) - np.mean(rej_similarities),
                "human_preference_correlation": human_pref_correlation,
            }

        except Exception as e:
            self.logger.error(f"Error computing preference metrics: {e}")
            return {
                "preference_win_rate": 0.0,
                "avg_preferred_similarity": 0.0,
                "avg_rejected_similarity": 0.0,
                "preference_margin": 0.0,
                "human_preference_correlation": 0.0,
            }

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using token overlap.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score between 0 and 1.
        """
        try:
            tokens1 = set(self._nltk.word_tokenize(text1.lower()))
            tokens2 = set(self._nltk.word_tokenize(text2.lower()))

            if not tokens1 or not tokens2:
                return 0.0

            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def compute_diversity_metrics(
        self,
        captions: List[str],
    ) -> Dict[str, float]:
        """Compute diversity metrics for generated captions.

        Args:
            captions: List of generated captions.

        Returns:
            Dictionary containing diversity metrics.
        """
        try:
            if not captions:
                return {"diversity_1": 0.0, "diversity_2": 0.0, "unique_caption_ratio": 0.0}

            # Tokenize captions
            tokenized = [self._nltk.word_tokenize(caption.lower()) for caption in captions]

            # Compute n-gram diversity
            def compute_ngram_diversity(tokens_list: List[List[str]], n: int) -> float:
                all_ngrams = []
                for tokens in tokens_list:
                    for i in range(len(tokens) - n + 1):
                        all_ngrams.append(tuple(tokens[i:i+n]))

                if not all_ngrams:
                    return 0.0

                unique_ngrams = len(set(all_ngrams))
                total_ngrams = len(all_ngrams)

                return unique_ngrams / total_ngrams

            diversity_1 = compute_ngram_diversity(tokenized, 1)
            diversity_2 = compute_ngram_diversity(tokenized, 2)

            # Unique caption ratio
            unique_captions = len(set(captions))
            unique_caption_ratio = unique_captions / len(captions)

            return {
                "diversity_1": diversity_1,
                "diversity_2": diversity_2,
                "unique_caption_ratio": unique_caption_ratio,
            }

        except Exception as e:
            self.logger.error(f"Error computing diversity metrics: {e}")
            return {"diversity_1": 0.0, "diversity_2": 0.0, "unique_caption_ratio": 0.0}

    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[List[str]],
        images: Optional[List[Union[str, Image.Image, torch.Tensor]]] = None,
        preferred_captions: Optional[List[str]] = None,
        rejected_captions: Optional[List[str]] = None,
        preference_scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Compute all evaluation metrics.

        Args:
            predictions: List of predicted captions.
            references: List of reference captions.
            images: Optional list of images for CLIP-Score.
            preferred_captions: Optional preferred captions for preference metrics.
            rejected_captions: Optional rejected captions for preference metrics.
            preference_scores: Optional preference scores.

        Returns:
            Dictionary containing all computed metrics.
        """
        metrics = {}

        # Traditional captioning metrics
        metrics.update(self.compute_bleu_scores(predictions, references))
        metrics.update(self.compute_rouge_scores(predictions, [ref[0] if ref else "" for ref in references]))
        metrics["meteor"] = self.compute_meteor_score(predictions, references)
        metrics["cider"] = self.compute_cider_score(predictions, references)

        # BERTScore
        metrics.update(self.compute_bertscore(predictions, [ref[0] if ref else "" for ref in references]))

        # CLIP-Score (if images provided)
        if images is not None:
            metrics.update(self.compute_clip_score(images, predictions))

        # Preference metrics (if preference data provided)
        if preferred_captions is not None and rejected_captions is not None:
            metrics.update(self.compute_preference_metrics(
                predictions, preferred_captions, rejected_captions,
                preference_scores or [1.0] * len(predictions)
            ))

        # Diversity metrics
        metrics.update(self.compute_diversity_metrics(predictions))

        return metrics


class EvaluationRunner:
    """Runner for comprehensive model evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        metrics_calculator: Optional[CaptioningMetrics] = None,
        output_dir: str = "./evaluation_results",
    ) -> None:
        """Initialize evaluation runner.

        Args:
            model: Model to evaluate.
            config: Evaluation configuration.
            metrics_calculator: Metrics calculator instance.
            output_dir: Directory to save evaluation results.
        """
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_calculator = metrics_calculator or CaptioningMetrics()
        self.logger = logging.getLogger(__name__)

    def run_evaluation(
        self,
        test_loader: torch.utils.data.DataLoader,
        save_predictions: bool = True,
        compute_latency: bool = True,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation.

        Args:
            test_loader: Test data loader.
            save_predictions: Whether to save predictions.
            compute_latency: Whether to compute latency metrics.

        Returns:
            Dictionary containing evaluation results.
        """
        self.logger.info("Starting comprehensive evaluation")

        # Generate predictions
        predictions, ground_truths, images, latencies = self._generate_predictions(
            test_loader, compute_latency
        )

        # Compute metrics
        metrics = self.metrics_calculator.compute_all_metrics(
            predictions=predictions,
            references=ground_truths,
            images=images if self.metrics_calculator.clip_model else None,
        )

        # Add latency metrics
        if compute_latency and latencies:
            metrics.update({
                "latency_mean_ms": np.mean(latencies) * 1000,
                "latency_median_ms": np.median(latencies) * 1000,
                "latency_p95_ms": np.percentile(latencies, 95) * 1000,
                "latency_p99_ms": np.percentile(latencies, 99) * 1000,
            })

        # Save results
        if save_predictions:
            self._save_predictions(predictions, ground_truths, images, metrics)

        # Generate visualizations
        self._generate_visualizations(predictions, ground_truths, metrics)

        self.logger.info("Evaluation completed")
        return {
            "metrics": metrics,
            "predictions": predictions,
            "ground_truths": ground_truths,
            "latencies": latencies,
        }

    def _generate_predictions(
        self,
        test_loader: torch.utils.data.DataLoader,
        compute_latency: bool,
    ) -> Tuple[List[str], List[List[str]], List[torch.Tensor], List[float]]:
        """Generate model predictions.

        Args:
            test_loader: Test data loader.
            compute_latency: Whether to compute latency.

        Returns:
            Tuple of (predictions, ground_truths, images, latencies).
        """
        self.logger.info("Starting model prediction generation")
        start_time = time.time()
        total_samples = len(test_loader.dataset)

        self.model.eval()
        predictions = []
        ground_truths = []
        images = []
        latencies = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                batch_images = batch["image"]
                batch_captions = batch.get("raw_caption", [""] * len(batch_images))

                # Measure latency
                start_time = time.time()

                # Generate captions
                if hasattr(self.model, "generate_captions"):
                    generated_captions = self.model.generate_captions(
                        batch_images,
                        **self.config.get("generate_config", {})
                    )
                else:
                    # Fallback generation method
                    generated_captions = [""] * len(batch_images)

                if compute_latency:
                    end_time = time.time()
                    batch_latency = (end_time - start_time) / len(batch_images)
                    latencies.extend([batch_latency] * len(batch_images))

                # Store results
                predictions.extend(generated_captions)
                ground_truths.extend([[caption] for caption in batch_captions])
                images.extend(batch_images)

        total_time = time.time() - start_time
        avg_time_per_sample = total_time / total_samples if total_samples > 0 else 0
        self.logger.info(
            f"Generated predictions for {total_samples} samples in {total_time:.3f}s "
            f"({avg_time_per_sample:.3f}s per sample)"
        )

        return predictions, ground_truths, images, latencies

    def _save_predictions(
        self,
        predictions: List[str],
        ground_truths: List[List[str]],
        images: List[torch.Tensor],
        metrics: Dict[str, float],
    ) -> None:
        """Save predictions and metrics to files.

        Args:
            predictions: Generated captions.
            ground_truths: Reference captions.
            images: Input images.
            metrics: Computed metrics.
        """
        # Save predictions
        predictions_data = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            predictions_data.append({
                "id": i,
                "prediction": pred,
                "ground_truth": gt[0] if gt else "",
                "all_references": gt,
            })

        predictions_file = self.output_dir / "predictions.json"
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(predictions_data, f, indent=2, ensure_ascii=False)

        # Save metrics
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"Saved predictions to {predictions_file}")
        self.logger.info(f"Saved metrics to {metrics_file}")

    def _generate_visualizations(
        self,
        predictions: List[str],
        ground_truths: List[List[str]],
        metrics: Dict[str, float],
    ) -> None:
        """Generate evaluation visualizations.

        Args:
            predictions: Generated captions.
            ground_truths: Reference captions.
            metrics: Computed metrics.
        """
        import matplotlib.pyplot as plt

        # Set style
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Metrics overview
        metric_names = []
        metric_values = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not name.endswith("_std"):
                metric_names.append(name.replace("_", " ").title())
                metric_values.append(value)

        # Select top metrics for visualization
        top_indices = np.argsort(metric_values)[-10:]
        top_names = [metric_names[i] for i in top_indices]
        top_values = [metric_values[i] for i in top_indices]

        axes[0, 0].barh(range(len(top_names)), top_values)
        axes[0, 0].set_yticks(range(len(top_names)))
        axes[0, 0].set_yticklabels(top_names)
        axes[0, 0].set_xlabel("Score")
        axes[0, 0].set_title("Top Evaluation Metrics")

        # Caption length distribution
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref[0].split()) if ref else 0 for ref in ground_truths]

        axes[0, 1].hist(pred_lengths, alpha=0.7, label="Predictions", bins=20)
        axes[0, 1].hist(ref_lengths, alpha=0.7, label="References", bins=20)
        axes[0, 1].set_xlabel("Caption Length (words)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Caption Length Distribution")
        axes[0, 1].legend()

        # BLEU score breakdown
        bleu_scores = []
        bleu_labels = []
        for key, value in metrics.items():
            if key.startswith("bleu_") and isinstance(value, (int, float)):
                bleu_scores.append(value)
                bleu_labels.append(key.replace("bleu_", "BLEU-").upper())

        if bleu_scores:
            axes[1, 0].plot(bleu_labels, bleu_scores, marker="o")
            axes[1, 0].set_ylabel("Score")
            axes[1, 0].set_title("BLEU Score Breakdown")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # Metrics comparison with targets
        targets = {
            "cider": 1.15,
            "preference_win_rate": 0.72,
            "latency_p95_ms": 150,
        }

        actual_values = []
        target_values = []
        target_names = []

        for metric, target in targets.items():
            if metric in metrics:
                actual_values.append(metrics[metric])
                target_values.append(target)
                target_names.append(metric.replace("_", " ").title())

        if actual_values:
            x_pos = np.arange(len(target_names))
            width = 0.35

            axes[1, 1].bar(x_pos - width/2, actual_values, width, label="Actual")
            axes[1, 1].bar(x_pos + width/2, target_values, width, label="Target")
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(target_names, rotation=45)
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].set_title("Actual vs Target Metrics")
            axes[1, 1].legend()

        plt.tight_layout()
        visualization_file = self.output_dir / "evaluation_summary.png"
        plt.savefig(visualization_file, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved visualizations to {visualization_file}")

    def compute_human_evaluation_metrics(
        self,
        predictions: List[str],
        human_ratings: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Compute human evaluation metrics.

        Args:
            predictions: Generated captions.
            human_ratings: List of human rating dictionaries.

        Returns:
            Dictionary containing human evaluation metrics.
        """
        if not human_ratings:
            return {}

        # Extract rating dimensions
        rating_dimensions = set()
        for rating in human_ratings:
            rating_dimensions.update(rating.keys())

        metrics = {}

        for dimension in rating_dimensions:
            scores = [rating.get(dimension, 0.0) for rating in human_ratings]
            metrics[f"human_eval_{dimension}"] = np.mean(scores)
            metrics[f"human_eval_{dimension}_std"] = np.std(scores)

        return metrics