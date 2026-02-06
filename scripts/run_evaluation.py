#!/usr/bin/env python3
"""Evaluation script for preference-guided image captioning alignment."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import torch
from torch.utils.data import DataLoader

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preference_guided_image_captioning_alignment.data.loader import (
    ConceptualCaptionsDataset,
    UltraFeedbackDataset,
)
from preference_guided_image_captioning_alignment.data.preprocessing import (
    ImageProcessor,
    TextProcessor,
)
from preference_guided_image_captioning_alignment.evaluation.metrics import (
    CaptioningMetrics,
    EvaluationRunner,
)
from preference_guided_image_captioning_alignment.models.model import (
    PreferenceGuidedCaptioningModel,
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
            logging.FileHandler("evaluation.log"),
        ],
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Config,
    device: torch.device,
) -> PreferenceGuidedCaptioningModel:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: Configuration object.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    logger = logging.getLogger(__name__)

    # Create model
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

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        logger.info(f"Successfully loaded model from {checkpoint_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}")
        raise


def create_test_dataset(
    config: Config,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
    dataset_type: str = "conceptual",
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Create test dataset and data loader.

    Args:
        config: Configuration object.
        image_processor: Image preprocessing pipeline.
        text_processor: Text preprocessing pipeline.
        dataset_type: Type of dataset ('conceptual' or 'ultrafeedback').
        max_samples: Maximum number of samples to evaluate.

    Returns:
        Test data loader.
    """
    data_config = config.get_data_config()

    if dataset_type == "conceptual":
        dataset_path = data_config["conceptual_captions_path"]
        dataset_class = ConceptualCaptionsDataset
    elif dataset_type == "ultrafeedback":
        dataset_path = data_config["ultrafeedback_path"]
        dataset_class = UltraFeedbackDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Check if dataset exists
    if not Path(dataset_path).exists():
        logging.warning(f"Dataset path not found: {dataset_path}. Creating dummy dataset.")
        return create_dummy_test_dataloader(text_processor, max_samples or 50)

    # Create dataset
    dataset = dataset_class(
        data_path=dataset_path,
        image_processor=image_processor,
        text_processor=text_processor,
        split="test",
        max_samples=max_samples,
    )

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=8,  # Small batch size for evaluation
        shuffle=False,
        num_workers=data_config.get("num_workers", 2),
        pin_memory=data_config.get("pin_memory", True),
        drop_last=False,
    )

    return data_loader


def create_dummy_test_dataloader(
    text_processor: TextProcessor,
    num_samples: int = 50,
) -> DataLoader:
    """Create dummy test data loader for demonstration.

    Args:
        text_processor: Text processor.
        num_samples: Number of dummy samples.

    Returns:
        Dummy test data loader.
    """
    from torch.utils.data import Dataset, DataLoader

    class DummyTestDataset(Dataset):
        def __init__(self, size: int):
            self.size = size
            self.captions = [
                "A cat sitting on a windowsill",
                "A dog playing in the garden",
                "A beautiful mountain landscape",
                "People walking in a busy street",
                "A red sports car on the highway",
                "Children playing in the park",
                "A bird flying over the ocean",
                "A train passing through the countryside",
                "A chef cooking in the kitchen",
                "Students studying in the library",
            ]

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Create dummy image
            image = torch.randn(3, 224, 224)

            # Get caption
            caption = self.captions[idx % len(self.captions)]

            # Process caption
            caption_encoding = text_processor.encode_caption(caption)

            return {
                "image": image,
                "caption_ids": caption_encoding["input_ids"],
                "caption_mask": caption_encoding["attention_mask"],
                "raw_caption": caption,
                "image_path": f"dummy_test_image_{idx}.jpg",
            }

    dataset = DummyTestDataset(num_samples)
    return DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def run_comprehensive_evaluation(
    model: PreferenceGuidedCaptioningModel,
    test_loader: DataLoader,
    config: Config,
    output_dir: str,
) -> Dict[str, Any]:
    """Run comprehensive model evaluation.

    Args:
        model: Model to evaluate.
        test_loader: Test data loader.
        config: Configuration object.
        output_dir: Output directory for results.

    Returns:
        Evaluation results.
    """
    logger = logging.getLogger(__name__)

    # Create metrics calculator
    metrics_calculator = CaptioningMetrics(
        device=str(model.device) if hasattr(model, "device") else "auto",
        cache_dir=config.get("paths.cache_dir", "./cache"),
    )

    # Create evaluation runner
    evaluation_runner = EvaluationRunner(
        model=model,
        config=config.get_evaluation_config(),
        metrics_calculator=metrics_calculator,
        output_dir=output_dir,
    )

    logger.info("Starting comprehensive evaluation...")

    # Run evaluation
    results = evaluation_runner.run_evaluation(
        test_loader=test_loader,
        save_predictions=True,
        compute_latency=True,
    )

    logger.info("Evaluation completed!")

    # Log key metrics
    metrics = results["metrics"]
    logger.info("Key Metrics:")
    key_metrics = [
        "bleu_4", "rouge1", "rougeL", "meteor", "cider",
        "clip_score", "bertscore_f1", "diversity_1",
        "latency_p95_ms", "preference_win_rate"
    ]

    for metric in key_metrics:
        if metric in metrics:
            logger.info(f"  {metric}: {metrics[metric]:.4f}")

    return results


def compare_with_targets(
    metrics: Dict[str, float],
    config: Config,
) -> Dict[str, bool]:
    """Compare metrics with target values.

    Args:
        metrics: Computed metrics.
        config: Configuration with target values.

    Returns:
        Dictionary indicating which targets were met.
    """
    targets = config.get_targets()
    comparisons = {}

    for target_name, target_value in targets.items():
        if target_name in metrics:
            actual_value = metrics[target_name]

            # Determine if target is met based on metric type
            if target_name.endswith("_ms"):
                # Lower is better for latency
                target_met = actual_value <= target_value
            else:
                # Higher is better for most metrics
                target_met = actual_value >= target_value

            comparisons[target_name] = target_met

    return comparisons


def generate_evaluation_report(
    results: Dict[str, Any],
    config: Config,
    output_dir: str,
) -> None:
    """Generate comprehensive evaluation report.

    Args:
        results: Evaluation results.
        config: Configuration object.
        output_dir: Output directory.
    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_dir)

    # Compare with targets
    target_comparisons = compare_with_targets(results["metrics"], config)

    # Create report
    report = {
        "evaluation_summary": {
            "total_samples": len(results["predictions"]),
            "metrics": results["metrics"],
            "target_comparisons": target_comparisons,
        },
        "model_performance": {
            "caption_quality": {
                "bleu_scores": {
                    k: v for k, v in results["metrics"].items()
                    if k.startswith("bleu_")
                },
                "rouge_scores": {
                    k: v for k, v in results["metrics"].items()
                    if k.startswith("rouge")
                },
                "semantic_similarity": {
                    "meteor": results["metrics"].get("meteor", 0.0),
                    "cider": results["metrics"].get("cider", 0.0),
                    "bertscore_f1": results["metrics"].get("bertscore_f1", 0.0),
                },
                "multimodal_alignment": {
                    "clip_score": results["metrics"].get("clip_score", 0.0),
                },
            },
            "preference_alignment": {
                k: v for k, v in results["metrics"].items()
                if k.startswith("preference_") or k.startswith("human_")
            },
            "generation_quality": {
                k: v for k, v in results["metrics"].items()
                if k.startswith("diversity_")
            },
            "efficiency": {
                k: v for k, v in results["metrics"].items()
                if k.startswith("latency_")
            },
        },
        "target_achievement": {
            "met_targets": sum(target_comparisons.values()),
            "total_targets": len(target_comparisons),
            "target_details": target_comparisons,
        },
    }

    # Save report
    report_path = output_path / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation report saved to {report_path}")

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total samples evaluated: {report['evaluation_summary']['total_samples']}")
    logger.info(f"Targets met: {report['target_achievement']['met_targets']}/{report['target_achievement']['total_targets']}")

    logger.info("\nKey Performance Indicators:")
    for target, met in target_comparisons.items():
        status = "✓" if met else "✗"
        value = results["metrics"].get(target, "N/A")
        target_value = config.get_targets().get(target, "N/A")
        logger.info(f"  {status} {target}: {value:.4f} (target: {target_value})")

    logger.info("="*50)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate preference-guided captioning model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["conceptual", "ultrafeedback", "both"],
        default="conceptual",
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name for logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting preference-guided image captioning evaluation")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Dataset: {args.dataset}")

    try:
        # Setup MLflow if specified
        if args.mlflow_experiment:
            mlflow.set_experiment(args.mlflow_experiment)
            mlflow.start_run()

        # Load configuration
        config = Config(args.config)

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load model
        logger.info("Loading model...")
        model = load_model_from_checkpoint(args.checkpoint, config, device)

        # Create data processors
        logger.info("Creating data processors...")
        data_config = config.get_data_config()
        model_config = config.get_model_config()

        image_processor = ImageProcessor(
            image_size=data_config["image_size"],
            augment=False,  # No augmentation for evaluation
        )

        text_processor = TextProcessor(
            model_name=model_config["text_model"],
            max_length=data_config["max_caption_length"],
            padding="max_length",
            truncation=True,
        )

        # Run evaluation on specified datasets
        all_results = {}

        datasets_to_eval = []
        if args.dataset == "both":
            datasets_to_eval = ["conceptual", "ultrafeedback"]
        else:
            datasets_to_eval = [args.dataset]

        for dataset_name in datasets_to_eval:
            logger.info(f"Evaluating on {dataset_name} dataset...")

            # Create test dataset
            test_loader = create_test_dataset(
                config, image_processor, text_processor,
                dataset_type=dataset_name, max_samples=args.max_samples
            )

            logger.info(f"Test dataset size: {len(test_loader.dataset)}")

            # Run evaluation
            dataset_output_dir = Path(args.output_dir) / dataset_name
            dataset_output_dir.mkdir(parents=True, exist_ok=True)

            results = run_comprehensive_evaluation(
                model, test_loader, config, str(dataset_output_dir)
            )

            all_results[dataset_name] = results

            # Generate report for this dataset
            generate_evaluation_report(results, config, str(dataset_output_dir))

            # Log to MLflow
            if mlflow.active_run():
                for metric_name, metric_value in results["metrics"].items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(f"{dataset_name}_{metric_name}", metric_value)

        # Generate combined report if evaluating multiple datasets
        if len(all_results) > 1:
            logger.info("Generating combined evaluation report...")

            combined_metrics = {}
            for dataset_name, results in all_results.items():
                for metric_name, metric_value in results["metrics"].items():
                    combined_metrics[f"{dataset_name}_{metric_name}"] = metric_value

            combined_results = {"metrics": combined_metrics}
            combined_output_dir = Path(args.output_dir) / "combined"
            combined_output_dir.mkdir(parents=True, exist_ok=True)

            generate_evaluation_report(combined_results, config, str(combined_output_dir))

        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise
    finally:
        # Clean up MLflow
        if mlflow.active_run():
            mlflow.end_run()
        logger.info("Evaluation script completed")


if __name__ == "__main__":
    main()