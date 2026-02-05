"""Configuration management utilities."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for the preference-guided captioning system.

    Handles loading and validation of YAML configuration files with support
    for environment variable overrides and nested parameter access.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file. If None, uses default.
        """
        self.logger = logging.getLogger(__name__)

        if config_path is None:
            config_path = self._get_default_config_path()

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._apply_env_overrides()

    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "configs",
            "default.yaml"
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError as e:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise e
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in configuration file: {e}")
            raise e

    def _validate_config(self) -> None:
        """Validate required configuration sections and parameters."""
        required_sections = ["data", "model", "training", "evaluation", "targets"]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate data configuration
        data_config = self.config["data"]
        required_data_keys = ["image_size", "max_caption_length", "num_workers"]
        for key in required_data_keys:
            if key not in data_config:
                raise ValueError(f"Missing required data config: {key}")

        # Validate model configuration
        model_config = self.config["model"]
        required_model_keys = ["vision_model", "text_model", "projection_dim"]
        for key in required_model_keys:
            if key not in model_config:
                raise ValueError(f"Missing required model config: {key}")

        # Validate training configuration
        training_config = self.config["training"]
        if "stage1" not in training_config or "stage2" not in training_config:
            raise ValueError("Training config must have stage1 and stage2 sections")

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Override configuration from environment variables
        env_overrides = {
            # Data paths
            "CONCEPTUAL_CAPTIONS_PATH": ["data", "conceptual_captions_path"],
            "ULTRAFEEDBACK_PATH": ["data", "ultrafeedback_path"],
            "CAPTION_ALIGNMENT_DATA_DIR": ["data", "conceptual_captions_path"],

            # Directory paths
            "OUTPUT_DIR": ["paths", "output_dir"],
            "CACHE_DIR": ["paths", "cache_dir"],
            "CAPTION_ALIGNMENT_CACHE_DIR": ["paths", "cache_dir"],
            "CAPTION_ALIGNMENT_OUTPUT_DIR": ["paths", "output_dir"],
            "CAPTION_ALIGNMENT_LOG_DIR": ["paths", "log_dir"],

            # Model configuration
            "CAPTION_ALIGNMENT_VISION_MODEL": ["model", "vision_model"],
            "CAPTION_ALIGNMENT_TEXT_MODEL": ["model", "text_model"],
            "CAPTION_ALIGNMENT_DEVICE": ["hardware", "device"],

            # Training configuration
            "CAPTION_ALIGNMENT_BATCH_SIZE": ["training", "stage1", "batch_size"],
            "CAPTION_ALIGNMENT_LEARNING_RATE": ["training", "stage1", "learning_rate"],
            "CAPTION_ALIGNMENT_NUM_EPOCHS": ["training", "stage1", "num_epochs"],
            "CAPTION_ALIGNMENT_LOG_LEVEL": ["logging", "level"],

            # Logging
            "WANDB_PROJECT": ["logging", "wandb_project"],
            "WANDB_ENTITY": ["logging", "wandb_entity"],
            "MLFLOW_EXPERIMENT": ["logging", "mlflow_experiment"],
            "MLFLOW_TRACKING_URI": ["logging", "mlflow_tracking_uri"],

            # Hardware
            "CAPTION_ALIGNMENT_NUM_WORKERS": ["data", "num_workers"],
            "CAPTION_ALIGNMENT_PIN_MEMORY": ["data", "pin_memory"],
            "CAPTION_ALIGNMENT_MIXED_PRECISION": ["hardware", "mixed_precision"],
        }

        for env_var, config_path in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Convert environment string values to appropriate types
                converted_value = self._convert_env_value(env_value)
                self._set_nested_config(config_path, converted_value)
                self.logger.info(f"Override from env {env_var}: {converted_value}")

    def _set_nested_config(self, path: list, value: Any) -> None:
        """Set nested configuration value.

        Args:
            path: List of keys to navigate to the configuration value.
            value: Value to set.
        """
        current = self.config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type.

        Args:
            value: String value from environment variable.

        Returns:
            Converted value with appropriate type.
        """
        # Handle boolean values
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        elif value.lower() in ("false", "0", "no", "off"):
            return False

        # Handle integer values
        try:
            if "." not in value and "e" not in value.lower():
                return int(value)
        except ValueError:
            pass

        # Handle float values
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string if no conversion applies
        return value

    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot notation path.

        Args:
            path: Dot-separated path to configuration value (e.g., "model.vision_model").
            default: Default value if path not found.

        Returns:
            Configuration value or default.
        """
        keys = path.split(".")
        current = self.config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, path: str, value: Any) -> None:
        """Set configuration value by dot notation path.

        Args:
            path: Dot-separated path to configuration value.
            value: Value to set.
        """
        keys = path.split(".")
        self._set_nested_config(keys, value)

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.config["data"]

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.config["model"]

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self.config["training"]

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration section."""
        return self.config["evaluation"]

    def get_targets(self) -> Dict[str, float]:
        """Get target metrics configuration."""
        return self.config["targets"]

    def get_stage1_config(self) -> Dict[str, Any]:
        """Get stage 1 training configuration."""
        return self.config["training"]["stage1"]

    def get_stage2_config(self) -> Dict[str, Any]:
        """Get stage 2 training configuration."""
        return self.config["training"]["stage2"]

    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to file.

        Args:
            path: Output path. If None, overwrites original config file.
        """
        output_path = path if path else self.config_path

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, indent=2)

        self.logger.info(f"Configuration saved to {output_path}")

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(path={self.config_path}, sections={list(self.config.keys())})"