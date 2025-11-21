from pathlib import Path
from typing import Any, Sequence
from contextlib import suppress
import mlflow
import mlflow.pytorch


class MLflowClient:
    """Client for tracking Flux LoRA training experiments with MLflow."""

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str | None,
        config: dict,
        sample_prompts: list[str],
    ):
        """Initialize MLflow tracking.

        Args:
            tracking_uri: MLflow tracking server URI (e.g., "http://localhost:5000")
            experiment_name: Name of the MLflow experiment
            run_name: Optional name for this specific run
            config: Training configuration dictionary to log as parameters
            sample_prompts: List of prompts used for sample generation
        """
        self.sample_prompts = sample_prompts
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Set or create experiment
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            raise ValueError(f"Failed to set MLflow experiment: {e}")

        # Start the run
        try:
            self.run = mlflow.start_run(run_name=run_name)
            # Log all config parameters
            mlflow.log_params(config)
            print(
                f"MLflow tracking initialized. View at: {self.tracking_uri}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{mlflow.active_run().info.run_id}"
            )
        except Exception as e:
            raise ValueError(f"Failed to start MLflow run: {e}")

    def log_loss(self, loss_dict: dict[str, Any], step: int | None):
        """Log training loss metrics.

        Args:
            loss_dict: Dictionary of loss values to log
            step: Training step number
        """
        try:
            mlflow.log_metrics(loss_dict, step=step)
        except Exception as e:
            print(f"Failed to log metrics to MLflow: {e}")

    def log_samples(self, image_paths: Sequence[Path], step: int | None):
        """Log generated sample images.

        Args:
            image_paths: List of paths to generated images
            step: Training step number
        """
        try:
            # Log each image with its corresponding prompt
            for prompt, path in zip(self.sample_prompts, image_paths):
                # Create a sanitized artifact name
                artifact_name = f"sample_step_{step}_{truncate(prompt, 30)}.jpg"
                mlflow.log_artifact(str(path), artifact_path=f"samples/step_{step}")
        except Exception as e:
            print(f"Failed to log samples to MLflow: {e}")

    def save_weights(self, lora_path: Path):
        """Save LoRA weights as an artifact.

        Args:
            lora_path: Path to the LoRA safetensors file
        """
        try:
            # Log the weights file as an artifact
            mlflow.log_artifact(str(lora_path), artifact_path="weights")
            print(f"Logged weights to MLflow: {lora_path.name}")
        except Exception as e:
            print(f"Failed to save weights to MLflow: {e}")

    def finish(self):
        """End the MLflow run."""
        with suppress(Exception):
            mlflow.end_run()
            print("MLflow run completed successfully")


def truncate(text, max_chars=50):
    """Truncate text to max_chars, adding ellipsis in the middle if needed.

    Args:
        text: Text to truncate
        max_chars: Maximum character length

    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    half = (max_chars - 3) // 2
    return f"{text[:half]}...{text[-half:]}"
