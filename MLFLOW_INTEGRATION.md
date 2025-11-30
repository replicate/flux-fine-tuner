# MLflow Integration for Flux Fine-Tuner

This document describes the MLflow integration for tracking Flux LoRA training experiments.

## Overview

MLflow has been integrated alongside Weights & Biases to provide experiment tracking capabilities. You can use MLflow independently or alongside W&B to track your training runs.

## Features

The MLflow integration tracks:
- **Parameters**: All training hyperparameters (learning rate, batch size, steps, LoRA rank, etc.)
- **Metrics**: Training loss logged at each step
- **Artifacts**: 
  - Sample images generated during training
  - Final LoRA weights (.safetensors file)

## Setup

### 1. Start MLflow Tracking Server

To use MLflow, you need a tracking server. You can run one locally:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Or use a remote MLflow server if you have one deployed.

### 2. Configure Training Parameters

When running training, provide the MLflow parameters:

```python
mlflow_tracking_uri="http://localhost:5000"  # Your MLflow server URI
mlflow_experiment_name="flux-lora-training"  # Experiment name
mlflow_run_name="my-custom-run"              # Optional: specific run name
```

## Training Parameters

### MLflow-specific Parameters

- **mlflow_tracking_uri** (string, optional): MLflow tracking server URI
  - Example: `"http://localhost:5000"` or `"https://your-mlflow-server.com"`
  - Default: `None` (MLflow disabled)

- **mlflow_experiment_name** (string): Name of the MLflow experiment
  - Default: `"flux-lora-training"`
  - Only applicable if `mlflow_tracking_uri` is set

- **mlflow_run_name** (string, optional): Name for this specific run
  - Default: `None` (auto-generated)
  - Only applicable if `mlflow_tracking_uri` is set

## Usage Example

```python
from pathlib import Path

result = train(
    input_images=Path("my_images.zip"),
    trigger_word="TOK",
    steps=1000,
    learning_rate=4e-4,
    lora_rank=16,
    # MLflow configuration
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="my-flux-experiment",
    mlflow_run_name="run-001",
)
```

## Viewing Results

Once training starts, MLflow will print the URL to view the run:

```
MLflow tracking initialized. View at: http://localhost:5000/#/experiments/1/runs/abc123
```

Open this URL in your browser to:
- View real-time training metrics
- Compare different runs
- Download artifacts (samples and weights)
- Analyze parameter impact

## MLflow UI Features

### Metrics Tab
- View training loss curves over time
- Compare metrics across multiple runs

### Parameters Tab
- See all hyperparameters for the run
- Filter and sort runs by parameters

### Artifacts Tab
- Download sample images generated during training
- Access final LoRA weights
- View organized by training step

## Using with Weights & Biases

You can use both MLflow and W&B simultaneously:

```python
result = train(
    input_images=Path("my_images.zip"),
    # W&B configuration
    wandb_api_key=my_wandb_key,
    wandb_project="my-project",
    # MLflow configuration
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="my-flux-experiment",
)
```

Both systems will receive the same metrics and artifacts independently.

## Architecture

### MLflow Client (`mlflow_client.py`)

The `MLflowClient` class provides:
- Experiment and run management
- Parameter logging
- Metric logging with step tracking
- Artifact (images and weights) logging

### Integration Points

The integration hooks into:
1. **CustomSDTrainer.hook_train_loop()**: Logs training loss at each step
2. **CustomSDTrainer.sample()**: Logs generated sample images
3. **CustomSDTrainer.post_save_hook()**: Logs LoRA weights

## Troubleshooting

### Connection Issues

If you see connection errors:
- Verify the MLflow server is running
- Check the tracking URI is correct
- Ensure network connectivity to the server

### Missing Artifacts

If artifacts aren't appearing:
- Check disk space on the MLflow server
- Verify artifact storage is configured correctly
- Review server logs for errors

### Performance

MLflow logging is non-blocking and won't slow down training. However:
- Large sample image sets may take time to upload
- Consider reducing sample frequency for very frequent sampling

## Advanced Configuration

### Remote Artifact Storage

MLflow can store artifacts in S3, Azure Blob Storage, or other backends:

```bash
mlflow server \
  --backend-store-uri postgresql://user:pass@localhost/mlflow \
  --default-artifact-root s3://my-mlflow-bucket/ \
  --host 0.0.0.0
```

### Database Backend

For production use, configure a database backend:

```bash
mlflow server \
  --backend-store-uri postgresql://user:pass@localhost/mlflow \
  --host 0.0.0.0
```

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/mlflow.html)
