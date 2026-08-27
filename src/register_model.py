import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from mlflow import MlflowClient


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.keras",
    )

    parser.add_argument(
        "--metrics_file",
        type=str,
        default="models/evaluation_metrics.json",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="FloraWatchClassifier",
    )

    parser.add_argument(
        "--alias",
        type=str,
        default="challenger",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_registry_uri("http://127.0.0.1:5000")

    model_path = Path(args.model_path)
    metrics_path = Path(args.metrics_file)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    model = tf.keras.models.load_model(model_path)

    img_size = model.input_shape[1]

    input_example = np.zeros(
        (1, img_size, img_size, 3),
        dtype=np.float32,
    )

    with mlflow.start_run():

        mlflow.log_metrics({
            "finetune_val_accuracy": metrics["finetune_val_accuracy"],
            "finetune_val_top3": metrics["finetune_val_top3"],
        })

        model_info = mlflow.tensorflow.log_model(
            model,
            name="model",
            input_example=input_example,
            registered_model_name=args.model_name,
        )

    client = MlflowClient()

    version = model_info.registered_model_version

    client.set_registered_model_alias(
        name=args.model_name,
        alias=args.alias,
        version=version,
    )

    client.set_model_version_tag(
        name=args.model_name,
        version=version,
        key="quality_gate",
        value="PASSED",
    )

    print("\n=== MLflow Model Registry ===")
    print(f"Model:   {args.model_name}")
    print(f"Version: {version}")
    print(f"Alias:   {args.alias}")
    print("Registered successfully ✅")


if __name__ == "__main__":
    main()