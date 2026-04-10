from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf


def load_label_mapping(mapping_csv: Path) -> List[str]:
    df = pd.read_csv(mapping_csv, encoding="utf-8")
    if not {"class_index", "species"}.issubset(df.columns):
        raise ValueError("label_mapping.csv must contain columns: class_index, species")
    df = df.sort_values("class_index")
    return df["species"].astype(str).tolist()


def preprocess_image(image_path: Path, img_size: int) -> np.ndarray:
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_size, img_size), method="bilinear")
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img.numpy()


def predict_topk(
    model: tf.keras.Model,
    labels: List[str],
    image_path: Path,
    img_size: int = 224,
    k: int = 3,
    threshold: float = 0.60,
) -> Dict[str, Any]:
    x = preprocess_image(image_path, img_size)
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)

    probs = model.predict(x, verbose=0)[0]  # (num_classes,)
    probs = probs.astype(float)

    topk_idx = np.argsort(probs)[::-1][:k]
    topk = [
        {"label": labels[i], "probability": float(probs[i])}
        for i in topk_idx
    ]

    top1 = topk[0]
    is_unknown = top1["probability"] < threshold

    result = {
        "image": str(image_path),
        "threshold": threshold,
        "unknown": bool(is_unknown),
        "top1": top1,
        "top3": topk,
    }

    # If unknown, set a friendly label
    if is_unknown:
        result["prediction"] = "Unbekannte Pflanze"
    else:
        result["prediction"] = top1["label"]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/best_model.keras")
    parser.add_argument("--mapping_csv", type=str, default="models/label_mapping.csv")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    labels = load_label_mapping(Path(args.mapping_csv))

    res = predict_topk(
        model=model,
        labels=labels,
        image_path=Path(args.image_path),
        img_size=args.img_size,
        k=3,
        threshold=args.threshold,
    )

    # Pretty print
    print("\n=== PREDICTION ===")
    print("Prediction:", res["prediction"])
    print("Unknown:", res["unknown"])
    print(f"Top-1: {res['top1']['label']}  ({res['top1']['probability']:.3f})")
    print("Top-3:")
    for i, item in enumerate(res["top3"], start=1):
        print(f"  {i}) {item['label']}: {item['probability']:.3f}")


if __name__ == "__main__":
    main()