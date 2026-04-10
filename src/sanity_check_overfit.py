from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def pick_classes(df: pd.DataFrame, k: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    classes = sorted(df["species"].unique().tolist())
    if len(classes) < k:
        raise ValueError(f"Need at least {k} classes, found {len(classes)}")
    return rng.choice(classes, size=k, replace=False).tolist()


def make_subset(df: pd.DataFrame, classes: List[str], per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for c in classes:
        d = df[df["species"] == c].copy()
        if len(d) < per_class:
            raise ValueError(f"Class '{c}' has only {len(d)} rows, need {per_class}.")
        idx = rng.choice(len(d), size=per_class, replace=False)
        parts.append(d.iloc[idx])
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def build_label_mapping(classes: List[str]) -> Tuple[dict, dict]:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def make_dataset(df: pd.DataFrame, class_to_idx: dict, img_size: int, batch_size: int) -> tf.data.Dataset:
    paths = df["local_path"].astype(str).values
    labels = df["species"].map(class_to_idx).astype(int).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (img_size, img_size), method="bilinear")
        img = tf.cast(img, tf.float32)
        # EfficientNet preprocessing (important!)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes: int, img_size: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    backbone.trainable = False  # SANITY CHECK: frozen

    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations_csv", type=str, default="data/meta/observations.csv")
    parser.add_argument("--k_classes", type=int, default=3)
    parser.add_argument("--per_class", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = read_csv_robust(Path(args.observations_csv))

    required = {"species", "local_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"observations.csv missing columns: {sorted(missing)}")

    # Drop missing files
    df["local_path"] = df["local_path"].astype(str)
    df = df[df["local_path"].apply(lambda p: Path(p).exists())].copy()

    classes = pick_classes(df, k=args.k_classes, seed=args.seed)
    sub = make_subset(df, classes=classes, per_class=args.per_class, seed=args.seed)

    class_to_idx, idx_to_class = build_label_mapping(classes)

    ds = make_dataset(sub, class_to_idx, img_size=args.img_size, batch_size=args.batch_size)

    model = build_model(num_classes=len(classes), img_size=args.img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    print("=== SANITY CHECK OVERFIT ===")
    print("Classes:", classes)
    print("Samples:", len(sub), f"({args.per_class} per class)")
    print("Goal: training acc should go very high (e.g. >0.90). If not, pipeline/input likely wrong.\n")

    hist = model.fit(ds, epochs=args.epochs, verbose=1)

    final_acc = float(hist.history["acc"][-1])
    print(f"\nFinal train acc: {final_acc:.4f}")
    if final_acc >= 0.90:
        print("✅ PASS: Model can overfit tiny dataset. Pipeline likely OK.")
    else:
        print("❌ FAIL: Model did NOT overfit. Likely an input/label/preprocessing issue.")


if __name__ == "__main__":
    main()