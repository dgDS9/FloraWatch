from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight


def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def load_metadata(observations_csv: Path) -> pd.DataFrame:
    df = read_csv_robust(observations_csv)

    required = {"species", "local_path", "observation_id", "observer_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"observations.csv missing columns: {sorted(missing)}")

    df["species"] = df["species"].astype(str).str.strip()
    df["local_path"] = df["local_path"].astype(str)

    # Drop rows with missing/invalid files
    df = df[df["local_path"].apply(lambda p: Path(p).exists())].copy()

    # Group key to avoid leakage: primary observation_id, fallback observer_id
    df["group_id"] = df["observation_id"].fillna(df["observer_id"]).astype(str)

    return df


def stratified_group_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test:
      - stratified by label (species)
      - grouped by group_id (observation_id/observer_id)
    """
    y = df["species"].values
    groups = df["group_id"].values

    # Split off test
    n_splits_test = max(2, int(round(1 / test_size)))
    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    train_idx, test_idx = list(sgkf_test.split(df, y, groups))[0]

    df_trainval = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # Split trainval into train/val
    y_tv = df_trainval["species"].values
    groups_tv = df_trainval["group_id"].values

    val_rel = val_size / (1 - test_size)
    n_splits_val = max(2, int(round(1 / val_rel)))
    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed + 1)
    train2_idx, val_idx = list(sgkf_val.split(df_trainval, y_tv, groups_tv))[0]

    df_train = df_trainval.iloc[train2_idx].copy()
    df_val = df_trainval.iloc[val_idx].copy()

    return df_train, df_val, df_test


def build_label_mapping(df_train: pd.DataFrame):
    classes = sorted(df_train["species"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return classes, class_to_idx, idx_to_class


def make_dataset(
    df: pd.DataFrame,
    class_to_idx: dict,
    batch_size: int,
    img_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = df["local_path"].values
    labels = df["species"].map(class_to_idx).astype(int).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (img_size, img_size), method="bilinear")
        img = tf.cast(img, tf.float32)
        # IMPORTANT: EfficientNet preprocessing
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(2048, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes: int, img_size: int):
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))

    # Mild augmentation (safe starter)
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
        ],
        name="augmentation",
    )
    x = aug(inputs)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    backbone.trainable = False  # Phase 1

    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="plant_classifier_effnetb0")
    return model, backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations_csv", type=str, default="data/meta/observations.csv")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_head", type=int, default=15)
    parser.add_argument("--epochs_finetune", type=int, default=12)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    obs_path = Path(args.observations_csv)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(obs_path)
    df_train, df_val, df_test = stratified_group_split(df, seed=args.seed)

    classes, class_to_idx, idx_to_class = build_label_mapping(df_train)
    num_classes = len(classes)

    print(f"Classes used: {num_classes}")
    print("Train/Val/Test:", len(df_train), len(df_val), len(df_test))
    print("Per-class counts (train) min/max:",
          int(df_train['species'].value_counts().min()),
          int(df_train['species'].value_counts().max()))

    # Save label mapping for inference
    mapping_path = model_dir / "label_mapping.csv"
    pd.DataFrame(
        {"class_index": list(idx_to_class.keys()), "species": list(idx_to_class.values())}
    ).to_csv(mapping_path, index=False, encoding="utf-8")
    print(f"Saved label mapping: {mapping_path}")

    ds_train = make_dataset(df_train, class_to_idx, args.batch_size, args.img_size, training=True, seed=args.seed)
    ds_val = make_dataset(df_val, class_to_idx, args.batch_size, args.img_size, training=False, seed=args.seed)
    ds_test = make_dataset(df_test, class_to_idx, args.batch_size, args.img_size, training=False, seed=args.seed)

    # Class weights (helps mild imbalance)
    y_train = df_train["species"].map(class_to_idx).values
    cw = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(cw)}

    model, backbone = build_model(num_classes=num_classes, img_size=args.img_size)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor="val_acc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=2, min_lr=1e-6),
    ]

    # Phase 1: train head
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_head),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )

    print("\n=== Phase 1: Train head (backbone frozen) ===")
    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs_head,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # Phase 2: fine-tune last part of backbone
    print("\n=== Phase 2: Fine-tune (partial unfreeze) ===")
    backbone.trainable = True
    n_layers = len(backbone.layers)
    freeze_until = int(n_layers * 0.70)  # unfreeze last 30%
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= freeze_until

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_finetune),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs_finetune,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    print("\n=== Test evaluation ===")
    best = tf.keras.models.load_model(model_dir / "best_model.keras")
    metrics = best.evaluate(ds_test, return_dict=True)
    print(metrics)

    print(f"Saved best model: {model_dir / 'best_model.keras'}")
    print(f"Saved label mapping: {mapping_path}")


if __name__ == "__main__":
    main()