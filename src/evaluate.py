from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold

import matplotlib.pyplot as plt


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
    df = df[df["local_path"].apply(lambda p: Path(p).exists())].copy()
    df["group_id"] = df["observation_id"].fillna(df["observer_id"]).astype(str)
    return df


def stratified_group_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["species"].values
    groups = df["group_id"].values

    n_splits_test = max(2, int(round(1 / test_size)))
    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    train_idx, test_idx = list(sgkf_test.split(df, y, groups))[0]

    df_trainval = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    y_tv = df_trainval["species"].values
    groups_tv = df_trainval["group_id"].values

    val_rel = val_size / (1 - test_size)
    n_splits_val = max(2, int(round(1 / val_rel)))
    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed + 1)
    train2_idx, val_idx = list(sgkf_val.split(df_trainval, y_tv, groups_tv))[0]

    df_train = df_trainval.iloc[train2_idx].copy()
    df_val = df_trainval.iloc[val_idx].copy()

    return df_train, df_val, df_test


def load_label_mapping(mapping_csv: Path) -> List[str]:
    m = read_csv_robust(mapping_csv)
    if not {"class_index", "species"}.issubset(m.columns):
        raise ValueError("label_mapping.csv must contain columns: class_index, species")
    m = m.sort_values("class_index")
    return m["species"].astype(str).tolist()


def make_dataset(df: pd.DataFrame, class_to_idx: dict, img_size: int, batch_size: int) -> tf.data.Dataset:
    paths = df["local_path"].values
    labels = df["species"].map(class_to_idx).astype(int).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (img_size, img_size), method="bilinear")
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_png: Path, normalize: bool = True) -> None:
    cm_plot = cm.astype(float)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, np.maximum(row_sums, 1.0))

    plt.figure(figsize=(12, 10))
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=90, fontsize=7)
    plt.yticks(ticks, labels, fontsize=7)
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def most_confused_pairs(cm: np.ndarray, labels: List[str], top_n: int = 10) -> pd.DataFrame:
    # off-diagonal counts
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    pairs = []
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            if cm2[i, j] > 0:
                pairs.append((labels[i], labels[j], int(cm2[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[:top_n]
    return pd.DataFrame(top, columns=["true_label", "pred_label", "count"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations_csv", type=str, default="data/meta/observations.csv")
    ap.add_argument("--model_path", type=str, default="models/best_model.keras")
    ap.add_argument("--mapping_csv", type=str, default="models/label_mapping.csv")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_n", type=int, default=15)
    ap.add_argument("--out_dir", type=str, default="models/eval")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(Path(args.observations_csv))
    df_train, df_val, df_test = stratified_group_split(df, seed=args.seed)

    labels = load_label_mapping(Path(args.mapping_csv))
    class_to_idx = {c: i for i, c in enumerate(labels)}

    # Ensure test labels match mapping
    unknown = set(df_test["species"].unique()) - set(labels)
    if unknown:
        raise ValueError(f"Test contains labels not in label_mapping.csv: {sorted(unknown)}")

    ds_test = make_dataset(df_test, class_to_idx, args.img_size, args.batch_size)
    model = tf.keras.models.load_model(args.model_path)

    # Predict
    probs = model.predict(ds_test, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    # True labels (collected from dataset order)
    y_true = np.concatenate([y.numpy() for _, y in ds_test], axis=0)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    cm_csv = out_dir / "confusion_matrix_counts.csv"
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_csv, encoding="utf-8")

    # plots
    save_confusion_matrix(cm, labels, out_dir / "confusion_matrix_normalized.png", normalize=True)
    save_confusion_matrix(cm, labels, out_dir / "confusion_matrix_counts.png", normalize=False)

    # most confused pairs
    top_pairs = most_confused_pairs(cm, labels, top_n=args.top_n)
    top_pairs_path = out_dir / "most_confused_pairs.csv"
    top_pairs.to_csv(top_pairs_path, index=False, encoding="utf-8")

    # classification report
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    print("\n=== Saved ===")
    print(cm_csv)
    print(out_dir / "confusion_matrix_normalized.png")
    print(out_dir / "confusion_matrix_counts.png")
    print(top_pairs_path)
    print(out_dir / "classification_report.txt")

    print("\n=== Most confused pairs (top) ===")
    print(top_pairs.to_string(index=False))


if __name__ == "__main__":
    main()