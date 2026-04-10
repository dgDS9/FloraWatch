from __future__ import annotations

import io
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


# ---------- Config ----------
BASE_DIR = Path(__file__).resolve().parents[2]   # geht von app/api/main.py zum Repo-Root
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
LABELS_PATH = BASE_DIR / "models" / "label_mapping.csv"
IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.60


# ---------- App ----------
app = FastAPI(title="Plant Recognition API", version="1.0")


# CORS: später für React anpassen (z.B. http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # im Produktivbetrieb einschränken!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: tf.keras.Model | None = None
labels: List[str] | None = None


def load_label_mapping(mapping_csv: Path) -> List[str]:
    df = pd.read_csv(mapping_csv, encoding="utf-8")
    if not {"class_index", "species"}.issubset(df.columns):
        raise ValueError("label_mapping.csv must contain columns: class_index, species")
    df = df.sort_values("class_index")
    return df["species"].astype(str).tolist()


def preprocess_pil_image(img: Image.Image, img_size: int) -> np.ndarray:
    # Ensure RGB
    img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype(np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr


def topk_from_probs(probs: np.ndarray, labels: List[str], k: int = 3) -> List[Dict[str, Any]]:
    topk_idx = np.argsort(probs)[::-1][:k]
    return [{"label": labels[i], "probability": float(probs[i])} for i in topk_idx]


@app.on_event("startup")
def startup_event() -> None:
    global model, labels
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not MAPPING_CSV.exists():
        raise FileNotFoundError(f"Label mapping not found: {MAPPING_CSV}")

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_label_mapping(MAPPING_CSV)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    """
    Accepts an image upload and returns:
      - prediction (Top-1 label or 'Unbekannte Pflanze')
      - unknown (bool)
      - top1
      - top3 (always included)
      - threshold
    """
    if model is None or labels is None:
        raise RuntimeError("Model not loaded")

    content = await file.read()
    img = Image.open(io.BytesIO(content))

    x = preprocess_pil_image(img, IMG_SIZE)
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)

    probs = model.predict(x, verbose=0)[0].astype(float)

    top3 = topk_from_probs(probs, labels, k=3)
    top1 = top3[0]
    is_unknown = top1["probability"] < threshold

    return {
        "filename": file.filename,
        "threshold": threshold,
        "unknown": bool(is_unknown),
        "prediction": "Unbekannte Pflanze" if is_unknown else top1["label"],
        "top1": top1,
        "top3": top3,  # always returned -> frontend can show collapsed
    }