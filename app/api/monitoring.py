from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List
from uuid import uuid4

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger("uvicorn.error")

_lock = Lock()
_predictions: Dict[str, Dict[str, Any]] = {}
_total_predictions = 0
_confidence_sum = 0.0
_unknown_count = 0
_class_distribution = Counter()


def log_prediction(
    prediction: str,
    top1: Dict[str, Any],
    top3: List[Dict[str, Any]],
    unknown: bool,
    threshold: float,
    model_version: str,
) -> str:
    global _total_predictions, _confidence_sum, _unknown_count

    prediction_id = str(uuid4())

    record = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction,
        "confidence": float(top1["probability"]),
        "top3": top3,
        "unknown": bool(unknown),
        "threshold": float(threshold),
        "model_version": model_version,
        "true_label": None,
    }

    with _lock:
        _predictions[prediction_id] = record
        _total_predictions += 1
        _confidence_sum += float(top1["probability"])

        if unknown:
            _unknown_count += 1

        _class_distribution[prediction] += 1

    logger.info(
        json.dumps(
            {
                "event": "model_prediction",
                **record,
            },
            ensure_ascii=False,
        )
    )

    return prediction_id


def log_feedback(
    prediction_id: str,
    true_label: str,
) -> None:
    with _lock:
        if prediction_id not in _predictions:
            raise KeyError(prediction_id)

        _predictions[prediction_id]["true_label"] = true_label
        prediction = _predictions[prediction_id]["prediction"]

    logger.info(
        json.dumps(
            {
                "event": "model_feedback",
                "prediction_id": prediction_id,
                "prediction": prediction,
                "true_label": true_label,
                "correct": prediction == true_label,
            },
            ensure_ascii=False,
        )
    )


def get_monitoring_summary() -> Dict[str, Any]:
    with _lock:
        total_predictions = _total_predictions

        average_confidence = (
            _confidence_sum / total_predictions
            if total_predictions > 0
            else 0.0
        )

        unknown_rate = (
            _unknown_count / total_predictions
            if total_predictions > 0
            else 0.0
        )

        feedback_records = [
            record.copy()
            for record in _predictions.values()
            if record["true_label"] is not None
        ]

        class_distribution = dict(_class_distribution)

    feedback_count = len(feedback_records)

    feedback_coverage = (
        feedback_count / total_predictions
        if total_predictions > 0
        else 0.0
    )

    if feedback_count == 0:
        return {
            "total_predictions": total_predictions,
            "average_confidence": average_confidence,
            "unknown_rate": unknown_rate,
            "class_distribution": class_distribution,
            "feedback_count": 0,
            "feedback_coverage": feedback_coverage,
            "production_accuracy": None,
            "production_top3_accuracy": None,
            "production_precision_macro": None,
            "production_recall_macro": None,
            "production_f1_macro": None,
        }

    y_true = [record["true_label"] for record in feedback_records]
    y_pred = [record["prediction"] for record in feedback_records]

    production_accuracy = accuracy_score(y_true, y_pred)

    production_precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    production_recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    production_f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    top3_correct = sum(
        1
        for record in feedback_records
        if record["true_label"]
        in [item["label"] for item in record["top3"]]
    )

    production_top3_accuracy = top3_correct / feedback_count

    return {
        "total_predictions": total_predictions,
        "average_confidence": average_confidence,
        "unknown_rate": unknown_rate,
        "class_distribution": class_distribution,
        "feedback_count": feedback_count,
        "feedback_coverage": feedback_coverage,
        "production_accuracy": float(production_accuracy),
        "production_top3_accuracy": float(production_top3_accuracy),
        "production_precision_macro": float(production_precision_macro),
        "production_recall_macro": float(production_recall_macro),
        "production_f1_macro": float(production_f1_macro),
    }
