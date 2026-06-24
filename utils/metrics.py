"""
Evaluation metrics helpers for the DeepFake Detector.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels (0=real, 1=fake).
        y_pred_probs: Predicted probabilities (float, 0-1).
        threshold: Classification threshold.

    Returns:
        Dict with accuracy, auc, f1, precision, recall, confusion_matrix.
    """
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_true, y_pred_probs), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]),
    }

    return metrics
