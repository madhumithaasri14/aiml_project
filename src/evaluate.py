"""
Evaluation metrics for SmartHire: Classification, Clustering, and Recommender.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)


def evaluate_classifier(y_true, y_pred, labels: List[str] = None) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }


def evaluate_clustering(X, cluster_labels) -> Dict[str, float]:
    """Calculate clustering validation metrics."""
    if len(set(cluster_labels)) > 1:
        sil = silhouette_score(X, cluster_labels)
    else:
        sil = 0.0
    return {
        "silhouette_score": round(float(sil), 4),
        "n_clusters": len(set(cluster_labels)),
    }


def evaluate_recommender_precision_at_k(recommended_categories: List[str], target_category: str, k: int = 5) -> float:
    """Compute Precision@K: fraction of top-K recommended jobs that match candidate target domain."""
    if not recommended_categories:
        return 0.0
    top_k = recommended_categories[:k]
    matches = sum(1 for cat in top_k if cat.strip().lower() == target_category.strip().lower())
    return round(matches / len(top_k), 4)
