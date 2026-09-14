"""
Category Classifier (Model A) for SmartHire.
Trains classical supervised ML models (Logistic Regression, Random Forest)
to predict job domain from Description + Skills.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import (
    CLASSIFIER_PATH,
    TFIDF_VECTORIZER_PATH,
    RANDOM_STATE,
    CATEGORIES,
)
from src.features.text_features import get_tfidf_vectorizer, load_vectorizer
from src.evaluate import evaluate_classifier


def train_category_classifier(
    df: pd.DataFrame,
    text_column: str = "composite_text",
    label_column: str = "Category",
    model_type: str = "logistic_regression",
    test_size: float = 0.2,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Train and evaluate a supervised classifier to predict category from text.
    """
    X_raw = df[text_column].fillna("")
    y = df[label_column]

    # Stratified Train-Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # TF-IDF Vectorization
    vectorizer = get_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # Model instantiation
    if model_type == "logistic_regression":
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    labels = sorted(list(y.unique()))
    metrics = evaluate_classifier(y_test, y_pred, labels=labels)

    if save_artifacts:
        CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, CLASSIFIER_PATH)
        joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "labels": labels,
    }


def predict_category(
    text: str,
    model=None,
    vectorizer=None,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict job category and confidence distribution for any text (e.g. candidate CV).
    Returns (top_category, confidence_score, all_probabilities).
    """
    if model is None:
        model = joblib.load(CLASSIFIER_PATH)
    if vectorizer is None:
        vectorizer = load_vectorizer(TFIDF_VECTORIZER_PATH)

    vec = vectorizer.transform([text])
    pred_cat = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(vec)[0]
        classes = model.classes_
        prob_dict = {classes[i]: round(float(probas[i]), 4) for i in range(len(classes))}
        confidence = round(float(max(probas)), 4)
    else:
        prob_dict = {pred_cat: 1.0}
        confidence = 1.0

    return pred_cat, confidence, prob_dict
