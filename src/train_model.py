"""
Training script for Titanic ML project.
Trains RandomForest and LogisticRegression using the preprocessing pipeline
from src.preprocessing.get_full_pipeline().
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.preprocessing import get_full_pipeline


# --------------------------------------
# Directories
# --------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "train.csv"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# --------------------------------------
# Load Data
# --------------------------------------
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {DATA_PATH}")

    print(f"Loading training data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if "Survived" not in df.columns:
        raise ValueError("train.csv must contain 'Survived' column")

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    return df, X, y


# --------------------------------------
# Cross-validation helper
# --------------------------------------
def evaluate_model(model, X_transformed, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc = cross_val_score(model, X_transformed, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(model, X_transformed, y, cv=cv, scoring="roc_auc")

    return acc.mean(), acc.std(), auc.mean(), auc.std()


# --------------------------------------
# Training Pipeline
# --------------------------------------
def train_and_evaluate(df):
    print("\nFitting preprocessing pipeline and transforming data...")

    pipeline = get_full_pipeline()
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_trans = pipeline.fit_transform(X)

    print(f"Transformed shape: {X_trans.shape}")

    # --------------------------------------
    # Evaluate Models
    # --------------------------------------
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    logreg = LogisticRegression(max_iter=500)

    print("\nEvaluating RandomForest with 5-fold stratified CV...")
    rf_acc, rf_sd, rf_auc, rf_auc_sd = evaluate_model(rf, X_trans, y)
    print(f"RandomForest Accuracy: {rf_acc*100:.2f}% ± {rf_sd*100:.2f}%")
    print(f"RandomForest ROC AUC: {rf_auc:.4f} ± {rf_auc_sd:.4f}")

    print("\nEvaluating LogisticRegression with 5-fold stratified CV...")
    lr_acc, lr_sd, lr_auc, lr_auc_sd = evaluate_model(logreg, X_trans, y)
    print(f"LogisticRegression Accuracy: {lr_acc*100:.2f}% ± {lr_sd*100:.2f}%")
    print(f"LogisticRegression ROC AUC: {lr_auc:.4f} ± {lr_auc_sd:.4f}")

    # --------------------------------------
    # Fit on full training data
    # --------------------------------------
    print("\nFitting models on full training data...")

    rf.fit(X_trans, y)
    logreg.fit(X_trans, y)

    # --------------------------------------
    # Save models + pipeline
    # --------------------------------------
    print("\nSaving models and preprocessing pipeline...")

    joblib.dump(pipeline, MODELS_DIR / "pipeline.joblib")
    joblib.dump(rf, MODELS_DIR / "rf.joblib")
    joblib.dump(logreg, MODELS_DIR / "logreg.joblib")

    print("Saved: pipeline.joblib, rf.joblib, logreg.joblib")

    # --------------------------------------
    # Generate confusion matrices + ROC curves
    # --------------------------------------
    print("\nGenerating evaluation plots...")

    for model, name in [(rf, "rf"), (logreg, "logreg")]:
        y_pred = model.predict(X_trans)
        y_proba = model.predict_proba(X_trans)[:, 1]

        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(REPORTS_DIR / f"confusion_matrix_{name}.png")
        plt.close()

        RocCurveDisplay.from_predictions(y, y_proba)
        plt.title(f"ROC Curve - {name}")
        plt.savefig(REPORTS_DIR / f"roc_curve_{name}.png")
        plt.close()

    # --------------------------------------
    # Write metrics summary
    # --------------------------------------
    summary = (
        f"RandomForest: Acc={rf_acc:.4f}, ROC_AUC={rf_auc:.4f}\n"
        f"LogisticRegression: Acc={lr_acc:.4f}, ROC_AUC={lr_auc:.4f}\n"
    )

    with open(REPORTS_DIR / "metrics.txt", "w") as f:
        f.write(summary)

    print("\nAll training complete.")
    print("Metrics saved → reports/metrics.txt")
    print("Confusion matrices + ROC curves saved → reports/")


# --------------------------------------
# Script Entry Point
# --------------------------------------
if __name__ == "__main__":
    df, X, y = load_data()
    train_and_evaluate(df)