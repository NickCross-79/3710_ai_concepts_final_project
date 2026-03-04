"""
ml/evaluate_models.py
=======================
Loads the trained models (from results/ml_models/) and the dataset
(from data/strategy_dataset.csv) to produce a deeper evaluation report.

Generates:
  results/ml_evaluation_report.csv  – unified model comparison table
  results/ml_roc_curves.png          – ROC curves for classifiers
  results/ml_pred_vs_actual.png      – predicted vs actual score scatter
  results/ml_feature_ranking.csv     – ranked feature importances

Can be run standalone (after train_models.py):
  python -m ml.evaluate_models
"""

import os
import sys
import csv
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics          import (roc_curve, auc, accuracy_score,
                                      confusion_matrix, r2_score,
                                      mean_absolute_error,
                                      classification_report)
from sklearn.preprocessing    import StandardScaler

from ml.train_models import FEATURE_NAMES, TEST_RATIO, RANDOM_SEED


# ---------------------------------------------------------------------------
# Loader utilities
# ---------------------------------------------------------------------------

def load_trained_models(models_dir):
    """Return dict of {model_name: fitted estimator} and the scaler."""
    models = {}
    scaler = None
    for fname in os.listdir(models_dir):
        if fname.endswith(".pkl") and fname != "scaler.pkl":
            name = fname.replace(".pkl", "").replace("_", " ").title()
            with open(os.path.join(models_dir, fname), "rb") as fh:
                models[name] = pickle.load(fh)
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    if os.path.isfile(scaler_path):
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
    return models, scaler


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(data_dir="data",
                   results_dir="results",
                   seed=RANDOM_SEED,
                   verbose=True):
    """
    Full evaluation of saved models.

    Returns
    -------
    report : list of dicts (one per model × metric)
    """
    models_dir = os.path.join(results_dir, "ml_models")

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(
            f"No trained models found at {models_dir}. "
            "Run ml/train_models.py first."
        )

    # Load data
    from ml.train_models import load_dataset
    X, y_score, y_label, df = load_dataset(data_dir)

    models, scaler = load_trained_models(models_dir)

    if verbose:
        print(f"\n===== ML Evaluation Report =====")
        print(f"  Models found   : {list(models.keys())}")
        print(f"  Dataset size   : {len(X)}")

    # Reproduce the same train/test split
    (X_tr_c, X_te_c,
     y_tr_c, y_te_c) = train_test_split(X, y_label,
                                         test_size=TEST_RATIO,
                                         random_state=seed,
                                         stratify=y_label)
    (X_tr_r, X_te_r,
     y_tr_r, y_te_r) = train_test_split(X, y_score,
                                         test_size=TEST_RATIO,
                                         random_state=seed)

    X_te_c_sc = scaler.transform(X_te_c) if scaler is not None else X_te_c

    report    = []
    roc_data  = {}    # for ROC plot
    reg_preds = {}    # for scatter plot
    feat_imps = {}    # model → importances array

    for name, estimator in models.items():
        is_classifier = hasattr(estimator, "predict_proba")
        is_regressor  = hasattr(estimator, "score") and not is_classifier

        # Decide which test data to use
        if "regressor" in name.lower():
            Xte, yte_true = X_te_r, y_te_r
            is_regression = True
        else:
            needs_scale = "logistic" in name.lower() or "mlp" in name.lower()
            Xte = X_te_c_sc if needs_scale else X_te_c
            yte_true = y_te_c
            is_regression = False

        y_pred = estimator.predict(Xte)

        if is_regression:
            r2  = r2_score(yte_true, y_pred)
            mae = mean_absolute_error(yte_true, y_pred)
            row = {"model": name, "type": "regression",
                   "r2": round(r2, 4), "mae": round(mae, 2),
                   "accuracy": "—", "auc": "—"}
            reg_preds[name] = (yte_true, y_pred)
            if verbose:
                print(f"  {name:<28} R²={r2:.4f}  MAE={mae:.2f}")
        else:
            acc = accuracy_score(yte_true, y_pred)
            try:
                y_prob = estimator.predict_proba(Xte)[:, 1]
                auc_v  = auc(*roc_curve(yte_true, y_prob)[:2])
                roc_data[name] = (y_prob, yte_true)
            except Exception:
                auc_v = float("nan")
            row = {"model": name, "type": "classification",
                   "accuracy": round(acc, 4),
                   "auc": round(auc_v, 4) if not _isnan(auc_v) else "n/a",
                   "r2": "—", "mae": "—"}
            if verbose:
                print(f"  {name:<28} acc={acc:.4f}  auc={str(auc_v)[:6]}")
                report_str = classification_report(
                    yte_true, y_pred,
                    target_names=["Bad", "Good"],
                    zero_division=0,
                )
                for line in report_str.splitlines():
                    print(f"    {line}")

        report.append(row)

        if hasattr(estimator, "feature_importances_"):
            feat_imps[name] = estimator.feature_importances_

    # Save report CSV
    os.makedirs(results_dir, exist_ok=True)
    _write_report(report, results_dir)

    # Feature importance ranking CSV
    if feat_imps:
        _save_feature_ranking(feat_imps, results_dir, verbose)

    # Plots
    if roc_data:
        _plot_roc_curves(roc_data, results_dir)
    if reg_preds:
        _plot_pred_vs_actual(reg_preds, results_dir)

    return report


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _isnan(v):
    import math
    try:
        return math.isnan(v)
    except TypeError:
        return False


def _write_report(report, results_dir):
    path = os.path.join(results_dir, "ml_evaluation_report.csv")
    fields = ["model", "type", "accuracy", "auc", "r2", "mae"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(report)
    print(f"\n  Saved evaluation report -> {path}")


def _save_feature_ranking(feat_imps, results_dir, verbose):
    """Average importances across models and rank them."""
    all_imp = np.stack(list(feat_imps.values()), axis=0)
    avg_imp = all_imp.mean(axis=0)

    rows = sorted(
        [{"feature": FEATURE_NAMES[i], "avg_importance": round(avg_imp[i], 6)}
         for i in range(len(FEATURE_NAMES))],
        key=lambda r: r["avg_importance"], reverse=True
    )
    path = os.path.join(results_dir, "ml_feature_ranking.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature", "avg_importance"])
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved feature ranking -> {path}")

    if verbose:
        print("\n  --- Top-10 Features (avg importance across tree models) ---")
        for r in rows[:10]:
            bar = "█" * int(r["avg_importance"] * 300)
            print(f"  {r['feature']:<28} {r['avg_importance']:.6f}  {bar}")


def _plot_roc_curves(roc_data, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_prob, y_true) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_v = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_v:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Classification Models")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(results_dir, "ml_roc_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved ROC curves -> {path}")


def _plot_pred_vs_actual(reg_preds, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(reg_preds)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, (y_true, y_pred)) in zip(axes, reg_preds.items()):
        ax.scatter(y_true, y_pred, alpha=0.3, s=15, color="steelblue")
        mn = min(min(y_true), min(y_pred))
        mx = max(max(y_true), max(y_pred))
        ax.plot([mn, mx], [mn, mx], "r--")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title(f"Pred vs Actual – {name}")
        ax.grid(True)
    plt.tight_layout()
    path = os.path.join(results_dir, "ml_pred_vs_actual.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved scatter plot -> {path}")


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation(seed=42, verbose=True)
