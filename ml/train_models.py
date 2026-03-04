"""
ml/train_models.py
====================
Builds the strategy dataset, trains classification and regression models,
saves everything to disk, and prints a performance summary.

Dataset construction
--------------------
Each sample is one IPD strategy (memory-3, 67 bits) enriched with
engineered features.  Labels are:
  score  – total tournament score (regression)
  label  – 1 if score >= median (classification: "good" vs "bad")

Sources of strategies
~~~~~~~~~~~~~~~~~~~~~
  • Random strategies (random_count)
  • Evolved strategies from short GA runs (evolved_ga_runs × ga_pop_size)

The combined dataset is saved to  data/strategy_dataset.csv.

Models trained
--------------
Classification: Logistic Regression, Random Forest, Gradient Boosting,
                MLP (optional, controlled by include_mlp flag)
Regression    : Random Forest, Gradient Boosting

Evaluation
----------
  • 80/20 stratified train/test split
  • 5-fold cross-validation on the training split
  • Accuracy, ROC-AUC (classification)
  • R², MAE (regression)
  • Confusion matrices
  • Feature importances (tree-based models)

Outputs
-------
  data/strategy_dataset.csv
  results/ml_classification_results.csv
  results/ml_regression_results.csv
  results/ml_confusion_matrix_<model>.png
  results/ml_feature_importance_<model>.png
  results/ml_models/          – pickled trained models

Run standalone:
  python -m ml.train_models
"""

import os
import sys
import csv
import pickle
import random
import statistics
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics       import (accuracy_score, roc_auc_score,
                                   confusion_matrix, r2_score,
                                   mean_absolute_error, classification_report)
from sklearn.preprocessing import StandardScaler

from strats import random_strategy, mutate, copy_strategy
from prisoner_dilemma import round_robin_score
from strategies.encoding import extract_features

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MEMORY_DEPTH     = 3
STRATEGY_LEN     = 67        # 3 + 64

RANDOM_COUNT     = 300       # random strategies in dataset
EVOLVED_GA_RUNS  = 20        # short GA runs to harvest evolved strategies
GA_GENERATIONS   = 10        # generations per short GA run
GA_POP_SIZE      = 20        # population per GA run
GA_MUTATION_RATE = 0.1
EVAL_ROUNDS      = 50        # rounds per match during dataset scoring
N_RANDOM_OPPS    = 30

CV_FOLDS         = 5
TEST_RATIO       = 0.20
RANDOM_SEED      = 42


# ---------------------------------------------------------------------------
# Feature / column definitions
# ---------------------------------------------------------------------------

def _feature_names():
    names  = [f"open_{i}"  for i in range(3)]          # opening bits
    names += [f"bit_{i:02d}" for i in range(64)]        # table bits
    names += [                                           # engineered
        "opening_coop_ratio",
        "table_coop_ratio",
        "coop_after_i_coop",
        "coop_after_i_defect",
        "coop_after_opp_coop",
        "coop_after_opp_defect",
        "coop_after_mutual_coop",
        "coop_after_mutual_def",
    ]
    return names


FEATURE_NAMES = _feature_names()


# ---------------------------------------------------------------------------
# Dataset construction helpers
# ---------------------------------------------------------------------------

def _strategy_to_row(strategy):
    """Return ordered feature list for one strategy."""
    eng = extract_features(strategy, memory_depth=MEMORY_DEPTH)
    row  = strategy[:3]               # opening bits
    row += strategy[3:]               # table bits
    row += [
        eng["opening_coop_ratio"],
        eng["table_coop_ratio"],
        eng["coop_after_i_coop"],
        eng["coop_after_i_defect"],
        eng["coop_after_opp_coop"],
        eng["coop_after_opp_defect"],
        eng["coop_after_mutual_coop"],
        eng["coop_after_mutual_def"],
    ]
    return row


def _build_opponents(seed=None):
    if seed is not None:
        random.seed(seed)
    from strats import (always_cooperate, always_defect,
                        tit_for_tat, suspicious_tit_for_tat)
    pool = [always_cooperate(), always_defect(),
            tit_for_tat(), suspicious_tit_for_tat()]
    for _ in range(N_RANDOM_OPPS):
        pool.append(random_strategy())
    return pool


def _crossover(a, b):
    pt = random.randint(1, len(a) - 1)
    return a[:pt] + b[pt:]


def _short_ga(opponents, generations=GA_GENERATIONS,
              pop_size=GA_POP_SIZE, mutation_rate=GA_MUTATION_RATE,
              rounds=EVAL_ROUNDS):
    """Run a short GA and return the entire final population."""
    population = [random_strategy() for _ in range(pop_size)]
    for _ in range(generations):
        scored = [(s, round_robin_score(s, opponents, rounds))
                  for s in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        survivors  = [s for s, _ in scored[:pop_size // 2]]
        next_gen   = survivors[:]
        while len(next_gen) < pop_size:
            child = _crossover(random.choice(survivors),
                                random.choice(survivors))
            child = mutate(child, mutation_rate)
            next_gen.append(child)
        population = next_gen
    return population


# ---------------------------------------------------------------------------
# Build (or load) dataset
# ---------------------------------------------------------------------------

def build_dataset(random_count=RANDOM_COUNT,
                  evolved_ga_runs=EVOLVED_GA_RUNS,
                  ga_generations=GA_GENERATIONS,
                  ga_pop_size=GA_POP_SIZE,
                  eval_rounds=EVAL_ROUNDS,
                  data_dir="data",
                  seed=None,
                  verbose=True):
    """
    Construct the strategy dataset and save it as a CSV.

    Returns (X, y_score, y_label, df).
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "strategy_dataset.csv")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    opponents = _build_opponents(seed=seed)

    strategies  = []
    origins     = []       # "random" or "evolved"
    scores_raw  = []

    if verbose:
        print(f"\n===== Building ML Dataset =====")
        print(f"  Random strategies  : {random_count}")
        print(f"  GA runs (evolved)  : {evolved_ga_runs} × pop={ga_pop_size}")
        print(f"  Eval rounds/match  : {eval_rounds}")
        print(f"  Opponents          : {len(opponents)}")

    # ── Random strategies ─────────────────────────────────────────────────
    if verbose:
        print(f"  Generating {random_count} random strategies …")
    for _ in range(random_count):
        s = random_strategy()
        strategies.append(s)
        origins.append("random")

    # ── Evolved strategies ─────────────────────────────────────────────────
    evolved_total = 0
    if verbose:
        print(f"  Running {evolved_ga_runs} short GAs …")
    for run in range(evolved_ga_runs):
        run_seed = (seed + run * 37) if seed is not None else None
        if run_seed is not None:
            random.seed(run_seed)
        pop = _short_ga(opponents, generations=ga_generations,
                        pop_size=ga_pop_size, rounds=eval_rounds)
        for s in pop:
            strategies.append(s)
            origins.append("evolved")
            evolved_total += 1
        if verbose and (run + 1) % 5 == 0:
            print(f"    GA runs done: {run+1}/{evolved_ga_runs}")

    if verbose:
        print(f"  Total strategies: {len(strategies)} "
              f"({random_count} random + {evolved_total} evolved)")
        print("  Evaluating all strategies …")

    # ── Evaluate ALL strategies ───────────────────────────────────────────
    for i, s in enumerate(strategies):
        sc = round_robin_score(s, opponents, eval_rounds)
        scores_raw.append(sc)
        if verbose and (i + 1) % 100 == 0:
            print(f"    Evaluated {i+1}/{len(strategies)}")

    # ── Build labels ──────────────────────────────────────────────────────
    median_score = statistics.median(scores_raw)
    labels = [1 if sc >= median_score else 0 for sc in scores_raw]

    # ── Assemble DataFrame ────────────────────────────────────────────────
    rows = []
    for strat, sc, lbl, orig in zip(strategies, scores_raw, labels, origins):
        row = _strategy_to_row(strat)
        rows.append(row + [float(sc), int(lbl), orig])

    col_names = FEATURE_NAMES + ["score", "label", "origin"]
    df = pd.DataFrame(rows, columns=col_names)
    df.to_csv(csv_path, index=False)

    if verbose:
        print(f"  Dataset saved -> {csv_path}")
        print(f"  Rows: {len(df)}  |  Good (label=1): "
              f"{int(df['label'].sum())}  |  Bad: "
              f"{len(df)-int(df['label'].sum())}")

    X       = df[FEATURE_NAMES].values.astype(float)
    y_score = df["score"].values.astype(float)
    y_label = df["label"].values.astype(int)

    return X, y_score, y_label, df


def load_dataset(data_dir="data"):
    """Load a previously saved dataset CSV."""
    csv_path = os.path.join(data_dir, "strategy_dataset.csv")
    df = pd.read_csv(csv_path)
    X       = df[FEATURE_NAMES].values.astype(float)
    y_score = df["score"].values.astype(float)
    y_label = df["label"].values.astype(int)
    return X, y_score, y_label, df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(X, y_score, y_label,
                 include_mlp=True,
                 test_ratio=TEST_RATIO,
                 cv_folds=CV_FOLDS,
                 results_dir="results",
                 models_dir=None,
                 seed=None,
                 verbose=True):
    """
    Train classification and regression models on the dataset.

    Returns
    -------
    cls_results  : list of dicts (classification)
    reg_results  : list of dicts (regression)
    trained      : dict  name → fitted estimator
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    os.makedirs(results_dir, exist_ok=True)
    models_dir = models_dir or os.path.join(results_dir, "ml_models")
    os.makedirs(models_dir, exist_ok=True)

    if verbose:
        print(f"\n===== Training ML Models =====")
        print(f"  Samples    : {len(X)}")
        print(f"  Features   : {X.shape[1]}")
        print(f"  Test ratio : {test_ratio}")
        print(f"  CV folds   : {cv_folds}")

    # ── Train/test split ──────────────────────────────────────────────────
    (X_tr_c, X_te_c,
     y_tr_c, y_te_c) = train_test_split(X, y_label,
                                         test_size=test_ratio,
                                         random_state=seed,
                                         stratify=y_label)
    (X_tr_r, X_te_r,
     y_tr_r, y_te_r) = train_test_split(X, y_score,
                                         test_size=test_ratio,
                                         random_state=seed)

    # Scale for LR / MLP
    scaler = StandardScaler()
    X_tr_c_sc = scaler.fit_transform(X_tr_c)
    X_te_c_sc = scaler.transform(X_te_c)

    # ── Classification models ─────────────────────────────────────────────
    cls_specs = [
        ("Logistic Regression",
         LogisticRegression(max_iter=2000, random_state=seed),
         True),       # True = use scaled features
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, random_state=seed),
         False),
        ("Gradient Boosting",
         GradientBoostingClassifier(n_estimators=200, random_state=seed,
                                    max_depth=4, learning_rate=0.05),
         False),
    ]
    if include_mlp:
        cls_specs.append((
            "MLP",
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                          random_state=seed, early_stopping=True,
                          validation_fraction=0.1),
            True,
        ))

    # ── Regression models ─────────────────────────────────────────────────
    reg_specs = [
        ("RF Regressor",
         RandomForestRegressor(n_estimators=200, random_state=seed)),
        ("GB Regressor",
         GradientBoostingRegressor(n_estimators=200, random_state=seed,
                                   max_depth=4, learning_rate=0.05)),
    ]

    cls_results = []
    reg_results = []
    trained     = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    kf  = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    if verbose:
        print("\n  [Classification]")

    for name, clf, use_scaled in cls_specs:
        if verbose:
            print(f"    Fitting {name} …", end=" ", flush=True)
        t0 = time.time()

        Xtr = X_tr_c_sc if use_scaled else X_tr_c
        Xte = X_te_c_sc if use_scaled else X_te_c
        Xfull_sc = scaler.transform(X) if use_scaled else X

        # CV on training set
        cv_acc = cross_val_score(clf, Xtr, y_tr_c, cv=skf,
                                  scoring="accuracy")
        cv_auc = cross_val_score(clf, Xtr, y_tr_c, cv=skf,
                                  scoring="roc_auc")

        clf.fit(Xtr, y_tr_c)
        y_pred = clf.predict(Xte)
        try:
            y_prob = clf.predict_proba(Xte)[:, 1]
            auc_te = roc_auc_score(y_te_c, y_prob)
        except Exception:
            auc_te = float("nan")

        acc_te = accuracy_score(y_te_c, y_pred)
        cm     = confusion_matrix(y_te_c, y_pred)

        elapsed = time.time() - t0
        if verbose:
            print(f"done ({elapsed:.1f}s)  acc={acc_te:.4f}  "
                  f"cv_acc={cv_acc.mean():.4f}±{cv_acc.std():.4f}")

        row = {
            "model":        name,
            "test_acc":     round(acc_te, 4),
            "test_auc":     round(auc_te, 4) if not _isnan(auc_te) else "n/a",
            "cv_acc_mean":  round(float(cv_acc.mean()), 4),
            "cv_acc_std":   round(float(cv_acc.std()), 4),
            "cv_auc_mean":  round(float(cv_auc.mean()), 4),
            "cv_auc_std":   round(float(cv_auc.std()), 4),
        }
        cls_results.append(row)
        trained[name] = clf

        _save_confusion_matrix(cm, name, results_dir)
        if hasattr(clf, "feature_importances_"):
            _save_feature_importance(clf.feature_importances_,
                                     FEATURE_NAMES, name, results_dir)

    # Save classification results CSV
    _write_csv(cls_results,
               ["model","test_acc","test_auc",
                "cv_acc_mean","cv_acc_std","cv_auc_mean","cv_auc_std"],
               os.path.join(results_dir, "ml_classification_results.csv"))

    if verbose:
        print("\n  [Regression]")

    for name, reg in reg_specs:
        if verbose:
            print(f"    Fitting {name} …", end=" ", flush=True)
        t0 = time.time()

        cv_r2  = cross_val_score(reg, X_tr_r, y_tr_r, cv=kf,
                                  scoring="r2")
        cv_mae = cross_val_score(reg, X_tr_r, y_tr_r, cv=kf,
                                  scoring="neg_mean_absolute_error")

        reg.fit(X_tr_r, y_tr_r)
        y_pred = reg.predict(X_te_r)
        r2_te  = r2_score(y_te_r, y_pred)
        mae_te = mean_absolute_error(y_te_r, y_pred)

        elapsed = time.time() - t0
        if verbose:
            print(f"done ({elapsed:.1f}s)  R²={r2_te:.4f}  "
                  f"cv_R²={cv_r2.mean():.4f}±{cv_r2.std():.4f}")

        row = {
            "model":       name,
            "test_r2":     round(r2_te, 4),
            "test_mae":    round(mae_te, 2),
            "cv_r2_mean":  round(float(cv_r2.mean()), 4),
            "cv_r2_std":   round(float(cv_r2.std()), 4),
            "cv_mae_mean": round(float(-cv_mae.mean()), 2),
            "cv_mae_std":  round(float(cv_mae.std()), 2),
        }
        reg_results.append(row)
        trained[name] = reg

        if hasattr(reg, "feature_importances_"):
            _save_feature_importance(reg.feature_importances_,
                                     FEATURE_NAMES, name, results_dir)

    _write_csv(reg_results,
               ["model","test_r2","test_mae",
                "cv_r2_mean","cv_r2_std","cv_mae_mean","cv_mae_std"],
               os.path.join(results_dir, "ml_regression_results.csv"))

    # Pickle all models + scaler
    for name, estimator in trained.items():
        safe_name = name.replace(" ", "_").lower()
        path = os.path.join(models_dir, f"{safe_name}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(estimator, fh)
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)

    if verbose:
        print(f"\n  Models saved -> {models_dir}/")
        _print_summary(cls_results, reg_results)

    return cls_results, reg_results, trained


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _isnan(v):
    import math
    try:
        return math.isnan(v)
    except TypeError:
        return False


def _write_csv(rows, fieldnames, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved -> {path}")


def _save_confusion_matrix(cm, name, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Pred Bad", "Pred Good"],
           yticklabels=["True Bad", "True Good"],
           title=f"Confusion Matrix – {name}",
           ylabel="True label", xlabel="Predicted label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    safe = name.replace(" ", "_").lower()
    path = os.path.join(results_dir, f"ml_confusion_matrix_{safe}.png")
    plt.savefig(path, dpi=110)
    plt.close()


def _save_feature_importance(importances, names, model_name, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    indices = sorted(range(len(importances)),
                     key=lambda i: importances[i], reverse=True)[:20]
    top_imp  = [importances[i] for i in indices]
    top_names = [names[i]       for i in indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(top_imp))[::-1], top_imp, color="steelblue",
            edgecolor="black")
    ax.set_yticks(range(len(top_imp))[::-1])
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-20 Feature Importances – {model_name}")
    ax.grid(axis="x")
    plt.tight_layout()
    safe = model_name.replace(" ", "_").lower()
    path = os.path.join(results_dir, f"ml_feature_importance_{safe}.png")
    plt.savefig(path, dpi=110)
    plt.close()


def _print_summary(cls_results, reg_results):
    print("\n--- Classification Results ---")
    hdr = (f"{'Model':<22} {'TestAcc':>8} {'TestAUC':>8} "
           f"{'CV_Acc':>10} {'CV_AUC':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in cls_results:
        print(f"{r['model']:<22} {str(r['test_acc']):>8} "
              f"{str(r['test_auc']):>8} "
              f"{r['cv_acc_mean']:.4f}±{r['cv_acc_std']:.4f} "
              f"{r['cv_auc_mean']:.4f}±{r['cv_auc_std']:.4f}")

    print("\n--- Regression Results ---")
    hdr2 = (f"{'Model':<22} {'TestR2':>8} {'TestMAE':>8} "
            f"{'CV_R2':>10}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in reg_results:
        print(f"{r['model']:<22} {r['test_r2']:>8} {r['test_mae']:>8} "
              f"{r['cv_r2_mean']:.4f}±{r['cv_r2_std']:.4f}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def run_study(random_count=RANDOM_COUNT,
              evolved_ga_runs=EVOLVED_GA_RUNS,
              include_mlp=True,
              data_dir="data",
              results_dir="results",
              seed=RANDOM_SEED,
              verbose=True):
    """Build dataset, train all models, save everything."""
    X, y_score, y_label, df = build_dataset(
        random_count=random_count,
        evolved_ga_runs=evolved_ga_runs,
        data_dir=data_dir,
        seed=seed,
        verbose=verbose,
    )
    cls_res, reg_res, trained = train_models(
        X, y_score, y_label,
        include_mlp=include_mlp,
        results_dir=results_dir,
        seed=seed,
        verbose=verbose,
    )
    return cls_res, reg_res, trained, df


if __name__ == "__main__":
    run_study(seed=42, verbose=True)
