"""
analysis/pattern_extraction.py
=================================
Extracts behavioural patterns that distinguish high-performing strategies
from low-performing ones.

Definitions:
  Top strategies    – top 20% by tournament score
  Bottom strategies – bottom 20% by tournament score

Analysis:
  1. Comparison of feature means (top vs bottom)
  2. Mean-difference ranking to identify most discriminating features
  3. Opening-move frequency analysis
  4. Per-state cooperation difference heatmap
  5. Human-readable rule generation

Outputs:
  results/pattern_feature_comparison.csv
  results/pattern_top20_features.png
  results/pattern_state_heatmap.png
  Console printed rules

Can be run standalone:
  python -m analysis.pattern_extraction
"""

import os
import sys
import csv
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ml.train_models import FEATURE_NAMES

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

TOP_PERCENTILE    = 0.20    # top 20%
BOTTOM_PERCENTILE = 0.20    # bottom 20%
MEMORY_DEPTH      = 3
N_TABLE_BITS      = 64      # 2^(2*3)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_study(data_dir="data",
              results_dir="results",
              top_pct=TOP_PERCENTILE,
              bottom_pct=BOTTOM_PERCENTILE,
              verbose=True):
    """
    Load the dataset and extract behavioural patterns.

    Returns
    -------
    comparison_df : DataFrame with per-feature statistics
    rules         : list of human-readable rule strings
    """
    from ml.train_models import load_dataset
    X, y_score, y_label, df = load_dataset(data_dir)

    os.makedirs(results_dir, exist_ok=True)

    n = len(df)
    cutoff_n = int(n * top_pct)

    # Sort by score
    sorted_df = df.sort_values("score", ascending=False).reset_index(drop=True)
    top_df    = sorted_df.iloc[:cutoff_n]
    bot_df    = sorted_df.iloc[n - cutoff_n:]

    if verbose:
        print(f"\n===== Pattern Extraction =====")
        print(f"  Total strategies : {n}")
        print(f"  Top-{top_pct:.0%} (n={len(top_df)})  "
              f"score range [{top_df['score'].min():.1f}, {top_df['score'].max():.1f}]")
        print(f"  Bot-{bottom_pct:.0%} (n={len(bot_df)})  "
              f"score range [{bot_df['score'].min():.1f}, {bot_df['score'].max():.1f}]")

    # ── Per-feature statistics ─────────────────────────────────────────────
    feat_rows = []
    for feat in FEATURE_NAMES:
        top_vals = top_df[feat].values
        bot_vals = bot_df[feat].values
        top_mean = float(np.mean(top_vals))
        bot_mean = float(np.mean(bot_vals))
        diff     = top_mean - bot_mean
        top_std  = float(np.std(top_vals))
        bot_std  = float(np.std(bot_vals))
        # Cohen's d (pooled std)
        pooled_std = math.sqrt((top_std**2 + bot_std**2) / 2) or 1e-9
        cohens_d   = diff / pooled_std

        feat_rows.append({
            "feature":   feat,
            "top_mean":  round(top_mean, 4),
            "bot_mean":  round(bot_mean, 4),
            "diff":      round(diff,     4),
            "top_std":   round(top_std,  4),
            "bot_std":   round(bot_std,  4),
            "cohens_d":  round(cohens_d, 4),
            "abs_diff":  round(abs(diff),4),
        })

    comparison_df = pd.DataFrame(feat_rows).sort_values("abs_diff",
                                                          ascending=False)

    # Save CSV
    csv_path = os.path.join(results_dir, "pattern_feature_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\n  Saved feature comparison -> {csv_path}")

    # ── Opening-move frequency ───────────────────────────────────────────
    open_feats = ["open_0", "open_1", "open_2"]
    if verbose:
        print("\n  --- Opening Move Patterns ---")
        for feat in open_feats:
            round_n = int(feat[-1]) + 1
            top_c = top_df[feat].mean()
            bot_c = bot_df[feat].mean()
            print(f"  Round {round_n}: top coop={top_c:.2%}  bot coop={bot_c:.2%}  "
                  f"Δ={top_c-bot_c:+.2%}")

    # ── Top-20 discriminating features ───────────────────────────────────
    if verbose:
        print("\n  --- Top-20 Discriminating Features (by |Δ mean|) ---")
        hdr = f"{'Feature':<28} {'TopMean':>9} {'BotMean':>9} {'Δ':>8} {'Cohen d':>9}"
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        for _, row in comparison_df.head(20).iterrows():
            print(f"  {row['feature']:<28} "
                  f"{row['top_mean']:>9.4f} "
                  f"{row['bot_mean']:>9.4f} "
                  f"{row['diff']:>+8.4f} "
                  f"{row['cohens_d']:>9.4f}")

    # ── Per-state cooperation difference heatmap ─────────────────────────
    state_diffs = []
    for bit_i in range(N_TABLE_BITS):
        feat = f"bit_{bit_i:02d}"
        if feat in top_df.columns:
            d = top_df[feat].mean() - bot_df[feat].mean()
            state_diffs.append(d)
        else:
            state_diffs.append(0.0)

    _save_heatmap(state_diffs, results_dir)

    # ── Top feature importance bar chart ─────────────────────────────────
    _save_top_features_plot(comparison_df.head(20), results_dir)

    # ── Generate human-readable rules ──────────────────────────────────
    rules = _generate_rules(comparison_df, top_df, bot_df, verbose)

    # Print rules
    if verbose:
        print("\n  --- Human-Readable Strategy Rules ---")
        for rule in rules:
            print(f"  • {rule}")

    # Save rules
    rules_path = os.path.join(results_dir, "pattern_rules.txt")
    with open(rules_path, "w") as fh:
        fh.write("Behavioural Pattern Rules\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"Based on top-{top_pct:.0%} vs bottom-{bottom_pct:.0%} "
                 f"strategies by tournament score.\n\n")
        for rule in rules:
            fh.write(f"• {rule}\n")
    if verbose:
        print(f"\n  Saved rules -> {rules_path}")

    return comparison_df, rules


# ---------------------------------------------------------------------------
# Rule generation
# ---------------------------------------------------------------------------

def _generate_rules(comparison_df, top_df, bot_df, verbose):
    """
    Derive human-readable rules from the statistical differences.
    """
    rules = []

    # Engineered feature rules
    eng_feats = [
        "table_coop_ratio",
        "opening_coop_ratio",
        "coop_after_i_coop",
        "coop_after_i_defect",
        "coop_after_opp_coop",
        "coop_after_opp_defect",
        "coop_after_mutual_coop",
        "coop_after_mutual_def",
    ]

    feat_means = {}
    for feat in eng_feats:
        if feat in top_df.columns:
            feat_means[feat] = (top_df[feat].mean(), bot_df[feat].mean())

    def _dir(top_v, bot_v, threshold=0.03):
        if top_v - bot_v > threshold:
            return "higher"
        if bot_v - top_v > threshold:
            return "lower"
        return "similar"

    if "table_coop_ratio" in feat_means:
        tv, bv = feat_means["table_coop_ratio"]
        d = _dir(tv, bv)
        if d != "similar":
            rules.append(
                f"High-performing strategies have a {d} overall cooperation rate "
                f"({tv:.1%} vs {bv:.1%}) across all history states."
            )

    if "opening_coop_ratio" in feat_means:
        tv, bv = feat_means["opening_coop_ratio"]
        d = _dir(tv, bv)
        rules.append(
            f"Opening cooperation: top strategies cooperate {tv:.1%} of the "
            f"time in early rounds vs {bv:.1%} for bottom strategies."
        )

    if "coop_after_mutual_coop" in feat_means:
        tv, bv = feat_means["coop_after_mutual_coop"]
        d = _dir(tv, bv)
        if d != "similar":
            rules.append(
                f"After mutual cooperation, top strategies cooperate "
                f"significantly {d} ({tv:.1%} vs {bv:.1%}), suggesting "
                f"{'strong conditional cooperation.' if d == 'higher' else 'more punishing behaviour.'}"
            )

    if "coop_after_opp_defect" in feat_means:
        tv, bv = feat_means["coop_after_opp_defect"]
        d = _dir(tv, bv, threshold=0.02)
        rules.append(
            f"After opponent defection, top strategies cooperate {tv:.1%} vs "
            f"{bv:.1%} — top performers tend to be "
            f"{'more forgiving.' if tv > bv else 'more retaliatory.'}"
        )

    if "coop_after_mutual_def" in feat_means:
        tv, bv = feat_means["coop_after_mutual_def"]
        rules.append(
            f"After mutual defection, top strategies cooperate {tv:.1%} vs "
            f"{bv:.1%} — suggesting top performers are "
            f"{'quicker to attempt reconciliation.' if tv > bv else 'more locked into defection cycles.'}"
        )

    if "coop_after_i_defect" in feat_means:
        tv, bv = feat_means["coop_after_i_defect"]
        rules.append(
            f"After the strategy itself defected, the cooperation rate is "
            f"{tv:.1%} (top) vs {bv:.1%} (bottom), indicating "
            f"{'self-correcting behaviour' if tv > bv else 'sustained exploitation'} in top strategies."
        )

    # Bit-level patterns
    top_20 = set(comparison_df.head(20)["feature"].values)
    table_top20 = [f for f in top_20 if f.startswith("bit_")]
    if table_top20:
        n_coop = sum(
            1 for f in table_top20
            if (top_df[f].mean() - bot_df[f].mean()) > 0
        )
        rules.append(
            f"Among the 20 most discriminating history-table bits, "
            f"{n_coop}/20 favour more cooperation in top strategies — "
            f"{'indicating that conditional cooperation is the key distinguisher.'  if n_coop > 12 else 'indicating nuanced state-specific decisions.'}"
        )

    return rules


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_top_features_plot(top20_df, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["steelblue" if d > 0 else "tomato"
              for d in top20_df["diff"]]
    y_pos = range(len(top20_df))
    ax.barh(list(y_pos)[::-1], top20_df["diff"].values,
            color=colors, edgecolor="black")
    ax.set_yticks(list(y_pos)[::-1])
    ax.set_yticklabels(top20_df["feature"].tolist(), fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Difference (Top − Bottom)")
    ax.set_title("Top-20 Discriminating Features\n(Top-20% vs Bottom-20% strategies)")
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(results_dir, "pattern_top20_features.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved pattern plot -> {path}")


def _save_heatmap(state_diffs, results_dir):
    """
    Plot an 8×8 grid showing (top − bottom) cooperation rate per history state.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    grid = np.array(state_diffs).reshape(8, 8)  # 64 states → 8×8

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="RdBu", vmin=-0.5, vmax=0.5,
                   interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, label="Coop diff (top − bottom)")
    ax.set_title("Per-State Cooperation Difference\n(Top-20% minus Bottom-20%)")
    ax.set_xlabel("State index (mod 8)")
    ax.set_ylabel("State index (// 8)")
    plt.tight_layout()
    path = os.path.join(results_dir, "pattern_state_heatmap.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved heatmap -> {path}")


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_study(verbose=True)
