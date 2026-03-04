"""
experiments/hyperparameter_study.py
====================================
Hyperparameter sensitivity study for the Genetic Algorithm.

Sweeps combinations of:
  - population_size  : number of individuals per generation
  - mutation_rate    : per-bit flip probability
  - crossover_rate   : probability that crossover is applied vs. clone
  - n_generations    : number of GA generations (= `matches` budget)

Each configuration is run RUNS_PER_CONFIG times.
Records average score, std deviation, best score, and convergence history.

Outputs:
  results/ga_hyperparameter_results.csv
  results/ga_hp_avg_score.png
  results/ga_hp_best_score.png

Can be run standalone:
  python -m experiments.hyperparameter_study
"""

import os
import sys
import csv
import time
import random
import statistics
import itertools
import math

# Ensure project root is on the path when run standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prisoner_dilemma import round_robin_score
from strats import random_strategy, mutate, copy_strategy

# ---------------------------------------------------------------------------
# Default sweep parameters  (all overridable via run_study())
# ---------------------------------------------------------------------------

DEFAULT_POPULATION_SIZES = [20, 50, 100]
DEFAULT_MUTATION_RATES   = [0.01, 0.05, 0.1, 0.2]
DEFAULT_CROSSOVER_RATES  = [0.6, 0.8, 1.0]
DEFAULT_N_GENERATIONS    = [50, 100, 200]

RUNS_PER_CONFIG  = 10
ROUNDS_PER_MATCH = 50    # rounds inside each play_match call


# ---------------------------------------------------------------------------
# Local GA with configurable crossover_rate
# ---------------------------------------------------------------------------

def _crossover(parent_a, parent_b, crossover_rate):
    """
    With probability `crossover_rate` perform single-point crossover;
    otherwise clone parent_a.
    """
    if random.random() < crossover_rate:
        pt = random.randint(1, len(parent_a) - 1)
        return parent_a[:pt] + parent_b[pt:]
    return copy_strategy(parent_a)


def _run_ga(opponents, population_size, mutation_rate, crossover_rate,
            n_generations, rounds, seed=None):
    """
    Run a single GA trial.

    Returns (best_strategy, best_score, convergence_history).
    convergence_history[g] = best score seen after generation g.
    """
    if seed is not None:
        random.seed(seed)

    population = [random_strategy() for _ in range(population_size)]
    history    = []

    for _ in range(n_generations):
        scored = [(s, round_robin_score(s, opponents, rounds))
                  for s in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        history.append(scored[0][1])

        survivors = [s for s, _ in scored[:population_size // 2]]

        next_gen = survivors[:]
        while len(next_gen) < population_size:
            pa = random.choice(survivors)
            pb = random.choice(survivors)
            child = _crossover(pa, pb, crossover_rate)
            child = mutate(child, mutation_rate)
            next_gen.append(child)

        population = next_gen

    # Final scoring pass
    final = [(s, round_robin_score(s, opponents, rounds))
             for s in population]
    best = max(final, key=lambda x: x[1])
    return best[0], best[1], history


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(opponents,
              population_sizes=None,
              mutation_rates=None,
              crossover_rates=None,
              n_generations_list=None,
              runs_per_config=RUNS_PER_CONFIG,
              rounds=ROUNDS_PER_MATCH,
              results_dir="results",
              seed=None,
              verbose=True):
    """
    Run the full hyperparameter sweep and save results.

    Parameters
    ----------
    opponents          : list of strategy bitstrings to evaluate against
    population_sizes   : list of ints
    mutation_rates     : list of floats
    crossover_rates    : list of floats
    n_generations_list : list of ints
    runs_per_config    : int  – independent runs per configuration
    rounds             : int  – rounds per match during evaluation
    results_dir        : str  – directory for CSV and PNG outputs
    seed               : int or None – base random seed (each run uses seed+run)
    verbose            : bool

    Returns
    -------
    records : list of dicts (one per configuration)
    """
    population_sizes   = population_sizes   or DEFAULT_POPULATION_SIZES
    mutation_rates     = mutation_rates     or DEFAULT_MUTATION_RATES
    crossover_rates    = crossover_rates    or DEFAULT_CROSSOVER_RATES
    n_generations_list = n_generations_list or DEFAULT_N_GENERATIONS

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "ga_hyperparameter_results.csv")

    configs = list(itertools.product(
        population_sizes, mutation_rates, crossover_rates, n_generations_list
    ))
    total = len(configs)

    if verbose:
        print(f"\n===== GA Hyperparameter Study =====")
        print(f"  Configurations : {total}")
        print(f"  Runs per cfg   : {runs_per_config}")
        print(f"  Rounds/match   : {rounds}")
        print(f"  Opponents      : {len(opponents)}")
        print(f"  Output CSV     : {csv_path}\n")

    records = []

    fieldnames = [
        "population_size", "mutation_rate", "crossover_rate", "n_generations",
        "avg_score", "std_score", "best_score", "worst_score",
        "avg_runtime_s",
        # convergence: best score at 25%, 50%, 75%, 100% of generations
        "conv_25pct", "conv_50pct", "conv_75pct", "conv_100pct",
    ]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for cfg_idx, (pop, mut, cr, ngen) in enumerate(configs, 1):
            scores   = []
            runtimes = []
            all_conv = []

            for run in range(runs_per_config):
                run_seed = (seed + cfg_idx * 1000 + run) if seed is not None else None
                t0 = time.time()
                _, score, conv = _run_ga(
                    opponents, pop, mut, cr, ngen, rounds, seed=run_seed
                )
                runtimes.append(time.time() - t0)
                scores.append(score)
                all_conv.append(conv)

            avg   = statistics.mean(scores)
            std   = statistics.stdev(scores) if len(scores) > 1 else 0.0
            best  = max(scores)
            worst = min(scores)
            avg_t = statistics.mean(runtimes)

            # Average convergence curve across runs
            min_len = min(len(c) for c in all_conv)
            avg_conv = [
                statistics.mean(c[g] for c in all_conv if g < len(c))
                for g in range(min_len)
            ]

            def _at(pct):
                idx = max(0, int(pct * len(avg_conv)) - 1)
                return round(avg_conv[idx], 2) if avg_conv else 0.0

            row = {
                "population_size": pop,
                "mutation_rate":   mut,
                "crossover_rate":  cr,
                "n_generations":   ngen,
                "avg_score":       round(avg,   2),
                "std_score":       round(std,   2),
                "best_score":      round(best,  2),
                "worst_score":     round(worst, 2),
                "avg_runtime_s":   round(avg_t, 3),
                "conv_25pct":      _at(0.25),
                "conv_50pct":      _at(0.50),
                "conv_75pct":      _at(0.75),
                "conv_100pct":     _at(1.00),
            }
            records.append(row)
            writer.writerow(row)
            fh.flush()

            if verbose:
                print(f"  [{cfg_idx:>3}/{total}] "
                      f"pop={pop:>3} mut={mut:.2f} cr={cr:.1f} gen={ngen:>3} | "
                      f"avg={avg:>7.1f}  std={std:>6.1f}  best={best:>7.1f}  "
                      f"t={avg_t:.1f}s")

    if verbose:
        print(f"\n  Saved {len(records)} rows -> {csv_path}")

    _print_table(records)
    _save_plots(records, results_dir)

    return records


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

def _print_table(records):
    print("\n--- Hyperparameter Study: Top-20 Configurations (by avg score) ---")
    hdr = (f"{'Pop':>5} {'Mut':>5} {'CR':>5} {'Gen':>5} | "
           f"{'Avg':>8} {'Std':>7} {'Best':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(records, key=lambda x: x["avg_score"], reverse=True)[:20]:
        print(f"{r['population_size']:>5} {r['mutation_rate']:>5.2f} "
              f"{r['crossover_rate']:>5.1f} {r['n_generations']:>5} | "
              f"{r['avg_score']:>8.1f} {r['std_score']:>7.2f} "
              f"{r['best_score']:>8.1f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_plots(records, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available – skipping plots.")
        return

    # ── Plot 1: avg score by mutation_rate (one line per population_size) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mut_rates = sorted(set(r["mutation_rate"] for r in records))
    pop_sizes = sorted(set(r["population_size"] for r in records))

    ax = axes[0]
    for pop in pop_sizes:
        ys = []
        for mut in mut_rates:
            subset = [r["avg_score"] for r in records
                      if r["population_size"] == pop and r["mutation_rate"] == mut]
            ys.append(statistics.mean(subset) if subset else 0)
        ax.plot(mut_rates, ys, marker="o", label=f"pop={pop}")
    ax.set_xlabel("Mutation Rate")
    ax.set_ylabel("Average Score")
    ax.set_title("Avg Score vs Mutation Rate\n(averaged over other params)")
    ax.legend()
    ax.grid(True)

    # ── Plot 2: avg score by population_size (one line per n_generations) ──
    gen_list = sorted(set(r["n_generations"] for r in records))
    ax = axes[1]
    for gen in gen_list:
        ys = []
        for pop in pop_sizes:
            subset = [r["avg_score"] for r in records
                      if r["n_generations"] == gen and r["population_size"] == pop]
            ys.append(statistics.mean(subset) if subset else 0)
        ax.plot(pop_sizes, ys, marker="s", label=f"gen={gen}")
    ax.set_xlabel("Population Size")
    ax.set_ylabel("Average Score")
    ax.set_title("Avg Score vs Population Size\n(averaged over other params)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    path = os.path.join(results_dir, "ga_hp_avg_score.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved plot -> {path}")

    # ── Plot 3: best score vs crossover_rate ──
    cr_vals = sorted(set(r["crossover_rate"] for r in records))
    fig, ax = plt.subplots(figsize=(7, 5))
    avg_by_cr = []
    std_by_cr = []
    for cr in cr_vals:
        subset = [r["best_score"] for r in records if r["crossover_rate"] == cr]
        avg_by_cr.append(statistics.mean(subset))
        std_by_cr.append(statistics.stdev(subset) if len(subset) > 1 else 0)
    ax.bar([str(c) for c in cr_vals], avg_by_cr, yerr=std_by_cr,
           capsize=5, color="steelblue", edgecolor="black")
    ax.set_xlabel("Crossover Rate")
    ax.set_ylabel("Average Best Score")
    ax.set_title("Best Score vs Crossover Rate\n(averaged over all other params)")
    ax.grid(axis="y")
    plt.tight_layout()
    path2 = os.path.join(results_dir, "ga_hp_best_score.png")
    plt.savefig(path2, dpi=120)
    plt.close()
    print(f"  Saved plot -> {path2}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from strats import (always_cooperate, always_defect,
                        tit_for_tat, suspicious_tit_for_tat, random_strategy)

    def _build_opponents():
        pool = [always_cooperate(), always_defect(),
                tit_for_tat(), suspicious_tit_for_tat()]
        for _ in range(30):
            pool.append(random_strategy())
        return pool

    opps = _build_opponents()
    run_study(opps, seed=42,
              population_sizes=[20, 50],
              mutation_rates=[0.05, 0.1, 0.2],
              crossover_rates=[0.7, 1.0],
              n_generations_list=[50, 100])
