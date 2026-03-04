"""
experiments/memory_depth_study.py
===================================
Compares GA performance across three memory depths:
  Memory-3  →   67 bits
  Memory-4  →  260 bits
  Memory-5  → 1029 bits

For each depth, the GA runs RUNS_PER_DEPTH independent times.
Metrics recorded: average score, std deviation, runtime, convergence history.

Outputs:
  results/memory_depth_results.csv
  results/memory_depth_avg_score.png
  results/memory_depth_convergence.png

Can be run standalone:
  python -m experiments.memory_depth_study
"""

import os
import sys
import csv
import time
import random
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.encoding import (
    strategy_length,
    random_strategy_md,
    mutate_md,
    crossover_md,
    round_robin_score_md,
    strategy_to_string_md,
)
from strategies.classic_strategies import (
    always_cooperate_md,
    always_defect_md,
    tit_for_tat_md,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MEMORY_DEPTHS    = [3, 4, 5]
RUNS_PER_DEPTH   = 10
POPULATION_SIZE  = 20
MUTATION_RATE    = 0.05
N_GENERATIONS    = 50    # GA generation budget per run
ROUNDS_PER_MATCH = 30    # rounds each play_match call
N_RANDOM_OPPS    = 20    # random opponents in the pool


# ---------------------------------------------------------------------------
# Opponent pool builder (memory-depth aware)
# ---------------------------------------------------------------------------

def build_opponents_md(memory_depth, n_random=N_RANDOM_OPPS, seed=None):
    if seed is not None:
        random.seed(seed)
    opponents = [
        always_cooperate_md(memory_depth),
        always_defect_md(memory_depth),
        tit_for_tat_md(memory_depth),
    ]
    for _ in range(n_random):
        opponents.append(random_strategy_md(memory_depth))
    return opponents


# ---------------------------------------------------------------------------
# Memory-depth-aware GA
# ---------------------------------------------------------------------------

def _run_ga_md(opponents, memory_depth, population_size, mutation_rate,
               n_generations, rounds, seed=None):
    """
    Run one GA trial for the given memory depth.
    Returns (best_strategy, best_score, convergence_history).
    """
    if seed is not None:
        random.seed(seed)

    population = [random_strategy_md(memory_depth) for _ in range(population_size)]
    history    = []

    for _ in range(n_generations):
        scored = [
            (s, round_robin_score_md(s, opponents, rounds, memory_depth))
            for s in population
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        history.append(scored[0][1])

        survivors = [s for s, _ in scored[:population_size // 2]]

        next_gen = survivors[:]
        while len(next_gen) < population_size:
            pa = random.choice(survivors)
            pb = random.choice(survivors)
            child = crossover_md(pa, pb)
            child = mutate_md(child, mutation_rate)
            next_gen.append(child)

        population = next_gen

    # Final scoring pass
    final = [
        (s, round_robin_score_md(s, opponents, rounds, memory_depth))
        for s in population
    ]
    best = max(final, key=lambda x: x[1])
    return best[0], best[1], history


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(memory_depths=None,
              runs_per_depth=RUNS_PER_DEPTH,
              population_size=POPULATION_SIZE,
              mutation_rate=MUTATION_RATE,
              n_generations=N_GENERATIONS,
              rounds=ROUNDS_PER_MATCH,
              n_random_opps=N_RANDOM_OPPS,
              results_dir="results",
              seed=None,
              verbose=True):
    """
    Run the memory-depth comparison study.

    Returns
    -------
    records : list of dicts, one per (memory_depth, run)
    depth_summaries : dict  memory_depth → summary dict
    best_by_depth   : dict  memory_depth → (strategy, score)
    """
    memory_depths = memory_depths or MEMORY_DEPTHS
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "memory_depth_results.csv")

    if verbose:
        print(f"\n===== Memory Depth Study =====")
        print(f"  Depths         : {memory_depths}")
        print(f"  Runs per depth : {runs_per_depth}")
        print(f"  Population     : {population_size}")
        print(f"  Mutation rate  : {mutation_rate}")
        print(f"  Generations    : {n_generations}")
        print(f"  Rounds/match   : {rounds}")
        print(f"  Output CSV     : {csv_path}\n")

    records       = []
    depth_summaries = {}
    best_by_depth   = {}

    fieldnames = [
        "memory_depth", "strategy_length", "run",
        "score", "runtime_s",
        "conv_25pct", "conv_50pct", "conv_75pct", "conv_100pct",
    ]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for depth in memory_depths:
            slen     = strategy_length(depth)
            opponents = build_opponents_md(depth, n_random=n_random_opps,
                                           seed=seed)
            scores   = []
            runtimes = []
            all_conv = []
            best_strat = None
            best_score = -1

            if verbose:
                print(f"  --- Memory-{depth} (strategy length = {slen}) ---")

            for run in range(runs_per_depth):
                run_seed = (seed + depth * 100 + run) if seed is not None else None
                t0 = time.time()
                strat, score, conv = _run_ga_md(
                    opponents, depth, population_size, mutation_rate,
                    n_generations, rounds, seed=run_seed
                )
                elapsed = time.time() - t0

                scores.append(score)
                runtimes.append(elapsed)
                all_conv.append(conv)

                if score > best_score:
                    best_score = score
                    best_strat = strat[:]

                def _at(pct, c=conv):
                    idx = max(0, int(pct * len(c)) - 1)
                    return round(c[idx], 2) if c else 0.0

                row = {
                    "memory_depth":    depth,
                    "strategy_length": slen,
                    "run":             run + 1,
                    "score":           round(score, 2),
                    "runtime_s":       round(elapsed, 3),
                    "conv_25pct":      _at(0.25),
                    "conv_50pct":      _at(0.50),
                    "conv_75pct":      _at(0.75),
                    "conv_100pct":     _at(1.00),
                }
                records.append(row)
                writer.writerow(row)
                fh.flush()

                if verbose:
                    print(f"    Run {run+1:>2}: score={score:>7.1f}  "
                          f"time={elapsed:.1f}s")

            avg = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            depth_summaries[depth] = {
                "avg":      avg,
                "std":      std,
                "best":     max(scores),
                "worst":    min(scores),
                "avg_time": statistics.mean(runtimes),
            }
            best_by_depth[depth] = (best_strat, best_score)

            if verbose:
                s = depth_summaries[depth]
                print(f"    Summary → avg={s['avg']:.1f}  std={s['std']:.1f}  "
                      f"best={s['best']:.1f}  avg_time={s['avg_time']:.1f}s\n")

    _print_table(memory_depths, depth_summaries)
    _save_plots(memory_depths, records, depth_summaries, results_dir,
                n_generations)

    return records, depth_summaries, best_by_depth


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

def _print_table(memory_depths, summaries):
    print("\n--- Memory Depth Comparison ---")
    print(f"{'Depth':>7} {'Bits':>6} | {'Avg':>8} {'Std':>7} "
          f"{'Best':>8} {'Worst':>8} {'AvgTime':>9}")
    print("-" * 60)
    for d in memory_depths:
        s = summaries[d]
        print(f"{d:>7} {strategy_length(d):>6} | "
              f"{s['avg']:>8.1f} {s['std']:>7.2f} "
              f"{s['best']:>8.1f} {s['worst']:>8.1f} "
              f"{s['avg_time']:>8.1f}s")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_plots(memory_depths, records, summaries, results_dir,
                n_generations=N_GENERATIONS):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available – skipping plots.")
        return

    depths = memory_depths
    avgs   = [summaries[d]["avg"]  for d in depths]
    stds   = [summaries[d]["std"]  for d in depths]
    bests  = [summaries[d]["best"] for d in depths]

    # ── Plot 1: Bar chart of avg score ± std ──
    fig, ax = plt.subplots(figsize=(7, 5))
    x = list(range(len(depths)))
    ax.bar(x, avgs, yerr=stds, capsize=8, color="steelblue",
           edgecolor="black", label="Avg ± Std")
    ax.plot(x, bests, marker="D", color="crimson",
            linestyle="--", label="Best")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Memory-{d}\n({strategy_length(d)} bits)"
                        for d in depths])
    ax.set_ylabel("Score")
    ax.set_title("GA Performance vs Memory Depth")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    p1 = os.path.join(results_dir, "memory_depth_avg_score.png")
    plt.savefig(p1, dpi=120)
    plt.close()
    print(f"  Saved plot -> {p1}")

    # ── Plot 2: Convergence curves per depth (average over runs) ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for d in depths:
        depth_records = [r for r in records if r["memory_depth"] == d]
        # Collect the 4 convergence checkpoints as a proxy curve
        pts = [(0.25, statistics.mean(r["conv_25pct"] for r in depth_records)),
               (0.50, statistics.mean(r["conv_50pct"] for r in depth_records)),
               (0.75, statistics.mean(r["conv_75pct"] for r in depth_records)),
               (1.00, statistics.mean(r["conv_100pct"] for r in depth_records))]
        xs = [int(p * n_generations) for p, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, marker="o", label=f"Memory-{d}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Score (avg over runs)")
    ax.set_title("Convergence by Memory Depth")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    p2 = os.path.join(results_dir, "memory_depth_convergence.png")
    plt.savefig(p2, dpi=120)
    plt.close()
    print(f"  Saved plot -> {p2}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_study(seed=42, verbose=True)
