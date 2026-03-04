"""
main.py — IPD Optimization Research: Full Experiment Runner
=============================================================
Runs all five experiment parts in sequence, producing every output file
in a single invocation.

Usage
-----
    python main.py                        # full run with default settings
    python main.py --seed 42              # fixed random seed
    python main.py --fast                 # reduced budgets for quick testing
    python main.py --skip hp md bc ml pa  # skip parts by code
    python main.py --parts hp bc          # run only named parts
    python main.py --help

Part codes
----------
  hp  – Part 1: Hyperparameter Sensitivity Study (GA)
  md  – Part 2: Memory Depth Experiments
  bc  – Part 3: Baseline Comparison vs Classic Strategies
  ml  – Part 4: ML Prediction of Strategy Quality
  pa  – Part 5: Pattern Extraction from Good Strategies

All outputs go to:
  data/          – generated datasets (CSV)
  results/        – plots (PNG), tables (CSV), model files (pkl)
"""

import argparse
import os
import sys
import random
import time

# ---------------------------------------------------------------------------
# Make sure the project root is on the Python path
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Build the shared opponent pool (memory-3)
# ---------------------------------------------------------------------------

def build_opponents(seed=None):
    from strats import (always_cooperate, always_defect,
                        tit_for_tat, suspicious_tit_for_tat, random_strategy)
    if seed is not None:
        random.seed(seed)
    pool = [always_cooperate(), always_defect(),
            tit_for_tat(), suspicious_tit_for_tat()]
    for _ in range(30):
        pool.append(random_strategy())
    return pool


# ---------------------------------------------------------------------------
# Part runners
# ---------------------------------------------------------------------------

def run_part1_hyperparameter(seed, fast, results_dir):
    """GA Hyperparameter Sensitivity Study."""
    from experiments.hyperparameter_study import run_study

    if fast:
        pop_sizes  = [20]
        mut_rates  = [0.05, 0.1]
        cr_rates   = [0.8, 1.0]
        n_gens     = [20, 40]
        runs_cfg   = 3
        rounds_hp  = 20
    else:
        pop_sizes  = [20, 50, 100]
        mut_rates  = [0.01, 0.05, 0.1, 0.2]
        cr_rates   = [0.6, 0.8, 1.0]
        n_gens     = [50, 100, 200]
        runs_cfg   = 10
        rounds_hp  = 50

    opponents = build_opponents(seed)

    records = run_study(
        opponents,
        population_sizes=pop_sizes,
        mutation_rates=mut_rates,
        crossover_rates=cr_rates,
        n_generations_list=n_gens,
        runs_per_config=runs_cfg,
        rounds=rounds_hp,
        results_dir=results_dir,
        seed=seed,
        verbose=True,
    )
    return records


def run_part2_memory_depth(seed, fast, results_dir):
    """Memory Depth Comparison Study."""
    from experiments.memory_depth_study import run_study

    if fast:
        depths   = [3, 4]
        runs     = 2
        pop      = 10
        gens     = 15
        rounds   = 15
    else:
        depths   = [3, 4, 5]
        runs     = 10
        pop      = 20
        gens     = 50
        rounds   = 30

    records, summaries, best_by_depth = run_study(
        memory_depths=depths,
        runs_per_depth=runs,
        population_size=pop,
        n_generations=gens,
        rounds=rounds,
        results_dir=results_dir,
        seed=seed,
        verbose=True,
    )
    return records, summaries, best_by_depth


def run_part3_baseline(seed, fast, results_dir, evolved_strategy=None):
    """Baseline Comparison vs Classic Strategies."""
    from experiments.baseline_comparison import run_study

    if fast:
        ga_gens    = 20
        ga_pop     = 15
        cmp_rounds = 200
    else:
        ga_gens    = 100
        ga_pop     = 50
        cmp_rounds = 1000

    records, evolved = run_study(
        evolved_strategy=evolved_strategy,
        comparison_rounds=cmp_rounds,
        ga_generations=ga_gens,
        ga_pop_size=ga_pop,
        results_dir=results_dir,
        seed=seed,
        verbose=True,
    )
    return records, evolved


def run_part4_ml(seed, fast, data_dir, results_dir):
    """ML Prediction of Strategy Quality."""
    from ml.train_models    import run_study   as train_run
    from ml.evaluate_models import run_evaluation

    if fast:
        rand_count  = 80
        ga_runs     = 4
        include_mlp = False
    else:
        rand_count  = 300
        ga_runs     = 20
        include_mlp = True

    cls_res, reg_res, trained, df = train_run(
        random_count=rand_count,
        evolved_ga_runs=ga_runs,
        include_mlp=include_mlp,
        data_dir=data_dir,
        results_dir=results_dir,
        seed=seed,
        verbose=True,
    )

    # Deeper evaluation using the just-saved models
    run_evaluation(
        data_dir=data_dir,
        results_dir=results_dir,
        seed=seed,
        verbose=True,
    )

    return cls_res, reg_res, df


def run_part5_patterns(data_dir, results_dir):
    """Pattern Extraction from Good Strategies."""
    from analysis.pattern_extraction import run_study

    comparison_df, rules = run_study(
        data_dir=data_dir,
        results_dir=results_dir,
        verbose=True,
    )
    return comparison_df, rules


# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="IPD Optimization Research – Full Experiment Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, default=67,
        help="Global random seed for reproducibility (default: 67).",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use reduced budgets for a quick end-to-end test run.",
    )
    parser.add_argument(
        "--parts", nargs="+",
        choices=["hp", "md", "bc", "ml", "pa"],
        default=["hp", "md", "bc", "ml", "pa"],
        metavar="PART",
        help=(
            "Which parts to run (default: all).\n"
            "  hp  – Hyperparameter study\n"
            "  md  – Memory depth study\n"
            "  bc  – Baseline comparison\n"
            "  ml  – ML training & evaluation\n"
            "  pa  – Pattern extraction"
        ),
    )
    parser.add_argument(
        "--skip", nargs="+",
        choices=["hp", "md", "bc", "ml", "pa"],
        default=[],
        metavar="PART",
        help="Parts to skip (applied after --parts).",
    )
    parser.add_argument(
        "--data-dir",    default=os.path.join(ROOT, "data"),
        help="Directory for dataset files.",
    )
    parser.add_argument(
        "--results-dir", default=os.path.join(ROOT, "results"),
        help="Directory for result files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    parts_to_run = [p for p in args.parts if p not in args.skip]

    os.makedirs(args.data_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    random.seed(args.seed)

    print("=" * 60)
    print(" IPD Optimization Research – Experiment Suite")
    print("=" * 60)
    print(f"  Seed        : {args.seed}")
    print(f"  Fast mode   : {args.fast}")
    print(f"  Parts       : {parts_to_run}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Results dir : {args.results_dir}")
    print("=" * 60)

    t_total = time.time()
    evolved_strategy = None   # carry best strategy from Part 1/3

    # ── Part 1: Hyperparameter Study ──────────────────────────────────────
    if "hp" in parts_to_run:
        t0 = time.time()
        run_part1_hyperparameter(args.seed, args.fast, args.results_dir)
        print(f"\n  [Part 1 done in {time.time()-t0:.1f}s]")

    # ── Part 2: Memory Depth ───────────────────────────────────────────────
    if "md" in parts_to_run:
        t0 = time.time()
        _, _, best_by_depth = run_part2_memory_depth(
            args.seed, args.fast, args.results_dir
        )
        # Grab the best memory-3 strategy for potential reuse
        if 3 in best_by_depth and evolved_strategy is None:
            evolved_strategy = best_by_depth[3][0]
        print(f"\n  [Part 2 done in {time.time()-t0:.1f}s]")

    # ── Part 3: Baseline Comparison ─────────────────────────────────────
    if "bc" in parts_to_run:
        t0 = time.time()
        _, evolved_strategy = run_part3_baseline(
            args.seed, args.fast, args.results_dir,
            evolved_strategy=evolved_strategy,
        )
        print(f"\n  [Part 3 done in {time.time()-t0:.1f}s]")

    # ── Part 4: ML ─────────────────────────────────────────────────────
    if "ml" in parts_to_run:
        t0 = time.time()
        run_part4_ml(args.seed, args.fast, args.data_dir, args.results_dir)
        print(f"\n  [Part 4 done in {time.time()-t0:.1f}s]")

    # ── Part 5: Pattern Extraction (requires dataset from Part 4) ──────
    if "pa" in parts_to_run:
        dataset_path = os.path.join(args.data_dir, "strategy_dataset.csv")
        if not os.path.isfile(dataset_path):
            print("\n  [Part 5] No dataset found – running Part 4 first …")
            run_part4_ml(args.seed, args.fast, args.data_dir, args.results_dir)
        t0 = time.time()
        run_part5_patterns(args.data_dir, args.results_dir)
        print(f"\n  [Part 5 done in {time.time()-t0:.1f}s]")

    # ── Final summary ──────────────────────────────────────────────────
    total_time = time.time() - t_total
    print("\n" + "=" * 60)
    print(f"  All experiments complete  (total: {total_time:.1f}s)")
    print("=" * 60)
    print("\n  Output files:")
    for root, _, files in os.walk(args.results_dir):
        for f in sorted(files):
            if not f.startswith("."):
                rel = os.path.relpath(os.path.join(root, f), ROOT)
                print(f"    {rel}")
    for root, _, files in os.walk(args.data_dir):
        for f in sorted(files):
            if not f.startswith("."):
                rel = os.path.relpath(os.path.join(root, f), ROOT)
                print(f"    {rel}")
    print()


if __name__ == "__main__":
    main()
