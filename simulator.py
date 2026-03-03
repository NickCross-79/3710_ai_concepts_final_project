import statistics
import time

from strats import (
    always_cooperate,
    always_defect,
    tit_for_tat,
    suspicious_tit_for_tat,
    random_strategy,
    strategy_to_string
)

from algorithms.random_search import random_search
from algorithms.genetic_algorithm import genetic_algorithm
from algorithms.hill_climbing import hill_climbing
from algorithms.tabu_search import tabu_search
from algorithms.simulated_annealing import simulated_annealing
from algorithms.memetic_algorithm import memetic_algorithm
from algorithms.differential_evolution import differential_evolution

# Experiment settings

RUNS = 20    # set to 20 for full experiment
MATCHES = 200  # set to 200 for full experiment
ROUNDS = 50


def build_opponents():
    """
    Opponent pool used to evaluate strats.
    """
    opponents = [
        always_cooperate(),
        always_defect(),
        tit_for_tat(),
        suspicious_tit_for_tat()
    ]
    for _ in range(30):
        opponents.append(random_strategy())
    return opponents


def run_experiment(name, algorithm, opponents, **kwargs):
    print(f"\n===== {name} =====")

    best_scores = []
    best_strats = []
    run_times   = []

    for i in range(RUNS):
        t0 = time.time()
        best, score, history = algorithm(opponents, matches=MATCHES, rounds=ROUNDS, **kwargs)
        elapsed = time.time() - t0
        best_scores.append(score)
        best_strats.append(best)
        run_times.append(elapsed)
        print(f"  Run {i+1}: {score}  ({elapsed:.1f}s)")

    print(f"\n--- {name} Results ---")
    print(f"Average Score: {statistics.mean(best_scores):.1f}")
    print(f"Median Score:  {statistics.median(best_scores):.1f}")
    print(f"Best Score:    {max(best_scores)}")
    print(f"Worst Score:   {min(best_scores)}")
    print(f"Std Dev:       {statistics.stdev(best_scores):.2f}")
    print(f"Avg Time/Run:  {statistics.mean(run_times):.1f}s")
    print(f"Total Time:    {sum(run_times):.1f}s")

    best_index = best_scores.index(max(best_scores))
    print(f"Best Strategy: {strategy_to_string(best_strats[best_index])}")

    return best_scores, run_times


if __name__ == "__main__":
    opponents = build_opponents()

    # Print experiment configuration
    print("========== EXPERIMENT CONFIGURATION ==========")
    print(f"  Runs per algorithm : {RUNS}")
    print(f"  Match budget       : {MATCHES}")
    print(f"  Rounds per match   : {ROUNDS}")
    print(f"  Opponent pool size : {len(opponents)}")
    print(f"    - Always Cooperate      x1")
    print(f"    - Always Defect         x1")
    print(f"    - Tit-for-Tat           x1")
    print(f"    - Suspicious TFT        x1")
    print(f"    - Random strategies     x30")

    # Run all seven algorithms
    scores_random, times_random = run_experiment("Random Search",       random_search,          opponents)
    scores_ga,     times_ga     = run_experiment("Genetic Algorithm",   genetic_algorithm,      opponents)
    scores_hc,     times_hc     = run_experiment("Hill Climbing",       hill_climbing,          opponents)
    scores_tabu,   times_tabu   = run_experiment("Tabu Search",         tabu_search,            opponents)
    scores_sa,     times_sa     = run_experiment("Simulated Annealing", simulated_annealing,    opponents)
    scores_ma,     times_ma     = run_experiment("Memetic Algorithm",   memetic_algorithm,      opponents)
    scores_de,     times_de     = run_experiment("Differential Evol.",  differential_evolution, opponents)

    # Final comparison
    print("\n========== FINAL COMPARISON ==========")
    print(f"{'Algorithm':<22} {'Avg':>8} {'Median':>8} {'Best':>8} {'Worst':>8} {'Std':>8} {'AvgTime':>9}")
    print("-" * 77)
    for name, scores, times in [
        ("Random Search",       scores_random, times_random),
        ("Genetic Algorithm",   scores_ga,     times_ga),
        ("Hill Climbing",       scores_hc,     times_hc),
        ("Tabu Search",         scores_tabu,   times_tabu),
        ("Simulated Annealing", scores_sa,     times_sa),
        ("Memetic Algorithm",   scores_ma,     times_ma),
        ("Differential Evol.",  scores_de,     times_de),
    ]:
        print(f"{name:<22} {statistics.mean(scores):>8.1f} {statistics.median(scores):>8.1f} "
              f"{max(scores):>8.1f} {min(scores):>8.1f} {statistics.stdev(scores):>8.2f} "
              f"{statistics.mean(times):>8.1f}s")