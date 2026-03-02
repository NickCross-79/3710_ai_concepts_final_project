import statistics

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

# Experiment settings

RUNS = 3 #20
MATCHES = 100 #2000
ROUNDS = 50 #200


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

    for i in range(RUNS):
        best, score, history = algorithm(opponents, matches=MATCHES, rounds=ROUNDS, **kwargs)
        best_scores.append(score)
        best_strats.append(best)
        print(f"  Run {i+1}: {score}")

    print(f"\n--- {name} Results ---")
    print("Average Score:", statistics.mean(best_scores))
    print("Best Score:   ", max(best_scores))
    print("Worst Score:  ", min(best_scores))
    print("Std Dev:      ", round(statistics.stdev(best_scores), 2))

    best_index = best_scores.index(max(best_scores))
    print("Best Strategy:", strategy_to_string(best_strats[best_index]))

    return best_scores


if __name__ == "__main__":
    opponents = build_opponents()

    # Run all four algorithms
    scores_random = run_experiment("Random Search",       random_search,       opponents)
    scores_ga     = run_experiment("Genetic Algorithm",   genetic_algorithm,   opponents)
    scores_hc     = run_experiment("Hill Climbing",       hill_climbing,       opponents)
    scores_tabu   = run_experiment("Tabu Search",         tabu_search,         opponents)

    # Final comparison
    print("\n========== FINAL COMPARISON ==========")
    print(f"{'Algorithm':<20} {'Avg':>8} {'Best':>8} {'Worst':>8} {'Std':>8}")
    print("-" * 56)
    for name, scores in [
        ("Random Search",     scores_random),
        ("Genetic Algorithm", scores_ga),
        ("Hill Climbing",     scores_hc),
        ("Tabu Search",       scores_tabu),
    ]:
        print(f"{name:<20} {statistics.mean(scores):>8.1f} {max(scores):>8.1f} "
              f"{min(scores):>8.1f} {statistics.stdev(scores):>8.2f}")