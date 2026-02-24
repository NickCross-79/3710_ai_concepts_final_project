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

# Experiment settings

RUNS = 20
MATCHES = 2000 # Each run has 2000 matches
ROUNDS = 200 # Each match has 200 rounds


def build_opponents():
    """
    Opponent pool used to evaluate strats.
    """
    opponents = [
        always_cooperate(),
        always_defect(),
        tit_for_tat(),
        suspicious_tit_for_tat()
        # TODO: add more known algorithms as opponents
    ]

    # Add some random opponents
    for _ in range(30):
        opponents.append(random_strategy())

    return opponents

# Example of algorithm being tested
def run_random_search_experiment():
    opponents = build_opponents()

    best_scores = []
    best_strats = []

    for i in range(RUNS):

        best, score, history = random_search(
            opponents,
            matches=MATCHES,
            rounds=ROUNDS
        )

        best_scores.append(score)
        best_strats.append(best)

        print(f"Run {i+1} best score: {score}")

    print("\n----- Random Search Results -----")

    print("Average Score:", statistics.mean(best_scores))
    print("Best Score:", max(best_scores))
    print("Worst Score:", min(best_scores))
    print("Std Dev:", statistics.stdev(best_scores))

    best_index = best_scores.index(max(best_scores))

    print("\nBest Strategy Found:")
    print(strategy_to_string(best_strats[best_index]))


if __name__ == "__main__":

    # Test all algorithms here
    run_random_search_experiment()