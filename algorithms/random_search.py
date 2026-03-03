import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from strats import random_strategy, strategy_to_string
from prisoner_dilemma import round_robin_score

# Example of an algorithm implementing the prisoner_dilemma and strats interfaces. This is a simple random search that samples random strategies and keeps the best one found.

def random_search(opponents, matches=5000, rounds=200):
    """
    Randomly sample strats and keep the best one found.
    """

    best_strategy = None
    best_score = -1

    history = []

    for i in range(matches):
        strat = random_strategy()
        score = round_robin_score(strat, opponents, rounds)

        history.append(score)

        if score > best_score:
            best_score = score
            best_strategy = strat
    return best_strategy, best_score, history


if __name__ == "__main__":

    # Example opponent pool
    from strats import (
        always_cooperate,
        always_defect,
        tit_for_tat,
        suspicious_tit_for_tat
    )

    opponents = [
        always_cooperate(),
        always_defect(),
        tit_for_tat(),
        suspicious_tit_for_tat()
    ]

    best, score, history = random_search(opponents, matches=2000)

    print("Best Strategy Found:")
    print(strategy_to_string(best))
    print("Score:", score)