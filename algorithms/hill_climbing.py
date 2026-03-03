import random
from strats import random_strategy, get_neighbors
from prisoner_dilemma import round_robin_score


def hill_climbing(opponents, matches=2000, rounds=200):
    """
    Hill Climbing for the Prisoner's Dilemma.

    - Start with a random strategy
    - Look at all neighboring strategies (flip one bit at a time)
    - Move to the best neighbor if it's better
    - If no neighbor is better, restart from a new random strategy
    - Repeat until we've used up our match budget

    Returns: best_strategy, best_score, history
    """

    best_ever_strategy = None
    best_ever_score = -1
    history = []

    matches_used = 0

    while matches_used < matches:

        # Start from a new random strategy
        current = random_strategy()
        current_score = round_robin_score(current, opponents, rounds)
        matches_used += 1

        # Track this as best if it's better than anything seen so far
        if current_score > best_ever_score:
            best_ever_score = current_score
            best_ever_strategy = current[:]

        while matches_used < matches:

            # Check all neighbors (5 neighbors for a 5-bit strategy)
            neighbors = get_neighbors(current)
            best_neighbor = None
            best_neighbor_score = -1

            for neighbor in neighbors:
                if matches_used >= matches:
                    break
                score = round_robin_score(neighbor, opponents, rounds)
                matches_used += 1

                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor = neighbor

            # Move to neighbor if it's better
            if best_neighbor_score > current_score:
                current = best_neighbor
                current_score = best_neighbor_score
            else:
                # Stuck at local optimum — restart
                break

            history.append(current_score)

            print(f"Match {matches_used} best score: {current_score}")

            # Track global best
            if current_score > best_ever_score:
                best_ever_score = current_score
                best_ever_strategy = current[:]

    return best_ever_strategy, best_ever_score, history
