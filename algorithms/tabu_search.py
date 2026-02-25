import random
from strats import random_strategy, get_neighbors
from prisoner_dilemma import round_robin_score


def tabu_search(opponents, matches=2000, rounds=200, tabu_size=10):
    """
    Tabu Search for the Prisoner's Dilemma.

    Like hill climbing, but with a "blacklist" (tabu list) of recently
    visited strategies. This stops it from going in circles and helps
    it escape local optima by allowing non-improving moves when needed.

    Returns: best_strategy, best_score, history
    """

    # Start from a random strategy
    current = random_strategy()
    current_score = round_robin_score(current, opponents, rounds)

    best_ever_strategy = current[:]
    best_ever_score = current_score
    history = [current_score]

    # Tabu list stores recently visited strategies (as tuples so they're hashable)
    tabu_list = [tuple(current)]

    matches_used = 1

    while matches_used < matches:

        neighbors = get_neighbors(current)
        best_neighbor = None
        best_neighbor_score = -1

        for neighbor in neighbors:
            if matches_used >= matches:
                break

            # Skip if this neighbor is on the blacklist
            if tuple(neighbor) in tabu_list:
                continue

            score = round_robin_score(neighbor, opponents, rounds)
            matches_used += 1

            if score > best_neighbor_score:
                best_neighbor_score = score
                best_neighbor = neighbor

        # If all neighbors were tabu, pick a random one to escape
        if best_neighbor is None:
            best_neighbor = random.choice(neighbors)
            best_neighbor_score = round_robin_score(best_neighbor, opponents, rounds)
            matches_used += 1

        # Move to best neighbor (even if it's worse — that's the key difference from hill climbing)
        current = best_neighbor
        current_score = best_neighbor_score

        # Add to tabu list, remove oldest if list is full
        tabu_list.append(tuple(current))
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        history.append(current_score)

        print(f"Match {matches_used} best score: {current_score}")

        # Track global best
        if current_score > best_ever_score:
            best_ever_score = current_score
            best_ever_strategy = current[:]

    return best_ever_strategy, best_ever_score, history
