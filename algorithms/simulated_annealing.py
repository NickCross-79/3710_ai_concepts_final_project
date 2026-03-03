import random
import math
from strats import random_strategy
from prisoner_dilemma import round_robin_score


def simulated_annealing(opponents, matches=2000, rounds=200,
                         initial_temp=1.0, cooling_rate=0.995):
    """
    Simulated Annealing for the Prisoner's Dilemma.

    Like hill climbing, but instead of always rejecting worse solutions,
    we accept them with probability exp(-delta / T), where T (temperature)
    decreases each step. This lets the search escape local optima early on
    while converging to a good solution as T approaches 0.

    - initial_temp: starting temperature (controls initial acceptance of bad moves)
    - cooling_rate: T is multiplied by this each step (e.g. 0.995 = slow cooling)

    Returns: best_strategy, best_score, history
    """

    # Start from a random solution
    current = random_strategy()
    current_score = round_robin_score(current, opponents, rounds)

    best_strategy = current[:]
    best_score = current_score
    history = [current_score]

    temp = initial_temp
    matches_used = 1

    while matches_used < matches:
        # Pick one random bit to flip (random neighbor)
        index = random.randint(0, len(current) - 1)
        neighbor = current[:]
        neighbor[index] = 1 - neighbor[index]

        score = round_robin_score(neighbor, opponents, rounds)
        matches_used += 1

        delta = score - current_score

        # Always accept improvements; accept worse with probability exp(delta/T)
        if delta > 0 or (temp > 1e-10 and random.random() < math.exp(delta / temp)):
            current = neighbor
            current_score = score

        # Cool down
        temp *= cooling_rate

        history.append(current_score)

        # Track global best
        if current_score > best_score:
            best_score = current_score
            best_strategy = current[:]

    return best_strategy, best_score, history
