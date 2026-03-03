import random
from strats import random_strategy, mutate, copy_strategy
from prisoner_dilemma import round_robin_score


def _crossover(parent_a, parent_b):
    """Single-point crossover."""
    point = random.randint(1, len(parent_a) - 1)
    return parent_a[:point] + parent_b[point:]


def _local_polish(strategy, opponents, rounds, local_iters, evals_counter):
    """
    Brief hill climb: try `local_iters` random single-bit flips and keep any
    improvement. Returns (polished_strategy, score, evals_used).
    """
    current = strategy[:]
    current_score = round_robin_score(current, opponents, rounds)
    evals = 1

    for _ in range(local_iters):
        idx = random.randint(0, len(current) - 1)
        neighbor = current[:]
        neighbor[idx] = 1 - neighbor[idx]
        s = round_robin_score(neighbor, opponents, rounds)
        evals += 1
        if s > current_score:
            current = neighbor
            current_score = s

    return current, current_score, evals


def memetic_algorithm(opponents, matches=2000, rounds=200,
                       population_size=30, mutation_rate=0.1, local_iters=5):
    """
    Memetic Algorithm for the Prisoner's Dilemma.

    A Genetic Algorithm (selection + crossover + mutation) where each new
    offspring undergoes a short local hill climb before being added to the
    next generation. This combines the global exploration of a GA with the
    local exploitation of hill climbing.

    - population_size: number of strategies per generation
    - mutation_rate:   per-bit flip probability after crossover
    - local_iters:     number of random bit-flip attempts per local polish

    Returns: best_strategy, best_score, history
    """

    evals = 0

    # Initialise and score the initial population
    population = [random_strategy() for _ in range(population_size)]
    scored = []
    for s in population:
        if evals >= matches:
            break
        sc = round_robin_score(s, opponents, rounds)
        evals += 1
        scored.append((s, sc))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_strategy = scored[0][0][:]
    best_score = scored[0][1]
    history = [best_score]

    while evals < matches:
        # Selection: keep the top half as parents
        survivors = [s for s, _ in scored[:population_size // 2]]

        # Build next generation: survivors carry over, rest are new children
        next_scored = list(scored[:population_size // 2])

        while len(next_scored) < population_size and evals < matches:
            parent_a = random.choice(survivors)
            parent_b = random.choice(survivors)
            child = _crossover(parent_a, parent_b)
            child = mutate(child, mutation_rate)

            # Local polish
            polished, pol_score, used = _local_polish(
                child, opponents, rounds, local_iters, evals
            )
            evals += used
            next_scored.append((polished, pol_score))

        scored = sorted(next_scored, key=lambda x: x[1], reverse=True)

        if scored[0][1] > best_score:
            best_score = scored[0][1]
            best_strategy = scored[0][0][:]

        history.append(best_score)

    return best_strategy, best_score, history
