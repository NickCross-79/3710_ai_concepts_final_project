import random
from strats import random_strategy, mutate, copy_strategy
from prisoner_dilemma import round_robin_score


def crossover(parent_a, parent_b):
    """Mix two strategies at a random point."""
    point = random.randint(1, len(parent_a) - 1)
    child = parent_a[:point] + parent_b[point:]
    return child


def genetic_algorithm(opponents, matches=2000, rounds=200,
                       population_size=50, mutation_rate=0.1):
    """
    Genetic Algorithm for the Prisoner's Dilemma.

    - Start with a population of random strategies
    - Score each one against the opponents
    - Keep the best half (selection)
    - Breed them together (crossover)
    - Randomly tweak some bits (mutation)
    - Repeat for `matches` generations

    Returns: best_strategy, best_score, history
    """

    # Start with random population
    population = [random_strategy() for _ in range(population_size)]
    history = []

    for generation in range(matches):

        # Score everyone
        scored = [(s, round_robin_score(s, opponents, rounds)) for s in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_strategy = scored[0][0]
        best_score = scored[0][1]
        history.append(best_score)

        # Keep the top half (survivors)
        survivors = [s for s, _ in scored[:population_size // 2]]

        # Build next generation via crossover + mutation
        next_generation = survivors[:]  # survivors carry over
        while len(next_generation) < population_size:
            parent_a = random.choice(survivors)
            parent_b = random.choice(survivors)
            child = crossover(parent_a, parent_b)
            child = mutate(child, mutation_rate)
            next_generation.append(child)

        population = next_generation


    # Final best
    final_scored = [(s, round_robin_score(s, opponents, rounds)) for s in population]
    best = max(final_scored, key=lambda x: x[1])

    return best[0], best[1], history
