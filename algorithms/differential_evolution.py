import random
from strats import random_strategy
from prisoner_dilemma import round_robin_score


def _de_mutant(a, b, c, scale_factor):
    """
    Binary DE mutation: v = a + F*(b - c).

    In binary space, "difference" b_i XOR c_i is 1 when b and c disagree.
    We flip a_i with probability scale_factor whenever b and c disagree,
    modelling the continuous F*(b-c) perturbation.
    """
    mutant = a[:]
    for i in range(len(a)):
        if b[i] != c[i] and random.random() < scale_factor:
            mutant[i] = 1 - mutant[i]
    return mutant


def _de_crossover(x, mutant, cr):
    """
    Binomial crossover between target x and mutant v.
    At least one bit is always taken from the mutant (mandatory index).
    """
    trial = x[:]
    mandatory = random.randint(0, len(x) - 1)
    for i in range(len(x)):
        if i == mandatory or random.random() < cr:
            trial[i] = mutant[i]
    return trial


def differential_evolution(opponents, matches=2000, rounds=200,
                             population_size=30, scale_factor=0.8,
                             crossover_rate=0.9):
    """
    Differential Evolution for the Prisoner's Dilemma.

    For each individual x in the population, three distinct others (a, b, c)
    are chosen at random. A mutant vector v is created by perturbing a in the
    direction of (b - c) scaled by F. A trial vector u is formed by crossing
    x and v. If u scores better than x, it replaces x in the next generation.

    This population-level competition makes DE more systematic than random
    search while avoiding the parent-selection pressure of a standard GA.

    - scale_factor (F):   Controls the amplification of differences (0-2).
    - crossover_rate (CR): Fraction of bits taken from the mutant per trial.

    Returns: best_strategy, best_score, history
    """

    evals = 0

    # Initialise population and score it
    population = [random_strategy() for _ in range(population_size)]
    scores = []
    for s in population:
        scores.append(round_robin_score(s, opponents, rounds))
        evals += 1

    best_idx = scores.index(max(scores))
    best_strategy = population[best_idx][:]
    best_score = scores[best_idx]
    history = [best_score]

    while evals < matches:
        for i in range(population_size):
            if evals >= matches:
                break

            # Select three distinct individuals different from i
            candidates = [j for j in range(population_size) if j != i]
            a_idx, b_idx, c_idx = random.sample(candidates, 3)
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Mutation + crossover
            mutant = _de_mutant(a, b, c, scale_factor)
            trial = _de_crossover(population[i], mutant, crossover_rate)

            # Greedy selection
            trial_score = round_robin_score(trial, opponents, rounds)
            evals += 1

            if trial_score >= scores[i]:
                population[i] = trial
                scores[i] = trial_score

            # Track global best
            if scores[i] > best_score:
                best_score = scores[i]
                best_strategy = population[i][:]

        history.append(best_score)

    return best_strategy, best_score, history
