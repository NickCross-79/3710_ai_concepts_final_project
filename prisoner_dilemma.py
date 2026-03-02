from strats import decide_move

# Payoff matrix
# (playerA, playerB)

PAYOFFS = {
    (1, 1): (3, 3),  # CC
    (1, 0): (0, 5),  # CD
    (0, 1): (5, 0),  # DC
    (0, 0): (1, 1)   # DD
}


def play_round(move_a, move_b):
    """Return payoff for a single round."""
    return PAYOFFS[(move_a, move_b)]


def play_match(strategy_a, strategy_b, rounds=200):
    """
    Play an iterated Prisoner's Dilemma match.
    Returns total scores (A, B).
    """

    score_a = 0
    score_b = 0

    history_a = []  # my moves as A
    history_b = []  # my moves as B

    for r in range(rounds):

        # Pass full history so decide_move can look back 3 rounds
        state_a = (history_a, history_b) if r > 0 else None
        state_b = (history_b, history_a) if r > 0 else None

        move_a = decide_move(strategy_a, state_a, r)
        move_b = decide_move(strategy_b, state_b, r)

        payoff_a, payoff_b = play_round(move_a, move_b)

        score_a += payoff_a
        score_b += payoff_b

        history_a.append(move_a)
        history_b.append(move_b)

    return score_a, score_b


def round_robin_score(strategy, opponents, rounds=200):
    """
    Evaluate strategy against a list of opponents.
    Returns total score.
    """

    total = 0

    for opp in opponents:
        score, _ = play_match(strategy, opp, rounds)
        total += score

    return total


def evaluate_population(population, opponents, rounds=200):
    """
    Evaluate many strategies at once.
    Returns list of (strategy, score).
    """

    results = []

    for strat in population:
        score = round_robin_score(strat, opponents, rounds)
        results.append((strat, score))

    return results


def best_strategy(results):
    """Return best strategy from results."""
    return max(results, key=lambda x: x[1])