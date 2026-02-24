import random

# 1 = Cooperate
# 0 = Defect

# Strategy format:
# [start, CC, CD, DC, DD]


def random_strategy():
    """Generate a random 5-bit strategy."""
    return [random.randint(0, 1) for _ in range(5)]


def copy_strategy(strategy):
    """Return a copy of a strategy."""
    return strategy[:]


def flip_bit(strategy, index):
    """Flip a specific bit in the strategy."""
    new_strategy = copy_strategy(strategy)
    new_strategy[index] = 1 - new_strategy[index]
    return new_strategy


def get_neighbors(strategy):
    """
    Generate all neighboring strategies
    (used for hill climbing).
    """
    neighbors = []
    for i in range(len(strategy)):
        neighbors.append(flip_bit(strategy, i))
    return neighbors


def mutate(strategy, mutation_rate=0.1):
    """
    Random mutation used in genetic algorithms.
    """
    new_strategy = copy_strategy(strategy)

    for i in range(len(new_strategy)):
        if random.random() < mutation_rate:
            new_strategy[i] = 1 - new_strategy[i]

    return new_strategy


def decide_move(strategy, last_state, round_number):
    """
    Determine the next move.

    last_state:
        None for first round
        otherwise tuple like (my_last_move, opponent_last_move)
    """

    # First move
    if round_number == 0 or last_state is None:
        return strategy[0]

    my_last, opp_last = last_state

    if my_last == 1 and opp_last == 1:
        return strategy[1]  # CC
    elif my_last == 1 and opp_last == 0:
        return strategy[2]  # CD
    elif my_last == 0 and opp_last == 1:
        return strategy[3]  # DC
    else:
        return strategy[4]  # DD


# -------------------------
# Baseline known strategies
# -------------------------

def always_cooperate():
    return [1, 1, 1, 1, 1]


def always_defect():
    return [0, 0, 0, 0, 0]


def tit_for_tat():
    """
    Cooperate first, then copy opponent.
    """
    return [1, 1, 0, 1, 0]


def suspicious_tit_for_tat():
    """
    Defect first, then copy opponent.
    """
    return [0, 1, 0, 1, 0]


def random_strategy_policy():
    """
    Strategy that behaves randomly each move.
    (Used only for testing environments)
    """
    return [random.randint(0, 1) for _ in range(5)]


def strategy_to_string(strategy):
    """Human readable format."""
    labels = ["Start", "CC", "CD", "DC", "DD"]
    parts = []
    for label, bit in zip(labels, strategy):
        move = "C" if bit == 1 else "D"
        parts.append(f"{label}:{move}")
    return " | ".join(parts)