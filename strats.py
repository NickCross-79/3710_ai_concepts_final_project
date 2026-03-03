import random

# 1 = Cooperate
# 0 = Defect

# Strategy format:
# 64-bit string representing all possible (my_last_3, opp_last_3) combinations
# Plus 6 bits for the first 3 moves (before we have full history)
# Total = 70 bits

STRATEGY_LENGTH = 67  # 3 opening-move bits + 64 bits for history lookup


def random_strategy():
    """Generate a random STRATEGY_LENGTH strategy."""
    return [random.randint(0, 1) for _ in range(STRATEGY_LENGTH)]


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
    Determine the next move using last 3 rounds of history.

    For the first 3 rounds we use the opening bits (indices 0-5).
    After that we look up the full history in the 64-bit table (indices 6-69).

    last_state is now a tuple of the full history:
        (my_moves_list, opp_moves_list)
    """

    # First 3 rounds — use opening moves
    if round_number == 0:
        return strategy[0]
    if round_number == 1:
        return strategy[1]
    if round_number == 2:
        return strategy[2]

    my_moves, opp_moves = last_state

    # Use last 3 moves from each player
    my_last3  = my_moves[-3:]
    opp_last3 = opp_moves[-3:]

    # Convert to a single index (0-63)
    # my3 bits + opp3 bits = 6 bits = index into the 64-entry lookup table
    bits = my_last3 + opp_last3
    index = int("".join(str(b) for b in bits), 2)

    # Offset by 3 (skip the 3 opening move bits)
    return strategy[3 + index]


# -------------------------
# Baseline known strategies
# -------------------------

def always_cooperate():
    return [1] * STRATEGY_LENGTH


def always_defect():
    return [0] * STRATEGY_LENGTH


def tit_for_tat():
    """Cooperate first, then copy opponent's last move."""
    strategy = [0] * STRATEGY_LENGTH
    # Opening: cooperate for first 3 rounds
    strategy[0] = 1
    strategy[1] = 1
    strategy[2] = 1
    # For each history combination, copy opponent's last move
    # Opponent's last move is the last bit of opp_last3
    for i in range(64):
        opp_last = i & 1  # last bit = opponent's most recent move
        strategy[3 + i] = opp_last
    return strategy


def suspicious_tit_for_tat():
    """Defect first, then copy opponent's last move."""
    strategy = tit_for_tat()
    strategy[0] = 0  # defect on first move
    return strategy


def strategy_to_string(strategy):
    """Human readable decode of a 67-bit strategy."""
    opening = "".join("C" if b == 1 else "D" for b in strategy[:3])
    coop_count = sum(strategy[3:])
    total = 64

    lines = []
    lines.append(f"Opening moves : {opening}")
    lines.append(f"Cooperates in : {coop_count}/{total} history states")
    lines.append("")
    lines.append("History table (my last 3 | opp last 3 → move):")
    lines.append(f"  {'My:-3 -2 -1':>14}  {'Opp:-3 -2 -1':>14}  {'Move':>4}")
    lines.append("  " + "-" * 38)

    for i in range(64):
        bits = f"{i:06b}"
        my3  = bits[:3]
        opp3 = bits[3:]
        my_str  = "  ".join("C" if b == "1" else "D" for b in my3)
        opp_str = "  ".join("C" if b == "1" else "D" for b in opp3)
        move = "C" if strategy[3 + i] == 1 else "D"
        lines.append(f"  {my_str:>14}  {opp_str:>14}  {move:>4}")

    return "\n".join(lines)
