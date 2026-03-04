"""
Memory-depth-aware strategy encoding for Iterated Prisoner's Dilemma.

Strategy format for memory depth d:
  - bits 0 .. d-1              : opening moves (one per early round)
  - bits d .. d + 2^(2d) - 1  : lookup table indexed by 2d-bit history

Compatibility table:
  Memory-3 (d=3): 3 + 64   =   67 bits  (matches strats.STRATEGY_LENGTH)
  Memory-4 (d=4): 4 + 256  =  260 bits
  Memory-5 (d=5): 5 + 1024 = 1029 bits
"""

import random

PAYOFFS = {
    (1, 1): (3, 3),
    (1, 0): (0, 5),
    (0, 1): (5, 0),
    (0, 0): (1, 1),
}


# ---------------------------------------------------------------------------
# Bitstring utilities
# ---------------------------------------------------------------------------

def strategy_length(memory_depth: int) -> int:
    """Return total number of bits for a strategy of given memory depth."""
    return memory_depth + (1 << (2 * memory_depth))


def random_strategy_md(memory_depth: int) -> list:
    """Return a random strategy bitstring for the given memory depth."""
    return [random.randint(0, 1) for _ in range(strategy_length(memory_depth))]


def copy_strategy_md(strategy: list) -> list:
    return strategy[:]


def mutate_md(strategy: list, mutation_rate: float = 0.1) -> list:
    """Flip each bit independently with probability mutation_rate."""
    new = strategy[:]
    for i in range(len(new)):
        if random.random() < mutation_rate:
            new[i] ^= 1
    return new


def crossover_md(parent_a: list, parent_b: list) -> list:
    """Single-point crossover."""
    n = len(parent_a)
    pt = random.randint(1, n - 1)
    return parent_a[:pt] + parent_b[pt:]


# ---------------------------------------------------------------------------
# Move decision
# ---------------------------------------------------------------------------

def decide_move_md(strategy: list, last_state, round_number: int,
                   memory_depth: int) -> int:
    """
    Return the move (1=cooperate, 0=defect) for the current round.

    For rounds 0 .. memory_depth-1 the opening bits are used.
    From round memory_depth onward the history lookup table is used.
    """
    if round_number < memory_depth:
        return strategy[round_number]

    my_moves, opp_moves = last_state
    my_last  = my_moves[-memory_depth:]
    opp_last = opp_moves[-memory_depth:]

    bits  = my_last + opp_last
    index = int("".join(str(b) for b in bits), 2)
    return strategy[memory_depth + index]


# ---------------------------------------------------------------------------
# Match / evaluation
# ---------------------------------------------------------------------------

def play_match_md(strategy_a: list, strategy_b: list,
                  rounds: int = 200, memory_depth: int = 3) -> tuple:
    """
    Play a full match between two strategies of the given memory depth.
    Returns (score_a, score_b).
    """
    score_a, score_b = 0, 0
    history_a: list = []
    history_b: list = []

    for r in range(rounds):
        state_a = (history_a, history_b) if r > 0 else None
        state_b = (history_b, history_a) if r > 0 else None

        move_a = decide_move_md(strategy_a, state_a, r, memory_depth)
        move_b = decide_move_md(strategy_b, state_b, r, memory_depth)

        pa, pb = PAYOFFS[(move_a, move_b)]
        score_a += pa
        score_b += pb

        history_a.append(move_a)
        history_b.append(move_b)

    return score_a, score_b


def round_robin_score_md(strategy: list, opponents: list,
                          rounds: int = 200, memory_depth: int = 3) -> float:
    """Evaluate a strategy against each opponent and return the total score."""
    total = 0
    for opp in opponents:
        s, _ = play_match_md(strategy, opp, rounds, memory_depth)
        total += s
    return total


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(strategy: list, memory_depth: int = 3) -> dict:
    """
    Return a dict of interpretable features for a single strategy.

    Always-on features (memory-depth independent):
      opening_coop_ratio      – fraction of opening bits that are C
      table_coop_ratio        – fraction of lookup-table bits that are C
      coop_after_i_coop       – P(cooperate | I cooperated last round)
      coop_after_i_defect     – P(cooperate | I defected last round)
      coop_after_opp_coop     – P(cooperate | opponent cooperated last round)
      coop_after_opp_defect   – P(cooperate | opponent defected last round)
      coop_after_mutual_coop  – P(cooperate | both cooperated last round)
      coop_after_mutual_def   – P(cooperate | both defected last round)
    """
    d  = memory_depth
    n  = 1 << (2 * d)       # number of table entries
    opening_bits = strategy[:d]
    table_bits   = strategy[d:]

    opening_coop = sum(opening_bits) / d if d else 0.0
    table_coop   = sum(table_bits)   / n if n else 0.0

    # Breakdown by last-round moves (we look at the innermost/most-recent bit)
    i_coop_idx   = []   # states where my last move was C (my_last[-1] == 1)
    i_defect_idx = []   # states where my last move was D
    o_coop_idx   = []   # states where opp last move was C
    o_defect_idx = []   # states where opp last move was D
    mutual_coop  = []
    mutual_def   = []

    for idx in range(n):
        bits = f"{idx:0{2*d}b}"
        my_last_bit  = int(bits[d - 1])   # my most-recent move
        opp_last_bit = int(bits[2*d - 1]) # opp's most-recent move

        (i_coop_idx if my_last_bit  == 1 else i_defect_idx).append(idx)
        (o_coop_idx if opp_last_bit == 1 else o_defect_idx).append(idx)
        if my_last_bit == 1 and opp_last_bit == 1:
            mutual_coop.append(idx)
        if my_last_bit == 0 and opp_last_bit == 0:
            mutual_def.append(idx)

    def _mean(indices):
        if not indices:
            return 0.0
        return sum(table_bits[i] for i in indices) / len(indices)

    return {
        "opening_coop_ratio":     opening_coop,
        "table_coop_ratio":       table_coop,
        "coop_after_i_coop":      _mean(i_coop_idx),
        "coop_after_i_defect":    _mean(i_defect_idx),
        "coop_after_opp_coop":    _mean(o_coop_idx),
        "coop_after_opp_defect":  _mean(o_defect_idx),
        "coop_after_mutual_coop": _mean(mutual_coop),
        "coop_after_mutual_def":  _mean(mutual_def),
    }


def strategy_to_string_md(strategy: list, memory_depth: int = 3) -> str:
    """Human-readable summary for any memory depth."""
    d     = memory_depth
    table = strategy[d:]
    n     = len(table)
    open_str = "".join("C" if b else "D" for b in strategy[:d])
    coop_n   = sum(table)

    lines = [
        f"Memory depth    : {d}",
        f"Strategy length : {len(strategy)} bits",
        f"Opening moves   : {open_str}",
        f"Cooperates in   : {coop_n}/{n} history states "
        f"({100*coop_n/n:.1f}%)",
    ]
    return "\n".join(lines)
