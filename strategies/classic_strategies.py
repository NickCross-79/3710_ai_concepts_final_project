"""
Classic hand-crafted IPD strategies.

Every function returns a bitstring representation compatible with
strategies.encoding (and with strats.py when memory_depth=3).

Classic strategies (memory-depth generic):
  always_cooperate_md   – always play C
  always_defect_md      – always play D
  tit_for_tat_md        – copy opponent's most-recent move; open with C
  tit_for_two_tats_md   – defect only after TWO consecutive opponent defections
  suspicious_tft_md     – like TFT but opens with D

For convenience the module also exposes the standard memory-3 versions that
are fully compatible with prisoner_dilemma.play_match.
"""

from strategies.encoding import strategy_length


# ---------------------------------------------------------------------------
# Generic (any memory depth)
# ---------------------------------------------------------------------------

def always_cooperate_md(memory_depth: int = 3) -> list:
    """Return a strategy that always cooperates."""
    return [1] * strategy_length(memory_depth)


def always_defect_md(memory_depth: int = 3) -> list:
    """Return a strategy that always defects."""
    return [0] * strategy_length(memory_depth)


def tit_for_tat_md(memory_depth: int = 3) -> list:
    """
    Cooperate on round 0; afterwards copy opponent's most-recent move.

    In the lookup table the opponent's most-recent move is the last bit of the
    opp_last_d block, i.e. bit (2d-1) of the 2d-bit index when written MSB→LSB.
    """
    d    = memory_depth
    n    = 1 << (2 * d)
    strat = [0] * strategy_length(d)

    # Opening: cooperate for all early rounds
    for i in range(d):
        strat[i] = 1

    # History table: copy opp's last move (LSB of the opp block = index bit 0)
    for idx in range(n):
        opp_last_bit = idx & 1        # last bit of opp_last block
        strat[d + idx] = opp_last_bit

    return strat


def suspicious_tft_md(memory_depth: int = 3) -> list:
    """Defect on round 0; afterwards copy opponent's most-recent move."""
    strat = tit_for_tat_md(memory_depth)
    strat[0] = 0           # first opening move → D
    return strat


def tit_for_two_tats_md(memory_depth: int = 3) -> list:
    """
    Cooperate unless the opponent defected on BOTH of the last two rounds.

    Requires memory_depth >= 2.  For depth-1 this falls back to TFT.
    """
    d    = memory_depth
    n    = 1 << (2 * d)
    strat = [1] * strategy_length(d)   # start from all-cooperate

    if d < 2:
        return tit_for_tat_md(d)

    # History table
    for idx in range(n):
        bits     = f"{idx:0{2*d}b}"
        opp_bits = bits[d:]                    # opponent's d most-recent moves
        # The last two opponent moves are opp_bits[-2] and opp_bits[-1]
        opp_penultimate = int(opp_bits[-2])
        opp_last        = int(opp_bits[-1])

        if opp_penultimate == 0 and opp_last == 0:
            strat[d + idx] = 0             # both were D → defect

    return strat


# ---------------------------------------------------------------------------
# Memory-3 convenience aliases (compatible with strats.py / prisoner_dilemma)
# ---------------------------------------------------------------------------

def always_cooperate()    -> list: return always_cooperate_md(3)
def always_defect()       -> list: return always_defect_md(3)
def tit_for_tat()         -> list: return tit_for_tat_md(3)
def suspicious_tft()      -> list: return suspicious_tft_md(3)
def tit_for_two_tats()    -> list: return tit_for_two_tats_md(3)


# ---------------------------------------------------------------------------
# Registry: name → builder function (memory-3)
# ---------------------------------------------------------------------------

CLASSIC_STRATEGIES = {
    "Always Cooperate":       always_cooperate,
    "Always Defect":          always_defect,
    "Tit-for-Tat":            tit_for_tat,
    "Tit-for-Two-Tats":       tit_for_two_tats,
    "Suspicious Tit-for-Tat": suspicious_tft,
}
