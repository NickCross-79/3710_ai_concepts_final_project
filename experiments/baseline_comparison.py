"""
experiments/baseline_comparison.py
=====================================
Compares the best GA-evolved strategy against each classic IPD strategy
over 1000 rounds.

Metrics:
  - Total payoff (evolved strategy)
  - Average payoff per round
  - Cooperation rate of the evolved strategy in each match

Outputs:
  results/baseline_comparison.csv
  Console table

Can be run standalone:
  python -m experiments.baseline_comparison
"""

import os
import sys
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prisoner_dilemma import round_robin_score
from strats            import random_strategy, mutate, copy_strategy
from strategies.classic_strategies import CLASSIC_STRATEGIES

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

COMPARISON_ROUNDS = 1000
GA_POP_SIZE       = 50
GA_MUTATION_RATE  = 0.1
GA_GENERATIONS    = 100
GA_ROUNDS         = 50
N_RANDOM_OPPS     = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_opponents():
    """Build the standard memory-3 opponent pool."""
    from strats import (always_cooperate, always_defect,
                        tit_for_tat, suspicious_tit_for_tat)
    pool = [always_cooperate(), always_defect(),
            tit_for_tat(), suspicious_tit_for_tat()]
    for _ in range(N_RANDOM_OPPS):
        pool.append(random_strategy())
    return pool


def _crossover(a, b):
    import random
    pt = random.randint(1, len(a) - 1)
    return a[:pt] + b[pt:]


def run_ga(opponents, population_size=GA_POP_SIZE,
           mutation_rate=GA_MUTATION_RATE,
           n_generations=GA_GENERATIONS,
           rounds=GA_ROUNDS, seed=None, verbose=True):
    """
    Convenience wrapper: run memory-3 GA and return (best_strategy, score).
    """
    import random as _rng

    if seed is not None:
        _rng.seed(seed)

    population = [random_strategy() for _ in range(population_size)]

    for gen in range(n_generations):
        scored = [(s, round_robin_score(s, opponents, rounds))
                  for s in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        if verbose and (gen % 20 == 0 or gen == n_generations - 1):
            print(f"  GA gen {gen+1:>4}/{n_generations}: "
                  f"best={scored[0][1]:.1f}")

        survivors = [s for s, _ in scored[:population_size // 2]]
        next_gen  = survivors[:]
        while len(next_gen) < population_size:
            pa = _rng.choice(survivors)
            pb = _rng.choice(survivors)
            child = _crossover(pa, pb)
            child = mutate(child, mutation_rate)
            next_gen.append(child)
        population = next_gen

    final  = [(s, round_robin_score(s, opponents, rounds))
               for s in population]
    best   = max(final, key=lambda x: x[1])
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Match analysis
# ---------------------------------------------------------------------------

def analyse_match(evolved, classic, rounds=COMPARISON_ROUNDS):
    """
    Play `rounds` rounds between evolved and classic strategy.

    Returns dict with:
      total_score      – evolved strategy total
      avg_per_round    – total_score / rounds
      coop_rate        – fraction of rounds where evolved cooperated
      opp_coop_rate    – fraction of rounds opponent cooperated
    """
    from prisoner_dilemma import play_round
    from strats import decide_move

    score_e = 0
    history_e: list = []
    history_c: list = []

    for r in range(rounds):
        state_e = (history_e, history_c) if r > 0 else None
        state_c = (history_c, history_e) if r > 0 else None

        move_e = decide_move(evolved,  state_e, r)
        move_c = decide_move(classic,  state_c, r)

        pe, pc = play_round(move_e, move_c)
        score_e += pe

        history_e.append(move_e)
        history_c.append(move_c)

    coop_e = sum(history_e) / rounds
    coop_c = sum(history_c) / rounds

    return {
        "total_score":   score_e,
        "avg_per_round": score_e / rounds,
        "coop_rate":     coop_e,
        "opp_coop_rate": coop_c,
    }


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(evolved_strategy=None,
              classic_registry=None,
              comparison_rounds=COMPARISON_ROUNDS,
              ga_generations=GA_GENERATIONS,
              ga_pop_size=GA_POP_SIZE,
              ga_mutation_rate=GA_MUTATION_RATE,
              ga_rounds=GA_ROUNDS,
              results_dir="results",
              seed=None,
              verbose=True):
    """
    Run the baseline comparison experiment.

    Parameters
    ----------
    evolved_strategy : list or None
        Pre-evolved strategy bitstring to use.  If None the function runs a
        fresh GA to find one.
    classic_registry : dict or None
        {name: builder_fn} mapping.  Defaults to CLASSIC_STRATEGIES.
    comparison_rounds : int
        Number of rounds per head-to-head match.
    ...

    Returns
    -------
    results : list of dicts
    evolved_strategy : the strategy used
    """
    import random as _rng

    if seed is not None:
        _rng.seed(seed)

    classic_registry = classic_registry or CLASSIC_STRATEGIES
    os.makedirs(results_dir, exist_ok=True)

    if verbose:
        print("\n===== Baseline Comparison =====")

    # ── Step 1: obtain evolved strategy ──────────────────────────────────────
    if evolved_strategy is None:
        if verbose:
            print("\n  [1/2] Training GA to find best strategy …")
        opponents = _build_opponents()
        t0 = time.time()
        evolved_strategy, evo_score = run_ga(
            opponents,
            population_size=ga_pop_size,
            mutation_rate=ga_mutation_rate,
            n_generations=ga_generations,
            rounds=ga_rounds,
            seed=seed,
            verbose=verbose,
        )
        if verbose:
            print(f"  Best evolved score (tournament): {evo_score:.1f}  "
                  f"({time.time()-t0:.1f}s)")
    else:
        if verbose:
            print("  Using provided evolved strategy.")

    # ── Step 2: head-to-head vs each classic strategy ─────────────────────
    if verbose:
        print(f"\n  [2/2] Running {comparison_rounds}-round matches …")

    records = []
    for name, builder in classic_registry.items():
        classic = builder()
        res = analyse_match(evolved_strategy, classic,
                            rounds=comparison_rounds)
        res["opponent"] = name
        records.append(res)
        if verbose:
            print(f"    vs {name:<25} | "
                  f"total={res['total_score']:>6}  "
                  f"avg/rnd={res['avg_per_round']:>5.3f}  "
                  f"coop={res['coop_rate']:.2%}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "baseline_comparison.csv")
    fieldnames = ["opponent", "total_score", "avg_per_round",
                  "coop_rate", "opp_coop_rate"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    if verbose:
        print(f"\n  Saved -> {csv_path}")
        _print_table(records, comparison_rounds)

    return records, evolved_strategy


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

def _print_table(records, rounds):
    print(f"\n--- Evolved Strategy vs. Classic Strategies ({rounds} rounds) ---")
    hdr = (f"{'Opponent':<28} | {'Total':>7} {'Avg/Rnd':>8} "
           f"{'EvolCoop':>9} {'OppCoop':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        print(f"{r['opponent']:<28} | "
              f"{r['total_score']:>7}  "
              f"{r['avg_per_round']:>8.3f}  "
              f"{r['coop_rate']:>8.2%}  "
              f"{r['opp_coop_rate']:>7.2%}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_study(seed=42, verbose=True)
