"""
Does the time-window penalty still prune at sizes beyond QAOA simulation?

The penalties are classical, so no circuit is needed: enumerate all n! routes of each
instance and count how many late routes (some window or the return deadline missed) get a
positive penalty.  pruning_check.py does this at n <= 7 for more methods; this script
covers n = 3..9 for E1 and E1+C with a t-interval over instances.

Instances: qubo.make_instance(n, seed), Salehi et al. (2022) protocol, seeds 90000..90099
(clear of the tuning seeds 10000-70000 and the QAOA grid's 80000-80049); the same seeds
at every n.  Soundness is asserted on every instance: no on-time route is ever penalised.
An instance with no late route at all has no fraction to report; it is left out of the
caught-% columns (the count is in the "no late" column).  "opt late" is the share of
instances whose cheapest tour, ties included, misses a window (grid.py's "binding").

Run:  python pruning_large.py | tee results/pruning_large.txt      (about 2 minutes)
"""
import math

import numpy as np
from scipy import stats

from qubo import all_routes, make_instance, route_penalty, tw_terms

SIZES = range(3, 10)
SEEDS = range(90000, 90100)


def mean_ci(v):
    """Mean and half-width of the 95% t-interval, both in percent."""
    v = 100 * np.asarray(v)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def measure(n):
    caught = {"E1": [], "E1+C": []}
    on_time, opt_late, no_late = [], [], 0
    for seed in SEEDS:
        inst = make_instance(n, seed)
        perms, feas, cost = all_routes(inst)
        on_time.append(feas.mean())
        opt_late.append(not feas[np.isclose(cost, cost.min())].any())
        no_late += feas.all()
        for kind in caught:
            pen = route_penalty(perms, *tw_terms(inst, kind))
            assert not (pen[feas] > 0).any(), f"{kind} penalised an on-time route (n={n}, seed={seed})"
            if not feas.all():
                caught[kind].append((pen[~feas] > 0).mean())
    return caught, on_time, opt_late, no_late


if __name__ == "__main__":
    print(f"Late routes penalised by each encoding, {len(SEEDS)} Salehi-protocol instances per size "
          f"(seeds {SEEDS[0]}..{SEEDS[-1]}), all n! routes enumerated.\n"
          f"Mean over instances with a 95% t-interval; every on-time route unpenalised (asserted).\n")
    print(f"{'n':>2} {'routes':>7} {'on time':>8} {'opt late':>9} {'no late':>8} "
          f"{'E1 caught %':>15} {'E1+C caught %':>15}")
    for n in SIZES:
        caught, on_time, opt_late, no_late = measure(n)
        cells = [f"{m:6.1f} ± {h:4.1f}" for m, h in map(mean_ci, caught.values())]
        print(f"{n:>2} {math.factorial(n):>7} "
              f"{np.mean(on_time):8.1%} {np.mean(opt_late):9.0%} {no_late:8d} {cells[0]:>15} {cells[1]:>15}",
              flush=True)
