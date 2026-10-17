"""
Experiments
===========
Simulated-annealing comparison of the encodings E0, E1, E1+C (qubo.py) and Salehi et al.'s
E2 (baselines.py), with the penalty weights tuned the same way for every encoding:

  weights  Salehi et al. (2022) §4.3: distance weight 1, route-validity weight C·p1 and
           time-window weight C/p2, where C is the largest travel cost
  tune     the same (p1, p2) grid for every encoding (p2 = ∞ switches the time-window term
           off), on 10 tuning instances per size (seeds 10000+), SA with 100 reads × 2000
           sweeps; keep the pair with the highest mean P(opt).  p1 ≥ 1 keeps the validity
           weight at least the largest edge cost (Salehi et al.'s range); p1 = 0.25, 0.5 never
           helped on held-out instances in a trial run, and for E2 at n = 5 p1 = 0.25 won the
           tuning but left 1% of held-out reads valid
  eval     that pair on 20 fresh instances per size (seeds 20000+), 100 reads × 10000
           sweeps (Salehi et al.'s SA budget)
  oracle   E2's best pair chosen per instance on 5 of the evaluation instances, knowing the
           optimum, as Salehi et al. did: E2's best case
  budget   E1 and E2 with their tuned pairs at 2000 and 50000 sweeps on those 5 instances:
           does a bigger SA budget change the picture?

P(opt) = fraction of reads that end in an optimal on-time route.  Writes results/sa.csv.
Run:  python experiments.py | tee results/sa_summary.txt      (≈ 45 min on 10 cores)

The tightness sweep (below main) repeats this on planted-window instances of five widths,
re-tuning every encoding's weights at each width:
Run:  python experiments.py tightness | tee results/sa_tightness_summary.txt   (≈ 1 h on 6 cores)

Circuit cost of each encoding (qubits, two-qubit terms):
Run:  python experiments.py resources | tee results/resources.txt
"""
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per worker process

import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from baselines import SalehiQUBO, anneal
from qubo import QUBOBuilder, all_routes, make_instance

ENCODINGS = ("E0", "E1", "E1+C", "E2")
SIZES = (3, 4, 5)
P1 = (1, 2, 5, 10, 20, 50)
P2 = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, float("inf"))   # inf: no window term
TUNE_SEEDS = range(10000, 10010)
EVAL_SEEDS = range(20000, 20020)
ORACLE_SEEDS = EVAL_SEEDS[:5]


def builder(encoding, inst, p1, p2):
    """Salehi et al.'s weights (A = C·p1, P = C/p2, B = 1) in the builders' units of S."""
    C, S = inst.d.max(), inst.d.max(axis=1).sum()
    A, P = C * p1 / S, C / p2 / S
    if encoding == "E2":
        return SalehiQUBO(inst, A=A, P=P)
    return QUBOBuilder(inst, A=A, P=P, tw=encoding)


def grid(encoding, p1s=P1, p2s=P2):
    return itertools.product(p1s, p2s if encoding != "E0" else (1,))   # E0 has no window term


def best_pairs(rows, *fields):
    """Values of `fields` → the (p1, p2) whose tuning rows have the highest mean
    (P(opt), P(on time), P(valid)), compared in that order."""
    scores = {}
    for r in rows:
        if r["phase"] == "tune":
            scores.setdefault(tuple(r[f] for f in fields), {}).setdefault((r["p1"], r["p2"]), []).append(
                (r["p_opt"], r["p_on_time"], r["p_valid"]))
    return {k: max(v, key=lambda pw: tuple(np.mean(v[pw], axis=0))) for k, v in scores.items()}


def save(rows, path):
    os.makedirs("results", exist_ok=True)
    with open(path, "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def run(job):
    """One SA run: (phase, encoding, n, seed, p1, p2, sweeps[, width]) → result row.
    With a width the instance has planted windows of that width (qubo.make_instance)."""
    phase, encoding, n, seed, p1, p2, sweeps, *rest = job
    width = rest[0] if rest else None
    inst = make_instance(n, seed, width)
    _, feasible, cost = all_routes(inst)
    opt = cost[feasible].min()
    b = builder(encoding, inst, p1, p2)
    samples, _ = anneal(b.Q, b.offset, reads=100, sweeps=sweeps, seed=seed)
    valid = np.array([b.is_valid_route(x) for x in samples])
    routes = [b.decode(x) for x in samples]
    on_time = np.array([v and not inst.late_stops(r) for v, r in zip(valid, routes)])
    optimal = np.array([t and np.isclose(inst.tour_cost(r), opt) for t, r in zip(on_time, routes)])
    row = dict(phase=phase, encoding=encoding, n=n, seed=seed, p1=p1, p2=p2, sweeps=sweeps,
               qubits=b.N, p_valid=valid.mean(), p_on_time=on_time.mean(), p_opt=optimal.mean())
    if width is not None:
        row.update(width=width, feasible_frac=feasible.mean(),
                   binding=not feasible[np.isclose(cost, cost.min())].any())   # TSP optimum is late
    return row


def main():
    tune = [("tune", enc, n, s, p1, p2, 2000)
            for enc in ENCODINGS for n in SIZES for s in TUNE_SEEDS for p1, p2 in grid(enc)]
    oracle = [("oracle", "E2", n, s, p1, p2, 2000)
              for n in SIZES for s in ORACLE_SEEDS for p1, p2 in grid("E2")]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run, tune + oracle, chunksize=4))
        best = best_pairs(rows, "encoding", "n")
        budget = [("budget", enc, n, s, *best[enc, n], sweeps)
                  for sweeps in (50000, 2000) for enc in ("E2", "E1") for n in SIZES[::-1]
                  for s in ORACLE_SEEDS]                     # longest jobs first
        evaluate = [("eval", enc, n, s, *best[enc, n], 10000)
                    for enc in ENCODINGS for n in SIZES for s in EVAL_SEEDS]
        rows += list(pool.map(run, budget + evaluate))
    save(rows, "results/sa.csv")

    def mean(phase, enc, n, key, **match):
        vals = [r[key] for r in rows if r["phase"] == phase and r["encoding"] == enc and r["n"] == n
                and all(r[k] == v for k, v in match.items())]
        return np.mean(vals), vals

    for n in SIZES:
        print(f"n = {n} customers: tuned weights on {len(TUNE_SEEDS)} instances, evaluated on "
              f"{len(EVAL_SEEDS)} fresh ones (100 reads x 10000 sweeps)")
        print(f"  {'enc':5} {'qubits':>6} {'p1':>3} {'p2':>4} {'tune P(opt)':>11} "
              f"{'P(valid)':>9} {'P(on time)':>10} {'P(opt)':>7} {'solved':>7}")
        for enc in ENCODINGS:
            p1, p2 = best[enc, n]
            tuned, _ = mean("tune", enc, n, "p_opt", p1=p1, p2=p2)
            q, _ = mean("eval", enc, n, "qubits")
            v, _ = mean("eval", enc, n, "p_valid")
            t, _ = mean("eval", enc, n, "p_on_time")
            o, opts = mean("eval", enc, n, "p_opt")
            print(f"  {enc:5} {q:6.0f} {p1:3} {p2:4} {tuned:11.3f} {v:9.3f} {t:10.3f} {o:7.3f} "
                  f"{sum(x > 0 for x in opts):>4}/{len(opts)}")
        oracle_best = [max(r["p_opt"] for r in rows if r["phase"] == "oracle" and r["n"] == n
                           and r["seed"] == s) for s in ORACLE_SEEDS]
        print(f"  E2 oracle (best pair per instance, 2000 sweeps): mean P(opt) "
              f"{np.mean(oracle_best):.3f}, solved {sum(x > 0 for x in oracle_best)}/{len(oracle_best)}")
        for enc in ("E1", "E2"):
            curve = [np.mean([r["p_opt"] for r in rows if r["phase"] == phase and r["sweeps"] == sw
                              and r["encoding"] == enc and r["n"] == n and r["seed"] in ORACLE_SEEDS])
                     for phase, sw in (("budget", 2000), ("eval", 10000), ("budget", 50000))]
            print(f"  {enc} P(opt) at 2000 / 10000 / 50000 sweeps (same 5 instances): "
                  + " / ".join(f"{c:.3f}" for c in curve))
        print()


WIDTHS = (0, 6, 12, 24, 48)                  # planted-window half-widths, tight → loose
SWEEP_P1 = (1, 2, 5)
SWEEP_P2 = (0.25, 1, 5, 20, 100, 1000, float("inf"))
SWEEP_TUNE = range(30000, 30005)
SWEEP_EVAL = range(40000, 40020)


def tightness(workers=6):
    """How the encodings fare as the windows tighten.  Planted-window instances (width
    from WIDTHS, same distances and tour at every width); at each (width, size) every
    encoding gets its own (p1, p2), chosen on 5 tuning instances from a coarser grid than
    main()'s, then 20 fresh instances at 10000 sweeps.  Writes results/sa_tightness.csv."""
    def biggest_first(jobs):                  # E2 (most variables), then larger n: load balance
        return sorted(jobs, key=lambda j: (j[1] == "E2", j[2]), reverse=True)

    tune = biggest_first([("tune", enc, n, s, p1, p2, 2000, w) for w in WIDTHS for enc in ENCODINGS
                          for n in SIZES for s in SWEEP_TUNE for p1, p2 in grid(enc, SWEEP_P1, SWEEP_P2)])
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(run, tune, chunksize=4))
        save(rows, "results/sa_tightness.csv")
        best = best_pairs(rows, "width", "encoding", "n")
        evaluate = biggest_first([("eval", enc, n, s, *best[w, enc, n], 10000, w) for w in WIDTHS
                                  for enc in ENCODINGS for n in SIZES for s in SWEEP_EVAL])
        rows += list(pool.map(run, evaluate, chunksize=2))
    save(rows, "results/sa_tightness.csv")

    def cell(w, enc, n, key):
        return np.mean([r[key] for r in rows if r["phase"] == "eval" and
                        (r["width"], r["encoding"], r["n"]) == (w, enc, n)])

    for n in SIZES:
        print(f"n = {n} customers, planted windows, weights re-tuned at every width "
              f"(5 tuning + 20 fresh instances, 100 reads x 10000 sweeps)")
        print(f"  {'width':>5} {'feasible':>9} {'binding':>8} | P(opt):"
              + "".join(f"{enc:>7}" for enc in ENCODINGS) + " | P(on time):"
              + "".join(f"{enc:>7}" for enc in ENCODINGS))
        for w in WIDTHS:
            print(f"  {w:>5} {cell(w, 'E0', n, 'feasible_frac'):9.3f} {cell(w, 'E0', n, 'binding'):8.2f} |"
                  f"{'':8}" + "".join(f"{cell(w, enc, n, 'p_opt'):7.3f}" for enc in ENCODINGS)
                  + f" |{'':12}" + "".join(f"{cell(w, enc, n, 'p_on_time'):7.3f}" for enc in ENCODINGS))
        print()


def resources():
    """What each encoding costs a QAOA circuit: qubits and two-qubit terms (every nonzero
    off-diagonal entry of Q is one ZZ rotation, two CNOTs, per layer), means over the 20
    evaluation instances, with p1 = p2 = 1.  Our encodings with the XY mixer drop the
    each-customer-once penalty (zero on the states the mixer reaches)."""
    print(f"{'n':>2} {'enc':5} {'qubits':>7} {'ZZ terms, X mixer':>18} {'ZZ terms, XY mixer':>19}")
    for n in SIZES:
        for enc in ENCODINGS:
            qubits, zz_x, zz_xy = [], [], []
            for s in EVAL_SEEDS:
                b = builder(enc, make_instance(n, s), 1, 1)
                qubits.append(b.N)
                zz_x.append(np.count_nonzero(np.triu(b.Q, 1)))
                zz_xy.append(np.count_nonzero(np.triu(b.qubo(rows=False)[0], 1)) if enc != "E2" else np.nan)
            xy = f"{np.mean(zz_xy):19.1f}" if enc != "E2" else f"{'-':>19}"
            print(f"{n:>2} {enc:5} {np.mean(qubits):7.0f} {np.mean(zz_x):18.1f} {xy}")
        print()


if __name__ == "__main__":
    {"tightness": tightness, "resources": resources}.get(sys.argv[1] if sys.argv[1:] else "", main)()
