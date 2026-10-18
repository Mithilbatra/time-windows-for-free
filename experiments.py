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

E2-UP (baselines.UnbalancedQUBO) under main()'s protocol, beside main()'s rows from results/sa.csv:
Run:  python experiments.py up | tee results/sa_up_summary.txt      (≈ 70 min on 8 cores)
      python experiments.py up-summary    reprints the summary from results/sa_up.csv
"""
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per worker process

import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from baselines import SalehiQUBO, UnbalancedQUBO, anneal
from qubo import QUBOBuilder, all_routes, make_instance

ENCODINGS = ("E0", "E1", "E1+C", "E2")
SIZES = (3, 4, 5)
P1 = (1, 2, 5, 10, 20, 50)
P2 = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, float("inf"))   # inf: no window term
TUNE_SEEDS = range(10000, 10010)
EVAL_SEEDS = range(20000, 20020)
ORACLE_SEEDS = EVAL_SEEDS[:5]


def builder(encoding, inst, p1, p2):
    """Salehi et al.'s weights (A = C·p1, P = C/p2, B = 1) in the builders' units of S.
    For E2-UP p2 is the pair (λ1, λ2) of its unbalanced penalty, in cost units."""
    C, S = inst.d.max(), inst.d.max(axis=1).sum()
    if encoding == "E2-UP":
        return UnbalancedQUBO(inst, A=C * p1 / S, lam1=p2[0], lam2=p2[1])
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


def resource_counts():
    """(n, encoding) → mean (qubits, two-qubit terms with the X mixer, with the XY mixer) over
    the 20 evaluation instances, p1 = p2 = 1.  Every nonzero off-diagonal entry of Q is one ZZ
    rotation, two CNOTs, per layer; with the XY mixer our encodings drop the each-customer-once
    penalty (zero on the states that mixer reaches), and E2 has no XY count (nan)."""
    out = {}
    for n in SIZES:
        for enc in ENCODINGS:
            counts = []
            for s in EVAL_SEEDS:
                b = builder(enc, make_instance(n, s), 1, 1)
                xy = np.count_nonzero(np.triu(b.qubo(rows=False)[0], 1)) if enc != "E2" else np.nan
                counts.append((b.N, np.count_nonzero(np.triu(b.Q, 1)), xy))
            out[n, enc] = np.mean(counts, axis=0)
    return out


def resources():
    """What each encoding costs a QAOA circuit (resource_counts), as a table."""
    print(f"{'n':>2} {'enc':5} {'qubits':>7} {'ZZ terms, X mixer':>18} {'ZZ terms, XY mixer':>19}")
    for (n, enc), (qubits, zz_x, zz_xy) in resource_counts().items():
        xy = f"{zz_xy:19.1f}" if enc != "E2" else f"{'-':>19}"
        print(f"{n:>2} {enc:5} {qubits:7.0f} {zz_x:18.1f} {xy}")
        if enc == ENCODINGS[-1]:
            print()


UP_LAM = [(0, 0)] + list(itertools.product((0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30),       # (0, 0): no window term
                                           (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1)))


def unbalanced(workers=8):
    """E2-UP tuned and evaluated exactly as main() does E2: the same P1, seeds, sweeps and
    selection rule, with (λ1, λ2) from UP_LAM in place of p2 (73 window settings vs E2's 14).
    Writes results/sa_up.csv, then prints unbalanced_summary()."""
    tune = [("tune", "E2-UP", n, s, p1, lam, 2000)                  # largest first: load balance
            for n in SIZES[::-1] for s in TUNE_SEEDS for p1 in P1 for lam in UP_LAM]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(run, tune, chunksize=4))
        best = best_pairs(rows, "n")
        evaluate = [("eval", "E2-UP", n, s, *best[n,], 10000) for n in SIZES[::-1] for s in EVAL_SEEDS]
        rows += list(pool.map(run, evaluate))
    save([dict({k: v for k, v in r.items() if k != "p2"}, lam1=r["p2"][0], lam2=r["p2"][1])
          for r in rows], "results/sa_up.csv")                     # (λ1, λ2) in columns of their own
    unbalanced_summary()


def unbalanced_summary():
    """results/sa_up.csv's E2-UP beside main()'s tuned encodings (results/sa.csv), paired
    P(opt) differences, and how far E2-UP's penalty is from exact on the evaluation instances."""
    def load(path):
        with open(path) as f:
            return [{k: v if k in ("phase", "encoding") else float(v) for k, v in r.items()}
                    for r in csv.DictReader(f)]

    rows = load("results/sa.csv") + load("results/sa_up.csv")
    encodings = ENCODINGS + ("E2-UP",)
    zz = {key: v[1] for key, v in resource_counts().items()}
    for n in SIZES:
        zz[n, "E2-UP"] = np.mean([np.count_nonzero(np.triu(builder("E2-UP", make_instance(n, s), 1, (1, 1)).Q, 1))
                                  for s in EVAL_SEEDS])
    t95 = stats.t.ppf(0.975, len(EVAL_SEEDS) - 1)
    print("E2-UP: E2 (Salehi et al. 2022) with its two time-window slack registers per stop replaced by")
    print("unbalanced penalization (Montañez-Barrera et al. 2024, Eq. 25): h >= 0 costs -l1*h + l2*h^2.")
    print(f"Tuned as main() tunes E2: p1 in {P1} x (l1, l2) in {sorted({a for a, _ in UP_LAM[1:]})} x "
          f"{sorted({b for _, b in UP_LAM[1:]})} plus (0, 0) (no window term), in cost units.")
    print("Other rows: results/sa.csv.  ZZ terms: nonzero couplings, mean over the evaluation instances at")
    print("p1 = p2 = 1 (E2-UP: l1 = l2 = 1), as in results/resources.txt.\n")
    for n in SIZES:
        ev = {enc: sorted((r for r in rows if r["phase"] == "eval" and r["encoding"] == enc and r["n"] == n
                           and r["sweeps"] == 10000), key=lambda r: r["seed"]) for enc in encodings}
        print(f"n = {n} customers: tuned weights on {len(TUNE_SEEDS)} instances, evaluated on "
              f"{len(EVAL_SEEDS)} fresh ones (100 reads x 10000 sweeps)")
        print(f"  {'enc':5} {'qubits':>6} {'ZZ':>7} {'p1':>3} {'window weight':>16} {'tune P(opt)':>11} "
              f"{'P(valid)':>9} {'P(on time)':>10} {'P(opt)':>7} {'solved':>7}")
        for enc in encodings:
            r0 = ev[enc][0]
            weight = (f"l1={r0['lam1']:g} l2={r0['lam2']:g}" if enc == "E2-UP" else
                      "-" if enc == "E0" else f"p2={r0['p2']:g}")
            same = ("lam1", "lam2") if enc == "E2-UP" else ("p2",)
            tuned = np.mean([r["p_opt"] for r in rows if r["phase"] == "tune" and r["encoding"] == enc
                             and r["n"] == n and all(r[k] == r0[k] for k in ("p1",) + same)])
            opts = [r["p_opt"] for r in ev[enc]]
            print(f"  {enc:5} {np.mean([r['qubits'] for r in ev[enc]]):6.1f} {zz[n, enc]:7.1f} {r0['p1']:3g} "
                  f"{weight:>16} {tuned:11.3f} {np.mean([r['p_valid'] for r in ev[enc]]):9.3f} "
                  f"{np.mean([r['p_on_time'] for r in ev[enc]]):10.3f} {np.mean(opts):7.3f} "
                  f"{sum(x > 0 for x in opts):>4}/{len(opts)}")
        qubits = [r["qubits"] for r in ev["E2-UP"]]
        print(f"  E2-UP qubits {min(qubits):.0f}..{max(qubits):.0f}: "
              f"{sum(q <= 25 for q in qubits)}/{len(qubits)} instances fit exact QAOA simulation (<= 25 qubits)")
        print("  paired P(opt) differences over the same instances, mean [95% t-interval]:")
        for a, b in (("E2-UP", "E2"), ("E1", "E2-UP"), ("E1+C", "E2-UP")):
            d = np.array([ra["p_opt"] - rb["p_opt"] for ra, rb in zip(ev[a], ev[b])])
            h = t95 * d.std(ddof=1) / np.sqrt(len(d))
            print(f"    {a:5} - {b:5} {d.mean():+.3f} [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}]")
        # exactness at the tuned (l1, l2): each route's least energy over the waiting bits
        r0 = ev["E2-UP"][0]
        nonzero, false_pos, unpen, gs_opt, gs_on_time, rank = [], [], [], [], [], []
        for s in EVAL_SEEDS:
            inst = make_instance(n, s)
            q = UnbalancedQUBO(inst, lam1=r0["lam1"], lam2=r0["lam2"])
            perms, feasible, cost = all_routes(inst)
            pen = np.array([q.window_penalty([0, *p]) for p in perms])
            E = q.B * cost + pen
            optimal = feasible & np.isclose(cost, cost[feasible].min())
            ground = np.isclose(E, E.min())
            nonzero.append((np.abs(pen[feasible]) > 1e-9).mean())
            false_pos.append((pen[feasible] > 1e-9).mean())
            if (~feasible).any():
                unpen.append((pen[~feasible] <= 1e-9).mean())
            gs_opt.append(optimal[ground].all())
            gs_on_time.append(feasible[ground].all())
            rank.append((E < E[optimal].min() - 1e-9).sum())
        print(f"  E2-UP penalty at the tuned weights, every route of the {len(EVAL_SEEDS)} evaluation instances,"
              " least over waiting times:")
        print(f"    feasible routes with a nonzero penalty {np.mean(nonzero):.1%}, penalised (> 0) "
              f"{np.mean(false_pos):.1%}; infeasible routes unpenalised (<= 0) {np.mean(unpen):.1%} "
              f"({len(unpen)} instances with a late route)")
        print(f"    lowest-energy route is the TSPTW optimum in {sum(gs_opt)}/{len(gs_opt)} instances, on time in "
              f"{sum(gs_on_time)}/{len(gs_on_time)}; routes below the optimum median {np.median(rank):.0f}, "
              f"max {max(rank)}")
        print()


if __name__ == "__main__":
    {"tightness": tightness, "resources": resources, "up": unbalanced, "up-summary": unbalanced_summary}.get(
        sys.argv[1] if sys.argv[1:] else "", main)()
