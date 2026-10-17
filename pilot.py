"""
QAOA pilot (n = 3 and 4), run on FastQAOA before the real grid is fixed.

  A  optimiser:  how do Adam from random angles (what the first runs used), L-BFGS from random
                 angles and the INTERP depth ladder compare, in P(opt) and in function evaluations?
                 (E1, weights p1 = p2 = 1, XY and X mixers)
  B  weights:    P(opt) over a (p1, p2) grid, the weights of experiments.builder, for E0 / E1 /
                 E1+C, on tuning instances; the best pair is picked for each depth
  C  check:      those pairs on fresh instances: P(valid), P(on time), P(opt), X vs XY, E0 vs E1
                 vs E1+C, against the uniform state
  D  depth:      how far does P(opt) climb with the XY mixer up to p = 12, at n = 4 and 5?

Every number is exact (state vector, no shots).  gap closed = (⟨E⟩ of the uniform state − ⟨E⟩)
/ (⟨E⟩ of the uniform state − ground-state energy of the simulated Hamiltonian).
Run:  python pilot.py A | tee results/pilot_a.txt      (B, C, D likewise; C reads the CSV that B writes)
"""
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per worker process

import csv
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments import builder
from qaoa import FastQAOA
from qubo import make_instance

WORKERS = 8                                       # the machine has 10 cores
DEPTHS = (1, 2, 3, 5)


def optimise(fq, method, depths):
    """{p: (params, function evaluations)} for every p in depths."""
    if method.startswith("ladder"):                   # "ladder" or "ladder+R": R random restarts per depth
        ladder = fq.ladder(max(depths), starts=5, restarts=int(method[7:] or 0))
        return {p: (ladder[p][0], ladder[p][2]) for p in depths}
    out = {}
    for p in depths:
        if method.startswith("adam"):
            steps, starts = map(int, method[4:].split("x"))
            x = min((fq.adam(fq.random_start(p), steps) for _ in range(starts)), key=lambda r: r[1])[0]
            out[p] = (x, steps * starts)
        else:                                         # lbfgs5
            runs = [fq.lbfgs(fq.random_start(p)) for _ in range(int(method[5:]))]
            out[p] = (min(runs, key=lambda r: r[1])[0], sum(r[2] for r in runs))
    return out


def job(args):
    """One instance, one optimiser: a result row for each depth (DEPTHS unless the job lists some)."""
    stage, n, mixer, enc, p1, p2, seed, method, *depths = args
    fq = FastQAOA(builder(enc, make_instance(n, seed), p1, p2), mixer=mixer, seed=seed)
    e_uniform, e_ground = fq.energy(np.zeros(0)), fq.energies.min()
    uniform = fq.route_masses(np.zeros(0))
    rows = []
    for p, (x, nfev) in optimise(fq, method, tuple(depths) or DEPTHS).items():
        valid, on_time, optimal = fq.route_masses(x)
        e = fq.energy(x)
        rows.append(dict(stage=stage, n=n, mixer=mixer, enc=enc, p1=p1, p2=p2, seed=seed,
                         method=method, p=p, nfev=nfev, energy=e,
                         gap_closed=(e_uniform - e) / (e_uniform - e_ground), p_valid=valid,
                         p_on_time=on_time, p_opt=optimal, uniform_p_opt=uniform[2]))
    return rows


def run_jobs(jobs, path):
    with ProcessPoolExecutor(WORKERS) as pool:
        rows = [row for rows in pool.map(job, jobs) for row in rows]
    os.makedirs("results", exist_ok=True)
    with open(path, "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    return rows


def stage_a():
    cases = [(4, "x", range(60000, 60004)), (4, "xy", range(60000, 60008)),
             (3, "x", range(60100, 60110)), (3, "xy", range(60100, 60110))]
    methods = ("adam60x3", "adam200x5", "lbfgs5", "ladder", "ladder+3")
    rows = run_jobs([("A", n, mixer, "E1", 1, 1, s, m) for n, mixer, seeds in cases
                     for s in seeds for m in methods], "results/pilot_a.csv")
    for n, mixer, seeds in cases:
        sub = [r for r in rows if r["n"] == n and r["mixer"] == mixer]
        print(f"n = {n}, {mixer.upper()} mixer, E1 with p1 = p2 = 1, {len(seeds)} instances; "
              f"uniform state P(opt) = {np.mean([r['uniform_p_opt'] for r in sub]):.4f}")
        print(f"  {'':10}" + "".join(f"   p = {p}  gap closed / P(opt) / evals    " for p in DEPTHS))
        for m in methods:
            cells = []
            for p in DEPTHS:
                c = [r for r in sub if r["method"] == m and r["p"] == p]
                cells.append(f"   {np.mean([r['gap_closed'] for r in c]):6.1%} / {np.mean([r['p_opt'] for r in c]):6.3f}"
                             f" / {np.mean([r['nfev'] for r in c]):6.0f}      ")
            print(f"  {m:10}" + "".join(cells))
        print()


METHOD = "ladder+3"                               # stage A's pick
GRID1, GRID2 = (0.5, 1, 2, 5), (0.25, 1, 4)
CASES = [(3, "xy", 10), (3, "x", 10), (4, "xy", 8), (4, "x", 4)]     # size, mixer, instances


def pairs(enc):
    return [(p1, 1) for p1 in GRID1] if enc == "E0" else [(p1, p2) for p1 in GRID1 for p2 in GRID2]


def stage_b():
    rows = run_jobs([("B", n, mixer, enc, p1, p2, 61000 + i, METHOD) for n, mixer, m in CASES[::-1]
                     for enc in ("E0", "E1", "E1+C") for p1, p2 in pairs(enc) for i in range(m)],
                    "results/pilot_b.csv")
    for n, mixer, m in CASES:
        for enc in ("E0", "E1", "E1+C"):
            print(f"n = {n}, {mixer.upper()}, {enc}: mean P(opt) over {m} tuning instances; best (p1, p2) "
                  f"per depth, the default (1, 1), the median and the worst pair")
            for p in DEPTHS:
                score = {pw: np.mean([r["p_opt"] for r in rows if (r["n"], r["mixer"], r["enc"], r["p"], r["p1"], r["p2"])
                                      == (n, mixer, enc, p, *pw)]) for pw in pairs(enc)}
                best = max(score, key=score.get)
                print(f"  p = {p}: best {best} {score[best]:.3f} | default {score[(1, 1)]:.3f} | "
                      f"median {np.median(list(score.values())):.3f} | worst {min(score.values()):.3f}")
        print()


def stage_c():
    tune = list(csv.DictReader(open("results/pilot_b.csv")))
    best = {}                                         # (n, mixer, enc, p) -> (p1, p2)
    for n, mixer, _ in CASES:
        for enc in ("E0", "E1", "E1+C"):
            for p in DEPTHS:
                score = {pw: np.mean([float(r["p_opt"]) for r in tune if (int(r["n"]), r["mixer"], r["enc"], int(r["p"]),
                                      float(r["p1"]), float(r["p2"])) == (n, mixer, enc, p, *map(float, pw))])
                         for pw in pairs(enc)}
                best[n, mixer, enc, p] = max(score, key=score.get)
    chosen = sorted({(n, mixer, enc, pw) for (n, mixer, enc, _), pw in best.items()})
    sizes = {(n, mixer): 20 if (n, mixer) != (4, "x") else 8 for n, mixer, _ in CASES}
    rows = run_jobs([("C", n, mixer, enc, *pw, 62000 + i, METHOD) for n, mixer, enc, pw in chosen[::-1]
                     for i in range(sizes[n, mixer])], "results/pilot_c.csv")
    rows = [r for r in rows if best[r["n"], r["mixer"], r["enc"], r["p"]] == (r["p1"], r["p2"])]
    for n in (3, 4):
        print(f"n = {n}: fresh instances, weights chosen per depth on the tuning instances; "
              f"mean (±95% CI) over instances")
        for key, label in (("p_opt", "P(opt)"), ("p_on_time", "P(on time)"), ("p_valid", "P(valid)")):
            print(f"  {label:10}" + "".join(f"        p = {p}   " for p in DEPTHS) + "      uniform")
            for mixer in ("xy", "x"):
                for enc in ("E0", "E1", "E1+C"):
                    cells = []
                    for p in DEPTHS:
                        v = [r[key] for r in rows if (r["n"], r["mixer"], r["enc"], r["p"]) == (n, mixer, enc, p)]
                        cells.append(f"  {np.mean(v):.3f}±{1.96 * np.std(v, ddof=1) / np.sqrt(len(v)):.3f}")
                    base = {"p_opt": "uniform_p_opt"}.get(key)
                    uni = (f"{np.mean([r[base] for r in rows if (r['n'], r['mixer'], r['enc']) == (n, mixer, enc)]):10.4f}"
                           if base else "")
                    print(f"    {mixer.upper():2} {enc:5}" + "".join(cells) + uni)
        print()


def stage_d():
    """P(opt) of the XY circuit up to p = 12.  Alongside: how often the Hamiltonian's ground state
    is the TSPTW optimum, the level P(opt) can reach as p grows."""
    weights = {"E0": (1, 1), "E1": (2, 0.25), "E1+C": (1, 1)}         # stage C's picks for XY
    deep = (1, 2, 3, 5, 8, 12)
    cases = [(4, range(63000, 63010)), (5, range(63100, 63106))]
    rows = run_jobs([("D", n, "xy", enc, *weights[enc], s, METHOD, *deep) for n, seeds in cases[::-1]
                     for enc in weights for s in seeds], "results/pilot_d.csv")
    for n, seeds in cases:
        print(f"n = {n}, XY mixer, {len(seeds)} instances: P(opt) (±95% CI) by depth; uniform state "
              f"{np.mean([r['uniform_p_opt'] for r in rows if r['n'] == n]):.4f}")
        print(f"  {'':6}" + "".join(f"      p = {p:<3}  " for p in deep))
        for enc in weights:
            cells = []
            for p in deep:
                v = [r["p_opt"] for r in rows if (r["n"], r["enc"], r["p"]) == (n, enc, p)]
                cells.append(f"  {np.mean(v):.3f}±{1.96 * np.std(v, ddof=1) / np.sqrt(len(v)):.3f}")
            cap = np.mean([FastQAOA(builder(enc, make_instance(n, s), *weights[enc]), mixer="xy").ground_optimal()
                           for s in seeds])
            print(f"  {enc:6}" + "".join(cells) + f"   ground state = optimum in {cap:.0%} of instances")
        print()


if __name__ == "__main__":
    {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d}[sys.argv[1]]()
