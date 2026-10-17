"""
Final QAOA grid
===============
Exact QAOA simulation (FastQAOA) of the encodings E0, E1 and E1+C with the XY mixer at
n = 3, 4, 5 customers and the X mixer at n = 3, 4 (n² qubits): the paper's main QAOA table.

  weights    Salehi et al.'s parametrisation as in experiments.builder: validity weight C·p1,
             window weight C/p2 (E0 has none), C = largest travel cost; p1 in P1, p2 in P2
  tune       every pair on 10 tuning instances (4 for X at n = 4, which is slow), depths 1..5;
             keep the pair with the highest P(opt) averaged over instances and depths, for
             each (mixer, size, encoding)
  eval       that pair on 50 fresh instances (20 for X at n = 4), depths 1..12
  optimiser  pilot stage A's pick: INTERP ladder, 5 random starts at p = 1 and 3 more at every
             later depth, L-BFGS with exact gradients
  summary    means with 95% t-intervals over instances; encodings compared per instance
             (paired), split by whether the instance's TSP optimum is late

Rows go to results/qaoa_grid.csv as jobs finish, and an interrupted run resumes where it
stopped.  Reruns agree to about 1e-5 in P(opt), not bit for bit: L-BFGS follows last-bit
differences in OpenBLAS's results, which vary between processes, so evaluation counts drift.  Run:  python grid.py run [workers]             (about 15 CPU hours: 2 h on 8 workers)
                python grid.py summary | tee results/qaoa_grid_summary.txt
The X mixer's chosen weights sat on the grid's edge (p1 = 20, p2 = 0.1); `edge` re-tunes with
pairs beyond it on the same instances:
                python grid.py edge | tee results/qaoa_grid_edge.txt   (about 5 min on 8 workers)
"""
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one core per worker process

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import stats

from experiments import builder, save
from qaoa import FastQAOA
from qubo import all_routes, make_instance

PATH = "results/qaoa_grid.csv"
CASES = [("xy", 3), ("xy", 4), ("xy", 5), ("x", 3), ("x", 4)]
ENCODINGS = ("E0", "E1", "E1+C")
P1 = (1, 2, 5, 10, 20)
P2 = (0.1, 0.25, 1, 4)
N_TUNE = {("x", 4): 4}                        # default 10
N_EVAL = {("x", 4): 20}                       # default 50
TUNE_DEPTH, DEPTH = 5, 12
SHOW = (1, 2, 3, 5, 8, 12)
METHOD = "ladder+3"                           # pilot stage A's pick
COST = {("x", 4): 20, ("xy", 5): 10, ("xy", 4): 3, ("x", 3): 1.5, ("xy", 3): 1}   # rough, for load balance


def optimise(fq, method, depths):
    """{p: (params, function evaluations)} for every p in depths.  method "ladder+R": INTERP
    ladder with R random restarts per depth; "adamSxR" (R Adam runs of S steps) and "lbfgsR"
    (R L-BFGS runs), both from random angles at every depth, are pilot stage A's comparisons."""
    if method.startswith("ladder"):
        ladder = fq.ladder(max(depths), starts=5, restarts=int(method[7:] or 0))
        return {p: (ladder[p][0], ladder[p][2]) for p in depths}
    out = {}
    for p in depths:
        if method.startswith("adam"):
            steps, starts = map(int, method[4:].split("x"))
            x = min((fq.adam(fq.random_start(p), steps) for _ in range(starts)), key=lambda r: r[1])[0]
            out[p] = (x, steps * starts)
        else:
            runs = [fq.lbfgs(fq.random_start(p)) for _ in range(int(method[5:]))]
            out[p] = (min(runs, key=lambda r: r[1])[0], sum(r[2] for r in runs))
    return out


def job(args):
    """(stage, n, mixer, encoding, p1, p2, seed, method, *depths) → one result row per depth."""
    stage, n, mixer, enc, p1, p2, seed, method, *depths = args
    inst = make_instance(n, seed)
    fq = FastQAOA(builder(enc, inst, p1, p2), mixer=mixer, seed=seed)
    e_uniform, e_ground = fq.energy(np.zeros(0)), fq.energies.min()
    _, feasible, cost = all_routes(inst)
    rows = []
    for p, (x, nfev) in optimise(fq, method, depths).items():
        valid, on_time, optimal = fq.route_masses(x)
        e = fq.energy(x)
        rows.append(dict(stage=stage, n=n, mixer=mixer, enc=enc, p1=p1, p2=p2, seed=seed,
                         method=method, p=p, nfev=nfev, energy=e,
                         gap_closed=(e_uniform - e) / (e_uniform - e_ground), p_valid=valid,
                         p_on_time=on_time, p_opt=optimal, uniform_p_opt=fq.route_masses(np.zeros(0))[2],
                         ground_opt=fq.ground_optimal(),
                         binding=not feasible[np.isclose(cost, cost.min())].any()))   # TSP optimum is late
    return rows


def pairs(enc):
    return [(p1, 1) for p1 in P1] if enc == "E0" else [(p1, p2) for p1 in P1 for p2 in P2]


def load():
    if not os.path.exists(PATH):
        return []
    rows = list(csv.DictReader(open(PATH)))
    for r in rows:
        for k in ("n", "seed", "p", "nfev"):
            r[k] = int(r[k])
        for k in ("p1", "p2", "energy", "gap_closed", "p_valid", "p_on_time", "p_opt",
                  "uniform_p_opt", "ground_opt"):
            r[k] = float(r[k])
        r["binding"] = r["binding"] == "True"
    return rows


def write(rows, mode="a"):
    new = mode == "w" or not os.path.exists(PATH)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, mode, newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            out.writeheader()
        out.writerows(rows)


def key(x):
    """(stage, n, mixer, encoding, p1, p2, seed) of a result row or of a job."""
    if isinstance(x, dict):
        return x["stage"], x["n"], x["mixer"], x["enc"], x["p1"], x["p2"], x["seed"]
    return x[0], x[1], x[2], x[3], float(x[4]), float(x[5]), x[6]


def best_weights(rows):
    """(mixer, n, encoding) → the (p1, p2) with the highest mean P(opt) over the tuning rows."""
    score = {}
    for r in rows:
        if r["stage"] == "tune":
            score.setdefault((r["mixer"], r["n"], r["enc"]), {}).setdefault((r["p1"], r["p2"]), []).append(r["p_opt"])
    return {k: min(v, key=lambda pw: (-np.mean(v[pw]), pw)) for k, v in score.items()}


def run(workers=8):
    rows = load()
    complete = {key(r) for r in rows if r["p"] == (TUNE_DEPTH if r["stage"] == "tune" else DEPTH)}
    if any(key(r) not in complete for r in rows):                 # a job cut off mid-write
        rows = [r for r in rows if key(r) in complete]
        if rows:
            write(rows, "w")
        else:
            os.remove(PATH)

    def finish(pool, jobs):
        todo = sorted((j for j in jobs if key(j) not in complete),
                      key=lambda j: COST[j[2], j[1]] * max(j[8:]), reverse=True)
        for f in as_completed([pool.submit(job, j) for j in todo]):
            write(f.result())

    tune = [("tune", n, mixer, enc, p1, p2, 70000 + i, METHOD, *range(1, TUNE_DEPTH + 1))
            for mixer, n in CASES for enc in ENCODINGS for p1, p2 in pairs(enc)
            for i in range(N_TUNE.get((mixer, n), 10))]
    with ProcessPoolExecutor(workers) as pool:
        finish(pool, tune)
        best = best_weights(load())
        finish(pool, [("eval", n, mixer, enc, *best[mixer, n, enc], 80000 + i, METHOD, *range(1, DEPTH + 1))
                      for mixer, n in CASES for enc in ENCODINGS for i in range(N_EVAL.get((mixer, n), 50))])


def ci(v):
    """Half-width of the 95% t-interval for the mean of v."""
    return stats.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))


def mean_ci(v):
    return f"{np.mean(v):+.3f}±{ci(v):.3f}" if len(v) > 1 else f"{np.mean(v):+.3f}"


def summary():
    rows = load()
    best = best_weights(rows)
    show = [p for p in SHOW if p <= DEPTH]
    for mixer, n in CASES:
        ev = [r for r in rows if r["stage"] == "eval" and (r["mixer"], r["n"]) == (mixer, n)
              and (r["p1"], r["p2"]) == best[mixer, n, r["enc"]]]                  # the chosen weights only
        if not ev:
            continue
        seeds = sorted({r["seed"] for r in ev})
        at = {(r["enc"], r["p"], r["seed"]): r for r in ev}

        def col(enc, p, field):
            return np.array([at[enc, p, s][field] for s in seeds])

        binding = np.array([at["E0", 1, s]["binding"] for s in seeds])
        print(f"n = {n}, {mixer.upper()} mixer: {len(seeds)} fresh instances, {binding.sum()} where the TSP optimum "
              f"is late; weights (p1, p2): " + ", ".join(f"{e} {best[mixer, n, e]}" for e in ENCODINGS))
        head = "".join(f"      p = {p:<2}  " for p in show)
        for field, label in (("p_opt", "P(opt)"), ("p_on_time", "P(on time)")):
            print(f"  {label:10}{head}" + ("   uniform" if field == "p_opt" else ""))
            for enc in ENCODINGS:
                cells = "".join(f"  {col(enc, p, field).mean():.3f}±{ci(col(enc, p, field)):.3f}" for p in show)
                uniform = np.mean([at["E0", 1, s]["uniform_p_opt"] for s in seeds])
                print(f"    {enc:6}{cells}" + (f"   {uniform:.4f}" if enc == "E0" and field == "p_opt" else ""))
        for field, label in (("p_opt", "P(opt)"), ("p_on_time", "P(on time)")):
            print(f"  {label} minus E0's, paired per instance (mean ± 95% CI)")
            for enc in ("E1", "E1+C"):
                print(f"    {enc:6}" + "".join(f"  {mean_ci(col(enc, p, field) - col('E0', p, field))}" for p in show))
        print(f"  P(opt) minus E0's at p = {DEPTH}, by whether the TSP optimum is late:")
        for enc in ("E1", "E1+C"):
            d = col(enc, DEPTH, "p_opt") - col("E0", DEPTH, "p_opt")
            print(f"    {enc:6}  late ({binding.sum()}) {mean_ci(d[binding])}  |  on time ({(~binding).sum()}) "
                  f"{mean_ci(d[~binding])}  |  late instances' share of the total gain {d[binding].sum() / d.sum():.0%}")
        print("  ground state is the optimum in: "
              + ", ".join(f"{enc} {col(enc, 1, 'ground_opt').mean():.0%}" for enc in ENCODINGS))
        print()


BEYOND = {"E0": [(50, 1)], "E1": [(50, 0.1), (50, 0.25), (10, 0.025), (20, 0.025), (50, 0.025)]}
BEYOND["E1+C"] = BEYOND["E1"]


def edge(workers=8):
    """Tuning P(opt) of the X mixer's chosen pair against pairs beyond the grid's edge, same
    tuning instances and depths.  Writes results/qaoa_grid_edge.csv."""
    jobs = [("edge", n, mixer, enc, p1, p2, 70000 + i, METHOD, *range(1, TUNE_DEPTH + 1))
            for mixer, n in CASES if mixer == "x" for enc in ENCODINGS for p1, p2 in BEYOND[enc]
            for i in range(N_TUNE.get((mixer, n), 10))]
    with ProcessPoolExecutor(workers) as pool:
        rows = [row for rows in pool.map(job, jobs) for row in rows]
    save(rows, "results/qaoa_grid_edge.csv")
    tune, best = load(), best_weights(load())
    for mixer, n in CASES:
        if mixer != "x":
            continue
        for enc in ENCODINGS:
            def score(rows, p1, p2):
                return np.mean([r["p_opt"] for r in rows if (r["n"], r["mixer"], r["enc"], r["p1"], r["p2"])
                                == (n, mixer, enc, p1, p2) and r["stage"] in ("tune", "edge")])
            b = best[mixer, n, enc]
            print(f"X n={n} {enc:5} chosen {b}: {score(tune, *b):.3f}  |  beyond the edge: "
                  + "  ".join(f"{pw}: {score(rows, *pw):.3f}" for pw in BEYOND[enc]))


if __name__ == "__main__":
    {"run": run, "summary": summary, "edge": edge}[sys.argv[1]](*map(int, sys.argv[2:]))
