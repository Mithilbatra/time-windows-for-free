"""
Classical check: how much of the infeasible route space does each time-window
penalty rule out, and does the QUBO ground state land on the TSPTW optimum?

Instances: qubo.make_instance (Salehi, Glos & Miszczak 2022 §4.1 protocol).
Depot 0 opens at 0, tour returns to depot, no depot deadline, no service times.

Methods (formulas in related_work.md §4):
  E0        no time-window term                                   (qubo.py)
  naive     the old qubo.py position proxy (sign fixed): arrival ~ position * T/n
  RG-AB     Rezk & Gora tiers A + B-hard (their sound part, no new couplings)
  RG-ABs    + B-soft (graded, heuristic)
  RG-full   + C look-ahead k->k+2 (their actual method)
  E1        ours: cumulative earliest-arrival bounds LB(j,k)      (qubo.py)
  E1+C      ablation: E1 plus a sound look-ahead built on LB(j,k) (qubo.py)

Run:  python pruning_check.py
"""
import numpy as np

from qubo import TSPTWInstance, all_routes, lookahead, make_instance, route_penalty, tw_terms

P = 1000.0       # time-window weight, >> any tour cost (max ~ 10 * 8)
INF = np.inf


def method_terms(name, inst):
    """(lin, pair, la) penalty terms: our encodings from qubo.py, baselines built here."""
    if name in ("E0", "E1", "E1+C"):
        return tw_terms(inst, name)
    n, t, e, l = inst.n, inst.d, inst.e, inst.l
    lin, pair, la = tw_terms(inst, "E0")          # zero arrays of the right shapes
    cust = np.arange(1, n + 1)
    if name == "naive":
        T = l[1:].max(); step = T / n
        for k in range(n):
            arr = (k + 1) * step
            lin[1:, k] = np.maximum(0, arr - l[1:]) ** 2 + np.maximum(0, e[1:] - arr) ** 2
        return lin, pair, la
    if name.startswith("RG"):
        ehat = np.maximum(e, e[0] + t[0])          # their "true earliest": straight from depot
        lin[1:, 0] = ehat[1:] > l[1:]
        for u in cust:
            for j in cust:
                if u == j:
                    continue
                if ehat[u] + t[u, j] > l[j]:
                    pair[:, u, j] = 1.0
                elif name != "RG-AB" and l[u] + t[u, j] > l[j]:
                    f = min(1.0, (l[u] + t[u, j] - l[j]) / max(1.0, l[u] - e[u]))
                    pair[:, u, j] = 0.5 * f
        if name == "RG-full":
            la = lookahead(inst, np.repeat(ehat[:, None], n, axis=1))
        return lin, pair, la
    raise ValueError(name)


METHODS = ["E0", "naive", "RG-AB", "RG-ABs", "RG-full", "E1", "E1+C"]
SOUND = {"E0", "RG-AB", "E1", "E1+C"}   # never penalise a feasible route (RG-AB: metric instances only)


def run(n, n_inst=200, seed=0):
    rng = np.random.default_rng(seed + n)
    stats = {m: dict(prune=[], false=[], gs=[], gs_bind=[], newzz=[], rank=[]) for m in METHODS}
    for _ in range(n_inst):
        inst = make_instance(n, rng)
        perms, feas, cost = all_routes(inst)
        opt = cost[feas].min()
        binding = not feas[np.argmin(cost)]          # TSP optimum violates a window
        for m in METHODS:
            lin, pair, la = method_terms(m, inst)
            pen = route_penalty(perms, lin, pair, la)
            if m in SOUND:
                assert not (pen[feas] > 0).any(), f"{m} penalised a feasible route"
            E = cost + P * pen
            ground = np.isclose(E, E.min())
            ok = bool(feas[ground].all() and np.isclose(cost[ground], opt).all())
            E_opt = E[feas & np.isclose(cost, opt)].min()
            s = stats[m]
            s["rank"].append(int((E < E_opt - 1e-9).sum()))  # routes a sampler would prefer over the optimum
            s["prune"].append((pen[~feas] > 0).mean() if (~feas).any() else 1.0)
            s["false"].append((pen[feas] > 0).mean())
            s["gs"].append(ok)
            if binding:
                s["gs_bind"].append(ok)
            s["newzz"].append(int((la > 0).sum()))  # k->k+2 pairs: couplings no other term has
    return stats


def check_soft_tier_unsound():
    """Hand example from related_work.md §4: RG soft tier penalises a feasible route."""
    t = np.array([[0, 2, 3.], [2, 0, 1], [3, 1, 0]])
    inst = TSPTWInstance(None, [[0, INF], [0, 10], [0, 5]], travel_times=t)
    perms, feas, _ = all_routes(inst)
    route = np.where((perms == [1, 2]).all(axis=1))[0][0]
    assert feas[route]
    pen = route_penalty(perms, *method_terms("RG-ABs", inst))
    assert np.isclose(pen[route], 0.3), pen[route]
    assert route_penalty(perms, *method_terms("E1", inst))[route] == 0


if __name__ == "__main__":
    check_soft_tier_unsound()
    n_base = lambda n: n * n * (n - 1) + n * (n - 1) ** 2   # ZZ terms: one-hot rows+cols + distance
    print(f"P = {P:g}; 200 feasible instances per size (Salehi protocol)\n")
    for n in (3, 4, 5, 6, 7):
        st = run(n)
        nb = len(st["E0"]["gs_bind"])
        print(f"n = {n} customers ({n*n} qubits, depot fixed) | TW-binding instances: {nb}/200")
        print(f"  {'method':8} {'pruned':>7} {'false+':>7} {'GS=opt':>7} {'GS=opt|bind':>12} "
              f"{'rank med/p90':>13} {'new ZZ':>7}")
        for m in METHODS:
            s = st[m]
            gsb = np.mean(s["gs_bind"]) if s["gs_bind"] else float("nan")
            nz = np.mean(s["newzz"])
            r = np.percentile(s["rank"], [50, 90])
            print(f"  {m:8} {np.mean(s['prune']):7.1%} {np.mean(s['false']):7.1%} "
                  f"{np.mean(s['gs']):7.1%} {gsb:12.1%} {r[0]:6.0f}/{r[1]:<6.0f} "
                  f"{nz:5.1f} (+{nz / n_base(n):.0%})")
        print()
