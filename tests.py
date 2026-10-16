"""Self-checks.   Run:  python tests.py   (or: pytest tests.py)"""
import itertools
import warnings

import numpy as np
import pennylane as qml

from baselines import SalehiQUBO, anneal, int_bits, int_to_bits
from qaoa import TSPTWQAOACircuit, qubo_to_hamiltonian
from qubo import (QUBOBuilder, TSPTWInstance, all_routes, make_instance, route_penalty,
                  solve_exact, tw_terms)

warnings.filterwarnings("ignore")          # PennyLane's NumPy-version notice


def tight_deadline(inst):
    """Same instance, with the depot deadline set to the optimal tour's return time."""
    route, _ = solve_exact(inst)
    tw = inst.time_windows.copy()
    tw[0, 1] = inst.arrivals(route)[-1]
    return TSPTWInstance(None, tw, travel_times=inst.d)


# ── qubo.py ───────────────────────────────────────────────────────────────────

def test_exact_solver_matches_route_by_route():
    for seed in range(20):
        for inst in (make_instance(4, seed), tight_deadline(make_instance(4, seed))):
            perms, feasible, cost = all_routes(inst)
            for p, f, c in zip(perms, feasible, cost):
                route = [0, *p]
                assert f == (not inst.late_stops(route)) and np.isclose(c, inst.tour_cost(route))


def test_feasible_routes_never_penalised():
    for seed in range(30):
        for inst in (make_instance(5, seed), tight_deadline(make_instance(5, seed))):
            perms, feasible, _ = all_routes(inst)
            for kind in ("E1", "E1+C"):
                assert not route_penalty(perms[feasible], *tw_terms(inst, kind)).any()


def test_energy_is_cost_plus_penalty():
    inst = tight_deadline(make_instance(4, seed=1))
    perms, _, cost = all_routes(inst)
    for kind in ("E0", "E1", "E1+C"):
        b = QUBOBuilder(inst, tw=kind)
        pen = route_penalty(perms, *tw_terms(inst, kind))
        energies = [b.energy(b.encode([0, *p])) for p in perms]
        assert np.allclose(energies, b.B * cost + b.P * pen)


def test_qubo_matrix_matches_formula_on_random_states():
    """H_A + H_B + H_T evaluated term by term (module docstring), invalid states included."""
    inst = tight_deadline(make_instance(4, seed=7))
    n, d = inst.n, inst.d
    rng = np.random.default_rng(0)
    for kind, rows in itertools.product(("E0", "E1", "E1+C"), (True, False)):
        b = QUBOBuilder(inst, tw=kind)
        Q, offset = b.qubo(rows)
        lin, pair, la = tw_terms(inst, kind)
        for bits in rng.integers(0, 2, size=(50, b.N)):
            X = np.vstack([np.zeros(n), bits.reshape(n, n)])      # X[j, k], row 0 unused
            H_A = rows * ((1 - X[1:].sum(axis=1)) ** 2).sum() + ((1 - X[1:].sum(axis=0)) ** 2).sum()
            H_B = d[0, 1:] @ X[1:, 0] + d[1:, 0] @ X[1:, -1] + sum(
                d[u, j] * X[u, k] * X[j, k + 1] for k in range(n - 1)
                for u in range(1, n + 1) for j in range(1, n + 1) if u != j)
            H_T = (lin * X).sum() + sum(
                pair[k, u, j] * X[u, k] * X[j, k + 1] + (la[k - 1, u, j] * X[u, k - 1] * X[j, k + 1] if k else 0)
                for k in range(n - 1) for u in range(n + 1) for j in range(n + 1))
            assert np.isclose(bits @ Q @ bits + offset, b.A * H_A + b.B * H_B + b.P * H_T)


def test_ground_state_is_a_valid_route():
    inst = make_instance(3, seed=2)
    b = QUBOBuilder(inst)
    best = min(itertools.product((0, 1), repeat=b.N), key=b.energy)   # all 2^9 states
    assert b.is_valid_route(best)
    assert b.decode(b.encode([0, 2, 3, 1])) == [0, 2, 3, 1]
    extra = list(b.encode([0, 2, 3, 1]))
    extra[b._idx(1, 1)] = 1            # customer 1 twice; decode still shows a permutation
    assert not b.is_valid_route(extra)


# ── qaoa.py ───────────────────────────────────────────────────────────────────

def test_ising_hamiltonian_matches_qubo():
    b = QUBOBuilder(make_instance(3, seed=5))
    states = np.array(list(itertools.product((0, 1), repeat=b.N)))
    for rows in (True, False):
        Q, offset = b.qubo(rows)
        H, nq, ising_offset = qubo_to_hamiltonian(Q)
        diag = np.real(np.diag(qml.matrix(H, wire_order=range(nq))))
        assert np.allclose(diag + ising_offset + offset, np.einsum("si,ij,sj->s", states, Q, states) + offset)


def test_reported_energy_and_xy_subspace():
    b = QUBOBuilder(make_instance(3, seed=5))
    states = np.array(list(itertools.product((0, 1), repeat=b.N)))
    E = np.array([b.energy(s) for s in states])
    params = np.array([0.3, 0.7, 0.4, 0.2])
    for mixer in ("x", "xy"):
        qc = TSPTWQAOACircuit(b, p=2, mixer=mixer)
        probs = np.abs(qc.get_statevector(params)) ** 2
        first = qc.optimize(steps=1, init_params=params)["loss_history"][0]
        assert np.isclose(first, probs @ E)               # normalisation and offsets undone
        if mixer == "xy":                                 # each customer at exactly one position
            one_hot = np.all([states[:, row].sum(axis=1) == 1 for row in qc.rows], axis=0)
            assert np.isclose(probs[one_hot].sum(), 1)
        res = qc.optimize(steps=3, starts=2)
        assert len(res["start_energies"]) == 2 and res["best_energy"] == min(res["start_energies"])


# ── baselines.py ──────────────────────────────────────────────────────────────

def test_integer_encoding_covers_its_range():
    for U in range(70):
        weights = int_bits(U)
        assert sum(weights) == U
        for y in range(U + 1):
            assert np.dot(int_to_bits(y, weights), weights) == y
        for y in (-1, U + 1):
            try:
                int_to_bits(y, weights)
                assert False, (y, U)
            except ValueError:
                pass


def test_e2_feasible_routes_cost_only_infeasible_cannot_be_encoded():
    for n, seed in itertools.product((3, 4), range(4)):
        inst = make_instance(n, seed)
        q = SalehiQUBO(inst)
        perms, feasible, cost = all_routes(inst)
        for p, f, c in zip(perms, feasible, cost):
            route = [0, *p]
            if f:
                bits = q.encode(route)
                assert q.is_valid_route(bits) and q.decode(bits) == route
                assert np.isclose(q.energy(bits), q.B * c)    # zero route and window penalty
            else:
                try:
                    q.encode(route)
                    assert False, route
                except ValueError:
                    pass


def test_e2_matrix_matches_formula_on_random_states():
    """Salehi et al. Eqs. 11 (shifted), 13, 22, 23 evaluated term by term, invalid states included."""
    q = SalehiQUBO(make_instance(3, seed=1))
    n, c, e, l = q.n, q.inst.d, q.inst.e, q.inst.l
    rng = np.random.default_rng(0)
    for bits in rng.integers(0, 2, size=(200, q.N)):
        def x(i, u, v):
            return bits[q.var["x", i, u, v]]

        def val(name, i):
            return sum(weight * bits[var] for weight, var in q.ints[name, i])

        H_R = sum((1 - sum(x(i, u, v) for u, v in q.steps[i])) ** 2 for i in range(1, n + 2))
        H_R += sum((1 - x(n + 1, u, 0) - sum(x(i, u, v) for i in range(2, n + 1)
                                             for v in range(1, n + 1) if v != u)) ** 2
                   for u in range(1, n + 1))
        H_R += sum(1 - sum(x(t, u, v) * x(t + 1, v, w) for u, v in q.steps[t]
                           for vv, w in q.steps[t + 1] if vv == v and u != w)
                   for t in range(1, n + 1))
        H_TW, A = 0, 0
        for i in range(1, n + 1):
            A += (val("w", i - 1) if i > 1 else 0) + sum(c[u, v] * x(i, u, v) for u, v in q.steps[i])
            due_e = sum(e[v] * x(i, u, v) for u, v in q.steps[i])
            due_l = sum(l[v] * x(i, u, v) for u, v in q.steps[i])
            H_TW += (due_e - A - val("w", i) + val("se", i)) ** 2 + (A - due_l + val("sl", i)) ** 2
        H_C = sum(c[u, v] * x(i, u, v) for i in range(1, n + 2) for u, v in q.steps[i])
        assert np.isclose(q.energy(bits), q.A * H_R + q.P * H_TW + q.B * H_C)


def test_anneal_finds_the_ground_state_and_ends_in_local_minima():
    b = QUBOBuilder(make_instance(3, seed=2))
    best = min(b.energy(s) for s in itertools.product((0, 1), repeat=b.N))
    samples, energies = anneal(b.Q, b.offset, reads=20, sweeps=200, seed=0)
    assert np.allclose(energies, [b.energy(s) for s in samples])
    assert np.isclose(energies.min(), best)
    b = QUBOBuilder(make_instance(4, seed=3))
    samples, _ = anneal(b.Q, b.offset, reads=50, sweeps=300, seed=1)
    W = b.Q + b.Q.T
    np.fill_diagonal(W, 0)
    flip_cost = (1 - 2 * samples) * (np.diag(b.Q) + samples @ W)
    assert (flip_cost >= -1e-9).all(axis=1).mean() >= 0.9    # no single flip lowers the energy


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
