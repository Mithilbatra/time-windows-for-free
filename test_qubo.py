"""Self-checks for qubo.py.   Run:  python test_qubo.py   (or: pytest test_qubo.py)"""
import itertools

import numpy as np

from qubo import (QUBOBuilder, TSPTWInstance, all_routes, make_instance, route_penalty,
                  solve_exact, tw_terms)


def tight_deadline(inst):
    """Same instance, with the depot deadline set to the optimal tour's return time."""
    route, _ = solve_exact(inst)
    tw = inst.time_windows.copy()
    tw[0, 1] = inst.arrivals(route)[-1]
    return TSPTWInstance(None, tw, travel_times=inst.d)


def test_exact_solver_matches_route_by_route():
    for seed in range(20):
        for inst in (make_instance(4, seed), tight_deadline(make_instance(4, seed))):
            perms, feasible, cost = all_routes(inst)
            for p, f, c in zip(perms, feasible, cost):
                route = [0, *p]
                arr = inst.arrivals(route)
                ok = all(a <= inst.l[v] for v, a in zip(route[1:], arr[1:-1])) and arr[-1] <= inst.l[0]
                assert f == ok and np.isclose(c, inst.tour_cost(route))


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


def test_ground_state_is_a_valid_route():
    inst = make_instance(3, seed=2)
    b = QUBOBuilder(inst)
    best = min(itertools.product((0, 1), repeat=b.N), key=b.energy)   # all 2^9 states
    assert b.is_valid_route(best)
    assert b.decode(b.encode([0, 2, 3, 1])) == [0, 2, 3, 1]
    extra = list(b.encode([0, 2, 3, 1]))
    extra[b._idx(1, 1)] = 1            # customer 1 twice; decode still shows a permutation
    assert not b.is_valid_route(extra)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
