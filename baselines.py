"""
Classical baselines
===================
E2      Edge-based TSPTW QUBO of Salehi, Glos & Miszczak (2022, §3.1; Eqs. 5, 11–13,
        18–24): exact time windows through binary-encoded waiting times and slacks.
        It needs far more qubits than QAOA simulation allows (at most n³ − 2n² + 3n + 3nδ,
        vs n² for qubo.QUBOBuilder), so it is solved with simulated annealing only.
anneal  Single-bit-flip simulated annealing for any upper-triangular QUBO (ours or E2).
"""

import numpy as np


def int_bits(U):
    """Weights of the bits encoding an integer in [0, U] (Salehi et al. Eq. 5):
    1, 2, …, 2^(k-2), U − 2^(k-1) + 1  with  k = ⌈log2(U + 1)⌉;  no bits for U ≤ 0."""
    U = int(U)
    if U <= 0:
        return []
    k = int(np.ceil(np.log2(U + 1)))
    return [2 ** i for i in range(k - 1)] + [U - 2 ** (k - 1) + 1]


def int_to_bits(y, weights):
    """Bit values that give integer y under int_bits weights; ValueError if y does not fit."""
    if not weights:
        if y != 0:
            raise ValueError(f"{y} does not fit an empty encoding")
        return []
    low = len(weights) - 1                    # plain binary part covers 0 .. 2^low − 1
    top = int(y > 2 ** low - 1)
    rest = y - top * weights[-1]
    if not 0 <= rest <= 2 ** low - 1:
        raise ValueError(f"{y} does not fit the encoding {weights}")
    return [(rest >> b) & 1 for b in range(low)] + [top]


class SalehiQUBO:
    """
    E2:  H = A·H_R + P·H_TW + B·H_C   (Salehi et al. Eqs. 11, 22, 23, 24) over
      x[i,u,v]          edge u→v taken at step i  (i = 1: depot→v,  i = n+1: v→depot)
      ω_i, ξe_i, ξl_i   waiting time and the two slacks at the i-th stop, binary-encoded
                        (Eq. 5) up to the paper's bounds (Eqs. 18–21)
    A and P are multiples of the safe scale S of qubo.QUBOBuilder.  H_R's continuity terms
    carry a constant of 1 per step transition so a valid tour has H_R = 0 (the paper's
    Eq. 11 differs by a constant).  As in the paper: integer data, departure at time 0 and
    no depot deadline.
    """

    def __init__(self, instance, A=1.0, B=1.0, P=1.0):
        inst = instance
        n, c, e, l = inst.n, inst.d, inst.e, inst.l
        if (e[0] != 0 or np.isfinite(l[0]) or not np.array_equal(c, np.round(c))
                or not np.array_equal(inst.time_windows[1:], np.round(inst.time_windows[1:]))):
            raise ValueError("E2 needs integer data, departure at 0 and no depot deadline")
        S = B * c.max(axis=1).sum()
        self.inst, self.n = inst, n
        self.A, self.B, self.P = A * S, B, P * S

        # edges available at each step
        cust = range(1, n + 1)
        self.steps = {1: [(0, v) for v in cust], n + 1: [(v, 0) for v in cust]}
        for i in range(2, n + 1):
            self.steps[i] = [(u, v) for u in cust for v in cust if u != v]
        self.var = {}
        for i in range(1, n + 2):
            for u, v in self.steps[i]:
                self.var["x", i, u, v] = len(self.var)

        # waiting time and slacks of stop i, bounded via the arrival lower bound A_lb (Eq. 18)
        arcs = np.sort([c[u, v] for u in cust for v in range(n + 1) if u != v])
        self.ints = {}                        # (name, i) -> [(weight, var), ...]
        for i in cust:
            A_lb = c[0, 1:].min() + arcs[:i - 1].sum()
            bounds = (("w", e[1:].max() - A_lb),          # Eq. 21
                      ("se", (l[1:] - e[1:]).max()),      # Eq. 20
                      ("sl", l[1:].max() - A_lb))         # Eq. 19
            for name, U in bounds:
                self.ints[name, i] = []
                for b, weight in enumerate(int_bits(U)):
                    self.var[name, i, b] = len(self.var)
                    self.ints[name, i].append((weight, self.var[name, i, b]))
        self.N = len(self.var)
        self.Q, self.offset = self._build()

    def _build(self):
        n, c, e, l = self.n, self.inst.d, self.inst.e, self.inst.l
        Q = np.zeros((self.N, self.N))
        offset = 0.0

        def x(i, u, v):
            return self.var["x", i, u, v]

        def add(a, b, value):
            Q[min(a, b), max(a, b)] += value

        def square(terms, const, weight):
            """weight · (Σ coef·var + const)²  for terms = [(coef, var), ...]"""
            nonlocal offset
            merged = {}
            for coef, var in terms:
                merged[var] = merged.get(var, 0.0) + coef
            items = list(merged.items())
            offset += weight * const ** 2
            for a, (va, ca) in enumerate(items):
                add(va, va, weight * (ca * ca + 2 * const * ca))
                for vb, cb in items[a + 1:]:
                    add(va, vb, weight * 2 * ca * cb)

        def neg(terms):
            return [(-coef, var) for coef, var in terms]

        # H_R: one edge per step, each customer left once, consecutive edges chain up
        for i in range(1, n + 2):
            square([(1, x(i, u, v)) for u, v in self.steps[i]], -1, self.A)
        for u in range(1, n + 1):
            leave = [(1, x(n + 1, u, 0))] + [(1, x(i, a, b)) for i in range(2, n + 1)
                                             for a, b in self.steps[i] if a == u]
            square(leave, -1, self.A)
        for t in range(1, n + 1):
            offset += self.A
            for u, v in self.steps[t]:
                for vv, w in self.steps[t + 1]:
                    if vv == v and u != w:
                        add(x(t, u, v), x(t + 1, v, w), -self.A)

        # H_TW at each stop i:  e_v ≤ A_i + ω_i  and  A_i ≤ l_v,  closed with the slacks
        arrival = []                                  # A_i (Eq. 13) as (coef, var) terms
        for i in range(1, n + 1):
            if i > 1:
                arrival = arrival + self.ints["w", i - 1]
            arrival = arrival + [(c[u, v], x(i, u, v)) for u, v in self.steps[i]]
            due_e = [(e[v], x(i, u, v)) for u, v in self.steps[i]]
            due_l = [(l[v], x(i, u, v)) for u, v in self.steps[i]]
            square(due_e + neg(arrival) + neg(self.ints["w", i]) + self.ints["se", i], 0, self.P)
            square(arrival + neg(due_l) + self.ints["sl", i], 0, self.P)

        # H_C: tour cost
        for i in range(1, n + 2):
            for u, v in self.steps[i]:
                add(x(i, u, v), x(i, u, v), self.B * c[u, v])
        return Q, offset

    def energy(self, bitstring):
        x = np.asarray(bitstring, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def _edges(self, bitstring):
        """Edge taken at each step 1..n+1, None where a step has not exactly one."""
        out = []
        for i in range(1, self.n + 2):
            on = [(u, v) for u, v in self.steps[i] if bitstring[self.var["x", i, u, v]]]
            out.append(on[0] if len(on) == 1 else None)
        return out

    def decode(self, bitstring):
        """Route [0, c1, ..., cn]; -1 where a step does not have exactly one edge."""
        return [0] + [edge[1] if edge else -1 for edge in self._edges(bitstring)[:-1]]

    def is_valid_route(self, bitstring):
        """One edge per step, each leaving where the previous one arrived, every customer once."""
        edges = self._edges(bitstring)
        if None in edges:
            return False
        route = [0] + [v for _, v in edges[:-1]]
        return (all(edge[0] == route[s] for s, edge in enumerate(edges))
                and sorted(route[1:]) == list(range(1, self.n + 1)))

    def encode(self, route):
        """Bits of a feasible route with the least waiting; ValueError if it misses a window."""
        n, c, e, l = self.n, self.inst.d, self.inst.e, self.inst.l
        stops = list(route) + [0]
        bits = np.zeros(self.N, dtype=int)
        for i in range(1, n + 2):
            bits[self.var["x", i, stops[i - 1], stops[i]]] = 1
        a = 0.0
        for i in range(1, n + 1):
            v = stops[i]
            a += c[stops[i - 1], v]                   # A_i: arrival, earlier waits included
            wait = max(0.0, e[v] - a)
            for name, y in (("w", wait), ("se", a + wait - e[v]), ("sl", l[v] - a)):
                weights = [weight for weight, _ in self.ints[name, i]]
                for (_, var), bit in zip(self.ints[name, i], int_to_bits(int(round(y)), weights)):
                    bits[var] = bit
            a += wait
        return tuple(bits)


def anneal(Q, offset=0.0, reads=100, sweeps=1000, seed=None):
    """
    Single-bit-flip Metropolis simulated annealing on E(x) = xᵀQx + offset, Q upper
    triangular.  Geometric schedule: the first sweep accepts the costliest possible flip
    half the time, the last accepts the cheapest non-zero coefficient 1% of the time.
    Each sweep visits every bit once, for all reads in parallel.
    Returns (samples, energies) of the final states, one row per read.
    """
    rng = np.random.default_rng(seed)
    N = len(Q)
    lin = np.diag(Q).copy()
    W = Q + Q.T
    np.fill_diagonal(W, 0)
    coeffs = np.abs(np.r_[lin, Q[np.triu_indices(N, 1)]])
    hot = (np.abs(lin) + np.abs(W).sum(axis=1)).max() / np.log(2)
    cold = coeffs[coeffs > 1e-9].min() / np.log(100)

    x = rng.integers(0, 2, size=(reads, N)).astype(float)
    field = lin + x @ W                           # energy change of switching each bit on
    for T in np.geomspace(hot, cold, sweeps):
        for i in range(N):
            dE = (1 - 2 * x[:, i]) * field[:, i]
            flip = rng.random(reads) < np.exp(-np.maximum(dE, 0) / T)
            step = np.where(flip, 1 - 2 * x[:, i], 0.0)
            x[:, i] += step
            field += step[:, None] * W[i]
    energies = np.einsum("ri,ij,rj->r", x, Q, x) + offset
    return x.astype(int), energies
