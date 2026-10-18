"""
Classical baselines
===================
E2      Edge-based TSPTW QUBO of Salehi, Glos & Miszczak (2022, §3.1; Eqs. 5, 11–13,
        18–24): exact time windows through binary-encoded waiting times and slacks.
        It needs far more qubits than QAOA simulation allows (at most n³ − 2n² + 3n + 3nδ,
        vs n² for qubo.QUBOBuilder), so it is solved with simulated annealing only.
E2-UP   E2 with its two slack registers per stop replaced by Montañez-Barrera et al.'s (2024)
        unbalanced penalization: slack-free, but not exact (UnbalancedQUBO).
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
    lam = None                                # (λ1, λ2): E2-UP's unbalanced penalty (UnbalancedQUBO)

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
            for name, U in bounds[:1] if self.lam else bounds:    # E2-UP keeps only ω_i
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

        def unbalanced(terms):
            """−λ1·h + λ2·h²  for h = Σ coef·var  (Montañez-Barrera et al. Eq. 25)"""
            lam1, lam2 = self.lam
            square(terms, 0, lam2)
            for coef, var in terms:
                add(var, var, -lam1 * coef)

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
            if self.lam:
                unbalanced(arrival + self.ints["w", i] + neg(due_e))      # h_e = A_i + ω_i − e_v
                unbalanced(due_l + neg(arrival))                          # h_l = l_v − A_i
            else:
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
        """Bits of a feasible route with the least waiting; ValueError if it misses a window
        (E2-UP has no slacks, so it encodes any route)."""
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
                if (name, i) not in self.ints:
                    continue
                weights = [weight for weight, _ in self.ints[name, i]]
                for (_, var), bit in zip(self.ints[name, i], int_to_bits(int(round(y)), weights)):
                    bits[var] = bit
            a += wait
        return tuple(bits)


class UnbalancedQUBO(SalehiQUBO):
    """
    E2-UP:  H = A·H_R + Σ_i [ζ(h_e,i) + ζ(h_l,i)] + B·H_C,  E2 (SalehiQUBO) with its slack
    registers ξe_i, ξl_i dropped and each stop's two window inequalities penalised by the
    unbalanced penalization of Montañez-Barrera, Willsch, Maldonado-Romo & Michielsen
    (Quantum Sci. Technol. 9:025022, 2024; arXiv 2211.13914, Eqs. 23 and 25):
      h(x) ≥ 0   →   ζ(x) = −λ1·h(x) + λ2·h(x)²       (a quadratic fit to e^−h, Eq. 24)
      h_e,i = A_i + ω_i − e_v ≥ 0     served no earlier than the window opens
      h_l,i = l_v − A_i ≥ 0           arrived no later than it closes
    with A_i the arrival (Salehi et al. Eq. 13) and the waiting bits ω_i kept from E2: these are
    the two inequalities E2 closes with ξe_i and ξl_i, so E2-UP differs from E2 only in how
    they are penalised.  λ1, λ2 are absolute (cost units, B = 1), the same pair for every
    inequality, as the paper uses one pair per problem (its Table I).
    Not exact: on the feasible side ζ < 0 for 0 < h < λ1/λ2 (least −λ1²/4λ2 at h = λ1/2λ2) and
    ζ > 0 beyond, so a route's energy shifts with how much slack it leaves, a feasible route
    can be penalised and an infeasible one unpenalised (window_penalty measures it).
    """

    def __init__(self, instance, A=1.0, B=1.0, lam1=1.0, lam2=0.01):
        self.lam = (lam1, lam2)
        super().__init__(instance, A=A, B=B)

    def window_penalty(self, route):
        """Least Σ_i ζ(h_e,i) + ζ(h_l,i) over the waiting times the ω registers allow, for
        route [0, c1, ..., cn]: its least energy is B·cost plus this.  Dynamic programming
        over the total wait so far."""
        c, e, l = self.inst.d, self.inst.e, self.inst.l
        lam1, lam2 = self.lam

        def zeta(h):
            return -lam1 * h + lam2 * h ** 2

        best = np.zeros(1)                    # least penalty so far, by total wait so far
        travel = 0.0
        for i, (u, v) in enumerate(zip(route, route[1:]), 1):
            travel += c[u, v]
            a = travel + np.arange(len(best))                 # A_i for each total wait
            late = zeta(l[v] - a)
            nxt = np.full(len(best) + sum(w for w, _ in self.ints["w", i]), np.inf)
            for w in range(len(nxt) - len(best) + 1):         # ω_i = w
                cand = best + zeta(a + w - e[v]) + late
                nxt[w:w + len(best)] = np.minimum(nxt[w:w + len(best)], cand)
            best = nxt
        return float(best.min())


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
