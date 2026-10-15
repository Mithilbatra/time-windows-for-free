"""
TSPTW → QUBO
============
Travelling Salesman Problem with Time Windows, encoded for QAOA.

Instance
  Depot = node 0, customers 1..n.  d[u, v] = travel time (= cost).
  Customer j must be served inside [e_j, l_j]; arriving early means waiting.
  The vehicle leaves the depot at e_0 and must be back by l_0 (inf = no deadline).

Encoding (depot fixed at position 0, so n customers need n² qubits)
  x[j, k] = 1  iff customer j is the k-th stop            qubit (j-1)*n + (k-1)

H = H_A + H_B + H_T, every term ≥ 0:

  H_A  A · Σ_j (1 − Σ_k x[j,k])²  +  A · Σ_k (1 − Σ_j x[j,k])²
       one stop per customer and one customer per position;
       (1 − Σx)² = 1 − Σx + 2·Σ_{a<b} x_a x_b  for binary x.

  H_B  B · [ Σ_j d[0,j] x[j,1] + Σ_k Σ_{u≠j} d[u,j] x[u,k] x[j,k+1] + Σ_j d[j,0] x[j,n] ]
       closed tour; the legs touching the fixed depot are linear.

  H_T  P · (time-window terms from tw_terms):
       E0    none
       E1    LB[j,k] = earliest possible service start of j as the k-th stop, over all
             k-hop walks from the depot whose stops are all on time:
               LB[j,1] = max(e_j, e_0 + d[0,j])
               LB[j,k] = max(e_j, min_{u≠j, LB[u,k-1] ≤ l_u} LB[u,k-1] + d[u,j])
             linear    x[j,k]           if LB[j,k] > l_j   (k = n: or LB[j,n] + d[j,0] > l_0)
             pairwise  x[u,k] x[j,k+1]  if max(e_j, LB[u,k] + d[u,j]) > l_j
       E1+C  E1 plus a look-ahead x[u,k] x[w,k+2] when no middle customer fits.
       Sound: a feasible route's service start at stop k is ≥ LB, so no feasible route is
       ever penalised (proof: related_work.md §4).  E1's pairs are pairs H_B already
       couples, so it adds no two-qubit gates; E1+C's (k, k+2) pairs are new couplings.

Weights
  A and P are multiples of S = B · Σ_v max_w d[v,w] ≥ B · (cost of any tour).  Since every
  term is non-negative, A, P ≥ 1 keep the TSPTW optimum below every invalid assignment
  (energy ≥ 2A) and every penalised route (energy > P).  Unpenalised infeasible routes
  compete on cost alone; how often that happens is what pruning_check.py measures.
"""

import itertools
import numpy as np


class TSPTWInstance:
    """Depot = node 0, customers 1..n.  time_windows[0] = (departure time, return deadline)."""

    def __init__(self, coords, time_windows, travel_times=None):
        """
        coords       : (n+1, 2) positions, or None when travel_times is given
        time_windows : (n+1, 2) rows of (earliest, latest)
        travel_times : optional (n+1, n+1) matrix; Euclidean distance between coords if None
        """
        self.coords = None if coords is None else np.array(coords, dtype=float)
        self.time_windows = np.array(time_windows, dtype=float)
        self.e, self.l = self.time_windows.T
        if travel_times is not None:
            self.d = np.array(travel_times, dtype=float)
        else:
            self.d = np.linalg.norm(self.coords[:, None] - self.coords[None], axis=-1)
        self.n = len(self.d) - 1          # customers

    def tour_cost(self, route):
        """route = [0, c1, ..., cn]: cost of the closed tour back to the depot."""
        route = list(route)
        return float(sum(self.d[u, v] for u, v in zip(route, route[1:] + route[:1])))

    def arrivals(self, route):
        """Service start at each stop of route (waiting if early), then the return time."""
        route = list(route)
        a = self.e[0]
        out = [a]
        for u, v in zip(route, route[1:]):
            a = max(self.e[v], a + self.d[u, v])
            out.append(a)
        return out + [a + self.d[route[-1], 0]]


def make_instance(n, seed=None):
    """Random feasible instance, protocol of Salehi, Glos & Miszczak (2022) §4.1:
    travel costs U{1..10} (made metric by shortest-path closure), e_j ~ U{1..20},
    l_j ~ U{e_j..40}; resampled until at least one route is feasible.
    seed may be an int or a np.random.Generator (drawn from in place)."""
    rng = np.random.default_rng(seed)
    while True:
        N = n + 1
        d = rng.integers(1, 11, size=(N, N)).astype(float)
        d = np.triu(d, 1)
        d = d + d.T
        for k in range(N):                    # Floyd–Warshall
            d = np.minimum(d, d[:, [k]] + d[[k], :])
        e = np.r_[0, rng.integers(1, 21, size=n)].astype(float)
        l = np.r_[np.inf, [rng.integers(ei, 41) for ei in e[1:]]].astype(float)
        inst = TSPTWInstance(None, np.c_[e, l], travel_times=d)
        if all_routes(inst)[1].any():
            return inst


def all_routes(inst):
    """Every customer order (rows of perms; depot implied at both ends), whether it meets
    every window including the return deadline, and its tour cost.  n! rows."""
    n, d, e, l = inst.n, inst.d, inst.e, inst.l
    perms = np.array(list(itertools.permutations(range(1, n + 1))))
    a = np.full(len(perms), e[0])
    prev = np.zeros(len(perms), dtype=int)
    feasible = np.ones(len(perms), dtype=bool)
    cost = np.zeros(len(perms))
    for k in range(n):
        c = perms[:, k]
        cost += d[prev, c]
        a = np.maximum(e[c], a + d[prev, c])  # wait if early
        feasible &= a <= l[c]
        prev = c
    cost += d[prev, 0]
    feasible &= a + d[prev, 0] <= l[0]
    return perms, feasible, cost


def solve_exact(inst):
    """Optimal TSPTW route [0, c1, ..., cn] and its cost, or (None, inf) if infeasible."""
    perms, feasible, cost = all_routes(inst)
    if not feasible.any():
        return None, np.inf
    i = np.flatnonzero(feasible)[np.argmin(cost[feasible])]
    return [0, *perms[i].tolist()], float(cost[i])


# ─────────────────────────────────────────────────────────────────────────────
# Time-window penalty terms
# ─────────────────────────────────────────────────────────────────────────────

def earliest_bounds(inst):
    """LB[j, k]: earliest possible service start of customer j at position k+1, over all
    walks from the depot whose stops are all on time.  inf if no such walk."""
    n, d, e, l = inst.n, inst.d, inst.e, inst.l
    LB = np.full((n + 1, n), np.inf)
    LB[1:, 0] = np.maximum(e[1:], e[0] + d[0, 1:])
    for k in range(1, n):
        prev = np.where(LB[:, k - 1] <= l, LB[:, k - 1], np.inf)
        prev[0] = np.inf
        M = prev[:, None] + d
        np.fill_diagonal(M, np.inf)
        LB[1:, k] = np.maximum(e[1:], M.min(axis=0)[1:])
    return LB


def lookahead(inst, start):
    """(n-2, n+1, n+1) look-ahead terms: penalise (u at k, w at k+2) when no middle
    customer v makes u → v → w fit, given u's service starts no earlier than start[u, k]."""
    n, d, e, l = inst.n, inst.d, inst.e, inst.l
    la = np.zeros((max(n - 2, 0), n + 1, n + 1))
    cust = range(1, n + 1)

    def fits(u, v, w, k):
        a2 = max(e[v], start[u, k] + d[u, v])
        return a2 <= l[v] and max(e[w], a2 + d[v, w]) <= l[w]

    for k in range(n - 2):
        for u in cust:
            for w in cust:
                if u != w:
                    la[k, u, w] = not any(fits(u, v, w, k) for v in cust if v not in (u, w))
    return la


def tw_terms(inst, kind="E1"):
    """Time-window terms for kind in {"E0", "E1", "E1+C"}, by 0-based position k:
    lin[j, k] on x[j,k];  pair[k, u, j] on x[u,k]·x[j,k+1];  la[k, u, w] on x[u,k]·x[w,k+2]."""
    n, d, e, l = inst.n, inst.d, inst.e, inst.l
    lin = np.zeros((n + 1, n))
    pair = np.zeros((max(n - 1, 0), n + 1, n + 1))
    la = np.zeros((max(n - 2, 0), n + 1, n + 1))
    if kind == "E0":
        return lin, pair, la
    LB = earliest_bounds(inst)
    lin[1:] = LB[1:] > l[1:, None]
    lin[1:, -1] = (lin[1:, -1] > 0) | (LB[1:, -1] + d[1:, 0] > l[0])   # must still get home
    mask = np.ones((n + 1, n + 1), dtype=bool)
    mask[0] = mask[:, 0] = False
    np.fill_diagonal(mask, False)
    for k in range(n - 1):
        pair[k] = mask & (np.maximum(e, LB[:, k, None] + d) > l)
    if kind == "E1+C":
        la = lookahead(inst, LB)
    elif kind != "E1":
        raise ValueError(f"unknown time-window encoding {kind!r}")
    return lin, pair, la


def route_penalty(perms, lin, pair, la):
    """Penalty units of each customer order (rows of perms, depot omitted)."""
    n = perms.shape[1]
    pen = sum(lin[perms[:, k], k] for k in range(n))
    pen = pen + sum(pair[k][perms[:, k], perms[:, k + 1]] for k in range(n - 1))
    return pen + sum(la[k][perms[:, k], perms[:, k + 2]] for k in range(n - 2))


# ─────────────────────────────────────────────────────────────────────────────
# QUBO builder
# ─────────────────────────────────────────────────────────────────────────────

class QUBOBuilder:
    """
    Upper-triangular QUBO matrix Q (linear terms on the diagonal) plus a constant offset,
    so energy(x) = xᵀQx + offset.  A valid, unpenalised route's energy is B · tour cost.
    """

    def __init__(self, instance: TSPTWInstance, A=1.0, B=1.0, P=1.0, tw="E1"):
        """A, P: multiples of the safe scale S (module docstring).  tw: "E0" | "E1" | "E1+C"."""
        S = B * instance.d.max(axis=1).sum()
        self.inst = instance
        self.n = instance.n
        self.N = self.n * self.n              # qubits
        self.A, self.B, self.P, self.tw = A * S, B, P * S, tw
        self.build()

    def _idx(self, j, k):
        """Qubit of customer j (1..n) at 0-based position k."""
        return (j - 1) * self.n + k

    def build(self):
        n, d = self.n, self.inst.d
        Q = np.zeros((self.N, self.N))

        def add(a, b, v):
            Q[min(a, b), max(a, b)] += v

        # H_A: each customer once, each position once
        rows = [[self._idx(j, k) for k in range(n)] for j in range(1, n + 1)]
        cols = [[self._idx(j, k) for j in range(1, n + 1)] for k in range(n)]
        for group in rows + cols:
            for a, qa in enumerate(group):
                add(qa, qa, -self.A)
                for qb in group[a + 1:]:
                    add(qa, qb, 2 * self.A)
        offset = self.A * (len(rows) + len(cols))

        # H_B: closed tour distance
        for j in range(1, n + 1):
            add(self._idx(j, 0), self._idx(j, 0), self.B * d[0, j])
            add(self._idx(j, n - 1), self._idx(j, n - 1), self.B * d[j, 0])
        for k in range(n - 1):
            for u in range(1, n + 1):
                for j in range(1, n + 1):
                    if u != j:
                        add(self._idx(u, k), self._idx(j, k + 1), self.B * d[u, j])

        # H_T: time windows
        lin, pair, la = tw_terms(self.inst, self.tw)
        for j in range(1, n + 1):
            for k in range(n):
                add(self._idx(j, k), self._idx(j, k), self.P * lin[j, k])
        for k, u, j in zip(*np.nonzero(pair)):
            add(self._idx(u, k), self._idx(j, k + 1), self.P * pair[k, u, j])
        for k, u, w in zip(*np.nonzero(la)):
            add(self._idx(u, k), self._idx(w, k + 2), self.P * la[k, u, w])

        self.Q, self.offset = Q, offset
        return Q

    def energy(self, bitstring):
        x = np.asarray(bitstring, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def decode(self, bitstring):
        """Route [0, c1, ..., cn]; -1 marks an empty position."""
        route = [-1] * self.n
        for j in range(1, self.n + 1):
            for k in range(self.n):
                if bitstring[self._idx(j, k)] == 1:
                    route[k] = j
        return [0] + route

    def encode(self, route):
        bits = [0] * self.N
        for k, j in enumerate(route[1:]):
            bits[self._idx(j, k)] = 1
        return tuple(bits)

    def is_valid_route(self, bitstring):
        """Exactly one customer per position and one position per customer."""
        route = self.decode(bitstring)[1:]
        return sum(bitstring) == self.n and sorted(route) == list(range(1, self.n + 1))
