"""
TSPTW QUBO Formulation with Time-Window Penalty Hamiltonian
===========================================================
Encodes the Travelling Salesman Problem with Time Windows (TSPTW)
as a Quadratic Unconstrained Binary Optimization (QUBO) problem.

Variables: x_{i,t} = 1 if city i is visited at position t in the route
           s_{i,k} = slack binary variables for time-window encoding

Objective terms:
  1. Route validity  – each city visited exactly once
  2. Position validity – each position filled exactly once
  3. Travel cost     – minimize total distance
  4. Time-window penalty – soft penalty if arrival outside [e_i, l_i]
"""

import numpy as np
from itertools import product


class TSPTWInstance:
    """Represents a TSPTW problem instance."""

    def __init__(self, coords, time_windows, travel_times=None, depot=0):
        """
        Parameters
        ----------
        coords       : list of (x, y) tuples, shape (n,)
        time_windows : list of (earliest, latest) per city, shape (n,)
        travel_times : optional n×n matrix; computed from Euclidean dist if None
        depot        : index of the depot / start city
        """
        self.coords = np.array(coords, dtype=float)
        self.time_windows = np.array(time_windows, dtype=float)
        self.depot = depot
        self.n = len(coords)

        if travel_times is not None:
            self.d = np.array(travel_times, dtype=float)
        else:
            self.d = self._euclidean_matrix()

        # Cumulative travel time from depot: used for time-window checking
        self.T_max = self.time_windows[:, 1].max()

    def _euclidean_matrix(self):
        n = self.n
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])
        return D

    @property
    def n_cities(self):
        return self.n


class QUBOBuilder:
    """
    Builds the QUBO matrix Q for TSPTW.

    Binary variables are indexed as:  var(i, t) = i*n + t
    Total variables                :  n * n   (for small n≤4 demos)
    """

    def __init__(self, instance: TSPTWInstance,
                 lam_city=5.0,
                 lam_pos=5.0,
                 lam_dist=1.0,
                 lam_tw=8.0):
        self.inst = instance
        self.n = instance.n
        self.lam_city = lam_city   # Lagrange multiplier: city-visit constraint
        self.lam_pos  = lam_pos    # Lagrange multiplier: position constraint
        self.lam_dist = lam_dist   # Lagrange multiplier: distance objective
        self.lam_tw   = lam_tw     # Lagrange multiplier: time-window penalty

        self.N = self.n * self.n   # total QUBO variables
        self.Q = np.zeros((self.N, self.N))

    def _idx(self, city, pos):
        return city * self.n + pos

    # ------------------------------------------------------------------
    # Constraint 1: Each city visited exactly once
    #   (sum_t x_{i,t} - 1)^2  expanded →  quadratic in x
    # ------------------------------------------------------------------
    def _add_city_constraints(self):
        n = self.n
        for i in range(n):
            # diagonal: -lambda * x_{i,t}  (from -2 * 1 * x_{i,t})
            for t in range(n):
                idx = self._idx(i, t)
                self.Q[idx, idx] += self.lam_city * (1 - 2)   # -λ(2-1)? let's expand properly
            # off-diagonal: 2*lambda * x_{i,t1} * x_{i,t2}
        # Proper expansion of (sum_t x_{i,t} - 1)^2:
        #  = sum_t x_{i,t}^2 + 2 sum_{t<t'} x_{i,t}x_{i,t'} - 2 sum_t x_{i,t} + 1
        # Since x^2 = x for binary: sum_t x_{i,t}(1-2) + 2 sum_{t<t'} x_{i,t}x_{i,t'}
        # Reset and redo properly
        for i in range(n):
            for t in range(n):
                idx = self._idx(i, t)
                self.Q[idx, idx] += self.lam_city * (-1)  # coefficient of x_{i,t}
                for t2 in range(t + 1, n):
                    idx2 = self._idx(i, t2)
                    self.Q[idx, idx2] += self.lam_city * 2

    # ------------------------------------------------------------------
    # Constraint 2: Each position filled exactly once
    # ------------------------------------------------------------------
    def _add_position_constraints(self):
        n = self.n
        for t in range(n):
            for i in range(n):
                idx = self._idx(i, t)
                self.Q[idx, idx] += self.lam_pos * (-1)
                for i2 in range(i + 1, n):
                    idx2 = self._idx(i2, t)
                    self.Q[idx, idx2] += self.lam_pos * 2

    # ------------------------------------------------------------------
    # Objective 3: Minimize travel distance
    #   sum_{i,j,t} d_{i,j} * x_{i,t} * x_{j,t+1}
    # ------------------------------------------------------------------
    def _add_distance_objective(self):
        n = self.n
        d = self.inst.d
        for t in range(n - 1):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        idx_i = self._idx(i, t)
                        idx_j = self._idx(j, t + 1)
                        r, c = min(idx_i, idx_j), max(idx_i, idx_j)
                        self.Q[r, c] += self.lam_dist * d[i, j]

    # ------------------------------------------------------------------
    # Penalty 4: Time-window penalty Hamiltonian
    #   Soft penalty P_tw = sum_i max(0, a_i - l_i)^2 + max(0, e_i - a_i)^2
    #   where a_i is the estimated arrival time at city i.
    #
    #   Arrival time approximation:
    #     a_i ≈ sum_{t} t * (T_max / n) * x_{i,t}     (position → time mapping)
    #
    #   Penalty for city i being visited at position t if arrival outside window:
    #     arrival_est = t * (T_max / n)
    #     pen_i_t = max(0, arrival_est - l_i) + max(0, e_i - arrival_est)
    #
    #   Added as diagonal QUBO term: lam_tw * pen_i_t^2 * x_{i,t}
    # ------------------------------------------------------------------
    def _add_time_window_penalties(self):
        n = self.n
        T_max = self.inst.T_max if self.inst.T_max > 0 else 1.0
        time_step = T_max / max(n - 1, 1)
        tw = self.inst.time_windows

        for i in range(n):
            e_i, l_i = tw[i, 0], tw[i, 1]
            for t in range(n):
                arr = t * time_step          # estimated arrival
                late  = max(0.0, arr - l_i)  # lateness
                early = max(0.0, e_i - arr)  # earliness
                penalty = (late ** 2 + early ** 2)
                if penalty > 0:
                    idx = self._idx(i, t)
                    self.Q[idx, idx] -= self.lam_tw * penalty  # negative on diagonal = penalise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self):
        """Return upper-triangular QUBO matrix Q."""
        self.Q = np.zeros((self.N, self.N))
        self._add_city_constraints()
        self._add_position_constraints()
        self._add_distance_objective()
        self._add_time_window_penalties()
        return self.Q

    def energy(self, bitstring):
        """Evaluate QUBO energy for a binary vector x."""
        x = np.array(bitstring, dtype=float)
        return float(x @ self.Q @ x)

    def decode(self, bitstring):
        """Decode a QUBO bitstring into a route (list of city indices)."""
        n = self.n
        route = [-1] * n
        for i in range(n):
            for t in range(n):
                if bitstring[self._idx(i, t)] == 1:
                    route[t] = i
        return route

    def is_valid_route(self, bitstring):
        route = self.decode(bitstring)
        return -1 not in route and len(set(route)) == self.n

    def route_cost(self, route):
        """Total travel distance for a decoded route list."""
        if -1 in route:
            return float('inf')
        cost = 0.0
        for t in range(len(route) - 1):
            cost += self.inst.d[route[t], route[t + 1]]
        return cost

    def time_window_violations(self, route):
        """Return list of (city, estimated_arrival, window) for TW violations."""
        n = self.n
        T_max = self.inst.T_max if self.inst.T_max > 0 else 1.0
        time_step = T_max / max(n - 1, 1)
        violations = []
        tw = self.inst.time_windows
        for t, city in enumerate(route):
            if city == -1:
                continue
            arr = t * time_step
            e, l = tw[city]
            if arr < e or arr > l:
                violations.append((city, arr, (e, l)))
        return violations


def make_sample_instance(n=3, seed=42):
    """Create a small random TSPTW instance for demo."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2)).tolist()
    # Time windows: give each city a reasonable window
    windows = []
    for i in range(n):
        e = rng.uniform(0, 5)
        l = e + rng.uniform(2, 6)
        windows.append((float(e), float(l)))
    return TSPTWInstance(coords, windows)
