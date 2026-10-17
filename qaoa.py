"""
PennyLane QAOA Circuit for TSPTW-QUBO
======================================
Implements the Quantum Approximate Optimization Algorithm (QAOA) with:
  - Problem Hamiltonian  H_C  derived from the QUBO matrix, divided by its largest
    |coefficient| so one step size and one range of initial angles suit every instance
  - Mixer "x"   transverse field Σ X_i, starting from |+⟩^N
    Mixer "xy"  XY ring inside each customer's row of qubits, starting from a W state per
                row: every state keeps each customer at exactly one position, so the
                each-customer-once penalty is zero there and is left out of H_C
  - Variational parameters (γ, β) trained with Adam from one or more random starts

n customers → n² qubits (depot fixed), simulated on lightning.qubit with adjoint gradients.
"""

import itertools

import autograd.numpy as anp
import numpy as np
import pennylane as qml
from autograd import value_and_grad
from pennylane import numpy as pnp
from scipy.optimize import minimize

from qubo import QUBOBuilder, all_routes, solve_exact


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian construction from QUBO
# ─────────────────────────────────────────────────────────────────────────────

def ising_terms(Q: np.ndarray):
    """
    QUBO energy  E(x) = xᵀQx  (Q upper triangular) with  x_i = (1 − Z_i)/2  becomes
      E = offset + Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j,  where
      J_ij = Q_ij/4,   h_i = −Q_ii/2 − Σ_{j≠i} J_ij,   offset = Σ_i Q_ii/2 + Σ_{i<j} J_ij

    Returns (h, J, offset)
    """
    J = np.triu(Q, 1) / 4
    h = -np.diag(Q) / 2 - J.sum(axis=0) - J.sum(axis=1)
    return h, J, float(np.diag(Q).sum() / 2 + J.sum())


def qubo_to_hamiltonian(Q: np.ndarray):
    """Upper-triangular QUBO matrix Q → PennyLane Hamiltonian (Z and ZZ terms of
    ising_terms).  Returns (H, n_qubits, offset)"""
    n = Q.shape[0]
    h, J, offset = ising_terms(Q)

    coeffs, ops = [], []
    for i in np.flatnonzero(np.abs(h) > 1e-12):
        coeffs.append(float(h[i]))
        ops.append(qml.PauliZ(int(i)))
    for i, j in zip(*np.nonzero(np.abs(J) > 1e-12)):
        coeffs.append(float(J[i, j]))
        ops.append(qml.PauliZ(int(i)) @ qml.PauliZ(int(j)))
    if not coeffs:
        coeffs, ops = [0.0], [qml.Identity(0)]
    return qml.Hamiltonian(coeffs, ops), n, offset


def w_state(wires):
    """Equal-amplitude superposition of the one-hot states on `wires`."""
    qml.PauliX(wires[0])
    m = len(wires)
    for i in range(m - 1):          # keep 1/√m on wires[i], pass the rest to wires[i+1]
        qml.SingleExcitation(-2 * np.arccos(1 / np.sqrt(m - i)), wires=[wires[i], wires[i + 1]])


def ring(wires):
    """Edges of a ring over `wires` (one edge for two wires)."""
    return [tuple(wires)] if len(wires) == 2 else list(zip(wires, wires[1:] + wires[:1]))


# ─────────────────────────────────────────────────────────────────────────────
# QAOA Circuit
# ─────────────────────────────────────────────────────────────────────────────

class TSPTWQAOACircuit:
    """
    QAOA circuit wrapping a TSPTW QUBO instance.

    Usage
    -----
    >>> qc = TSPTWQAOACircuit(builder, p=2, mixer="xy")
    >>> result = qc.optimize(steps=60, starts=3)
    >>> best_bs, best_E, route, counts = qc.sample_best(n_shots=512)
    """

    def __init__(self, builder: QUBOBuilder, p: int = 2, seed=0, mixer="x", normalize=True):
        if mixer not in ("x", "xy"):
            raise ValueError(f"unknown mixer {mixer!r}")
        self.builder = builder
        self.p = p
        self.mixer = mixer
        self.n_qubits = builder.N
        self.rng = np.random.default_rng(seed)   # parameter init and shot sampling
        n = builder.n
        self.rows = [[builder._idx(j, k) for k in range(n)] for j in range(1, n + 1)]

        Q, self.qubo_offset = builder.qubo(rows=(mixer == "x"))
        self.H_C, _, self.offset = qubo_to_hamiltonian(Q)
        self.scale = max(abs(c) for c in self.H_C.coeffs) if normalize else 1.0
        self.scale = self.scale or 1.0
        self._H = qml.Hamiltonian([c / self.scale for c in self.H_C.coeffs], self.H_C.ops)

        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        self._circuit = qml.QNode(self._qaoa_circuit, self.dev, diff_method=None)
        self._cost_fn = qml.QNode(self._cost_circuit, self.dev,
                                  interface="autograd", diff_method="adjoint")

        # Optimisation state
        self.params = None
        self.loss_history = []
        self.best_params = None

    # ── Circuit definition ────────────────────────────────────────────────────

    def _ansatz(self, params):
        gammas = params[:self.p]
        betas  = params[self.p:]

        if self.mixer == "x":
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
        else:
            for row in self.rows:
                w_state(row)

        for layer in range(self.p):
            # Problem unitary  U_C(γ): exact in one step, the Z terms commute
            qml.ApproxTimeEvolution(self._H, gammas[layer], 1)
            # Mixer unitary    U_B(β)
            if self.mixer == "x":
                for i in range(self.n_qubits):
                    qml.RX(2 * betas[layer], wires=i)
            else:
                for row in self.rows:
                    for a, b in ring(row):
                        qml.IsingXY(-2 * betas[layer], wires=[a, b])

    def _qaoa_circuit(self, params):
        """Full QAOA ansatz returning state."""
        self._ansatz(params)
        return qml.state()

    def _cost_circuit(self, params):
        """QAOA ansatz returning ⟨H_C⟩ (normalised units)."""
        self._ansatz(params)
        return qml.expval(self._H)

    # ── Optimisation ─────────────────────────────────────────────────────────

    def optimize(self, steps: int = 80, stepsize: float = 0.05,
                 init_params=None, callback=None, starts: int = 1):
        """
        Adam on ⟨H_C⟩ from `starts` random initial angles (or once from init_params),
        keeping the best.  Energies are expected QUBO energies, in QUBO units.

        Returns dict with keys: params, loss_history (winning start), best_energy,
        start_energies (best energy reached by each start)
        """
        runs = 1 if init_params is not None else starts
        best_energy, start_energies = float('inf'), []

        for run in range(runs):
            init = init_params if init_params is not None else \
                self.rng.uniform(0, np.pi, size=2 * self.p)
            params = pnp.array(init, requires_grad=True)
            opt = qml.AdamOptimizer(stepsize=stepsize)
            history, run_best, run_params = [], float('inf'), params

            for step in range(steps):
                before = params
                params, loss = opt.step_and_cost(self._cost_fn, params)   # loss is at `before`
                energy = float(loss) * self.scale + self.offset + self.qubo_offset
                history.append(energy)
                if energy < run_best:
                    run_best, run_params = energy, before
                if callback is not None:
                    callback(run * steps + step, energy, params)

            start_energies.append(run_best)
            if run_best < best_energy:
                best_energy, self.best_params, self.loss_history = run_best, run_params, history

        self.params = params
        return {
            "params": np.array(self.best_params),
            "loss_history": self.loss_history,
            "best_energy": best_energy,
            "start_energies": start_energies,
        }

    # ── Sampling ──────────────────────────────────────────────────────────────

    def get_statevector(self, params=None):
        if params is None:
            params = self.best_params if self.best_params is not None else self.params
        return np.array(self._circuit(params))

    def sample_bitstrings(self, n_shots: int = 1024, params=None):
        """Sample bitstrings from the QAOA output state."""
        sv = self.get_statevector(params)
        probs = np.abs(sv) ** 2
        probs = probs / probs.sum()
        indices = self.rng.choice(len(probs), size=n_shots, p=probs)
        counts = {}
        for idx in indices:
            bs = tuple(int(b) for b in format(idx, f'0{self.n_qubits}b'))
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    def sample_best(self, n_shots: int = 1024, params=None):
        """Return the most-probable valid bitstring and its QUBO energy."""
        counts = self.sample_bitstrings(n_shots, params)
        sorted_bs = sorted(counts, key=lambda b: -counts[b])

        best_valid = None
        best_energy = float('inf')
        best_route = None

        for bs in sorted_bs:
            energy = self.builder.energy(bs)
            if self.builder.is_valid_route(bs):
                if energy < best_energy:
                    best_energy = energy
                    best_valid = bs
                    best_route = self.builder.decode(bs)

        if best_valid is None:
            # Fall back to lowest-energy sample
            best_valid = min(sorted_bs[:20], key=lambda b: self.builder.energy(b))
            best_energy = self.builder.energy(best_valid)
            best_route = self.builder.decode(best_valid)

        return best_valid, best_energy, best_route, counts

    def probability_landscape(self, params=None):
        """Return (bitstring_ints, probabilities) arrays."""
        sv = self.get_statevector(params)
        probs = np.abs(sv) ** 2
        return np.arange(len(probs)), probs

    # ── Classical exact solver (for comparison) ─────────────────────────────

    def brute_force(self):
        """True TSPTW optimum by enumerating the n! routes: (bitstring, QUBO energy, route),
        or (None, None, []) when no route meets every time window."""
        route, _ = solve_exact(self.builder.inst)
        if route is None:
            return None, None, []
        bs = self.builder.encode(route)
        return bs, self.builder.energy(bs), route


# ─────────────────────────────────────────────────────────────────────────────
# Fast exact simulation (no PennyLane)
# ─────────────────────────────────────────────────────────────────────────────

def interp(x):
    """INTERP (Zhou et al. 2020): depth-p angles → a depth-(p+1) start, by linear
    interpolation of the γ schedule and of the β schedule."""
    p = len(x) // 2

    def up(v):
        v = np.r_[0.0, v, 0.0]
        i = np.arange(1, p + 2)
        return (i - 1) / p * v[i - 1] + (p - i + 1) / p * v[i]
    return np.r_[up(x[:p]), up(x[p:])]


class FastQAOA:
    """
    The circuit of TSPTWQAOACircuit, simulated exactly on the states its mixer can reach:
      "xy"  every customer at exactly one position: n^n amplitudes (27 / 256 / 3125 for
            n = 3 / 4 / 5) instead of 2^(n²); each row's XY ring is a rotation of that row's
            single excitation, applied gate by gate in the circuit's order
      "x"   all 2^(n²) states (n ≤ 4)
    Same normalised H_C, same angles (γ_1..γ_p, β_1..β_p; p = len(params) / 2) and the same
    energies, in QUBO units.  Gradients are exact (autograd).  tests.py checks the state, the
    energy and the gradient against the PennyLane circuit.

    Speed: about 2 / 3 / 5 ms per value-and-gradient at p = 3 for "xy" with n = 3 / 4 / 5, and
    22 ms for "x" with n = 4.  numpy's BLAS threads fight when several processes run at once
    (100× slower): set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 before numpy is imported
    in each worker, as experiments.py does.
    """

    def __init__(self, builder: QUBOBuilder, mixer="xy", seed=0, normalize=True):
        n = self.n = builder.n
        self.builder, self.mixer = builder, mixer
        self.rng = np.random.default_rng(seed)
        if mixer == "xy":
            pos = np.array(list(itertools.product(range(n), repeat=n)))   # pos[s, j-1]: customer j's position
            bits = np.zeros((len(pos), n, n))
            bits[np.arange(len(pos))[:, None], np.arange(n), pos] = 1
            self.shape = (n,) * n
            self._eye = np.eye(n)
            self._edges = []
            for a, b in ring(list(range(n))):
                P, X = np.zeros((n, n)), np.zeros((n, n))
                P[a, a] = P[b, b] = 1
                X[a, b] = X[b, a] = 1
                self._edges.append((P, X))
        elif mixer == "x":
            if n * n > 16:
                raise ValueError("the X mixer needs all 2^(n²) states: n ≤ 4 here")
            N = n * n
            bits = (np.arange(2 ** N)[:, None] >> np.arange(N - 1, -1, -1)) & 1
            # the RX layer is a tensor product of 2×2 matrices: group the qubits by four, so
            # each group is one axis of the amplitude tensor and gets one 16×16 block whose
            # (a, b) entry is c^(g−f)(−is)^f, f = number of qubits where a and b differ
            self._sizes = [4] * (N // 4) + ([N % 4] if N % 4 else [])
            self.shape = tuple(2 ** g for g in self._sizes)
            self._flips = {}
            for g in set(self._sizes):
                b = (np.arange(2 ** g)[:, None] >> np.arange(g)) & 1
                self._flips[g] = (b[:, None, :] != b[None, :, :]).sum(-1)
        else:
            raise ValueError(f"unknown mixer {mixer!r}")

        Q, self.qubo_offset = builder.qubo(rows=(mixer == "x"))
        h, J, self.offset = ising_terms(Q)
        self.scale = (max(np.abs(h).max(), np.abs(J).max()) if normalize else 1.0) or 1.0
        self.bits = bits.reshape(len(bits), n * n).astype(float)    # qubit (j-1)·n + k
        xQx = np.einsum("si,ij,sj->s", self.bits, Q, self.bits)
        self.energies = xQx + self.qubo_offset                      # QUBO energy of each basis state
        self.D = (xQx - self.offset) / self.scale                   # diagonal of the normalised H_C
        self.psi0 = np.full(len(self.bits), len(self.bits) ** -0.5, dtype=complex)
        self._vg = value_and_grad(self._loss)
        self._flag_cache = None

    # ── simulation ───────────────────────────────────────────────────────────

    @staticmethod
    def _apply(mats, psi):
        """Matrix mats[i] on axis i of psi: contract the last axis with a matrix product and
        rotate it to the front, for the last axis first, so the axes come back in order."""
        for M in reversed(mats):
            psi = anp.dot(psi.reshape(-1, len(M)), M.T).reshape(psi.shape)
            psi = anp.transpose(psi, (psi.ndim - 1, *range(psi.ndim - 1)))
        return psi

    def _mix(self, psi, beta):
        c, s = anp.cos(beta), anp.sin(beta)
        if self.mixer == "x":                    # RX(2β) = exp(−iβX) on every qubit
            block = {g: c ** (g - f) * (-1j * s) ** f for g, f in self._flips.items()}
            mats = [block[g] for g in self._sizes]
        else:                                    # IsingXY(−2β) around the ring, first edge first
            M = self._eye + 0j
            for P, X in self._edges:
                M = anp.dot(self._eye + (c - 1) * P - 1j * s * X, M)
            mats = [M] * psi.ndim
        return self._apply(mats, psi)

    def state(self, params):
        """Final amplitudes over the basis states (flat)."""
        p = len(params) // 2
        D = self.D.reshape(self.shape)
        psi = self.psi0.reshape(self.shape)
        for layer in range(p):
            psi = psi * anp.exp(-1j * params[layer] * D)
            psi = self._mix(psi, params[p + layer])
        return psi.reshape(-1)

    def _loss(self, params):
        psi = self.state(params)
        return anp.sum((anp.real(psi) ** 2 + anp.imag(psi) ** 2) * self.D)

    def loss_and_grad(self, params):
        """⟨H_C⟩ in normalised units and its exact gradient."""
        return self._vg(np.asarray(params, dtype=float))

    def energy(self, params):
        """Expected QUBO energy."""
        return float(self._loss(np.asarray(params, dtype=float))) * self.scale + self.offset + self.qubo_offset

    def to_energy(self, loss):
        return loss * self.scale + self.offset + self.qubo_offset

    # ── what the final state says about routes ────────────────────────────────

    def route_masses(self, params):
        """(P(valid route), P(valid and on time), P(optimal on-time route))."""
        if self._flag_cache is None:
            n, inst = self.n, self.builder.inst
            b = self.bits.reshape(-1, n, n)                          # [state, customer, position]
            valid = (b.sum(2) == 1).all(1) & (b.sum(1) == 1).all(1)
            perms, feasible, cost = all_routes(inst)
            key = lambda orders: orders @ (n + 1) ** np.arange(n)
            by_key = np.argsort(key(perms))
            seq = np.argsort(np.argmax(b[valid], axis=2), axis=1) + 1   # customer at each position
            idx = by_key[np.searchsorted(key(perms)[by_key], key(seq))]
            on_time, optimal = np.zeros(len(b), bool), np.zeros(len(b), bool)
            on_time[valid] = feasible[idx]
            optimal[valid] = feasible[idx] & np.isclose(cost[idx], cost[feasible].min())
            self._flag_cache = valid, on_time, optimal
        pr = np.abs(self.state(np.asarray(params, dtype=float))) ** 2
        return tuple(float(pr[mask].sum()) for mask in self._flag_cache)

    def ground_optimal(self):
        """Share of the ground states (lowest-energy basis states) that are optimal on-time
        routes.  For a unique ground state it is the limit of P(opt) as p → ∞: 1 if the
        Hamiltonian's minimum is the TSPTW optimum, 0 if an infeasible route undercuts it."""
        self.route_masses(np.zeros(0))
        return float(self._flag_cache[2][np.isclose(self.energies, self.energies.min())].mean())

    # ── optimisers ───────────────────────────────────────────────────────────

    def random_start(self, p):
        return self.rng.uniform(0, np.pi, size=2 * p)

    def adam(self, x0, steps=60, stepsize=0.05):
        """PennyLane's Adam (β = 0.9, 0.99, ε = 1e-8).  Returns (best params, best energy);
        'best' = lowest ⟨H_C⟩ seen, at the parameters before an update, as TSPTWQAOACircuit."""
        x = np.array(x0, dtype=float)
        fm, sm = np.zeros_like(x), np.zeros_like(x)
        best_f, best_x = np.inf, x
        for t in range(1, steps + 1):
            f, g = self.loss_and_grad(x)
            if f < best_f:
                best_f, best_x = f, x
            fm = 0.9 * fm + 0.1 * g
            sm = 0.99 * sm + 0.01 * g * g
            x = x - stepsize * np.sqrt(1 - 0.99 ** t) / (1 - 0.9 ** t) * fm / (np.sqrt(sm) + 1e-8)
        return best_x, self.to_energy(best_f)

    def lbfgs(self, x0, maxiter=200):
        """L-BFGS-B with exact gradients.  Returns (params, energy, function evaluations)."""
        r = minimize(self.loss_and_grad, np.asarray(x0, dtype=float), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=maxiter))
        return r.x, self.to_energy(r.fun), r.nfev

    def ladder(self, pmax, starts=5, maxiter=200, restarts=0):
        """Depths 1..pmax by INTERP: best of `starts` random L-BFGS starts at p = 1, then each
        depth starts from the previous optimum (plus `restarts` random starts at every depth
        ≥ 2, keeping the best).  Returns {p: (params, energy, function evaluations so far)}."""
        runs = [self.lbfgs(self.random_start(1), maxiter) for _ in range(starts)]
        nfev = sum(r[2] for r in runs)
        x, e, _ = min(runs, key=lambda r: r[1])
        out = {1: (x, e, nfev)}
        for p in range(2, pmax + 1):
            runs = [self.lbfgs(interp(x), maxiter)]
            runs += [self.lbfgs(self.random_start(p), maxiter) for _ in range(restarts)]
            nfev += sum(r[2] for r in runs)
            x, e, _ = min(runs, key=lambda r: r[1])
            out[p] = (x, e, nfev)
        return out
