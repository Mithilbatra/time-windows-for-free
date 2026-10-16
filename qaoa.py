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

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from qubo import QUBOBuilder, solve_exact


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian construction from QUBO
# ─────────────────────────────────────────────────────────────────────────────

def qubo_to_hamiltonian(Q: np.ndarray):
    """
    Convert upper-triangular QUBO matrix Q to a PennyLane Hamiltonian.

    QUBO energy  E(x) = xᵀQx  with  x_i = (1 − Z_i)/2  becomes
      E = offset + Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j,  where
      J_ij = Q_ij/4,   h_i = −Q_ii/2 − Σ_{j≠i} J_ij,   offset = Σ_i Q_ii/2 + Σ_{i<j} J_ij

    Returns (H, n_qubits, offset)
    """
    n = Q.shape[0]
    J = np.triu(Q, 1) / 4
    h = -np.diag(Q) / 2 - J.sum(axis=0) - J.sum(axis=1)
    offset = float(np.diag(Q).sum() / 2 + J.sum())

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
