"""
PennyLane QAOA Circuit for TSPTW-QUBO
======================================
Implements the Quantum Approximate Optimization Algorithm (QAOA) with:
  - Problem Hamiltonian  H_C  derived from the QUBO matrix
  - Mixer Hamiltonian    H_B  (transverse field / X-mixer)
  - Variational parameters (γ, β) optimised via gradient descent

For small instances (n≤3 cities → n²≤9 qubits) this runs on the
default.qubit simulator; larger instances use the sparse simulator.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from qubo import QUBOBuilder, TSPTWInstance


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian construction from QUBO
# ─────────────────────────────────────────────────────────────────────────────

def qubo_to_hamiltonian(Q: np.ndarray):
    """
    Convert upper-triangular QUBO matrix Q to a PennyLane Hamiltonian.

    QUBO energy:  E(x) = x^T Q x
    Using  x_i = (1 - Z_i)/2  (spin encoding), we get:
      E = const + sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j

    Returns (H, n_qubits, offset)
    """
    n = Q.shape[0]
    coeffs = []
    ops    = []
    offset = 0.0

    # Diagonal terms:  Q_{ii} * x_i  →  Q_{ii}/4 * (1-Z)
    for i in range(n):
        q = Q[i, i]
        if abs(q) > 1e-12:
            offset += q / 2          # constant part  (Q_ii * 1/2)
            coeffs.append(-q / 2)   # Z_i part
            ops.append(qml.PauliZ(i))

    # Off-diagonal terms: Q_{ij} * x_i * x_j  → Q_{ij}/4*(1-Z_i)(1-Z_j)
    for i in range(n):
        for j in range(i + 1, n):
            q = Q[i, j]
            if abs(q) > 1e-12:
                offset +=  q / 4
                coeffs.append(-q / 4)   # Z_i
                ops.append(qml.PauliZ(i))
                coeffs.append(-q / 4)   # Z_j
                ops.append(qml.PauliZ(j))
                coeffs.append( q / 4)   # Z_i Z_j
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    if not coeffs:
        # trivial
        coeffs = [0.0]
        ops    = [qml.Identity(0)]

    H = qml.Hamiltonian(coeffs, ops)
    return H, n, offset


# ─────────────────────────────────────────────────────────────────────────────
# QAOA Circuit
# ─────────────────────────────────────────────────────────────────────────────

class TSPTWQAOACircuit:
    """
    QAOA circuit wrapping a TSPTW QUBO instance.

    Usage
    -----
    >>> qc = TSPTWQAOACircuit(builder, p=2)
    >>> result = qc.optimize(steps=60)
    >>> best_bs, best_E = qc.sample_best(n_shots=512)
    """

    def __init__(self, builder: QUBOBuilder, p: int = 2):
        self.builder = builder
        self.p = p
        self.n_qubits = builder.N

        Q = builder.build()
        self.H_C, _, self.offset = qubo_to_hamiltonian(Q)

        # Mixer: H_B = -sum_i X_i
        self.H_B = qml.Hamiltonian(
            [-1.0] * self.n_qubits,
            [qml.PauliX(i) for i in range(self.n_qubits)]
        )

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Build QNode
        self._circuit = qml.QNode(self._qaoa_circuit, self.dev,
                                  interface="autograd")
        self._cost_fn = qml.QNode(self._cost_circuit, self.dev,
                                  interface="autograd")

        # Optimisation state
        self.params = None
        self.loss_history = []
        self.best_params = None

    # ── Circuit definition ────────────────────────────────────────────────────

    def _qaoa_circuit(self, params):
        """Full QAOA ansatz returning state."""
        gammas = params[:self.p]
        betas  = params[self.p:]

        # Equal superposition
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)

        for layer in range(self.p):
            # Problem unitary  U_C(γ)
            qml.ApproxTimeEvolution(self.H_C, gammas[layer], 1)
            # Mixer unitary    U_B(β)
            for i in range(self.n_qubits):
                qml.RX(2 * betas[layer], wires=i)

        return qml.state()

    def _cost_circuit(self, params):
        """QAOA ansatz returning ⟨H_C⟩."""
        gammas = params[:self.p]
        betas  = params[self.p:]

        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)

        for layer in range(self.p):
            qml.ApproxTimeEvolution(self.H_C, gammas[layer], 1)
            for i in range(self.n_qubits):
                qml.RX(2 * betas[layer], wires=i)

        return qml.expval(self.H_C)

    # ── Optimisation ─────────────────────────────────────────────────────────

    def optimize(self, steps: int = 80, stepsize: float = 0.05,
                 init_params=None, callback=None):
        """
        Gradient-descent optimisation of QAOA variational parameters.

        Returns dict with keys: params, loss_history, best_energy
        """
        rng = np.random.default_rng(0)
        if init_params is not None:
            params = pnp.array(init_params, requires_grad=True)
        else:
            params = pnp.array(
                rng.uniform(0, np.pi, size=2 * self.p),
                requires_grad=True
            )

        opt = qml.AdamOptimizer(stepsize=stepsize)
        self.loss_history = []
        best_loss = float('inf')
        self.best_params = params.copy()

        for step in range(steps):
            params, loss = opt.step_and_cost(self._cost_fn, params)
            loss_val = float(loss)
            self.loss_history.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                self.best_params = params.copy()

            if callback is not None:
                callback(step, loss_val, params)

        self.params = params
        return {
            "params": np.array(self.best_params),
            "loss_history": self.loss_history,
            "best_energy": best_loss + self.offset
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
        indices = np.random.choice(len(probs), size=n_shots, p=probs)
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

    # ── Classical brute-force (for comparison) ──────────────────────────────

    def brute_force(self):
        """Enumerate all 2^N bitstrings and return the best valid one."""
        N = self.n_qubits
        best_bs = None
        best_e  = float('inf')
        for i in range(2 ** N):
            bs = tuple(int(b) for b in format(i, f'0{N}b'))
            if self.builder.is_valid_route(bs):
                e = self.builder.energy(bs)
                if e < best_e:
                    best_e = e
                    best_bs = bs
        route = self.builder.decode(best_bs) if best_bs else []
        return best_bs, best_e, route
