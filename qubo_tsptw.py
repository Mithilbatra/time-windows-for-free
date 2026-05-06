"""
================================================================================
TSPTW — QUBO & Ising Hamiltonian Builder
================================================================================

MATHEMATICAL DERIVATION
========================

The Traveling Salesman Problem with Time Windows (TSPTW) is encoded as a
three-part Quadratic Unconstrained Binary Optimization (QUBO) problem:

    H_total = H_A  +  H_B  +  H_T
                ↑       ↑       ↑
           Constraint  Route  Time-window
           penalty     cost   penalty

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUBIT ENCODING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Binary variable:  x_{v,i} ∈ {0,1}
  = 1  →  city v is visited at position i in the tour
  = 0  →  city v is NOT at position i

Qubit index mapping:  q = v * n + i
  (city v at position i maps to qubit number v*n + i)

Total qubits needed: n²   (n cities × n positions)

For n=4 cities: 16 qubits
For n=5 cities: 25 qubits

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
H_A: CONSTRAINT HAMILTONIAN  (penalty = A)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rule 1: Each CITY visited exactly once
    H_A1 = A · Σ_v  (1 - Σ_i x_{v,i})²

Rule 2: Each POSITION has exactly one city
    H_A2 = A · Σ_i  (1 - Σ_v x_{v,i})²

Expansion of (1 - Σ x)²:
    = 1 - 2·Σx + (Σx)² = 1 - Σx + 2·Σ_{a<b} x_a·x_b
    (using x² = x for binary variables)

→ Constant: A per constraint group
→ Linear terms: -A per variable
→ Quadratic terms: 2A per pair in the same group

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
H_B: OBJECTIVE HAMILTONIAN  (weight = B)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    H_B = B · Σ_{u≠v} W[u,v] · Σ_i  x_{u,i} · x_{v,(i+1) mod n}

Meaning: if city u is at position i AND city v is at position i+1,
         add the travel cost W[u,v] to the objective.

→ Quadratic terms: B·W[u,v] for each (u@pos_i, v@pos_{i+1}) pair

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
H_T: TIME-WINDOW PENALTY HAMILTONIAN  (weight = T_pen)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each city v has a time window [e_v, l_v]:
  e_v = earliest acceptable arrival time
  l_v = latest acceptable arrival time

Key insight: If city v is at position i in the tour, its expected arrival
time can be estimated as:

    t_expected(i) = i · mean_edge_weight

This is an approximation — the actual arrival depends on which cities
preceded v in the tour. For the QUBO formulation, we use this position-
based estimate to create a LINEAR penalty (no auxiliary qubits needed).

Incompatibility score  ψ[v,i]:
  ψ[v,i] = max(0, e_v - t_expected(i))   if arriving too EARLY
           + max(0, t_expected(i) - l_v)   if arriving too LATE
           normalized to [0, 1]

Time-window Hamiltonian:
    H_T = T_pen · Σ_{v,i}  ψ[v,i] · x_{v,i}

This is a PURELY LINEAR term — it only adds to the diagonal of the
QUBO matrix (no new qubit-qubit interactions needed).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ISING CONVERSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quantum computers work with Pauli operators Z ∈ {+1, -1}.
We convert using:   x = (1 - Z) / 2

    Z = +1  →  |0⟩  →  x = 0  (city NOT at this position)
    Z = -1  →  |1⟩  →  x = 1  (city IS at this position)

After substitution:

    From c_i · x_i:
        → constant: +c_i/2
        → linear Z: -c_i/2 · Z_i

    From c_ij · x_i · x_j:
        → constant: +c_ij/4
        → linear Z_i: -c_ij/4 · Z_i
        → linear Z_j: -c_ij/4 · Z_j
        → quadratic: +c_ij/4 · Z_i · Z_j

Final Ising Hamiltonian:
    H_Ising = Σ_i h_i · Z_i + Σ_{i<j} J_ij · Z_i · Z_j + const

Where:
    h_i   = -c_i/2 - Σ_{j≠i} c_{ij}/4     (Ising linear coupling)
    J_ij  = c_ij/4                           (Ising quadratic coupling)

================================================================================
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def qubit_index(city: int, position: int, n: int) -> int:
    """
    Map (city, position) pair to a single qubit index.

    The n² qubits are arranged as a flattened n×n matrix:
        qubit 0     = city 0, position 0
        qubit 1     = city 0, position 1
        ...
        qubit n     = city 1, position 0
        ...
        qubit n²-1  = city n-1, position n-1

    Args:
        city:     City index (0 to n-1)
        position: Position in tour (0 to n-1)
        n:        Total number of cities

    Returns:
        Integer qubit index in [0, n²-1]
    """
    return city * n + position


def compute_time_incompatibility(
    n: int,
    W: np.ndarray,
    time_windows: list,
) -> np.ndarray:
    """
    Compute the time-window incompatibility matrix ψ[city, position].

    For each (city v, tour position i) pair, ψ[v,i] measures how badly
    city v's delivery time window conflicts with arriving at position i.

    Method:
        - Expected arrival at position i ≈ i × mean_edge_weight
        - ψ[v,i] = max(0, e_v - t_exp) + max(0, t_exp - l_v)
        - Normalized to [0, 1] range

    Args:
        n:            Number of cities
        W:            n×n distance matrix
        time_windows: List of (earliest, latest) tuples per city

    Returns:
        incompatibility: (n, n) float array, values in [0, 1]
    """
    nonzero = W[W > 0]
    mean_w  = float(np.mean(nonzero)) if len(nonzero) > 0 else 1.0
    max_w   = float(np.max(W))

    incompatibility = np.zeros((n, n))

    for v in range(n):
        e_v, l_v = float(time_windows[v][0]), float(time_windows[v][1])

        for i in range(n):
            # Expected arrival time if city v is at position i
            t_exp = i * mean_w

            early_penalty = max(0.0, e_v - t_exp)   # arrive before window opens
            late_penalty  = max(0.0, t_exp - l_v)   # arrive after window closes

            incompatibility[v, i] = early_penalty + late_penalty

    # Normalize so the maximum violation = 1.0
    max_inc = incompatibility.max()
    if max_inc > 0:
        incompatibility /= max_inc

    return incompatibility


# ─────────────────────────────────────────────────────────────────────────────
# QUBO BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_qubo(
    n: int,
    W: np.ndarray,
    time_windows: list,
    A: float = 10.0,
    B: float = 1.0,
    T_pen: float = 5.0,
) -> tuple:
    """
    Build the complete TSPTW QUBO matrix.

    H_total = H_A (constraints) + H_B (distance) + H_T (time windows)

    Args:
        n:            Number of cities
        W:            n×n symmetric distance matrix
        time_windows: List of (earliest, latest) per city
        A:            Constraint penalty weight (must be > B·max(W))
        B:            Distance objective weight
        T_pen:        Time-window penalty weight

    Returns:
        qubo_linear:       (n²,) array of linear QUBO coefficients c_i
        qubo_quad:         (n², n²) array of quadratic coefficients c_ij (upper tri)
        qubo_constant:     float energy constant
        incompatibility:   (n, n) time-window incompatibility scores
    """
    N   = n * n   # total binary variables (= qubits needed)

    qubo_linear   = np.zeros(N)
    qubo_quad     = np.zeros((N, N))   # upper triangular
    qubo_constant = 0.0

    incompatibility = compute_time_incompatibility(n, W, time_windows)

    # ── H_A Part 1: Each city visited exactly once ────────────────────────────
    # A · Σ_v (1 - Σ_i x_{v,i})²
    # Expanded: A · Σ_v [1 - Σ_i x_{v,i} + 2·Σ_{i<j} x_{v,i}·x_{v,j}]
    for v in range(n):
        qubo_constant += A
        for i in range(n):
            qi = qubit_index(v, i, n)
            qubo_linear[qi] += -A                # linear: -A·x_{v,i}
            for j in range(i + 1, n):
                qj = qubit_index(v, j, n)
                qubo_quad[qi, qj] += 2.0 * A     # quadratic: 2A·x_{v,i}·x_{v,j}

    # ── H_A Part 2: Each position has exactly one city ────────────────────────
    # A · Σ_i (1 - Σ_v x_{v,i})²
    for i in range(n):
        qubo_constant += A
        for v in range(n):
            qi = qubit_index(v, i, n)
            qubo_linear[qi] += -A
            for u in range(v + 1, n):
                qu = qubit_index(u, i, n)
                qubo_quad[qi, qu] += 2.0 * A

    # ── H_B: Distance objective ───────────────────────────────────────────────
    # B · Σ_{u≠v} W[u,v] · Σ_i x_{u,i} · x_{v,(i+1) mod n}
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            for i in range(n):
                next_i = (i + 1) % n
                qu     = qubit_index(u, i, n)
                qv     = qubit_index(v, next_i, n)
                # Ensure upper-triangular storage
                lo, hi = min(qu, qv), max(qu, qv)
                qubo_quad[lo, hi] += B * W[u, v]

    # ── H_T: Time-window penalty ──────────────────────────────────────────────
    # T_pen · Σ_{v,i} ψ[v,i] · x_{v,i}   (purely linear)
    for v in range(n):
        for i in range(n):
            qi = qubit_index(v, i, n)
            qubo_linear[qi] += T_pen * incompatibility[v, i]

    return qubo_linear, qubo_quad, qubo_constant, incompatibility


# ─────────────────────────────────────────────────────────────────────────────
# ISING CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

def qubo_to_ising(
    qubo_linear:   np.ndarray,
    qubo_quad:     np.ndarray,
    qubo_constant: float,
) -> tuple:
    """
    Convert QUBO to Ising Hamiltonian via x = (1 - Z) / 2.

    Derivation:
        c_i · x_i   → -c_i/2 · Z_i + c_i/2   (constant)
        c_ij · x_ix_j → c_ij/4 · Z_iZ_j
                       - c_ij/4 · Z_i
                       - c_ij/4 · Z_j
                       + c_ij/4              (constant)

    Returns:
        h_ising:    (N,) linear Ising couplings
        J_ising:    (N, N) quadratic Ising couplings (upper triangular)
        ising_const: total energy constant
    """
    N = len(qubo_linear)

    h_ising    = np.zeros(N)
    J_ising    = np.zeros((N, N))
    ising_const = qubo_constant

    # Contribution from linear QUBO terms c_i · x_i
    for i in range(N):
        h_ising[i]   += -qubo_linear[i] / 2.0
        ising_const  +=  qubo_linear[i] / 2.0

    # Contribution from quadratic QUBO terms c_ij · x_i · x_j (upper tri)
    for i in range(N):
        for j in range(i + 1, N):
            c = qubo_quad[i, j]
            if abs(c) < 1e-12:
                continue
            J_ising[i, j] +=  c / 4.0
            h_ising[i]    += -c / 4.0
            h_ising[j]    += -c / 4.0
            ising_const   +=  c / 4.0

    return h_ising, J_ising, ising_const


# ─────────────────────────────────────────────────────────────────────────────
# PENNYLANE HAMILTONIAN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_pennylane_hamiltonian(
    n: int,
    W: np.ndarray,
    time_windows: list,
    A: float = 10.0,
    B: float = 1.0,
    T_pen: float = 5.0,
):
    """
    Full pipeline: TSPTW problem → PennyLane Hamiltonian.

    Args:
        n:            Number of cities
        W:            n×n distance matrix
        time_windows: List of (earliest, latest) per city
        A:            Constraint penalty
        B:            Distance weight
        T_pen:        Time-window penalty

    Returns:
        coeffs:           List of Hamiltonian coefficients
        obs:              List of PennyLane Pauli observables
        ising_const:      Energy constant offset
        incompatibility:  (n,n) time-window incompatibility matrix
        h_ising:          (n²,) Ising linear couplings (for diagnostics)
        J_ising:          (n²,n²) Ising quadratic couplings (for diagnostics)
    """
    import pennylane as qml

    N = n * n  # total qubits

    # Step 1: Build QUBO
    qubo_lin, qubo_quad, qubo_const, incompatibility = build_qubo(
        n, W, time_windows, A, B, T_pen
    )

    # Step 2: Convert to Ising
    h_ising, J_ising, ising_const = qubo_to_ising(qubo_lin, qubo_quad, qubo_const)

    # Step 3: Build PennyLane observables
    coeffs = []
    obs    = []

    # Z terms (linear Ising couplings)
    for i in range(N):
        if abs(h_ising[i]) > 1e-10:
            coeffs.append(float(h_ising[i]))
            obs.append(qml.PauliZ(i))

    # ZZ terms (quadratic Ising couplings)
    for i in range(N):
        for j in range(i + 1, N):
            if abs(J_ising[i, j]) > 1e-10:
                coeffs.append(float(J_ising[i, j]))
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

    return coeffs, obs, ising_const, incompatibility, h_ising, J_ising


# ─────────────────────────────────────────────────────────────────────────────
# HAMILTONIAN DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def hamiltonian_summary(n: int, coeffs: list, obs: list, ising_const: float) -> dict:
    """
    Return a summary dict describing the Hamiltonian structure.

    Useful for displaying circuit complexity in the dashboard.
    """
    n_z  = sum(1 for o in obs if len(o.wires) == 1)
    n_zz = sum(1 for o in obs if len(o.wires) == 2)

    max_coeff = max(abs(c) for c in coeffs) if coeffs else 0.0
    min_coeff = min(abs(c) for c in coeffs if abs(c) > 1e-10) if coeffs else 0.0

    return {
        "n_qubits":       n * n,
        "n_terms":        len(coeffs),
        "n_Z_terms":      n_z,
        "n_ZZ_terms":     n_zz,
        "energy_constant": round(ising_const, 4),
        "max_coefficient": round(max_coeff, 4),
        "min_coefficient": round(min_coeff, 4),
        "circuit_cnot_count": n_zz * 2,   # each ZZ needs 2 CNOTs
        "circuit_rz_count":   n_z + n_zz, # one RZ per Z and ZZ term
    }
