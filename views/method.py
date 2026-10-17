"""
Method: the encoding, the time-window penalties and their soundness, the circuit cost, and
the code that implements each piece (shown straight from the source, so it never drifts).
"""
import inspect

import streamlit as st

import baselines
import qaoa
import qubo
import ui
from experiments import ENCODINGS, SIZES

SNIPPETS = [
    (qubo.make_instance, "Random instances: Salehi et al.'s protocol, or a planted tour with windows of a given width."),
    (qubo.all_routes, "The exact solver: every customer order, its feasibility (waiting allowed) and its cost."),
    (qubo.earliest_bounds, "E1's dynamic programme: the earliest possible service start of each customer at each position."),
    (qubo.tw_terms, "Turns the bounds into penalty terms: linear for E1, plus pairwise; the look-ahead for E1+C."),
    (qubo.QUBOBuilder.qubo, "The full QUBO with the depot fixed: one-hot constraints, closed-tour distance, window terms."),
    (qaoa.FastQAOA.state, "Exact QAOA on the states the mixer reaches: a diagonal phase, then the mixer, layer by layer."),
    (qaoa.FastQAOA._mix, "The mixers: RX on every qubit (X), or the XY ring inside each customer's row (XY)."),
    (baselines.SalehiQUBO._build, "E2, the slack-qubit baseline, written from Salehi et al.'s equations."),
    (baselines.anneal, "The simulated annealer used for every encoding."),
]

st.markdown("# 🧮 Method")
encoding, penalty, cost, code = st.tabs(["Encoding", "Time-window penalties", "Circuit cost", "Code"])

with encoding:
    st.markdown(
        "A route visits customers 1…n once each and returns to the depot 0. Customer $j$ must be served "
        "inside its window $[e_j, l_j]$; arriving early means waiting. With the depot fixed at position 0, "
        "one qubit per (customer, position) pair is enough: **n² qubits**.")
    st.latex(r"x_{j,k} = 1 \iff \text{customer } j \text{ is the } k\text{-th stop}, \qquad H = H_A + H_B + H_T")
    st.latex(r"H_A = A\sum_j \Big(1 - \sum_k x_{j,k}\Big)^2 + A\sum_k \Big(1 - \sum_j x_{j,k}\Big)^2")
    st.latex(r"H_B = B\Big[\sum_j d_{0j}\,x_{j,1} + \sum_k \sum_{u \neq j} d_{uj}\,x_{u,k}\,x_{j,k+1} "
             r"+ \sum_j d_{j0}\,x_{j,n}\Big]")
    st.markdown(
        "The encodings differ only in $H_T$, the time-window term:\n"
        "- **E0**: none (the control).\n"
        "- **E1** (ours): penalties from earliest-arrival bounds. No extra qubits, no extra two-qubit gates.\n"
        "- **E1+C** (ours): E1 plus a look-ahead two positions ahead, at about 9% more two-qubit terms.\n"
        "- **E2** (Salehi et al., 2022): an edge-based QUBO that enforces the windows exactly with "
        "binary-encoded slack and waiting variables. Too many qubits for QAOA simulation, so it is run "
        "with simulated annealing only.\n\n"
        "With the **XY mixer** every state keeps each customer at exactly one position, so the first sum "
        "of $H_A$ is always zero and is left out of the circuit.")

with penalty:
    st.latex(r"LB_{j,1} = \max\big(e_j,\; e_0 + d_{0j}\big)")
    st.latex(r"LB_{j,k} = \max\Big(e_j,\; \min_{u \neq j,\; LB_{u,k-1} \le l_u} LB_{u,k-1} + d_{uj}\Big)")
    st.latex(r"\text{penalise } x_{j,k} \text{ if } LB_{j,k} > l_j, \qquad "
             r"\text{penalise } x_{u,k}\,x_{j,k+1} \text{ if } \max\big(e_j,\, LB_{u,k} + d_{uj}\big) > l_j")
    section = ui.related_work()["4. Rezk & Gora's time-window penalty: exact formulas"]
    ours = section[section.index("### Our E1 penalty"):section.index("### Differences that matter")]
    differences = section[section.index("### Differences that matter"):section.index("**Example of the unsound")]
    st.markdown(ours.replace("### ", "#### "))
    st.markdown(differences.replace("### ", "#### "))

with cost:
    st.markdown("Qubits and two-qubit (ZZ) terms per QAOA layer, mean over the 20 evaluation instances. "
                "Every nonzero off-diagonal QUBO entry is one ZZ rotation (two CNOTs).")
    counts = ui.costs()
    st.dataframe({"n": [n for n in SIZES for _ in ENCODINGS], "encoding": [e for _ in SIZES for e in ENCODINGS],
                  "qubits": [f"{counts[n, e][0]:.0f}" for n in SIZES for e in ENCODINGS],
                  "ZZ terms, X mixer": [f"{counts[n, e][1]:,.0f}" for n in SIZES for e in ENCODINGS],
                  "ZZ terms, XY mixer": [f"{counts[n, e][2]:,.0f}" if e != "E2" else "–"
                                         for n in SIZES for e in ENCODINGS]},
                 hide_index=True, use_container_width=True)

with code:
    st.markdown("The functions behind each step, read from the source files when the page loads.")
    for fn, what in SNIPPETS:
        name = f"{fn.__module__}.{fn.__qualname__}"
        with st.expander(f"`{name}` — {what}"):
            st.code(inspect.getsource(fn), language="python")
