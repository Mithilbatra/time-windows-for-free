"""
Paper: the claim, headline numbers computed from results/, the figures and the related work.
"""
import numpy as np
import streamlit as st

import grid
import ui

ABSTRACT = """
QUBO encodings of the travelling salesman problem with time windows (TSPTW) usually enforce
the windows with slack qubits, which multiplies the qubit count and the circuit size. We bound
every customer's earliest possible arrival at every tour position with a dynamic programme and
turn the bounds into penalties that sit on couplings the distance term already has: no extra
qubits, no extra two-qubit gates, and provably no penalty on any feasible route (**E1**; **E1+C**
adds a look-ahead at about 9% more two-qubit terms). In exact simulation of QAOA with a
constraint-preserving XY mixer, E1+C beats the window-free encoding at 3, 4 and 5 customers,
with the gain growing with circuit depth. With simulated annealing, E1 and E1+C beat both the
window-free encoding and a tuned slack-qubit encoding (Salehi et al., 2022) at every size and
window width, with 6–7× fewer qubits.
"""

CONTRIBUTIONS = """
1. **A sound time-window penalty (E1)** from cumulative earliest-arrival bounds: zero extra
   qubits, zero extra two-qubit gates, and no feasible route is ever penalised. A look-ahead
   variant (E1+C) trades about 9% more two-qubit terms for more pruning.
2. **A fair comparison** with the slack-qubit encoding of Salehi et al. (E2) under simulated
   annealing, with every encoding's penalty weights tuned the same way on separate instances.
3. **Exact QAOA simulation** with the XY mixer on the states it can reach (up to 25 qubits), to
   depth 12, with paired statistics over 50 fresh instances per size.
4. **A window-tightness study** on instances with planted windows, from zero slack to loose.
"""

CAPTIONS = {
    "qaoa_depth": "QAOA P(opt) against depth: mean over 50 fresh instances (20 for the X mixer at "
                  "n = 4) with 95% t-interval bands. Each panel has its own y-scale.",
    "sa_tightness": "Simulated annealing P(opt) against the half-width of planted time windows (ordinal "
                    "axis: 0, 6, 12, 24, 48), weights re-tuned at every width. At width 0 only the "
                    "planted route is feasible.",
    "resources": "Qubits and two-qubit terms per QAOA layer with the X mixer, mean over 20 instances. "
                 "E1 needs exactly E0's terms.",
}


def tile(column, label, value, sub):
    column.markdown(f'<div class="metric-card"><h4>{label}</h4><div class="val">{value}</div>'
                    f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


st.markdown("# 📄 Time windows for free")
st.markdown("### A sound, ancilla-free time-window encoding for QAOA on the TSP with time windows")
st.caption("Working title. Data, code and figures are done; the draft is in progress. Target: arXiv by "
           "31 Dec 2026, then ICCS 2027 (submission around February; dates to confirm).")

overview, figures, related = st.tabs(["Overview", "Figures", "Related work"])

with overview:
    rows = ui.grid_rows()
    best = grid.best_weights(rows)
    at = {(r["enc"], r["p"], r["seed"]): r["p_opt"] for r in rows if r["stage"] == "eval"
          and (r["mixer"], r["n"]) == ("xy", 4) and (r["p1"], r["p2"]) == best["xy", 4, r["enc"]]}
    seeds = sorted({k[2] for k in at})
    gain = np.array([at["E1+C", grid.DEPTH, s] - at["E0", grid.DEPTH, s] for s in seeds])
    sa = [r for r in ui.csv_rows("results/sa.csv") if r["phase"] == "eval" and r["n"] == 5]
    sa_opt = {e: np.mean([r["p_opt"] for r in sa if r["encoding"] == e]) for e in ("E1", "E2")}
    cost = ui.costs()
    t1, t2, t3, t4 = st.columns(4)
    tile(t1, "QAOA gain over E0", f"{gain.mean():+.3f}",
         f"E1+C, P(opt), XY mixer, n = 4, p = {grid.DEPTH}; 95% CI ±{grid.ci(gain):.3f}")
    tile(t2, "Annealing, n = 5", f"{sa_opt['E1'] / sa_opt['E2']:.1f}×",
         f"P(opt) {sa_opt['E1']:.2f} for E1 vs {sa_opt['E2']:.2f} for the tuned slack-qubit E2")
    tile(t3, "Qubits, n = 5", f"{cost[5, 'E2'][0] / cost[5, 'E1'][0]:.1f}× fewer",
         f"E1 {cost[5, 'E1'][0]:.0f} vs E2 {cost[5, 'E2'][0]:.0f}")
    tile(t4, "Two-qubit terms, n = 5", f"{cost[5, 'E2'][1] / cost[5, 'E1'][1]:.0f}× fewer",
         f"per QAOA layer, X mixer: E1 {cost[5, 'E1'][1]:,.0f} vs E2 {cost[5, 'E2'][1]:,.0f}")
    st.markdown("#### Abstract (draft)")
    st.markdown(ABSTRACT)
    st.markdown("#### Contributions")
    st.markdown(CONTRIBUTIONS)

with figures:
    xy = ", ".join(f"{cost[n, 'E1'][2]:.0f}" for n in (3, 4, 5))
    for name, caption in CAPTIONS.items():
        st.image(f"figures/{name}.png", use_column_width=True)
        st.caption(caption + (f" With the XY mixer our encodings need fewer ({xy} at n = 3, 4, 5)."
                              if name == "resources" else ""))
        st.download_button(f"Download {name}.pdf", open(f"figures/{name}.pdf", "rb").read(),
                           file_name=f"{name}.pdf", key=name)
        st.divider()

with related:
    st.markdown("From `related_work.md`: the closest papers, what to cite, and where this work fits.")
    for heading, body in list(ui.related_work().items())[:3]:
        with st.expander(heading, expanded=heading.startswith("3.")):
            st.markdown(body)
