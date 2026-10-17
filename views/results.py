"""
Results: the saved experiments as interactive charts and tables.  Nothing is recomputed;
everything is read from results/ (the Reproduce page lists the commands that wrote it).
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import grid
import ui
from experiments import SIZES, WIDTHS

MEASURES = {"p_opt": "P(opt)", "p_on_time": "P(on time)", "p_valid": "P(valid route)",
            "gap_closed": "energy gap closed"}
SEQ = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"]   # one hue, brighter = more


def download(path):
    st.download_button(f"Download {path.split('/')[-1]}", open(path, "rb").read(), file_name=path.split("/")[-1])


ui.header("011", "Saved experiments", "Results",
          "Read straight from <code>results/</code>. Bands and ± values are 95% t-intervals over instances; "
          "differences between encodings are paired per instance.", "results")
tab_qaoa, tab_sa, tab_tight, tab_pilot = st.tabs(["QAOA grid", "Annealing", "Window tightness", "Pilot runs"])

# ── QAOA grid ─────────────────────────────────────────────────────────────────────
with tab_qaoa:
    rows = ui.grid_rows()
    best = grid.best_weights(rows)
    c1, c2 = st.columns([1, 2])
    mixer, n = c1.selectbox("Mixer and size", grid.CASES, format_func=lambda c: f"{c[0].upper()} mixer, n = {c[1]}")
    metric = c2.radio("Measure", list(MEASURES), format_func=MEASURES.get, horizontal=True)
    ev = [r for r in rows if r["stage"] == "eval" and (r["mixer"], r["n"]) == (mixer, n)
          and (r["p1"], r["p2"]) == best[mixer, n, r["enc"]]]
    at = {(r["enc"], r["p"], r["seed"]): r for r in ev}
    seeds = sorted({r["seed"] for r in ev})

    def col(enc, p, field=metric):
        return np.array([at[enc, p, s][field] for s in seeds])

    depths = range(1, grid.DEPTH + 1)
    fig = go.Figure()
    for enc in (e for e in ui.ENC if e in grid.ENCODINGS):
        ui.band(fig, depths, [col(enc, p) for p in depths], enc,
                clip=(0, 1) if metric != "gap_closed" else (-np.inf, np.inf))
    if metric == "p_opt":
        fig.add_hline(y=np.mean([at["E0", 1, s]["uniform_p_opt"] for s in seeds]), line=dict(color="#475569", width=1),
                      annotation_text="uniform state", annotation_font_color="#94a3b8")
    st.plotly_chart(ui.style(fig, height=380, xaxis_title="depth p", yaxis_title=MEASURES[metric]),
                    use_container_width=True)
    st.caption(f"{len(seeds)} fresh instances. Weights (p1, p2) chosen on the tuning instances: "
               + ", ".join(f"{e} {best[mixer, n, e]}" for e in grid.ENCODINGS) + ".")

    st.markdown(f"**{MEASURES[metric]} minus E0's, paired per instance**")
    st.dataframe({"": ["E1", "E1+C"],
                  **{f"p = {p}": [grid.mean_ci(col(e, p) - col("E0", p)) for e in ("E1", "E1+C")] for p in grid.SHOW}},
                 hide_index=True, use_container_width=True)
    late = np.array([at["E0", 1, s]["binding"] for s in seeds])
    gain = {e: col(e, grid.DEPTH, "p_opt") - col("E0", grid.DEPTH, "p_opt") for e in ("E1", "E1+C")}
    st.markdown(f"**P(opt) minus E0's at p = {grid.DEPTH}, by whether the instance's TSP optimum is late**")
    st.dataframe({"": ["E1", "E1+C"],
                  f"late ({late.sum()} instances)": [grid.mean_ci(gain[e][late]) for e in gain],
                  f"on time ({(~late).sum()} instances)": [grid.mean_ci(gain[e][~late]) for e in gain],
                  "late instances' share of the gain": [f"{gain[e][late].sum() / gain[e].sum():.0%}" for e in gain]},
                 hide_index=True, use_container_width=True)
    st.caption("Ground state is the optimum in: "
               + ", ".join(f"{e} {col(e, 1, 'ground_opt').mean():.0%}" for e in grid.ENCODINGS)
               + " of instances. That is where P(opt) heads as p → ∞, not a bound at finite p.")

    st.markdown("**Weight tuning** (mean P(opt) over the tuning instances and depths 1–5; the box is the pick)")
    tune = [r for r in rows if r["stage"] == "tune" and (r["mixer"], r["n"]) == (mixer, n)]
    for column, enc in zip(st.columns(2), ("E1", "E1+C")):
        z = [[np.mean([r["p_opt"] for r in tune if (r["enc"], r["p1"], r["p2"]) == (enc, p1, p2)])
              for p2 in grid.P2] for p1 in grid.P1]
        x, y = [f"{v:g}" for v in grid.P2], [f"{v:g}" for v in grid.P1]
        p1, p2 = best[mixer, n, enc]
        fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=SEQ, colorbar=dict(title="P(opt)"),
                                   hovertemplate="p1 = %{y}, p2 = %{x}: %{z:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[f"{p2:g}"], y=[f"{p1:g}"], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(symbol="square-open", size=30, color="#e2e8f0", line=dict(width=2))))
        column.plotly_chart(ui.style(fig, height=300, title=enc, xaxis_title="p2 (window weight C/p2)",
                                     yaxis_title="p1 (validity weight C·p1)"), use_container_width=True)
    with st.expander("X mixer: weights beyond the grid's edge (grid.py edge)"):
        st.code(open("results/qaoa_grid_edge.txt").read(), language=None)
    download("results/qaoa_grid.csv")

# ── Annealing ─────────────────────────────────────────────────────────────────────
with tab_sa:
    sa = ui.csv_rows("results/sa.csv")
    ev = [r for r in sa if r["phase"] == "eval"]
    fig = go.Figure()
    for enc, (colour, _, label) in ui.ENC.items():
        vals = [[r["p_opt"] for r in ev if (r["encoding"], r["n"]) == (enc, n)] for n in SIZES]
        fig.add_trace(go.Bar(x=[f"n = {n}" for n in SIZES], y=[np.mean(v) for v in vals], name=label,
                             marker=dict(color=colour, line=dict(color=ui.SURFACE, width=2)),
                             error_y=dict(type="data", array=[grid.ci(v) for v in vals], color="#94a3b8", thickness=1),
                             hovertemplate=f"{label}<br>%{{x}}: P(opt) %{{y:.3f}}<extra></extra>"))
    st.plotly_chart(ui.style(fig, height=340, barmode="group", yaxis_title="P(opt)", yaxis_range=[0, 1],
                             title="Simulated annealing, 20 fresh instances per size, 100 reads × 10,000 sweeps"),
                    use_container_width=True)
    table = {k: [] for k in ("n", "encoding", "qubits", "weights (p1, p2)", "P(valid)", "P(on time)", "P(opt)", "solved")}
    for n in SIZES:
        for enc in ui.ENC:
            r_ = [r for r in ev if (r["encoding"], r["n"]) == (enc, n)]
            for k, v in (("n", n), ("encoding", enc), ("qubits", f"{np.mean([r['qubits'] for r in r_]):.0f}"),
                         ("weights (p1, p2)", f"({r_[0]['p1']:g}, {r_[0]['p2']:g})"),
                         ("P(valid)", f"{np.mean([r['p_valid'] for r in r_]):.3f}"),
                         ("P(on time)", f"{np.mean([r['p_on_time'] for r in r_]):.3f}"),
                         ("P(opt)", grid.mean_ci([r["p_opt"] for r in r_]).lstrip("+")),
                         ("solved", f"{sum(r['p_opt'] > 0 for r in r_)}/{len(r_)}")):
                table[k].append(v)
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.markdown("**Does a bigger annealing budget change the picture?** (same 5 instances, tuned weights)")
    oracle_seeds = sorted({r["seed"] for r in sa if r["phase"] == "oracle"})
    fig = make_subplots(1, 3, shared_yaxes=True, subplot_titles=[f"n = {n}" for n in SIZES])
    for i, n in enumerate(SIZES, 1):
        for enc in ("E2", "E1"):
            pts = {sw: [r["p_opt"] for r in sa if r["phase"] in ("budget", "eval") and r["sweeps"] == sw
                        and (r["encoding"], r["n"]) == (enc, n) and r["seed"] in oracle_seeds]
                   for sw in (2000, 10000, 50000)}
            ui.band(fig, [f"{sw:,}" for sw in pts], list(pts.values()), enc, row=1, col=i)
    fig.update_xaxes(type="category")
    st.plotly_chart(ui.style(fig, height=300, yaxis_title="P(opt)", xaxis2_title="sweeps per read"),
                    use_container_width=True)
    oracle = {n: np.mean([max(r["p_opt"] for r in sa if r["phase"] == "oracle" and r["n"] == n and r["seed"] == s)
                          for s in oracle_seeds]) for n in SIZES}
    st.caption("E2's best case (weights picked per instance knowing the optimum, 2,000 sweeps): mean P(opt) "
               + ", ".join(f"{oracle[n]:.3f} at n = {n}" for n in SIZES) + ".")
    with st.expander("Full summary (results/sa_summary.txt)"):
        st.code(open("results/sa_summary.txt").read(), language=None)
    download("results/sa.csv")

# ── Window tightness ──────────────────────────────────────────────────────────────
with tab_tight:
    rows_t = [r for r in ui.csv_rows("results/sa_tightness.csv") if r["phase"] == "eval"]
    fig = make_subplots(1, 3, shared_yaxes=True, subplot_titles=[f"n = {n}" for n in SIZES])
    for i, n in enumerate(SIZES, 1):
        for enc in ui.ENC:
            ui.band(fig, [str(w) for w in WIDTHS],
                    [[r["p_opt"] for r in rows_t if (r["n"], r["width"], r["encoding"]) == (n, w, enc)] for w in WIDTHS],
                    enc, row=1, col=i)
    fig.update_xaxes(type="category")                # ordinal, as in the paper figure
    st.plotly_chart(ui.style(fig, height=340, yaxis_title="P(opt)", xaxis2_title="window half-width (planted windows)"),
                    use_container_width=True)
    st.caption("Annealing on instances with a planted feasible tour and windows of the given half-width around it; "
               "weights re-tuned at every width. At width 0 only the planted route is feasible.")
    st.dataframe({"n": [n for n in SIZES for _ in WIDTHS], "half-width": [w for _ in SIZES for w in WIDTHS],
                  "feasible routes": [f"{np.mean([r['feasible_frac'] for r in rows_t if (r['n'], r['width'], r['encoding']) == (n, w, 'E0')]):.1%}"
                                      for n in SIZES for w in WIDTHS],
                  "TSP optimum late": [f"{np.mean([r['binding'] for r in rows_t if (r['n'], r['width'], r['encoding']) == (n, w, 'E0')]):.0%}"
                                       for n in SIZES for w in WIDTHS]},
                 hide_index=True, use_container_width=True)
    download("results/sa_tightness.csv")

# ── Pilot ─────────────────────────────────────────────────────────────────────────
with tab_pilot:
    st.markdown("The QAOA pilot that fixed the final grid's design (pilot.py). Small samples; the final "
                "numbers are on the QAOA grid tab.")
    for stage, what in (("a", "Optimiser: Adam from random angles vs L-BFGS vs the INTERP depth ladder"),
                        ("b", "Weight grid on tuning instances"),
                        ("c", "Those weights on fresh instances"),
                        ("d", "How far P(opt) climbs with depth, up to p = 12")):
        with st.expander(f"Stage {stage.upper()}: {what}"):
            st.code(open(f"results/pilot_{stage}.txt").read(), language=None)
