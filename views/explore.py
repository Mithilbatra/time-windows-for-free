"""
Explore an instance: pick any instance, see its windows and exact optimum, which routes each
encoding penalises, and run QAOA on it live with the exact simulator (FastQAOA).
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import grid
import ui
from experiments import builder
from qaoa import FastQAOA
from qubo import all_routes, make_instance, route_penalty, solve_exact, tw_terms

GOOD, LATE = "#0ca30c", "#d03b3b"                    # status colours: on time / late (always labelled)


def route_name(order):
    return "0 → " + " → ".join(map(str, order)) + " → 0"


@st.cache_data(show_spinner=False)
def live(n, seed, width, enc, mixer, p1, p2, pmax, restarts):
    """QAOA by the INTERP ladder on one instance: route probabilities at every depth up to pmax."""
    fq = FastQAOA(builder(enc, make_instance(n, seed, width), p1, p2), mixer=mixer, seed=seed)
    ladder = fq.ladder(pmax, starts=5, restarts=restarts)
    b = fq.bits.reshape(-1, n, n)[fq.valid]
    orders = [tuple(o) for o in np.argsort(np.argmax(b, axis=2), axis=1) + 1]
    probs = np.abs(fq.state(ladder[pmax][0])) ** 2
    return dict(curve=[fq.route_masses(ladder[p][0]) for p in range(1, pmax + 1)],
                uniform=fq.route_masses(np.zeros(0)), ground_opt=fq.ground_optimal(),
                routes=dict(zip(orders, probs[fq.valid])))


st.markdown("# 🔍 Explore an instance")

with st.sidebar:
    st.markdown('<div class="section-head">Instance</div>', unsafe_allow_html=True)
    n = st.radio("Customers", (3, 4, 5), index=1, horizontal=True, key="ex_n")
    planted = st.radio("Windows", ("Salehi et al. protocol", "planted tour"), key="ex_kind") == "planted tour"
    seed = int(st.number_input("Seed", value=80000, step=1, key="ex_seed",
                               help="80000–80049 are the QAOA grid's evaluation instances"))
    width = st.slider("Half-width of the planted windows", 0, 48, 12, key="ex_width") if planted else None
    st.markdown('<div class="section-head">QAOA</div>', unsafe_allow_html=True)
    mixer = st.radio("Mixer", ("xy", "x") if n <= 4 else ("xy",), format_func=str.upper, horizontal=True,
                     key="ex_mixer")
    encodings = st.multiselect("Encodings", list(grid.ENCODINGS), default=list(grid.ENCODINGS), key="ex_enc")
    pmax = st.slider("Depth p", 1, 8 if (mixer, n) != ("x", 4) else 5, 3, key="ex_p")
    restarts = st.radio("Random restarts per depth", (0, 3), horizontal=True, key="ex_restarts",
                        help="The saved grid used 3; 0 is faster")
    tuned = grid.best_weights(ui.grid_rows())
    st.caption("Weights (p1, p2) are the grid's tuned picks for this mixer and size.")
    run = st.button("⚡ Run QAOA", use_container_width=True, key="ex_run_button")

inst = make_instance(n, seed, width)
perms, feasible, cost = all_routes(inst)
optimum, opt_cost = solve_exact(inst)
optimal = feasible & np.isclose(cost, opt_cost)
late_tsp = not feasible[np.isclose(cost, cost.min())].any()

# ── the instance ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">The instance</div>', unsafe_allow_html=True)
left, right = st.columns([3, 2])
xy = ui.layout(inst.d)
fig = go.Figure()
loop = optimum + [0]
fig.add_trace(go.Scatter(x=xy[loop, 0], y=xy[loop, 1], mode="lines", line=dict(color="#38bdf8", width=2),
                         name="optimal route", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=["depot"] + list(range(1, n + 1)),
                         textposition="top center", marker=dict(size=14, color=[ui.city_colour(i) for i in range(n + 1)],
                                                                line=dict(color=ui.SURFACE, width=2)),
                         hovertext=[f"window [{e:g}, {l:g}]" for e, l in inst.time_windows], hoverinfo="text",
                         showlegend=False))
left.plotly_chart(ui.style(fig, height=330, xaxis_visible=False, yaxis_visible=False,
                           title="Positions approximate the travel times"), use_container_width=True)
arrive = inst.arrivals(optimum)
right.dataframe({"stop": ["depot"] + [str(c) for c in optimum[1:]] + ["depot"],
                 "window": [f"[{inst.e[c]:g}, {inst.l[c]:g}]" for c in optimum + [0]],
                 "service starts": [f"{a:g}" for a in arrive]}, hide_index=True, use_container_width=True)
right.markdown(f"**Optimum** {route_name(optimum[1:])}, cost {opt_cost:g}  \n"
               f"{feasible.sum()} of {len(perms)} routes are on time. "
               + ("The plain TSP optimum is **late** here." if late_tsp else "The plain TSP optimum is on time."))

# ── which routes each encoding penalises ──────────────────────────────────────────
st.markdown('<div class="section-head">Which routes each encoding penalises</div>', unsafe_allow_html=True)
pen = {e: route_penalty(perms, *tw_terms(inst, e)) for e in ("E1", "E1+C")}
energy = {}
for e in grid.ENCODINGS:
    b = builder(e, inst, *tuned[mixer, n, e])
    energy[e] = np.array([b.energy(b.encode([0, *p])) for p in perms])
cols = st.columns(3)
for column, e in zip(cols, grid.ENCODINGS):
    caught = (pen[e][~feasible] > 0).mean() if e in pen and (~feasible).any() else 0.0
    ground = np.isclose(energy[e], energy[e].min())
    column.markdown(
        f'<div class="metric-card"><h4>{e}</h4><div class="val">{caught:.0%}</div>'
        f'<div class="sub">of late routes penalised; on-time routes penalised: '
        f'{int((pen[e][feasible] > 0).sum()) if e in pen else 0}<br>lowest-energy routes that are the optimum: '
        f'{optimal[ground].sum()} of {ground.sum()}</div></div>', unsafe_allow_html=True)
order = np.argsort(cost, kind="stable")
st.dataframe({"route": [route_name(perms[i]) for i in order], "cost": [f"{cost[i]:g}" for i in order],
              "on time": ["✓ on time" if feasible[i] else "✗ late" for i in order],
              "E1 penalty": [f"{pen['E1'][i]:g}" for i in order], "E1+C penalty": [f"{pen['E1+C'][i]:g}" for i in order]},
             hide_index=True, use_container_width=True, height=240)
st.caption("Penalty = number of window terms a route triggers. E1 never penalises an on-time route (the "
           "soundness lemma on the Method page); a late route it misses competes on cost alone.")

# ── live QAOA ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">QAOA, run live</div>', unsafe_allow_html=True)
settings = (n, seed, width, tuple(encodings), mixer, pmax, restarts)
if run:
    st.session_state.ex_ran = settings
if st.session_state.get("ex_ran") != settings or not encodings:
    st.info("Choose the settings in the sidebar and press **Run QAOA**. Up to 5 customers with the XY mixer "
            "takes a few seconds per encoding.")
else:
    with st.spinner("Running the depth ladder…"):
        res = {e: live(n, seed, width, e, mixer, *tuned[mixer, n, e], pmax, restarts) for e in encodings}
    saved = {(r["enc"], r["p"]): r["p_opt"] for r in ui.grid_rows() if r["stage"] == "eval" and not planted
             and (r["mixer"], r["n"], r["seed"]) == (mixer, n, seed) and (r["p1"], r["p2"]) == tuned[mixer, n, r["enc"]]}
    fig = go.Figure()
    for e in (k for k in ui.ENC if k in res):
        colour, symbol, label = ui.ENC[e]
        fig.add_trace(go.Scatter(x=list(range(1, pmax + 1)), y=[c[2] for c in res[e]["curve"]], name=label,
                                 mode="lines+markers", line=dict(color=colour, width=2),
                                 marker=dict(symbol=symbol, size=8, color=colour, line=dict(color=ui.SURFACE, width=2)),
                                 hovertemplate=f"{label}<br>p = %{{x}}: P(opt) %{{y:.3f}}<extra></extra>"))
        if saved:
            fig.add_trace(go.Scatter(x=list(range(1, pmax + 1)), y=[saved.get((e, p)) for p in range(1, pmax + 1)],
                                     mode="markers", marker=dict(symbol=symbol + "-open", size=11, color=colour),
                                     name=f"{e}, saved grid run", hovertemplate="saved: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=res[encodings[0]]["uniform"][2], line=dict(color="#475569", width=1),
                  annotation_text="uniform state", annotation_font_color="#94a3b8")
    st.plotly_chart(ui.style(fig, height=340, xaxis_title="depth p", yaxis_title="P(opt)", xaxis_dtick=1),
                    use_container_width=True)
    st.dataframe({"encoding": list(res), "P(opt)": [f"{res[e]['curve'][-1][2]:.3f}" for e in res],
                  "P(on time)": [f"{res[e]['curve'][-1][1]:.3f}" for e in res],
                  "P(valid route)": [f"{res[e]['curve'][-1][0]:.3f}" for e in res],
                  "p → ∞ (ground state is the optimum)": [f"{res[e]['ground_opt']:.0%}" for e in res]},
                 hide_index=True, use_container_width=True)
    if saved:
        st.caption("Open markers: the saved grid's run on this instance (3 restarts per depth). With 3 restarts "
                   "here too, the live curve should match it.")

    shown = st.selectbox("Route probabilities at the final depth, for", list(res), key="ex_shown")
    probs = res[shown]["routes"]
    fig = go.Figure()
    for ok, colour, name in ((True, GOOD, "on time"), (False, LATE, "late")):
        idx = [i for i in order if feasible[i] == ok]
        fig.add_trace(go.Bar(x=[route_name(perms[i]) for i in idx], y=[probs.get(tuple(perms[i]), 0.0) for i in idx],
                             marker=dict(color=colour), name=name,
                             hovertemplate="%{x}<br>probability %{y:.4f}<extra>" + name + "</extra>"))
    fig.add_annotation(x=route_name(optimum[1:]), y=probs.get(tuple(optimum[1:]), 0.0), text="optimum",
                       showarrow=True, arrowcolor="#e2e8f0", font=dict(color="#e2e8f0"))
    st.plotly_chart(ui.style(fig, height=320, xaxis_title="routes, cheapest first", yaxis_title="probability",
                             xaxis_showticklabels=n <= 4, xaxis_categoryorder="array",
                             xaxis_categoryarray=[route_name(perms[i]) for i in order]),
                    use_container_width=True)
    st.caption(f"The rest of the probability, {1 - res[shown]['curve'][-1][0]:.3f}, sits on states that are not routes.")
