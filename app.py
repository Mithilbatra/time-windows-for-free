"""
TSPTW-QAOA Interactive Dashboard
==================================
Streamlit app for interactive quantum route optimisation.
Run:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

from qubo import TSPTWInstance, QUBOBuilder, make_instance
from qaoa import TSPTWQAOACircuit, qubo_to_hamiltonian

# ──────────────────────────────────────────────────────────────────────────────
# Page config & custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSPTW · QAOA",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .stApp { background: #0b0e1a; color: #e2e8f0; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%);
    border-right: 1px solid #1e2d45;
  }

  /* Cards */
  .metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2333 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    box-shadow: 0 4px 24px rgba(0,120,255,0.08);
  }
  .metric-card h4 { color: #64b5f6; font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; margin: 0 0 6px; }
  .metric-card .val { font-size: 1.6rem; font-weight: 700; color: #e2f0ff; font-family: 'Space Mono', monospace; }
  .metric-card .sub { font-size: 0.75rem; color: #8fa3c0; margin-top: 2px; }

  /* Section headers */
  .section-head {
    font-family: 'Space Mono', monospace;
    color: #38bdf8;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 24px 0 16px;
  }

  /* Badges */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.06em;
  }
  .badge-blue  { background:#1e3a5f; color:#7dd3fc; border:1px solid #2d5a8a; }
  .badge-green { background:#14532d; color:#86efac; border:1px solid #16a34a; }
  .badge-red   { background:#450a0a; color:#fca5a5; border:1px solid #dc2626; }
  .badge-yellow{ background:#451a03; color:#fcd34d; border:1px solid #d97706; }

  /* Route table */
  .route-table { width:100%; border-collapse:collapse; font-family:'Space Mono',monospace; font-size:0.8rem; }
  .route-table th { background:#0f1e35; color:#7dd3fc; padding:8px 12px; text-align:left; border-bottom:1px solid #1e3a5f; }
  .route-table td { padding:8px 12px; border-bottom:1px solid #1a2535; color:#cbd5e1; }
  .route-table tr:hover td { background:#111f35; }

  /* Qubit grid */
  .qubit-row { display:flex; gap:4px; flex-wrap:wrap; margin:8px 0; }
  .qubit-bit { width:26px; height:26px; border-radius:5px; display:flex; align-items:center; justify-content:center;
               font-family:'Space Mono',monospace; font-size:0.7rem; font-weight:700; }
  .qubit-1 { background:#1d4ed8; color:#bfdbfe; border:1px solid #3b82f6; }
  .qubit-0 { background:#1a2535; color:#475569; border:1px solid #243447; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background:#0f1629; border-bottom:1px solid #1e3a5f; gap:0; }
  .stTabs [data-baseweb="tab"] { color:#64748b; font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:0.08em; }
  .stTabs [aria-selected="true"] { color:#38bdf8 !important; border-bottom:2px solid #38bdf8 !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(90deg, #1d4ed8 0%, #0ea5e9 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    padding: 10px 24px;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity:0.85; }

  /* Sliders and inputs */
  .stSlider label, .stNumberInput label, .stSelectbox label { color:#94a3b8 !important; font-size:0.75rem !important; }

  h1 { font-family:'Space Mono',monospace; color:#e2f0ff; font-size:1.4rem !important; }
  h2 { font-family:'Space Mono',monospace; color:#bfdbfe; font-size:1rem !important; }
  h3 { color:#94a3b8; font-size:0.85rem !important; }
  .stMarkdown p { color:#94a3b8; font-size:0.85rem; line-height:1.6; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────
def init_state():
    if "instance" not in st.session_state:
        st.session_state.instance = None
    if "builder" not in st.session_state:
        st.session_state.builder = None
    if "qaoa" not in st.session_state:
        st.session_state.qaoa = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "bf_result" not in st.session_state:
        st.session_state.bf_result = None
    if "cities" not in st.session_state:
        st.session_state.cities = []     # list of dicts {x,y,e,l,name}
    if "optimizing" not in st.session_state:
        st.session_state.optimizing = False

init_state()


# ──────────────────────────────────────────────────────────────────────────────
# Helper – colour palette for cities
# ──────────────────────────────────────────────────────────────────────────────
CITY_COLOURS = ["#f87171","#fb923c","#facc15","#4ade80","#38bdf8","#a78bfa","#f472b6","#67e8f9"]

def city_colour(i):
    return CITY_COLOURS[i % len(CITY_COLOURS)]


def layout(d):
    """2-D positions whose distances roughly match d (classical MDS), for display only."""
    J = np.eye(len(d)) - 1 / len(d)
    w, V = np.linalg.eigh(-0.5 * J @ d**2 @ J)
    return V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0))


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – problem configuration
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚛️ TSPTW · QAOA")
    st.markdown("<p>Quantum Route Optimiser</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-head">Problem Setup</div>', unsafe_allow_html=True)

    mode = st.selectbox("Input mode", ["Random instance", "Manual cities", "Preset examples"])

    if mode == "Random instance":
        n_cities = st.slider("Number of customers (+ depot)", 2, 4, 3)
        seed     = st.number_input("Random seed", value=42, step=1)
        st.caption("Salehi et al. protocol: random travel times. Map positions are an approximate 2-D layout.")
        if st.button("🎲 Generate Instance"):
            inst = make_instance(n_cities, seed=int(seed))
            inst.coords = layout(inst.d)      # display only; travel times stay inst.d
            st.session_state.cities = [
                {"name": ("Depot" if i == 0 else f"City {i}"),
                 "x": float(inst.coords[i, 0]),
                 "y": float(inst.coords[i, 1]),
                 "e": float(inst.time_windows[i, 0]),
                 "l": float(inst.time_windows[i, 1])}
                for i in range(inst.n + 1)
            ]
            st.session_state.instance = inst
            st.session_state.builder  = None
            st.session_state.qaoa     = None
            st.session_state.result   = None

    elif mode == "Preset examples":
        preset = st.selectbox("Preset", ["3-city triangle", "4-city square"])
        if st.button("📦 Load Preset"):
            # depot window = (departure, return deadline)
            if preset == "3-city triangle":
                coords = [(0,0),(5,0),(2.5,4)]
                windows = [(0,20),(2,8),(4,12)]
            else:
                coords = [(0,0),(4,0),(4,4),(0,4)]
                windows = [(0,20),(1,6),(3,9),(5,12)]
            inst = TSPTWInstance(coords, windows)
            st.session_state.cities = [
                {"name": ("Depot" if i == 0 else f"City {i}"),
                 "x": float(inst.coords[i,0]),
                 "y": float(inst.coords[i,1]),
                 "e": float(inst.time_windows[i,0]),
                 "l": float(inst.time_windows[i,1])}
                for i in range(inst.n + 1)
            ]
            st.session_state.instance = inst
            st.session_state.builder  = None
            st.session_state.qaoa     = None
            st.session_state.result   = None

    elif mode == "Manual cities":
        n_manual = st.slider("Number of customers (+ depot)", 2, 4, 3)
        new_cities = []
        for i in range(n_manual + 1):
            lbl = "Depot" if i == 0 else f"City {i}"
            with st.expander(f"📍 {lbl}", expanded=(i == 0)):
                cx = st.number_input(f"x ({lbl})", value=float(i*3), key=f"cx{i}")
                cy = st.number_input(f"y ({lbl})", value=float(i*2), key=f"cy{i}")
                ce = st.number_input(f"Earliest ({lbl})", value=0.0, key=f"ce{i}")
                cl = st.number_input(f"Latest ({lbl})", value=30.0, key=f"cl{i}")
                new_cities.append({"name":lbl,"x":cx,"y":cy,"e":ce,"l":cl})
        if st.button("✅ Use These Cities"):
            coords  = [(c["x"], c["y"]) for c in new_cities]
            windows = [(c["e"], c["l"])  for c in new_cities]
            inst = TSPTWInstance(coords, windows)
            st.session_state.cities   = new_cities
            st.session_state.instance = inst
            st.session_state.builder  = None
            st.session_state.qaoa     = None
            st.session_state.result   = None

    st.divider()
    st.markdown('<div class="section-head">QUBO Weights</div>', unsafe_allow_html=True)
    A_mult = st.slider("A  constraints (× safe bound)", 0.1, 3.0, 1.0, 0.1)
    B_dist = st.slider("B  travel distance",            0.1, 5.0, 1.0, 0.1)
    P_mult = st.slider("P  time windows (× safe bound)", 0.1, 3.0, 1.0, 0.1)

    st.divider()
    st.markdown('<div class="section-head">QAOA Parameters</div>', unsafe_allow_html=True)
    mixer    = st.selectbox("Mixer", ["xy", "x"], format_func=lambda m: {
        "xy": "XY ring per customer (keeps each customer once)", "x": "X (transverse field)"}[m])
    p_layers = st.slider("QAOA depth p", 1, 4, 2)
    steps    = st.slider("Optimisation steps", 20, 200, 80, 10)
    starts   = st.slider("Random starts", 1, 5, 1)
    stepsize = st.select_slider("Step size", [0.01,0.02,0.05,0.1,0.15,0.2], value=0.05)
    n_shots  = st.select_slider("Measurement shots", [256,512,1024,2048], value=1024)

    st.divider()
    run_btn = st.button("⚡ Run QAOA Optimisation", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# ⚛️  TSPTW · QAOA  Route Planner")
st.markdown(
    "**Travelling Salesman Problem with Time Windows** solved via "
    "**Quantum Approximate Optimisation Algorithm** — "
    "QUBO formulation · PennyLane circuit · interactive dashboard"
)

# ── Run optimisation ──────────────────────────────────────────────────────────
if run_btn:
    if st.session_state.instance is None:
        st.error("⚠️  Generate or load a problem instance first (sidebar).")
    else:
        inst    = st.session_state.instance
        with st.spinner("⚙️  Building QUBO & constructing Hamiltonian…"):
            builder = QUBOBuilder(inst, A=A_mult, B=B_dist, P=P_mult)
            qaoa = TSPTWQAOACircuit(builder, p=p_layers, mixer=mixer)

        progress_bar = st.progress(0, text="Running QAOA…")
        loss_placeholder = st.empty()
        losses = []
        total = steps * starts

        def cb(step, loss, _params):
            losses.append(loss)
            pct = int((step + 1) / total * 100)
            progress_bar.progress(pct, text=f"Step {step+1}/{total} · ⟨H_C⟩ = {loss:.4f}")

        result = qaoa.optimize(steps=steps, stepsize=stepsize, callback=cb, starts=starts)
        progress_bar.empty()

        with st.spinner("🔬 Sampling quantum state…"):
            bs, energy, route, counts = qaoa.sample_best(n_shots=n_shots)

        with st.spinner("🧮 Running classical brute-force for comparison…"):
            try:
                bf_bs, bf_e, bf_route = qaoa.brute_force()
            except Exception:
                bf_bs, bf_e, bf_route = None, None, []
        if not bf_route:
            st.warning("No route meets every time window for this instance.")

        st.session_state.builder   = builder
        st.session_state.qaoa      = qaoa
        st.session_state.result    = {
            "bitstring": bs, "energy": energy, "route": route,
            "counts": counts, "opt_result": result
        }
        st.session_state.bf_result = {"bs": bf_bs, "energy": bf_e, "route": bf_route}
        st.success("✅  QAOA optimisation complete!")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Route Map",
    "📊 QUBO Matrix",
    "⚛️ Circuit",
    "📈 Convergence",
    "🔬 State Analysis"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Route Map
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    inst   = st.session_state.instance
    result = st.session_state.result

    if inst is None:
        st.info("👈  Generate or load an instance from the sidebar to begin.")
    else:
        cities = st.session_state.cities
        n      = inst.n + 1          # nodes incl. depot
        coords = inst.coords
        tw     = inst.time_windows

        col_map, col_info = st.columns([3, 1])

        with col_map:
            fig = go.Figure()

            # Draw all possible edges (faded)
            for i in range(n):
                for j in range(i+1, n):
                    fig.add_trace(go.Scatter(
                        x=[coords[i,0], coords[j,0]],
                        y=[coords[i,1], coords[j,1]],
                        mode="lines",
                        line=dict(color="#1e3a5f", width=1, dash="dot"),
                        showlegend=False, hoverinfo="skip"
                    ))

            # Draw QAOA route (if available)
            route = result["route"] if result else None
            if route and -1 not in route:
                route_loop = route + [route[0]]
                rx = [coords[c,0] for c in route_loop]
                ry = [coords[c,1] for c in route_loop]
                # gradient path
                for seg in range(len(route)):
                    alpha = 0.4 + 0.6 * seg / max(len(route)-1, 1)
                    fig.add_trace(go.Scatter(
                        x=[rx[seg], rx[seg+1]], y=[ry[seg], ry[seg+1]],
                        mode="lines",
                        line=dict(color=f"rgba(56,189,248,{alpha:.2f})", width=3),
                        showlegend=(seg == 0),
                        name="QAOA Route",
                        hoverinfo="skip"
                    ))
                # Arrows
                for seg in range(len(route)):
                    mx = (rx[seg] + rx[seg+1]) / 2
                    my = (ry[seg] + ry[seg+1]) / 2
                    fig.add_annotation(
                        x=mx, y=my, ax=rx[seg], ay=ry[seg],
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowsize=1.2,
                        arrowwidth=2, arrowcolor="#38bdf8",
                    )

            # Draw brute-force route
            bf = st.session_state.bf_result
            if bf and bf["route"] and -1 not in bf["route"]:
                bfr = bf["route"] + [bf["route"][0]]
                bx = [coords[c,0] for c in bfr]
                by = [coords[c,1] for c in bfr]
                fig.add_trace(go.Scatter(
                    x=bx, y=by, mode="lines",
                    line=dict(color="rgba(250,204,21,0.35)", width=2, dash="dash"),
                    name="Classical Optimal",
                ))

            # City markers
            for i, city in enumerate(cities):
                e, l = tw[i]
                col  = city_colour(i)
                fig.add_trace(go.Scatter(
                    x=[city["x"]], y=[city["y"]],
                    mode="markers+text",
                    marker=dict(size=22, color=col, line=dict(width=2, color="#0b0e1a"),
                                symbol="circle"),
                    text=[city["name"]],
                    textfont=dict(color="#0b0e1a", size=10, family="Space Mono"),
                    textposition="middle center",
                    name=city["name"],
                    hovertemplate=(
                        f"<b>{city['name']}</b><br>"
                        f"Position: ({city['x']:.2f}, {city['y']:.2f})<br>"
                        f"Time window: [{e:.1f}, {l:.1f}]<extra></extra>"
                    )
                ))

                # Time-window arc indicator
                fig.add_shape(type="circle",
                    xref="x", yref="y",
                    x0=city["x"]-0.6, y0=city["y"]-0.6,
                    x1=city["x"]+0.6, y1=city["y"]+0.6,
                    line=dict(color=col, width=1.5, dash="dot"),
                    fillcolor="rgba(0,0,0,0)"
                )

            fig.update_layout(
                paper_bgcolor="#0b0e1a", plot_bgcolor="#0f1225",
                font=dict(color="#94a3b8", family="DM Sans"),
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=True, gridcolor="#1a2535", zeroline=False,
                           showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor="#1a2535", zeroline=False,
                           showticklabels=False, scaleanchor="x"),
                legend=dict(bgcolor="#0f1629", bordercolor="#1e3a5f",
                            borderwidth=1, font=dict(size=11)),
                title=dict(text="City Map & Optimised Route", font=dict(color="#7dd3fc")),
                height=440,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown('<div class="section-head">Instance</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
              <h4>Customers</h4>
              <div class="val">{inst.n}</div>
              <div class="sub">+ depot</div>
            </div>
            <div class="metric-card">
              <h4>Qubits</h4>
              <div class="val">{inst.n**2}</div>
              <div class="sub">n² encoding, depot fixed</div>
            </div>
            """, unsafe_allow_html=True)

            if result:
                route = result["route"]
                builder = st.session_state.builder
                valid = builder.is_valid_route(result["bitstring"])
                cost  = inst.tour_cost(route) if valid else 0
                viol  = inst.late_stops(route) if valid else []

                st.markdown(f"""
                <div class="metric-card">
                  <h4>Route Cost</h4>
                  <div class="val">{cost:.2f}</div>
                  <div class="sub">total distance</div>
                </div>
                <div class="metric-card">
                  <h4>QUBO Energy</h4>
                  <div class="val">{result['energy']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

                badge = '<span class="badge badge-green">✓ VALID</span>' if valid else '<span class="badge badge-red">✗ INVALID</span>'
                tw_badge = (f'<span class="badge badge-green">No violations</span>' 
                           if not viol else f'<span class="badge badge-red">{len(viol)} violation(s)</span>')
                st.markdown(f"""
                <div class="metric-card">
                  <h4>Route</h4>
                  {badge}<br><br>{tw_badge}
                </div>""", unsafe_allow_html=True)

                if valid and bf and bf["route"]:
                    gap = cost - inst.tour_cost(bf["route"])
                    st.markdown(f"""
                    <div class="metric-card">
                      <h4>vs Classical</h4>
                      <div class="val">{gap:.2f}</div>
                      <div class="sub">QAOA cost − exact optimum</div>
                    </div>""", unsafe_allow_html=True)

            # Time windows table
            st.markdown('<div class="section-head">Time Windows</div>', unsafe_allow_html=True)
            rows = ""
            for i, city in enumerate(cities):
                rows += f"<tr><td><span style='color:{city_colour(i)};'>●</span> {city['name']}</td><td>[{city['e']:.1f}, {city['l']:.1f}]</td></tr>"
            st.markdown(f"""
            <table class="route-table">
              <tr><th>City</th><th>Window</th></tr>
              {rows}
            </table>""", unsafe_allow_html=True)

    # Route sequence (if result)
    if result and st.session_state.builder.is_valid_route(result["bitstring"]):
        inst   = st.session_state.instance
        cities = st.session_state.cities
        stops  = result["route"] + [0]          # ends back at the depot
        arr    = inst.arrivals(result["route"])

        st.markdown('<div class="section-head">Route Sequence</div>', unsafe_allow_html=True)
        rows = ""
        for t, city_idx in enumerate(stops):
            cname = cities[city_idx]["name"] + (" (return)" if t == len(stops) - 1 else "")
            e, l  = inst.time_windows[city_idx]
            ok    = arr[t] <= l                 # arriving early just means waiting
            status = '<span class="badge badge-green">✓ On time</span>' if ok else '<span class="badge badge-red">✗ Late</span>'
            dist  = "" if t == 0 else f"{inst.d[stops[t-1], city_idx]:.2f}"
            rows += f"<tr><td>{t}</td><td><span style='color:{city_colour(city_idx)};'>●</span> {cname}</td><td>{arr[t]:.2f}</td><td>[{e:.1f},{l:.1f}]</td><td>{status}</td><td>{dist}</td></tr>"
        st.markdown(f"""
        <table class="route-table">
          <tr><th>Step</th><th>City</th><th>Service start</th><th>Window</th><th>Status</th><th>Leg dist.</th></tr>
          {rows}
        </table>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – QUBO Matrix
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    builder = st.session_state.builder
    if builder is None:
        st.info("Run optimisation first to view the QUBO matrix.")
    else:
        Q = builder.Q
        n = builder.n

        col_heat, col_desc = st.columns([3, 1])
        with col_heat:
            # Labels: x_{city,pos}
            labels = [f"x{j},{k}" for j in range(1, n + 1) for k in range(1, n + 1)]
            fig = go.Figure(go.Heatmap(
                z=Q, x=labels, y=labels,
                colorscale="RdBu_r",
                zmid=0,
                text=np.round(Q, 2),
                texttemplate="%{text}",
                textfont=dict(size=9, family="Space Mono"),
                hovertemplate="Q[%{x},%{y}] = %{z:.3f}<extra></extra>"
            ))
            fig.update_layout(
                title="QUBO Matrix Q  (upper triangular encoding)",
                paper_bgcolor="#0b0e1a", plot_bgcolor="#0f1225",
                font=dict(color="#94a3b8", family="DM Sans"),
                height=420,
                xaxis=dict(tickfont=dict(size=8, family="Space Mono", color="#7dd3fc")),
                yaxis=dict(tickfont=dict(size=8, family="Space Mono", color="#7dd3fc"), autorange="reversed"),
                coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8")),
                margin=dict(l=20,r=20,t=50,b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_desc:
            N = Q.shape[0]
            nnz = np.count_nonzero(Q)
            sparsity = 1 - nnz / N**2
            eigs = np.linalg.eigvalsh(Q + Q.T - np.diag(np.diag(Q)))
            st.markdown(f"""
            <div class="metric-card">
              <h4>Matrix Size</h4>
              <div class="val">{N}×{N}</div>
              <div class="sub">{N} binary variables</div>
            </div>
            <div class="metric-card">
              <h4>Non-zeros</h4>
              <div class="val">{nnz}</div>
              <div class="sub">sparsity {sparsity:.0%}</div>
            </div>
            <div class="metric-card">
              <h4>Min eigenvalue</h4>
              <div class="val">{eigs.min():.2f}</div>
            </div>
            <div class="metric-card">
              <h4>Max eigenvalue</h4>
              <div class="val">{eigs.max():.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-head">Penalty Terms</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table class="route-table">
              <tr><th>Term</th><th>Weight</th></tr>
              <tr><td>A · one-hot constraints</td><td>{builder.A:.2f}</td></tr>
              <tr><td>B · travel distance</td><td>{builder.B:.2f}</td></tr>
              <tr><td>P · time windows ({builder.tw})</td><td>{builder.P:.2f}</td></tr>
            </table>
            """, unsafe_allow_html=True)

        # Eigenspectrum
        st.markdown('<div class="section-head">QUBO Eigenspectrum</div>', unsafe_allow_html=True)
        Qsym = Q + Q.T - np.diag(np.diag(Q))
        eigs_all = np.linalg.eigvalsh(Qsym)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=list(range(len(eigs_all))), y=eigs_all,
            marker=dict(color=eigs_all, colorscale="RdBu_r", cmid=0, showscale=False),
            hovertemplate="Eigenvalue %{x}: %{y:.3f}<extra></extra>"
        ))
        fig2.update_layout(
            paper_bgcolor="#0b0e1a", plot_bgcolor="#0f1225",
            font=dict(color="#94a3b8"), height=220,
            xaxis=dict(title="Index", gridcolor="#1a2535"),
            yaxis=dict(title="Eigenvalue", gridcolor="#1a2535"),
            margin=dict(l=20,r=20,t=10,b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Circuit
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    qaoa = st.session_state.qaoa
    if qaoa is None:
        st.info("Run optimisation first to inspect the circuit.")
    else:
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown('<div class="section-head">Circuit Architecture</div>', unsafe_allow_html=True)
            n_q = qaoa.n_qubits
            p   = qaoa.p

            # Build a visual circuit diagram as SVG-like HTML
            layers = ["H"] + [f"U_C(γ{i+1}), U_B(β{i+1})" for i in range(p)] + ["Measure"]
            html = "<div style='font-family:Space Mono,monospace;font-size:0.72rem;color:#94a3b8;overflow-x:auto;'>"
            for q in range(n_q):
                inst2 = st.session_state.instance
                cities = st.session_state.cities
                city_i = q // inst2.n + 1       # customers are 1..n
                pos_t  = q %  inst2.n + 1
                cname  = cities[city_i]["name"] if cities else f"c{city_i}"
                col    = city_colour(city_i)
                line = f"<div style='display:flex;align-items:center;gap:4px;margin:2px 0;'>"
                line += f"<span style='width:80px;color:{col};'>q{q}[{cname},t{pos_t}]</span>"
                line += "<span style='color:#1d4ed8;'>──</span>"
                line += "<span style='background:#1e3a5f;color:#7dd3fc;padding:2px 6px;border-radius:4px;'>H</span>"
                for _ in range(p):
                    line += "<span style='color:#334155;'>────</span>"
                    line += "<span style='background:#14532d;color:#86efac;padding:2px 4px;border-radius:4px;'>U_C</span>"
                    line += "<span style='color:#334155;'>──</span>"
                    line += f"<span style='background:#3b0764;color:#d8b4fe;padding:2px 4px;border-radius:4px;'>{'XY' if qaoa.mixer == 'xy' else 'RX'}</span>"
                line += "<span style='color:#334155;'>────</span>"
                line += "<span style='background:#451a03;color:#fcd34d;padding:2px 6px;border-radius:4px;'>M</span>"
                line += "</div>"
                html += line
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

        with col_c2:
            st.markdown('<div class="section-head">Optimised Parameters</div>', unsafe_allow_html=True)
            if qaoa.best_params is not None:
                params = np.array(qaoa.best_params)
                gammas = params[:p]
                betas  = params[p:]

                rows = ""
                for i, (g, b) in enumerate(zip(gammas, betas)):
                    rows += f"<tr><td>{i+1}</td><td>{g:.4f}</td><td>{b:.4f}</td></tr>"
                st.markdown(f"""
                <table class="route-table">
                  <tr><th>Layer</th><th>γ (problem)</th><th>β (mixer)</th></tr>
                  {rows}
                </table>""", unsafe_allow_html=True)

                # Bloch sphere representation of angles
                fig = go.Figure()
                theta_vals = np.linspace(0, 2*np.pi, 100)
                fig.add_trace(go.Scatter(
                    x=np.cos(theta_vals), y=np.sin(theta_vals),
                    mode="lines", line=dict(color="#1e3a5f", width=1),
                    showlegend=False, hoverinfo="skip"
                ))
                for i, (g, b) in enumerate(zip(gammas, betas)):
                    g_mod = g % (2*np.pi)
                    b_mod = b % (2*np.pi)
                    fig.add_trace(go.Scatter(
                        x=[0, np.cos(g_mod)], y=[0, np.sin(g_mod)],
                        mode="lines+markers",
                        line=dict(color="#38bdf8", width=2),
                        marker=dict(size=[0,10], color="#38bdf8"),
                        name=f"γ{i+1}={g:.3f}",
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, np.cos(b_mod)], y=[0, np.sin(b_mod)],
                        mode="lines+markers",
                        line=dict(color="#a78bfa", width=2, dash="dash"),
                        marker=dict(size=[0,10], color="#a78bfa"),
                        name=f"β{i+1}={b:.3f}",
                    ))
                fig.update_layout(
                    paper_bgcolor="#0b0e1a", plot_bgcolor="#0f1225",
                    font=dict(color="#94a3b8"), height=260,
                    xaxis=dict(range=[-1.3,1.3], showgrid=False, zeroline=True, zerolinecolor="#1e3a5f"),
                    yaxis=dict(range=[-1.3,1.3], showgrid=False, zeroline=True, zerolinecolor="#1e3a5f", scaleanchor="x"),
                    legend=dict(bgcolor="#0f1629", bordercolor="#1e3a5f", borderwidth=1, font=dict(size=9)),
                    margin=dict(l=10,r=10,t=10,b=10),
                    title=dict(text="Parameter angle space", font=dict(color="#7dd3fc", size=11))
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-head">Hamiltonian Terms</div>', unsafe_allow_html=True)
        H_C = qaoa.H_C
        n_terms = len(H_C.coeffs)
        n_zz = sum(1 for op in H_C.ops if hasattr(op, 'name') and op.name == "Prod")
        st.markdown(f"""
        <div style='display:flex;gap:12px;flex-wrap:wrap;'>
          <div class="metric-card" style='flex:1;min-width:120px;'>
            <h4>Hamiltonian terms</h4><div class="val">{n_terms}</div>
          </div>
          <div class="metric-card" style='flex:1;min-width:120px;'>
            <h4>Qubits</h4><div class="val">{qaoa.n_qubits}</div>
          </div>
          <div class="metric-card" style='flex:1;min-width:120px;'>
            <h4>QAOA depth p</h4><div class="val">{p}</div>
          </div>
          <div class="metric-card" style='flex:1;min-width:120px;'>
            <h4>Circuit gates ~</h4><div class="val">{qaoa.n_qubits + p*(n_terms + qaoa.n_qubits)}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Convergence
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    result = st.session_state.result
    if result is None:
        st.info("Run optimisation first to view convergence plots.")
    else:
        losses = result["opt_result"]["loss_history"]
        steps_arr = list(range(1, len(losses)+1))

        col_cv, col_stats = st.columns([3, 1])
        with col_cv:
            fig = go.Figure()
            # Fill area
            fig.add_trace(go.Scatter(
                x=steps_arr, y=losses,
                fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
                line=dict(color="#38bdf8", width=0),
                showlegend=False, hoverinfo="skip"
            ))
            # Main loss line
            fig.add_trace(go.Scatter(
                x=steps_arr, y=losses,
                mode="lines",
                line=dict(color="#38bdf8", width=2),
                name="⟨H_C⟩",
                hovertemplate="Step %{x}: %{y:.4f}<extra></extra>"
            ))
            # Moving average
            window = max(1, len(losses)//10)
            ma = np.convolve(losses, np.ones(window)/window, mode="valid")
            fig.add_trace(go.Scatter(
                x=list(range(window, len(losses)+1)), y=ma.tolist(),
                mode="lines",
                line=dict(color="#f87171", width=1.5, dash="dash"),
                name=f"MA({window})",
            ))
            # Best point
            best_step = int(np.argmin(losses)) + 1
            best_val  = min(losses)
            fig.add_trace(go.Scatter(
                x=[best_step], y=[best_val],
                mode="markers",
                marker=dict(size=12, color="#4ade80", symbol="star"),
                name=f"Best: {best_val:.4f}",
            ))

            fig.update_layout(
                paper_bgcolor="#0b0e1a", plot_bgcolor="#0f1225",
                font=dict(color="#94a3b8"), height=360,
                xaxis=dict(title="Optimisation Step", gridcolor="#1a2535", title_font=dict(color="#64748b")),
                yaxis=dict(title="⟨H_C⟩  Expectation Value", gridcolor="#1a2535", title_font=dict(color="#64748b")),
                legend=dict(bgcolor="#0f1629", bordercolor="#1e3a5f", borderwidth=1),
                margin=dict(l=20,r=20,t=20,b=60),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            delta = losses[0] - losses[-1] if len(losses) > 1 else 0
            st.markdown(f"""
            <div class="metric-card">
              <h4>Initial ⟨H_C⟩</h4>
              <div class="val">{losses[0]:.3f}</div>
            </div>
            <div class="metric-card">
              <h4>Final ⟨H_C⟩</h4>
              <div class="val">{losses[-1]:.3f}</div>
            </div>
            <div class="metric-card">
              <h4>Best ⟨H_C⟩</h4>
              <div class="val">{best_val:.3f}</div>
              <div class="sub">step {best_step}</div>
            </div>
            <div class="metric-card">
              <h4>Improvement</h4>
              <div class="val">{delta:.3f}</div>
              <div class="sub">{delta/abs(losses[0])*100:.1f}% reduction</div>
            </div>
            """, unsafe_allow_html=True)

        # Gradient magnitude proxy (successive differences)
        if len(losses) > 5:
            st.markdown('<div class="section-head">Gradient magnitude proxy  |Δ⟨H_C⟩|</div>', unsafe_allow_html=True)
            grads = np.abs(np.diff(losses))
            fig2 = go.Figure(go.Scatter(
                x=list(range(2, len(losses)+1)), y=grads.tolist(),
                mode="lines", fill="tozeroy",
                fillcolor="rgba(167,139,250,0.1)",
                line=dict(color="#a78bfa", width=1.5),
            ))
            fig2.update_layout(
                paper_bgcolor="#0b0e1a", plot_bgcolor="#0f1225",
                font=dict(color="#94a3b8"), height=180,
                xaxis=dict(gridcolor="#1a2535"),
                yaxis=dict(gridcolor="#1a2535", type="log"),
                margin=dict(l=20,r=20,t=10,b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – State Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    qaoa   = st.session_state.qaoa
    result = st.session_state.result
    builder= st.session_state.builder

    if qaoa is None or result is None:
        st.info("Run optimisation first to analyse the quantum state.")
    else:
        col_prob, col_bs = st.columns([2, 1])

        with col_prob:
            st.markdown('<div class="section-head">Probability Landscape</div>', unsafe_allow_html=True)
            _, probs = qaoa.probability_landscape()
            N = len(probs)

            # Colour each bar by its QUBO energy
            energies = np.array([builder.energy(
                tuple(int(b) for b in format(i, f'0{qaoa.n_qubits}b'))
            ) for i in range(min(N, 256))])

            fig = go.Figure(go.Bar(
                x=list(range(min(N, 256))),
                y=probs[:256].tolist(),
                marker=dict(
                    color=energies, colorscale="Plasma", showscale=True,
                    colorbar=dict(title="QUBO E", tickfont=dict(color="#94a3b8", size=9))
                ),
                hovertemplate="State %{x}: P=%{y:.4f}, E=%{marker.color:.2f}<extra></extra>"
            ))
            fig.update_layout(
                paper_bgcolor="#0b0e1a", plot_bgcolor="#0f1225",
                font=dict(color="#94a3b8"), height=320,
                xaxis=dict(title="Basis state index", gridcolor="#1a2535"),
                yaxis=dict(title="Probability", gridcolor="#1a2535"),
                margin=dict(l=20,r=20,t=10,b=50),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_bs:
            st.markdown('<div class="section-head">Best Bitstring</div>', unsafe_allow_html=True)
            bs = result["bitstring"]
            n  = builder.n
            # Display as qubit grid
            html = "<div>"
            for i in range(n):
                city_name = st.session_state.cities[i + 1]["name"] if st.session_state.cities else f"C{i + 1}"
                html += f"<div style='margin-bottom:6px;'><span style='color:{city_colour(i + 1)};font-family:Space Mono;font-size:0.65rem;'>{city_name}</span><div class='qubit-row'>"
                for t in range(n):
                    bit = bs[i*n + t]
                    cls = "qubit-1" if bit else "qubit-0"
                    html += f"<div class='qubit-bit {cls}'>{bit}</div>"
                html += "</div></div>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

            valid = builder.is_valid_route(bs)
            st.markdown(f"""
            <div class="metric-card">
              <h4>Valid route?</h4>
              {'<span class="badge badge-green">✓ YES</span>' if valid else '<span class="badge badge-red">✗ NO</span>'}
            </div>
            <div class="metric-card">
              <h4>QUBO Energy</h4>
              <div class="val">{result['energy']:.3f}</div>
            </div>""", unsafe_allow_html=True)

        # Top-k sampled states
        st.markdown('<div class="section-head">Top Sampled States</div>', unsafe_allow_html=True)
        counts = result["counts"]
        total  = sum(counts.values())
        top_k  = sorted(counts, key=lambda b: -counts[b])[:16]
        rows = ""
        for bs_k in top_k:
            cnt   = counts[bs_k]
            prob  = cnt / total
            energy= builder.energy(bs_k)
            valid = builder.is_valid_route(bs_k)
            route = builder.decode(bs_k)
            route_str = "→".join(str(c) for c in route if c != -1)
            badge = '<span class="badge badge-green">✓</span>' if valid else '<span class="badge badge-red">✗</span>'
            bits  = "".join(str(b) for b in bs_k)
            rows += f"<tr><td><code style='font-size:0.65rem;color:#64748b;'>{bits}</code></td><td>{cnt}</td><td>{prob:.3f}</td><td>{energy:.2f}</td><td>{badge}</td><td>{route_str}</td></tr>"

        st.markdown(f"""
        <table class="route-table">
          <tr><th>Bitstring</th><th>Counts</th><th>Prob</th><th>QUBO E</th><th>Valid</th><th>Route</th></tr>
          {rows}
        </table>""", unsafe_allow_html=True)

        # Classical vs QAOA comparison
        bf = st.session_state.bf_result
        if bf and bf["route"]:
            st.markdown('<div class="section-head">Quantum vs Classical Comparison</div>', unsafe_allow_html=True)
            q_route = result["route"]
            c_route = bf["route"]
            inst    = st.session_state.instance

            q_cost = inst.tour_cost(q_route) if builder.is_valid_route(result["bitstring"]) else float('nan')
            c_cost = inst.tour_cost(c_route)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["QAOA (quantum)", "Brute-force (classical)"],
                y=[q_cost, c_cost],
                marker=dict(color=["#38bdf8", "#facc15"]),
                text=[f"{q_cost:.3f}", f"{c_cost:.3f}"],
                textposition="outside",
                textfont=dict(color="#e2e8f0", family="Space Mono"),
            ))
            fig.update_layout(
                paper_bgcolor="#0b0e1a", plot_bgcolor="#0f1225",
                font=dict(color="#94a3b8"), height=240,
                yaxis=dict(title="Route Cost", gridcolor="#1a2535"),
                margin=dict(l=20,r=20,t=20,b=40),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#334155;font-family:Space Mono,monospace;font-size:0.65rem;letter-spacing:0.1em;padding:12px 0;'>
  TSPTW · QAOA  ·  PennyLane + Streamlit  ·  QUBO formulation with time-window penalty Hamiltonian
</div>
""", unsafe_allow_html=True)
