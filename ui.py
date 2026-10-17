"""
Shared pieces of the Streamlit app (app.py and views/): the dark theme, colours, plotly
styling and the cached loaders for the saved results.
"""
import csv

import numpy as np
import plotly.graph_objects as go
import streamlit as st

CSS = """
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
  .stTabs [data-baseweb="tab"] { color:#64748b; font-family:'Space Mono',monospace; font-size:0.7rem; letter-spacing:0.08em; padding:0 14px; }
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
"""

CITY_COLOURS = ["#f87171","#fb923c","#facc15","#4ade80","#38bdf8","#a78bfa","#f472b6","#67e8f9"]


def city_colour(i):
    return CITY_COLOURS[i % len(CITY_COLOURS)]


def layout(d):
    """2-D positions whose distances roughly match d (classical MDS), for display only."""
    J = np.eye(len(d)) - 1 / len(d)
    w, V = np.linalg.eigh(-0.5 * J @ d**2 @ J)
    return V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0))


# ── encodings: colour, marker, label ─────────────────────────────────────────────
# dark steps of the palette behind figures.py; with this gray for E0 the four pass the dataviz
# validator (all pairs, colour-blind and normal vision) on the plot surface #0f1225

ENC = {"E0": ("#64635f", "circle", "E0  no window term"),
       "E2": ("#199e70", "triangle-up", "E2  Salehi et al."),
       "E1": ("#3987e5", "square", "E1  ours"),
       "E1+C": ("#d95926", "diamond", "E1+C  ours, look-ahead")}       # drawing and legend order
SURFACE = "#0f1225"


def rgba(hex_colour, alpha):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def style(fig, height=360, **layout):
    """The dark plotly look of the demo page, applied to a figure."""
    fig.update_layout(paper_bgcolor="#0b0e1a", plot_bgcolor=SURFACE, height=height,
                      font=dict(color="#94a3b8", family="DM Sans"), margin=dict(l=20, r=20, t=40, b=40),
                      legend=dict(bgcolor="#0f1629", bordercolor="#1e3a5f", borderwidth=1), **layout)
    fig.update_xaxes(gridcolor="#1a2535", zeroline=False)
    fig.update_yaxes(gridcolor="#1a2535", zeroline=False)
    return fig


def band(fig, x, groups, enc, clip=(0, 1), **pos):
    """Mean line with markers over its 95% t-interval band (grid.ci); groups[i] holds the
    per-instance values at x[i].  pos = row/col for subplots."""
    import grid
    colour, symbol, label = ENC[enc]
    x = list(x)
    mean = np.array([np.mean(g) for g in groups])
    half = np.array([grid.ci(g) if len(g) > 1 else 0.0 for g in groups])
    lo, hi = np.clip(mean - half, *clip), np.clip(mean + half, *clip)
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(hi) + list(lo[::-1]), fill="toself", fillcolor=rgba(colour, 0.12),
                             mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=enc),
                  **pos)                           # mode set: plotly adds markers to traces under 20 points
    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines+markers", name=label, legendgroup=enc,
                             showlegend=pos.get("col", 1) == 1 and pos.get("row", 1) == 1,
                             line=dict(color=colour, width=2),
                             marker=dict(symbol=symbol, size=8, color=colour, line=dict(color=SURFACE, width=2)),
                             customdata=np.c_[lo, hi],
                             hovertemplate=f"{label}<br>%{{x}}: %{{y:.3f}}  (95% CI %{{customdata[0]:.3f}}–"
                                           "%{customdata[1]:.3f})<extra></extra>"), **pos)


# ── saved results, read once per session ───────────────────────────────────────────

def _value(v):
    if v in ("True", "False"):
        return v == "True"
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and v.lstrip("-").isdigit() else f


@st.cache_data
def csv_rows(path):
    """A results CSV as a list of dicts with numbers, booleans and strings restored."""
    return [{k: _value(v) for k, v in r.items()} for r in csv.DictReader(open(path))]


@st.cache_data
def grid_rows():
    import grid
    return grid.load()


@st.cache_data
def costs():
    """experiments.resource_counts(): (n, encoding) → mean (qubits, ZZ terms X mixer, XY mixer)."""
    from experiments import resource_counts
    return resource_counts()


@st.cache_data
def related_work():
    """related_work.md split into its '## ' sections: {heading: markdown}."""
    parts = open("related_work.md").read().split("\n## ")[1:]
    return {p.split("\n", 1)[0].strip(): p.split("\n", 1)[1] for p in parts}
