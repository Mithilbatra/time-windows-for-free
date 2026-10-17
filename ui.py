"""
Shared pieces of the Streamlit app (app.py and views/): the dark theme, colours, plotly
styling and the cached loaders for the saved results.
"""
import csv

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── the look: a quiet dark theme with a few quantum touches ───────────────────────
# a faint qubit-lattice backdrop, ket-numbered page headers (|000⟩ … |101⟩), a Bloch sphere on
# the landing page and one cyan→violet gradient used sparingly. IBM Plex is a nod to IBM
# Quantum's typeface; offline it falls back to the system font.

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

  :root {
    --bg: #070b16; --surface: #0c1222; --surface-2: #111a2f;
    --line: #1a2440; --line-2: #26335a;
    --text: #e8ebf4; --text-2: #b3bbd0; --muted: #7c86a0;
    --cyan: #5ce1e6; --blue: #7aa2ff; --violet: #a78bfa;
    --grad: linear-gradient(90deg, #5ce1e6 0%, #7aa2ff 50%, #a78bfa 100%);
    --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  }

  /* ── canvas: soft glows and a faint lattice of points ── */
  .stApp {
    font-family: var(--sans); color: var(--text); isolation: isolate;
    background:
      radial-gradient(1100px 560px at 88% -8%, rgba(122,162,255,0.10), transparent 62%),
      radial-gradient(900px 520px at -12% 108%, rgba(167,139,250,0.08), transparent 60%),
      var(--bg);
  }
  .stApp::before {
    content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background-image: radial-gradient(rgba(148,163,255,0.08) 1px, transparent 1.4px);
    background-size: 28px 28px;
    -webkit-mask-image: linear-gradient(180deg, #000 0%, transparent 75%);
            mask-image: linear-gradient(180deg, #000 0%, transparent 75%);
  }
  header[data-testid="stHeader"] { background: rgba(7,11,22,0.72); backdrop-filter: blur(10px); }
  [data-testid="stDecoration"] { background: var(--grad); height: 2px; }
  .block-container { padding-top: 3rem; max-width: 1280px; }

  .stApp h1, .stApp h2, .stApp h3, .stApp h5, .stApp h6,
  [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li, .stApp button, .stApp input, .stApp textarea { font-family: var(--sans); }
  .stApp code, .stApp pre, .stApp pre * { font-family: var(--mono) !important; }

  .stMarkdown p, .stMarkdown li { color: var(--text-2); font-size: 0.95rem; line-height: 1.65; }
  .stMarkdown strong { color: var(--text); font-weight: 600; }
  .stMarkdown h2 { color: var(--text); font-size: 1.15rem !important; font-weight: 600; }
  .stMarkdown h3 { color: var(--text-2); font-size: 1rem !important; font-weight: 500; }
  .stMarkdown h4 {
    font-family: var(--mono); color: var(--cyan); font-size: 0.72rem !important; font-weight: 500;
    letter-spacing: 0.16em; text-transform: uppercase; padding: 1.2rem 0 0.2rem;
  }
  .stMarkdown :not(pre) > code {
    color: #c4b5fd; background: rgba(167,139,250,0.10); border-radius: 5px; padding: 0.1em 0.35em; font-size: 0.85em;
  }
  .stMarkdown hr, [data-testid="stMarkdownContainer"] hr {
    border: none; height: 1px; background: linear-gradient(90deg, transparent, var(--line-2), transparent);
  }
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); font-size: 0.82rem; font-family: var(--sans); }
  body, [data-baseweb="popover"] li { font-family: var(--sans); }
  .katex-display {
    margin: 0.2rem 0 !important; padding: 0.85rem 1rem; border-radius: 12px; overflow-x: auto;
    background: rgba(12,18,34,0.6); border: 1px solid var(--line); box-shadow: inset 2px 0 0 rgba(122,162,255,0.5);
  }

  /* ── sidebar: brand, navigation, controls ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1020 0%, #080c18 100%); border-right: 1px solid var(--line);
  }
  [data-testid="stSidebarHeader"] { padding: 1.4rem 1rem 0.6rem 1.4rem; align-items: flex-start; }
  section[data-testid="stSidebar"] [data-testid="stLogoSpacer"] { flex: 1; height: auto; }
  section[data-testid="stSidebar"] [data-testid="stLogoSpacer"]::before {
    content: "⟨TSPTW | QAOA⟩"; display: block; font-family: var(--mono); font-weight: 500;
    font-size: 1.02rem; letter-spacing: 0.02em;
    background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  section[data-testid="stSidebar"] [data-testid="stLogoSpacer"]::after {
    content: "Time windows for free"; display: block; margin-top: 4px;
    font-family: var(--sans); font-size: 0.74rem; letter-spacing: 0.04em; color: var(--muted);
  }
  [data-testid="stSidebarNav"] { padding-top: 0.4rem; border-bottom: 1px solid var(--line); padding-bottom: 0.8rem; }
  a[data-testid="stSidebarNavLink"] { border-radius: 9px; margin: 1px 0.6rem; padding: 0.38rem 0.7rem; transition: background 0.15s; }
  a[data-testid="stSidebarNavLink"]:hover { background: rgba(122,162,255,0.08); }
  a[data-testid="stSidebarNavLink"] span { color: var(--text-2); font-size: 0.9rem; }
  a[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] { color: var(--muted); font-size: 1.15rem; }
  [data-testid="stWidgetLabel"] p { color: var(--text-2); font-size: 0.82rem; font-weight: 500; }

  /* ── page headers and the landing hero ── */
  .eyebrow {
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--muted); display: flex; align-items: center; gap: 0.6rem;
  }
  .ket {
    font-family: var(--mono); letter-spacing: 0.02em; text-transform: none; color: var(--cyan);
    border: 1px solid rgba(92,225,230,0.28); background: rgba(92,225,230,0.06);
    border-radius: 6px; padding: 0.1rem 0.45rem;
  }
  .page-head { padding: 0.2rem 0 1.4rem; margin-bottom: 1.2rem; border-bottom: 1px solid var(--line); }
  .page-title { font-size: 2rem; font-weight: 600; letter-spacing: -0.02em; color: var(--text); margin: 0.7rem 0 0.35rem; line-height: 1.15; }
  .page-head .page-sub { color: var(--text-2); font-size: 1rem; max-width: 62rem; margin: 0; }

  .hero {
    display: grid; grid-template-columns: minmax(0, 1fr) 240px; gap: 2rem; align-items: center;
    padding: 2rem 2.2rem; margin-bottom: 1.6rem; border: 1px solid var(--line); border-radius: 20px;
    background:
      radial-gradient(600px 300px at 100% 0%, rgba(122,162,255,0.12), transparent 70%),
      linear-gradient(180deg, rgba(17,26,47,0.85), rgba(12,18,34,0.85));
    position: relative; overflow: hidden;
  }
  .hero::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 1px; background: var(--grad); opacity: 0.8; }
  .hero-title { font-size: 2.9rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.05; color: var(--text); margin: 1rem 0 0.7rem; }
  .grad { background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .hero .hero-sub { font-size: 1.08rem; color: var(--text-2); max-width: 40rem; margin: 0 0 1.3rem; line-height: 1.55; }
  .chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
  .chip {
    font-size: 0.78rem; color: var(--text-2); border: 1px solid var(--line-2); background: rgba(7,11,22,0.5);
    border-radius: 999px; padding: 0.28rem 0.75rem; display: inline-flex; align-items: center; gap: 0.45rem;
  }
  .chip .dot { width: 7px; height: 7px; border-radius: 50%; background: #34d399; box-shadow: 0 0 8px #34d399; }
  @media (max-width: 900px) { .hero { grid-template-columns: 1fr; } .hero-art { display: none; } .hero-title { font-size: 2.2rem; } }

  /* the Bloch sphere (ui._bloch) */
  .hero-art { width: 240px; height: 240px; }
  .hero-art svg { width: 100%; height: 100%; overflow: visible; }
  .bloch-tip { filter: drop-shadow(0 0 6px rgba(167,139,250,0.9)); }
  .bloch-still { display: none; }
  @media (prefers-reduced-motion: reduce) { .bloch-live { display: none; } .bloch-still { display: inline; } }

  /* ── cards ── */
  .metric-card {
    position: relative; overflow: hidden; margin-bottom: 12px; padding: 18px 20px; border-radius: 14px;
    border: 1px solid var(--line); background: linear-gradient(180deg, rgba(17,26,47,0.92), rgba(12,18,34,0.92));
    transition: border-color 0.2s, transform 0.2s;
  }
  .metric-card::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 1px; background: var(--grad); opacity: 0.55; }
  .metric-card:hover { border-color: var(--line-2); }
  .metric-card h4 {
    font-family: var(--mono); color: var(--muted); font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.14em; text-transform: uppercase; margin: 0 0 8px; padding: 0;
  }
  .metric-card .val {
    font-family: var(--sans); font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em;
    color: var(--text); font-variant-numeric: tabular-nums; line-height: 1.15;
  }
  .metric-card .sub { font-size: 0.8rem; color: var(--text-2); margin-top: 6px; line-height: 1.45; }
  .metric-card.tile { min-height: 142px; }
  .metric-card [data-testid="stHeaderActionElements"] { display: none; }    /* no anchor links on card labels */

  /* section labels: a glowing qubit, the label, a fading rule */
  .section-head {
    display: flex; align-items: center; gap: 10px; margin: 30px 0 14px;
    font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--cyan);
  }
  .section-head::before { content: ""; flex: none; width: 6px; height: 6px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 10px var(--cyan); }
  .section-head::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, var(--line-2), transparent); }

  /* badges */
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px; font-family: var(--mono);
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.04em;
  }
  .badge-blue   { background: rgba(122,162,255,0.12); color: #a9c1ff; border: 1px solid rgba(122,162,255,0.35); }
  .badge-green  { background: rgba(52,211,153,0.12);  color: #6ee7b7; border: 1px solid rgba(52,211,153,0.35); }
  .badge-red    { background: rgba(248,113,113,0.12); color: #fca5a5; border: 1px solid rgba(248,113,113,0.35); }
  .badge-yellow { background: rgba(251,191,36,0.12);  color: #fcd34d; border: 1px solid rgba(251,191,36,0.35); }

  /* tables written as HTML (demo page) */
  .route-table {
    width: 100%; border-collapse: separate; border-spacing: 0; font-family: var(--mono); font-size: 0.78rem;
    border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: var(--surface);
  }
  .route-table th {
    background: var(--surface-2); color: var(--muted); padding: 9px 12px; text-align: left; font-weight: 500;
    font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; border-bottom: 1px solid var(--line);
  }
  .route-table td { padding: 8px 12px; border-bottom: 1px solid var(--line); color: var(--text-2); }
  .route-table tr:last-child td { border-bottom: none; }
  .route-table tr:hover td { background: rgba(122,162,255,0.05); }

  /* qubit grid: |1⟩ glows, |0⟩ stays dark */
  .qubit-row { display: flex; gap: 5px; flex-wrap: wrap; margin: 8px 0; }
  .qubit-bit {
    width: 28px; height: 28px; border-radius: 7px; display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 0.72rem; font-weight: 500;
  }
  .qubit-1 { background: linear-gradient(135deg, #4f7cff, #8b5cf6); color: #fff; box-shadow: 0 0 12px rgba(122,120,255,0.45); }
  .qubit-0 { background: var(--surface-2); color: var(--muted); border: 1px solid var(--line); }

  /* ── Streamlit elements ── */
  .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; background: transparent; border-bottom: 1px solid var(--line); }
  .stTabs [data-baseweb="tab"] { padding: 0.55rem 0.9rem; height: auto; background: transparent; }
  .stTabs [data-baseweb="tab"] p { font-size: 0.9rem; font-weight: 500; color: var(--muted); }
  .stTabs [data-baseweb="tab"]:hover p { color: var(--text-2); }
  .stTabs [aria-selected="true"] p { color: var(--text); }
  .stTabs [data-baseweb="tab-highlight"] { background: var(--grad); height: 2px; }
  .stTabs [data-baseweb="tab-border"] { display: none; }

  .stButton > button {
    background: linear-gradient(135deg, #3b63f0 0%, #7c5cf5 100%); color: #fff;
    border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 0.55rem 1.2rem;
    box-shadow: 0 6px 20px -8px rgba(124,92,245,0.7), inset 0 1px 0 rgba(255,255,255,0.15);
    transition: filter 0.15s, box-shadow 0.15s, transform 0.15s;
  }
  .stButton > button p { font-weight: 600; font-size: 0.88rem; }
  .stButton > button:hover { color: #fff; border-color: rgba(255,255,255,0.18); filter: brightness(1.1);
                             box-shadow: 0 10px 28px -8px rgba(124,92,245,0.9), inset 0 1px 0 rgba(255,255,255,0.2); }
  .stButton > button:active, .stButton > button:focus, .stButton > button:focus:not(:active) { color: #fff; }
  .stButton > button:active { transform: translateY(1px); }
  .stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 2px solid #a5b4fc; outline-offset: 2px; }
  .stDownloadButton > button {
    background: var(--surface-2); color: var(--text-2); border: 1px solid var(--line-2); border-radius: 10px;
  }
  .stDownloadButton > button:hover { border-color: var(--blue); color: var(--text); }

  [data-testid="stPlotlyChart"] {
    background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 8px 8px 2px; overflow: hidden;
  }
  [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
  [data-testid="stImage"] img {
    background: #fff; border-radius: 14px; padding: 14px; border: 1px solid var(--line-2);
    max-width: min(100%, 940px); margin: 0 auto; display: block;
  }
  [data-testid="stExpander"] details {
    border: 1px solid var(--line); border-radius: 12px; background: rgba(12,18,34,0.65);
  }
  [data-testid="stExpander"] summary:hover { color: var(--text); }
  [data-testid="stCodeBlock"] pre { background: #0a1020 !important; border: 1px solid var(--line); border-radius: 12px; }
  [data-testid="stAlert"] > div { border-radius: 12px; }
</style>
"""

NAV_ACTIVE = """<style>
  a[data-testid="stSidebarNavLink"][href$="/%s"] {
    background: linear-gradient(90deg, rgba(122,162,255,0.16), rgba(167,139,250,0.03)); box-shadow: inset 2px 0 0 var(--blue);
  }
  a[data-testid="stSidebarNavLink"][href$="/%s"] span { color: var(--text); }
  a[data-testid="stSidebarNavLink"][href$="/%s"] [data-testid="stIconMaterial"] { color: var(--cyan); }
</style>"""


def header(ket, eyebrow, title, sub, path):
    """A page's header: ket-numbered eyebrow, title and subtitle.  It also highlights the page's
    sidebar link, since Streamlit 1.37 gives the active link no attribute to style; path is the
    page's URL path ("" for the default page)."""
    st.markdown(NAV_ACTIVE % (path, path, path) +
                f'<div class="page-head"><div class="eyebrow"><span class="ket">|{ket}⟩</span>{eyebrow}</div>'
                f'<div class="page-title" role="heading" aria-level="1">{title}</div>'
                f'<p class="page-sub">{sub}</p></div>', unsafe_allow_html=True)


def _bloch():
    """A Bloch sphere seen from 16° above the equator, its state vector (θ = 40°) precessing
    about z.  Positions are sampled around the latitude circle and animated with SMIL; with
    reduced motion only the still vector is shown."""
    c, r, tilt, theta = 120, 88, np.radians(16), np.radians(40)
    rx, h = r * np.sin(theta), r * np.cos(theta) * np.cos(tilt)          # latitude circle, height
    phi = np.linspace(0, 2 * np.pi, 49)
    xs, foot = c + rx * np.cos(phi), c + rx * np.sin(tilt) * np.sin(phi)
    vals = lambda a: ";".join(f"{v:.1f}" for v in a)
    anim = lambda attr, a: f'<animate attributeName="{attr}" values="{vals(a)}" dur="18s" repeatCount="indefinite"/>'
    pole, eq = r * np.cos(tilt), r * np.sin(tilt)
    still = (c + rx * np.cos(2.2), c - h + rx * np.sin(tilt) * np.sin(2.2))
    return f"""
<svg viewBox="0 0 240 240" role="img" aria-label="Bloch sphere with a precessing qubit state">
  <defs>
    <radialGradient id="bloch-fill" cx="38%" cy="32%" r="75%">
      <stop offset="0%" stop-color="#7aa2ff" stop-opacity="0.22"/><stop offset="100%" stop-color="#7aa2ff" stop-opacity="0.02"/>
    </radialGradient>
    <linearGradient id="bloch-vec" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0%" stop-color="#5ce1e6"/><stop offset="100%" stop-color="#a78bfa"/>
    </linearGradient>
  </defs>
  <circle cx="{c}" cy="{c}" r="{r}" fill="url(#bloch-fill)" stroke="rgba(148,163,255,0.38)" stroke-width="1"/>
  <path d="M {c - r} {c} A {r} {eq:.1f} 0 0 1 {c + r} {c}" fill="none" stroke="rgba(148,163,255,0.18)" stroke-dasharray="3 4"/>
  <path d="M {c - r} {c} A {r} {eq:.1f} 0 0 0 {c + r} {c}" fill="none" stroke="rgba(148,163,255,0.38)"/>
  <line x1="{c}" y1="{c - pole:.1f}" x2="{c}" y2="{c + pole:.1f}" stroke="rgba(148,163,255,0.22)" stroke-dasharray="2 4"/>
  <line x1="{c - r}" y1="{c}" x2="{c + r}" y2="{c}" stroke="rgba(148,163,255,0.12)" stroke-dasharray="2 4"/>
  <ellipse cx="{c}" cy="{c - h:.1f}" rx="{rx:.1f}" ry="{rx * np.sin(tilt):.1f}" fill="none" stroke="rgba(92,225,230,0.5)" stroke-dasharray="2 3"/>
  <circle cx="{c}" cy="{c - pole:.1f}" r="2.5" fill="#7c86a0"/><circle cx="{c}" cy="{c + pole:.1f}" r="2.5" fill="#7c86a0"/>
  <text x="{c}" y="{c - pole - 10:.1f}" text-anchor="middle" font-size="13" fill="#b3bbd0" font-family="IBM Plex Mono, monospace">|0⟩</text>
  <text x="{c}" y="{c + pole + 20:.1f}" text-anchor="middle" font-size="13" fill="#b3bbd0" font-family="IBM Plex Mono, monospace">|1⟩</text>
  <g class="bloch-live">{anim("opacity", 0.72 + 0.28 * np.sin(phi))}
    <line x1="{c}" y1="{c}" stroke="rgba(167,139,250,0.6)" stroke-dasharray="2 3">{anim("x2", xs)}{anim("y2", foot)}</line>
    <line stroke="rgba(167,139,250,0.6)" stroke-dasharray="2 3">{anim("x1", xs)}{anim("x2", xs)}{anim("y1", foot - h)}{anim("y2", foot)}</line>
    <line x1="{c}" y1="{c}" stroke="url(#bloch-vec)" stroke-width="2.6" stroke-linecap="round">{anim("x2", xs)}{anim("y2", foot - h)}</line>
    <circle r="5.5" fill="#a78bfa" class="bloch-tip">{anim("cx", xs)}{anim("cy", foot - h)}</circle>
  </g>
  <g class="bloch-still">
    <line x1="{c}" y1="{c}" x2="{still[0]:.1f}" y2="{still[1]:.1f}" stroke="url(#bloch-vec)" stroke-width="2.6" stroke-linecap="round"/>
    <circle cx="{still[0]:.1f}" cy="{still[1]:.1f}" r="5.5" fill="#a78bfa" class="bloch-tip"/>
  </g>
</svg>"""


def hero(eyebrow, title_html, sub, chips):
    """The landing page's header: title, subtitle, status chips and the Bloch sphere."""
    chips = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(NAV_ACTIVE % ("", "", "") +
                f'<div class="hero"><div><div class="eyebrow"><span class="ket">|000⟩</span>{eyebrow}</div>'
                f'<div class="hero-title" role="heading" aria-level="1">{title_html}</div>'
                f'<p class="hero-sub">{sub}</p><div class="chips">{chips}</div></div>'
                f'<div class="hero-art">{_bloch()}</div></div>', unsafe_allow_html=True)

CITY_COLOURS = ["#f87171","#fb923c","#facc15","#4ade80","#38bdf8","#a78bfa","#f472b6","#67e8f9"]


def city_colour(i):
    return CITY_COLOURS[i % len(CITY_COLOURS)]


def layout(d):
    """2-D positions whose distances roughly match d (classical MDS), for display only."""
    J = np.eye(len(d)) - 1 / len(d)
    w, V = np.linalg.eigh(-0.5 * J @ d**2 @ J)
    return V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0))


# ── encodings: colour, marker, label ─────────────────────────────────────────────
# dark steps of the palette behind figures.py; the four pass the dataviz validator (all pairs,
# colour-blind and normal vision, contrast) on the plot surface #0c1222, E0's gray being
# deliberately neutral (the control)

ENC = {"E0": ("#64635f", "circle", "E0  no window term"),
       "E2": ("#199e70", "triangle-up", "E2  Salehi et al."),
       "E1": ("#3987e5", "square", "E1  ours"),
       "E1+C": ("#d95926", "diamond", "E1+C  ours, look-ahead")}       # drawing and legend order
SURFACE = "#0c1222"                                  # = --surface in CSS: charts sit in their cards
FONT = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"


def rgba(hex_colour, alpha):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def style(fig, height=360, **layout):
    """The app's dark plotly look, applied to a figure."""
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
                      font=dict(color="#9aa3b8", family=FONT), margin=dict(l=20, r=20, t=40, b=40),
                      hoverlabel=dict(bgcolor="#111a2f", bordercolor="#26335a", font=dict(color="#e8ebf4", family=FONT)),
                      legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a2440", borderwidth=1), **layout)
    fig.update_xaxes(gridcolor="#18213a", zeroline=False, linecolor="#1a2440")
    fig.update_yaxes(gridcolor="#18213a", zeroline=False, linecolor="#1a2440")
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
