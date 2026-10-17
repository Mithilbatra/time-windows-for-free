"""
TSPTW-QAOA paper companion
==========================
A multipage Streamlit app over the project's code and saved results: the paper, the method
and its code, an instance explorer that runs QAOA live, the results, the PennyLane circuit
demo, and how to reproduce everything.  The pages live in views/, shared pieces in ui.py.
Run:  streamlit run app.py
"""
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)                   # the project's modules,
os.chdir(ROOT)                             # and results/, figures/, ... from any launch directory
from ui import CSS

st.set_page_config(page_title="TSPTW · QAOA", page_icon="⚛️", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
st.navigation([
    st.Page("views/paper.py", title="Paper", icon=":material/article:", default=True),
    st.Page("views/method.py", title="Method", icon=":material/functions:"),
    st.Page("views/explore.py", title="Explore an instance", icon=":material/travel_explore:"),
    st.Page("views/results.py", title="Results", icon=":material/monitoring:"),
    st.Page("views/demo.py", title="QAOA circuit demo", icon=":material/memory:"),
    st.Page("views/reproduce.py", title="Reproduce", icon=":material/terminal:"),
]).run()
