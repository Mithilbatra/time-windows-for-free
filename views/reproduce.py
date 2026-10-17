"""
Reproduce: what each file does, the command behind every saved result, the project's
self-checks (run them from here), the source itself, and the git history.
"""
import ast
import glob
import os
import re
import subprocess
import sys

import streamlit as st

SOURCES = sorted(glob.glob("*.py")) + sorted(glob.glob("views/*.py"))


def doc(path):
    return ast.get_docstring(ast.parse(open(path).read())) or ""


def summary(path):
    """The docstring's first paragraph; a bare title gets the paragraph after it too."""
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", re.sub(r"\n=+\n", "\n\n", doc(path)).strip())]
    return " — ".join(paras[:2]) if len(paras[0]) < 50 and len(paras) > 1 else paras[0]


st.markdown("# 🔁 Reproduce")
files, commands, checks, source, history = st.tabs(["Files", "Commands", "Self-checks", "Source", "History"])

with files:
    st.dataframe({"file": SOURCES,
                  "lines": [sum(1 for _ in open(p)) for p in SOURCES],
                  "what it is": [summary(p) for p in SOURCES]},
                 hide_index=True, use_container_width=True, height=460)

with commands:
    st.markdown("Every saved result comes from one command, taken here from the `Run:` lines in the "
                "files' docstrings. Results are written to `results/`, figures to `figures/`.")
    rows = [(p, line.strip()) for p in SOURCES if not p.startswith("views/")
            for line in re.findall(r"((?:python|streamlit run) [^\n]+)", doc(p))]
    st.dataframe({"file": [p for p, _ in rows], "command": [c for _, c in rows]}, hide_index=True,
                 use_container_width=True)
    st.code("pip install -r requirements.txt", language="bash")
    st.caption("requirements.txt (pinned to the versions the results were produced with)")
    st.code(open("requirements.txt").read(), language=None)

with checks:
    st.markdown("`tests.py` checks the QUBOs against their formulas term by term, the exact solver, the "
                "fast simulator against PennyLane, the baselines and the grid pipeline.")
    if st.button("▶ Run the self-checks (about 15 s)"):
        with st.spinner("Running tests.py…"):
            out = subprocess.run([sys.executable, "tests.py"], capture_output=True, text=True, timeout=900,
                                 env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        lines = [l for l in (out.stdout + out.stderr).splitlines() if "warn" not in l.lower()]
        passed = sum(l.startswith("ok ") for l in lines)
        (st.success if out.returncode == 0 else st.error)(
            f"{passed} checks passed" if out.returncode == 0 else f"failed after {passed} passing checks")
        st.code("\n".join(lines), language=None)

with source:
    path = st.selectbox("File", SOURCES + ["related_work.md", "requirements.txt"])
    st.code(open(path).read(), language="python" if path.endswith(".py") else "markdown", line_numbers=True)

with history:
    log = subprocess.run(["git", "log", "--format=%h  %ad  %s", "--date=short"], capture_output=True, text=True)
    st.code(log.stdout or "git history not available", language=None)
