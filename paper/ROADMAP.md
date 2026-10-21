# Paper roadmap

Targets: **arXiv by 31 Dec 2026**, then **ICCS 2027: submit by 5 Feb 2027 AoE** (notification 5 Apr, camera-ready 23 Apr, conference 12–14 Jul, Kraków).
Status 2026-10-18: full draft in `main.tex`, 15 pages including references (the ICCS limit), every number traced to `results/`. This file stays out of the arXiv bundle.

## What is left, all of it yours

1. **Read the whole paper** (`paper/main.pdf`, rebuild with `tectonic main.tex` in `paper/`). The text was drafted with AI assistance at your request, and the paper says so ("Use of AI Tools"). Change anything you would not say yourself. Check that the two closing statements are true for you: "no competing interests", and "the author checked every equation, number and citation". Check your college's rules on AI-written text too.
2. **Authors:** the affiliation (CSE, NIT Delhi) and the repository URL are filled in. Add co-authors or an acknowledgements section (e.g. your advisor) if you have them, in `main.tex` and `elsevier.tex`.
3. **Code and data** are public at https://github.com/Mithilbatra/time-windows-for-free (MIT):
   - push later changes with `git push`
   - optional DOI: sign in to zenodo.org with GitHub, switch the repository on, publish a GitHub release
   - Local tool files are untracked and stay private.
4. **Build the arXiv upload:** `sh paper/arxiv.sh`. It refuses while a TODO or an untraceable number remains, and writes `paper/main.pdf` and `paper/arxiv.zip` (main.tex, main.bbl, llncs.cls, two figures), checking that the zip compiles on its own.
5. **arXiv** (by 31 Dec): your first submission needs an endorsement, either an institutional email on your arXiv account or an endorser such as your advisor. Start now. Suggested categories: primary `quant-ph`, cross-list `math.OC`. Upload `arxiv.zip` and paste the plain-text abstract below.
6. **ICCS** (EasyChair, by 5 Feb 2027): upload the PDF, paste the abstract and the relevance statement below. Review is single-blind, so keep your name on the paper. If it is rejected on 5 Apr: GECCO QuantOpt workshop or IEEE QCE. No dual submission.

## Ready to paste

**Abstract (plain text):**
QUBO encodings of the travelling salesman problem with time windows (TSPTW) usually enforce the windows with slack variables, which multiplies the number of qubits and two-qubit gates. We bound every customer's earliest possible service start at every tour position with a dynamic programme and turn the bounds into penalties. The resulting encoding, E1, needs no qubits or two-qubit terms beyond those of the window-free position encoding and provably never penalises a feasible route; a look-ahead variant, E1+C, prunes more for a few percent more two-qubit terms. In exact QAOA simulation with an XY mixer on three to five customers, E1+C raises the probability of sampling the optimal route over a window-free encoding at every depth, by 0.094 to 0.217 at depth 12. Under simulated annealing, E1 and E1+C outperform the slack-qubit encoding of Salehi et al. and a slack-free unbalanced-penalization variant of it with far fewer qubits. Over all routes of instances with up to nine customers, E1 penalises 68.8-85.2% of the late routes and, unlike a recent heuristic penalty, never an on-time one.

**ICCS "relevance to computational science" (draft, edit freely):**
The paper develops and benchmarks a computational method for a classical combinatorial problem: a QUBO encoding of the travelling salesman problem with time windows for quantum optimisation algorithms. It pairs a classical dynamic-programming preprocessing step with exact state-vector simulation of QAOA and large simulated-annealing comparisons, and so falls under the call's Quantum Computing and Hybrid Computational Methods topics.

## Done

| Step | Where |
|---|---|
| Research, results, figures | `results/`, `figures/` (earlier commits) |
| E2-UP baseline (unbalanced penalization) | 117ce2e, merged f20cda0 |
| Pruning analysis, LaTeX skeleton, references checked against arXiv, Crossref and Springer | a7a9704 and later |
| ICCS facts, LNCS volumes, INTERP citation | bb4b0ba and later |
| README and MIT licence | 3152ce8 |
| Full draft, 15 pages; Tectonic build; `arxiv.sh` | ae0d124 |
| Checks: `python3 paper/check_numbers.py`, `python3 tests.py` (19 tests) | run on every change |

Skipped on purpose: a P(on time) table and the gap-matrix table (no room in 15 pages); the `resources` figure is left out of the paper because Table 4 holds the same numbers.
