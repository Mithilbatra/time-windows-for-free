# Time windows for free

Code, data and paper draft for *Time windows for free: a sound, ancilla-free time-window encoding for QAOA on the TSP with time windows*.

E1 turns each customer's earliest possible arrival at each tour position into penalties. They sit on couplings the distance term already has: no extra qubits, no extra two-qubit gates, and no feasible route is ever penalised. E1+C adds a look-ahead. Both are compared with a window-free encoding (E0) and the slack-qubit QUBO of Salehi et al. (2022) (E2), under simulated annealing and exact QAOA simulation with X and XY mixers.

## Layout

| Path | What it is |
|---|---|
| `qubo.py` | Instances, exact solver, E0 / E1 / E1+C encodings |
| `baselines.py` | E2 (Salehi et al.) and the simulated annealer |
| `qaoa.py` | Exact QAOA simulator (X and XY mixers) |
| `experiments.py`, `grid.py`, `pilot.py` | Annealing comparison, final QAOA grid, QAOA pilots |
| `pruning_check.py`, `pruning_large.py` | Classical pruning power, including Rezk and Gora's penalty |
| `results/` | Every number in the paper |
| `figures.py`, `figures/` | Paper figures |
| `paper/` | LaTeX draft (Springer LNCS) |
| `app.py`, `views/`, `ui.py` | Streamlit companion app |
| `related_work.md` | Literature review |

## Reproduce

```bash
pip install -r requirements.txt
python tests.py
```

Each script's docstring gives its exact command and run time, for example `python experiments.py | tee results/sa_summary.txt` (about 45 min on 10 cores) or `python grid.py run 8` (about 15 CPU hours). The app's Reproduce page lists them all: `streamlit run app.py`.

Rezk and Gora's time-window penalty (arXiv:2609.04593) is re-implemented from its formulas for comparison; none of their code is included.

## Citation

TODO: arXiv identifier once posted.

## Licence

MIT, see `LICENSE`.
