# Paper roadmap

Targets: **arXiv by 31 Dec 2026**, then **ICCS 2027: submit by 5 Feb 2027 AoE** (notification 5 Apr, camera-ready 23 Apr, conference 12–14 Jul, Kraków).
Rules: you write all the prose (Springer AI policy); every number comes from `results/` (`python3 paper/check_numbers.py`). Writing items in detail: the TODO block at the top of `main.tex`.
This file stays out of the arXiv bundle. Last updated 2026-10-18.

| # | Step | Who | Status |
|---|------|-----|--------|
| 1 | Compile on Overleaf (`main.tex`, `refs.bib`, `figures/*.pdf`); send the log if it breaks | you, the assistant fixes | **todo** |
| 2 | Pruning-power table from `results/pruning_check.txt` | the assistant | done (d22af28) |
| 3 | Commit on branch `paper-draft` | the assistant | done (d22af28); merging into main still pending |
| 4 | Check ICCS facts, LNCS volumes, INTERP citation, weight grids | the assistant | done |
| 5 | Write the prose: Method → Setup → Results → Limitations → Related work → Intro → Conclusion → Abstract; Lemma 1 proof; captions | you | **todo** |
| 6 | Review each section as you finish it: numbers, claims vs data, citations, LNCS format, ≤ 15 pages incl. references | the assistant | waits on 5 |
| 7 | Optional extras: unbalanced-penalization baseline, P(on time) table, gap-matrix table | you decide | open |
| 8 | Public code and data (repository, licence, Zenodo DOI); arXiv zip (figures inside, `.bbl` from Overleaf, no outline bullets or TODOs) | the assistant, your decisions | open |
| 9 | arXiv submission. First submission needs an endorsement: add an institutional email to your arXiv account, or ask an endorser (e.g. your advisor). **Start early.** | you | todo |
| 10 | ICCS submission on EasyChair: fill in the "relevance to computational science" field; include the AI disclosure | you | by 5 Feb 2027 |

If ICCS rejects (5 Apr): GECCO QuantOpt workshop or IEEE QCE. No dual submission.
