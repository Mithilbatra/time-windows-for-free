# Related work: TSPTW encodings for QAOA

Literature review compiled 2026-10-15. Each entry's bibliographic details were checked against arXiv or the publisher page. "?" means the paper's abstract or text did not say, so check before citing.

## 1. Closest work

| # | Paper | Problem | How time windows (TW) are handled | Solver, size | What it means for us |
|---|---|---|---|---|---|
| 1 | **Rezk & Gora (2026)**, arXiv [2609.04593](https://arxiv.org/abs/2609.04593) | CVRPTW | Position encoding (from #7). TW penalty in 3 parts: depot-reachability check, pairwise term between consecutive steps, look-ahead over steps k→k+2. Exact formulas are only in their code ([repo](https://github.com/YoussefKamelKamel1/cvrptw-qubo-coarsening)). | Simulated annealing + D-Wave Advantage2; Solomon, N=10–100 after coarsening | **Closest prior work.** Our pairwise term resembles theirs. We must set ourselves apart on: position-dependent *cumulative* earliest-arrival bounds; staying on existing couplings (their k→k+2 term adds new ones); gate-model QAOA + XY mixer; a soundness proof. |
| 2 | **Salehi, Glos & Miszczak (2022)**, *Quantum Inf. Process.* 21:67, [arXiv 2106.09056](https://arxiv.org/abs/2106.09056) | TSPTW | Corrected edge-based QUBO with binary-encoded slack and waiting variables, n³−2n²+3n+3nδ qubits. Also a node-based HOBO and an ILP-based QUBO. | D-Wave Advantage; 10 instances each for n=3,4,5. The lowest-energy sample missed the optimum on 3 of 10 instances at n=4. | **Our E2 baseline.** Adopt their instance protocol (§4.1) so results are comparable. |
| 3 | **Papalitsas et al. (2019)**, *Algorithms* 12(11):224, [doi](https://doi.org/10.3390/a12110224) | TSPTW | First TSPTW QUBO, using slack variables. Assumes every earliest time e_v = 0. | Formulation aimed at D-Wave | Cite as the origin. #2 reports a flaw in it, so use #2's corrected version. |
| 4 | **Holliday, Blount, Osaba & Luu (2025)**, arXiv [2503.24285](https://arxiv.org/abs/2503.24285) | TSPTW, CVRPTW | Hard constraints in the D-Wave CQM solver, plus classical swap repair | D-Wave hybrid; 3.86% average gap | TW feasibility drops as size grows, which supports reporting feasibility first. |
| 5 | **Harwood et al. (2021)**, *IEEE Trans. Quantum Eng.* 2:1–17, [doi](https://doi.org/10.1109/TQE.2021.3049230) | Inventory routing cast as VRPTW | Several formulations compared using QUBO-hardness metrics | Simulated VQE / QAOA / ADMM | Shows that "compare encodings" papers get accepted at good venues. |
| 6 | **Leonidas, Dukakis, Tan & Angelakis (2023)**, arXiv [2306.08507](https://arxiv.org/abs/2306.08507) | VRPTW | ? (instance sizes are given in routes, which suggests a route-based formulation) | Variational solver on IBMQ, Rigetti, IonQ; 11–3,964 routes | Alternative approach: handle TW classically before the quantum step. Contrast with it in related work. |
| 7 | **Borowski, Gora et al. (2020)**, ICCS 2020, LNCS, [doi](https://doi.org/10.1007/978-3-030-50433-5_42) | VRP, CVRP | No TW. Introduces the position encoding that #1 uses. | D-Wave hybrid | Shows ICCS publishes quantum routing work. Authors of #1 and #7 are plausible reviewers there. |
| 8 | **Azad et al. (2023)**, *IEEE Trans. ITS* 24(7):7564–7573, [arXiv 2002.01351](https://arxiv.org/abs/2002.01351) | VRP | No TW | Qiskit QAOA, ≤15 qubits, vs CPLEX | Results depend on the optimizer, p and initialisation, so we need multiple initialisations and fixed seeds. |
| 9 | **Qian et al. (2023)**, *Entropy* 25(8):1238, [arXiv 2307.07243](https://arxiv.org/abs/2307.07243) | TSP | No TW | 3 QAOA mixers compared, 3–5 cities, noise simulations | Mixer comparison for plain TSP; we extend it to TSPTW. |
| 10 | **do Carmo et al. (2025)**, arXiv [2504.19934](https://arxiv.org/abs/2504.19934) | TSP (VRP subroutine) | No TW | Goemans–Williamson warm start + XY mixer, 5 cities | The XY mixer on one-hot TSP is current practice; TW are absent. |
| 11 | **Azfar et al. (2025)**, arXiv [2505.01614](https://arxiv.org/abs/2505.01614) | VRP | No TW (flow and subtour constraints as penalties) | QAOA on IBM Quantum System One | Penalty scaling and coefficient normalisation decide feasibility, so normalise our Hamiltonian. |
| 12 | **Kalleri Madhu et al. (2025)**, arXiv [2512.17291](https://arxiv.org/abs/2512.17291) | TSP | No TW | Edge-based + subspace-reduction encoding; 4–6 cities; IQM hardware | Competing qubit-reduction approach for TSP, without TW. |
| 13 | **Glos, Krawiec & Zimborás (2022)**, *npj Quantum Inf.* 8:39, [doi](https://doi.org/10.1038/s41534-022-00546-y) | TSP | No TW | Binary (log) encoding: fewer qubits, deeper circuits | A different trade-off (qubits vs depth); belongs in the discussion. |
| 14 | **Montañez-Barrera et al. (2024)**, *Quantum Sci. Technol.* 9(2), [arXiv 2211.13914](https://arxiv.org/abs/2211.13914) | TSP, bin packing, knapsack | Slack-free "unbalanced penalization" for inequality constraints | Mainly quantum annealing | The main slack-free alternative. Reviewers may ask for it as a baseline. |
| 15 | **Ayodele (2022)**, EvoCOP 2022, LNCS, [doi](https://doi.org/10.1007/978-3-031-04148-8_11) | Permutation QUBOs (TSP etc.) | — | Static rules for penalty weights | Use these rules to set the λ weights, so they aren't hand-tuned. |

## 2. Foundations to cite

- Farhi, Goldstone & Gutmann (2014), QAOA, arXiv:1411.4028.
- Lucas (2014), *Ising formulations of many NP problems*, Front. Phys. 2:5. Source of the n² position encoding for TSP.
- Hadfield et al. (2019), *Quantum Alternating Operator Ansatz*, Algorithms 12(2):34, [arXiv 1709.03489](https://arxiv.org/abs/1709.03489). Mixers that keep states feasible; includes TSP mappings.
- Wang, Rubin, Dominy & Rieffel (2020), *XY mixers*, Phys. Rev. A 101:012320, [arXiv 1904.09314](https://arxiv.org/abs/1904.09314).
- Dumas, Desrosiers, Gélinas & Solomon (1995), *An optimal algorithm for the TSPTW*, Oper. Res. 43(2):367–371, [doi](https://doi.org/10.1287/opre.43.2.367). Uses time windows to rule out states and transitions in advance, which is where our penalty idea comes from classically.
- Soulignac (2025), *Beware of the classical benchmark instances for the TSPTW*, [arXiv 2512.01064](https://arxiv.org/abs/2512.01064). Argues the classical benchmarks are no longer representative, which justifies generating our own instances.
- Liu et al. (2024), *A comparison on constraint encoding methods for QAOA*, [arXiv 2410.04030](https://arxiv.org/abs/2410.04030). Compares penalty, dephasing and Zeno approaches on knapsack.

## 3. Gap matrix

| Work | Time windows | Gate-model QAOA | Feasibility-preserving mixer | No extra qubits for TW | No extra 2-qubit gates for TW | Cumulative arrival time |
|---|---|---|---|---|---|---|
| Rezk & Gora 2026 (#1) | ✓ | ✗ annealing | ✗ | ✓ | ✗ (k→k+2 look-ahead) | ✗ (bound from the depot only, same at every step; see §4) |
| Salehi et al. 2022 (#2) | ✓ | ✗ annealing | ✗ | ✗ slack | ✗ | ✓ exact |
| Holliday et al. 2025 (#4) | ✓ | ✗ CQM hybrid | ✗ | — | — | ✓ |
| Leonidas et al. 2023 (#6) | ✓ | ✓ variational | ? | ? | ? | ? |
| Qian 2023 / do Carmo 2025 (#9, #10) | ✗ | ✓ | ✓ XY | — | — | — |
| **This work** | ✓ | ✓ | ✓ XY | ✓ | ✓ | ✓ sound lower bound |

**Qubit cost comparison**, using #2's protocol where δ = ⌊log₂ 40⌋+1 = 6:

| Customers | E2 (exact, #2): n³−2n²+3n+3nδ | E1 (ours, depot fixed): n² |
|---|---|---|
| 3 | 72 | 9 |
| 4 | 116 | 16 |
| 5 | 180 | 25 |

## 4. Rezk & Gora's time-window penalty: exact formulas

Source: `src/qubo.py`, `VRPProblem.get_qubo`, section "4. TIME WINDOWS (physics-aware, 3 tiers)", lines 240–291, repo commit `a82c9e3` (2026-08-06). The repo has **no LICENSE file**. We can re-implement from these formulas with a citation, but must not copy their code.

**Notation**

- x_{i,j,k} = 1 if vehicle i visits customer j at step k (step 0 = first customer after the depot).
- [e_j, l_j] = time window of j; s_j = service time at j; τ_ab = travel time from a to b; depot 0 opens at e_0.
- P = time-window weight. The static default is 3·10⁶; adaptive calibration sets it to 5·B, where B is the distance scale.

They first compute one earliest-arrival value per customer. It **does not depend on the step k**:

```
ê_j = max(e_j, e_0 + τ_0j)          # "true earliest": drive straight from the depot
```

**A. Depot check** (single variables, step 0 only)
```
if ê_j > l_j:   + P · x_{i,j,0}
```

**B. Consecutive pairs k → k+1** (the same coefficient at every step k)
```
hard:  if ê_j1 + s_j1 + τ_j1j2 > l_j2:
           + P · x_{i,j1,k} x_{i,j2,k+1}                         for all k
soft:  elif l_j1 + s_j1 + τ_j1j2 > l_j2:
           f = min(1, (l_j1 + s_j1 + τ_j1j2 − l_j2) / max(1, l_j1 − e_j1))
           + (P/2) · f · x_{i,j1,k} x_{i,j2,k+1}                 for all k
```

**C. Two-step look-ahead k → k+2** (adds couplings between steps two apart)
```
for j1 ≠ j3: if NO j2 ∉ {j1, j3} satisfies
      a2 = ê_j1 + s_j1 + τ_j1j2 ≤ l_j2   and   max(a2, e_j2) + s_j2 + τ_j2j3 ≤ l_j3
   then  + P · x_{i,j1,k} x_{i,j3,k+2}                           for all k
```

### Our E1 penalty, written to the same level of detail

Single vehicle, depot fixed at position 0, customers at positions k = 1..n. No service times: as in Salehi et al., service time is folded into τ.

**Earliest-arrival bound for every (customer, position) pair.** This is a dynamic program over walks of k hops from the depot:
```
LB(j,1) = max(e_j, e_0 + τ_0j)
LB(j,k) = max(e_j,  min over u ≠ j with LB(u,k−1) ≤ l_u  of  LB(u,k−1) + τ_uj)     k = 2..n
          (a min over no candidates = +∞)
```

**Penalty terms**
```
linear:    if LB(j,k) > l_j:                         + P · x_{j,k}
pairwise:  if max(e_j, LB(u,k) + τ_uj) > l_j:        + P · x_{u,k} x_{j,k+1}     k = 1..n−1
return:    if LB(j,n) + τ_j0 > l_0:                  + P · x_{j,n}
```

**Soundness lemma: no feasible route is ever penalised.** Let π be a feasible route, a_k its arrival time at position k, and j = π(k). By induction:

- a_1 = LB(j,1).
- For k ≥ 2: a_k = max(e_j, a_{k−1} + τ) ≥ max(e_j, LB(π(k−1),k−1) + τ) ≥ LB(j,k). The predecessor π(k−1) is one of the candidates in the min, because LB(π(k−1),k−1) ≤ a_{k−1} ≤ l_{π(k−1)}.

So a_k ≤ l_j implies LB(j,k) ≤ l_j, and no linear term fires. The pairwise and return terms are handled the same way. ∎

Consequence: every feasible route has exactly its plain TSP energy. The ground state is the TSPTW optimum whenever every infeasible route that beats it on distance receives at least one penalty. **Pruning power** measures how often that holds.

### Differences that matter for the paper

| | Rezk & Gora | Ours (E1) |
|---|---|---|
| **Arrival bound** | ê_j: one value per customer, the same at every step | LB(j,k): grows with the position k, so it reflects cumulative travel time |
| **Single-variable term** | Only at step 0, and only when j is unreachable even when driving straight from the depot | Every (j,k) whose cumulative bound already misses the window |
| **Pairwise coefficient** | The same at every k | Depends on k, so strictly more pairs get pruned |
| **Soundness** | The hard tiers are sound only when travel times obey the triangle inequality, because ê_j assumes driving straight from the depot is fastest. The **soft tier is unsound**: it can penalise feasible routes (example below). | Sound for any travel times (lemma above) |
| **New couplings for time windows** | Tier C adds couplings between x_{j1,k} and x_{j3,k+2}. Those pairs sit in different cities and positions two apart, so no other term couples them. In QAOA each one is an extra ZZ term, which costs 2 CNOTs. | None. The pairwise term sits on (k, k+1) pairs that the distance term already couples. |
| **What it catches that we don't** | Tier C catches infeasibility that only shows up two steps later | — (motivates the E1 + look-ahead ablation) |
| **Setting** | CVRPTW, many vehicles, Clarke–Wright savings objective, annealing | TSPTW, gate-model QAOA, X and XY mixers |

**Prior art we must credit (Salehi et al. 2022, Eq. 18 and footnote 1).** They already define an arrival-time lower bound that depends on the position:
```
A_i = min_v c_0v + (sum of the i−1 smallest arc costs)
```
It is the same for every city and does not account for waiting. They use it only to decide how many slack bits to allocate. A footnote adds that it "could even [be improved] into shortest path from depot to any vertex of distance k". Our LB(j,k) carries that idea further in four ways:

- it is specific to each city, not just each position;
- it includes waiting, through max(e_j, ·);
- it only extends prefixes that are themselves on time;
- it is used **as the penalty itself**, which removes the slack bits entirely.

Present LB(j,k) as building on their footnote, not as entirely new.

**Example of the unsound soft tier**

- Depot opens at 0, τ_0,j1 = 2, so ê_j1 = 2. Windows: j1 = [0,10], j2 = [0,5]. τ_j1,j2 = 1.
- Route depot → j1 (arrive 2) → j2 (arrive 3) is **feasible**.
- Hard tier: 2 + 1 = 3 ≤ 5, so it does not fire.
- Soft tier: 10 + 1 = 11 > 5, so f = min(1, 6/10) = 0.6, and the route gets **+0.3·P**.
- Since P = 5B, which outweighs any distance difference, this extra penalty can push the TSPTW optimum out of the ground state.

**Note for writing it up:** their soft tier is a reasonable heuristic for annealing with classical repair. Describe it as "heuristic", not "wrong".

**Note for the plan:** the E1 + look-ahead ablation re-implements Tier C, but with LB(j1,k) in place of ê_j1, so that it stays sound.

## 5. Plan changes from this review

1. **Novelty claim.** Pairwise consecutive penalties alone are not new (#1). Our claim rests on:
   - cumulative, position-dependent earliest-arrival bounds;
   - staying on existing couplings, so no extra 2-qubit gates;
   - gate-model QAOA combined with the XY mixer;
   - a soundness proof.

   Add an ablation that adds #1-style k→k+2 look-ahead terms to E1. It measures how much extra pruning those terms buy and how many 2-qubit gates they cost.
2. **E2 = Salehi et al.'s corrected edge-based QUBO.** It needs 72–180 qubits, so solve it with simulated annealing only.
3. **Instances: #2's protocol** (§4.1):
   - travel costs: random integers 1–10
   - earliest times e: random integers 1–20
   - latest times l: random integers from e to 40
   - 10 instances each for n = 3, 4, 5, every one feasible
   - in half, the TSP optimum differs from the TSPTW optimum

   Then scale the same protocol up to 50 instances per size and add a window-tightness sweep.
4. **Metrics put feasibility first**, because #4 and #11 show feasibility is the bottleneck.
5. **Penalty weights:** set them with Ayodele's static rules (#15) plus Hamiltonian normalisation (#11).
6. **Optional January baseline:** unbalanced penalization (#14), if time allows.
7. **Venue:** ICCS fits (#7). Cite #1 precisely and generously.
8. **First classical result** (`pruning_check.py`; 200 instances per size, Salehi protocol, P = 1000):
   - E1 never penalises a feasible route.
   - It flags 70–81% of infeasible routes at n = 4–7 (82% at n = 3), with zero new couplings. Instances with no late route are left out (`results/pruning_check.txt`).
   - Its ground state is the TSPTW optimum in 83.5%, 59.5% and 46.5% of instances at n = 3, 4, 5. Rezk & Gora's full method reaches 64.5%, 33% and 17.5%, and adds about 8% more ZZ terms.
   - E1+C (sound look-ahead, about 9% more ZZ terms) reaches 100%, 85% and 67%.
   - Under E1, the optimum sits among the 2 lowest-energy routes in 90% of n = 5 instances.

   **Framing:** two sound operating points, E1 (no gate overhead) and E1+C (+9% ZZ terms). Both beat RG-full on these instances.
