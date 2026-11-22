# HFLMLD as implemented for P-PE — differences from the paper

**Source:** Bi et al., "HFLMLD: Enhancing Robustness in Hierarchical FL with Multiple Layer Defenses", IEEE IoT-J 12(24):53396–53409, 2025, DOI 10.1109/JIOT.2025.3617774. Library: `01_Literature/RelatedWork_Library/07_Malicious_Edge_HFL_COMPETITORS/2025_Bi_HFLMLD_Multiple-Layer-Defenses_IoTJ.pdf`. Sections used: §IV-A to §IV-C, Algorithm 1, equations (4)–(14), Table I.
**Rule:** PR §5: P-PE is "implemented from the HFLMLD paper as faithfully as possible; differences are logged". DL-37 fixes the round structure.
**Code:** `04_Code/defences/hflmld.py` (rules), `04_Code/fl_system/placements.py` (`ppe_gateway`, `cloud_ppe`). Unit tests: `TestHFLMLD`, `TestNewPlacementsM201` in `04_Code/tests/test_g0a.py`.
**Created:** 16 Nov 2026 (plan task M2-01).

Legend: **Same** = as in the paper. **Choice** = the paper does not say; we chose. **Different** = deliberately different, with the reason.

| # | Element | Paper | P-PE here | Status |
|---|---|---|---|---|
| 1 | Topology | 10 edge servers × 10 clients (Table I) | 7 gateways × 30 devices (PR §3 item 12) | Different: study design |
| 2 | Edge iterations per global iteration *T_e* | 5 (Table I) | 5 (DL-37) | Same |
| 3 | Global iterations *T_g* | 35–80 (Table I) | 20, so devices train 100 times, like every other placement (T·S = 500 DP-SGD steps, same ε) (DL-37) | Different: equal privacy budget |
| 4 | Local training | SGD, 1–3 local iterations, batch 128–256, η 0.001–0.01 (Table I) | **DP-SGD** on each device (PR §5: "Devices run DP-SGD"): Poisson q = 0.2267, S = 5 steps (one epoch), *C_ex* 2.0, *z* = 3.3545 (ε = 8), lr 0.01 (DL-36 default point) | Different: P-PE definition. E5's "HFLMLD native" row runs without DP (DL-06). |
| 5 | Edge input to detection | Clients' local **model parameters** *w_c* (§IV-A) | Client models *w_c* = edge model + device update | Same |
| 6 | Min–max normalization, eq. (4) | (w − w_min)/(w_max − w_min); the text does not say whether min/max are per client or per coordinate | **Per client**: each client's vector is scaled to [0, 1]. Reason: the paper says it "removes scale discrepancies … across clients". | Choice |
| 7 | PCA dimension, eq. (5) | Not given | 2 components (`random_state` from the run seed, device round and gateway) | Choice |
| 8 | K-means | "K-means clustering"; *k* and the rule that labels a cluster malicious are not given | 2 clusters, 10 initializations, deterministic seed; the **smaller** cluster is malicious; with two equal clusters nobody is flagged | Choice |
| 9 | Suspicion score, eq. (6)–(7) | S ← S + 2V − 1 per edge iteration; logger of length *K* = *T_e*, reset after each global aggregation | Same, reset at every cloud round | Same |
| 10 | Admission to the edge mean | Benign in the current edge iteration **and** S ≤ 0 (§IV-B) | Same | Same |
| 11 | Edge aggregation | Plain mean of the admitted clients' models (Algorithm 1, line 18; FedAvg, §IV-D) | Same | Same |
| 12 | No client admitted | Not discussed | The edge model stays unchanged for that edge iteration | Choice |
| 13 | Cloud pseudo-gradient, eq. (8) | g_e = (w* − w_e)/η, then ln(1 + ‖g_e‖) · g_e/‖g_e‖; η is also the local learning rate (Algorithm 1, lines 8, 24, 28) | E1 (`JOB_20261117_M3-01_e1`): same, η = the learning rate (0.01). From DL-40: a separate cloud step η_c, selected on fold 9; local DP-SGD keeps lr 0.01 | Different (DL-40): with η = 0.01 the cloud moved 4.4% of a plain-mean step and P-PE hardly learned |
| 14 | Cloud weight, eq. (9)–(12) | CR_e = ReLU(cos(w_e, w*) · (1 − euc_e)), with euc_e min–max normalized over the edges | Same. Consequence (also in the paper): the farthest edge model always gets weight 0. If all distances are equal, euc_e = 0 for all (choice). | Same + choice |
| 15 | Global update, eq. (13)–(14) | w* ← w* − η Σ CR_e g_e / Σ CR_e | Same, with η_c in place of η from DL-40 (row 13). If every weight is 0, the global model is kept (choice). | Same + choice; η_c from DL-40 |
| 16 | Data and model | MNIST, Fashion-MNIST, CIFAR-10, CINIC-10; CNNs | PTB-XL, 12-lead ECG, 5 superclasses, multi-label; our 1-D CNN (53,765 parameters) | Different: study design |
| 17 | Transport | Edge sends its model to the cloud | The gateway sends (edge model − global model); the cloud adds the global model back before the HFLMLD rule | Same result |
| 18 | Signed model (our D1, PR §5) | Not part of HFLMLD | Devices check the cloud-signed global model once per cloud round; the edge models of edge iterations 2–5 come from the gateway unsigned, as in HFLMLD (M2-02) | Added by our design |

## Notes for the paper
- Items 6–8, 12, 14 and 15 (choices) should be stated in the method section: "Details not specified in [HFLMLD] were set as follows …".
- With DP-SGD on every device (item 4), the edge clusters noisy models. This is the privacy-constrained setting of contribution C4. Its effect on HFLMLD's detection is part of what E4/E5 measure.
- The HFLMLD paper is the published IEEE IoT-J version (DOI above), so no arXiv reminder is needed.
