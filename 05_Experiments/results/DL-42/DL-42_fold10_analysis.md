# DL-42 on fold 10: the DP designs at their final configurations (plan task X-5) — 19 Nov 2026

**Opened on 19 Nov 2026**, once, with `04_Code/fl_system/dl42_fold10.py`. All numbers: `dl42_fold10.json`. Fold-9 numbers and the grid: `DL-42_results_note.md`.

## 1. How the opening rules were applied
| Rule | What happened |
|---|---|
| Script committed before opening | `dl42_fold10.py` committed in `77046ff`, before any DL-42 result existed; it also refuses to open if it or the register is uncommitted |
| Seal register committed before opening | `dl42_seal_register.json` (commit `37bbdd1`): 10 files, hashed without reading, all matching the workstation manifest |
| Seals at opening | All 10 new files matched; the kept designs' E1/DL-40 files were re-read through the G1 register (same final configuration, already opened in M3-03) |
| Missing seeds | none |
| Thresholds | Every file was scored with its own run's fold-9 thresholds (checked by the script) |
| Reruns | Each P-TG-c rerun carries the selected point (lr 0.1, *C_ex* 2.0; checked by the script) |

## 2. Fold-10 results at the final configurations (10 seeds)
| Design | Configuration | Fold-9 F1 | **Fold-10 F1** | Fold-10 F1, seeds 3–9 | AUROC | MI sens. @ 90% spec. | Backdoor rate, no attack |
|---|---|---|---|---|---|---|---|
| P-TG | E1: lr 0.01, *C_ex* 2 | 0.519 | **0.503 ± 0.013** | 0.506 | 0.740 | 0.320 | 0.242 |
| P-TG-c | **tuned**: lr 0.1, *C_ex* 2 | 0.613 | **0.596 ± 0.006** | 0.597 | 0.826 | 0.410 | 0.285 |
| P-PE | E1: lr 0.01, *C_ex* 2 | 0.523 | **0.508 ± 0.016** | 0.513 | 0.744 | 0.307 | 0.227 |
| P-FR | E1: lr 0.01, *C_ex* 2 | 0.526 | **0.511 ± 0.013** | 0.515 | 0.749 | 0.317 | 0.269 |
| P-TA | E1: lr 0.01, *C_ex* 2 | 0.525 | **0.510 ± 0.015** | 0.513 | 0.748 | 0.320 | 0.257 |

- P-TG-c loses 1.7 points from fold 9 to fold 10, like the other designs in E1 (1.4–1.6 points): no sign that the tuning over-fitted fold 9.
- The four kept designs show exactly their E1 fold-10 values (`E1_analysis.md`).

## 3. The price of distrust: shared point vs tuned points (P-TG-c − P-TA, paired by seed, points)
| Comparison | Mean | 95% CI | Status |
|---|---|---|---|
| **Shared E1 point** (pre-registered H3(a), ε₀ = 8) | +0.51 | -0.29 to +1.31 | **H3(a) FAIL, as recorded** |
| **Each design at its tuned point**, 10 seeds | **+8.60** | +7.50 to +9.69 | Sensitivity analysis (DL-42 (4)) |
| Tuned, seeds 3–9 only (not used to choose the point) | +8.34 | +6.73 to +9.95 | Check (DL-43 (16)) |

- P-TG-c tuned − P-TG-c shared: +8.08 (95% CI +6.99 to +9.18) points; seeds 3–9: +7.68 (95% CI +6.28 to +9.08).
- **H5(a), F1 part, at the final configurations:** P-TA − P-FR = -0.08, 90% CI -0.50 to +0.33: **equivalent** (unchanged: neither design changed its point).
- P-TA − local-only: +0.73 (95% CI -0.25 to +1.72) points, as in E1.

**All pairs of DP designs at their final configurations (points)**

| Pair | All seeds: mean | 95% CI | Seeds 3–9: mean | 95% CI |
|---|---|---|---|---|
| P-TG - P-TG-c | -9.30 | -10.19 to -8.41 | -9.05 | -10.05 to -8.04 |
| P-TG - P-PE | -0.49 | -1.14 to +0.15 | -0.72 | -1.58 to +0.14 |
| P-TG - P-FR | -0.79 | -1.33 to -0.24 | -0.89 | -1.51 to -0.26 |
| P-TG - P-TA | -0.70 | -1.65 to +0.25 | -0.71 | -1.97 to +0.55 |
| P-TG-c - P-PE | +8.81 | +7.74 to +9.87 | +8.33 | +7.02 to +9.64 |
| P-TG-c - P-FR | +8.51 | +7.58 to +9.45 | +8.16 | +6.89 to +9.43 |
| P-TG-c - P-TA | +8.60 | +7.50 to +9.69 | +8.34 | +6.73 to +9.95 |
| P-PE - P-FR | -0.29 | -0.85 to +0.26 | -0.17 | -0.99 to +0.65 |
| P-PE - P-TA | -0.21 | -1.11 to +0.69 | +0.01 | -1.28 to +1.30 |
| P-FR - P-TA | +0.08 | -0.43 to +0.59 | +0.18 | -0.56 to +0.92 |

## 4. What this means for the paper (draft wording for the student)
1. **At the pre-registered shared settings, distrusting the gateway costs almost nothing (+0.5 points, H3(a) FAIL). When each design gets its own small tuning grid, the trusted-gateway design is about 8.6 points better on the sealed test fold (95% CI 7.5–9.7).** Both results are reported; the design guidance uses the tuned one and says so (DL-42 (4)).
2. **Mechanism:** device-level DP makes each device update almost pure noise, so the four device-DP designs stay near 0.51 whatever the step size; the trusted gateway adds noise once to its pooled sum and can use a large step (fold 9 rises from 0.46 to 0.61 across the grid).
3. Among the four device-DP placements nothing changes: P-TA is still equivalent to P-FR on F1 (H5(a)) and differs from no other device-DP design.
4. The tuned P-TG-c (0.596) closes about 59% of the gap between P-TA (0.510) and non-private FedAvg (0.655, E1 fold 10).

## 5. Limitations
- P-TG-c's selected point is at the edge of the grid and still improving at round 100 (fold 9); the tuned price of distrust is a **lower bound** (DL-42 (2): the grid is not extended).
- The selection used seeds 0–2 on fold 9; the seeds 3–9 check gives the same answer.
- G1 is a fold-9 gate at the shared point and stays PASSED (DL-02, DL-42 (6)). At the tuned points, P-TA − P-TG-c on fold 10 is -8.6 points, inside the −10 limit.
- One dataset (PTB-XL), one ε₀ (8), simulation; the E6 ε line will show how the gap changes with ε.
