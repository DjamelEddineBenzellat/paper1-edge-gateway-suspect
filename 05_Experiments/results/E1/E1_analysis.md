# E1 analysis on fold 10 (plan task M3-03) — 18 Nov 2026

**Fold 10 was opened on 18 Nov 2026, after G1 passed** (approved by the student; `05_Experiments/results/G1/G1_decision.md`).
The opening script `04_Code/fl_system/e1_fold10.py` was committed (`6b18228`) **before** any fold-10 file was read. All numbers: `e1_fold10.json`.

> **WARNING added 18 Nov 2026 (same day): the H3(a) result below is probably a tuning artefact.** The E1 settings (lr 0.01, *C_ex* 2.0) were tuned on P-TA only (DL-36) and given to every design (DL-37). In the DL-35 diagnostic (15 Nov, seeds 0–2, fold 9), P-TG-c at lr 0.1 / *C_ex* 1.0 scored **0.574** tuned F1 (AUROC 0.804), against 0.529 (0.761) at the E1 settings, while P-TA barely changed (0.514 vs 0.525). The price of distrust was then **≈ 6 points**, not 0.5 (`05_Experiments/results/G0/G0a_DL35_diagnostic_results.md`; raised first in `02_Research_Reports/10_publication_plan_reduced_scope_2026-11-18.md` §3).
> Do **not** use the wording "distrust is cheap" (section 5, point 2) until a fair per-design tuning on fold 9 is done (a DL row is needed). The H3(a) result stays as computed for the pre-registered shared point; both results will be reported.
> Note on order: this fold-10 analysis was run before that tuning. Tuned runs will have new sealed fold-10 files, which nobody has seen.
> **Update 19 Nov 2026 (DL-42 done):** with each design at its own tuned point, the price of distrust on fold 10 is **+8.60 points (95% CI +7.50 to +9.69)**; only P-TG-c changed its point. H3(a) at the shared point stays FAIL as recorded. See `05_Experiments/results/DL-42/DL-42_fold10_analysis.md`.

**Main results**
- Fold 10 confirms fold 9: the five DP designs keep **exactly the same order**, and all lose 1.4–1.6 points (the thresholds were tuned on fold 9).
- **H3(a) FAILS** (expected, rule O7): the cost of distrusting the gateway at ε₀ = 8 is +0.51 points (95% CI −0.29 to +1.31), not clearly above 0. Written fallback: **"distrust is cheap at ε₀" becomes a finding.**
- **H5(a), F1 part: EQUIVALENT** (DL-07): P-TA − P-FR = −0.08 points, 90% CI −0.50 to +0.33, inside ±1.0.
- P-TA still does not clearly beat local-only: +0.73 points (95% CI −0.25 to +1.72).

## 1. How the rules O1–O9 were applied
| Rule | What happened |
|---|---|
| O1 (which files) | 90 final files opened: 80 E1 files (all designs except P-PE) and 10 DL-40 P-PE files. The 10 superseded E1 P-PE files were **not** opened. |
| O2 (seal check) | All 90 SHA-256 values matched the seal register. Missing seeds: none. |
| O3 (reading) | Every file belonged to its run, and every single-model file used its run's fold-9 thresholds (`thresholds_tuned_on_this_fold = false`). |
| O4 (G1 on fold 10) | Not triggered: P-TA − P-TG-c = −0.51 points (95% CI −1.31 to +0.29), far above −10. G1 stays PASSED (it is a fold-9 gate). |
| O5 (fold 9 → fold 10 drop) | Expected and seen: every design drops (table in §2). Comparisons between designs are the result, not the level. |
| O6 (H5(a)) | Outcome "equivalent" (§3). |
| O7 (H3(a)) | H3(a) fails; the written fallback applies (§3). |
| O8 (order of the DP designs) | The same order on both folds: P-TG-c > P-FR > P-TA > P-PE > P-TG. |
| O9 (early reads) | None. The seal register (made before G1) matched every file. |

## 2. Fold-10 results (10 seeds, mean ± SD)
| Design | Macro-F1 (DL-36), fold 10 | Fold 9 | Change, points (95% CI) | Macro-AUROC | MI sens. @ 90% spec. | Backdoor rate, no attack |
|---|---|---|---|---|---|---|
| P-TG | 0.503 ± 0.013 | 0.519 | −1.60 (−1.91 to −1.29) | 0.740 | 0.320 | 0.242 |
| P-TG-c | 0.515 ± 0.014 | 0.529 | −1.36 (−1.75 to −0.97) | 0.755 | 0.345 | 0.247 |
| P-PE (DL-40) | 0.508 ± 0.016 | 0.523 | −1.52 (−1.96 to −1.08) | 0.744 | 0.307 | 0.227 |
| P-FR | 0.511 ± 0.013 | 0.526 | −1.44 (−1.83 to −1.04) | 0.749 | 0.317 | 0.269 |
| **P-TA** | **0.510 ± 0.016** | 0.525 | −1.48 (−1.92 to −1.05) | **0.748** | 0.320 | 0.257 |
| Centralized (no DP) | 0.739 ± 0.004 | 0.747 | −0.77 (−1.15 to −0.39) | 0.920 | 0.742 | 0.152 |
| Local-only (no DP) | 0.503 ± 0.005 | 0.518 | −1.55 (−1.60 to −1.49) | 0.722 | 0.313 | — |
| FedAvg (no DP) | 0.655 ± 0.003 | 0.673 | −1.77 (−1.93 to −1.62) | 0.866 | 0.488 | 0.233 |
| HierFAVG (no DP) | 0.655 ± 0.004 | 0.673 | −1.78 (−2.05 to −1.51) | 0.866 | 0.489 | 0.237 |

Local-only on fold 10: the mean over the 210 device models, each with its own fold-9 thresholds (DL-36); its MI sensitivity is the mean over devices.

**Per-class macro-F1 on fold 10**

| Design | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| P-TA | 0.727 | 0.517 | 0.488 | 0.488 | 0.332 |
| P-TG-c | 0.736 | 0.538 | 0.498 | 0.489 | 0.316 |
| P-PE | 0.721 | 0.507 | 0.483 | 0.492 | 0.339 |
| FedAvg | 0.823 | 0.597 | 0.717 | 0.637 | 0.501 |
| Centralized | 0.853 | 0.727 | 0.760 | 0.748 | 0.606 |

## 3. Hypotheses and comparisons (fold 10, paired by seed, points)
| Item | Mean | 95% CI | 90% CI | Result |
|---|---|---|---|---|
| **H3(a)** at ε₀ = 8 (DL-15): P-TG-c − P-TA > 0 with a 95% CI excluding 0 | +0.51 | −0.29 to +1.31 | — | **FAIL** → fallback: "distrust is cheap at ε₀" is a finding |
| **H5(a), F1 part** (DL-07): P-TA − P-FR within ±1.0 (TOST, α = 0.05) | −0.08 | −0.59 to +0.43 | −0.50 to +0.33 | **EQUIVALENT** |
| G1 on fold 10 (reported only): P-TA − P-TG-c | −0.51 | −1.31 to +0.29 | — | Consistent with fold 9 |
| P-TA − local-only (reported only) | +0.73 | −0.25 to +1.72 | — | Not clearly above 0 (as on fold 9) |

- H5 is not finished: its backdoor part comes from E4, and its WAN part from E7. A preview of H5(b) from the byte count: P-FR uploads 45.16 MB per device round, P-TA 1.51 MB, a **30.0×** cut (rule ≥ 10×) [Computed from `metrics.wan_bytes`, not measured; E7 measures it on the hardware].
- H3(b) and H3(c) come from E6.

**All 10 pairs of DP designs (fold 10)**

| Pair | Mean | 95% CI |
|---|---|---|
| P-TG − P-TG-c | −1.22 | −1.95 to −0.48 (excludes 0) |
| P-TG − P-FR | −0.79 | −1.33 to −0.24 (excludes 0) |
| P-TG-c − P-PE | +0.72 | +0.07 to +1.37 (excludes 0, barely) |
| P-TG-c − P-FR | +0.43 | −0.02 to +0.88 |
| P-TG − P-TA | −0.70 | −1.65 to +0.25 |
| P-TG − P-PE | −0.49 | −1.14 to +0.15 |
| P-TG-c − P-TA | +0.51 | −0.29 to +1.31 |
| P-PE − P-FR | −0.29 | −0.85 to +0.26 |
| P-PE − P-TA | −0.21 | −1.11 to +0.69 |
| P-FR − P-TA | +0.08 | −0.43 to +0.59 |

- These 10 comparisons are not corrected for multiple testing; with 10 tests about 0.5 "excludes 0" results are expected by chance. P-TG's lower score (the median of plaintext DP updates) appears on both folds; P-TG-c − P-PE is new on fold 10 and borderline.
- **P-TA differs from no other DP design on either fold.**

## 4. DL-05 data (gateway-view canary AUC, from the fold-9 run files)
| Pair | Mean AUC difference | 95% CI | Within ±0.09? |
|---|---|---|---|
| P-TA − P-TG | −0.007 | −0.030 to +0.017 | yes |
| P-TA − P-PE (DL-40) | −0.002 | −0.018 to +0.015 | yes |

**Warning (E1 finding F3, risk R6):** the canary audit shows no signal even without DP, so "within ±0.09" says nothing about leakage here. DL-05 cannot be applied until the stronger canaries (risk plan R6) give a working positive control.

## 5. What this means for the paper (draft wording for the student)
1. With device-level DP at ε₀ = 8, **where the defences are placed does not change benign accuracy**: the five DP designs lie within about 1.2 points on both folds, and P-TA differs from none of them.
2. ~~Distrusting the gateway is cheap at ε₀ = 8~~ **[withdrawn, see the warning at the top]:** at the shared settings P-TA costs 0.5 points against P-TG-c, but P-TG-c was not tuned; at its pilot settings the gap was about 6 points.
3. **P-TA is equivalent to the flat relay on accuracy** (H5(a), F1 part) while uploading 30× fewer bytes (to be measured in E7).
4. **Limitation:** at ε₀ = 8, none of the DP designs clearly beats local-only training (+0.7 points), although their AUROC is higher (0.748 vs 0.722 for P-TA).
5. The differences between the placements must therefore be shown in E3–E5 (leakage, poisoning, gateway tampering).
