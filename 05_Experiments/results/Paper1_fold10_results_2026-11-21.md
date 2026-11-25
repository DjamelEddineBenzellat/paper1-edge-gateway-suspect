# Paper 1 — fold-10 results of E4, E5 and E6 (opened 21 Nov 2026)

**How opened.** Each experiment's sealed fold-10 files were opened once, by its opening script committed before its results existed (`e4_fold10.py` 8b29bfd, `e6_fold10.py` 2f28dec, `e5_fold10.py` a19e965), against seal registers committed before opening (8c70c77, 669f29a). No file was missing; every SHA-256 matched. Outputs: `05_Experiments/results/E4/e4_fold10.json`, `E5/e5_fold10.json`, `E6/e6_fold10.json`. Earlier results (E1, DL-42) are in `E1/` and `DL-42/`.
**Machine.** All runs: <workstation>, GTX 1660 SUPER, torch 2.15.0+cu132, Flower 1.43.0, Opacus 1.6.0, Ray 2.55.1 (`05_Experiments/logs/<JOB>/system.txt`).

## 1. All Paper-1 hypotheses
| Hypothesis | Rule | Result | Verdict |
|---|---|---|---|
| H3(a) price of distrust at ε₀ | gap > 0, CI excludes 0 (shared point) | +0.51 [−0.29, +1.31] (E1) | **FAIL** (recorded); tuned sensitivity: +8.60 [+7.50, +9.69] (DL-42) |
| H3(b) gap shrinks with ε | gap(1) > gap(3) > gap(8) | +11.97 > +9.52 > +8.96 | **Not evaluable** (DL-11: P-TA at ε = 1 is 0.408, within 2 points of the trivial 0.401); the direction is as predicted, descriptive only |
| H3(c) gap independent of *k* | fails if \|gap(k) − gap(3)\| > 1.5 and Holm p < 0.05 | +0.57, +0.51, +0.73 points (k = 1, 2, 5); all CIs include 0 | **Holds** |
| H4 our check detects tampering | each type ≥ 90% AND pooled false alarms ≤ 5% | false alarms 1.9%; scale 100%; drop 89.4%; sign flip 3.5%; injection 3.1% | **FAIL** (fallback: report it and compare with HFLMLD) |
| H5(a) P-TA ≡ P-FR | F1 TOST ±1.0; backdoor TOST ±3 | F1: equivalent (E1); backdoor: — | F1 part **equivalent**; backdoor part **not evaluable** (positive control failed, DL-49) |
| H5(b) traffic | ≥ 10× fewer uplink bytes | 30.0× (byte count) | **Holds** (protocol check, DL-12) |
| DL-24 | P-TG / P-PE lower backdoor than P-TA | — | **Not evaluable** (DL-49) |
| S1 (secondary, DL-49) | P-TA − P-FR F1 drop, TOST ±3 | −4.26 [−9.02, +0.49]; CI90 [−8.12, −0.41] | **Inconclusive** (P-TA's drop is smaller, but not significantly, and not within ±3) |
| S2 (secondary, DL-49) | P-TG, P-PE − P-TA F1 drop, CI below 0 | P-TG −6.79 [−11.38, −2.20], p_Holm 0.009; P-PE −19.09 [−25.30, −12.88], p_Holm 0.0001 | **Both lower than P-TA** |

## 2. E4: device poisoning by label flipping (fold 10, 10 seeds, macro-F1 drop in points vs each design's attack-free run)
| Design | Attack-free F1 | Uniform (≈10% attackers everywhere) | Concentrated (15/30 at one gateway) |
|---|---|---|---|
| P-TG | 0.503 | **−12.9** [8.9, 16.9]; AUROC 0.39 | **−6.7** [5.2, 8.2]; AUROC 0.44 |
| P-TG-c | 0.596 | **−19.5** [15.1, 24.0]; AUROC 0.44 | **−8.5** [7.9, 9.0]; AUROC 0.75 |
| P-PE (HFLMLD) | 0.508 | −0.6 [0.1, 1.2]; AUROC 0.74 | 0.0 [−0.4, 0.4]; AUROC 0.74 |
| P-FR | 0.511 | **−24.0** [18.3, 29.7]; AUROC 0.35 | **−9.3** [7.3, 11.4]; AUROC 0.43 |
| **P-TA** | 0.512 | **−19.7** [13.4, 26.0]; AUROC 0.39 | **−0.2** [−0.2, 0.6]; AUROC 0.75 |

Positive control passed (uniform −29.2, concentrated −22.2 points on undefended FedAvg). τ in P-TA dropped 21% of the uniform poisoned group sums and 60% of the concentrated ones; P-FR has no τ.

**Reading.** (1) **Concentrated attack:** only the two-tier P-TA and P-PE contain it; P-TA's cloud check removes the overrun gateway (see the DL-48 smoke: unexplained drops flag it). (2) **Uniform attack:** every design that relies on medians over DP-noised updates loses 13–24 points, including P-TA; P-PE, which inspects each plaintext device update, loses 0.6. Under device DP the honest updates are mostly noise, so a coordinated minority in every group shifts coordinate-wise medians consistently (a known weakness of median rules when honest updates are widely spread; mechanism = interpretation, not tested here). (3) Hiding individual updates (secure aggregation in P-TA, P-FR) removes exactly the per-device inspection that P-PE uses: a measured privacy–robustness trade-off.
**Correction of a preview.** On 20 Nov I wrote, from fold 9, that P-TG-c "holds" under the concentrated attack. That compared it with the wrong level: its attack-free F1 is higher (0.596), and on fold 10 it loses 8.5 points.
**Correction (22 Nov 2026; the text above is left as approved).** Two values in this section were rounded twice (first to two decimals, then to one). Rounded once from the exact values in `E4/e4_fold10.json`: P-TG-c concentrated is **−8.4** (exact 8.446), not −8.5, also in the line above ("loses 8.5 points" → 8.4); P-PE concentrated is 0.0 [−0.4, **0.3**] (exact upper limit 0.348), not 0.4. P-TG uniform [8.9, 16.9] is correct here; the paper had [9.0, 16.9] and is now fixed. No reading or verdict changes.

## 3. E5: one tampering gateway out of 7 (fold 10, 10 seeds; mixed partition)
| Detector | False alarms (pooled) | Scale ×2 | Drop 50% | Sign flip | Injection | Adaptive (reported only) |
|---|---|---|---|---|---|---|
| **Ours** (size-aware, DL-44 + DL-48) | **1.9%** [1.8, 2.0] | **100%** | **89.4%** [87.3, 91.2] | 3.5% | 3.1% | 0%; largest undetected F1 drop 1.8 points |
| Ours with the old DL-10 normalization (same runs) | 15.1% | 52.2% | 40.4% | 19.7% | 19.6% | — |
| HFLMLD native (no DP) | 12.6% | 99.5% | 18.5% | 8.5% | 7.0% | 0%; 2.4 points |
| HFLMLD cloud rule on P-TA | 13.3% | 46.7% | 31.9% | 15.0% | 15.0% | 0%; 2.8 points |

Accuracy effect of the tampering on our runs (F1 drop vs the run without tampering): scale −0.15, drop −0.04, sign flip +0.79, injection −0.09, adaptive +0.63 points. False alarms per gateway (ours): 0.1–0.2% at the skewed gateways 0–2 (DL-48 works), 1.4–4.7% at gateways 3–6.

**Reading.** (1) H4 fails: a distance check cannot see a sign flip or a same-norm injection when every gateway output is mostly DP noise (DL-10 already warned of this); the drop is detected in 89.4% of rounds, 0.6 points under the bar. (2) The size-aware check is still far better than HFLMLD on every type and on false alarms (1.9% vs 12.6–13.3%); HFLMLD's 8–15% "detection" of sign flips is close to its built-in rate of flagging the farthest edge every round. (3) With one tampering gateway in seven, the cloud median keeps every attack's accuracy damage under about 2 points, detected or not: the check matters for attribution more than for accuracy at *b* = 1. (4) DL-44 and DL-48 together cut false alarms from 15.1% (DL-10 rule) to 1.9%.
Harmless misses (DL-10 rule, per run: F1 within 2 points AND backdoor rate within 3 points): sign flip 389 of 965, injection 397 of 969. The backdoor-rate part varies by several points even without attack, so many misses count as "not harmless" although their F1 change is small.
**Correction (added 21 Nov, after drafting Section V).** Reading (4) credits DL-44 and DL-48 together. That is not right: the DL-10 flags in this table use the same DL-48 rate alert as ours (`placements.py`, `cloud_pta`), so the drop from 15.1% to 1.9% comes from the size-aware distance (DL-44) alone. DL-48's effect is shown only indirectly: 0.1–0.2% false alarms at gateways 0–2, where counting every τ drop was predicted to alert in 85–93% of rounds (DL-48, "Why"). No number in the tables changes.

## 4. E6: privacy level and group size (fold 10, seeds 0–2)
| Line | P-TA F1 | P-TG-c F1 | Gap (P-TG-c − P-TA) |
|---|---|---|---|
| ε = 1 | 0.408 (trivial 0.401) | 0.528 | +12.0 |
| ε = 3 | 0.477 | 0.573 | +9.5 |
| ε = 8 | 0.506 | 0.595 | +9.0 |

| *k* | 1 | 2 | 3 | 5 |
|---|---|---|---|---|
| P-TA F1 | 0.500 | 0.501 | 0.506 | 0.498 |
| P-FR F1 | 0.498 | 0.505 | 0.502 | 0.506 |

Group size has no measurable effect on accuracy (H3(c) holds). At ε = 1 device-DP P-TA is at the trivial level.

## 5. What this means for Paper 1 (honest)
- The frozen plan's **favourable** hypotheses mostly did not hold: H3(a) fails at the shared point, H3(b) is not evaluable, H4 fails, the backdoor parts are not evaluable. H3(c), H5(a)-F1 and H5(b) hold.
- The results still form a coherent, measured answer to the research question, but as a **trade-off map**, not as "P-TA is best": trusting the gateway buys about 9 F1 points (H3(a) tuned, E6); group size does not matter (H3(c)); hiding device updates protects privacy and costs robustness against spread-out poisoning (E4 uniform), while the two-tier check contains a concentrated attack (E4) and attributes scaling and dropping gateways with few false alarms (E5), but not sign flips or injections.
- The pre-registered fallbacks apply: H4 → "report it and compare with HFLMLD; the contribution shifts to the evaluation"; H5 → the paper must not claim P-TA is as robust as alternatives.
- Single dataset, single machine, ε₀ = 8; E6 has 3 seeds.
