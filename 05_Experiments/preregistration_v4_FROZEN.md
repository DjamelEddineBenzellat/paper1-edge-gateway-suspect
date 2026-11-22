# Pre-registration v4 — PhD Paper 1 experiments

**Status:** FINAL and FROZEN
**Date:** 11 Nov 2026
**Approved by:** the student (no supervisor meeting)
**Goes with:** `CONTRIBUTION_Paper1_EdgeGatewaySuspect_v4_FINAL.md`. The research question (§1), hypotheses (§4), experiment tables (§6), contributions (§8) and timeline with freeze rules (§9) are identical in both files.
**Replaces:** the draft `05_Experiments/preregistration.md`, kept unchanged for the record.
**Labels:** [Set by student, 11 Nov 2026] = a final value; [Computed] = arithmetic for this plan; [Rule] = the value is fixed later by the rule written here.

## 0. Changes from the draft pre-registration
| Change | Review IDs (`02_Research_Reports/05_v3_consistency_review_2026-11-11.md`) |
|---|---|
| FROZEN; approved by the student; no supervisor steps | D02, D03 |
| Option-A items removed: *h* in D1, the *h* sweep, the *k* × *h* slice | A07, A08 |
| H1 rule changed to isolation rate + DP audit; G2 stops only on a bug | C01, C02 |
| H2 prediction rule added; H3 rule matched to P5 under D6-B | C06, A06 |
| G1 has one numeric rule; gates placed in the agreed order | C03, C04, C08 |
| BRPPFL removed from G0; searches S1–S4 | D04 |
| All placeholders set (§3); run total recomputed | D05, C09, C17, C18 |
| Mixed gateways (E5); MedMNIST in E2; canaries per run | C07, C10, C19 |

---

## 1. Research question and decisions
**Research question:** In a device → gateway → cloud federated learning system for health monitoring, **where should privacy and poisoning defences be placed** when the gateway may be curious or malicious, and what does each placement cost and buy in privacy, robustness, accuracy, traffic, latency and energy?

| ID | Final decision |
|---|---|
| D1 | Signed model + cloud-certified groups + **t = k** (no dropouts, no retry) |
| D2 | Placement study: four placements, five configurations |
| D3 | Ward setting (PTB-XL) is the main setting; the home setting is optional (E9) |
| D4 | Core: E1–E7. Optional: E8, E9, FLamby, FedProx, SCAFFOLD, FedAdam |
| D5 | The fallback framing for H5 is accepted in advance |
| D6 | **Option B:** patient-level DP-SGD on each device (Opacus) |

---

## 2. Gate G0 (month 1, before any training)
| Check | How | Pass rule |
|---|---|---|
| G0-a DP feasibility | Opacus Rényi-DP accountant for ε = 3, δ = 10⁻⁵, sampling rate ≈ 0.19, about 600 steps (100 rounds × 6 steps) [Computed] | Noise multiplier *z* ≤ 1.5 |
| G0-b Literature | Searches S1–S4 (§11), logged in `01_Literature/search_log.md` | No paper found that compares these placements under a suspect gateway |
| G0-c Compute | Time one default P-TA run; multiply by the run plan (§6.1) | Projected E1–E6 compute ≤ 3,000 machine-hours |
| G0-d MedMNIST subset | Pick one 2D, 28×28 subset whose source licence allows publishing reconstructed images (MedMNIST v2 licence table) | Subset and licence recorded |

If G0-a or G0-c fails, use the allowed adjustment (§10).

---

## 3. Values set by the student
| # | Item | Value | Reason | Mark |
|---|---|---|---|---|
| 1 | Patient-level ε | 3 | Middle of the planned values {1, 3, 8}; one allowed change to 8 | [Set by student, 11 Nov 2026] |
| 2 | δ | 10⁻⁵ | Below 1/(about 17,470 training records) ≈ 5.7 × 10⁻⁵ [Computed] | [Set by student, 11 Nov 2026] |
| 3 | Rounds *T* | 100 | Enough for a small CNN; the composition is checked at G0 | [Set by student, 11 Nov 2026] |
| 4 | Local training | 1 epoch per round, SGD | Keeps devices light; about 6 steps per round [Computed] | [Set by student, 11 Nov 2026] |
| 5 | DP-SGD batch size | 16 | About 83 records per device gives a sampling rate of about 0.19 [Computed] | [Set by student, 11 Nov 2026] |
| 6 | Learning rate | Best of {0.01, 0.05, 0.1} on fold 9 | Small grid; validation fold only | [Set by student, 11 Nov 2026] |
| 7 | Per-example clip *C_ex* | Best of {0.5, 1.0, 2.0} on fold 9, by benign macro-F1 at ε = 3 | Small grid; validation fold only | [Rule], set in months 2–3 before E1 |
| 8 | Noise multiplier *z* | From the accountant (G0-a) | Cannot be known before G0 | [Rule], set at G0 |
| 9 | Cut-off for *z* | ≤ 1.5 | The student's cut-off: above it, about 83 records per device are expected to be too noisy to learn from; G1 checks the real accuracy | [Set by student, 11 Nov 2026] |
| 10 | ECG input | 12 leads at 100 Hz | Ward monitors record clinical 12-lead ECG. 100 Hz is the downsampled release of PTB-XL (Wagner et al. 2020) and keeps the model small. | [Set by student, 11 Nov 2026] |
| 11 | Labels | 5 diagnostic superclasses, multi-label | As planned | [Set by student, 11 Nov 2026] |
| 12 | Partition | 7 gateways × 30 devices; a patient is never split | As planned | [Set by student, 11 Nov 2026] |
| 13 | Dirichlet α | 0.5 (main); 0.1 for 3 of the 7 gateways in E5 | Moderate non-IID; stronger for the mixed-gateway test | [Set by student, 11 Nov 2026] |
| 14 | Model | 4 conv blocks (12→32→32→64→64 channels, kernel 7, GroupNorm with 8 groups, ReLU, max-pool 2) + global average pooling + linear 64→5; about 53.8k parameters [Computed] | Small, for DP-SGD (Guerraoui et al. 2021) and the Pi. GroupNorm, because DP-SGD needs per-example gradients. | [Set by student, 11 Nov 2026] |
| 15 | Update clip *C* | 99th percentile of benign update norms on fold 9 | Bounds each update for secure aggregation | [Rule], set in months 2–3 |
| 16 | Quantization | 32-bit field elements, 16-bit fixed point | Room for sums of up to *k* = 5; about 210 KiB per update [Computed] | [Set by student, 11 Nov 2026] |
| 17 | P-TG-c noise | Devices clip per example and add no noise. The gateway adds Gaussian noise once per round to its devices' summed update, calibrated by the accountant to the same patient-level ε. | The equal-ε reference for the price of distrust | [Rule], set at G0 |
| 18 | Group size *k* | 3 | Safe at *f* = 0.1 by P1 | [Set by student, 11 Nov 2026] |
| 19 | Dropout threshold *t* | = *k* | D1 | [Set by student, 11 Nov 2026] |
| 20 | τ | 99th percentile of benign group-sum norms on fold 9 | About 1% of benign sums are dropped by design | [Rule], set in months 2–3 |
| 21 | Gateway-check factor | Flag if distance > median + 3·(1.4826·MAD) | About 3 standard deviations; margin for different gateways under the 5% false-alarm target | [Set by student, 11 Nov 2026] |
| 22 | Reputation decay | 0.9 | A memory of about 10 rounds | [Set by student, 11 Nov 2026] |
| 23 | Abort-rate alert | > 10% of a gateway's groups in a round | Beyond this, lost updates slow training; it also flags selective aborts | [Set by student, 11 Nov 2026] |
| 24 | Flagged gateway | Excluded from the cloud median in that round | Simplest response | [Set by student, 11 Nov 2026] |
| 25 | Default attack | *f* = 0.1 uniform, backdoor, attacker level K1, *b* = 0 | As planned | [Set by student, 11 Nov 2026] |
| 26 | Backdoor trigger | Rectangular pulse, 0.2 s, +0.5 mV, on lead II, starting at t = 1.0 s; target label NORM only | Simple to inject; "hide disease as normal" is the clinically harmful case | [Set by student, 11 Nov 2026] |
| 27 | Concentrated attacker | One gateway with *M_g* = 15 of 30 (overall *f* ≈ 0.07 [Computed]) | Enough to overrun that gateway at *k* = 3 (P1) | [Set by student, 11 Nov 2026] |
| 28 | Tampering types | Scaling ×2; dropping 50% of groups; injection (the update replaced by a random vector of the same norm); sign flip | As in H4 | [Set by student, 11 Nov 2026] |
| 29 | Adaptive gateway | Keeps its distance at 0.95 × the current threshold | Stays just under the check | [Set by student, 11 Nov 2026] |
| 30 | Canaries | 100 per run: 50 inserted at random into honest devices, 50 held out (about 0.29% of training records [Computed]) | A per-run membership test that also works with 3 seeds | [Set by student, 11 Nov 2026] |
| 31 | Clinical operating point | MI-superclass sensitivity at 90% specificity | A fixed false-positive rate of 10% | [Set by student, 11 Nov 2026] |
| 32 | ECG "recognizable" | Mean per-lead Pearson *r* ≥ 0.8 | Strong linear agreement | [Set by student, 11 Nov 2026] |
| 33 | MedMNIST subset | Licence rule (G0-d) | Some MedMNIST sources are CC-BY-NC, redistributed with permission (MedMNIST v2 paper) | [Rule], set at G0 |
| 34 | G0-c budget | ≤ 3,000 machine-hours for E1–E6 (5 months × 25 days × 24 h [Computed]) | Leaves 2 of the 7 working months as margin | [Set by student, 11 Nov 2026] |
| 35 | G1 threshold | P-TA benign macro-F1 ≥ P-TG-c − 10 points | A larger loss would make the protected model clinically uninteresting | [Set by student, 11 Nov 2026] |
| 36 | MedMNIST model and topology (E2) | Small CNN (2 conv layers with 32 and 64 channels + linear layer), on the same 7 × 30 topology | Inversion needs a small model; E2 needs no new topology | [Set by student, 11 Nov 2026] |

**Reasons for the hypothesis thresholds** (all [Set by student, 11 Nov 2026]):
- **H1:**
  - ≥ 90% isolation shows that the attack works reliably, since secure aggregation promises 0%;
  - any isolation with protections is a bug;
  - an ε_emp above ε is a DP violation.
- **H2:**
  - 80% agreement allows a few borderline configurations to disagree;
  - a 10% backdoor success leaves room for the clean model's own errors on triggered records;
  - 2 points is the smallest macro-F1 change treated as meaningful.
- **H3:** 1.5 points is the H5 margin plus seed noise at 3 seeds.
- **H4:** an operator cannot use a check that misses more than 1 attack in 10, or raises more than 1 false alarm in 20 honest rounds.
- **H5:**
  - ±1.0 point and ±3 points are the smallest differences treated as meaningful;
  - the 10× traffic cut is a third of the expected 30× [Computed], so the claim survives protocol overhead.
- **H6:** 10 ms is one sample period at 100 Hz.

---

## 4. Hypotheses (identical in v4 §5)
| ID | Hypothesis | Pass rule (all parts must hold) | Experiment | If it fails |
|---|---|---|---|---|
| H1 | Without the signed model and certified groups, a malicious gateway can isolate one device's update. With them and t = k, it cannot, and the measured leakage stays within the DP guarantee. | (a) Without protections (no model signature, no group certificates, dropouts allowed down to t = 2), at least one gateway attack isolates the update of **≥ 90%** of targeted devices (5 seeds). (b) With protections (P-TA, P-FR): isolation rate **= 0**. (c) The lower 95% bound of ε_emp from a canary audit on individual device updates (the worst-case view) is **≤ ε = 3** (10 seeds, pooled). Membership AUC and ECG *r* are reported. | E3 | (a): strengthen the attack, never weaken the defences. (b) or (c): **G2 stop**, a protocol or DP-accounting bug; fix it first. |
| H2 | Poisoning damage follows P1–P3. | **≥ 80%** of the 28 configurations below agree with their prediction. Predicted safe: backdoor success **≤ 10%** and macro-F1 drop **≤ 2 points** against the same setting with *f* = 0. Predicted unsafe: backdoor success > 10% or drop > 2 points. | E4, E6 | Report where P1–P3 fail as a limitation; keep the measured map. |
| H3 | At equal ε, distrusting the gateway costs accuracy; the cost shrinks as ε grows and does not depend on *k* (P5). | (a) The gap (P-TG-c − P-TA) in macro-F1 at ε = 3 is **> 0**, with a paired 95% CI excluding 0 (10 seeds). (b) Mean gap at ε = 1 > ε = 3 > ε = 8. (c) For *k* ∈ {1, 2, 5}: \|gap(*k*) − gap(3)\| **≤ 1.5 points**. | E1, E6 | (a): "distrust is cheap at ε = 3" becomes a finding. (b) or (c): P5 mis-predicts; report it and discuss why. |
| H4 | The cloud check detects gateway tampering with few false alarms, even when gateways differ. | Non-adaptive tampering (scaling ×2, dropping 50% of groups, injection, sign flip), with mixed gateways: detection **≥ 90%** of tampering gateway-rounds and false alarms **≤ 5%** of honest gateway-rounds (Wilson 95% CIs reported). Adaptive gateway: the largest undetected accuracy drop and backdoor success are reported (no pass/fail). | E5 | Report it, and compare with the HFLMLD cloud rule. The contribution shifts to the evaluation. |
| H5 | The tier-aware placement matches the flat relay on accuracy and robustness, while cutting WAN traffic by about *N_g*. The flat relay leaks less to the gateway (empirically). | (a) TOST equivalence (α = 0.05, 10 seeds): benign macro-F1 within **±1.0 point** (E1); backdoor success within **±3 points** (E4, default attack). (b) WAN bytes per round cut **≥ 10×** (about 30× expected [Computed]). Gateway-view leakage is reported. | E1, E4, E7 | **Fallback framing:** "A suspect gateway should relay, not aggregate; aggregate at the cloud with certified groups." |
| H6 | The gateway-side cost of P-TA fits a low-cost gateway. | P-TA adds **≤ 10%** to the gateway's round time against plain aggregation, and raises the P95 forwarding delay of a concurrent 100 Hz sensor stream by **≤ 10 ms** (30 repetitions). Energy per round is reported. | E7 | Report the cost; recommend a stronger gateway class. |

**H2 prediction rule** [Computed from P1–P3]:
- A gateway is **overrun** if its expected malicious count reaches half its groups. For uniform attackers that means *f* ≥ 1/(2*k*); for the concentrated gateway, *M_g* ≥ *N_g*/(2*k*).
- **P-TA** is predicted safe if (overrun gateways + *b*) < *G*/2 = 3.5.
- **P-FR** is predicted safe if the total expected malicious count is below half of all groups.

**The 28 configurations:**

| Group | Configurations | Predicted safe | Predicted unsafe |
|---|---|---|---|
| *k* × *f* slice (P-TA and P-FR each; *f* > 0) | 24 | *k* = 1: all *f*; *k* = 2: *f* = 0.1, 0.2; *k* = 3: *f* = 0.1 (12 in total) | *k* = 2: *f* = 0.3; *k* = 3: *f* = 0.2, 0.3; *k* = 5: all *f* (12 in total) |
| Concentrated attacker (P-TA, P-FR; *k* = 3) | 2 | 2 | 0 |
| Tampering gateways *b* = 1, 2 (P-TA; *k* = 3, *f* = 0.1) | 2 | 2 | 0 |

---

## 5. Placement definitions
| Code | Definition |
|---|---|
| P-TG | The gateway sees plaintext DP-SGD updates and takes their coordinate-wise median; the cloud takes the median over gateways. |
| P-TG-c | Devices clip per example and add no noise; the trusted gateway adds central Gaussian noise (§3, item 17); medians at both levels. |
| P-PE | HFLMLD-style: the edge runs dimensionality-reduction detection with suspicion scores; the cloud runs normalized weighted aggregation. Implemented from the HFLMLD paper (IEEE IoT-J 2025) as faithfully as possible; differences are logged. Devices run DP-SGD. |
| P-FR | The gateway relays masked messages only. The cloud draws certified groups of *k* inside each gateway's devices; secure aggregation runs per group; the cloud takes the median over all (group sum / *k*). The abort-rate alert applies. |
| P-TA | v4 §3: certified groups, t = k, group-sum norm check and median at the gateway; median and gateway check at the cloud. |

The signed model applies to all five configurations. Certified groups and t = k apply to P-FR and P-TA.

---

## 6. Experiments, operating point and sweeps (tables identical to v4 §6; this file adds the run breakdown)

### 6.1 Order, gates, seeds and runs
| Step | What | Hypotheses | Seeds | Runs | Month |
|---|---|---|---|---|---|
| **G0** | DP accountant; searches S1–S4; time one default run; choose the MedMNIST subset. **Pass:** *z* ≤ 1.5; no paper found that compares these placements under a suspect gateway; projected E1–E6 compute ≤ 3,000 machine-hours | – | – | – | 1 |
| E1 | Benign accuracy: P-TG, P-TG-c, P-PE, P-FR, P-TA + centralized, local-only, FedAvg, HierFAVG | H3, H5 | 10 | 90 | 2–3 |
| **G1** | **Pass:** P-TA benign macro-F1 ≥ P-TG-c − 10 points. Otherwise: one allowed adjustment, then re-run E1 | – | – | – | end of 3 |
| E2 | Attacks on non-private HierFAVG (the problem is real): ECG inversion, MedMNIST inversion, membership inference from the gateway and cloud views | descriptive | 5 | 20 | 4 |
| E4 | Device poisoning at the default point: 5 configurations × 4 attacks (uniform backdoor, adaptive attack on the median, label flipping, concentrated backdoor); plus K0/K2 for the backdoor and the adaptive attack on P-TA and P-FR | H2, H5 | 10 (K0/K2: 3) | 224 | 4–5 |
| E3 | Malicious-gateway attacks with and without protections; canary audit | H1 | 5 (a), 10 (b, c) | 65 | 6 |
| **G2** | **Pass:** isolation = 0 with protections, and the lower 95% bound of ε_emp ≤ ε. Otherwise: stop and fix the bug | – | – | – | 6 |
| E5 | Gateway tampering (4 types + adaptive) × 3 detectors (ours on P-TA; HFLMLD native on P-PE; HFLMLD cloud rule on P-TA group sums); mixed gateways | H4 | 10 | 150 | 7 |
| E6 | Sweeps (§6.3) | H2, H3 | 3 | 102 | 8 |
| E7 | Cost on Raspberry Pi with INA219: 9 mechanisms × 30 repetitions | H5, H6 | – | 270 measurements | 9 |
| E8 / E9 | Optional: ablations (MAD vs fixed threshold, norm check on/off, re-grouping on/off, RoFL-style proofs) / home setting | – | 3 | not counted | 10 |
| **Total** | | | | **651 runs + 270 measurements** [Computed] | |

**How the runs add up** [Computed]:
- E1: 9 configurations × 10 seeds = 90.
- E2: 4 attacks × 5 seeds = 20.
- E4: 5 × 4 × 10 = 200, plus 2 attacks × 2 knowledge levels × 2 placements × 3 seeds = 24.
- E3: 5 attacks × 5 seeds (a) = 25, plus 2 placements × 10 seeds (b, c) = 20, plus 5 attacks × 2 placements × 2 seeds (refusal checks) = 20.
- E5: 5 × 3 × 10 = 150.
- E6: 34 configurations × 3 seeds = 102.
- E7 mechanisms:
  1. idle baseline;
  2. plain aggregation;
  3. secure-aggregation server side;
  4. norm check;
  5. median over group sums;
  6. certificate checks;
  7. relaying masked messages (P-FR);
  8. forwarding a concurrent sensor stream under load;
  9. a device-proxy round (DP-SGD + secure-aggregation client).

### 6.2 Default operating point
| Item | Value |
|---|---|
| Data | PTB-XL, 12 leads at 100 Hz, 5 diagnostic superclasses (multi-label); folds 1–8 train, 9 validation, 10 test (used once per final configuration) [Set by student, 11 Nov 2026] |
| Partition | *G* = 7 gateways × *N_g* = 30 devices (*N* = 210), about 83 records per device [Computed]; a patient is never split; Dirichlet α = 0.5 [Set by student, 11 Nov 2026] |
| Model | 1-D CNN: 4 blocks (12→32→32→64→64 channels, kernel 7, GroupNorm, ReLU, max-pool) + global average pooling + linear layer; about 53.8k parameters [Computed] [Set by student, 11 Nov 2026] |
| Training | *T* = 100 rounds; 1 local epoch; SGD; learning rate from {0.01, 0.05, 0.1} on fold 9 [Set by student, 11 Nov 2026] |
| Privacy (D6-B) | Patient-level ε = 3, δ = 10⁻⁵; batch 16 (sampling rate ≈ 0.19 [Computed]); *z* from the accountant at G0; *C_ex* from {0.5, 1.0, 2.0} on fold 9 [Set by student, 11 Nov 2026] |
| Grouping | *k* = 3, t = k, re-drawn every round |
| Default attack | *f* = 0.1 uniform, backdoor, attacker level K1, *b* = 0 |
| Filters | Median at both levels; τ = 99th percentile of benign group-sum norms (fold 9); MAD factor 3; abort alert 10%; reputation decay 0.9 [Set by student, 11 Nov 2026] |

### 6.3 Sweeps (option-A items removed: no *h* sweep, no *k* × *h* slice)
| Line | Placements | Values | Attack | New configurations |
|---|---|---|---|---|
| *k* × *f* slice | P-TA, P-FR | *k* ∈ {1, 2, 3, 5} × *f* ∈ {0, 0.1, 0.2, 0.3} | backdoor (none at *f* = 0) | 28 (32 minus 4 already run in E1/E4) |
| ε line | P-TA, P-TG-c | ε ∈ {1, 8} (ε = 3 comes from E1) | none (*f* = 0) | 4 |
| *b* line | P-TA | *b* ∈ {1, 2} (scaling ×2) | *f* = 0.1 backdoor | 2 |

34 configurations × 3 seeds = 102 runs. *k* = 1 means no secure aggregation: each group is one device. Every E6 run also records the gateway-view canary AUC (descriptive), which measures the *k* trade-off.

---

## 7. Attacks, metrics and statistics
**Attacks:**
- Attacker levels:
  - K0: the protocol only;
  - **K1 (main):** + the defence and its parameters;
  - K2: + estimates of benign statistics.
- Device attacks:
  - backdoor (§3, item 26);
  - label flipping;
  - an adaptive attack on the median (Fang et al. 2020);
  - low-norm poisoning tuned to stay under τ;
  - FLPoison where possible, with its commit pinned.
- Gateway attacks:
  - model inconsistency (Pasquini 2022), imprint module (Fowl 2022), trap weights (Boenisch 2023);
  - Sybil devices, selective abort;
  - tampering (§3, item 28), and the adaptive gateway (§3, item 29).
- Privacy attacks:
  - gradient inversion with `breaching` (pinned commit);
  - membership inference with ML Privacy Meter (pinned commit).
- **Isolation** (H1): the gateway obtains one device's update, alone, in a round. It is measured as the share of targeted devices for which that happens.

**Metrics:**
- **Utility:** macro-F1 (main), AUROC, per-class recall, MI sensitivity at 90% specificity.
- **Privacy:** patient-level ε and δ; canary membership AUC and the canary-audit ε_emp (lower 95% bound, at δ = 10⁻⁵); ECG *r* and normalized DTW; PSNR/SSIM (MedMNIST, E2).
- **Robustness:** backdoor success on triggered test records; macro-F1 drop against *f* = 0; detection and false-alarm rates per gateway-round.
- **Cost:** WAN bytes per round; round time broken down (training, secure aggregation, filtering, transfer); P95 forwarding delay; peak RAM; joules per round (INA219, minus the idle baseline).

**Statistics:**
- Main comparisons: **10 seeds**, paired by seed. Report the mean difference and its 95% CI (paired *t*).
- H5: TOST.
- Proportions: Wilson 95% CIs.
- Sweeps: 3 seeds, mean ± SD, descriptive.
- Holm correction within one hypothesis.
- Why 10 seeds: with 5 seeds, a two-sided Wilcoxon test can never reach p < 0.05.

---

## 8. Contributions, novelty and claims to avoid (identical in v4 §7)
| # | Contribution |
|---|---|
| C1 | A threat model for a **suspect gateway** in device–gateway–cloud health FL: curious and malicious cases, Sybil groups, selective abort, gateway–device collusion, with ward and home deployment scenarios. |
| C2 | **Hardened** flat-relay and tier-aware placements: signed model, certified groups, t = k, group-sum norm check, cloud gateway check. Each part is standard engineering; the contribution is making them work together against a suspect gateway. |
| C3 | **Five propositions** (P1–P5) that predict when each placement is safe and what distrust costs, tested empirically. |
| C4 | A **same-run comparison** of four placements (five configurations) under privacy attacks (including malicious-server and Sybil attacks), device poisoning (uniform and concentrated) and gateway tampering, against HFLMLD in its native setting and in a privacy-constrained setting. [To confirm with S1–S4] |
| C5 | **Measured cost** per placement and per mechanism on Raspberry Pi gateways (INA219): WAN traffic, round-latency breakdown, forwarding delay, then energy; plus a device proxy. |

**Novelty sentence:**
> Prior work treats the middle tier of hierarchical FL as malicious for robustness or integrity only (HFLMLD, RaSA, VHFL, Fed-MS), or for privacy and robustness together using homomorphic encryption or verifiable computing (PPBR, RaSA). Other hierarchical defences assume honest aggregators (SHIELD). In these works, the edge either inspects individual client updates (HFLMLD, SHIELD, VHFL, Fed-MS) or relies on heavy cryptography (PPBR, RaSA). Flat systems combine secure aggregation with robust filtering (SHARE, BREA, RoFL) or DP with Byzantine robustness (Ma et al.), but with a single server. **To the best of our knowledge**, no prior work compares, in the same runs, where privacy and poisoning defences should be placed when the gateway of a device–gateway–cloud health FL system is itself suspect; predicts with simple propositions when each placement is safe and what distrust costs; and measures the traffic, latency and energy of each placement on low-cost gateways.

**Claims to avoid:**
- "First to treat the edge as malicious" (PPBR, RaSA, VHFL, Fed-MS, HFLMLD).
- "First cloud-side defence without clean data" (HFLMLD).
- "First to consider collusion" (HFLMLD, PPBR).
- "A new aggregation or detection rule" (median and MAD are standard).
- "Certified groups, PKI, signed models or the norm check are new" (standard engineering).
- "Group size is a new idea" (SHARE; Li et al., KBS 2023 [Unverified]).
- "The price of distrust is new in general" (the local-vs-central DP gap is well known; only its measurement here is ours).
- "First privacy attack on PTB-XL" (PRIFLEX [Unverified]).
- "First FL energy measured on a Raspberry Pi" (Gao 2020, Schwermer 2023; Feng 2025 and Akyol 2026 [Unverified]).
- "First to combine DP and Byzantine robustness" (Ma 2022, BREA 2021, RoFL 2023).

---

## 9. Timeline and freeze rules (identical in v4 §9)
| Month | Work | Gate |
|---|---|---|
| 1 | G0 checks; searches S1–S4 with a log in `01_Literature/search_log.md`; copy the threat model (v4 §2) into `03_Threat_Model/threat_model.md`; pin all versions | G0 |
| 2–3 | Build the five configurations (Flower + Opacus); tune on fold 9; E1 | G1 (end of month 3) |
| 4 | E2; start E4 | – |
| 5 | Finish E4 | – |
| 6 | E3 | G2 |
| 7 | E5 | – |
| 8 | E6 | – |
| 9 | E7 on Raspberry Pi + INA219 | – |
| 10 | Optional E8/E9; statistics; figures | – |
| 11–12 | Write, release the code, submit to IEEE IoT-J | – |

**Freeze rules**
1. Frozen on **11 Nov 2026** by the student.
2. Change the plan only if (a) a gate fails (G0, G1 or G2) and the allowed adjustment is used, or (b) searches S1–S4 find a paper that compares these placements under a suspect gateway.
3. Log every change in the deviation log of the pre-registration, with the date and the reason.
4. No new experiments, datasets, placements or hypotheses after the freeze. New ideas go to Paper 2.
5. A failed hypothesis is a result: use its written fallback and do not redesign.
6. Fold 10 is used once per final configuration.

---

## 10. Allowed adjustments (each used at most once, and logged)
1. **G0-a or G1 fails:** choose **one** of: ε 3 → 8; *T* 100 → 60; batch 16 → 32.
2. **G0-c fails:** E6 seeds 3 → 2 (E6 = 68 runs; total 617 runs [Computed]).
3. **A dependency fails on the Raspberry Pi:** replace it and record the new version.

Anything else is a deviation (§13).

---

## 11. Literature searches (month 1, part of G0-b)
Run each search in Scopus and IEEE Xplore, for 2019–2026. Record the date, query, hit count and selected papers.

| # | Purpose | Query |
|---|---|---|
| S1 | Byzantine-robust secure aggregation | ("secure aggregation") AND ("Byzantine" OR "poisoning" OR "robust aggregation") AND "federated" |
| S2 | DP at a distrusted edge in hierarchical FL | ("hierarchical federated" OR "edge-assisted" OR "client-edge-cloud") AND ("differential privacy") AND ("untrusted" OR "malicious" OR "curious") |
| S3 | Sybil and dropout attacks on secure aggregation | ("secure aggregation") AND ("Sybil" OR "dropout" OR "malicious server" OR "active adversary") |
| S4 | Placement comparisons | ("federated") AND ("trusted aggregator" OR "untrusted edge" OR "aggregation placement" OR "design space") AND ("privacy" AND "poisoning") |

---

## 12. Reproducibility
- Pin and record:
  - OS, Python, PyTorch, Flower and Opacus versions;
  - the FLPoison, `breaching` and ML Privacy Meter commits;
  - the PTB-XL version, the MedMNIST version and subset;
  - the Raspberry Pi OS image.
- Seeds 0–9.
- One config file per run in `05_Experiments/configs/`; logs in `05_Experiments/logs/`; results in `05_Experiments/results/`; hardware data in `06_Hardware_Measurements/`.
- Every table and figure is generated by a script from the result files.

---

## 13. Deviation log
Rows DL-01 to DL-19 come from `02_Research_Reports/07_v4_senior_review_2026-11-12.md` (findings F01–F20).
- They are amendments made **before any experiment ran**: no data was seen.
- They are not gate failures or search results, so they fall outside freeze rule 2. The student approved them on 12 Nov 2026.
- **Where a row differs from the text above (or from v4_FINAL), the row wins.**

| Date | What changed | Why | Approved by |
|---|---|---|---|
| 12 Nov 2026 | **DL-01.** G0-a: the rule "*z* ≤ 1.5" is replaced by a fold-9 utility pilot. P-TA at the default point must beat non-private local-only training by ≥ 5 macro-F1 points. If it fails, the default ε becomes 8, and ε = 3 stays in the ε line. The options "*T* 100 → 60" and "batch 16 → 32" are removed from §10. | F01: *z* = 6.67 at ε = 3 [Computed]. No allowed adjustment reaches 1.5. *T* and batch changes leave (B/z)·√steps at 54–55. Batch 32 raises *z* to 9.39. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-02.** G1 is evaluated on fold 9 only. It also requires P-TA ≥ local-only + 5 macro-F1 points. Fold 10 is first used after G1. | F01: G1 had no absolute floor. F10: the test fold could steer the operating point. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-03.** P-TG-c is central DP-SGD at the gateway. At every local step, the gateway sums its devices' clipped per-example gradients and adds N(0, *z*²*C_ex*²) once, with the same *q*, steps and *z* as device DP-SGD. The gateway takes the **mean** (no gateway median); the cloud takes the median. P5 and v4 §3 are read this way. | F02: "summed update" and "medians at both levels" conflict. One noise per multi-step round has no valid accountant. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-04.** H2 scores the event that P1–P3 predict: in a round, the aggregate leaves the range of the honest inputs (measured in the experimenter's view). Backdoor success and macro-F1 drop are reported for each configuration, not scored. The H2 backdoor is a K1 attack scaled to 0.99·τ. "Uniform" means each device is malicious independently with probability *f*. | F03: P1–P3 do not predict backdoor success. An in-range attacker moves the model 3.8–8.5× faster than honest learning [Computed]. F20: "spread uniformly" was undefined. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-05.** New written outcome: if P-TG or P-PE match P-TA on gateway-view canary AUC (within ±0.09) and beat it on robustness, the paper concludes: "Under device DP-SGD, the gateway may filter individual updates; secure aggregation at the gateway is defence in depth only." | F04: P2 + P4 predict that *k* = 1 is at least as good. The audit resolution is ±0.093 AUC [Computed]. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-06.** In E5, the "HFLMLD native" row uses devices without DP (ε = ∞), as in the HFLMLD paper. E5 stays at 150 runs. | F05: HFLMLD was never run in its native setting. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-07.** H5(a) has three outcomes. **Equivalent:** TOST passes. **P-TA worse:** the paired 95% CI excludes 0 against P-TA, and the fallback framing applies. **Inconclusive:** neither; it is reported as inconclusive, with no fallback claim. | F06: TOST power is 0.80 only if the paired SD ≤ 1.0 point [Computed]. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-08.** Training keeps one record per patient, chosen at random with seed 0, so record-level DP-SGD gives patient-level DP (about 72 records per device; *z* ≈ 7.16 at ε = 3 [Computed]). | F07: the sampling rate was computed per record. 2,952 records belong to patients with more than one record. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-09.** H3(c) fails only if \|gap(*k*) − gap(3)\| > 1.5 points **and** its 95% CI excludes 0. H2 configurations with 3 seeds use the 3-seed mean and are labelled "directional". §7 "Sweeps … descriptive" is read with this exception. | F08: with no true effect, H3(c) would fail by chance in 33–57% of cases at a per-seed SD of 1.5–2.0 [Computed]. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-10.** Each gateway reports the IDs of the group sums it dropped at τ; the cloud tracks this drop rate like the abort rate. The cloud divides each gateway's distance by the expected noise norm for the number of sums used. H4 is scored per tampering type. A missed tampering with macro-F1 drop ≤ 2 points and backdoor change ≤ 3 points is reported as a "harmless miss". | F09: τ drops alone give about 9.6% false alarms [Computed]. Sign flips are hard to detect under heavy noise. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-11.** H3(b) is an empirical expectation, not a P5 prediction. If P-TA at ε = 1 is within 2 macro-F1 points of the trivial classifier, (b) is "not evaluable". | F11: P5 predicts nothing about ε; the models may sit at the floor at ε = 1. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-12.** H1(a), H1(b) and H5(b) are "protocol checks". They stay as bug gates but do not count as support for the hypotheses. | F12: they hold by construction. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-13.** H1(c) is reported as "no DP violation detected (best possible ε_lb = 5.1 with 1,000 canaries)". | F13: a pass does not show that the accounting is right [Computed]. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-14.** E5 uses *b* = 1 tampering gateway per run. | F14: *b* was not stated for E5. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-15.** All rules refer to the default ε (ε₀), which is 3 unless DL-01 sets it to 8. The ε line covers {1, 3, 8} without ε₀. | F15: the rules hard-coded ε = 3. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-16.** If projected compute exceeds the budget, cut in this order and stop when it fits: E6 seeds 3 → 2; E5 seeds 10 → 5; drop the E4 K0/K2 runs. | F16: E6 seeds 3 → 2 alone saves only 5.2% of runs [Computed]. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-17.** New check G0-e: run `breaching`, ML Privacy Meter and FLPoison once each on the 1-D ECG model inside Flower. If a tool fails, use an own implementation and log it here. | F17: tool support for 1-D ECG and Flower is [Unverified]. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-18.** Related-work wording: SHARE is a single-server, two-level design ("Secure Hierarchical Robust Aggregation"). P-FR = SHARE with one clustering per round (R = 1), because R ≥ *k* re-clusterings of the same updates let the aggregator solve for each update [Computed]. VHFL is described as "verification without retrieving client models". TW-Fed (IJACSA 2026) is added to the single-server list. | F18: two competitor descriptions did not match the papers' own words; one library competitor was missing. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-19.** H6 round time = the gateway's compute time plus its network time per round, without device training. Both parts are reported. | F19: "round time" was undefined. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-20.** The fold-9 grid of PR:58–59 (learning rate {0.01, 0.05, 0.1} × *C_ex* {0.5, 1.0, 2.0}) runs inside G0-a, with P-TA at ε₀ and seed 0. The pair with the best fold-9 macro-F1 is the default point. *C* and τ are the 99th percentiles of the benign update norms and group-sum norms in that run (PR:67, PR:72). The G0-a pilot and the G0-c timing use these values. If DL-01 switches ε₀ to 8, the grid is re-run at ε = 8. Fold 10 is not used. | OQ1 (plan §1.9): DL-01 needs the default point in month 1. An allowed fold-9 step moves earlier; no value set changes. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-21.** If G0-a fails at ε = 8, or G1 fails after ε₀ is already 8, work stops. Claude writes a decision note; the student adds a new DL row before any further experiment. There is no automatic action. | OQ2: no allowed adjustment is left, and every other option would change fixed values. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-22.** H2 judges the cloud aggregate. Honest inputs are those not influenced by any malicious device or tampering gateway: honest group means (P-FR); outputs of gateways with no gateway-level breakdown that are not tampering (P-TA). A round is a breakdown if any coordinate of the aggregate lies outside the honest min–max for that coordinate. A configuration is "unsafe" if breakdowns occur in ≥ 50% of rounds (mean over seeds). The 3-seed "directional" configurations count toward the 80%. | OQ3: DL-04 and DL-09 changed the event but gave no decision threshold. 50% matches the expected-count prediction rule (v4:155–157). | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-23.** H4 passes only if each of the 4 non-adaptive tampering types reaches detection ≥ 90% of its tampering gateway-rounds, and false alarms are ≤ 5% of all honest gateway-rounds (pooled). A "harmless miss" (DL-10) counts as a miss; its share is reported next to the detection rate. | OQ4: per type or pooled, and the role of harmless misses, were not stated. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-24.** "Beat it on robustness" (DL-05) means lower backdoor success in E4 at the default attack, with a paired 95% CI over 10 seeds that excludes 0. The macro-F1 drop is also reported. | OQ5: the metric was not stated. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-25.** The trivial classifier (DL-11) predicts every class as positive. Its macro-F1 is the mean over the 5 classes of 2*p*/(1+*p*), where *p* is the class prevalence on the fold used for the comparison. | OQ6: "trivial classifier" was not defined. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-26.** All devices use the same Poisson sampling rate *q* = 16 / (mean training records per device) and the same number of local steps per round, *S* = ⌈1/*q*⌉. One noise multiplier *z* holds for all devices, from the Opacus RDP accountant with (*q*, *T*·*S*, ε₀, δ). Device *i* has an expected batch of *q*·*n_i*. | OQ7: PR:56–57 assume equal device sizes; DL-08 and the Dirichlet partition make them unequal. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-27.** G0-a and G1 use the mean over seeds of the fold-9 macro-F1. For local-only, each device's own model is scored on fold 9, the 210 scores are averaged, and then the mean over seeds is taken. G0-a uses 3 seeds (0–2); G1 uses the 10 E1 seeds. | OQ8: the gate statistic was not stated. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-28.** If G0-b finds a paper that compares these placements under a suspect gateway, work stops. Claude writes a comparison note; the student decides with a new DL row. There is no automatic action. | OQ9: freeze rule 2(b) allows a change but none was written. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-29.** Data preparation. (a) A superclass is positive if any diagnostic SCP code of that class appears in `scp_codes`, whatever its likelihood value. (b) Records with no diagnostic superclass are dropped from all folds. (c) Two-level partition: training records are split at random into 7 equal gateways; inside each gateway, records are split over its 30 devices by Dirichlet(α_g) on each record's primary label (its rarest positive superclass, by training prevalence); α_g = 0.5 in the main setting and 0.1 for gateways 0–2 in E5 (PR:65). (d) Every device keeps ≥ 10 training records; the Dirichlet draw is repeated, up to 1,000 times, until this holds. (e) The partition depends on the run seed *s* and is shared by all configurations with seed *s*. | OQ10 (found in G0-07): PR:64–65 do not say how Dirichlet works with multi-label data or per-gateway α, or whether the partition changes with the seed. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-30.** If no Dirichlet draw in 1,000 meets the floor of DL-29(d), the last draw is kept and repaired: one record at a time is moved at random from the largest device of that gateway to its smallest device, until every device has ≥ 10 records. The number of moved records is logged for each gateway. On synthetic data with PTB-XL sizes, the repair moved about 2.5% (at most 4.8%) of a gateway's records at α = 0.1, and the label skew stayed strong (dominant-label share per device 0.77, against 0.34 for an IID split) [Computed; recheck on real data]. The main setting (α = 0.5) normally meets the floor without repair. | OQ11 (found in G0-07): PR:65 (α = 0.1 for 3 of 7 gateways in E5) and DL-29(d) cannot both hold; 0 of 2,000 simulated draws met the floor. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-31.** In the 9 DL-20 grid runs (benign, *f* = 0), the update clip *C*, the norm check τ and the quantization are off; the device-update norms and the group-sum norms are recorded. *C* and τ are the 99th percentiles of those norms in the best grid run. The G0-a pilot then uses *C*, τ and the quantization (PR:68). | G0-09: DL-20 sets *C* and τ from "that run", which needs them already; this removes the loop. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-32.** The local-only baseline (DL-01, DL-02) trains each device alone, without DP: plain SGD, batch 16, up to 100 epochs. Its learning rate in {0.01, 0.05, 0.1} and its stopping epoch in {10, 25, 50, 100} are the pair with the best mean fold-9 macro-F1 at seed 0 (3 runs; the epochs are checkpoints of the same runs). | G0-09: the baseline was not defined. Tuning it makes it the strongest local-only baseline, so the G0-a test is conservative. | the student, 12 Nov 2026 |
| 12 Nov 2026 | **DL-33.** Macro-F1 (and per-class recall) use a fixed decision threshold of 0.5 on each class's sigmoid output, for every configuration. Thresholds are not tuned. AUROC is reported as well. | G0-09: the multi-label decision rule was not stated. | the student, 12 Nov 2026 |
| 15 Nov 2026 | **DL-34.** Gate G0-a at ε = 3 **failed**. The mean fold-9 macro-F1 over seeds 0–2 (DL-27) was 0.134 for P-TA and 0.310 for local-only, so P-TA − local-only = −17.6 points; the rule needs ≥ +5. As written in DL-01, the default ε becomes **ε₀ = 8** from now on, and ε = 3 stays in the ε line (DL-15). The DL-20 grid and the G0-a pilot are re-run at ε = 8. The local-only runs are not repeated: they use no DP, so they are the same for any ε. | Gate result (G0-14, job `JOB_20261115_G0-14_g0a-grid-pilot`); the action is the pre-written one (DL-01). Note: `05_Experiments/results/G0/G0a_decision_eps3.md`. | the student, 15 Nov 2026 |
| 15 Nov 2026 | **DL-35.** Gate G0-a at ε = 8 **failed** too: mean fold-9 macro-F1 0.174 for P-TA and 0.310 for local-only (−13.6 points); P-TG-c 0.361. As DL-21 requires, work stops. One diagnostic job is allowed. It makes no gate decision, and its results are used only to choose the next DL row. (a) P-TA and P-TG-c **without DP noise** (*z* = 0; Poisson sampling and the per-example clip *C_ex* kept; the update clip *C*, τ and quantization off, as in the DL-31 grid, because they are set for noisy updates), seeds 0–2, at the ε = 8 default point (lr 0.1, *C_ex* 1.0). (b) The ε = 8 pilot again (P-TA, P-TG-c, local-only; seeds 0–2; same settings), with the final fold-9 scores of P-TA and P-TG-c saved. Every run of (a) and (b) also reports the per-class AUROC and the macro-F1 with per-class thresholds tuned on the training data and, as an optimistic bound, on fold 9. These are diagnostics only: DL-33 still holds for every gate and hypothesis. | Gate result (G0-14, job `JOB_20261115_G0-14_g0a-eps3-eps8`); DL-21. Note: `05_Experiments/results/G0/G0a_decision_eps8.md`. | the student, 15 Nov 2026 |
| 15 Nov 2026 | **DL-36.** Path A after the DL-35 diagnostic. (1) **Metric: DL-33 is replaced.** Macro-F1 and per-class recall use, for each model, a per-class threshold tuned on fold 9: the threshold on the grid 0.01–0.99 (step 0.01) with the best F1 on fold 9 (ties: the middle one; a class without positive records keeps 0.5). The same thresholds are applied to fold 10. For local-only, each device's model gets its own thresholds. This tuned macro-F1 decides every gate and hypothesis; all margins stay in macro-F1 points. Macro-AUROC is reported next to every result; macro-F1 at threshold 0.5 is reported in an appendix. (2) **G0-a stays FAILED** at ε = 3 (DL-34) and ε = 8. With the new metric the ε = 8 result is P-TA 0.514 vs local-only 0.506 (+0.8 points; macro-AUROC +1.6), so it still fails; it is not re-scored to pass. ε₀ stays 8. The DL-21 stop is lifted for G0-a: the study continues, and the measured price of distrust (H3) becomes a main result. (3) **G1 (DL-02):** part 1, P-TA ≥ P-TG-c − 10 points, stays the gate. Part 2, P-TA ≥ local-only + 5 points, becomes a reported comparison. If part 1 fails, DL-21 applies. (4) **Selection:** DL-20 (P-TA default point, *C*, τ) and DL-32 (local-only lr and epoch) select with the new metric. The earlier grid runs saved no scores, so before E1 the ε = 8 grid (9 P-TA runs) and the 3 local-only grid runs are run again with the new metric, and the default point is taken from them. (5) DL-25 (trivial classifier) and the MI sensitivity at 90% specificity (PR:83) are unchanged. | DL-35 (`05_Experiments/results/G0/G0a_DL35_diagnostic_results.md`): at threshold 0.5 every method scored below the trivial classifier (0.402); the best thresholds for MI, STTC, CD and HYP were 0.03–0.2; without noise P-TA ≈ P-TG-c (macro-AUROC 0.802 vs 0.805). The new metric does not make G0-a pass. | the student, 15 Nov 2026 (path A; tuned-threshold F1 as the deciding metric; G1 part 2 reported only) |
| 16 Nov 2026 | **DL-37.** E1 configurations not fixed by PR §5 (plan task M2-01). (1) **Baselines without DP** (ε = ∞), like local-only (DL-32): plain SGD, batch 16, 1 local epoch per device round. *Centralized*: one model on all training records. *FedAvg*: all 210 devices every round; the gateways relay; the cloud takes the mean weighted by record counts. *HierFAVG*: weighted mean at the gateway and at the cloud. (2) **P-PE and HierFAVG use 5 edge aggregations per cloud aggregation** (T_e = 5, as in HFLMLD Table I) and T_g = 20 cloud rounds. Devices still train in 100 rounds, so P-PE keeps T·S = 500 DP-SGD steps and the same ε. HFLMLD's suspicion score accumulates over the 5 edge iterations and is reset at each cloud round. Details the HFLMLD paper does not give are chosen by Claude and logged in `05_Experiments/hflmld_differences.md`. (3) **Baseline tuning:** the learning rate of centralized, FedAvg and HierFAVG is the best of {0.01, 0.05, 0.1} on fold 9 with the DL-36 score (seed 0); centralized also picks its epoch in {10, 25, 50, 100}. (4) **Reading of PR §5:** P-TG and P-PE see plaintext updates, so they use no update clip *C*, no quantization and no τ. P-FR uses *C* and quantization as in P-TA (PR §3 items 15–16: they exist for secure aggregation), but no τ and no gateway check; the cloud tracks the abort rate. P-TG, P-PE and P-FR use the DL-36 default point (lr 0.01, *C_ex* 2.0; *C* = 15.31) at ε₀ = 8. | PR §5 and the E1 list name these configurations but do not set their privacy, tuning or round structure. | the student, 16 Nov 2026 (baselines without DP; T_e = 5 × T_g = 20; fold-9 grid for the baselines) |
| 16 Nov 2026 | **DL-38.** Canaries and metric definitions not fixed by PR §3 item 30 and PR §7 (plan task M2-03). (1) **Canaries:** every run (all configurations, so all train on the same data) draws, with its seed, 100 records from the training pool. Each gets a random label vector, different from its own and not empty (mislabeled real ECGs). The 100 are removed from their devices; 50 at random are inserted into random honest devices (members); 50 stay out. *q*, *S* and *z* are unchanged (record-level DP does not depend on device size). (2) **Canary scores:** *cloud view* = minus the BCE loss of the canary on the final model; *gateway view* = the sum over rounds of the cosine between what the gateway sees from the canary's host device (its update in P-TG and P-PE, the group sum that contains it in P-TA) and minus the canary's gradient at that round's model. P-FR (masked messages) and P-TG-c (trusted, pre-noise view) have no gateway view. Out-canaries get a would-be host drawn the same way. For each view: membership AUC, and ε_emp as a lower bound: the largest, over score thresholds, of ln((TPR_lo − δ)/FPR_hi) and ln((TNR_lo − δ)/FNR_hi), with one-sided 95% Clopper–Pearson bounds and δ = 10⁻⁵ (as in report 07: 5.11 at best with 500 in / 500 out). (3) **MI sensitivity at 90% specificity:** the largest true-positive rate with false-positive rate ≤ 10% on the ROC curve of the MI class, on the scored fold. (4) **Backdoor success:** among the test records whose true labels include a disease class (MI, STTC, CD or HYP), the share that the model labels exactly 'NORM only' after the trigger (PR §3 item 26), at the model's DL-36 thresholds. (5) **Normalized DTW:** per lead, DTW with absolute-difference cost between the z-normalized true and reconstructed signals, divided by the length of the optimal warping path; the mean over the 12 leads. | PR §3 item 30 and PR §7 name these measures but do not define the canary records or the scores. Mislabeled canaries give the audit (H1(c), DL-13) more power than natural records. | the student, 16 Nov 2026 (mislabeled real ECGs; definitions as proposed) |
| 17 Nov 2026 | **DL-39.** Fold 10 in E1 (DL-02: fold 10 is first used after G1, once per final configuration). Each E1 run scores fold 10 at its end, with the model's fold-9 thresholds (DL-36), and writes these numbers to a separate **sealed** file (`*.fold10.sealed.json`). G1 and every other decision before G1 use fold 9 only. The sealed files are opened only after G1 passes (task M3-03). If G1 fails and E1 is re-run after the allowed adjustment, the sealed files of the first E1 are never opened. | The local-only device models (210 per run) are not kept, so fold 10 can only be scored during the run. | the student, 17 Nov 2026 |
| 17 Nov 2026 | **DL-40.** P-PE's cloud step (plan task M3-01). (1) **Two step sizes.** HFLMLD uses one η for local training and for the cloud update (Algorithm 1, lines 8, 24 and 28; eq. 8 and 14). P-PE keeps η = 0.01 for local DP-SGD (the DL-36 default point, as for every DP design) and uses a separate cloud step **η_c** in eq. (8) and (14). (2) **Grid on fold 9:** η_c ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}, seeds 0–4 (30 runs on the workstation; P-PE carries DP noise, so one seed is not enough), scored with DL-36 at the final round. η_c = 0.01 repeats the E1 setting as a check. The grid runs write no fold-10 file. (3) **Selection:** the **smallest** η_c whose mean macro-F1 is within **1.0 point** of the best mean. A smaller η_c keeps more of HFLMLD's log compression, which the paper uses to weaken corrupted edge models; 1.0 point is below one seed SD of the DP designs in E1 (1.2–1.4 points). (4) **Stop rule:** if the best mean is below the trivial classifier + 5 points (0.452), no η_c is selected and E1 is not run again; the edge stage is then examined (its clustering flags 2–13 of 30 honest devices per edge iteration). (5) **Use:** one workstation job runs the grid, selects η_c by this rule and runs the 10 E1 P-PE runs again with it (sealed fold-10 files, DL-39); these replace the E1 P-PE runs of `JOB_20261117_M3-01_e1`, which stay in the project as superseded. The selected η_c is also used for P-PE in E4 and E5. The E5 "HFLMLD native" row (DL-06, no DP) keeps one η, as in the paper. (6) **Unchanged:** device DP (η_c acts at the cloud on the gateways' outputs, so ε stays 7.99), the edge stage, T_e = 5 × T_g = 20, every other design (P-TA is not re-tuned), fold 10. Each P-PE run also logs, per cloud round, the cloud step and the mean edge distance. | E1 (`05_Experiments/results/M3-01/M3-01_E1_results_note.md`, finding F2): P-PE scored 0.422 macro-F1 (AUROC 0.573), 2 points above the trivial classifier. Measured (`ppe_step_diagnostic/`): with η = 0.01 the cloud moves the global model 0.023 per cloud round, against 0.52 for a plain mean of the edge models (4.4%); eq. (8) bounds the step by η · ln(1 + d/η) for an edge distance d. P-TA (DL-36) and the baselines (DL-37) had their main setting selected on fold 9; P-PE had none. Fold 9 now both selects and reports P-PE, as for P-TA and the baselines; fold 10 (M3-03) is the clean comparison. | the student, 17 Nov 2026 ("choose the best that you see work"; Claude chose a separate η_c, 5 seeds and one job, `05_Experiments/DL-40_draft_PPE_cloud_step.md`) |
| 18 Nov 2026 | **DL-41.** Scope split (Option B of report 10). The frozen plan is reported in two journal papers. **No hypothesis, pass rule, fallback or run setting changes; only the paper in which it is reported.** (1) **Paper 1** reports: E1 (done, with DL-40) and the DL-42 per-design tuning; the E6 ε line (P-TA and P-TG-c, ε ∈ {1, 3}, seeds 0–2: 12 runs); the *f* = 0 row of the E6 *k* × *f* slice (P-TA and P-FR, *k* ∈ {1, 2, 5}, seeds 0–2: 18 runs); E4 with three of its four attacks (uniform backdoor = default attack, concentrated backdoor, label flipping; the 5 DP designs; 10 seeds: 150 runs); E5 as pre-registered (150 runs; DL-06, DL-14, DL-23). (2) **Hypotheses in Paper 1:** H3(a), H3(b), H3(c); H4; H5(a), both parts (F1 from E1; backdoor from the E4 default attack); H5(b), as the byte count of `metrics.wan_bytes` (a protocol check, DL-12; not measured on hardware); the DL-24 robustness comparison. (3) **Moved unchanged to Paper 2:** E2; E3 with G2 and H1; the E4 adaptive attack on the median and the K0/K2 runs (74 runs); the *f* > 0 part of the E6 *k* × *f* slice and the *b* line (H2, P1–P3; 72 runs); E7 (H6, energy, H5(b) measured); E8/E9; DL-05, which needs a working canary audit (E1 finding F3; risk plan R6). H2 in Paper 2 uses the two *k* = 3, *f* = 0.1 cells from the Paper-1 E4 default runs, as planned (v4 §6.3). (4) **Order:** E5 runs before G2. G2 checks group isolation and the DP audit for E3; E5 does not depend on it. (5) **Claims of Paper 1:** it does not claim C5 (Raspberry Pi cost), and its title drops "measured on low-cost hardware". From C3, it uses P4 (argument) and P5 (tested by H3); P1–P3 go to Paper 2. Its privacy claims rest on formal DP and on what each party can see. It makes no empirical-leakage claim. (6) **Reporting:** Paper 1 reports every hypothesis in its scope with its pass rule and fallback, including failures. It states that the frozen plan is larger and that the rest is reported separately, and it publishes this deviation log. It says "frozen analysis plan with a deviation log", not "pre-registered". Paper 2 cites Paper 1's runs as a baseline and does not report them as new. (7) **Unchanged:** every DL row before this one; fold-10 rules; seeds; machines (one machine per experiment). | E1 on both folds shows that the five DP designs do not differ in benign accuracy (within about 1.2 points; P-TA differs from none). So the designs can differ only in robustness, tampering and cost. One complete, fair comparison on these axes is enough for a Q1/Q2 journal paper (report 10). The split is decided before any E2–E7 data exist. | the student, 18 Nov 2026 ("Option B"; wording: "Approve both", `00_Admin/DRAFT_DL-41_DL-42_2026-11-18.md`) |
| 18 Nov 2026 | **DL-42.** Per-design tuning of the five DP designs on fold 9 (fairness, as in DL-37 for the baselines and DL-40 for P-PE). (1) **Grid:** the pre-registered grid lr ∈ {0.01, 0.05, 0.1} × *C_ex* ∈ {0.5, 1.0, 2.0} (PR:58–59), for each of P-TG, P-TG-c, P-PE (η_c = 1.0, DL-40), P-FR and P-TA, at ε₀ = 8, seeds 0–2. One seed is not enough under DP noise (DL-40); seed SD is 1.2–1.4 points in E1. For P-TA and P-FR, *C*, τ and quantization are off in the grid and recorded, as in DL-31. For P-TG, P-TG-c and P-PE, the shared point reuses E1 seeds 0–2. The grid runs write no fold-10 file. About 126 new runs. (2) **Selection, per design:** compute the mean fold-9 macro-F1 (DL-36) over seeds 0–2 at each grid point. The shared point (lr 0.01, *C_ex* 2.0) is **kept** unless the best point's mean is **more than 1.0 point higher**; then the best point is selected. For P-TA and P-FR, *C* (and τ for P-TA) are the 99th percentiles of the norms recorded at the selected point (DL-31). A selected point at the edge of the grid is reported as a limitation; the grid is not extended. (3) **Runs:** for each design whose point changes, the 10 E1 seeds are run again at the new point (one workstation job, the E1 workstation, sealed fold-10 files as in DL-39). (4) **Reporting:** the H3(a) verdict at the pre-registered shared point (**FAIL**, fold 10: +0.51 points, 95% CI −0.29 to +1.31; `E1_analysis.md`) **stays as recorded**. The per-design tuned comparison is reported next to it as a sensitivity analysis, with its own fold-10 numbers. The fold-10 files of the new runs are opened once, by a script committed before opening, as in M3-03. The paper reports both results. Its design guidance uses the tuned comparison and says so. (5) **Use:** the selected point of each design is its final configuration for E4, E5 and the E6 lines (as DL-40 did for P-PE). (6) **Unchanged:** ε₀ = 8, *z*, *q*, *S*, *T*, δ; fold-10 rules; the baselines (DL-37); G1 (PASSED; a gap of about 6 points is still far from −10). (7) **Order note:** this row is written after fold 10 of the shared point was opened (M3-03). The selection uses fold 9 only. | DL-35 vs E1, seeds 0–2, fold 9: P-TG-c 0.574 (lr 0.1, *C_ex* 1.0) against 0.521 (shared point), while P-TA was 0.514 vs 0.516. The shared point was selected for P-TA alone (DL-20, DL-36). E1 finding: the five DP designs are within about 1.2 points only at that shared point. | the student, 18 Nov 2026 ("Approve both"; option A of `00_Admin/DRAFT_DL-41_DL-42_2026-11-18.md`) |
| 18 Nov 2026 | **DL-43.** Open details of E4, E5 and E6 (plan task X-3; Paper-1 scope of DL-41). (1) **Fold 10:** every E4, E5 and E6 run writes fold 10 to a sealed file (DL-39) with its fold-9 thresholds; each experiment's fold 10 is opened once, after all its runs, by a script committed before opening, with a seal register. (2) **Malicious devices (K1):** no per-example clipping, no noise, no update clip *C*; plain SGD (batch 16, 1 local epoch per device round, the design's lr) from the received global model; they follow each design's message format (signed model; in P-TA and P-FR also certified groups, t = *k*, quantization within ±*C* and secure aggregation) and attack in every round from round 1. (3) **Attack norm:** every malicious backdoor update in this study (E4, and the Paper-2 H2 runs of DL-41), and every label-flipping update, is scaled to the norm *C* of P-TA's final configuration (DL-42), the same absolute norm in every design and at every *k*; this replaces DL-04's "scaled to 0.99·τ". A backdoor device poisons 50% of its records (PR item 26 trigger, label "NORM only"). (4) **Label flipping:** on all records of a malicious device, any disease → "NORM only"; no disease → "MI only". (5) **Malicious devices:** uniform, each device with probability *f* = 0.1 (DL-04), from a separate random stream keyed by the run seed, fixed for the run; concentrated, 15 of the 30 devices of one gateway drawn from the run seed; canaries are drawn first as in E1, and a member-canary host is never malicious; honest devices train exactly as in the *f* = 0 run of the same seed. (6) **Measures:** backdoor success (DL-38 (4)) and macro-F1 on fold 10; the pre-registered tests (H5(a), DL-24) use the raw success; the success minus the same design's and seed's clean rate, and the macro-F1 drop against its *f* = 0 run (E1 or DL-42), are reported next to it. (7) **Positive control (R11):** in the E4 job preflight, both attacks on undefended non-DP FedAvg, seed 0, 100 rounds, fold 9; pass if the backdoor success ≥ the clean rate + 30 points and the label flip lowers macro-F1 by ≥ 5 points; otherwise the job stops and the attack is strengthened inside K1 by a new DL row; the trigger and the defences never change. (8) **E5 tampering gateway:** *b* = 1, drawn from the run seed, the same in every round, tampering from round 1, with *f* = 0; partition `partition_E5.yaml`. (9) **Dropping 50%:** silent (not reported; the gateway claims all sums were used). (10) **Adaptive gateway:** honest in round 1, which is not scored; from round 2 it sends out = (1 − 2*s*)·*v*, with *v* its honest output and *s* = min(1, 0.95·*T*₋₁ / (2·‖*v*‖·√(used/full))) (*s* = 0 if *v* = 0), *T*₋₁ being the cloud's threshold of the previous round and used/full its own honest counts (DL-10); a unit test checks the formula; reported only. (11) **HFLMLD rows:** native, P-PE without DP with one η = 0.1 (DL-06, DL-37); cloud rule, HFLMLD eq. (8)–(14) on the 7 P-TA gateway outputs with η_c = 1.0 (DL-40). A gateway-round counts as flagged when its HFLMLD cloud weight is 0; the result reports that eq. (13) gives weight 0 to the farthest gateway in every round. (12) **E6 *C* and τ:** P-TA's τ(*k*) for *k* ∈ {1, 2, 5}, and P-TA's *C* and τ at ε ∈ {1, 3}, are the 99th percentiles of the norms of benign runs at P-TA's final lr and *C_ex*, with *C*, τ and quantization off, seeds 0–2 pooled (DL-31, DL-42); *C* stays the same across *k*; P-FR keeps its *C* (no τ, DL-37). (13) **E6 settings:** each design's DL-42 final point; *z* from `G0a_noise_multipliers.csv`. (14) **H3(c):** gap(*k*) − gap(3), paired over seeds 0–2, with gap(3) from the E1/DL-42 runs; DL-09 rule; Holm over *k*. (15) **H3(b):** the mean gap over seeds 0–2 at ε = 1, 3, 8, at the final configurations; DL-11. (16) **DL-42 fold 10:** the tuned comparison is reported on all 10 seeds and, as a check, on seeds 3–9. (17) **Unchanged:** every hypothesis, pass rule and fallback; seeds; canaries on (DL-38); one machine per experiment. | PR §3, §6 and §7 name these attacks, tampering types and lines but do not set their parameters. Measured facts that drive the choices (§4): with norm 0.99·τ a poisoned P-TA group sum passes τ only if its two honest members sum to ≤ 2.46, while the median honest 3-sum is 5.95 (E1); the update norms grow with lr·*C_ex* and *z*, and are almost pure DP noise (DL-36 grid: *C* 15.3 / 76.6 / 153.1 at lr·*C_ex* 0.02 / 0.1 / 0.2); E1's τ drops 0.34% of benign sums; HFLMLD eq. (13) gives weight 0 to the farthest edge. Written before any E4–E6 run. | the student, 18 Nov 2026 ("Approve DL-43"; amendments A and B added first at the student's request, "Send A and B"; `00_Admin/DRAFT_DL-43_2026-11-18.md`) |
| 18 Nov 2026 | **DL-44.** The gateway check (decision D3) and the attack on P-TG-c (plan tasks X-4, X-6). (1) **Size-aware gateway check** (from E4 on, every P-TA run): each gateway's distance is divided by *s_g*, the standard deviation of the coordinate-wise median of the group means of the sums it reports as used in that round, when each device *d* adds independent noise with standard deviation 1/*n_d* (*n_d* = its record count, registered at enrollment); *s_g* is computed by the cloud by Monte Carlo (2,000 draws, seeded by run seed, round and gateway) and divided by the median of the 7 values; it replaces the √(used/full) factor of DL-10. Factor 3, the 10% rate alert, reputation 0.9 and exclusion (PR items 21–24) are unchanged. The DL-10 flags (√(used/full)) are computed offline from the same logged distances and reported next to them. (2) **P-TA reference:** P-TA's *f* = 0 runs at its final point are run again with the size-aware check (10 seeds, sealed fold 10); they are the reference for the E4 macro-F1 drop and for the E6 gaps at *k* = 3 and ε = 8 (seeds 0–2). The E1 results (H3(a), H5(a) F1) stay as recorded. (3) **P-TG-c in E4:** the gateway runs DL-03 DP-SGD on the pooled records of its honest devices (update Δ_h); each malicious device *m* computes its attack update *a_m* as in DL-43 (2)–(4); the gateway update is (*N_h*/*N*)·Δ_h + Σ_m (*n_m*/*N*)·*a_m*, where *N*, *N_h* and *n_m* are the record counts of all its accepted devices, of its honest devices and of device *m*. (4) **Unchanged:** every hypothesis, pass rule and fallback (H4 is still scored by DL-23); seeds; one machine per experiment. | DL-10 asks for the "expected noise norm", but √(used/full) is that only for equal devices, while DP noise grows as 1/*n_d* (`dpsgd.py`). Report 09 F5: the benign flags are the same at every grid point and ε and follow the device sizes (rank correlation 0.89–0.96); the simulated false alarms on the E5 partition are 29.5% as coded (honest α = 0.1 gateways flagged in 92–98% of rounds) and 2.4% size-aware. DL-03 gives P-TG-c no device update, so DL-43 (3) needs a rule for it. Written before any E4, E5 or E6 run. | the student, 18 Nov 2026 ("Approve DL-44": option B, the size-aware check, and option 1, the record-share update for P-TG-c; `00_Admin/DRAFT_DL-44_2026-11-18.md`) |
| 19 Nov 2026 | **DL-45.** Three open details of E5 (plan task X-7; DL-43 (8)-(11)). (1) **Adaptive gateway against the HFLMLD detectors** (HFLMLD native, and the HFLMLD cloud rule on P-TA): honest in round 1, which is not scored; from round 2 it sends out = −min(1, 0.95·*D*₋₁/‖*v*‖)·*v*, where *v* is its honest output and *D*₋₁ is the largest distance ‖*v_e*‖ among the other edges in the previous round (HFLMLD's distance is ‖*v*‖, because it sees each edge model as w* + *v*); like DL-43 (10), the gateway knows the cloud's previous-round values; reported only (no pass/fail, H4). E5 keeps 5 types × 3 detectors × 10 seeds = 150 runs. (2) **Runs without tampering:** for each detector and seeds 0–9, one run on the E5 partition with no tampering gateway (30 runs, sealed fold 10). A missed tampering counts as harmless (DL-10, DL-23) if, on fold 10, its run's macro-F1 is at most 2 points below the run without tampering of the same detector and seed, and its backdoor rate (DL-38 (4)) differs by at most 3 points; the judgement is per run. These runs also give each detector's false alarms without tampering. (3) **Silent drop in P-PE:** at each edge iteration the edge runs its HFLMLD detection, then averages a random half of the admitted device models and reports all as used (in P-TA: half of the group sums, DL-43 (9)). (4) **Unchanged:** every hypothesis, pass rule and fallback; H4 is scored on our check (DL-23, DL-44); seeds; one machine per experiment. | DL-43 (10) needs our threshold, which HFLMLD lacks; HFLMLD's distance is ‖*v*‖ (`placements.cloud_ppe`, `hflmld.cloud_aggregate`), so a sign flip is invisible to it and the adaptive gateway only has to avoid being the farthest edge. DL-10 and DL-23 report harmless misses, which need a run without tampering. DL-43 (9) defines the drop for group sums only. Written before any E5 run. | the student, 19 Nov 2026 ("Approve DL-45": options A, A, A; `00_Admin/DRAFT_DL-45_2026-11-19.md`) |
| 19 Nov 2026 | **DL-46.** The adaptive gateway of E5 (plan task X-7; PR §3 item 29, DL-43 (10), DL-45 (1)). (1) **Against our check** (P-TA with the size-aware check, DL-44): honest in round 1, which is not scored; from round 2 the gateway tries *s* = 1.00, 0.99, …, 0.00 in turn and sends out = (1 − 2*s*)·*v* (*v* its honest output) for the largest *s* at which the cloud's check of the current round, run on the six other outputs and this candidate (size-aware scales, median + 3·1.4826·MAD, rate alert), does not flag it and gives it a distance ≤ 0.95 × that round's threshold; if no *s* qualifies, it sends the *s* with the smallest distance / threshold ratio. This replaces the formula of DL-43 (10). (2) **Against the HFLMLD detectors:** DL-45 (1), with *D* the largest ‖*v_e*‖ among the other edges of the **current** round instead of the previous one. (3) **Knowledge:** the adaptive gateway knows the other gateways' outputs of the current round. This is more than K2 and a worst case for every detector; its result is read as an upper bound on the harm that passes each detector unflagged. (4) **Unchanged:** the adaptive row is reported only (no pass/fail, H4); round 1 is not scored; the other four tampering types; 5 types × 3 detectors × 10 seeds + 30 runs without tampering = 180 runs; H4 scoring (DL-23, DL-44); seeds; one machine per experiment. | DL-43 (10) was written before DL-44. It scales with √(used/full), while the threshold it reads is that of the size-aware check (`tampering.adaptive_strength`, `run_pilot.py`). It also takes the distance of (1 − 2*s*)·*v* to be 2*s*‖*v*‖, the distance to its own honest output, while the check measures the distance to the median of the seven outputs. Under device DP each output is mostly its own noise, so *T*₋₁ ≈ ‖*v*‖ and the formula gives *s* ≈ 0.5, an almost silent gateway [reasoning; E1 did not log the distances]. PR item 29 asks for "0.95 × the current threshold". Written before any E5 run; found while drafting Section VI (note N2). | the student, 19 Nov 2026 ("Approve DL-46 with option A"; `00_Admin/DRAFT_DL-46_2026-11-19.md`) |
| 19 Nov 2026 | **DL-47.** The E4 attacks after the failed positive control (DL-43 (7), risk plan R19; plan task X-6). (1) **Boost:** each malicious update (plain SGD from the received model, as DL-43 (2)) is multiplied by a boost β and capped at P-TA's final *C*; this replaces "scaled to the norm *C*" of DL-43 (3) for every backdoor and label-flipping update in this study (E4, and the Paper-2 H2 runs of DL-41). (2) **Backdoor recipe:** "replace" (DL-43 (3)) or "copies" (the device keeps every record with its true label and adds a triggered copy, labelled "NORM only", of 50% of its records); the trigger (PR item 26) is unchanged. (3) **Uniform attackers:** a device that hosts a member canary stays honest, and every other device is malicious with probability 0.1·*N*/(*N* − hosts), so the expected malicious share is *f* = 0.1 over all *N* devices (DL-04); the draw stays independent per device and keyed by the run seed. (4) **Gate job before E4:** the positive control of DL-43 (7) (undefended non-DP FedAvg, seed 0, 100 rounds, fold 9, same pass rules) is run for the backdoor with recipe ∈ {replace, copies} × β ∈ {1, 3, 10, 30} and for label flipping with β ∈ {1, 3, 10, 30} and the DL-43 setting. Selection: the backdoor recipe and β with the highest backdoor success (ties: smaller β, then "replace"); the label-flip β with the largest macro-F1 drop (ties: smaller β); the concentrated attack uses the backdoor choice; each must pass its rule. If no backdoor candidate passes, no E4 job is built and the student decides by a new DL row. (5) **The E4 job** is built only after the gate job passed, with the selected settings; its positive control is repeated as a fail-fast first step (the job stops at once if it fails). (6) **Unchanged:** the trigger; the defences; *C* as the upper bound; the label rules; the PC pass rules; every hypothesis, pass rule and fallback (H5(a), DL-24); seeds; one machine per experiment; the E6 runs and the 10 P-TA references of `JOB_20261119_X-9_e6-e4` stay valid. | The positive control of `JOB_20261119_X-9_e6-e4` failed: backdoor 23.7% against 18.3% clean (+5.4 points; rule +30); label flip −23.9 points (passes). Measured on the laptop: a non-DP update has norm about 0.3, so scaling it to *C* = 15.31 boosts it about 50 times; the global model then swings between "all NORM" and "all disease", the tuned thresholds cancel the shift, and the trigger is not learned (trigger effect 0.000 with 5 local epochs or with triggered copies). Without any boost (3.5% poisoned records) the training is stable but the trigger is not learned in 20 rounds either; in centralized training the trigger is learned with 10% triggered copies (+19 points after 5 epochs) but not with 3.5%. The uniform attack had 5.2–10.0% malicious devices (mean 7.7%), not 10%, because canary hosts stay honest. Written before any E4 attack run. | the student, 19 Nov 2026 ("Approve DL-47"; `00_Admin/DRAFT_DL-47_2026-11-19.md`) |
| 19 Nov 2026 | **DL-48.** The rate alert of the P-TA gateway check (PR §3 items 23–24, DL-10, DL-44 (1)). From E4 and E5 on, the rate alert counts the aborted groups and only those tau drops that the device sizes do not explain. Each gateway reports the members of the group sums it dropped at τ (DL-10). The cloud predicts the norm of each dropped group as √Σ min(*C*, K/*n*_i)², with K = √d·lr·*z*·*C*_ex·√*S*/*q* from the protocol values and *n*_i the registered record counts (DL-44 (1)); a drop is explained if this prediction is at least 0.95·τ. A gateway is flagged if (aborts + unexplained drops) exceed 10% of its groups in the round. Every tau drop is still logged. The distance part of the check (size-aware, DL-44 (1)), factor 3, the reputation and the exclusion are unchanged; the DL-10 flags reported next to ours use the same rate. Unchanged: every hypothesis, pass rule and fallback (H4 by DL-23); E1 as recorded; the E4 references and E6 runs of `JOB_20261119_X-9_e6-e4`; seeds; one machine per experiment. | A DP update is almost pure noise with norm about 153.5/*n*_d, so groups with two devices of ≤ 13 records exceed τ = 17.40. On the E5 mixed partition (α = 0.1 for gateways 0–2), these gateways have 8.7–10.2 such devices of 30: predicted tau-drop rate 26–34%, rate alert in 85–93% of rounds (all 10 seeds), against about 0% on the main partition. A real 2-round run with the DL-46 code flagged gateways 0–2 in both rounds by this alert while their distances were below the threshold. H4 would then fail (pooled false alarms about 30–45%) whatever the distance check does. Counting aborts only (option A) would also stop counting poisoned groups dropped at τ in E4. Written before any E5 run (`05_Experiments/diagnostics/2026-11-19_pc-failure_and_tau-drops/`). | the student, 19 Nov 2026 ("Approve DL-48 (B)"; `00_Admin/DRAFT_DL-48_2026-11-19.md`) |
| 19 Nov 2026 | **DL-49.** E4 without the backdoor (DL-43 (7), DL-47 (4); plan task X-6). The positive control failed for every backdoor setting of the gate job (best +4.2 points over the clean rate; rule +30), while label flipping passed. (1) **Attacks:** uniform label flipping (DL-43 (4), host-adjusted f, DL-47 (3)) and concentrated label flipping (15 of the 30 devices of one gateway, DL-43 (5)), both with every malicious update scaled to P-TA's final *C* (the gate's label-flip choice); the uniform and concentrated backdoors are not run. 2 attacks × 5 designs × 10 seeds = 100 runs, sealed fold 10. (2) **Fail-fast positive control:** both label flips on undefended non-DP FedAvg, seed 0, 100 rounds, fold 9, each at least 5 macro-F1 points below E1 FedAvg seed 0; if either fails the job stops before any E4 run. (3) **Hypotheses:** the backdoor part of H5(a) and DL-24 are not evaluable and are reported as such; the F1 part of H5(a) stands (E1). (4) **Secondary tests, added before any E4 data:** S1, P-TA − P-FR macro-F1 drop under the uniform label flip, TOST ±3 points with the DL-07 outcomes; S2, P-TG − P-TA and P-PE − P-TA macro-F1 drop under the uniform label flip, "lower" if the paired 95% CI is below 0, Holm-adjusted p-values reported. Every design and both attacks are also reported descriptively with paired 95% CIs. (5) **References:** as DL-43 (6) and DL-44 (2); the P-TA references stay valid under DL-48 (in their 1,000 rounds the rate alert flagged nothing on its own). (6) **Unchanged:** the trigger-free label rules, *C*, the defences, DL-48, seeds, one machine per experiment; the Paper-2 H2 runs (DL-41 (3)) are decided in Paper 2. | Gate job `JOB_20261119_X-9_e4-gate`: 8 backdoor settings (replace or copies × boost 1, 3, 10, 30) all failed the positive control (best +4.2 points; rule +30); label flipping scaled to *C* passed (−28.3 points). A backdoor that does not work on undefended FedAvg makes "equal backdoor success" between placements uninterpretable. Label flipping measures the same robustness axis without a trigger. The concentrated attack keeps the case where one gateway is overrun. The secondary tests give the robustness comparison a decision rule fixed before the data. Decisions taken by the student on 19 Nov 2026 (experiment-agent plan mode). | the student, 19 Nov 2026 ("Approve DL-49"; plan-mode decisions: label flips, concentrated label flip, fail-fast control with both, margin ±3 points, secondary tests; `00_Admin/DRAFT_DL-49_2026-11-19.md`) |
