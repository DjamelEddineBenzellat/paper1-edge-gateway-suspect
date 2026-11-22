# Data versions and checks — G0-02 record (12 Nov 2026)

**Downloaded by:** the student, 12 Nov 2026.
**Checked by:** Claude, 12 Nov 2026.
**Location:** outside the project, so data is never in git.

## PTB-XL
| Item | Value |
|---|---|
| Version | **1.0.3**, published on PhysioNet on 9 Nov 2022 |
| Page | https://physionet.org/content/ptb-xl/1.0.3/ |
| DOI of this version (cite it, with Wagner et al. 2020) | 10.13026/kfzx-aw45 |
| DOI of the latest version | 10.13026/6sec-a640 |
| Licence | Creative Commons Attribution 4.0 International |
| Downloaded as | ZIP, 1.7 GB (3.0 GB unpacked) |
| Folder | `<BASE_DIR>/PhD_data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/` |
| `ptbxl_database.csv` SHA-256 | 7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b |
| `scp_statements.csv` SHA-256 | ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f |
| `SHA256SUMS.txt` SHA-256 | b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d |
| `LICENSE.txt` SHA-256 | 9a78e7f22742dde9f66ae235ec793ba2212019dc4d0ced75c4da09ced0b35fb2 |

**Important:** version 1.0.3 has **21,799 records from 18,869 patients**.
- The numbers 21,837 records and 18,885 patients come from the 2020 paper (Wagner et al.), which describes an older version. They appear in v4 and the pre-registration.
- PhysioNet removed 38 duplicate records in versions 1.0.2 and 1.0.3. See `ptbxl_v103_changelog.txt`.
- Use 21,799 and v1.0.3 in the paper.

### Checks
| Check | Result |
|---|---|
| All files against the official `SHA256SUMS.txt` (`sha256sum -c`) | 87,203 of 87,203 match; 0 failures |
| 100 Hz files | 21,799 `.hea` + 21,799 `.dat` |
| Every 100 Hz record read with `04_Code/fl_system/data_ptbxl.py` | 21,799 of 21,799 read, all 12 × 1,000; 0 errors; 0 records with NaN; largest absolute value 20.03 mV |
| Reader against the `wfdb` library | Identical values on synthetic WFDB files, and on 200 random real records with wfdb 4.3.1 (max difference 0.0 mV) |
| 500 Hz files (`records500/`) | Present and checksum-verified, but **not used** by the plan (PR:62) |

### Values on this data (DL-08, DL-25, DL-26, DL-29) [Computed]
Full output: `05_Experiments/results/G0/data_summary.json`.

| Item | Value |
|---|---|
| Records without a diagnostic superclass (dropped, DL-29b) | 411 |
| Training, folds 1–8, before DL-08 | 17,084 |
| Training after DL-08 (one record per patient) | **14,823** |
| Validation (fold 9) / test (fold 10) | 2,146 / 2,158 |
| Mean training records per device (210 devices) | 70.59 |
| Training prevalence NORM / MI / STTC / CD / HYP | 0.468 / 0.241 / 0.233 / 0.221 / 0.120 |
| Trivial macro-F1 (DL-25), fold 9 / fold 10 | 0.402 / 0.401 |
| Sampling plan (DL-26) | *q* = 0.226675; *S* = 5 steps per round; *T* = 100; 500 steps in total |
| δ = 10⁻⁵ against 1/14,823 = 6.7 × 10⁻⁵ | δ is smaller, as PR:54 requires |
| Canaries (PR:82): 50 / 14,823 | 0.34% of training records |

### Partition checks
All checks PASS for each partition. Files are in `05_Experiments/results/G0/`.

| Partition | Smallest device | Records moved by DL-30 | Report |
|---|---|---|---|
| Main, seed 0 | 10 | 0 | `partition_check_seed0.md` |
| Main, seed 1 | 10 | 0 | `partition_check_seed1.md` |
| Main, seed 2 | 10 | 0 | `partition_check_seed2.md` |
| E5 (α = 0.1 for gateways 0–2), seed 0 | 10 | 90 / 52 / 80 in gateways 0 / 1 / 2 (2.5–4.2% of 2,118) | `partition_E5_check_seed0.md` |

## MedMNIST v2 — PneumoniaMNIST
| Item | Value |
|---|---|
| Record | Zenodo 5208230, "Version 2.0", published 16 Aug 2021 (https://zenodo.org/records/5208230). Zenodo also lists a newer version; the plan uses v2. |
| DOI | 10.5281/zenodo.5208230 (MedMNIST v2 paper, ref. 38) |
| Licence | CC BY 4.0, from the MedMNIST website (only DermaMNIST is CC BY-NC 4.0) |
| Licence of the original source data (G0-d) | **Not yet checked [Unverified]**, to do in G0-13 |
| File | `<BASE_DIR>/PhD_data/medmnist/pneumoniamnist.npz` |
| md5 | 28209eda62fecd6e6a2d98b1501bb15f (matches Zenodo) |
| SHA-256 | e1792d3f03751cb101e99f19a63b3c1941436c988665f47853417b05be250cd8 |
| Contents | 4,708 / 524 / 624 images (train / val / test), 28 × 28 uint8; binary labels {0, 1}. This matches Table 2 of the MedMNIST v2 paper. |

## Noise multiplier (G0-a accountant, DL-26)
**Official values** (Opacus 1.6.0 RDP accountant), from `05_Experiments/results/G0/G0a_noise_multipliers.csv`:

| ε | *z* | ε read back |
|---|---|---|
| 1 | 20.625 | 0.998 |
| 3 | 7.676 | 2.994 |
| 8 | 3.355 | 7.994 |

- The inputs are *q* = 0.226675, 500 steps and δ = 10⁻⁵. They are the same for seeds 0–2.
- The earlier file `G0a_noise_multipliers_FALLBACK_preview.csv` is superseded; it agrees with these values to within 0.04.
- **G0-a itself is not decided yet.** DL-01 decides it with the fold-9 utility pilot, not with *z*.

## Environment (G0-03)
- **Recorded by `capture_env.py`:** `05_Experiments/env/environment_G0.md` and `requirements_G0.txt`.
- **Project environment:** `<BASE_DIR>/venvs/paper1/`, on the data drive; the system drive is 95% full.

## Still open
- **G0-d:** the licence of PneumoniaMNIST's original source data [Unverified].
- **G0-e:** tool commits (`breaching`, ML Privacy Meter, FLPoison). Re-run `capture_env.py` with `--tool` after cloning them.
  - Done 16 Nov 2026: commits and the separate tools venv are recorded in `05_Experiments/results/G0/G0e_tool_smoke.md`.
