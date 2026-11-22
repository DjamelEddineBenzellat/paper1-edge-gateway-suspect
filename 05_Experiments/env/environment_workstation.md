# Workstation environment record (rule 5) — 12 Nov 2026

**Source:** setup job `JOB_20261112_G0-03_workstation-setup`, built from git commit `81a0169`. It ran on the workstation on 12 Nov 2026, 18:30–18:46.
**Logs:** `05_Experiments/logs/JOB_20261112_G0-03_workstation-setup/`
**Results:** `05_Experiments/results/G0-03/JOB_20261112_G0-03_workstation-setup/`
**Job status:**
- **First run** (18:30): `OK_WITH_DATA_PROBLEM`. The environment was complete, but the data was not found.
- **Re-run** (20:00): **`OK`**, after the data was copied. Logs: `05_Experiments/logs/JOB_20261112_G0-03_workstation-setup_rerun_20261112-200036/`. Results: `05_Experiments/results/G0-03/JOB_20261112_G0-03_workstation-setup_rerun_20261112-200036/`. The package list is identical to the first run.

## Hardware and system
| Item | Value |
|---|---|
| Host | <workstation> |
| OS | Ubuntu: Linux 7.0.0-43-generic x86_64, glibc 2.43 |
| CPU | 12th Gen Intel Core i5-12400, 12 logical cores |
| RAM | 14.7 GB |
| Disk at `<WORKSTATION_HOME>/paper1` | 269.6 GB free of 467.3 GB |
| GPU | NVIDIA GeForce GTX 1660 SUPER: 6 GB total, 5.61 GB seen by torch; compute capability 7.5 |
| NVIDIA driver | 595.99.02; supports CUDA up to 13.2 |
| Python | 3.12.14 (`<WORKSTATION_HOME>/.local/bin/python3.12`). The laptop has 3.12.3. |

## Environment
| Item | Value |
|---|---|
| Path | `<WORKSTATION_HOME>/paper1/venvs/paper1-gpu` |
| torch / torchvision | **2.15.0+cu132 / 0.29.2+cu132** (CUDA 13.2). CUDA 13.4 was not tried, because the driver supports only 13.2. |
| cuDNN / triton | nvidia-cudnn-cu13 9.26.0.51 / triton 3.9.0 |
| GPU check | CUDA available; torch includes sm_75; the same matrix product on CPU and GPU differs by at most 5.3 × 10⁻⁵ |
| Pinned packages | **Identical to the laptop:** opacus 1.6.0, flwr 1.43.0, ray 2.55.1, numpy 2.5.4, pandas 3.1.1, wfdb 4.3.1, scikit-learn 1.9.1, PyYAML 6.0.3, medmnist 3.0.2 |
| Other differences from the laptop | The CUDA libraries (expected); uv 0.13.8 against 0.13.7 on the laptop. uv is a tool used by Flower, not by our computations. |
| Opacus on the Paper 1 model | 53,765 parameters; no validator errors; per-sample gradients work on the GPU |
| Flower / Ray | Flower simulation imports; Ray starts in 2.2 s |
| `pip check` | No broken requirements |

Full list: `requirements_workstation.txt` and `freeze_diff_vs_laptop.txt` in the results folder.

## First timings (one DP-SGD step, batch 16, 12 × 1000 input)
| Machine | ms per step | DP-SGD-only lower bound for one 100-round run [Computed] |
|---|---|---|
| Laptop CPU (i7-1165G7) | 22.5–25.9 | about 0.7 h |
| Workstation CPU (i5-12400) | 16.28 (first run); 11.17 (re-run) | 0.33–0.47 h |
| **Workstation GPU (GTX 1660 SUPER)** | **3.39 (first run); 3.44 (re-run)** | **0.10 h** |

The GPU time is stable between the two runs. The CPU time varies by about 30%.

The lower bound is 210 devices × 5 steps × 100 rounds × time per step. It leaves out Flower and Ray overhead, aggregation and evaluation. The real time per run is measured in G0-c, from the G0-a pilot.

## Data on the workstation (re-run, 12 Nov 2026)
| Check | Result |
|---|---|
| Folder | `<WORKSTATION_HOME>/paper1/PhD_data` |
| Key fingerprints (`ptbxl_database.csv`, `scp_statements.csv`, `SHA256SUMS.txt`, `pneumoniamnist.npz`) | 4 of 4 OK |
| PTB-XL 100 Hz headers | 21,799 of 21,799 |
| PTB-XL full checksum check (`SHA256SUMS.txt`) | **87,203 of 87,203 match**; `records500` was copied too |

The data on the workstation is identical to the laptop.
