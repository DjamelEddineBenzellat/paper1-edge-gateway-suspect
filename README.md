# Treating the Edge Gateway as a Suspect — code and analysis plan

Code, run configurations, frozen analysis plan with its deviation log, and summary results for the paper
*Treating the Edge Gateway as a Suspect: Defence Placement in Federated ECG Learning* (to be submitted to the IEEE Internet of Things Journal).

The study compares five placements of privacy and poisoning defences (P-TG, P-TG-c, P-PE, P-FR, P-TA) and four
baselines in device–gateway–cloud federated learning on the PTB-XL electrocardiograms, with patient-level DP-SGD on
the devices.

## Contents

| Folder / file | Content |
|---|---|
| `04_Code/fl_system/` | Simulation (Flower, Opacus), placements, metrics, config makers, the scripts that open the sealed test folds, figure and number scripts |
| `04_Code/defences/`, `04_Code/attacks/` | DP-SGD, secure-aggregation simulation, certificates, aggregation rules, HFLMLD; device attacks and gateway tampering |
| `04_Code/workstation/` | The job packages used to run the experiments on the GPU workstation |
| `04_Code/tests/` | Unit tests (134) |
| `05_Experiments/configs/` | One YAML file per run (see below) and the partition settings |
| `05_Experiments/env/` | Pinned versions (`requirements_G0.txt`), the dataset version and SHA-256 fingerprints, the environments used |
| `05_Experiments/preregistration_v4_FROZEN.md` | The frozen analysis plan (11 Nov 2026) with its full **deviation log** (DL-01 to DL-52) |
| `04_Code/defences/certificates.py` | Signed model, certified groups and the **device limit** (DL-52), which was added after all runs of the paper; honest rounds give the same outputs with and without it (unit tests) |
| `05_Experiments/hflmld_differences.md` | How P-PE implements HFLMLD: choices and departures |
| `05_Experiments/results/` | Summary results: the test-fold analyses of E1, DL-42, E4, E5 and E6; the seal registers (SHA-256 of every sealed test-fold file); the numbers of the paper's Section V and Appendix C |
| `RELEASE_SOURCE.txt` | The project commit this release was built from |

Paths and names of the authors' machines in the notes are replaced by placeholders such as `<BASE_DIR>`,
`<WORKSTATION_HOME>`, `<user>` and `<workstation>`.

**Not included:** the PTB-XL data (download it from PhysioNet, below), the per-run result files and sealed test-fold
files (about 100 MB), and the paper itself.

## Setup
Python 3.12. CPU environment, as on the development laptop:
```
python3 -m venv venv
venv/bin/pip install -r 05_Experiments/env/requirements_G0.txt
```
The experiments ran on one NVIDIA GTX 1660 SUPER with the CUDA build of the same PyTorch version (2.15.0); see
`05_Experiments/env/environment_workstation.md`. Main versions: Flower 1.43.0, Opacus 1.6.0, PyTorch 2.15.0.

## Data
PTB-XL v1.0.3 from PhysioNet (DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45), Creative Commons
Attribution 4.0). Check the files against the SHA-256 fingerprints in `05_Experiments/env/data_versions.md` and
`data_fingerprints.json` before running.

## Run the tests
```
cd 04_Code/tests
python -m unittest discover -p "test_*.py"
```
134 tests; one is skipped unless `PTBXL_DIR` points to the PTB-XL folder.

## Run one experiment
One run = one config file:
```
python 04_Code/fl_system/run_experiment.py --config 05_Experiments/configs/E1/E1_pta_s0.yaml \
    --ptbxl-dir <PTB-XL folder> --out-dir <results folder> [--device cuda]
```
The test-fold scores of each run go to a separate sealed file; each experiment's sealed files were first opened
by the `*_fold10.py` scripts, after a register of their SHA-256 hashes had been committed; later reads are recorded in
DL-51 of the plan.

| Config folder | Runs | Experiment |
|---|---|---|
| `E1/` | 90 | Benign accuracy: 5 placements and 4 baselines × 10 seeds |
| `DL40_grid/`, `DL42_grid/`, `DL42_shared_check/` | 30, 126, 9 | Validation-fold tuning (P-PE cloud step; per-placement grid) |
| `E4_reference/` | 10 | P-TA reference runs without attack |
| `E4_DL49/`, `E4_DL49_pc/` | 100, 2 | Device poisoning by label flipping (DL-49) and its positive control |
| `E4_gate/`, `E4_positive_control/` | 13, 2 | The backdoor gate and control that failed (DL-47, DL-49) |
| `E4/` | 150 | The original E4 backdoor configs, replaced by DL-49 and not run |
| `E5/` | 180 | Gateway tampering: 5 types × 3 detectors × 10 seeds, plus runs without tampering |
| `E6_norms/` | 15 | Norm runs of the privacy-level and group-size lines. The 30 line configs are made from their results by `make_e6_configs.py lines` inside the job (`04_Code/workstation/jobs/e6_e4/run_job.py`) |

## Figures and numbers of the paper
- Figures 2–4: `python 04_Code/fl_system/paper_figures.py --results 05_Experiments/results --out <folder>` (uses the included summary results).
- Figure 1: `python 04_Code/fl_system/figure_f1.py --out <folder>` (a drawing; no data).
- Figure 5, Section V and Appendix C (`paper_figures.py --ptbxl-dir …`, `section5_numbers.py`, `appendix_threshold05.py`) also need the PTB-XL data and the per-run or sealed result files, which are not in this release; their outputs are included in `05_Experiments/results/X-11/`.

## Licence
MIT (see `LICENSE`).

## Citation
The paper's reference will be added after publication.
