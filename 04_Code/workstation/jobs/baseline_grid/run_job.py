"""Job: the fold-9 learning-rate grid of the non-private baselines (DL-37 point 3; plan task M2-04).

Runs on the laptop CPU (16 Nov 2026: the workstation could not be reached) or on the workstation.
See README_RUN.txt for the exact command.

Steps (every finished run is kept in state/, so a second start resumes):
 1. record the machine; check the data fingerprints; unit tests;
 1b. preflight: 1 short run of each baseline on this device;
 2. grid, seed 0, no DP: centralized (lr x 100 epochs, scored at epochs 10, 25, 50, 100),
    FedAvg (lr, 100 rounds) and HierFAVG (lr, 100 device rounds = 20 cloud rounds x T_e = 5),
    for lr in {0.01, 0.05, 0.1};
 3. selection with the DL-36 score (select_baselines.py).
The results zip is always made, also if a step fails.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from job_runtime import Job, StepFailed, sha256_file, system_report  # noqa: E402

CODE = os.path.join(HERE, "code", "fl_system")
CONFIG = os.path.join(HERE, "configs", "G0a.yaml")
NOISE = os.path.join(HERE, "inputs", "G0a_noise_multipliers.csv")
EPS = "8"  # DL-34; recorded only: the baselines use no DP (DL-37)


def job_value(key):
    with open(os.path.join(HERE, "job.yaml"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip().strip('"')
    return None


def read_yaml_list(text, key):
    """Tiny reader for the few G0a.yaml values (the system Python has no PyYAML)."""
    import ast
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped.startswith(key + ":"):
            return ast.literal_eval(stripped.split(":", 1)[1].strip())
    raise KeyError(key)


def main():
    ap = argparse.ArgumentParser(description="Paper 1 baseline grid job (DL-37)")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--venv-python", help="default: <base-dir>/venvs/paper1-gpu/bin/python")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gpus-per-node", type=float, default=0.14,
                    help="GPU share per gateway node (CUDA only); use 0.25 if the GPU runs out of memory")
    ap.add_argument("--smoke", action="store_true", help="TEST: 2 rounds, 2 epochs, one learning rate")
    args = ap.parse_args()

    base = os.path.abspath(os.path.expanduser(args.base_dir))
    os.makedirs(base, exist_ok=True)
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    py = args.venv_python or (os.path.join(base, "venvs", "paper1-gpu", "Scripts", "python.exe") if os.name == "nt"
                              else os.path.join(base, "venvs", "paper1-gpu", "bin", "python"))
    py = os.path.abspath(os.path.expanduser(py))
    state = os.path.join(HERE, "state_smoke" if args.smoke else "state")
    os.makedirs(state, exist_ok=True)
    job = Job(HERE, sys.argv)
    status = "FAILED"
    with open(CONFIG, encoding="utf-8") as fh:
        grid_lr = read_yaml_list(fh.read(), "lr")
    fl_rounds = []  # default: T = 100 device rounds
    smoke = []
    if args.smoke:
        grid_lr = grid_lr[:1]
        fl_rounds = ["--rounds", "5"]  # a multiple of T_e = 5 (HierFAVG)
        smoke = ["--local-epochs", "2"]
        job.note("SMOKE TEST: 5 device rounds, 2 epochs, one learning rate; not valid for DL-37")

    with open(os.path.join(HERE, "inputs", "data_fingerprints.json"), encoding="utf-8") as fh:
        prints = json.load(fh)
    ptbxl = os.path.join(data_dir, *prints["ptbxl_folder"].split("/"))
    common = [py, os.path.join(CODE, "run_pilot.py"), "--config", CONFIG, "--noise-csv", NOISE,
              "--ptbxl-dir", ptbxl, "--device", args.device, "--eps0", EPS,
              "--git-commit", job_value("git_commit") or "unknown"]
    # Attempts 1, 2, 3. On CUDA a larger GPU share means fewer gateway nodes on the GPU at once;
    # on the CPU the share is ignored, and the retry only repeats the run.
    shares = sorted({args.gpus_per_node, max(args.gpus_per_node, 0.25), max(args.gpus_per_node, 0.5)})

    def run(run_id, extra, folder=None, settings=None):
        folder = folder or state
        out = os.path.join(folder, f"{run_id}.json")
        if os.path.exists(out):
            job.log(f"{run_id}: already done, skipped (resume)")
            return out
        tmp = out + ".partial"
        for attempt, share in enumerate(shares, start=1):
            code, _ = job.run(common + (settings if settings is not None else []) +
                              ["--gpus-per-node", str(share), "--run-id", run_id, "--out", tmp] + extra, check=False)
            if code == 0:
                break
            job.note(f"{run_id}: attempt {attempt} (GPU share {share}) failed with exit code {code}")
        else:
            raise RuntimeError(f"{run_id} failed {len(shares)} times; see logs/run.log")
        os.replace(tmp, out)
        return out

    selection = os.path.join(job.results, "selection_baselines.json")
    try:
        with job.step("1 record the machine and check the data"):
            text, _, _ = system_report(base)
            with open(os.path.join(job.logs, "system.txt"), "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            job.log(text)
            if not os.path.exists(py):
                raise RuntimeError(f"Python environment not found: {py}")
            bad = []
            for rel, want in prints["files"].items():
                path = os.path.join(data_dir, *rel.split("/"))
                if not os.path.exists(path) or sha256_file(path) != want:
                    bad.append(rel)
            if bad:
                raise RuntimeError(f"data fingerprints do not match for {bad}")
            job.log("data fingerprints OK")
            env = dict(os.environ, PTBXL_DIR=ptbxl)  # also runs the real-data test
            job.run([py, "-B", "-W", "ignore", "-m", "unittest", "discover", "-s",
                     os.path.join(HERE, "code", "tests"), "-v"], env=env)
            code, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_used.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")

        with job.step("1b preflight: 1 short run of each baseline on this device"):
            # Catches setup errors in minutes. These short runs are not results.
            pre = os.path.join(job.out_dir, "preflight")
            os.makedirs(pre, exist_ok=True)
            run("pre_centralized", ["--mode", "centralized", "--seed", "0", "--lr", "0.05"], folder=pre,
                settings=["--local-epochs", "1"])
            run("pre_fedavg", ["--mode", "fedavg", "--seed", "0", "--lr", "0.05"], folder=pre,
                settings=["--rounds", "1"])
            run("pre_hierfavg", ["--mode", "hierfavg", "--seed", "0", "--lr", "0.05"], folder=pre,
                settings=["--rounds", "5"])

        with job.step("2 baseline grid, seed 0, no DP (DL-37)"):
            for lr in grid_lr:
                run(f"centralized_grid_lr{lr}_s0", ["--mode", "centralized", "--seed", "0", "--lr", str(lr)],
                    settings=smoke)
            for lr in grid_lr:
                run(f"fedavg_grid_lr{lr}_s0", ["--mode", "fedavg", "--seed", "0", "--lr", str(lr)],
                    settings=fl_rounds)
            for lr in grid_lr:
                run(f"hierfavg_grid_lr{lr}_s0", ["--mode", "hierfavg", "--seed", "0", "--lr", str(lr)],
                    settings=fl_rounds)

        with job.step("3 selection with the DL-36 score"):
            job.run([py, os.path.join(CODE, "select_baselines.py"), "--runs", state, "--out", selection])
        status = "TEST_OK" if args.smoke else "OK"
    except StepFailed:
        status = "FAILED"
    except Exception as err:  # still write the zip
        job.log(f"UNEXPECTED ERROR: {type(err).__name__}: {err}")
    finally:
        runs_out = os.path.join(job.results, "runs")
        os.makedirs(runs_out, exist_ok=True)
        for name in sorted(os.listdir(state)):
            if name.endswith(".json"):
                shutil.copy2(os.path.join(state, name), runs_out)
        job.finish(status)
    sys.exit(0 if status.endswith("OK") else 1)


if __name__ == "__main__":
    main()
