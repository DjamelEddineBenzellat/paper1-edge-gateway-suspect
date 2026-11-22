"""Workstation job: experiment E1, benign accuracy (plan task M3-01; 90 runs = 9 configurations x seeds 0-9).

Run it with Python 3.12 on the workstation (see README_RUN.txt):
    python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data

Steps (every finished run is kept in state/, so a second start resumes):
 1. record the machine; check the data fingerprints; unit tests;
 1b. preflight: a short run of each of the 9 configurations on this device;
 2. the 90 E1 runs (configs/E1/*.yaml), seed by seed (all configurations of seed 0, then seed 1, ...);
 3. summary on fold 9 only (summarize_e1.py). The sealed fold-10 files (DL-39) are copied to
    results/sealed_fold10/ and are not opened.
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
CONFIGS = os.path.join(HERE, "configs", "E1")
NOISE = os.path.join(HERE, "inputs", "G0a_noise_multipliers.csv")
MODES = ["ptg", "ptgc", "ppe", "pfr", "pta", "centralized", "local", "fedavg", "hierfavg"]  # PR §6 order
EDGE_MODES = ("ppe", "hierfavg")  # T_e = 5: their rounds must be a multiple of 5 (DL-37)
SEALED = ".fold10.sealed.json"


def job_value(key):
    with open(os.path.join(HERE, "job.yaml"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip().strip('"')
    return None


def main():
    ap = argparse.ArgumentParser(description="Paper 1 E1 job")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--venv-python", help="default: <base-dir>/venvs/paper1-gpu/bin/python")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gpus-per-node", type=float, default=0.14,
                    help="GPU share per gateway node; use 0.25 if the GPU runs out of memory")
    ap.add_argument("--smoke", action="store_true", help="TEST: seed 0 only, 2 rounds, 2 epochs")
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
    seeds = [0] if args.smoke else list(range(10))
    if args.smoke:
        job.note("SMOKE TEST: seed 0 only, 2 rounds (5 for P-PE/HierFAVG), 2 epochs; not valid for E1")

    with open(os.path.join(HERE, "inputs", "data_fingerprints.json"), encoding="utf-8") as fh:
        prints = json.load(fh)
    ptbxl = os.path.join(data_dir, *prints["ptbxl_folder"].split("/"))
    commit = job_value("git_commit") or "unknown"
    # Attempts 1, 2, 3: a larger GPU share means fewer gateway nodes on the GPU at once (less memory).
    shares = sorted({args.gpus_per_node, max(args.gpus_per_node, 0.25), max(args.gpus_per_node, 0.5)})

    def done(folder, run_id):
        out = os.path.join(folder, f"{run_id}.json")
        if not (os.path.exists(out) and os.path.exists(os.path.join(folder, run_id + SEALED))):
            return False
        with open(out, encoding="utf-8") as fh:
            return "experiment_config" in json.load(fh)  # written last by run_experiment.py

    def run(mode, seed, folder, smoke_rounds=None):
        run_id = f"E1_{mode}_s{seed}"
        if done(folder, run_id):
            job.log(f"{run_id}: already done, skipped (resume)")
            return
        cmd = [py, os.path.join(CODE, "run_experiment.py"), "--config", os.path.join(CONFIGS, f"{run_id}.yaml"),
               "--ptbxl-dir", ptbxl, "--out-dir", folder, "--device", args.device, "--noise-csv", NOISE,
               "--git-commit", commit]
        if smoke_rounds:
            cmd += ["--smoke-rounds", str(smoke_rounds)]
        for attempt, share in enumerate(shares, start=1):
            code, _ = job.run(cmd + ["--gpus-per-node", str(share)], check=False)
            if code == 0 and done(folder, run_id):
                return
            job.note(f"{run_id}: attempt {attempt} (GPU share {share}) failed with exit code {code}")
        raise RuntimeError(f"{run_id} failed {len(shares)} times; see logs/run.log")

    def short(mode, smoke_test):
        if mode in EDGE_MODES:
            return 5
        return 2 if smoke_test else 1

    summary = os.path.join(job.results, "e1_summary_fold9.json")
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
            missing = [m for m in MODES for s in seeds
                       if not os.path.exists(os.path.join(CONFIGS, f"E1_{m}_s{s}.yaml"))]
            if missing:
                raise RuntimeError(f"configs missing: {missing}")
            env = dict(os.environ, PTBXL_DIR=ptbxl)  # also runs the real-data test
            job.run([py, "-B", "-W", "ignore", "-m", "unittest", "discover", "-s",
                     os.path.join(HERE, "code", "tests"), "-v"], env=env)
            code, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_used.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")

        with job.step("1b preflight: a short run of each configuration on this device"):
            # Catches memory or setup errors in minutes. These short runs are not results.
            pre = os.path.join(job.out_dir, "preflight")
            os.makedirs(pre, exist_ok=True)
            for mode in MODES:
                run(mode, 0, pre, smoke_rounds=short(mode, False))

        with job.step("2 the 90 E1 runs, seed by seed (plan task M3-01)"):
            for seed in seeds:
                for mode in MODES:
                    run(mode, seed, state, smoke_rounds=short(mode, True) if args.smoke else None)
                job.log(f"seed {seed}: all {len(MODES)} configurations done")

        with job.step("3 summary on fold 9 only (sealed fold 10 not opened)"):
            job.run([py, os.path.join(CODE, "summarize_e1.py"), "--runs", state, "--out", summary])
        status = "TEST_OK" if args.smoke else "OK"
    except StepFailed:
        status = "FAILED"
    except Exception as err:  # still write the zip
        job.log(f"UNEXPECTED ERROR: {type(err).__name__}: {err}")
    finally:
        runs_out = os.path.join(job.results, "runs")
        sealed_out = os.path.join(job.results, "sealed_fold10")  # DL-39: copied, not opened
        os.makedirs(runs_out, exist_ok=True)
        os.makedirs(sealed_out, exist_ok=True)
        for name in sorted(os.listdir(state)):
            if name.endswith(SEALED):
                shutil.copy2(os.path.join(state, name), sealed_out)
            elif name.endswith(".json"):
                shutil.copy2(os.path.join(state, name), runs_out)
        if not os.path.exists(summary) and os.listdir(runs_out):
            job.log("writing a partial summary from the finished runs (fold 9 only)")
            job.run([py, os.path.join(CODE, "summarize_e1.py"), "--runs", runs_out,
                     "--out", os.path.join(job.results, "e1_summary_fold9_PARTIAL.json")], check=False)
        job.finish(status)
    sys.exit(0 if status.endswith("OK") else 1)


if __name__ == "__main__":
    main()
