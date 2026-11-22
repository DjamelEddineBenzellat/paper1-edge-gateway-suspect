"""Workstation job: DL-40, P-PE's cloud step (plan task M3-01; 30 grid runs + 10 E1 P-PE runs).

Run it with Python 3.12 on the workstation (see README_RUN.txt):
    python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data

Steps (every finished run is kept in state_grid/ or state_e1/, so a second start resumes):
 1. record the machine; check the data fingerprints; unit tests;
 1b. preflight: one short P-PE run with a cloud step, on this device;
 2. the grid on fold 9: 6 cloud steps x seeds 0-4 (configs/DL40_grid), seed by seed; no fold 10;
 3. select the cloud step by the DL-40 rule (select_ppe_eta.py); if the stop rule fires, the job ends here;
 4. the 10 E1 P-PE runs again with the selected cloud step (configs written by make_dl40_configs.py;
    sealed fold-10 files, DL-39);
 5. summary of the new P-PE runs on fold 9 only (summarize_e1.py).
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
GRID = os.path.join(HERE, "configs", "DL40_grid")
E1_CONFIGS = os.path.join(HERE, "configs", "E1")
NOISE = os.path.join(HERE, "inputs", "G0a_noise_multipliers.csv")
ETA_TAGS = ["0p01", "0p03", "0p1", "0p3", "1", "3"]  # as make_dl40_configs.eta_tag(ETAS)
SEALED = ".fold10.sealed.json"
SHORT = 5  # P-PE: device rounds must be a multiple of T_e = 5


def job_value(key):
    with open(os.path.join(HERE, "job.yaml"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip().strip('"')
    return None


def main():
    ap = argparse.ArgumentParser(description="Paper 1 DL-40 job (P-PE cloud step)")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--venv-python", help="default: <base-dir>/venvs/paper1-gpu/bin/python")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gpus-per-node", type=float, default=0.14,
                    help="GPU share per gateway node; use 0.25 if the GPU runs out of memory")
    ap.add_argument("--smoke", action="store_true", help="TEST: seed 0 only, 5 device rounds")
    args = ap.parse_args()

    base = os.path.abspath(os.path.expanduser(args.base_dir))
    os.makedirs(base, exist_ok=True)
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    py = args.venv_python or (os.path.join(base, "venvs", "paper1-gpu", "Scripts", "python.exe") if os.name == "nt"
                              else os.path.join(base, "venvs", "paper1-gpu", "bin", "python"))
    py = os.path.abspath(os.path.expanduser(py))
    tag = "_smoke" if args.smoke else ""
    state_grid = os.path.join(HERE, "state_grid" + tag)
    state_e1 = os.path.join(HERE, "state_e1" + tag)
    e1_new = os.path.join(HERE, "configs", "E1_DL40" + tag)  # written in step 4 from the selection
    for d in (state_grid, state_e1):
        os.makedirs(d, exist_ok=True)
    job = Job(HERE, sys.argv)
    status = "FAILED"
    grid_seeds = [0] if args.smoke else list(range(5))
    e1_seeds = [0] if args.smoke else list(range(10))
    if args.smoke:
        job.note("SMOKE TEST: seed 0 only, 5 device rounds, no stop rule; not valid for DL-40")

    with open(os.path.join(HERE, "inputs", "data_fingerprints.json"), encoding="utf-8") as fh:
        prints = json.load(fh)
    ptbxl = os.path.join(data_dir, *prints["ptbxl_folder"].split("/"))
    commit = job_value("git_commit") or "unknown"
    # Attempts 1, 2, 3: a larger GPU share means fewer gateway nodes on the GPU at once (less memory).
    shares = sorted({args.gpus_per_node, max(args.gpus_per_node, 0.25), max(args.gpus_per_node, 0.5)})

    def done(folder, run_id, sealed):
        out = os.path.join(folder, f"{run_id}.json")
        if not os.path.exists(out) or (sealed and not os.path.exists(os.path.join(folder, run_id + SEALED))):
            return False
        with open(out, encoding="utf-8") as fh:
            return "experiment_config" in json.load(fh)  # written last by run_experiment.py

    def run(config, folder, sealed, smoke_rounds=None):
        run_id = os.path.splitext(os.path.basename(config))[0]
        if done(folder, run_id, sealed):
            job.log(f"{run_id}: already done, skipped (resume)")
            return
        cmd = [py, os.path.join(CODE, "run_experiment.py"), "--config", config, "--ptbxl-dir", ptbxl,
               "--out-dir", folder, "--device", args.device, "--noise-csv", NOISE, "--git-commit", commit]
        if smoke_rounds:
            cmd += ["--smoke-rounds", str(smoke_rounds)]
        for attempt, share in enumerate(shares, start=1):
            code, _ = job.run(cmd + ["--gpus-per-node", str(share)], check=False)
            if code == 0 and done(folder, run_id, sealed):
                return
            job.note(f"{run_id}: attempt {attempt} (GPU share {share}) failed with exit code {code}")
        raise RuntimeError(f"{run_id} failed {len(shares)} times; see logs/run.log")

    selection = os.path.join(job.results, "selection_dl40.json")
    summary = os.path.join(job.results, "e1_ppe_dl40_summary_fold9.json")
    stopped = False
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
            missing = [f"DL40_ppe_eta{t}_s{s}" for s in grid_seeds for t in ETA_TAGS
                       if not os.path.exists(os.path.join(GRID, f"DL40_ppe_eta{t}_s{s}.yaml"))]
            missing += [f"E1_ppe_s{s}" for s in e1_seeds
                        if not os.path.exists(os.path.join(E1_CONFIGS, f"E1_ppe_s{s}.yaml"))]
            if missing:
                raise RuntimeError(f"configs missing: {missing}")
            env = dict(os.environ, PTBXL_DIR=ptbxl)  # also runs the real-data test
            job.run([py, "-B", "-W", "ignore", "-m", "unittest", "discover", "-s",
                     os.path.join(HERE, "code", "tests"), "-v"], env=env)
            code, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_used.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")

        with job.step("1b preflight: a short P-PE run with a cloud step on this device"):
            pre = os.path.join(job.out_dir, "preflight")
            os.makedirs(pre, exist_ok=True)
            run(os.path.join(GRID, "DL40_ppe_eta0p3_s0.yaml"), pre, sealed=False, smoke_rounds=SHORT)

        with job.step("2 the DL-40 grid on fold 9: 6 cloud steps x seeds, seed by seed"):
            for seed in grid_seeds:
                for t in ETA_TAGS:
                    run(os.path.join(GRID, f"DL40_ppe_eta{t}_s{seed}.yaml"), state_grid, sealed=False,
                        smoke_rounds=SHORT if args.smoke else None)
                job.log(f"grid seed {seed}: all {len(ETA_TAGS)} cloud steps done")

        with job.step("3 select the cloud step (DL-40 rule)"):
            job.run([py, os.path.join(CODE, "select_ppe_eta.py"), "--runs", state_grid, "--out", selection,
                     "--n-seeds", str(len(grid_seeds))] + (["--smoke"] if args.smoke else []))
            with open(selection, encoding="utf-8") as fh:
                chosen = json.load(fh)["selected_cloud_eta"]
            if chosen is None:
                stopped = True
                job.note("DL-40 STOP RULE: no cloud step reached the trivial classifier + 5 points. "
                         "E1 is not run again; the edge stage must be examined (see selection_dl40.json).")
            else:
                job.log(f"selected cloud step eta_c = {chosen}")

        if not stopped:
            with job.step("4 the 10 E1 P-PE runs with the selected cloud step (sealed fold 10)"):
                job.run([py, os.path.join(CODE, "make_dl40_configs.py"), "rerun", "--e1", E1_CONFIGS,
                         "--selection", selection, "--out", e1_new, "--commit", commit])
                for seed in e1_seeds:
                    run(os.path.join(e1_new, f"E1_ppe_s{seed}.yaml"), state_e1, sealed=True,
                        smoke_rounds=SHORT if args.smoke else None)

            with job.step("5 summary of the new P-PE runs on fold 9 only (sealed fold 10 not opened)"):
                job.run([py, os.path.join(CODE, "summarize_e1.py"), "--runs", state_e1, "--out", summary])
        status = "TEST_OK" if args.smoke else "OK"
    except StepFailed:
        status = "FAILED"
    except Exception as err:  # still write the zip
        job.log(f"UNEXPECTED ERROR: {type(err).__name__}: {err}")
    finally:
        outs = {"grid_runs": state_grid, "e1_ppe_runs": state_e1}
        for name, folder in outs.items():
            dest = os.path.join(job.results, name)
            os.makedirs(dest, exist_ok=True)
            for f in sorted(os.listdir(folder)):
                if f.endswith(SEALED):  # DL-39: copied, not opened
                    os.makedirs(os.path.join(job.results, "sealed_fold10"), exist_ok=True)
                    shutil.copy2(os.path.join(folder, f), os.path.join(job.results, "sealed_fold10"))
                elif f.endswith(".json"):
                    shutil.copy2(os.path.join(folder, f), dest)
        if os.path.isdir(e1_new):
            shutil.copytree(e1_new, os.path.join(job.results, "configs_E1_DL40"), dirs_exist_ok=True)
        if not os.path.exists(selection) and os.listdir(os.path.join(job.results, "grid_runs")):
            job.log("writing a partial grid table (no selection: the grid is not complete)")
            job.run([py, os.path.join(CODE, "select_ppe_eta.py"), "--runs", os.path.join(job.results, "grid_runs"),
                     "--out", os.path.join(job.results, "selection_dl40_PARTIAL.json"), "--partial"]
                    + (["--smoke"] if args.smoke else []), check=False)
        job.finish(status)
    sys.exit(0 if status.endswith("OK") else 1)


if __name__ == "__main__":
    main()
