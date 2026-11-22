"""Workstation job: the E4 gate for Paper 1 (plan task X-9; DL-43 (7), DL-47).

Run it with Python 3.12 on the workstation (see README_RUN.txt):
    python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data

The positive control of DL-43 (7) at full length, for every attack strength of DL-47 (4), BEFORE any E4 job is built
(DL-47 (5)). Undefended non-DP FedAvg, seed 0, 100 rounds, fold 9 only; no run writes a fold-10 file.
Steps (every finished run is kept in state_gate/, so a second start resumes):
 1. record the machine; check the data fingerprints, the 13 gate configs and the inputs; unit tests;
 1b. preflight (about 6-8 minutes): the 13 gate runs shortened to 5 rounds, then e4_gate.py --smoke on them;
 2. the 13 gate runs at full length;
 3. e4_gate.py applies the DL-47 selection rule and writes results/gate_selection.json.
Status: OK = the gate passed (Claude builds the E4 job with the selected settings); GATE_FAILED = no backdoor or no
label-flip candidate passed (no E4 job; the student decides by a new DL row); INCOMPLETE = a run failed 3 times (run
the same command again); FAILED = a setup step failed. The results zip is always made.
"""
import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from job_runtime import Job, StepFailed, sha256_file, system_report  # noqa: E402

CODE = os.path.join(HERE, "code", "fl_system")
CFG = os.path.join(HERE, "configs")
E1_FEDAVG = os.path.join(HERE, "inputs", "E1_fedavg_s0.json")
NOISE = os.path.join(HERE, "inputs", "G0a_noise_multipliers.csv")
N_GATE = 13  # DL-47 (4): 8 backdoor + 5 label-flip runs
SHORT = 5


class Incomplete(Exception):
    """Some runs were skipped after 3 failed attempts (R13)."""


def job_value(key):
    with open(os.path.join(HERE, "job.yaml"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip().strip('"')
    return None


def main():
    ap = argparse.ArgumentParser(description="Paper 1 E4 gate job (DL-47)")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--venv-python", help="default: <base-dir>/venvs/paper1-gpu/bin/python")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gpus-per-node", type=float, default=0.14,
                    help="GPU share per gateway node; use 0.25 if the GPU runs out of memory")
    args = ap.parse_args()

    base = os.path.abspath(os.path.expanduser(args.base_dir))
    os.makedirs(base, exist_ok=True)
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    py = args.venv_python or (os.path.join(base, "venvs", "paper1-gpu", "Scripts", "python.exe") if os.name == "nt"
                              else os.path.join(base, "venvs", "paper1-gpu", "bin", "python"))
    py = os.path.abspath(os.path.expanduser(py))
    state = os.path.join(HERE, "state_gate")
    pre = os.path.join(HERE, "preflight")  # not zipped: shortened runs only
    pre_runs = os.path.join(pre, "runs")
    for d in (state, pre_runs):
        os.makedirs(d, exist_ok=True)
    job = Job(HERE, sys.argv)
    status = "FAILED"
    skipped = []

    with open(os.path.join(HERE, "inputs", "data_fingerprints.json"), encoding="utf-8") as fh:
        prints = json.load(fh)
    ptbxl = os.path.join(data_dir, *prints["ptbxl_folder"].split("/"))
    commit = job_value("git_commit") or "unknown"
    shares = sorted({args.gpus_per_node, max(args.gpus_per_node, 0.25), max(args.gpus_per_node, 0.5)})
    gate_configs = sorted(glob.glob(os.path.join(CFG, "E4_gate", "*.yaml")))

    def done(folder, run_id):
        out = os.path.join(folder, f"{run_id}.json")
        if not os.path.exists(out):
            return False
        with open(out, encoding="utf-8") as fh:
            return "experiment_config" in json.load(fh)  # written last by run_experiment.py

    def run(config, folder, smoke_rounds=None, strict=False):
        """True when the run is done. After 3 failed attempts: raise if strict, else skip and list it (R13)."""
        run_id = os.path.splitext(os.path.basename(config))[0]
        if done(folder, run_id):
            job.log(f"{run_id}: already done, skipped (resume)")
            return True
        cmd = [py, os.path.join(CODE, "run_experiment.py"), "--config", config, "--ptbxl-dir", ptbxl,
               "--out-dir", folder, "--device", args.device, "--noise-csv", NOISE, "--git-commit", commit]
        if smoke_rounds:
            cmd += ["--smoke-rounds", str(smoke_rounds)]
        for attempt, share in enumerate(shares, start=1):
            code, _ = job.run(cmd + ["--gpus-per-node", str(share)], check=False)
            if code == 0 and done(folder, run_id):
                return True
            job.note(f"{run_id}: attempt {attempt} (GPU share {share}) failed with exit code {code}")
        if strict:
            raise RuntimeError(f"{run_id} failed {len(shares)} times; see logs/run.log")
        skipped.append(run_id)
        job.note(f"{run_id}: SKIPPED after {len(shares)} failed attempts; the same command runs it again (R13)")
        return False

    def gate(folder, out, smoke):
        cmd = [py, os.path.join(CODE, "e4_gate.py"), "--runs", folder, "--e1-fedavg", E1_FEDAVG, "--out", out]
        return job.run(cmd + (["--smoke"] if smoke else []), check=False)[0]

    selection = os.path.join(job.results, "gate_selection.json")
    try:
        with job.step("1 record the machine and check the data, configs and inputs"):
            text, _, _ = system_report(base)
            with open(os.path.join(job.logs, "system.txt"), "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            job.log(text)
            if not os.path.exists(py):
                raise RuntimeError(f"Python environment not found: {py}")
            bad = [rel for rel, want in prints["files"].items()
                   if not os.path.exists(os.path.join(data_dir, *rel.split("/")))
                   or sha256_file(os.path.join(data_dir, *rel.split("/"))) != want]
            if bad:
                raise RuntimeError(f"data fingerprints do not match for {bad}")
            job.log("data fingerprints OK")
            if len(gate_configs) != N_GATE:
                raise RuntimeError(f"{len(gate_configs)} gate configs, expected {N_GATE}")
            missing = [p for p in (E1_FEDAVG, NOISE) if not os.path.exists(p)]
            if missing:
                raise RuntimeError(f"inputs missing: {missing}")
            job.log(f"{len(gate_configs)} gate configs present")
            env = dict(os.environ, PTBXL_DIR=ptbxl)  # also runs the real-data test
            job.run([py, "-B", "-W", "ignore", "-m", "unittest", "discover", "-s",
                     os.path.join(HERE, "code", "tests"), "-v"], env=env)
            code, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_used.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")

        with job.step("1b preflight: the 13 gate runs shortened, then the selection on their outputs"):
            for p in gate_configs:
                run(p, pre_runs, smoke_rounds=SHORT, strict=True)
            if gate(pre_runs, os.path.join(pre, "gate_selection_smoke.json"), smoke=True) != 0:
                raise RuntimeError("e4_gate.py --smoke failed on the preflight runs")
            job.log("preflight OK: 13 shortened gate runs and the selection script")

        with job.step("2 the 13 gate runs at full length (fold 9 only)"):
            for p in gate_configs:
                run(p, state)
        if skipped:
            raise Incomplete()

        with job.step("3 the DL-47 selection rule"):
            code = gate(state, selection, smoke=False)
            if code not in (0, 2):
                raise RuntimeError(f"e4_gate.py ended with exit code {code}")
        if code == 0:
            status = "OK"
            job.note("GATE PASSED: Claude builds the E4 job with results/gate_selection.json (DL-47 (5)).")
        else:
            status = "GATE_FAILED"
            job.note("GATE FAILED: no backdoor or no label-flip candidate passed its positive-control rule. "
                     "No E4 job is built; the student decides by a new DL row (DL-47 (4)).")
    except Incomplete:
        status = "INCOMPLETE"
        job.note(f"INCOMPLETE: {len(skipped)} runs skipped: {skipped}. Run the same command again.")
    except StepFailed:
        status = "FAILED"
    except Exception as err:  # still write the zip
        job.log(f"UNEXPECTED ERROR: {type(err).__name__}: {err}")
    finally:
        dest = os.path.join(job.results, "gate_runs")
        os.makedirs(dest, exist_ok=True)
        for f in sorted(os.listdir(state)):
            if f.endswith(".json"):
                shutil.copy2(os.path.join(state, f), dest)
        if os.path.exists(os.path.join(pre, "gate_selection_smoke.json")):
            os.makedirs(os.path.join(job.results, "preflight"), exist_ok=True)
            shutil.copy2(os.path.join(pre, "gate_selection_smoke.json"), os.path.join(job.results, "preflight"))
        job.finish(status)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
