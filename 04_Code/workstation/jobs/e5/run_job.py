"""Workstation job: E5, gateway tampering, for Paper 1 (plan task X-9; DL-06, DL-14, DL-23, DL-43 to DL-46).

Run it with Python 3.12 on the workstation (see README_RUN.txt):
    python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data

Steps (every finished run is kept in state_e5/, so a second start resumes):
 1. record the machine; check the data fingerprints and the 180 configs; unit tests;
 1b. preflight (about 15 minutes): every run type shortened: each of the 3 detectors (ours with the size-aware check;
     HFLMLD native without DP; the HFLMLD cloud rule on P-TA) with an output tampering, the silent drop, the adaptive
     gateway and no tampering, on the mixed partition (partition_E5.yaml);
 2. the 180 E5 runs, seed by seed (sealed fold 10): 5 types + no tampering x 3 detectors x 10 seeds.
Each run stores its own detection and false-alarm score (DL-23); fold 10 stays sealed.
A run that fails 3 times is skipped and listed; the job then ends INCOMPLETE and the same command runs only the
missing runs (risk plan R13). The results zip is always made, also if a step fails.
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
E5 = os.path.join(HERE, "configs", "E5")
NOISE = os.path.join(HERE, "inputs", "G0a_noise_multipliers.csv")
EXPECTED = 180
SEALED = ".fold10.sealed.json"
SHORT = 5  # P-PE: device rounds must be a multiple of T_e = 5
PREFLIGHT = [f"E5_{d}_{k}_s0" for d in ("ours", "hflmld_native", "hflmld_cloud")
             for k in ("scale", "drop", "adaptive", "none")]


class Incomplete(Exception):
    """Some runs were skipped after 3 failed attempts (R13)."""


def job_value(key):
    with open(os.path.join(HERE, "job.yaml"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip().strip('"')
    return None


def seed_of(path):
    return int(os.path.basename(path).rsplit("_s", 1)[1].split(".")[0])


def main():
    ap = argparse.ArgumentParser(description="Paper 1 E5 job (gateway tampering)")
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
    state = os.path.join(HERE, "state_e5")
    pre_runs = os.path.join(HERE, "preflight", "runs")  # not zipped: shortened runs only
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

    def done(folder, run_id, sealed):
        out = os.path.join(folder, f"{run_id}.json")
        if not os.path.exists(out) or (sealed and not os.path.exists(os.path.join(folder, run_id + SEALED))):
            return False
        with open(out, encoding="utf-8") as fh:
            return "experiment_config" in json.load(fh)  # written last by run_experiment.py

    def run(config, folder, sealed, smoke_rounds=None, strict=False):
        """True when the run is done. After 3 failed attempts: raise if strict, else skip and list it (R13)."""
        run_id = os.path.splitext(os.path.basename(config))[0]
        if done(folder, run_id, sealed):
            job.log(f"{run_id}: already done, skipped (resume)")
            return True
        cmd = [py, os.path.join(CODE, "run_experiment.py"), "--config", config, "--ptbxl-dir", ptbxl,
               "--out-dir", folder, "--device", args.device, "--noise-csv", NOISE, "--git-commit", commit]
        if smoke_rounds:
            cmd += ["--smoke-rounds", str(smoke_rounds)]
        for attempt, share in enumerate(shares, start=1):
            code, _ = job.run(cmd + ["--gpus-per-node", str(share)], check=False)
            if code == 0 and done(folder, run_id, sealed):
                return True
            job.note(f"{run_id}: attempt {attempt} (GPU share {share}) failed with exit code {code}")
        if strict:
            raise RuntimeError(f"{run_id} failed {len(shares)} times; see logs/run.log")
        skipped.append(run_id)
        job.note(f"{run_id}: SKIPPED after {len(shares)} failed attempts; the same command runs it again (R13)")
        return False

    try:
        with job.step("1 record the machine and check the data and the configs"):
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
            configs = sorted(glob.glob(os.path.join(E5, "*.yaml")))
            missing = [n for n in PREFLIGHT if not os.path.exists(os.path.join(E5, n + ".yaml"))]
            if len(configs) != EXPECTED or missing or not os.path.exists(NOISE):
                raise RuntimeError(f"{len(configs)} E5 configs (expected {EXPECTED}); missing: {missing}")
            job.log(f"{len(configs)} E5 configs present")
            env = dict(os.environ, PTBXL_DIR=ptbxl)  # also runs the real-data test
            job.run([py, "-B", "-W", "ignore", "-m", "unittest", "discover", "-s",
                     os.path.join(HERE, "code", "tests"), "-v"], env=env)
            code, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_used.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")

        with job.step("1b preflight: the 12 run types shortened (3 detectors x scale, drop, adaptive, none)"):
            for n in PREFLIGHT:
                run(os.path.join(E5, n + ".yaml"), pre_runs, sealed=True, smoke_rounds=SHORT, strict=True)
            scored = []
            for n in PREFLIGHT:
                with open(os.path.join(pre_runs, n + ".json"), encoding="utf-8") as fh:
                    r = json.load(fh)
                if n.endswith("_none_s0") != ("tampering" not in r):
                    raise RuntimeError(f"{n}: the tampering score is missing or unexpected")
                scored.append(n)
            job.log(f"preflight OK: {len(scored)} run types, each with the expected tampering score")

        with job.step("2 the 180 E5 runs, seed by seed (sealed fold 10)"):
            for seed in range(10):
                for p in [q for q in configs if seed_of(q) == seed]:
                    run(p, state, sealed=True)
                job.log(f"E5 seed {seed}: done ({len(skipped)} runs skipped so far)")
        if skipped:
            raise Incomplete()
        status = "OK"
    except Incomplete:
        status = "INCOMPLETE"
        job.note(f"INCOMPLETE: {len(skipped)} runs skipped: {skipped}. Run the same command again.")
    except StepFailed:
        status = "FAILED"
    except Exception as err:  # still write the zip
        job.log(f"UNEXPECTED ERROR: {type(err).__name__}: {err}")
    finally:
        dest = os.path.join(job.results, "e5_runs")
        os.makedirs(dest, exist_ok=True)
        for f in sorted(os.listdir(state)):
            if f.endswith(SEALED):  # DL-39: copied, not opened
                os.makedirs(os.path.join(job.results, "sealed_fold10"), exist_ok=True)
                shutil.copy2(os.path.join(state, f), os.path.join(job.results, "sealed_fold10"))
            elif f.endswith(".json"):
                shutil.copy2(os.path.join(state, f), dest)
        job.finish(status)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
