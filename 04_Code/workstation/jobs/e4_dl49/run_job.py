"""Workstation job: E4 (device poisoning by label flipping) for Paper 1 (plan task X-9; DL-43, DL-44, DL-47, DL-48,
DL-49).

Run it with Python 3.12 on the workstation (see README_RUN.txt):
    python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data

Steps (every finished run is kept in a state_* folder, so a second start resumes):
 1. record the machine; check the data fingerprints, the 100 + 2 configs and the inputs; unit tests;
 1b. preflight (about 10 minutes): both positive-control runs and one E4 run of every design and attack, shortened to
     5 rounds; then the positive-control check on the shortened runs;
 2. FAIL-FAST positive control (DL-49 (2)): uniform and concentrated label flipping on undefended non-DP FedAvg, seed 0,
    100 rounds, fold 9. If either lowers macro-F1 by less than 5 points, the job STOPS here (status PC_FAILED) and no
    E4 run starts;
 3. the 100 E4 runs, seed by seed (sealed fold 10): 2 attacks x 5 designs x 10 seeds.
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
CFG = os.path.join(HERE, "configs")
E1_FEDAVG = os.path.join(HERE, "inputs", "E1_fedavg_s0.json")
NOISE = os.path.join(HERE, "inputs", "G0a_noise_multipliers.csv")
EXPECTED = {"E4_DL49": 100, "E4_DL49_pc": 2}
SEALED = ".fold10.sealed.json"
SHORT = 5  # P-PE: device rounds must be a multiple of T_e = 5
PREFLIGHT = [f"E4_{mode}_{attack}_s0" for mode in ("ptg", "ptgc", "ppe", "pfr", "pta")
             for attack in ("labelflip", "concentrated_labelflip")]


class Incomplete(Exception):
    """Some runs were skipped after 3 failed attempts (R13)."""


class PCFailed(Exception):
    """The fail-fast positive control did not pass (DL-49 (2)): the job stops on purpose."""


def job_value(key):
    with open(os.path.join(HERE, "job.yaml"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip().strip('"')
    return None


def configs(folder):
    return sorted(glob.glob(os.path.join(CFG, folder, "*.yaml")))


def seed_of(path):
    return int(os.path.basename(path).rsplit("_s", 1)[1].split(".")[0])


def main():
    ap = argparse.ArgumentParser(description="Paper 1 E4 job (DL-49)")
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
    state = {k: os.path.join(HERE, f"state_{k}") for k in ("pc", "e4")}
    pre = os.path.join(HERE, "preflight")  # not zipped: shortened runs only (their sealed files are tests)
    pre_runs = os.path.join(pre, "runs")
    for d in list(state.values()) + [pre_runs]:
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

    def pc_check(folder, out, smoke):
        cmd = [py, os.path.join(CODE, "e4_positive_control.py"), "--dl49", "--runs", folder, "--e1-fedavg", E1_FEDAVG,
               "--out", out]
        return job.run(cmd + (["--smoke"] if smoke else []), check=False)[0]

    pc_out = os.path.join(job.results, "positive_control_dl49.json")
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
            counts = {k: len(configs(k)) for k in EXPECTED}
            missing = [p for p in (E1_FEDAVG, NOISE) if not os.path.exists(p)]
            missing += [n for n in PREFLIGHT if not os.path.exists(os.path.join(CFG, "E4_DL49", n + ".yaml"))]
            if counts != EXPECTED or missing:
                raise RuntimeError(f"config counts {counts} (expected {EXPECTED}); missing: {missing}")
            job.log(f"configs present: {counts}")
            env = dict(os.environ, PTBXL_DIR=ptbxl)  # also runs the real-data test
            job.run([py, "-B", "-W", "ignore", "-m", "unittest", "discover", "-s",
                     os.path.join(HERE, "code", "tests"), "-v"], env=env)
            code, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_used.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")

        with job.step("1b preflight: every run type shortened, then the positive-control check"):
            for p in configs("E4_DL49_pc"):
                run(p, pre_runs, sealed=False, smoke_rounds=SHORT, strict=True)
            if pc_check(pre_runs, os.path.join(pre, "positive_control_dl49_smoke.json"), smoke=True) != 0:
                raise RuntimeError("e4_positive_control.py --dl49 --smoke failed on the preflight runs")
            for n in PREFLIGHT:
                run(os.path.join(CFG, "E4_DL49", n + ".yaml"), pre_runs, sealed=True, smoke_rounds=SHORT, strict=True)
            job.log("preflight OK: both positive-control runs, the check, and every design x attack")

        with job.step("2 fail-fast positive control (DL-49 (2))"):
            ok = all(run(p, state["pc"], sealed=False) for p in configs("E4_DL49_pc"))
            code = pc_check(state["pc"], pc_out, smoke=False) if ok else None
        if code is None:
            raise Incomplete()
        if code != 0:
            status = "PC_FAILED"
            job.note("POSITIVE CONTROL FAILED (DL-49 (2)): a label flip lowered macro-F1 by less than 5 points. "
                     "The job STOPS here; no E4 run was started.")
            raise PCFailed()

        with job.step("3 the 100 E4 runs, seed by seed (sealed fold 10)"):
            for seed in range(10):
                for p in [q for q in configs("E4_DL49") if seed_of(q) == seed]:
                    run(p, state["e4"], sealed=True)
                job.log(f"E4 seed {seed}: done ({len(skipped)} runs skipped so far)")
        if skipped:
            raise Incomplete()
        status = "OK"
    except PCFailed:
        pass  # status PC_FAILED: stopped on purpose
    except Incomplete:
        status = "INCOMPLETE"
        job.note(f"INCOMPLETE: {len(skipped)} runs skipped: {skipped}. Run the same command again.")
    except StepFailed:
        status = "FAILED"
    except Exception as err:  # still write the zip
        job.log(f"UNEXPECTED ERROR: {type(err).__name__}: {err}")
    finally:
        names = {"pc": "e4_positive_control_runs", "e4": "e4_runs"}
        for key, folder in state.items():
            dest = os.path.join(job.results, names[key])
            os.makedirs(dest, exist_ok=True)
            for f in sorted(os.listdir(folder)):
                if f.endswith(SEALED):  # DL-39: copied, not opened
                    os.makedirs(os.path.join(job.results, "sealed_fold10"), exist_ok=True)
                    shutil.copy2(os.path.join(folder, f), os.path.join(job.results, "sealed_fold10"))
                elif f.endswith(".json"):
                    shutil.copy2(os.path.join(folder, f), dest)
        if os.path.exists(os.path.join(pre, "positive_control_dl49_smoke.json")):
            os.makedirs(os.path.join(job.results, "preflight"), exist_ok=True)
            shutil.copy2(os.path.join(pre, "positive_control_dl49_smoke.json"), os.path.join(job.results, "preflight"))
        job.finish(status)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
