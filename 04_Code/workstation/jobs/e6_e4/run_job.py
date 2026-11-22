"""Workstation job: E6 (eps and k lines at f = 0) and E4 (device poisoning) for Paper 1 (plan task X-9; DL-41,
DL-42 (5), DL-43, DL-44).

Run it with Python 3.12 on the workstation (see README_RUN.txt):
    python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data

Steps (every finished run is kept in a state_* folder, so a second start resumes):
 1. record the machine; check the data fingerprints, the configs and the inputs; unit tests;
 1b. preflight (about 20-25 minutes): every run type shortened, then every later step on those outputs;
 2. the E4 positive control (DL-43 (7)): both attacks on undefended non-DP FedAvg, seed 0, 100 rounds. If it fails,
    the 150 E4 attack runs are NOT started (the status becomes PC_FAILED; E6 and the references still run);
 3. the 15 E6 norm runs; their pooled norms set P-TA's tau(k) and C, tau per eps (e6_norms.py); the 30 E6 line
    configs are then written (make_e6_configs.py lines);
 4. the 30 E6 line runs (sealed fold 10);
 5. the 10 P-TA f = 0 reference runs with the size-aware check (DL-44 (2); sealed fold 10);
 6. the 150 E4 runs, seed by seed (sealed fold 10).
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
E1_CONFIGS, DL40_CONFIGS = os.path.join(CFG, "E1"), os.path.join(CFG, "E1_DL40")
SELECTION = os.path.join(HERE, "inputs", "selection_dl42.json")
E1_FEDAVG = os.path.join(HERE, "inputs", "E1_fedavg_s0.json")
NOISE = os.path.join(HERE, "inputs", "G0a_noise_multipliers.csv")
EXPECTED = {"E4": 150, "E4_positive_control": 2, "E4_reference": 10, "E6_norms": 15}
SEALED = ".fold10.sealed.json"
SHORT = 5  # P-PE: device rounds must be a multiple of T_e = 5
PREFLIGHT = {  # one shortened run of every type
    "E6_norms": ["E6n_pta_k1_s0", "E6n_pta_k2_s0", "E6n_pta_k5_s0", "E6n_pta_eps1_s0", "E6n_pta_eps3_s0"],
    "E6_lines": ["E6_pta_k1_s0", "E6_pfr_k5_s0", "E6_ptgc_eps1_s0", "E6_pta_eps3_s0"],
    "E4": ["E4_pta_backdoor_s0", "E4_ptgc_concentrated_s0", "E4_ppe_labelflip_s0", "E4_pfr_backdoor_s0",
           "E4_ptg_labelflip_s0"],
    "E4_reference": ["E4ref_pta_s0"],
    "E4_positive_control": ["E4pc_fedavg_backdoor_s0", "E4pc_fedavg_labelflip_s0"],
}


class Incomplete(Exception):
    """Some runs were skipped after 3 failed attempts (R13)."""


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
    ap = argparse.ArgumentParser(description="Paper 1 E6 + E4 job")
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
    state = {k: os.path.join(HERE, f"state_{k}") for k in ("pc", "e6n", "e6", "e4ref", "e4")}
    pre = os.path.join(HERE, "preflight")  # not zipped: shortened runs only (their sealed files are tests)
    pre_runs = os.path.join(pre, "runs")
    for d in list(state.values()) + [pre_runs]:
        os.makedirs(d, exist_ok=True)
    e6_lines = os.path.join(CFG, "E6")  # written in step 3 from the pooled norms (next to G0a.yaml)
    job = Job(HERE, sys.argv)
    status = "FAILED"
    skipped = []
    pc_passed = None

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

    def py_step(script, *extra, check=True):
        return job.run([py, os.path.join(CODE, script)] + list(extra), check=check)

    def lines_from(norms, subdir):
        """The E6 line configs from pooled norms, in configs/<subdir>/ (next to G0a.yaml)."""
        py_step("make_e6_configs.py", "lines", "--e1", E1_CONFIGS, "--dl40", DL40_CONFIGS, "--selection", SELECTION,
                "--norms", norms, "--out", CFG, "--subdir", subdir, "--commit", commit)
        return os.path.join(CFG, subdir)

    e6_norms = os.path.join(job.results, "e6_norms.json")
    pc_out = os.path.join(job.results, "positive_control.json")
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
            if counts != EXPECTED:
                raise RuntimeError(f"config counts {counts}, expected {EXPECTED}")
            missing = [p for p in (SELECTION, E1_FEDAVG, NOISE) if not os.path.exists(p)]
            missing += [n for k in ("E6_norms", "E4", "E4_reference", "E4_positive_control") for n in PREFLIGHT[k]
                        if not os.path.exists(os.path.join(CFG, k, n + ".yaml"))]
            if missing:
                raise RuntimeError(f"inputs or configs missing: {missing}")
            job.log(f"configs present: {counts}")
            env = dict(os.environ, PTBXL_DIR=ptbxl)  # also runs the real-data test
            job.run([py, "-B", "-W", "ignore", "-m", "unittest", "discover", "-s",
                     os.path.join(HERE, "code", "tests"), "-v"], env=env)
            code, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_used.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")

        with job.step("1b preflight: every run type shortened, then every later step on its outputs"):
            for n in PREFLIGHT["E6_norms"]:
                run(os.path.join(CFG, "E6_norms", n + ".yaml"), pre_runs, sealed=False, smoke_rounds=SHORT, strict=True)
            py_step("e6_norms.py", "--runs", pre_runs, "--out", os.path.join(pre, "e6_norms_smoke.json"), "--smoke")
            made = lines_from(os.path.join(pre, "e6_norms_smoke.json"), "E6_preflight")
            for n in PREFLIGHT["E6_lines"]:
                run(os.path.join(made, n + ".yaml"), pre_runs, sealed=True, smoke_rounds=SHORT, strict=True)
            for k in ("E4", "E4_reference"):
                for n in PREFLIGHT[k]:
                    run(os.path.join(CFG, k, n + ".yaml"), pre_runs, sealed=True, smoke_rounds=SHORT, strict=True)
            for n in PREFLIGHT["E4_positive_control"]:
                run(os.path.join(CFG, "E4_positive_control", n + ".yaml"), pre_runs, sealed=False,
                    smoke_rounds=SHORT, strict=True)
            py_step("e4_positive_control.py", "--runs", pre_runs, "--e1-fedavg", E1_FEDAVG, "--out",
                    os.path.join(pre, "positive_control_smoke.json"), "--smoke")
            job.log("preflight OK: E6 norms, pooling, line configs and runs; E4 attacks on all 5 designs; the P-TA "
                    "reference; the positive control and its check")

        with job.step("2 E4 positive control on FedAvg (DL-43 (7))"):
            ok = all(run(p, state["pc"], sealed=False) for p in configs("E4_positive_control"))
            if ok:
                code, _ = py_step("e4_positive_control.py", "--runs", state["pc"], "--e1-fedavg", E1_FEDAVG,
                                  "--out", pc_out, check=False)
                pc_passed = code == 0
                if not pc_passed:
                    job.note("POSITIVE CONTROL FAILED (DL-43 (7)): the 150 E4 attack runs are not started. "
                             "E6 and the P-TA references still run. A new DL row must strengthen the attack (R19).")
            else:
                job.note("positive control runs incomplete: the E4 attack runs wait for the next start")

        with job.step("3 E6 norm runs, pooled norms, E6 line configs"):
            for p in configs("E6_norms"):
                run(p, state["e6n"], sealed=False)
            if all(done(state["e6n"], os.path.splitext(os.path.basename(p))[0], False) for p in configs("E6_norms")):
                py_step("e6_norms.py", "--runs", state["e6n"], "--out", e6_norms)
                made = lines_from(e6_norms, "E6")
                job.log(f"{len(glob.glob(os.path.join(made, '*.yaml')))} E6 line configs written")
            else:
                job.note("E6 norm runs incomplete: the E6 line runs wait for the next start")

        with job.step("4 E6 line runs (sealed fold 10)"):
            for p in sorted(glob.glob(os.path.join(e6_lines, "*.yaml")), key=lambda q: (seed_of(q), q)):
                run(p, state["e6"], sealed=True)

        with job.step("5 P-TA f = 0 references with the size-aware check (DL-44 (2); sealed fold 10)"):
            for p in configs("E4_reference"):
                run(p, state["e4ref"], sealed=True)

        with job.step("6 E4 runs, seed by seed (sealed fold 10)"):
            if pc_passed:
                for seed in range(10):
                    for p in [q for q in configs("E4") if seed_of(q) == seed]:
                        run(p, state["e4"], sealed=True)
                    job.log(f"E4 seed {seed}: done ({len(skipped)} runs skipped so far)")
            else:
                job.note("E4 attack runs not started (the positive control did not pass or is incomplete)")
        if pc_passed is False:
            status = "PC_FAILED"
        elif skipped or pc_passed is None:
            raise Incomplete()
        else:
            status = "OK"
    except Incomplete:
        status = "INCOMPLETE"
        job.note(f"INCOMPLETE: {len(skipped)} runs skipped: {skipped}. Run the same command again.")
    except StepFailed:
        status = "FAILED"
    except Exception as err:  # still write the zip
        job.log(f"UNEXPECTED ERROR: {type(err).__name__}: {err}")
    finally:
        names = {"pc": "e4_positive_control_runs", "e6n": "e6_norm_runs", "e6": "e6_runs",
                 "e4ref": "e4_reference_runs", "e4": "e4_runs"}
        for key, folder in state.items():
            dest = os.path.join(job.results, names[key])
            os.makedirs(dest, exist_ok=True)
            for f in sorted(os.listdir(folder)):
                if f.endswith(SEALED):  # DL-39: copied, not opened
                    os.makedirs(os.path.join(job.results, "sealed_fold10"), exist_ok=True)
                    shutil.copy2(os.path.join(folder, f), os.path.join(job.results, "sealed_fold10"))
                elif f.endswith(".json"):
                    shutil.copy2(os.path.join(folder, f), dest)
        if os.path.isdir(e6_lines):
            shutil.copytree(e6_lines, os.path.join(job.results, "configs_E6"), dirs_exist_ok=True)
        for f in ("e6_norms_smoke.json", "positive_control_smoke.json"):
            if os.path.exists(os.path.join(pre, f)):
                os.makedirs(os.path.join(job.results, "preflight"), exist_ok=True)
                shutil.copy2(os.path.join(pre, f), os.path.join(job.results, "preflight"))
        job.finish(status)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
