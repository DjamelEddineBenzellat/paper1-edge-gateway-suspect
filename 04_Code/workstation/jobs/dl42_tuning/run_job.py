"""Workstation job: DL-42, the same fold-9 grid for the five DP designs (plan tasks X-1/X-2).

Run it with Python 3.12 on the workstation (see README_RUN.txt):
    python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data

Steps (every finished run is kept in state_grid/ or state_rerun/, so a second start resumes):
 1. record the machine; check the data fingerprints, the configs and the reused runs; unit tests;
 1b. preflight (about 10 minutes): one shortened grid run per design, then every later step on those outputs
     (reuse check, selection, rerun configs, two shortened reruns with sealed fold 10, summary);
 2. reuse check: P-TG, P-TG-c and P-PE at the shared point, seed 0, must reproduce the reused E1/DL-40 runs;
    a design that does not is run at the shared point for seeds 1-2 too, and those runs are used instead;
 3. the grid on fold 9: 126 runs, seed by seed; no fold 10;
 4. select each design's point by the DL-42 rule (select_dl42.py);
 5. the 10 E1 seeds again for each design whose point changed (sealed fold-10 files, DL-39);
 6. summary of the new runs on fold 9 only.
A run that fails 3 times is skipped and listed; the job then ends with status INCOMPLETE, and the same command
runs only the missing runs (risk plan R13). The results zip is always made, also if a step fails.
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
GRID = os.path.join(HERE, "configs", "DL42_grid")
CHECK = os.path.join(HERE, "configs", "DL42_shared_check")
E1_CONFIGS = os.path.join(HERE, "configs", "E1")
DL40_CONFIGS = os.path.join(HERE, "configs", "E1_DL40")
REUSE = os.path.join(HERE, "inputs", "reuse")
NOISE = os.path.join(HERE, "inputs", "G0a_noise_multipliers.csv")
MODES = ["ptg", "ptgc", "ppe", "pfr", "pta"]  # as make_dl42_configs.DESIGNS
REUSE_MODES = ["ptg", "ptgc", "ppe"]
POINTS = [(lr, c_ex) for lr in (0.01, 0.05, 0.1) for c_ex in (0.5, 1.0, 2.0)]
SHARED = (0.01, 2.0)
SEEDS = [0, 1, 2]
SEALED = ".fold10.sealed.json"
SHORT = 5  # P-PE: device rounds must be a multiple of T_e = 5


class Incomplete(Exception):
    """Some runs were skipped after 3 failed attempts (R13)."""


def point_tag(lr, c_ex):  # as make_dl42_configs.point_tag
    return f"lr{lr:g}_cex{c_ex:g}".replace(".", "p")


def grid_id(mode, point, seed):
    return f"DL42_{mode}_{point_tag(*point)}_s{seed}"


def job_value(key):
    with open(os.path.join(HERE, "job.yaml"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip().strip('"')
    return None


def main():
    ap = argparse.ArgumentParser(description="Paper 1 DL-42 job (fair per-design tuning)")
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
    state_grid, state_rerun = os.path.join(HERE, "state_grid"), os.path.join(HERE, "state_rerun")
    pre = os.path.join(HERE, "preflight")  # not zipped: shortened runs only (their sealed files are tests)
    rerun_configs = os.path.join(HERE, "configs", "E1_DL42")  # written in step 5 from the selection
    for d in (state_grid, state_rerun, os.path.join(pre, "runs")):
        os.makedirs(d, exist_ok=True)
    job = Job(HERE, sys.argv)
    status = "FAILED"
    skipped = []

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

    def select(what, runs, out, extra=()):
        job.run([py, os.path.join(CODE, "select_dl42.py"), what, "--runs", runs, "--reuse", REUSE, "--out", out]
                + list(extra))
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)

    selection = os.path.join(job.results, "selection_dl42.json")
    summary = os.path.join(job.results, "e1_tuned_summary_fold9.json")
    try:
        with job.step("1 record the machine and check the data, configs and reused runs"):
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
            need = [os.path.join(CHECK if (m in REUSE_MODES and p == SHARED) else GRID, grid_id(m, p, s) + ".yaml")
                    for s in SEEDS for m in MODES for p in POINTS]
            need += [os.path.join(DL40_CONFIGS if m == "ppe" else E1_CONFIGS, f"E1_{m}_s{s}.yaml")
                     for m in MODES for s in range(10)]
            need += [os.path.join(REUSE, f"E1_{m}_s{s}.json") for m in REUSE_MODES for s in SEEDS]
            missing = [os.path.relpath(p, HERE) for p in need if not os.path.exists(p)]
            if missing:
                raise RuntimeError(f"configs or reused runs missing: {missing}")
            if len(os.listdir(GRID)) != 126 or len(os.listdir(CHECK)) != 9:
                raise RuntimeError("expected 126 grid configs and 9 shared-check configs")
            job.log("135 grid/check configs, 50 E1 configs and 9 reused runs present")
            env = dict(os.environ, PTBXL_DIR=ptbxl)  # also runs the real-data test
            job.run([py, "-B", "-W", "ignore", "-m", "unittest", "discover", "-s",
                     os.path.join(HERE, "code", "tests"), "-v"], env=env)
            code, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_used.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")

        with job.step("1b preflight: every run type shortened, then every later step on its outputs"):
            pre_runs = os.path.join(pre, "runs")
            for m in MODES:  # P-TA/P-FR with C, tau and quantization off and saved norms; P-PE with eta_c 1.0
                run(os.path.join(GRID, grid_id(m, (0.05, 1.0), 0) + ".yaml"), pre_runs, sealed=False,
                    smoke_rounds=SHORT, strict=True)
            select("check", pre_runs, os.path.join(pre, "reuse_check_smoke.json"), ["--smoke"])
            select("select", pre_runs, os.path.join(pre, "selection_smoke.json"), ["--n-seeds", "1", "--smoke"])
            pre_configs = os.path.join(HERE, "configs", "E1_DL42_preflight")  # next to G0a.yaml (../G0a.yaml)
            job.run([py, os.path.join(CODE, "make_dl42_configs.py"), "rerun", "--e1", E1_CONFIGS, "--dl40",
                     DL40_CONFIGS, "--selection", os.path.join(pre, "selection_smoke.json"), "--out", pre_configs,
                     "--commit", commit])
            for m in ("pta", "pfr"):  # C (and tau) from the norms, quantization on, sealed fold 10
                run(os.path.join(pre_configs, f"E1tuned_{m}_s0.yaml"), pre_runs, sealed=True, smoke_rounds=SHORT,
                    strict=True)
            job.run([py, os.path.join(CODE, "summarize_e1.py"), "--runs", pre_runs, "--pattern", "E1tuned_*.json",
                     "--out", os.path.join(pre, "summary_smoke.json")])
            job.log("preflight OK: grid runs, reuse check, selection, rerun configs, sealed reruns and summary work")

        with job.step("2 reuse check: the shared point of P-TG, P-TG-c and P-PE, seed 0, run again"):
            for m in REUSE_MODES:
                run(os.path.join(CHECK, grid_id(m, SHARED, 0) + ".yaml"), state_grid, sealed=False)
            check = select("check", state_grid, os.path.join(job.results, "reuse_check.json"))
            for m, v in check.items():
                if v is not None and not v["match"]:
                    job.note(f"{m}: the reused E1 run is NOT reproduced (difference {v['max_abs_diff']:.2e}); its "
                             f"shared point is run for seeds 1-2 too, and the new runs are used")
                    for s in SEEDS[1:]:
                        run(os.path.join(CHECK, grid_id(m, SHARED, s) + ".yaml"), state_grid, sealed=False)

        with job.step("3 the DL-42 grid on fold 9: 126 runs, seed by seed"):
            for s in SEEDS:
                for m in MODES:
                    for p in POINTS:
                        if not (m in REUSE_MODES and p == SHARED):
                            run(os.path.join(GRID, grid_id(m, p, s) + ".yaml"), state_grid, sealed=False)
                job.log(f"grid seed {s}: done ({len(skipped)} runs skipped so far)")
        if skipped:
            raise Incomplete()

        with job.step("4 select each design's point (DL-42 rule)"):
            chosen = select("select", state_grid, selection)
            job.log(f"designs that change their point: {chosen['changed'] or 'none'}")

        with job.step("5 the 10 E1 seeds at the new point of each changed design (sealed fold 10)"):
            job.run([py, os.path.join(CODE, "make_dl42_configs.py"), "rerun", "--e1", E1_CONFIGS, "--dl40",
                     DL40_CONFIGS, "--selection", selection, "--out", rerun_configs, "--commit", commit])
            for m in chosen["changed"]:
                for s in range(10):
                    run(os.path.join(rerun_configs, f"E1tuned_{m}_s{s}.yaml"), state_rerun, sealed=True)
        if skipped:
            raise Incomplete()

        if chosen["changed"]:
            with job.step("6 summary of the new runs on fold 9 only (sealed fold 10 not opened)"):
                job.run([py, os.path.join(CODE, "summarize_e1.py"), "--runs", state_rerun, "--pattern",
                         "E1tuned_*.json", "--out", summary])
        else:
            job.note("no design changed its point: the E1 runs stand, no rerun was needed")
        status = "OK"
    except Incomplete:
        status = "INCOMPLETE"
        job.note(f"INCOMPLETE: {len(skipped)} runs skipped: {skipped}. Run the same command again.")
    except StepFailed:
        status = "FAILED"
    except Exception as err:  # still write the zip
        job.log(f"UNEXPECTED ERROR: {type(err).__name__}: {err}")
    finally:
        for name, folder in {"grid_runs": state_grid, "tuned_runs": state_rerun}.items():
            dest = os.path.join(job.results, name)
            os.makedirs(dest, exist_ok=True)
            for f in sorted(os.listdir(folder)):
                if f.endswith(SEALED):  # DL-39: copied, not opened
                    os.makedirs(os.path.join(job.results, "sealed_fold10"), exist_ok=True)
                    shutil.copy2(os.path.join(folder, f), os.path.join(job.results, "sealed_fold10"))
                elif f.endswith(".json"):
                    shutil.copy2(os.path.join(folder, f), dest)
        if os.path.isdir(rerun_configs):
            shutil.copytree(rerun_configs, os.path.join(job.results, "configs_E1_DL42"), dirs_exist_ok=True)
        for f in ("reuse_check_smoke.json", "selection_smoke.json", "summary_smoke.json"):
            if os.path.exists(os.path.join(pre, f)):
                os.makedirs(os.path.join(job.results, "preflight"), exist_ok=True)
                shutil.copy2(os.path.join(pre, f), os.path.join(job.results, "preflight"))
        if not os.path.exists(selection) and os.listdir(os.path.join(job.results, "grid_runs")):
            job.log("writing a partial grid table (no selection: the grid is not complete)")
            job.run([py, os.path.join(CODE, "select_dl42.py"), "select", "--runs",
                     os.path.join(job.results, "grid_runs"), "--reuse", REUSE, "--out",
                     os.path.join(job.results, "selection_dl42_PARTIAL.json"), "--partial"], check=False)
        job.finish(status)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
