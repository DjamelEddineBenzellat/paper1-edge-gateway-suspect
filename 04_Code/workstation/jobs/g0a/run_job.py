"""Workstation job for gate G0-a (plan tasks G0-14 and G0-15).

Run it with Python 3.12 on the workstation (see README_RUN.txt):
    python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data [--also-eps 8]

Steps (every finished run is kept in state/, so a second start resumes):
 1. record the machine; check the data fingerprints; reuse earlier runs; unit tests;
 1b. preflight: 1 round of every mode on this device;
 2. grid (DL-20, DL-31, DL-32): 9 P-TA runs (3 lr x 3 C_ex, seed 0) and
    3 local-only runs (3 lr, seed 0);
 3. selection: the P-TA default point, C, tau, and the local-only lr and epoch;
 4. pilot (DL-27): P-TA, local-only and P-TG-c, seeds 0-2 (local-only seed 0
    is the identical grid run);
 5. summary: the G0-a numbers (DL-01) and the run times for G0-c.
Steps 2-5 run for eps0 from G0a.yaml, then again for each --also-eps value
(DL-34), in state/eps<value>/; local-only runs are copied, not repeated.
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
    ap = argparse.ArgumentParser(description="Paper 1 G0-a job")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--venv-python", help="default: <base-dir>/venvs/paper1-gpu/bin/python")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gpus-per-node", type=float, default=0.14,
                    help="GPU share per gateway node; use 0.25 if the GPU runs out of memory")
    ap.add_argument("--smoke", action="store_true", help="TEST: 2 rounds, 2 epochs, smaller grid")
    ap.add_argument("--reuse-state", metavar="OLD_JOB_FOLDER",
                    help="copy the finished runs from an earlier G0-a job folder (its state/)")
    ap.add_argument("--also-eps", type=float, action="append", default=[], metavar="EPS",
                    help="after the eps0 stage, also run the grid and pilot at this epsilon (DL-34: 8)")
    args = ap.parse_args()

    base = os.path.abspath(os.path.expanduser(args.base_dir))
    os.makedirs(base, exist_ok=True)
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    py = args.venv_python or (os.path.join(base, "venvs", "paper1-gpu", "Scripts", "python.exe") if os.name == "nt"
                              else os.path.join(base, "venvs", "paper1-gpu", "bin", "python"))
    state = os.path.join(HERE, "state_smoke" if args.smoke else "state")
    os.makedirs(state, exist_ok=True)
    job = Job(HERE, sys.argv)
    status = "FAILED"
    with open(CONFIG, encoding="utf-8") as fh:
        conf_text = fh.read()
    grid_lr = read_yaml_list(conf_text, "lr")
    grid_cex = read_yaml_list(conf_text, "c_ex")
    seeds = read_yaml_list(conf_text, "pilot_seeds")
    margin = read_yaml_list(conf_text, "gate_margin_points")
    smoke = []
    if args.smoke:
        grid_lr, grid_cex, seeds = grid_lr[:2], grid_cex[:1], seeds[:2]
        smoke = ["--rounds", "2", "--local-epochs", "2"]
        job.note("SMOKE TEST: 2 rounds, 2 epochs, reduced grid; not valid for G0-a")

    with open(os.path.join(HERE, "inputs", "data_fingerprints.json"), encoding="utf-8") as fh:
        prints = json.load(fh)
    ptbxl = os.path.join(data_dir, *prints["ptbxl_folder"].split("/"))
    common = [py, os.path.join(CODE, "run_pilot.py"), "--config", CONFIG, "--noise-csv", NOISE,
              "--ptbxl-dir", ptbxl, "--device", args.device,
              "--git-commit", job_value("git_commit") or "unknown"]
    # GPU share per gateway node for attempts 1, 2, 3: a larger share means fewer gateway
    # nodes on the GPU at once (0.14 -> 7, 0.25 -> 4, 0.5 -> 2), so less GPU memory.
    shares = sorted({args.gpus_per_node, max(args.gpus_per_node, 0.25), max(args.gpus_per_node, 0.5)})

    def run(run_id, extra, folder=None, settings=None):
        folder = folder or state
        out = os.path.join(folder, f"{run_id}.json")
        if os.path.exists(out):
            job.log(f"{run_id}: already done, skipped (resume)")
            return out
        tmp = out + ".partial"
        for attempt, share in enumerate(shares, start=1):
            code, _ = job.run(common + (smoke if settings is None else settings) +
                              ["--gpus-per-node", str(share), "--run-id", run_id, "--out", tmp] + extra, check=False)
            if code == 0:
                break
            job.note(f"{run_id}: attempt {attempt} (GPU share {share}) failed with exit code {code}")
        else:
            raise RuntimeError(f"{run_id} failed {len(shares)} times; see logs/run.log")
        os.replace(tmp, out)
        return out

    # Stage 1 uses eps0 from G0a.yaml (3). Each --also-eps adds a stage with its own folder (DL-34).
    stages = [{"label": "", "suffix": "", "folder": state, "eps_args": [], "local_from": None}]
    for eps in args.also_eps:
        tag = f"{eps:g}"
        stages.append({"label": f" (eps {tag})", "suffix": f"_eps{tag}", "folder": os.path.join(state, f"eps{tag}"),
                       "eps_args": ["--eps0", tag], "local_from": state})

    def run_stage(st):
        folder, label, eps_args = st["folder"], st["label"], st["eps_args"]
        os.makedirs(folder, exist_ok=True)
        with job.step(f"2 grid runs{label} (DL-20, DL-31, DL-32)"):
            for lr in grid_lr:
                for c_ex in grid_cex:
                    run(f"pta_grid_lr{lr}_cex{c_ex}_s0", ["--mode", "pta", "--seed", "0", "--lr", str(lr),
                                                         "--c-ex", str(c_ex)] + eps_args, folder=folder)
            if st["local_from"]:
                # local-only uses no DP, so the same runs hold for any epsilon (DL-34)
                for name in sorted(os.listdir(st["local_from"])):
                    if name.startswith("local_") and name.endswith(".json") and \
                            not os.path.exists(os.path.join(folder, name)):
                        shutil.copy2(os.path.join(st["local_from"], name), os.path.join(folder, name))
                        job.log(f"{name}: copied from the eps0 stage (local-only has no DP)")
            for lr in grid_lr:
                run(f"local_grid_lr{lr}_s0", ["--mode", "local", "--seed", "0", "--lr", str(lr)], folder=folder)

        with job.step(f"3 select the default point{label}"):
            sel_path = os.path.join(folder, "selection.json")
            job.run([py, os.path.join(CODE, "select_grid.py"), "--runs", folder, "--out", sel_path])
            with open(sel_path, encoding="utf-8") as fh:
                sel = json.load(fh)
            p, lo = sel["pta_default_point"], sel["local_only"]

        with job.step(f"4 pilot runs{label} (DL-27)"):
            for s in seeds:
                run(f"pta_pilot_s{s}", ["--mode", "pta", "--seed", str(s), "--lr", str(p["lr"]),
                                        "--c-ex", str(p["c_ex"]), "--clip-C", str(p["clip_C"]),
                                        "--tau", str(p["tau"]), "--quantize"] + eps_args, folder=folder)
            for s in seeds:
                target = os.path.join(folder, f"local_pilot_s{s}.json")
                grid_twin = os.path.join(folder, f"local_grid_lr{lo['lr']}_s{s}.json")
                if s == 0 and os.path.exists(grid_twin) and not os.path.exists(target):
                    shutil.copy2(grid_twin, target)  # identical settings and seed: reuse
                    job.log("local_pilot_s0: reused the identical grid run")
                run(f"local_pilot_s{s}", ["--mode", "local", "--seed", str(s), "--lr", str(lo["lr"])], folder=folder)
            for s in seeds:
                run(f"ptgc_pilot_s{s}", ["--mode", "ptgc", "--seed", str(s), "--lr", str(p["lr"]),
                                         "--c-ex", str(p["c_ex"])] + eps_args, folder=folder)

        with job.step(f"5 summary{label} (DL-01, G0-c inputs)"):
            pilot = os.path.join(job.out_dir, "pilot_runs" + st["suffix"])
            os.makedirs(pilot, exist_ok=True)
            for name in os.listdir(folder):
                if "_pilot_s" in name and name.endswith(".json"):
                    shutil.copy2(os.path.join(folder, name), pilot)
            job.run([py, os.path.join(CODE, "summarize_g0a.py"), "--runs", pilot, "--selection",
                     os.path.join(folder, "selection.json"), "--margin", str(margin),
                     "--out", os.path.join(job.results, f"g0a_summary{st['suffix']}.json")])
            shutil.rmtree(pilot)

    try:
        with job.step("1 record the machine and check the data"):
            text, _, _ = system_report(base)
            with open(os.path.join(job.logs, "system.txt"), "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            job.log(text)
            if not os.path.exists(py):
                raise RuntimeError(f"environment not found: {py}. Run the setup job first.")
            bad = []
            for rel, want in prints["files"].items():
                path = os.path.join(data_dir, *rel.split("/"))
                if not os.path.exists(path) or sha256_file(path) != want:
                    bad.append(rel)
            if bad:
                raise RuntimeError(f"data fingerprints do not match for {bad}")
            job.log("data fingerprints OK")
            if args.reuse_state:
                # Finished runs only (a .json file exists only when a run completed). The later fixes
                # (local-only plain model, 15 Nov chunked per-example gradients) do not change the
                # results of finished runs beyond float rounding (tests and checks of 15 Nov).
                old = os.path.join(os.path.abspath(os.path.expanduser(args.reuse_state)), "state")
                if not os.path.isdir(old):
                    raise RuntimeError(f"--reuse-state: no state folder in {args.reuse_state}")
                for name in sorted(os.listdir(old)):
                    if not (name.startswith(("pta_", "local_", "ptgc_")) and name.endswith(".json")):
                        continue
                    target = os.path.join(state, name)
                    if os.path.exists(target):
                        continue
                    with open(os.path.join(old, name), encoding="utf-8") as fh:
                        commit = json.load(fh).get("git_commit")
                    shutil.copy2(os.path.join(old, name), target)
                    job.note(f"reused {name} from {old} (code commit {commit})")
            env = dict(os.environ, PTBXL_DIR=ptbxl)  # also runs the real-data test
            job.run([py, "-B", "-W", "ignore", "-m", "unittest", "discover", "-s",
                     os.path.join(HERE, "code", "tests"), "-v"], env=env)
            code, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_workstation.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")

        with job.step("1b preflight: 1 round of every mode on this device"):
            # Catches memory or setup errors in about 2 minutes instead of after hours.
            # These short runs are not results: they go to a separate folder.
            pre = os.path.join(job.out_dir, "preflight")
            os.makedirs(pre, exist_ok=True)
            quick = ["--rounds", "1", "--local-epochs", "1"]
            run("pre_pta", ["--mode", "pta", "--seed", "0", "--lr", "0.01", "--c-ex", "2.0", "--clip-C", "35",
                            "--tau", "40", "--quantize"], folder=pre, settings=quick)
            run("pre_ptgc", ["--mode", "ptgc", "--seed", "0", "--lr", "0.01", "--c-ex", "2.0"], folder=pre,
                settings=quick)
            run("pre_local", ["--mode", "local", "--seed", "0", "--lr", "0.1"], folder=pre, settings=quick)

        for st in stages:
            run_stage(st)
        status = "TEST_OK" if args.smoke else "OK"
    except StepFailed:
        status = "FAILED"
    except Exception as err:  # still write the zip
        job.log(f"UNEXPECTED ERROR: {type(err).__name__}: {err}")
    finally:
        for st in stages:
            if not os.path.isdir(st["folder"]):
                continue
            runs_out = os.path.join(job.results, "runs" + st["suffix"])
            os.makedirs(runs_out, exist_ok=True)
            for name in sorted(os.listdir(st["folder"])):
                if name.endswith(".json"):
                    shutil.copy2(os.path.join(st["folder"], name), runs_out)
            summary = os.path.join(job.results, f"g0a_summary{st['suffix']}.json")
            selection = os.path.join(st["folder"], "selection.json")
            if not os.path.exists(summary) and os.path.exists(selection):
                # a later step failed: still summarize the pilot runs that finished
                job.log(f"writing a partial summary{st['suffix']} from the finished pilot runs")
                job.run([py, os.path.join(CODE, "summarize_g0a.py"), "--runs", runs_out, "--selection", selection,
                         "--margin", str(margin),
                         "--out", os.path.join(job.results, f"g0a_summary{st['suffix']}_PARTIAL.json")], check=False)
        job.finish(status)
    sys.exit(0 if status.endswith("OK") else 1)


if __name__ == "__main__":
    main()
