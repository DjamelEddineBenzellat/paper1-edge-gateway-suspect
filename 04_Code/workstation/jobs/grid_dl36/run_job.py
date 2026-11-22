"""Workstation job: the eps = 8 grid again, scored with the DL-36 metric (before E1).

Run it with Python 3.12 on the workstation (see README_RUN.txt):
    python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data

Steps (every finished run is kept in state/, so a second start resumes):
 1. record the machine; check the data fingerprints; unit tests;
 1b. preflight: 1 round of P-TA and 1 epoch of local-only on this device;
 2. grid (DL-20, DL-31, DL-32, DL-36): 9 P-TA runs (3 lr x 3 C_ex, seed 0, eps 8;
    C, tau and quantization off) and 3 local-only runs (3 lr, seed 0);
 3. selection with the DL-36 metric: the default point, C, tau, and the
    local-only lr and epoch; compared with the old (threshold 0.5) selection.
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
OLD_SELECTION = os.path.join(HERE, "inputs", "selection_eps8_dl33.json")  # threshold 0.5 (DL-33)
EPS = "8"  # DL-34: eps0 = 8


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
    ap = argparse.ArgumentParser(description="Paper 1 DL-36 grid job")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--venv-python", help="default: <base-dir>/venvs/paper1-gpu/bin/python")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gpus-per-node", type=float, default=0.14,
                    help="GPU share per gateway node; use 0.25 if the GPU runs out of memory")
    ap.add_argument("--smoke", action="store_true", help="TEST: 2 rounds, 2 epochs, smaller grid")
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
    local_lr = list(grid_lr)
    smoke = []
    if args.smoke:
        grid_lr, grid_cex, local_lr = grid_lr[:2], grid_cex[:1], local_lr[:1]
        smoke = ["--rounds", "2", "--local-epochs", "2"]
        job.note("SMOKE TEST: 2 rounds, 2 epochs, reduced grid; not valid for DL-36")

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

    selection = os.path.join(job.results, "selection_dl36.json")
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
            env = dict(os.environ, PTBXL_DIR=ptbxl)  # also runs the real-data test
            job.run([py, "-B", "-W", "ignore", "-m", "unittest", "discover", "-s",
                     os.path.join(HERE, "code", "tests"), "-v"], env=env)
            code, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_workstation.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")

        with job.step("1b preflight: 1 round of P-TA and 1 epoch of local-only on this device"):
            # Catches memory or setup errors in minutes. These short runs are not results.
            pre = os.path.join(job.out_dir, "preflight")
            os.makedirs(pre, exist_ok=True)
            quick = ["--rounds", "1", "--local-epochs", "1"]
            run("pre_pta", ["--mode", "pta", "--seed", "0", "--lr", "0.1", "--c-ex", "1.0", "--eps0", EPS],
                folder=pre, settings=quick)
            run("pre_local", ["--mode", "local", "--seed", "0", "--lr", "0.1"], folder=pre, settings=quick)

        with job.step("2 grid runs at eps 8 (DL-20, DL-31, DL-32, DL-36)"):
            for lr in grid_lr:
                for c_ex in grid_cex:
                    run(f"pta_grid_lr{lr}_cex{c_ex}_s0", ["--mode", "pta", "--seed", "0", "--lr", str(lr),
                                                         "--c-ex", str(c_ex), "--eps0", EPS])
            for lr in local_lr:
                run(f"local_grid_lr{lr}_s0", ["--mode", "local", "--seed", "0", "--lr", str(lr)])

        with job.step("3 selection with the DL-36 metric"):
            job.run([py, os.path.join(CODE, "select_grid.py"), "--runs", state, "--out", selection])
            with open(selection, encoding="utf-8") as fh:
                new = json.load(fh)
            with open(OLD_SELECTION, encoding="utf-8") as fh:
                old = json.load(fh)
            for part, keys in (("pta_default_point", ("lr", "c_ex", "clip_C", "tau")),
                               ("local_only", ("lr", "epoch"))):
                job.log(f"{part}: old (threshold 0.5) " + json.dumps({k: old[part][k] for k in keys}) +
                        " -> new (DL-36) " + json.dumps({k: new[part][k] for k in keys}))
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
