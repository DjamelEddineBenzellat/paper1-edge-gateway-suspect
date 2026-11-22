"""Workstation job for the DL-35 diagnostic (after G0-a failed at eps = 8). No gate decision.

Run it with Python 3.12 on the workstation (see README_RUN.txt):
    python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data

Steps (every finished run is kept in state/, so a second start resumes):
 1. record the machine; check the data fingerprints; unit tests;
 1b. preflight: 1 round of every mode on this device, with the new options;
 2. DL-35 (a): P-TA and P-TG-c WITHOUT DP noise (z = 0), seeds 0-2, at the
    eps = 8 default point (lr, C_ex); update clip C, tau and quantization off;
 3. DL-35 (b): the eps = 8 pilot again (P-TA, P-TG-c, local-only, seeds 0-2,
    same settings), with diagnostics and the fold-9 scores of P-TA and P-TG-c;
 4. summary: diag_summary.json (summarize_diag.py).
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
SELECTION = os.path.join(HERE, "inputs", "selection_eps8.json")  # from JOB_20261115_G0-14_g0a-eps3-eps8
EARLIER = os.path.join(HERE, "inputs", "g0a_summary_eps8.json")
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
    ap = argparse.ArgumentParser(description="Paper 1 DL-35 diagnostic job")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--venv-python", help="default: <base-dir>/venvs/paper1-gpu/bin/python")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gpus-per-node", type=float, default=0.14,
                    help="GPU share per gateway node; use 0.25 if the GPU runs out of memory")
    ap.add_argument("--smoke", action="store_true", help="TEST: 2 rounds, 2 epochs, 2 seeds")
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
        seeds = read_yaml_list(fh.read(), "pilot_seeds")
    with open(SELECTION, encoding="utf-8") as fh:
        sel = json.load(fh)
    p, lo = sel["pta_default_point"], sel["local_only"]
    local_epoch = str(lo["epoch"])
    smoke = []
    if args.smoke:
        seeds = seeds[:2]
        smoke = ["--rounds", "2", "--local-epochs", "2"]
        local_epoch = "2"
        job.note("SMOKE TEST: 2 rounds, 2 epochs, 2 seeds; not valid for DL-35")

    with open(os.path.join(HERE, "inputs", "data_fingerprints.json"), encoding="utf-8") as fh:
        prints = json.load(fh)
    ptbxl = os.path.join(data_dir, *prints["ptbxl_folder"].split("/"))
    common = [py, os.path.join(CODE, "run_pilot.py"), "--config", CONFIG, "--noise-csv", NOISE,
              "--ptbxl-dir", ptbxl, "--device", args.device,
              "--git-commit", job_value("git_commit") or "unknown"]
    # GPU share per gateway node for attempts 1, 2, 3: a larger share means fewer gateway
    # nodes on the GPU at once (0.14 -> 7, 0.25 -> 4, 0.5 -> 2), so less GPU memory.
    shares = sorted({args.gpus_per_node, max(args.gpus_per_node, 0.25), max(args.gpus_per_node, 0.5)})

    def run(run_id, extra, folder=None, settings=None, scores=False):
        folder = folder or state
        out = os.path.join(folder, f"{run_id}.json")
        if os.path.exists(out):
            job.log(f"{run_id}: already done, skipped (resume)")
            return out
        tmp = out + ".partial"
        if scores:
            extra = extra + ["--scores-out", os.path.join(folder, f"{run_id}_fold9_scores.csv.gz")]
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

    point = ["--lr", str(p["lr"]), "--c-ex", str(p["c_ex"]), "--eps0", EPS, "--diagnostics"]
    pta_filters = ["--clip-C", str(p["clip_C"]), "--tau", str(p["tau"]), "--quantize"]
    summary = os.path.join(job.results, "diag_summary.json")

    def summarize(out, check=True):
        return job.run([py, os.path.join(CODE, "summarize_diag.py"), "--runs", state, "--local-epoch", local_epoch,
                        "--earlier", EARLIER, "--out", out], check=check)

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
            job.log(f"eps = 8 default point (inputs/selection_eps8.json): {json.dumps(p)}; local-only {json.dumps(lo)}")
            env = dict(os.environ, PTBXL_DIR=ptbxl)  # also runs the real-data test
            job.run([py, "-B", "-W", "ignore", "-m", "unittest", "discover", "-s",
                     os.path.join(HERE, "code", "tests"), "-v"], env=env)
            code, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_workstation.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")

        with job.step("1b preflight: 1 round of every mode on this device"):
            # Catches memory or setup errors in minutes instead of after an hour.
            # These short runs are not results: they go to a separate folder.
            pre = os.path.join(job.out_dir, "preflight")
            os.makedirs(pre, exist_ok=True)
            quick = ["--rounds", "1", "--local-epochs", "1"]
            run("pre_nonoise_pta", ["--mode", "pta", "--seed", "0", "--no-noise"] + point, folder=pre,
                settings=quick, scores=True)
            run("pre_pta", ["--mode", "pta", "--seed", "0"] + point + pta_filters, folder=pre, settings=quick)
            run("pre_ptgc", ["--mode", "ptgc", "--seed", "0"] + point, folder=pre, settings=quick)
            run("pre_local", ["--mode", "local", "--seed", "0", "--lr", str(lo["lr"]), "--diagnostics"], folder=pre,
                settings=quick)

        with job.step("2 DL-35 (a): P-TA and P-TG-c without DP noise"):
            for s in seeds:
                run(f"nonoise_pta_s{s}", ["--mode", "pta", "--seed", str(s), "--no-noise"] + point, scores=True)
            for s in seeds:
                run(f"nonoise_ptgc_s{s}", ["--mode", "ptgc", "--seed", str(s), "--no-noise"] + point, scores=True)

        with job.step("3 DL-35 (b): the eps = 8 pilot again, with diagnostics"):
            for s in seeds:
                run(f"pta_pilot_s{s}", ["--mode", "pta", "--seed", str(s)] + point + pta_filters, scores=True)
            for s in seeds:
                run(f"ptgc_pilot_s{s}", ["--mode", "ptgc", "--seed", str(s)] + point, scores=True)
            for s in seeds:
                run(f"local_pilot_s{s}", ["--mode", "local", "--seed", str(s), "--lr", str(lo["lr"]),
                                          "--diagnostics"])

        with job.step("4 summary (DL-35, no gate decision)"):
            summarize(summary)
        status = "TEST_OK" if args.smoke else "OK"
    except StepFailed:
        status = "FAILED"
    except Exception as err:  # still write the zip
        job.log(f"UNEXPECTED ERROR: {type(err).__name__}: {err}")
    finally:
        runs_out = os.path.join(job.results, "runs")
        os.makedirs(runs_out, exist_ok=True)
        for name in sorted(os.listdir(state)):
            if name.endswith((".json", ".csv.gz")):
                shutil.copy2(os.path.join(state, name), runs_out)
        if not os.path.exists(summary) and any(n.endswith(".json") for n in os.listdir(state)):
            # a later step failed: still summarize the runs that finished
            job.log("writing a partial summary from the finished runs")
            summarize(os.path.join(job.results, "diag_summary_PARTIAL.json"), check=False)
        job.finish(status)
    sys.exit(0 if status.endswith("OK") else 1)


if __name__ == "__main__":
    main()
