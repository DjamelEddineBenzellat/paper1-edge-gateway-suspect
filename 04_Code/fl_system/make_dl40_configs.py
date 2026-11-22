"""Write the DL-40 configs for P-PE's cloud step eta_c (plan task M3-01).

grid:  6 cloud steps x seeds 0-4 = 30 fold-9 runs that never score fold 10.
rerun: the 10 E1 P-PE runs again, with the eta_c chosen by select_ppe_eta.py (sealed fold 10, DL-39).
Every value except eta_c is copied from the E1 config of the same seed, so P-PE changes in eta_c only.

Examples:
    python 04_Code/fl_system/make_dl40_configs.py grid --e1 05_Experiments/configs/E1 \
        --out 05_Experiments/configs/DL40_grid
    python make_dl40_configs.py rerun --e1 configs/E1 --selection selection_dl40.json --out configs/E1_DL40 --commit <c>
"""
import argparse
import hashlib
import json
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
ETAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)  # log grid: the paper's eta (= lr 0.01) up to almost a plain mean
GRID_SEEDS = range(5)
E1_SEEDS = range(10)
E1_JOB = "JOB_20261117_M3-01_e1"


def eta_tag(eta):
    return f"{eta:g}".replace(".", "p")


def e1_config(e1_dir, seed):
    with open(os.path.join(e1_dir, f"E1_ppe_s{seed}.yaml"), "rb") as fh:
        raw = fh.read()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def write(out_dir, cfg, header):
    path = os.path.join(out_dir, f"{cfg['run_id']}.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def make_grid(e1_dir, out_dir, commit):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for seed in GRID_SEEDS:
        base, base_sha = e1_config(e1_dir, seed)
        for eta in ETAS:
            cfg = dict(base, experiment="DL-40 grid", run_id=f"DL40_ppe_eta{eta_tag(eta)}_s{seed}",
                       settings=dict(base["settings"], cloud_eta=eta), fold10="none",
                       rules=base["rules"] + ["DL-40"], code_commit=commit,
                       sources_sha256=dict(base["sources_sha256"], **{f"E1_ppe_s{seed}.yaml": base_sha}))
            written.append(write(out_dir, cfg, "# DL-40 grid config, written by make_dl40_configs.py. "
                                               "Do not edit by hand.\n"))
    return written


def make_rerun(e1_dir, selection_path, out_dir, commit):
    with open(selection_path, "rb") as fh:
        raw = fh.read()
    sel = json.loads(raw)
    eta = sel["selected_cloud_eta"]
    if eta is None:
        raise SystemExit("no cloud step was selected (DL-40 stop rule): E1 is not run again")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for seed in E1_SEEDS:
        base, base_sha = e1_config(e1_dir, seed)
        cfg = dict(base, settings=dict(base["settings"], cloud_eta=eta), rules=base["rules"] + ["DL-40"],
                   code_commit=commit, supersedes=f"E1_ppe_s{seed} of {E1_JOB} (cloud step = lr)",
                   dl40_selection={"file": os.path.basename(selection_path), "cloud_eta": eta,
                                   "smoke": bool(sel.get("smoke"))},
                   sources_sha256=dict(base["sources_sha256"], **{f"E1_ppe_s{seed}.yaml": base_sha,
                                                                  os.path.basename(selection_path):
                                                                      hashlib.sha256(raw).hexdigest()}))
        written.append(write(out_dir, cfg, "# E1 P-PE config with the DL-40 cloud step, written by "
                                           "make_dl40_configs.py. Do not edit by hand.\n"))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["grid", "rerun"])
    ap.add_argument("--e1", required=True, help="folder with the E1 configs (E1_ppe_s<seed>.yaml)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", help="rerun: selection_dl40.json from select_ppe_eta.py")
    ap.add_argument("--commit", help="code commit to record (default: git HEAD)")
    args = ap.parse_args(argv)
    commit = args.commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True,
                                           text=True, check=True).stdout.strip()
    if args.what == "grid":
        written = make_grid(args.e1, args.out, commit)
    else:
        if not args.selection:
            ap.error("rerun needs --selection")
        written = make_rerun(args.e1, args.selection, args.out, commit)
    print(f"{len(written)} configs written to {args.out} (code commit {commit[:7]})", flush=True)


if __name__ == "__main__":
    main()
