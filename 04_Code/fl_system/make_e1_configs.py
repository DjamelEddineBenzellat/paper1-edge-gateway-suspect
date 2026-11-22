"""Write the 90 E1 run configs, one file per run (plan task M2-05; PR §6, PR:320).

E1 = 9 configurations x seeds 0-9. All values come from the selection files, not typed by hand:
- DP placements (P-TA, P-TG, P-TG-c, P-PE, P-FR): the DL-36 default point at eps0 = 8 (DL-34);
  C and quantization for P-TA and P-FR, tau for P-TA only (DL-37 point 4).
- Local-only: lr and epoch of the DL-36 grid. Centralized, FedAvg, HierFAVG: the DL-37 baseline grid.
Every config records the code commit, the pinned library versions and the SHA-256 of its sources.

Example:
    python 04_Code/fl_system/make_e1_configs.py --out 05_Experiments/configs/E1
"""
import argparse
import hashlib
import json
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
SEL_DL36 = "05_Experiments/results/G0-14/JOB_20261115_G0-14_grid-dl36/selection_dl36.json"
SEL_DL37 = "05_Experiments/results/M2-04/JOB_20261116_M2-04_baseline-grid/selection_baselines.json"
REQUIREMENTS = "05_Experiments/env/requirements_G0.txt"
EPS0 = 8.0  # DL-34
SEEDS = range(10)  # PR:319
PINNED = ("torch", "opacus", "flwr", "numpy", "scikit-learn", "scipy", "cryptography")
# (name in the paper, mode); the order of PR §6 / v4 E1
CONFIGURATIONS = [("P-TG", "ptg"), ("P-TG-c", "ptgc"), ("P-PE", "ppe"), ("P-FR", "pfr"), ("P-TA", "pta"),
                  ("centralized", "centralized"), ("local-only", "local"), ("FedAvg", "fedavg"),
                  ("HierFAVG", "hierfavg")]


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def pinned_versions():
    out = {}
    with open(os.path.join(PROJECT, REQUIREMENTS), encoding="utf-8") as fh:
        for line in fh:
            name, _, version = line.strip().partition("==")
            if name in PINNED:
                out[name] = version
    return out


def settings_for(mode, dl36, dl37):
    p, lo = dl36["pta_default_point"], dl36["local_only"]
    if mode in ("pta", "ptg", "ptgc", "ppe", "pfr"):
        s = {"lr": p["lr"], "c_ex": p["c_ex"], "eps0": EPS0, "rounds": 100}
        if mode in ("pta", "pfr"):
            s.update({"clip_C": p["clip_C"], "quantize": True})
        if mode == "pta":
            s["tau"] = p["tau"]
        return s
    if mode == "local":
        return {"lr": lo["lr"], "local_epochs": lo["epoch"]}
    if mode == "centralized":
        return {"lr": dl37["centralized"]["lr"], "local_epochs": dl37["centralized"]["epoch"]}
    return {"lr": dl37[mode]["lr"], "rounds": 100}  # fedavg, hierfavg


def make(out_dir, commit):
    with open(os.path.join(PROJECT, SEL_DL36), encoding="utf-8") as fh:
        dl36 = json.load(fh)
    with open(os.path.join(PROJECT, SEL_DL37), encoding="utf-8") as fh:
        dl37 = json.load(fh)
    sources = {SEL_DL36: sha256(os.path.join(PROJECT, SEL_DL36)), SEL_DL37: sha256(os.path.join(PROJECT, SEL_DL37))}
    versions = pinned_versions()
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, mode in CONFIGURATIONS:
        for seed in SEEDS:
            run_id = f"E1_{mode}_s{seed}"
            cfg = {"experiment": "E1", "run_id": run_id, "configuration": name, "mode": mode, "seed": seed,
                   "settings": settings_for(mode, dl36, dl37), "canaries": True,
                   "fold10": "sealed (DL-39)", "base_config": "../G0a.yaml",
                   "rules": ["DL-34", "DL-36", "DL-37", "DL-38", "DL-39"],
                   "code_commit": commit, "versions": versions, "sources_sha256": sources}
            path = os.path.join(out_dir, f"{run_id}.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# E1 run config, written by 04_Code/fl_system/make_e1_configs.py. Do not edit by hand.\n")
                yaml.safe_dump(cfg, fh, sort_keys=False)
            written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", help="code commit to record (default: git HEAD)")
    args = ap.parse_args()
    commit = args.commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True,
                                           text=True, check=True).stdout.strip()
    written = make(args.out, commit)
    print(f"{len(written)} configs written to {args.out} (code commit {commit[:7]})")


if __name__ == "__main__":
    main()
