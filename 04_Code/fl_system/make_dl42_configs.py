"""Write the DL-42 configs: the same fold-9 grid for the five DP designs (plan task X-1).

grid:  lr {0.01, 0.05, 0.1} x C_ex {0.5, 1.0, 2.0} x seeds 0-2 for P-TG, P-TG-c, P-PE, P-FR and P-TA; fold 9 only.
       - P-TA and P-FR: C, tau and quantization off, every norm saved (DL-31; C and tau come from these norms).
       - P-TG, P-TG-c and P-PE: the shared point (lr 0.01, C_ex 2.0) reuses the E1 runs of seeds 0-2 (P-PE: the
         DL-40 runs), so it is not in grid/. Its configs go to shared_check/: seed 0 checks that this code
         reproduces the reused runs; seeds 1-2 run only if that check fails.
       126 grid configs + 9 shared-check configs.
rerun: the 10 E1 seeds again, at the selected point, for each design whose point changed (select_dl42.py);
       sealed fold 10 (DL-39). Run IDs E1tuned_<mode>_s<seed>.
Every value except lr and C_ex (and C, tau and quantization for P-TA and P-FR) is copied from the E1 config of the
same design and seed; for P-PE from the DL-40 config (eta_c = 1.0). Canaries, partition and G0a.yaml stay as in E1.

Examples:
    python 04_Code/fl_system/make_dl42_configs.py grid --e1 05_Experiments/configs/E1 \
        --dl40 05_Experiments/results/M3-01/JOB_20261117_M3-01_dl40-ppe/configs_E1_DL40 \
        --out 05_Experiments/configs
    python make_dl42_configs.py rerun --e1 configs/E1 --dl40 configs/E1_DL40 --selection selection_dl42.json \
        --out configs/E1_DL42 --commit <c>
"""
import argparse
import hashlib
import json
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
LRS = (0.01, 0.05, 0.1)  # PR:58
C_EXS = (0.5, 1.0, 2.0)  # PR:59
SHARED = (0.01, 2.0)  # the E1 point (DL-36, DL-37)
DESIGNS = {"ptg": "P-TG", "ptgc": "P-TG-c", "ppe": "P-PE", "pfr": "P-FR", "pta": "P-TA"}
REUSE_MODES = ("ptg", "ptgc", "ppe")  # shared point from the E1 / DL-40 runs
NORM_MODES = ("pta", "pfr")  # C (and tau for P-TA) from the norms at the selected point (DL-31)
GRID_SEEDS = range(3)
E1_SEEDS = range(10)
GRID_DIR, CHECK_DIR = "DL42_grid", "DL42_shared_check"


def point_tag(lr, c_ex):
    return f"lr{lr:g}_cex{c_ex:g}".replace(".", "p")


def grid_run_id(mode, lr, c_ex, seed):
    return f"DL42_{mode}_{point_tag(lr, c_ex)}_s{seed}"


def base_config(e1_dir, dl40_dir, mode, seed):
    """The E1 config of this design and seed (P-PE: the DL-40 one), its file name and SHA-256."""
    name = f"E1_{mode}_s{seed}.yaml"
    with open(os.path.join(dl40_dir if mode == "ppe" else e1_dir, name), "rb") as fh:
        raw = fh.read()
    cfg = yaml.safe_load(raw)
    if (cfg["mode"], cfg["seed"]) != (mode, seed) or (mode == "ppe" and cfg["settings"].get("cloud_eta") != 1.0):
        raise SystemExit(f"{name}: not the E1 config of {mode} seed {seed} (P-PE needs eta_c = 1.0, DL-40)")
    return cfg, name, hashlib.sha256(raw).hexdigest()


def write(out_dir, cfg, header):
    path = os.path.join(out_dir, f"{cfg['run_id']}.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def grid_config(base, base_name, base_sha, lr, c_ex, commit):
    mode = base["mode"]
    s = dict(base["settings"], lr=lr, c_ex=c_ex)
    if mode in NORM_MODES:  # DL-31: the norms are recorded without C, tau or quantization
        for key in ("clip_C", "tau", "quantize"):
            s.pop(key, None)
        s["save_norms"] = True
    cfg = dict(base, experiment="DL-42 grid", run_id=grid_run_id(mode, lr, c_ex, base["seed"]), settings=s,
               fold10="none", rules=base["rules"] + ["DL-42"], code_commit=commit,
               sources_sha256=dict(base["sources_sha256"], **{base_name: base_sha}))
    cfg.pop("supersedes", None)  # P-PE: the DL-40 note belongs to the E1 rerun only
    return cfg


def make_grid(e1_dir, dl40_dir, out_root, commit):
    """126 grid configs in DL42_grid/ and 9 shared-point check configs in DL42_shared_check/."""
    grid_dir, check_dir = os.path.join(out_root, GRID_DIR), os.path.join(out_root, CHECK_DIR)
    for d in (grid_dir, check_dir):
        os.makedirs(d, exist_ok=True)
    grid, check = [], []
    for seed in GRID_SEEDS:
        for mode in DESIGNS:
            base, name, sha = base_config(e1_dir, dl40_dir, mode, seed)
            for lr in LRS:
                for c_ex in C_EXS:
                    cfg = grid_config(base, name, sha, lr, c_ex, commit)
                    if mode in REUSE_MODES and (lr, c_ex) == SHARED:
                        check.append(write(check_dir, cfg, "# DL-42 shared-point check config (reuse check), "
                                                           "written by make_dl42_configs.py. Do not edit by hand.\n"))
                    else:
                        grid.append(write(grid_dir, cfg, "# DL-42 grid config, written by make_dl42_configs.py. "
                                                         "Do not edit by hand.\n"))
    return grid, check


def make_rerun(e1_dir, dl40_dir, selection_path, out_dir, commit):
    """The 10 E1 seeds at the selected point of every design whose point changed; [] if none changed."""
    with open(selection_path, "rb") as fh:
        raw = fh.read()
    sel = json.loads(raw)
    if sel.get("partial"):
        raise SystemExit("a partial grid table selects nothing: no rerun configs")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for mode, d in sel["designs"].items():
        if d["decision"] != "change":
            continue
        point = d["selected"]
        for seed in E1_SEEDS:
            base, name, sha = base_config(e1_dir, dl40_dir, mode, seed)
            s = dict(base["settings"], lr=point["lr"], c_ex=point["c_ex"])
            if mode in NORM_MODES:
                s.update(clip_C=point["clip_C"], quantize=True)
                if mode == "pta":
                    s["tau"] = point["tau"]
            cfg = dict(base, experiment="E1 tuned (DL-42)", run_id=f"E1tuned_{mode}_s{seed}", settings=s,
                       rules=base["rules"] + ["DL-42"], code_commit=commit,
                       dl42_selection={"file": os.path.basename(selection_path), "point": point,
                                       "smoke": bool(sel.get("smoke"))},
                       sources_sha256=dict(base["sources_sha256"], **{name: sha, os.path.basename(selection_path):
                                                                      hashlib.sha256(raw).hexdigest()}))
            cfg.pop("supersedes", None)
            written.append(write(out_dir, cfg, "# E1 config at the DL-42 selected point, written by "
                                               "make_dl42_configs.py. Do not edit by hand.\n"))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["grid", "rerun"])
    ap.add_argument("--e1", required=True, help="folder with the E1 configs (E1_<mode>_s<seed>.yaml)")
    ap.add_argument("--dl40", required=True, help="folder with the DL-40 P-PE configs (E1_ppe_s<seed>.yaml)")
    ap.add_argument("--out", required=True, help="grid: the configs folder (DL42_grid/ and DL42_shared_check/ "
                                                 "are made in it); rerun: the output folder")
    ap.add_argument("--selection", help="rerun: selection_dl42.json from select_dl42.py")
    ap.add_argument("--commit", help="code commit to record (default: git HEAD)")
    args = ap.parse_args(argv)
    commit = args.commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True,
                                           text=True, check=True).stdout.strip()
    if args.what == "grid":
        grid, check = make_grid(args.e1, args.dl40, args.out, commit)
        print(f"{len(grid)} grid + {len(check)} shared-check configs written to {args.out} "
              f"(code commit {commit[:7]})", flush=True)
    else:
        if not args.selection:
            ap.error("rerun needs --selection")
        written = make_rerun(args.e1, args.dl40, args.selection, args.out, commit)
        print(f"{len(written)} rerun configs written to {args.out} (code commit {commit[:7]})"
              + ("; no design changed its point" if not written else ""), flush=True)


if __name__ == "__main__":
    main()
