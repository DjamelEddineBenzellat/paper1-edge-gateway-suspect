"""Pick the learning rate of the non-private baselines (DL-37 point 3) with the DL-36 score.

- Centralized: the (lr, epoch) pair with the best fold-9 macro-F1 (DL-36), epochs {10, 25, 50,
  100}, seed 0. Ties: smaller lr, then fewer epochs.
- FedAvg, HierFAVG: the lr with the best final fold-9 macro-F1 (DL-36) after T = 100 device
  rounds, seed 0. Ties: smaller lr.
Runs made without the DL-36 score (macro_f1_tuned) are refused.

Example:
    python select_baselines.py --runs <folder with *_grid_lr*_s0.json> --out selection_baselines.json
"""
import argparse
import glob
import json
import os


def load(runs_dir, mode):
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, f"{mode}_grid_*.json"))):
        with open(path, encoding="utf-8") as fh:
            out.append((os.path.basename(path), json.load(fh)))
    if not out:
        raise SystemExit(f"no {mode} grid runs in {runs_dir}")
    return out


def select(runs_dir):
    sel = {"rules": "DL-37 point 3, DL-36 score (macro-F1, thresholds tuned on fold 9), seed 0"}
    table = []
    for f, r in load(runs_dir, "centralized"):
        for epoch, v in r["centralized"].items():
            if "macro_f1_tuned" not in v:
                raise SystemExit(f"{f} has no DL-36 score")
            table.append({"file": f, "lr": r["settings"]["lr"], "epoch": int(epoch),
                          "macro_f1_tuned": v["macro_f1_tuned"], "macro_auroc": v["macro_auroc"],
                          "macro_f1_0.5": v["macro_f1"]})
    table.sort(key=lambda t: (-t["macro_f1_tuned"], t["lr"], t["epoch"]))
    sel["centralized"] = {k: table[0][k] for k in ("lr", "epoch", "macro_f1_tuned", "macro_auroc", "file")}
    sel["centralized_grid"] = table
    for mode in ("fedavg", "hierfavg"):
        rows = []
        for f, r in load(runs_dir, mode):
            fin = r["final_fold9"]
            if "macro_f1_tuned" not in fin:
                raise SystemExit(f"{f} has no DL-36 score")
            rows.append({"file": f, "lr": r["settings"]["lr"], "macro_f1_tuned": fin["macro_f1_tuned"],
                         "macro_auroc": fin["macro_auroc"], "macro_f1_0.5": fin["macro_f1"],
                         "rounds": r["settings"]["rounds"], "edge_iterations": r["settings"]["edge_iterations"]})
        rows.sort(key=lambda t: (-t["macro_f1_tuned"], t["lr"]))
        sel[mode] = {k: rows[0][k] for k in ("lr", "macro_f1_tuned", "macro_auroc", "file")}
        sel[f"{mode}_grid"] = rows
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sel = select(args.runs)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(sel, fh, indent=1)
    print(json.dumps({k: sel[k] for k in ("centralized", "fedavg", "hierfavg")}, indent=1))


if __name__ == "__main__":
    main()
