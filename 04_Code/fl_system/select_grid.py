"""Pick the default point from the grid runs (DL-20, DL-31, DL-32), with the DL-36 metric.

- P-TA: the (lr, C_ex) pair with the best final fold-9 macro-F1 (DL-36: per-class
  thresholds tuned on fold 9), seed 0. Ties: smaller lr, then smaller C_ex. C and tau
  are the 99th percentiles of the device-update norms and group-sum norms in that
  run (DL-31).
- Local-only: the (lr, epoch) pair with the best mean fold-9 macro-F1 (DL-36) over
  the 210 devices (seed 0). Ties: smaller lr, then fewer epochs (DL-32).
Runs made before DL-36 (no macro_f1_tuned) are refused, so the old threshold-0.5
runs cannot be mixed in. The DL-33 values (threshold 0.5) are kept in the tables.

Example:
    python select_grid.py --runs <folder with pta_grid_*.json and local_grid_*.json> --out selection.json
"""
import argparse
import glob
import json
import os


def load(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fh:
            out.append((os.path.basename(path), json.load(fh)))
    return out


def select(runs_dir):
    pta = load(os.path.join(runs_dir, "pta_grid_*.json"))
    local = load(os.path.join(runs_dir, "local_grid_*.json"))
    if not pta or not local:
        raise SystemExit(f"grid runs missing in {runs_dir}: {len(pta)} P-TA, {len(local)} local-only")
    old = [f for f, r in pta if "macro_f1_tuned" not in r["final_fold9"]] + \
          [f for f, r in local if any("mean_macro_f1_tuned" not in v for v in r["local_only"].values())]
    if old:
        raise SystemExit(f"runs without the DL-36 metric (made before DL-36): {old}")
    pta_table = sorted(({"file": f, "lr": r["settings"]["lr"], "c_ex": r["settings"]["c_ex"],
                         "macro_f1_tuned": r["final_fold9"]["macro_f1_tuned"],
                         "macro_auroc": r["final_fold9"]["macro_auroc"], "macro_f1_0.5": r["final_fold9"]["macro_f1"],
                         "C_p99": r["norms"]["device_update"]["p99"], "tau_p99": r["norms"]["group_sum"]["p99"]}
                        for f, r in pta), key=lambda t: (-t["macro_f1_tuned"], t["lr"], t["c_ex"]))
    best = pta_table[0]
    local_table = []
    for f, r in local:
        for epoch, v in r["local_only"].items():
            local_table.append({"file": f, "lr": r["settings"]["lr"], "epoch": int(epoch),
                                "mean_macro_f1_tuned": v["mean_macro_f1_tuned"],
                                "mean_macro_auroc": v["mean_macro_auroc"], "mean_macro_f1_0.5": v["mean_macro_f1"]})
    local_table.sort(key=lambda t: (-t["mean_macro_f1_tuned"], t["lr"], t["epoch"]))
    best_local = local_table[0]
    return {"rules": "DL-20, DL-31, DL-32 with the DL-36 metric (macro-F1, thresholds tuned on fold 9)",
            "pta_default_point": {"lr": best["lr"], "c_ex": best["c_ex"], "clip_C": best["C_p99"],
                                  "tau": best["tau_p99"], "grid_macro_f1_tuned": best["macro_f1_tuned"],
                                  "grid_macro_auroc": best["macro_auroc"], "from": best["file"]},
            "local_only": {"lr": best_local["lr"], "epoch": best_local["epoch"],
                           "grid_mean_macro_f1_tuned": best_local["mean_macro_f1_tuned"],
                           "grid_mean_macro_auroc": best_local["mean_macro_auroc"], "from": best_local["file"]},
            "pta_grid": pta_table, "local_grid": local_table}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sel = select(args.runs)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(sel, fh, indent=1)
    print(json.dumps({k: sel[k] for k in ("pta_default_point", "local_only")}, indent=1))


if __name__ == "__main__":
    main()
