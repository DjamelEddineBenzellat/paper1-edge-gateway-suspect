"""Choose P-PE's cloud step eta_c from the DL-40 fold-9 grid (plan task M3-01).

Rule (fixed before the grid ran): the smallest eta_c whose mean fold-9 macro-F1 (DL-36, final round) over the seeds
is within TOLERANCE of the best mean. A smaller eta_c keeps more of HFLMLD's log compression, its defence against
corrupted edge models. Stop rule: if the best mean is below the trivial classifier + STOP_MARGIN, nothing is
selected and E1 is not run again. Every eta_c must have the same, complete set of seeds.

Example:
    python select_ppe_eta.py --runs <folder with DL40_ppe_*.json> --out selection_dl40.json
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

TOLERANCE = 0.010  # 1.0 macro-F1 point
STOP_MARGIN = 0.05  # 5 points above the trivial classifier
SEALED_SUFFIX = ".fold10.sealed.json"


def load_runs(runs_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "DL40_ppe_*.json"))):
        if path.endswith(SEALED_SUFFIX):
            continue
        with open(path, "rb") as fh:
            raw = fh.read()
        out.append((os.path.basename(path), hashlib.sha256(raw).hexdigest(), json.loads(raw)))
    return out


def select(runs, n_seeds, smoke=False, partial=False):
    by_eta = {}
    for name, _, r in runs:
        cfg = r["experiment_config"]["content"]
        if cfg["experiment"] != "DL-40 grid" or cfg["mode"] != "ppe":
            raise SystemExit(f"{name}: not a DL-40 grid run")
        if r["experiment_config"].get("smoke_rounds") and not smoke:
            raise SystemExit(f"{name}: a shortened (smoke) run cannot select eta_c")
        rounds = r["rounds"]
        by_eta.setdefault(float(cfg["settings"]["cloud_eta"]), {})[cfg["seed"]] = {
            "f1": r["final_fold9"]["macro_f1_tuned"], "auroc": r["final_fold9"]["macro_auroc"],
            "trivial": r["data"]["trivial_macro_f1_fold9"],
            "cloud_step": float(np.mean([x["cloud_step"] for x in rounds])),
            "edge_distance": float(np.mean([x["edge_distance_mean"] for x in rounds]))}
    seed_sets = {tuple(sorted(s)) for s in by_eta.values()}
    if not by_eta or (not partial and (len(seed_sets) != 1 or len(next(iter(seed_sets))) != n_seeds)):
        raise SystemExit(f"incomplete grid: seeds per eta_c = { {e: sorted(s) for e, s in by_eta.items()} }, "
                         f"expected {n_seeds} each")
    per_eta = {}
    for eta in sorted(by_eta):
        rows = [by_eta[eta][s] for s in sorted(by_eta[eta])]
        f1 = [x["f1"] for x in rows]
        per_eta[eta] = {"seeds": sorted(by_eta[eta]), "macro_f1_tuned_mean": float(np.mean(f1)),
                        "macro_f1_tuned_sd": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
                        "macro_f1_tuned_per_seed": f1,
                        "macro_auroc_mean": float(np.mean([x["auroc"] for x in rows])),
                        "cloud_step_mean": float(np.mean([x["cloud_step"] for x in rows])),
                        "edge_distance_mean": float(np.mean([x["edge_distance"] for x in rows]))}
    best_eta = max(per_eta, key=lambda e: per_eta[e]["macro_f1_tuned_mean"])
    best = per_eta[best_eta]["macro_f1_tuned_mean"]
    trivial = float(np.mean([x["trivial"] for s in by_eta.values() for x in s.values()]))
    stop = (not smoke) and best < trivial + STOP_MARGIN
    within = [e for e in per_eta if per_eta[e]["macro_f1_tuned_mean"] >= best - TOLERANCE - 1e-12]
    return {"rule": "DL-40: smallest eta_c with mean fold-9 macro-F1 (DL-36) >= best mean - 1.0 point; "
                    "stop if the best mean < trivial + 5 points",
            "tolerance": TOLERANCE, "stop_margin": STOP_MARGIN, "n_seeds": n_seeds, "smoke": smoke,
            "partial": partial, "per_eta": {f"{e:g}": v for e, v in per_eta.items()},
            "best_cloud_eta": best_eta, "best_mean": best, "trivial_macro_f1_fold9": trivial,
            "within_tolerance": sorted(within), "stop_rule_triggered": stop,
            "selected_cloud_eta": None if (stop or partial) else min(within)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="tests only: shortened runs, no stop rule")
    ap.add_argument("--partial", action="store_true", help="table of an unfinished grid; selects nothing")
    args = ap.parse_args(argv)
    runs = load_runs(args.runs)
    sel = select(runs, args.n_seeds, smoke=args.smoke, partial=args.partial)
    sel["runs_sha256"] = {name: sha for name, sha, _ in runs}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(sel, fh, indent=1)
    for eta, v in sel["per_eta"].items():
        print(f"eta_c {eta:>5}: macro-F1 {v['macro_f1_tuned_mean']:.4f} +/- {v['macro_f1_tuned_sd']:.4f}  "
              f"AUROC {v['macro_auroc_mean']:.3f}  cloud step {v['cloud_step_mean']:.4f}", flush=True)
    print(f"best eta_c {sel['best_cloud_eta']:g} ({sel['best_mean']:.4f}); within 1 point: {sel['within_tolerance']}; "
          f"stop rule: {sel['stop_rule_triggered']}; SELECTED eta_c = {sel['selected_cloud_eta']}", flush=True)


if __name__ == "__main__":
    main()
