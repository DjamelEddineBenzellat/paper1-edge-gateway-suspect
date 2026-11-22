"""Summarize E1 on fold 9 only (plan task M3-01; DL-36 score, DL-39 sealed fold 10).

Never opens the sealed fold-10 files. For each configuration: the seeds, and the mean and SD of
the fold-9 macro-F1 (DL-36), AUROC, MI sensitivity at 90% specificity, the clean backdoor rate,
the canary AUC and eps_emp per view, WAN bytes and run time.
Also a preview of gate G1 (DL-02 with DL-36/DL-37): part 1, P-TA >= P-TG-c - 10 points (paired by
seed); part 2, P-TA - local-only, reported only. The G1 decision itself is task M3-02.

Example:
    python summarize_e1.py --runs <folder with E1_*.json> --out e1_summary_fold9.json
"""
import argparse
import glob
import json
import os

import numpy as np

SEALED_SUFFIX = ".fold10.sealed.json"


def load_runs(runs_dir, pattern="E1_*.json"):
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, pattern))):
        if path.endswith(SEALED_SUFFIX):  # DL-39: never read here
            continue
        with open(path, encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def fold9_measures(r):
    cfg = r["experiment_config"]["content"]
    if cfg["mode"] == "local":
        epochs = r["experiment_config"].get("smoke_rounds") or cfg["settings"]["local_epochs"]  # as run_experiment.py
        lo = r["local_only"][str(epochs)]
        m = {"macro_f1_tuned": lo["mean_macro_f1_tuned"], "macro_auroc": lo["mean_macro_auroc"],
             "mi_sens_at_90spec": None, "backdoor_success_clean": None}
    else:
        f = r["final_fold9"]
        m = {k: f.get(k) for k in ("macro_f1_tuned", "macro_auroc", "mi_sens_at_90spec", "backdoor_success_clean")}
    for view, v in (r.get("canaries") or {}).get("views", {}).items():
        m[f"canary_auc_{view}"] = v["auc"]
        m[f"canary_eps_lb_{view}"] = v["eps_lb"]
    m["wan_up_per_device_round_mb"] = r["wan_bytes"]["uplink_per_device_round"] / 1e6
    m["wall_min"] = r["timing"]["wall_s"] / 60
    return m


def mean_sd(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def summarize(runs):
    by_cfg = {}
    for r in runs:
        cfg = r["experiment_config"]["content"]
        by_cfg.setdefault(cfg["configuration"], {})[cfg["seed"]] = fold9_measures(r)
    table = {}
    for name, seeds in by_cfg.items():
        keys = sorted({k for m in seeds.values() for k in m})
        table[name] = {"seeds": sorted(seeds),
                       **{k: mean_sd([seeds[s].get(k) for s in sorted(seeds)]) for k in keys},
                       "per_seed_macro_f1_tuned": {str(s): seeds[s]["macro_f1_tuned"] for s in sorted(seeds)}}
    devices = sorted({r["machine"]["device"] for r in runs})
    out = {"note": "fold 9 only; the sealed fold-10 files were not opened (DL-39)", "runs": len(runs),
           "devices": devices, "configurations": table}

    def paired(a, b):
        if a not in by_cfg or b not in by_cfg:
            return None
        common = sorted(set(by_cfg[a]) & set(by_cfg[b]))
        d = [100 * (by_cfg[a][s]["macro_f1_tuned"] - by_cfg[b][s]["macro_f1_tuned"]) for s in common]
        return {"seeds": common, "per_seed_points": d, **(mean_sd(d) or {})} if d else None

    g1 = paired("P-TA", "P-TG-c")
    out["g1_preview"] = {
        "part1_pta_minus_ptgc_points": g1,
        "part1_rule": "P-TA >= P-TG-c - 10 macro-F1 points (fold 9, mean over seeds)",
        "part1_would_pass": None if g1 is None else bool(g1["mean"] >= -10.0),
        "part2_pta_minus_local_points (reported only, DL-36)": paired("P-TA", "local-only"),
        "decision": "task M3-02 (Claude drafts, the student approves)"}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="E1_*.json", help="run files to read (DL-42 reruns: E1tuned_*.json)")
    args = ap.parse_args()
    summary = summarize(load_runs(args.runs, args.pattern))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    for name, t in summary["configurations"].items():
        f = t["macro_f1_tuned"]
        print(f"{name:12} n={f['n']:2} macro-F1 (DL-36) {f['mean']:.3f} +/- {f['sd']:.3f}", flush=True)
    print("G1 preview:", json.dumps({k: v for k, v in summary["g1_preview"].items() if "rule" in k or "would" in k}))


if __name__ == "__main__":
    main()
