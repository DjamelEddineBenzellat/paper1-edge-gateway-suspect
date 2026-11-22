"""Summarize the DL-35 diagnostic job. It makes no gate decision (DL-21, DL-35).

Groups (fold 9, mean over seeds):
- nonoise_pta, nonoise_ptgc: P-TA and P-TG-c without DP noise (z = 0);
- pta_pilot, ptgc_pilot: the eps = 8 pilot again;
- local_pilot: local-only (no DP), at the epoch chosen by select_grid.py.
For each group: macro-F1 at threshold 0.5 (DL-33), macro-AUROC, per-class AUROC,
and macro-F1 with per-class thresholds tuned on the training data and on fold 9
(optimistic). Also P-TA - local-only under each measure (at the gate epoch and at
local-only's best scored epoch), local-only at every scored epoch, and the change
of the eps-8 macro-F1 against the earlier job (same machine: reproducibility).

Example:
    python summarize_diag.py --runs <runs folder> --local-epoch 100 --earlier g0a_summary_eps8.json --out diag.json
"""
import argparse
import glob
import json
import os

import numpy as np

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
GROUPS = ("nonoise_pta", "nonoise_ptgc", "pta_pilot", "ptgc_pilot", "local_pilot")
EARLIER = {"pta_pilot": "P-TA", "ptgc_pilot": "P-TG-c", "local_pilot": "local-only"}
MEASURES = ("macro_f1_threshold_0.5", "macro_auroc", "macro_f1_train_thresholds", "macro_f1_fold9_thresholds")


def run_measures(r, epoch):
    """The four measures and the per-class AUROC of one run."""
    if r["mode"] == "local":
        lo = r["local_only"][epoch]
        d = lo["diag"]
        return ({"macro_f1_threshold_0.5": lo["mean_macro_f1"], "macro_auroc": lo["mean_macro_auroc"],
                 "macro_f1_train_thresholds": d["mean_macro_f1_train_thresholds"],
                 "macro_f1_fold9_thresholds": d["mean_macro_f1_fold9_thresholds"]}, d["mean_per_class_auroc"])
    f, d = r["final_fold9"], r["diagnostics"]
    return ({"macro_f1_threshold_0.5": f["macro_f1"], "macro_auroc": f["macro_auroc"],
             "macro_f1_train_thresholds": d["macro_f1_train_thresholds"],
             "macro_f1_fold9_thresholds": d["macro_f1_fold9_thresholds"]}, d["per_class_auroc"])


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--local-epoch", required=True, help="local-only epoch from selection.json (DL-32)")
    ap.add_argument("--earlier", help="g0a_summary_eps8.json of the earlier job (reproducibility check)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary = {"note": "DL-35 diagnostics only; no gate decision. DL-33 (threshold 0.5) still holds for every gate.",
               "groups": {}}
    loaded = {}
    for g in GROUPS:
        runs = []
        for path in sorted(glob.glob(os.path.join(args.runs, f"{g}_s*.json"))):
            with open(path, encoding="utf-8") as fh:
                runs.append(json.load(fh))
        if not runs:
            continue
        loaded[g] = runs
        per_seed, per_class = {}, []
        for r in runs:
            m, pc = run_measures(r, str(args.local_epoch))
            per_seed[r["seed"]] = m
            per_class.append(pc)
        summary["groups"][g] = {
            "seeds": sorted(per_seed),
            "noise": runs[0]["privacy"].get("noise", "on") if g != "local_pilot" else "no DP (local-only)",
            "z_used": runs[0]["privacy"].get("z_used") if g != "local_pilot" else None,
            "mean": {k: mean([per_seed[s][k] for s in per_seed]) for k in MEASURES},
            "per_class_auroc_mean": {c: mean([pc[c] for pc in per_class]) for c in CLASSES},
            "per_seed": per_seed,
        }
    groups = summary["groups"]
    for name in ("nonoise_pta", "pta_pilot"):
        if name in groups and "local_pilot" in groups:
            summary[f"{name}_minus_local_points"] = {
                k: 100 * (groups[name]["mean"][k] - groups["local_pilot"]["mean"][k]) for k in MEASURES}
    if "local_pilot" in loaded:
        # local-only at every scored epoch (DL-32 checkpoints)
        by_epoch = {}
        for r in loaded["local_pilot"]:
            for e in r["local_only"]:
                by_epoch.setdefault(int(e), []).append(run_measures(r, e)[0])
        table = {e: {k: mean([m[k] for m in ms]) for k in MEASURES} for e, ms in sorted(by_epoch.items())}
        summary["local_by_epoch"] = {str(e): v for e, v in table.items()}
        best = {k: max(table, key=lambda e, k=k: table[e][k]) for k in MEASURES}
        summary["local_best_epoch"] = {k: {"epoch": e, "value": table[e][k]} for k, e in best.items()}
        for name in ("nonoise_pta", "pta_pilot"):
            if name in groups:
                summary[f"{name}_minus_local_best_epoch_points"] = {
                    k: 100 * (groups[name]["mean"][k] - table[best[k]][k]) for k in MEASURES}
    if args.earlier and os.path.exists(args.earlier):
        with open(args.earlier, encoding="utf-8") as fh:
            old = json.load(fh)["per_seed_macro_f1"]
        summary["reproducibility_vs_earlier_job_points"] = {
            g: {str(s): 100 * (m["macro_f1_threshold_0.5"] - old[EARLIER[g]][str(s)])
                for s, m in groups[g]["per_seed"].items() if str(s) in old.get(EARLIER[g], {})}
            for g in EARLIER if g in groups}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    print(f"{'group':14} {'F1@0.5':>7} {'AUROC':>7} {'F1 train-thr':>13} {'F1 fold9-thr':>13}")
    for g, v in groups.items():
        m = v["mean"]
        print(f"{g:14} " + " ".join(f"{(m[k] if m[k] is not None else float('nan')):{w}.3f}"
                                     for k, w in zip(MEASURES, (7, 7, 13, 13))))
    for e, v in summary.get("local_by_epoch", {}).items():
        print(f"{'local ep ' + e:14} " + " ".join(f"{v[k]:{w}.3f}" for k, w in zip(MEASURES, (7, 7, 13, 13))))
    for key in ("nonoise_pta_minus_local_points", "pta_pilot_minus_local_points",
                "nonoise_pta_minus_local_best_epoch_points", "pta_pilot_minus_local_best_epoch_points",
                "reproducibility_vs_earlier_job_points"):
        if key in summary:
            print(key, json.dumps(summary[key]))


if __name__ == "__main__":
    main()
