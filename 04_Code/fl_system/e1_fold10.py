"""Open the final E1 fold-10 files and analyse E1 on fold 10 (plan task M3-03; DL-02, DL-39; G1 rules O1-O9).

O1: only the 90 final files are read (the 80 E1 files of the non-P-PE designs and the 10 DL-40 P-PE files); the
E1 P-PE files superseded by DL-40 are never opened. O2: before a file is read, its SHA-256 must match the seal
register (g1_check.json); otherwise it is not opened and its seed counts as missing. Each file must also belong to
its run and carry that run's fold-9 thresholds (DL-36).
Output: fold-10 scores per design; the fold-9 -> fold-10 change (O5); G1 on fold 10, reported only (O4);
H3(a) at eps0 = 8 (DL-15); H5(a) F1 part with the DL-07 outcomes (O6); all pairs of DP designs (O8);
the DL-05 data (gateway-view canary AUC, from the fold-9 run files).

Example:
    python e1_fold10.py --e1 <E1 job folder> --dl40 <DL-40 job folder> --register g1_check.json --out e1_fold10.json
"""
import argparse
import hashlib
import itertools
import json
import os

import numpy as np
from scipy import stats

SEEDS = list(range(10))
CONFIGURATIONS = [("P-TG", "ptg"), ("P-TG-c", "ptgc"), ("P-PE", "ppe"), ("P-FR", "pfr"), ("P-TA", "pta"),
                  ("centralized", "centralized"), ("local-only", "local"), ("FedAvg", "fedavg"),
                  ("HierFAVG", "hierfavg")]
DP_DESIGNS = ["P-TG", "P-TG-c", "P-PE", "P-FR", "P-TA"]
TOST_MARGIN = 1.0  # points, H5(a)
DL05_BAND = 0.09  # gateway-view canary AUC, DL-05


def sources(e1_job, dl40_job, mode, seed):
    """(job folder, run JSON, sealed file) of the final run (O1: P-PE from DL-40)."""
    if mode == "ppe":
        return dl40_job, os.path.join(dl40_job, "e1_ppe_runs", f"E1_ppe_s{seed}.json"), \
            os.path.join(dl40_job, "sealed_fold10", f"E1_ppe_s{seed}.fold10.sealed.json")
    return e1_job, os.path.join(e1_job, "runs", f"E1_{mode}_s{seed}.json"), \
        os.path.join(e1_job, "sealed_fold10", f"E1_{mode}_s{seed}.fold10.sealed.json")


def open_sealed(path, job, register):
    """The sealed file's content, or (None, reason) if O2 forbids opening it."""
    key = f"{os.path.basename(os.path.normpath(job))}/{os.path.basename(path)}"
    if key not in register:
        return None, "not in the seal register"
    if not os.path.exists(path):
        return None, "file missing"
    with open(path, "rb") as fh:
        raw = fh.read()
    if hashlib.sha256(raw).hexdigest() != register[key]["sha256"]:
        return None, "SHA-256 differs from the seal register"
    return json.loads(raw), None


def fold10_measures(sealed, run):
    """Fold-10 measures, after checking that the file belongs to the run and uses its fold-9 thresholds."""
    if (sealed["run_id"], sealed["seed"], sealed["mode"]) != (run["run_id"], run["seed"], run["mode"]):
        raise SystemExit(f"{sealed['run_id']}: the sealed file does not belong to its run")
    if "final_fold10" in sealed:
        f = sealed["final_fold10"]
        if f["thresholds_tuned_on_this_fold"] or f["thresholds"] != run["final_fold9"]["thresholds"]:
            raise SystemExit(f"{sealed['run_id']}: fold 10 was not scored with the fold-9 thresholds")
        return {"f1": f["macro_f1_tuned"], "auroc": f["macro_auroc"], "mi_sens": f["mi_sens_at_90spec"],
                "backdoor": f["backdoor_success_clean"], "per_class_f1": f["per_class_f1_tuned"]}
    lo = sealed["local_only_fold10"]
    mi = [d["mi_sens_at_90spec"] for d in sealed["devices"] if d.get("mi_sens_at_90spec") is not None]
    return {"f1": lo["mean_macro_f1_tuned"], "auroc": lo["mean_macro_auroc"],
            "mi_sens": float(np.mean(mi)) if mi else None, "backdoor": None, "per_class_f1": None}


def fold9_f1(run):
    if run["mode"] == "local":
        cfg = run["experiment_config"]["content"]
        return run["local_only"][str(cfg["settings"]["local_epochs"])]["mean_macro_f1_tuned"]
    return run["final_fold9"]["macro_f1_tuned"]


def summary(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def paired(a, b, key="f1"):
    """Paired difference a - b in points over the seeds both have."""
    common = sorted(set(a) & set(b))
    d = np.array([100 * (a[s][key] - b[s][key]) for s in common])
    if len(d) < 2:  # O2 can leave too few seeds for a CI
        return {"seeds": common, "per_seed_points": [round(float(x), 3) for x in d],
                "mean": float(d.mean()) if len(d) else None, "sd": None, "ci95": [None, None], "ci90": [None, None]}
    se = d.std(ddof=1) / np.sqrt(len(d))
    t95, t90 = stats.t.ppf(0.975, len(d) - 1), stats.t.ppf(0.95, len(d) - 1)
    return {"seeds": common, "per_seed_points": [round(float(x), 3) for x in d], "mean": float(d.mean()),
            "sd": float(d.std(ddof=1)), "ci95": [float(d.mean() - t95 * se), float(d.mean() + t95 * se)],
            "ci90": [float(d.mean() - t90 * se), float(d.mean() + t90 * se)]}


def dl07(p):
    if p["sd"] is None:
        return "not evaluable (too few seeds)"
    if -TOST_MARGIN < p["ci90"][0] and p["ci90"][1] < TOST_MARGIN:
        return "equivalent"
    return "P-TA worse" if p["ci95"][1] < 0 else "inconclusive"


def analyse(e1_job, dl40_job, register):
    f10, f9, canary, missing = {}, {}, {}, []
    for name, mode in CONFIGURATIONS:
        for seed in SEEDS:
            job, run_path, sealed_path = sources(e1_job, dl40_job, mode, seed)
            with open(run_path, encoding="utf-8") as fh:
                run = json.load(fh)
            f9.setdefault(name, {})[seed] = {"f1": fold9_f1(run)}
            views = (run.get("canaries") or {}).get("views", {})
            if "gateway" in views:
                canary.setdefault(name, {})[seed] = {"auc": views["gateway"]["auc"]}
            sealed, reason = open_sealed(sealed_path, job, register)
            if sealed is None:
                missing.append({"configuration": name, "seed": seed, "reason": reason})
                continue
            f10.setdefault(name, {})[seed] = fold10_measures(sealed, run)
    table = {}
    for name, _ in CONFIGURATIONS:
        rows = f10.get(name, {})
        table[name] = {k: summary([rows[s][k] for s in rows]) for k in ("f1", "auroc", "mi_sens", "backdoor")}
        table[name]["fold9_f1"] = summary([f9[name][s]["f1"] for s in f9[name]])
        table[name]["fold10_minus_fold9_points"] = paired(rows, f9[name]) if rows else None
        pcs = [rows[s]["per_class_f1"] for s in rows if rows[s]["per_class_f1"]]
        table[name]["per_class_f1"] = {c: float(np.mean([p[c] for p in pcs])) for c in pcs[0]} if pcs else None
    h3 = paired(f10["P-TG-c"], f10["P-TA"])
    h5 = paired(f10["P-TA"], f10["P-FR"])
    dl05 = {f"P-TA - {o}": paired(canary["P-TA"], canary[o], "auc") for o in ("P-TG", "P-PE")}
    for v in dl05.values():  # AUC units here, not points
        v["mean_auc"] = None if v["mean"] is None else v["mean"] / 100
        v["within_band"] = None if v["mean"] is None else bool(abs(v["mean_auc"]) <= DL05_BAND)
    return {"note": "fold 10 opened after G1 (approved 18 Nov 2026); 90 final files only (O1), seals checked (O2)",
            "missing": missing, "opened": sum(len(v) for v in f10.values()), "designs": table,
            "g1_on_fold10_reported_only": dict(paired(f10["P-TA"], f10["P-TG-c"]), rule_fold9_only="DL-02, O4"),
            "pta_minus_local_only": paired(f10["P-TA"], f10["local-only"]),
            "h3a_eps0_8": dict(h3, passes=None if h3["sd"] is None else bool(h3["mean"] > 0 and h3["ci95"][0] > 0),
                               rule="gap P-TG-c - P-TA > 0 with a paired 95% CI excluding 0 (DL-15: at eps0)"),
            "h5a_f1": dict(h5, outcome=dl07(h5), margin=TOST_MARGIN),
            "dp_pairs": {f"{a} - {b}": paired(f10[a], f10[b]) for a, b in itertools.combinations(DP_DESIGNS, 2)},
            "dl05_gateway_canary_auc": dict(dl05, warning="the canary audit has no power (E1 finding F3, risk R6)")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--e1", required=True, help="results folder of JOB_20261117_M3-01_e1")
    ap.add_argument("--dl40", required=True, help="results folder of JOB_20261117_M3-01_dl40-ppe")
    ap.add_argument("--register", required=True, help="g1_check.json with the seal register")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with open(args.register, encoding="utf-8") as fh:
        register = json.load(fh)["seal_register"]
    out = analyse(args.e1, args.dl40, register)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"opened {out['opened']} sealed files; missing: {out['missing'] or 'none'}", flush=True)


if __name__ == "__main__":
    main()
