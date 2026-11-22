"""G1 on fold 9 (plan task M3-02; DL-02, DL-27, DL-36), recomputed from the raw run files.

Never opens a sealed fold-10 file: it only records their SHA-256 (the seal register) and, if the results zips
are given, checks them against the workstation's MANIFEST.sha256.
Reports: the input checks; G1 part 1 (the gate) and part 2 (reported only); sensitivity of part 1 to the
metric (F1 at 0.5, best round, AUROC); fold-9 previews of H5(a) (TOST, DL-07) and of the H3 gap at eps = 8.

Example:
    python g1_check.py --e1 <E1 runs> --dl40 <DL-40 E1 P-PE runs> --sealed <dir> [--sealed <dir>] \
        [--zip <results zip>] --out g1_check.json
"""
import argparse
import glob
import hashlib
import json
import os
import zipfile

import numpy as np
from scipy import stats

SEALED_SUFFIX = ".fold10.sealed.json"
SEEDS = list(range(10))
G1_MARGIN = -10.0  # points (DL-35 table item 35)
TOST_MARGIN = 1.0  # points (H5(a))


def load(e1_dir, dl40_dir):
    """{configuration: {seed: run}}; P-PE from DL-40 replaces the E1 P-PE runs (DL-40 (5))."""
    runs = {}
    for folder, keep in ((e1_dir, lambda m: m != "ppe"), (dl40_dir, lambda m: m == "ppe")):
        for path in sorted(glob.glob(os.path.join(folder, "E1_*.json"))):
            if path.endswith(SEALED_SUFFIX):
                continue
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
            cfg = r["experiment_config"]["content"]
            if keep(cfg["mode"]):
                runs.setdefault(cfg["configuration"], {})[cfg["seed"]] = r
    return runs


def measures(r):
    cfg = r["experiment_config"]["content"]
    if cfg["mode"] == "local":
        lo = r["local_only"][str(cfg["settings"]["local_epochs"])]
        return {"f1": lo["mean_macro_f1_tuned"], "f1_at_05": lo["mean_macro_f1"], "auroc": lo["mean_macro_auroc"],
                "f1_best_round": lo["mean_macro_f1_tuned"]}
    f = r["final_fold9"]
    best = max([c["macro_f1_tuned"] for c in r.get("curve", [])] + [f["macro_f1_tuned"]])
    return {"f1": f["macro_f1_tuned"], "f1_at_05": f["macro_f1"], "auroc": f["macro_auroc"], "f1_best_round": best}


def paired(runs, a, b, key="f1"):
    d = np.array([100 * (measures(runs[a][s])[key] - measures(runs[b][s])[key]) for s in SEEDS])
    se = d.std(ddof=1) / np.sqrt(len(d))
    t95, t90 = stats.t.ppf(0.975, len(d) - 1), stats.t.ppf(0.95, len(d) - 1)
    return {"per_seed_points": [round(float(x), 3) for x in d], "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "ci95": [float(d.mean() - t95 * se), float(d.mean() + t95 * se)],
            "ci90": [float(d.mean() - t90 * se), float(d.mean() + t90 * se)], "min_seed": float(d.min())}


def checks(runs):
    out = {"configurations": sorted(runs), "seeds_complete": all(sorted(v) == SEEDS for v in runs.values()),
           "devices": sorted({r["machine"]["device"] for v in runs.values() for r in v.values()}),
           "commits": sorted({r["git_commit"][:7] for v in runs.values() for r in v.values()}),
           "smoke_runs": sum(1 for v in runs.values() for r in v.values() if r["experiment_config"].get("smoke_rounds")),
           "dp_epsilon_max": max((r["privacy"]["epsilon_accountant"] or 0) for v in runs.values() for r in v.values()),
           "ppe_cloud_eta": sorted({r["settings"].get("cloud_eta") for r in runs.get("P-PE", {}).values()})}
    out["ok"] = (out["seeds_complete"] and out["devices"] == ["cuda"] and out["smoke_runs"] == 0
                 and out["dp_epsilon_max"] <= 8.0 and len(runs) == 9)
    return out


def seal_register(sealed_dirs, zips):
    reg = {}
    for d in sealed_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*" + SEALED_SUFFIX))):
            folder = os.path.basename(os.path.dirname(os.path.dirname(path)))  # the job: the same name can recur
            with open(path, "rb") as fh:
                reg[f"{folder}/{os.path.basename(path)}"] = {"sha256": hashlib.sha256(fh.read()).hexdigest(),
                                                             "folder": folder, "file": os.path.basename(path)}
    manifest = {}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for line in zf.read("MANIFEST.sha256").decode("utf-8").splitlines():
                sha, rel = line.split(None, 1)
                if rel.endswith(SEALED_SUFFIX):
                    manifest[(os.path.basename(z), os.path.basename(rel))] = sha
    for v in reg.values():
        hits = [sha for (zname, base), sha in manifest.items() if base == v["file"] and v["folder"] in zname]
        v["matches_workstation_manifest"] = (hits == [v["sha256"]]) if zips else None
    return reg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--e1", required=True, help="the E1 run JSONs (JOB_20261117_M3-01_e1/runs)")
    ap.add_argument("--dl40", required=True, help="the DL-40 E1 P-PE run JSONs")
    ap.add_argument("--sealed", action="append", default=[], help="a sealed_fold10 folder (not opened)")
    ap.add_argument("--zip", action="append", default=[], help="a results zip, to check the seals against")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    runs = load(args.e1, args.dl40)
    part1 = paired(runs, "P-TA", "P-TG-c")
    se = part1["sd"] / np.sqrt(len(SEEDS))
    out = {"note": "fold 9 only; sealed fold-10 files were hashed, not opened (DL-39)",
           "checks": checks(runs),
           "g1_part1": dict(part1, rule="mean(P-TA - P-TG-c) >= -10 points (DL-27, DL-36)",
                            passes=bool(part1["mean"] >= G1_MARGIN),
                            p_true_gap_below_minus10=float(stats.t.cdf((G1_MARGIN - part1["mean"]) / se,
                                                                        len(SEEDS) - 1))),
           "g1_part1_sensitivity": {k: paired(runs, "P-TA", "P-TG-c", k) for k in ("f1_at_05", "f1_best_round",
                                                                                   "auroc")},
           "g1_part2_reported": paired(runs, "P-TA", "local-only"),
           "preview_h5a_tost_pta_vs_pfr": dict(paired(runs, "P-TA", "P-FR"), margin=TOST_MARGIN),
           "preview_h3_gap_eps8_ptgc_minus_pta": paired(runs, "P-TG-c", "P-TA"),
           "means": {c: {k: float(np.mean([measures(runs[c][s])[k] for s in SEEDS]))
                         for k in ("f1", "f1_at_05", "auroc", "f1_best_round")} for c in sorted(runs)},
           "seal_register": seal_register(args.sealed, args.zip)}
    t = out["preview_h5a_tost_pta_vs_pfr"]
    out["preview_h5a_tost_pta_vs_pfr"]["dl07_outcome_on_fold9"] = (
        "equivalent" if -TOST_MARGIN < t["ci90"][0] and t["ci90"][1] < TOST_MARGIN
        else "P-TA worse" if t["ci95"][1] < 0 else "inconclusive")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    p = out["g1_part1"]
    print(f"checks ok: {out['checks']['ok']}  G1 part 1: {p['mean']:+.2f} points, 95% CI [{p['ci95'][0]:+.2f}, "
          f"{p['ci95'][1]:+.2f}] -> {'PASS' if p['passes'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
