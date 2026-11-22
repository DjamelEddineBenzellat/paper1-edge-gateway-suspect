"""Appendix C of Paper 1: macro-F1 at the fixed threshold 0.5 (required by DL-36 (1)) and per-class F1, test fold.

Reads the final E1 test-fold files (the 90 of e1_fold10.py, P-PE from the DL-40 job) and the 10 tuned P-TG-c files
of DL-42. All were opened before (M3-03, X-5); each is re-read only if its SHA-256 matches its committed register
(G1 register, DL-42 register). As a check, the tuned macro-F1 means must reproduce Table T4 (e1_fold10.json,
dl42_fold10.json) within 0.0005, otherwise the script stops. Local-only files store only each device's tuned
macro-F1, so its threshold-0.5 and per-class values are reported as not recorded.

Example:
    python appendix_threshold05.py --results 05_Experiments/results --out 05_Experiments/results/X-11/appendix_threshold05.json
"""
import argparse
import hashlib
import json
import os
import statistics

from e1_fold10 import sources

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ROWS = [("P-TG", "ptg"), ("P-TG-c, shared point", "ptgc"), ("P-TG-c, tuned", "ptgc-tuned"), ("P-PE", "ppe"),
        ("P-FR", "pfr"), ("P-TA", "pta"), ("FedAvg (no DP)", "fedavg"), ("HierFAVG (no DP)", "hierfavg"),
        ("Centralized (no DP)", "centralized"), ("Local-only (no DP)", "local")]
T4_KEYS = {"ptg": "P-TG", "ptgc": "P-TG-c", "ppe": "P-PE", "pfr": "P-FR", "pta": "P-TA", "fedavg": "FedAvg",
           "hierfavg": "HierFAVG", "centralized": "centralized", "local": "local-only"}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checked(path, register, key):
    want = register[key]["sha256"] if isinstance(register[key], dict) else register[key]
    if sha256(path) != want:
        raise SystemExit(f"{path}: SHA-256 differs from its register")
    return load(path)


def summary(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values), "n": len(values)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="05_Experiments/results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    r = args.results
    e1_job = os.path.join(r, "M3-01", "JOB_20261117_M3-01_e1")
    dl40_job = os.path.join(r, "M3-01", "JOB_20261117_M3-01_dl40-ppe")
    dl42_job = os.path.join(r, "DL-42", "JOB_20261118_DL-42_dl42-tuning-v2")
    g1 = load(os.path.join(r, "G1", "g1_check.json"))["seal_register"]
    dl42 = load(os.path.join(r, "DL-42", "dl42_seal_register.json"))["seal_register"]
    e1_t4 = load(os.path.join(r, "E1", "e1_fold10.json"))["designs"]
    dl42_t4 = load(os.path.join(r, "DL-42", "dl42_fold10.json"))
    out = {"note": "test fold (fold 10); files re-read after their SHA-256 matched the committed registers",
           "rows": {}}
    for label, mode in ROWS:
        tuned, at05 = [], []
        pc_tuned = {c: [] for c in CLASSES}
        pc05 = {c: [] for c in CLASSES}
        for seed in range(10):
            if mode == "ptgc-tuned":
                name = f"E1tuned_ptgc_s{seed}.fold10.sealed.json"
                path = os.path.join(dl42_job, "sealed_fold10", name)
                sealed = checked(path, dl42, f"JOB_20261118_DL-42_dl42-tuning-v2/{name}")
            else:
                job, _, path = sources(e1_job, dl40_job, mode, seed)
                sealed = checked(path, g1, f"{os.path.basename(job)}/{os.path.basename(path)}")
            if mode == "local":
                tuned.append(statistics.mean(d["macro_f1_tuned"] for d in sealed["devices"]))
                continue
            f = sealed["final_fold10"]
            if f["thresholds_tuned_on_this_fold"]:
                raise SystemExit(f"{path}: thresholds tuned on the test fold")
            tuned.append(f["macro_f1_tuned"])
            at05.append(f["macro_f1"])
            for c in CLASSES:
                pc_tuned[c].append(f["per_class_f1_tuned"][c])
                pc05[c].append(f["per_class_f1"][c])
        row = {"macro_f1_tuned": summary(tuned)}
        ref = dl42_t4 if mode == "ptgc-tuned" else e1_t4[T4_KEYS[mode]]["f1"]
        ref_mean = find_mean(ref) if mode == "ptgc-tuned" else ref["mean"]
        if abs(row["macro_f1_tuned"]["mean"] - ref_mean) > 5e-4:
            raise SystemExit(f"{label}: tuned macro-F1 {row['macro_f1_tuned']['mean']:.4f} does not reproduce "
                             f"Table T4 ({ref_mean:.4f})")
        if at05:
            row["macro_f1_at_0_5"] = summary(at05)
            row["per_class_f1_tuned"] = {c: statistics.mean(v) for c, v in pc_tuned.items()}
            row["per_class_f1_at_0_5"] = {c: statistics.mean(v) for c, v in pc05.items()}
        out["rows"][label] = row
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for label, row in out["rows"].items():
        t, z = row["macro_f1_tuned"], row.get("macro_f1_at_0_5")
        pc = row.get("per_class_f1_tuned")
        print(f"{label:22s} tuned {t['mean']:.3f} ± {t['sd']:.3f} | 0.5 "
              + (f"{z['mean']:.3f} ± {z['sd']:.3f}" if z else "not recorded")
              + (" | " + " ".join(f"{c} {pc[c]:.3f}" for c in CLASSES) if pc else ""), flush=True)


def find_mean(dl42):
    """Tuned P-TG-c macro-F1 mean (10 seeds) in dl42_fold10.json."""
    for key in ("final_configurations", "designs", "table"):
        block = dl42.get(key)
        if isinstance(block, dict):
            for k, v in block.items():
                if "P-TG-c" in k and isinstance(v, dict):
                    f1 = v.get("f1", v)
                    if isinstance(f1, dict) and "mean" in f1:
                        return f1["mean"]
    raise SystemExit("tuned P-TG-c macro-F1 not found in dl42_fold10.json; check its keys")


if __name__ == "__main__":
    main()
