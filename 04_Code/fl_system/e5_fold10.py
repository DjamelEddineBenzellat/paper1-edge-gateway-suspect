"""Open the E5 fold-10 files and score H4 (plan task X-10; PR H4, DL-10, DL-14, DL-23, DL-43 (8)-(11), DL-44,
DL-45).

Two steps, each run once, in this order:
  register  hashes every sealed file of the E5 job WITHOUT reading it and checks them against the workstation's
            results MANIFEST. The register is committed before `open`.
  open      only after this script and the register are committed.

Detection and false alarms come from the run files (the per-round flags; not sealed). Fold 10 (sealed) is needed only
for the "harmless miss" (DL-10, DL-45 (2)) and the accuracy effects.
- Per detector (ours = P-TA with the size-aware check, DL-44; HFLMLD native; HFLMLD cloud rule on P-TA) and type:
  detection = flagged tampering gateway-rounds / tampering gateway-rounds, pooled over the 10 seeds, with a Wilson
  95% CI (the adaptive gateway's round 1 is not scored, DL-43 (10), DL-45 (1)).
- False alarms (DL-23, "pooled"; fixed here before any E5 result exists): flagged honest gateway-rounds / honest
  gateway-rounds, pooled over EVERY E5 run of the detector (the 5 types and the runs without tampering); the rate
  without tampering, the rate in the tampering runs only, and the rate per gateway are reported next to it.
- Harmless miss (DL-10, DL-45 (2)): the misses of a run are harmless if, on fold 10, its macro-F1 is at most 2 points
  below the run without tampering of the same detector and seed AND its backdoor rate (DL-38 (4)) differs by at most
  3 points. A harmless miss counts as a miss; its share of the misses is reported (DL-23).
- H4 (on our check, DL-23, DL-45 (4)): holds if each of scale, sign flip, injection and drop reaches detection >= 90%
  AND the pooled false alarms are <= 5%. The adaptive gateway is reported only: its detection and its largest
  undetected accuracy drop and backdoor change (PR H4).
- Also: the DL-10 flags computed on the same outputs of our runs (DL-44 (1)), with the same rules.

Examples:
    python e5_fold10.py register --sealed <E5 job>/sealed_fold10 --zip <results zip> --out e5_seal_register.json
    python e5_fold10.py open --job <E5 job> --register e5_seal_register.json --out e5_fold10.json
"""
import argparse
import json
import os

import numpy as np

from dl42_fold10 import committed, register
from e1_fold10 import SEEDS, paired, summary
from e4_fold10 import load_scores, merged_registers

DETECTORS = ("ours", "hflmld_native", "hflmld_cloud")
TYPES = ("scale", "signflip", "inject", "drop", "adaptive")
NON_ADAPTIVE = ("scale", "signflip", "inject", "drop")
DETECTION_MIN, FALSE_ALARM_MAX = 0.90, 0.05  # DL-23
HARMLESS_F1, HARMLESS_BACKDOOR = 2.0, 3.0  # points, DL-10
N_GATEWAYS = 7
SEALED = ".fold10.sealed.json"


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"rate": p, "k": int(k), "n": int(n), "wilson95": [float(centre - half), float(centre + half)]}


def paths(job, detector, kind, seed):
    run_id = f"E5_{detector}_{kind or 'none'}_s{seed}"
    return job, os.path.join(job, "e5_runs", run_id + ".json"), os.path.join(job, "sealed_fold10", run_id + SEALED)


def load_run(job, detector, kind, seed):
    path = paths(job, detector, kind, seed)[1]
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        r = json.load(fh)
    cfg = r["experiment_config"]["content"]
    t = r.get("tampering")
    if cfg.get("detector") != detector or (cfg["settings"].get("tamper") or None) != kind or r["seed"] != seed:
        raise SystemExit(f"{path}: not the E5 run of {detector} {kind} seed {seed}")
    if (kind is None) != (t is None) or (t and t["kind"] != kind):
        raise SystemExit(f"{path}: its tampering record does not match its config")
    return r


def round_counts(run, flag_key="flagged"):
    """(tampering detected, tampering scored, honest flagged, honest total, per-gateway honest flags and totals)."""
    t = run.get("tampering")
    g_t = t["gateway"] if t else None
    tp = tn_total = fa = scored = 0
    per = {g: [0, 0] for g in range(N_GATEWAYS)}
    for rec in run["rounds"]:
        flagged = set(rec.get(flag_key) or [])
        for g in range(N_GATEWAYS):
            if g == g_t:
                if t["kind"] == "adaptive" and rec["round"] == 1:
                    continue  # DL-43 (10), DL-45 (1)
                scored += 1
                tp += g in flagged
            else:
                tn_total += 1
                fa += g in flagged
                per[g][0] += g in flagged
                per[g][1] += 1
    return tp, scored, fa, tn_total, per


def analyse(job, registers):
    missing, out = [], {}
    for det in DETECTORS:
        refs, miss = load_scores({s: paths(job, det, None, s) for s in SEEDS}, registers)
        missing += [dict(m, configuration=f"{det} none") for m in miss]
        res = {"types": {}}
        pooled = {"fa": 0, "n": 0, "fa_tamper": 0, "n_tamper": 0, "fa_none": 0, "n_none": 0,
                  "per_gateway": {g: [0, 0] for g in range(N_GATEWAYS)}}
        pooled10 = {"fa": 0, "n": 0}
        for kind in TYPES + (None,):
            runs = {s: load_run(job, det, kind, s) for s in SEEDS}
            runs = {s: r for s, r in runs.items() if r is not None}
            missing += [{"configuration": f"{det} {kind or 'none'}", "seed": s, "reason": "run file missing"}
                        for s in SEEDS if s not in runs]
            if kind is not None:
                f10, miss = load_scores({s: paths(job, det, kind, s) for s in runs}, registers)
                missing += [dict(m, configuration=f"{det} {kind}") for m in miss]
            tp = scored = harmless = 0
            per_run = {}
            for s, r in runs.items():
                a, n, fa, ntot, per = round_counts(r)
                pooled["fa"] += fa
                pooled["n"] += ntot
                pooled["fa_tamper" if kind else "fa_none"] += fa
                pooled["n_tamper" if kind else "n_none"] += ntot
                for g, (f, t) in per.items():
                    pooled["per_gateway"][g][0] += f
                    pooled["per_gateway"][g][1] += t
                if det == "ours" and any("flagged_dl10" in rec for rec in r["rounds"]):
                    _, _, fa10, ntot10, _ = round_counts(r, "flagged_dl10")
                    pooled10["fa"] += fa10
                    pooled10["n"] += ntot10
                if kind is None:
                    continue
                tp += a
                scored += n
                ok = s in f10 and s in refs
                drop = 100 * (refs[s]["f1"] - f10[s]["f1"]) if ok else None
                bd = 100 * (f10[s]["backdoor"] - refs[s]["backdoor"]) if ok else None
                is_harmless = ok and drop <= HARMLESS_F1 and abs(bd) <= HARMLESS_BACKDOOR
                harmless += (n - a) if is_harmless else 0
                per_run[s] = {"detected": a, "scored": n, "f1_drop_points": drop, "backdoor_change_points": bd,
                              "misses_harmless": bool(is_harmless)}
            if kind is None:
                continue
            misses = scored - tp
            entry = {"detection": wilson(tp, scored), "misses": misses, "harmless_misses": harmless,
                     "harmless_share_of_misses": None if misses == 0 else harmless / misses,
                     "f1_drop_points": paired(refs, f10, "f1"), "backdoor_change_points": paired(f10, refs, "backdoor"),
                     "per_run": per_run}
            if det == "ours" and kind in NON_ADAPTIVE:
                d10 = [round_counts(r, "flagged_dl10")[:2] for r in runs.values()
                       if any("flagged_dl10" in rec for rec in r["rounds"])]
                entry["detection_dl10"] = wilson(sum(a for a, _ in d10), sum(n for _, n in d10)) if d10 else None
            if kind == "adaptive":  # PR H4: reported only
                undetected = [v for v in per_run.values() if v["detected"] < v["scored"] and v["f1_drop_points"] is not None]
                entry["largest_undetected_f1_drop_points"] = max((v["f1_drop_points"] for v in undetected), default=None)
                entry["largest_undetected_backdoor_change_points"] = max(
                    (abs(v["backdoor_change_points"]) for v in undetected), default=None)
            res["types"][kind] = entry
        res["false_alarms"] = {
            "pooled_all_runs": wilson(pooled["fa"], pooled["n"]),
            "tampering_runs_only": wilson(pooled["fa_tamper"], pooled["n_tamper"]),
            "without_tampering": wilson(pooled["fa_none"], pooled["n_none"]),
            "per_gateway": {g: wilson(f, t) for g, (f, t) in pooled["per_gateway"].items()}}
        if det == "ours":
            res["false_alarms"]["dl10_pooled_all_runs"] = wilson(pooled10["fa"], pooled10["n"])
        res["reference_f0"] = {k: summary([refs[s][k] for s in refs]) for k in ("f1", "backdoor")}
        out[det] = res
    ours = out["ours"]
    det_ok = {k: ours["types"][k]["detection"] is not None and ours["types"][k]["detection"]["rate"] >= DETECTION_MIN
              for k in NON_ADAPTIVE}
    fa = ours["false_alarms"]["pooled_all_runs"]
    fa_ok = fa is not None and fa["rate"] <= FALSE_ALARM_MAX
    return {"note": "E5: detection and false alarms from the run files; fold 10 opened once, after the register was "
                    "committed, for the harmless misses and the accuracy effects",
            "missing": missing, "detectors": out,
            "h4": {"detection_ok_per_type": det_ok, "false_alarms_ok": fa_ok, "holds": bool(all(det_ok.values())
                                                                                         and fa_ok),
                   "rule": "each of scale, sign flip, injection, drop >= 90% detection AND pooled false alarms <= 5% "
                           "(our check; DL-23, DL-44, DL-45 (4)); fallback: report it and compare with HFLMLD"}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["register", "open"])
    ap.add_argument("--sealed", help="register: the E5 job's sealed_fold10 folder (not opened)")
    ap.add_argument("--zip", action="append", default=[], help="register: the E5 results zip")
    ap.add_argument("--job", help="open: results folder of the E5 job")
    ap.add_argument("--register", help="open: the E5 register (committed)")
    ap.add_argument("--no-git-check", action="store_true", help="tests only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.what == "register":
        if not args.sealed or not args.zip:
            ap.error("register needs --sealed and --zip")
        out = {"note": "hashes only; no sealed file was read", "seal_register": register(args.sealed, args.zip)}
        print(f"{len(out['seal_register'])} sealed files registered; all match the workstation manifest", flush=True)
    else:
        if not args.job or not args.register:
            ap.error("open needs --job and --register")
        if not args.no_git_check:
            for path in (os.path.abspath(__file__), args.register):
                if not committed(path):
                    raise SystemExit(f"{path} must be committed before any sealed file is opened")
        out = analyse(args.job, merged_registers([args.register]))
        print(f"E5 fold 10 opened; missing: {out['missing'] or 'none'}; H4 holds: {out['h4']['holds']}", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
