"""Choose each DP design's point from the DL-42 fold-9 grid (plan task X-1).

Rule (DL-42 (2), fixed before the grid runs), per design:
- the mean fold-9 macro-F1 (DL-36, final round) over seeds 0-2 at each of the 9 grid points;
- the shared point (lr 0.01, C_ex 2.0) is KEPT unless the best mean is MORE than 1.0 point higher; then the best
  point is selected (ties: the shared point, then the smaller lr, then the smaller C_ex);
- P-TA and P-FR at a new point: C (and tau for P-TA) = the 99th percentiles of the device-update (and group-sum)
  norms at that point, pooled over the seeds (DL-31);
- a missing seed at any point is an error, not a skip.
P-TG, P-TG-c and P-PE take the shared point from the E1 runs (P-PE: DL-40) of seeds 0-2, after a check that this
code reproduces seed 0 (`check`). If the check fails, their shared point is run again for every seed and those runs
are used instead.

Examples:
    python select_dl42.py check --runs <DL42_*.json folder> --reuse <reused E1_*.json folder> --out reuse_check.json
    python select_dl42.py select --runs <DL42_*.json folder> --reuse <reused E1_*.json folder> --out selection_dl42.json
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

from make_dl42_configs import C_EXS, DESIGNS, LRS, NORM_MODES, REUSE_MODES, SHARED, point_tag

TOLERANCE = 0.010  # 1.0 macro-F1 point
REPRO_TOL = 1e-6  # the reuse check: same code, same seed, same device -> the same fold-9 scores
PERCENTILE = 99  # DL-31
SEALED_SUFFIX = ".fold10.sealed.json"


def load(folder, pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(folder, pattern))):
        if path.endswith(SEALED_SUFFIX):  # DL-39: never read here
            continue
        with open(path, "rb") as fh:
            raw = fh.read()
        out.append((os.path.basename(path), hashlib.sha256(raw).hexdigest(), json.loads(raw)))
    return out


def record(name, r):
    f = r["final_fold9"]
    return {"file": name, "f1": f["macro_f1_tuned"], "auroc": f["macro_auroc"], "device": r["machine"]["device"],
            "norms_raw": r.get("norms_raw")}


def index_grid(runs, smoke=False):
    """grid[(mode, (lr, c_ex))][seed] for the DL-42 grid and shared-check runs."""
    grid = {}
    for name, _, r in runs:
        cfg, s = r["experiment_config"]["content"], r["settings"]
        if cfg["experiment"] != "DL-42 grid" or cfg["mode"] not in DESIGNS:
            raise SystemExit(f"{name}: not a DL-42 grid run")
        if r["experiment_config"].get("smoke_rounds") and not smoke:
            raise SystemExit(f"{name}: a shortened (smoke) run cannot select a point")
        if cfg["mode"] in NORM_MODES and (s.get("clip_C") is not None or s.get("tau") is not None
                                          or s.get("quantize") or "norms_raw" not in r):
            raise SystemExit(f"{name}: P-TA/P-FR grid runs need C, tau and quantization off and saved norms (DL-31)")
        if cfg["mode"] == "ppe" and s.get("cloud_eta") != 1.0:
            raise SystemExit(f"{name}: P-PE needs eta_c = 1.0 (DL-40)")
        grid.setdefault((cfg["mode"], (float(s["lr"]), float(s["c_ex"]))), {})[cfg["seed"]] = record(name, r)
    return grid


def index_reuse(runs):
    """reuse[mode][seed] for the reused E1 (P-PE: DL-40) runs at the shared point."""
    reuse = {}
    for name, _, r in runs:
        cfg, s = r["experiment_config"]["content"], r["settings"]
        if cfg["mode"] not in REUSE_MODES or r["experiment_config"].get("smoke_rounds"):
            raise SystemExit(f"{name}: not a full E1 run of P-TG, P-TG-c or P-PE")
        if (float(s["lr"]), float(s["c_ex"])) != SHARED or (cfg["mode"] == "ppe" and s.get("cloud_eta") != 1.0):
            raise SystemExit(f"{name}: not at the shared point (lr 0.01, C_ex 2.0; P-PE eta_c 1.0)")
        reuse.setdefault(cfg["mode"], {})[cfg["seed"]] = record(name, r)
    return reuse


def reuse_check(grid, reuse):
    """Seed 0 of the shared point, run again with this code, against the reused run (None: not run)."""
    out = {}
    for mode in REUSE_MODES:
        new, old = grid.get((mode, SHARED), {}).get(0), reuse.get(mode, {}).get(0)
        if new is None or old is None:
            out[mode] = None
            continue
        diff = max(abs(new["f1"] - old["f1"]), abs(new["auroc"] - old["auroc"]))
        out[mode] = {"new_f1": new["f1"], "reused_f1": old["f1"], "new_auroc": new["auroc"],
                     "reused_auroc": old["auroc"], "max_abs_diff": diff, "match": bool(diff <= REPRO_TOL)}
    return out


def pooled_norms(rows):
    dev = np.concatenate([np.asarray(r["norms_raw"]["device_update"], dtype=np.float64) for r in rows])
    grp = np.concatenate([np.asarray(r["norms_raw"]["group_sum"], dtype=np.float64) for r in rows])
    return {"clip_C": float(np.percentile(dev, PERCENTILE)), "tau": float(np.percentile(grp, PERCENTILE)),
            "n_device_norms": int(dev.size), "n_group_norms": int(grp.size)}


def select(grid, reuse, n_seeds=3, smoke=False, partial=False):
    seeds = list(range(n_seeds))
    check = reuse_check(grid, reuse)
    designs, problems = {}, []
    for mode, name in DESIGNS.items():
        points, used = {}, {}
        for lr in LRS:
            for c_ex in C_EXS:
                p = (lr, c_ex)
                rows, source = grid.get((mode, p), {}), "grid"
                if mode in REUSE_MODES and p == SHARED and not all(s in rows for s in seeds):
                    rows, source = reuse.get(mode, {}), "reused E1 runs" + (" (DL-40)" if mode == "ppe" else "")
                    if not (smoke or partial):
                        if check[mode] is None:
                            problems.append(f"{name}: the reuse check (shared point, seed 0) was not run")
                        elif not check[mode]["match"]:
                            problems.append(f"{name}: the reuse check failed; run its shared point for every seed")
                have = [s for s in seeds if s in rows]
                if len(have) < n_seeds and not (smoke or partial):
                    problems.append(f"{name} lr {lr:g} C_ex {c_ex:g}: seeds {have}, expected {seeds}")
                if not have:
                    continue
                f1 = [rows[s]["f1"] for s in have]
                points[p] = {"lr": lr, "c_ex": c_ex, "source": source, "seeds": have,
                             "macro_f1_tuned_mean": float(np.mean(f1)),
                             "macro_f1_tuned_sd": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
                             "macro_f1_tuned_per_seed": f1,
                             "macro_auroc_mean": float(np.mean([rows[s]["auroc"] for s in have])),
                             "files": [rows[s]["file"] for s in have]}
                used[p] = [rows[s] for s in have]
        designs[mode] = decide(mode, points, used, smoke, partial)
    if problems:
        raise SystemExit("the DL-42 grid is not complete or not valid:\n  " + "\n  ".join(problems))
    devices = sorted({r["device"] for g in (grid, reuse) for rows in g.values() for r in rows.values()})
    return {"rule": "DL-42 (2): per design, keep the shared point (lr 0.01, C_ex 2.0) unless the best mean fold-9 "
                    "macro-F1 (DL-36) over seeds 0-2 is more than 1.0 point higher; P-TA/P-FR C and tau = 99th "
                    "percentiles of the pooled norms at the selected point (DL-31)",
            "tolerance": TOLERANCE, "n_seeds": n_seeds, "smoke": smoke, "partial": partial, "devices": devices,
            "reuse_check": check, "designs": designs,
            "changed": [m for m, d in designs.items() if d["decision"] == "change"]}


def decide(mode, points, used, smoke, partial):
    out = {"configuration": DESIGNS[mode], "points": {point_tag(*p): v for p, v in points.items()},
           "decision": None, "selected": None}
    if not points:
        return out
    order = sorted(points, key=lambda p: (-points[p]["macro_f1_tuned_mean"], p != SHARED, p[0], p[1]))
    best = order[0]
    shared = points.get(SHARED, {}).get("macro_f1_tuned_mean")
    out.update(best_point=point_tag(*best), best_mean=points[best]["macro_f1_tuned_mean"], shared_mean=shared,
               gain_points=None if shared is None else 100 * (points[best]["macro_f1_tuned_mean"] - shared))
    if partial or (shared is None and not smoke):  # no shared point: select() stops with the reason
        return out
    if smoke:  # tests only: switch to the best other point, so that the rerun path is exercised
        others = [p for p in order if p != SHARED]
        chosen = others[0] if others else SHARED
        out["smoke_forced"] = True
    else:
        chosen = best if points[best]["macro_f1_tuned_mean"] - shared > TOLERANCE + 1e-12 else SHARED
    out["decision"] = "keep" if chosen == SHARED else "change"
    sel = {"lr": chosen[0], "c_ex": chosen[1]}
    if out["decision"] == "change" and mode in NORM_MODES:
        norms = pooled_norms(used[chosen])
        if mode == "pfr":  # P-FR has no norm check (DL-37)
            norms.pop("tau")
        sel.update(norms)
    out["selected"] = sel
    sides = [f"{name} at the {'lowest' if v == min(grid) else 'highest'} grid value"
             for name, v, grid in (("lr", chosen[0], LRS), ("C_ex", chosen[1], C_EXS)) if v in (min(grid), max(grid))]
    out["at_grid_edge"], out["grid_edge_sides"] = bool(sides), sides  # DL-42 (2): reported as a limitation
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["check", "select"])
    ap.add_argument("--runs", required=True, help="folder with the DL42_*.json runs")
    ap.add_argument("--reuse", required=True, help="folder with the reused E1_ptg/E1_ptgc/E1_ppe runs, seeds 0-2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="tests only: shortened runs; every design switches")
    ap.add_argument("--partial", action="store_true", help="table of an unfinished grid; selects nothing")
    args = ap.parse_args(argv)
    grid_runs, reuse_runs = load(args.runs, "DL42_*.json"), load(args.reuse, "E1_*.json")
    grid, reuse = index_grid(grid_runs, smoke=args.smoke), index_reuse(reuse_runs)
    if args.what == "check":
        out = reuse_check(grid, reuse)
        for mode, v in out.items():
            print(f"reuse check {DESIGNS[mode]:7}: " + ("not run" if v is None else
                  f"new {v['new_f1']:.6f} vs reused {v['reused_f1']:.6f} -> {'MATCH' if v['match'] else 'DIFFERENT'}"),
                  flush=True)
    else:
        out = select(grid, reuse, args.n_seeds, smoke=args.smoke, partial=args.partial)
        out["runs_sha256"] = {name: sha for name, sha, _ in grid_runs + reuse_runs}
        for mode, d in out["designs"].items():
            for tag, v in d["points"].items():
                print(f"{d['configuration']:7} {tag:15} macro-F1 {v['macro_f1_tuned_mean']:.4f} "
                      f"+/- {v['macro_f1_tuned_sd']:.4f}  AUROC {v['macro_auroc_mean']:.3f}  ({v['source']})",
                      flush=True)
            print(f"{d['configuration']:7} -> {d['decision']} {d['selected']}", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
