"""Pool the E6 norm runs into P-TA's tau(k) and its C and tau per eps (plan task X-8; DL-43 (12)).

tau(k) = the 99th percentile of the group-sum norms at k, pooled over seeds 0-2 (C stays P-TA's final C);
C(eps) and tau(eps) = the 99th percentiles of the device-update and group-sum norms at eps (k = 3), pooled over seeds
0-2. Same procedure as tau(3) (DL-31, DL-42). Every k and eps must have all three seeds; otherwise it stops.

Example:
    python e6_norms.py --runs <folder with E6n_*.json> --out e6_norms.json
"""
import argparse
import json

from make_e6_configs import EPS, KS, SEEDS, tag
from select_dl42 import load, pooled_norms


def pool(runs, smoke=False):
    by = {}
    for name, _, r in runs:
        cfg, s = r["experiment_config"]["content"], r["settings"]
        if cfg["experiment"] != "E6 norms" or cfg["mode"] != "pta" or "norms_raw" not in r:
            raise SystemExit(f"{name}: not an E6 norm run with saved norms")
        if s.get("clip_C") is not None or s.get("tau") is not None or s.get("quantize"):
            raise SystemExit(f"{name}: C, tau and quantization must be off (DL-31)")
        if r["experiment_config"].get("smoke_rounds") and not smoke:
            raise SystemExit(f"{name}: a shortened run cannot set tau or C")
        c = cfg["settings"]  # a k run sets k (eps0 stays 8); an eps run sets eps0 and keeps k = 3
        key = ("k", str(c["k"])) if "k" in c else ("eps", tag(float(c["eps0"])))
        by.setdefault(key, {})[cfg["seed"]] = {"norms_raw": r["norms_raw"], "file": name}
    want = [("k", str(k)) for k in KS] + [("eps", tag(e)) for e in EPS]
    missing = [f"{a}={b}: seeds {sorted(by.get((a, b), {}))}" for a, b in want
               if sorted(by.get((a, b), {})) != list(SEEDS)]
    if missing and not smoke:
        raise SystemExit("E6 norm runs incomplete: " + "; ".join(missing))
    out = {"rule": "DL-43 (12): 99th percentiles of the pooled norms of seeds 0-2, C, tau and quantization off",
           "k": {}, "eps": {}, "smoke": smoke}
    for (a, b), rows in sorted(by.items()):
        p = pooled_norms(list(rows.values()))
        entry = {"tau": p["tau"], "n_group_norms": p["n_group_norms"], "files": [r["file"] for r in rows.values()]}
        if a == "eps":
            entry.update(clip_C=p["clip_C"], n_device_norms=p["n_device_norms"])
        out[a][b] = entry
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tests only: shortened or incomplete runs")
    args = ap.parse_args(argv)
    runs = load(args.runs, "E6n_*.json")
    out = pool(runs, smoke=args.smoke)
    out["runs_sha256"] = {name: sha for name, sha, _ in runs}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out["k"].items():
        print(f"tau(k={k}) = {v['tau']:.3f}", flush=True)
    for e, v in out["eps"].items():
        print(f"eps {e}: C = {v['clip_C']:.3f}, tau = {v['tau']:.3f}", flush=True)


if __name__ == "__main__":
    main()
