"""Summarize the G0-a pilot (DL-01, DL-27) and the run times for G0-c.

G0-a statistic (DL-27): the mean over seeds 0-2 of the fold-9 macro-F1.
- P-TA: the final-round score of each pilot run.
- Local-only: each device's own model scored on fold 9, averaged over the 210
  devices, at the epoch chosen by select_grid.py (DL-32); then the mean over seeds.
Rule (DL-01): P-TA >= local-only + 5 macro-F1 points.
This script reports the numbers and the rule. The G0 decision itself is
written by Claude and approved by the student (plan task G0-18).
P-TG-c and the G1 comparison are shown for information only.

Example:
    python summarize_g0a.py --runs <pilot folder> --selection selection.json --margin 5 --out g0a_summary.json
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--margin", type=float, default=5.0, help="DL-01 margin in macro-F1 points")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.selection, encoding="utf-8") as fh:
        sel = json.load(fh)
    epoch = str(sel["local_only"]["epoch"])

    def runs(prefix):
        out = []
        for path in sorted(glob.glob(os.path.join(args.runs, f"{prefix}_s*.json"))):
            with open(path, encoding="utf-8") as fh:
                out.append(json.load(fh))
        return out

    pta, ptgc, local = runs("pta_pilot"), runs("ptgc_pilot"), runs("local_pilot")
    f1 = {"P-TA": {r["seed"]: r["final_fold9"]["macro_f1"] for r in pta},
          "P-TG-c": {r["seed"]: r["final_fold9"]["macro_f1"] for r in ptgc},
          "local-only": {r["seed"]: r["local_only"][epoch]["mean_macro_f1"] for r in local}}
    mean = {k: (float(np.mean(list(v.values()))) if v else None) for k, v in f1.items()}
    trivial = (pta or ptgc or local)[0]["data"]["trivial_macro_f1_fold9"]
    gap_points = None if None in (mean["P-TA"], mean["local-only"]) else 100 * (mean["P-TA"] - mean["local-only"])
    private = pta or ptgc
    eps_values = sorted({r["privacy"]["eps0"] for r in private})
    if len(eps_values) > 1:
        raise SystemExit(f"pilot runs mix several epsilons: {eps_values}")
    summary = {
        "rule": f"DL-01: P-TA >= local-only + {args.margin:g} macro-F1 points (fold 9, mean over seeds, DL-27)",
        "eps0_private_runs": eps_values[0] if eps_values else None,
        "z_private_runs": private[0]["privacy"]["z"] if private else None,
        "seeds": {k: sorted(v) for k, v in f1.items()},
        "per_seed_macro_f1": f1,
        "mean_macro_f1": mean,
        "pta_minus_local_points": gap_points,
        "g0a_rule_result": None if gap_points is None else ("PASS" if gap_points >= args.margin else "FAIL"),
        "trivial_macro_f1_fold9": trivial,
        "info_g1_preview_pta_minus_ptgc_points": (None if None in (mean["P-TA"], mean["P-TG-c"])
                                                  else 100 * (mean["P-TA"] - mean["P-TG-c"])),
        "selection": {k: sel[k] for k in ("pta_default_point", "local_only")},
        "g0c_inputs": {
            "pta_wall_s": [r["timing"]["wall_s"] for r in pta],
            "pta_wall_s_mean": float(np.mean([r["timing"]["wall_s"] for r in pta])) if pta else None,
            "ptgc_wall_s_mean": float(np.mean([r["timing"]["wall_s"] for r in ptgc])) if ptgc else None,
            "local_wall_s_mean": float(np.mean([r["timing"]["wall_s"] for r in local])) if local else None,
            "device": (pta or ptgc or local)[0]["machine"],
        },
        "info_h4_benign_flag_rate_pta": [r["gateway_check"]["flag_rate"] for r in pta],
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: summary[k] for k in ("eps0_private_runs", "mean_macro_f1", "pta_minus_local_points",
                                               "g0a_rule_result", "trivial_macro_f1_fold9")}, indent=1))


if __name__ == "__main__":
    main()
