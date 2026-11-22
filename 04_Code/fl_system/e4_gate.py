"""The E4 gate (DL-47 (4)): choose the attack strength on the positive control, before any E4 job is built.

The 13 gate runs are undefended non-DP FedAvg, seed 0, 100 rounds, fold 9 (as DL-43 (7)):
- backdoor: recipe "replace" or "copies" x boost 1, 3, 10, 30;
- label flipping: boost 1, 3, 10, 30, or scaled to C (DL-43; "dl43").
Selection (written in DL-47 before any gate run):
- backdoor: the recipe and boost with the highest backdoor success (ties: the smaller boost, then "replace");
- label flipping: the boost with the largest macro-F1 drop against E1 FedAvg seed 0 (ties: the smaller boost; "dl43"
  counts as the largest);
- the concentrated attack uses the backdoor choice;
- each choice must pass its positive-control rule (e4_positive_control.py): backdoor success >= the clean rate of E1
  FedAvg seed 0 + 30 points; label flip >= 5 macro-F1 points below it.
Exit code 0 = PASS (an E4 job may be built with gate_selection.json), 2 = GATE_FAILED, 3 = runs missing.
--smoke (job preflight on shortened runs) checks the inputs and the rule but never exits 2.

Example:
    python e4_gate.py --runs <folder with E4gate_*.json> --e1-fedavg E1_fedavg_s0.json --out gate_selection.json
"""
import argparse
import json
import math
import os
import sys

from e4_positive_control import BACKDOOR_MARGIN, LABELFLIP_DROP
from make_e4_configs import gate_runs


def load_runs(folder, smoke=False):
    """{run_id: run JSON} of the gate runs found, and the missing run IDs. Each run is checked against its name."""
    runs, missing = {}, []
    for run_id, want in gate_runs():
        path = os.path.join(folder, f"{run_id}.json")
        if not os.path.exists(path):
            missing.append(run_id)
            continue
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        att = r.get("attack") or {}
        if (r["mode"] != "fedavg" or r["seed"] != 0 or att.get("kind") != want["kind"]
                or att.get("boost") != want["boost"] or not att.get("host_adjust")
                or (want["kind"] == "backdoor" and att.get("poison_mode") != want["poison_mode"])
                or (bool(r["experiment_config"].get("smoke_rounds")) != smoke)):
            raise SystemExit(f"{path}: not the {'smoke' if smoke else 'full'} gate run {run_id}")
        runs[run_id] = r
    return runs, missing


def select(runs, e1_fedavg):
    clean = e1_fedavg["final_fold9"]["backdoor_success_clean"]
    f1_clean = e1_fedavg["final_fold9"]["macro_f1_tuned"]
    backdoor, labelflip = [], []
    for run_id, want in gate_runs():
        if run_id not in runs:
            continue
        f = runs[run_id]["final_fold9"]
        att = runs[run_id]["attack"]
        row = {"run_id": run_id, "boost": want["boost"], "n_malicious": att.get("n_malicious"),
               "macro_f1": f["macro_f1_tuned"], "macro_auroc": f["macro_auroc"]}
        if want["kind"] == "backdoor":
            row.update(poison_mode=want["poison_mode"], success=f["backdoor_success_clean"],
                       passes=bool(f["backdoor_success_clean"] >= clean + BACKDOOR_MARGIN))
            backdoor.append(row)
        else:
            row.update(drop_points=100 * (f1_clean - f["macro_f1_tuned"]),
                       passes=bool(f1_clean - f["macro_f1_tuned"] >= LABELFLIP_DROP))
            labelflip.append(row)

    def boost_key(b):
        return math.inf if b is None else b

    bd = min(backdoor, key=lambda r: (-r["success"], boost_key(r["boost"]), r["poison_mode"] != "replace"),
             default=None)
    lf = min(labelflip, key=lambda r: (-r["drop_points"], boost_key(r["boost"])), default=None)
    return {"rule": "DL-47 (4): backdoor = highest success (ties: smaller boost, then replace); label flip = largest "
                    "macro-F1 drop (ties: smaller boost); each must pass its DL-43 (7) rule",
            "clean_rate_e1": clean, "macro_f1_e1": f1_clean,
            "backdoor": {"choice": None if bd is None else {"poison_mode": bd["poison_mode"], "boost": bd["boost"]},
                         "success": None if bd is None else bd["success"],
                         "passes": bool(bd and bd["passes"]), "candidates": backdoor},
            "labelflip": {"choice": None if lf is None else {"boost": lf["boost"]},
                          "drop_points": None if lf is None else lf["drop_points"],
                          "passes": bool(lf and lf["passes"]), "candidates": labelflip},
            "concentrated": "uses the backdoor choice (DL-47 (4))",
            "passes": bool(bd and bd["passes"] and lf and lf["passes"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, help="folder with the 13 E4gate_*.json run files")
    ap.add_argument("--e1-fedavg", required=True, help="the E1 run E1_fedavg_s0.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="job preflight only: shortened runs; never exits 2")
    args = ap.parse_args(argv)
    runs, missing = load_runs(args.runs, smoke=args.smoke)
    with open(args.e1_fedavg, encoding="utf-8") as fh:
        e1 = json.load(fh)
    out = dict(select(runs, e1), missing=missing, smoke=args.smoke)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for kind in ("backdoor", "labelflip"):
        for r in out[kind]["candidates"]:
            value = f"success {r['success']:.3f}" if kind == "backdoor" else f"drop {r['drop_points']:.1f} points"
            print(f"  {r['run_id']}: {value} -> {'pass' if r['passes'] else 'fail'}", flush=True)
    print(f"gate: backdoor choice {out['backdoor']['choice']} ({'PASS' if out['backdoor']['passes'] else 'FAIL'}); "
          f"label flip choice {out['labelflip']['choice']} ({'PASS' if out['labelflip']['passes'] else 'FAIL'}); "
          f"missing {len(missing)}", flush=True)
    if missing:
        sys.exit(0 if args.smoke else 3)
    sys.exit(0 if out["passes"] or args.smoke else 2)


if __name__ == "__main__":
    main()
