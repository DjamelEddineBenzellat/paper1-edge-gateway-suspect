"""The E4 positive control (DL-43 (7), risk plan R11): do the attacks work at all?

On undefended non-DP FedAvg, seed 0, 100 rounds, fold 9:
- backdoor: success >= the clean "NORM only" rate of E1 FedAvg seed 0 + 30 points;
- label flipping: macro-F1 (DL-36) at least 5 points below E1 FedAvg seed 0.
If either fails, the E4 runs must not start: the attack is strengthened inside K1 by a new DL row (R19); the trigger
and the defences never change. Exit code 0 = PASS, 2 = FAIL.

Example:
    python e4_positive_control.py --runs <folder with E4pc_*.json> --e1-fedavg E1_fedavg_s0.json --out pc.json
"""
import argparse
import json
import os
import sys

BACKDOOR_MARGIN = 0.30  # 30 points above the clean rate
LABELFLIP_DROP = 0.05  # 5 macro-F1 points


def check(backdoor_run, labelflip_run, e1_fedavg, smoke=False):
    for r, kind in ((backdoor_run, "backdoor"), (labelflip_run, "labelflip")):
        a = r.get("attack") or {}
        if r["mode"] != "fedavg" or a.get("kind") != kind or (r["experiment_config"].get("smoke_rounds")
                                                              and not smoke):
            raise SystemExit(f"{r['run_id']}: not a full FedAvg {kind} positive-control run")
    clean = e1_fedavg["final_fold9"]["backdoor_success_clean"]
    bd = backdoor_run["final_fold9"]["backdoor_success_clean"]  # DL-38 (4), measured on the attacked model
    f1_clean = e1_fedavg["final_fold9"]["macro_f1_tuned"]
    f1_flip = labelflip_run["final_fold9"]["macro_f1_tuned"]
    out = {"backdoor": {"success": bd, "clean_rate_e1": clean, "rule": f">= clean + {BACKDOOR_MARGIN:.2f}",
                        "passes": bool(bd >= clean + BACKDOOR_MARGIN)},
           "labelflip": {"macro_f1": f1_flip, "macro_f1_e1": f1_clean, "drop_points": 100 * (f1_clean - f1_flip),
                         "rule": f"drop >= {100 * LABELFLIP_DROP:.0f} points",
                         "passes": bool(f1_clean - f1_flip >= LABELFLIP_DROP)}}
    out["passes"] = out["backdoor"]["passes"] and out["labelflip"]["passes"]
    out["smoke"] = smoke
    return out


DL49_KINDS = ("labelflip", "concentrated_labelflip")  # DL-49 (2): the fail-fast control of the E4 job


def check_dl49(runs, e1_fedavg, smoke=False):
    """DL-49 (2): each label flip lowers macro-F1 by >= 5 points against E1 FedAvg seed 0. runs: {kind: run JSON}."""
    f1_clean = e1_fedavg["final_fold9"]["macro_f1_tuned"]
    out = {"macro_f1_e1": f1_clean, "rule": f"drop >= {100 * LABELFLIP_DROP:.0f} points", "smoke": smoke}
    for kind in DL49_KINDS:
        r = runs[kind]
        if (r["mode"] != "fedavg" or (r.get("attack") or {}).get("kind") != kind
                or bool(r["experiment_config"].get("smoke_rounds")) != smoke):
            raise SystemExit(f"{r['run_id']}: not the {'smoke' if smoke else 'full'} DL-49 positive-control run {kind}")
        f1 = r["final_fold9"]["macro_f1_tuned"]
        out[kind] = {"macro_f1": f1, "drop_points": 100 * (f1_clean - f1),
                     "passes": bool(f1_clean - f1 >= LABELFLIP_DROP)}
    out["passes"] = all(out[k]["passes"] for k in DL49_KINDS)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, help="folder with E4pc_fedavg_backdoor_s0.json and _labelflip_s0.json")
    ap.add_argument("--e1-fedavg", required=True, help="the E1 run E1_fedavg_s0.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="job preflight only: shortened runs; never exits 2")
    ap.add_argument("--dl49", action="store_true", help="the DL-49 control: both label flips (E4pc49_fedavg_*_s0)")
    args = ap.parse_args(argv)
    if args.dl49:
        runs = {}
        for kind in DL49_KINDS:
            with open(os.path.join(args.runs, f"E4pc49_fedavg_{kind}_s0.json"), encoding="utf-8") as fh:
                runs[kind] = json.load(fh)
        with open(args.e1_fedavg, encoding="utf-8") as fh:
            out = check_dl49(runs, json.load(fh), smoke=args.smoke)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print("positive control (DL-49): " + "; ".join(
            f"{k} drop {out[k]['drop_points']:.1f} points -> {'PASS' if out[k]['passes'] else 'FAIL'}"
            for k in DL49_KINDS), flush=True)
        sys.exit(0 if out["passes"] or args.smoke else 2)
    runs = {}
    for kind in ("backdoor", "labelflip"):
        with open(os.path.join(args.runs, f"E4pc_fedavg_{kind}_s0.json"), encoding="utf-8") as fh:
            runs[kind] = json.load(fh)
    with open(args.e1_fedavg, encoding="utf-8") as fh:
        e1 = json.load(fh)
    out = check(runs["backdoor"], runs["labelflip"], e1, smoke=args.smoke)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"positive control: backdoor {out['backdoor']['success']:.3f} (clean {out['backdoor']['clean_rate_e1']:.3f}) "
          f"-> {'PASS' if out['backdoor']['passes'] else 'FAIL'}; label flip drop "
          f"{out['labelflip']['drop_points']:.1f} points -> {'PASS' if out['labelflip']['passes'] else 'FAIL'}",
          flush=True)
    sys.exit(0 if out["passes"] or args.smoke else 2)


if __name__ == "__main__":
    main()
