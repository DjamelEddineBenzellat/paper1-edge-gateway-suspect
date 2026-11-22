"""Open the E6 fold-10 files and test H3(b) and H3(c) (plan task X-10; DL-09, DL-11, DL-25, DL-41, DL-43 (1),
(14), (15), DL-44 (2)).

Run only after this script and the job's seal register (made by `e4_fold10.py register`) are committed.
The runs (seeds 0-2): E6_pta_eps{1,3}, E6_ptgc_eps{1,3}, E6_pta_k{1,2,5}, E6_pfr_k{1,2,5}. The k = 3, eps = 8 values
of the same seeds come from the final f = 0 runs: P-TA the E4 references with the size-aware check (DL-44 (2)),
P-TG-c the DL-42 reruns, P-FR the E1 runs. Every sealed file is opened only with a matching seal and its run's
fold-9 thresholds.
- H3(b) (DL-11, DL-43 (15)): the mean gap P-TG-c - P-TA (fold-10 macro-F1, points, seeds 0-2) at eps 1, 3 and 8;
  it holds if gap(1) > gap(3) > gap(8). "Not evaluable" if P-TA at eps 1 is within 2 points of the trivial classifier
  on fold 10 (DL-25: the mean over classes of 2p/(1+p), p the fold-10 prevalence, from the PTB-XL labels).
- H3(c) (DL-09, DL-43 (14)): for k in {1, 2, 5}, gap(k) - gap(3), paired over seeds 0-2; P-TG-c has no groups, so this
  equals P-TA(k=3) - P-TA(k). A k fails only if |mean| > 1.5 points AND its CI excludes 0, tested with Holm-adjusted
  p < 0.05 over the three k (PR §7). H3(c) fails if any k fails.
- Reported only: P-FR at each k; AUROC; the tau drop rate of each P-TA run; the pooled norms (e6_norms.json).

Example:
    python e6_fold10.py --job <job> --e1 <E1 job> --dl40 <DL-40 job> --dl42 <DL-42 job> --ptbxl-dir <PTB-XL> \
        --g1-register g1_check.json --dl42-register dl42_seal_register.json --register e6e4_seal_register.json \
        --out e6_fold10.json
"""
import argparse
import json
import os

import numpy as np

from dl42_fold10 import committed
from e1_fold10 import paired, summary
from e4_fold10 import holm, load_scores, merged_registers, p_value, reference

SEEDS3 = [0, 1, 2]
EPS = ("1", "3")
KS = ("1", "2", "5")
H3C_LIMIT = 1.5  # points (DL-09)
DL11_MARGIN = 2.0  # points (DL-11)
SEALED = ".fold10.sealed.json"


def e6(job, run_id):
    return job, os.path.join(job, "e6_runs", run_id + ".json"), os.path.join(job, "sealed_fold10", run_id + SEALED)


def trivial_fold10(ptbxl_dir):
    """DL-25 on fold 10, from the labels only (no model output)."""
    from data_ptbxl import label_matrix, load_records, split_folds
    from metrics import trivial_macro_f1
    records, _ = load_records(ptbxl_dir)
    return trivial_macro_f1(label_matrix(split_folds(records)["test"]))


def tau_drop_rates(job, run_ids):
    out = {}
    for run_id in run_ids:
        path = e6(job, run_id)[1]
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                g = json.load(fh).get("gateway_check") or {}
            if g.get("group_sums"):
                out[run_id] = g["dropped_group_sums"] / g["group_sums"]
    return out


def analyse(job, e1_job, dl40_job, dl42_job, registers, trivial):
    missing = []

    def scores(paths, label):
        got, miss = load_scores(paths, registers)
        missing.extend(dict(m, configuration=label) for m in miss)
        return got

    base = {name: scores({s: reference(job, e1_job, dl40_job, dl42_job, mode, s) for s in SEEDS3}, f"{name} k3 eps8")
            for name, mode in (("P-TA", "pta"), ("P-TG-c", "ptgc"), ("P-FR", "pfr"))}
    eps = {(m, e): scores({s: e6(job, f"E6_{m}_eps{e}_s{s}") for s in SEEDS3}, f"{m} eps {e}")
           for m in ("pta", "ptgc") for e in EPS}
    kk = {(m, k): scores({s: e6(job, f"E6_{m}_k{k}_s{s}") for s in SEEDS3}, f"{m} k {k}")
          for m in ("pta", "pfr") for k in KS}
    gaps = {"8": paired(base["P-TG-c"], base["P-TA"])}
    gaps.update({e: paired(eps[("ptgc", e)], eps[("pta", e)]) for e in EPS})
    means = {e: gaps[e]["mean"] for e in ("1", "3", "8")}
    pta1 = summary([eps[("pta", "1")][s]["f1"] for s in eps[("pta", "1")]])
    evaluable = pta1 is not None and 100 * (pta1["mean"] - trivial) > DL11_MARGIN
    h3b = {"gap_points": gaps, "mean_gap_points": means, "pta_eps1_f1": pta1, "trivial_fold10": trivial,
           "evaluable": evaluable,
           "holds": None if not evaluable or None in means.values()
           else bool(means["1"] > means["3"] > means["8"]),
           "rule": "mean gap(eps 1) > gap(eps 3) > gap(eps 8); not evaluable if P-TA at eps 1 is within 2 points of "
                   "the trivial classifier on fold 10 (DL-11, DL-25)"}
    diffs = {k: paired(base["P-TA"], kk[("pta", k)]) for k in KS}  # gap(k) - gap(3) = P-TA(3) - P-TA(k)
    p = {k: p_value(v["per_seed_points"]) for k, v in diffs.items()}
    adj = holm(p)
    per_k = {}
    for k, v in diffs.items():
        large = v["mean"] is not None and abs(v["mean"]) > H3C_LIMIT
        per_k[k] = dict(v, p=p[k], p_holm=adj.get(k), fails=bool(large and adj.get(k) is not None and adj[k] < 0.05))
    h3c = {"per_k": per_k, "fails": any(v["fails"] for v in per_k.values()),
           "gap_k_points": {k: paired(base["P-TG-c"], kk[("pta", k)]) for k in KS},
           "rule": "fails only if |gap(k) - gap(3)| > 1.5 points and the CI excludes 0 (Holm-adjusted p < 0.05 "
                   "over k = 1, 2, 5) (DL-09, DL-43 (14))"}
    table = {f"{m} eps {e}": {x: summary([v[s][x] for s in v]) for x in ("f1", "auroc")} for (m, e), v in eps.items()}
    table.update({f"{m} k {k}": {x: summary([v[s][x] for s in v]) for x in ("f1", "auroc")} for (m, k), v in kk.items()})
    table.update({f"{n} k 3 eps 8 (reference)": {x: summary([v[s][x] for s in v]) for x in ("f1", "auroc")}
                  for n, v in base.items()})
    norms_path = os.path.join(job, "e6_norms.json")
    norms = None
    if os.path.exists(norms_path):
        with open(norms_path, encoding="utf-8") as fh:
            norms = {k: v for k, v in json.load(fh).items() if k in ("k", "eps")}
    run_ids = [f"E6_pta_{x}_s{s}" for x in [f"eps{e}" for e in EPS] + [f"k{k}" for k in KS] for s in SEEDS3]
    return {"note": "E6 fold 10, seeds 0-2; sealed files opened once, after the register was committed",
            "missing": missing, "table": table, "h3b": h3b, "h3c": h3c,
            "pfr_k_points_vs_k3": {k: paired(kk[("pfr", k)], base["P-FR"]) for k in KS},
            "pta_tau_drop_rate": tau_drop_rates(job, run_ids), "pooled_norms": norms}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for a in ("--job", "--e1", "--dl40", "--dl42", "--ptbxl-dir", "--g1-register", "--dl42-register", "--register",
              "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--no-git-check", action="store_true", help="tests only")
    args = ap.parse_args(argv)
    if not args.no_git_check:
        for path in (os.path.abspath(__file__), args.register):
            if not committed(path):
                raise SystemExit(f"{path} must be committed before any sealed file is opened")
    regs = merged_registers([args.g1_register, args.dl42_register, args.register])
    out = analyse(args.job, args.e1, args.dl40, args.dl42, regs, trivial_fold10(args.ptbxl_dir))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"E6 fold 10 opened; missing: {out['missing'] or 'none'}; H3(b) holds: {out['h3b']['holds']} "
          f"(evaluable: {out['h3b']['evaluable']}); H3(c) fails: {out['h3c']['fails']}", flush=True)


if __name__ == "__main__":
    main()
