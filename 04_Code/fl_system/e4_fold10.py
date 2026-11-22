"""Open the E4 fold-10 files and analyse E4 (plan task X-10; DL-07, DL-41, DL-43 (1), (6), DL-44 (2), DL-49).

Two steps, each run once, in this order:
  register  hashes every sealed file of a job WITHOUT reading it and checks them against the workstation's results
            MANIFEST. The E6 + E4 job (E6 runs and the P-TA references) and the DL-49 E4 job each get a register;
            a register is committed before any `open`.
  open      only after this script and the registers are committed. The attacked runs (DL-49 E4 job) are
            E4_<mode>_<attack>_s<seed>, attack = labelflip (uniform) or concentrated_labelflip. The f = 0 reference of
            the same design and seed: P-TA the E4 references with the size-aware check of the E6 + E4 job (DL-44 (2);
            valid under DL-48, DL-49 (5)); P-TG-c the DL-42 reruns; P-TG and P-FR the E1 runs; P-PE the DL-40 runs.
            A sealed file is opened only if its SHA-256 matches its register and only with its run's fold-9
            thresholds.
Output (fold 10, 10 seeds, points):
- per design and attack: macro-F1, AUROC; the macro-F1 drop and the AUROC drop, paired with the same design's and
  seed's f = 0 reference (DL-43 (6)); P-TA and P-FR: poisoned group sums and those dropped at tau;
- H5(a) backdoor part and DL-24: not evaluable (the backdoor failed its positive control, DL-47 (4), DL-49 (3));
- S1 (secondary, DL-49 (4)): P-TA - P-FR macro-F1 drop under the uniform label flip, TOST +-3 points, DL-07 outcomes
  ("P-TA worse": its drop is larger, the paired 95% CI is above 0);
- S2 (secondary, DL-49 (4)): P-TG - P-TA and P-PE - P-TA macro-F1 drop under the uniform label flip; "lower" if the
  paired 95% CI is below 0; Holm-adjusted p-values reported next to it;
- the DL-49 positive control of the job (from its results; not sealed).

Examples:
    python e4_fold10.py register --sealed <job>/sealed_fold10 --zip <results zip> --out e4dl49_seal_register.json
    python e4_fold10.py open --job <DL-49 E4 job> --ref-job <E6 + E4 job> --e1 <E1 job> --dl40 <DL-40 job> \
        --dl42 <DL-42 job> --g1-register g1_check.json --dl42-register dl42_seal_register.json \
        --ref-register e6e4_seal_register.json --register e4dl49_seal_register.json --out e4_fold10.json
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

from dl42_fold10 import committed, load_scores as _load_scores, register
from e1_fold10 import SEEDS, paired, sources, summary

DESIGNS = (("P-TG", "ptg"), ("P-TG-c", "ptgc"), ("P-PE", "ppe"), ("P-FR", "pfr"), ("P-TA", "pta"))
ATTACKS = ("labelflip", "concentrated_labelflip")  # DL-49 (1)
DEFAULT = "labelflip"  # DL-49 (4): S1 and S2 use the uniform label flip
S1_MARGIN = 3.0  # points, DL-49 (4)
NOT_EVALUABLE = ("not evaluable: the backdoor failed its positive control on undefended FedAvg (gate job "
                 "JOB_20261119_X-9_e4-gate, best +4.2 points over the clean rate, rule +30; DL-47 (4), DL-49 (3))")
SEALED = ".fold10.sealed.json"


def attacked(job, mode, attack, seed):
    run_id = f"E4_{mode}_{attack}_s{seed}"
    return job, os.path.join(job, "e4_runs", run_id + ".json"), os.path.join(job, "sealed_fold10", run_id + SEALED)


def reference(job, e1_job, dl40_job, dl42_job, mode, seed):
    """The design's f = 0 run at its final configuration (DL-42 (5), DL-44 (2))."""
    if mode == "pta":
        run_id = f"E4ref_pta_s{seed}"
        return job, os.path.join(job, "e4_reference_runs", run_id + ".json"), \
            os.path.join(job, "sealed_fold10", run_id + SEALED)
    if mode == "ptgc":
        run_id = f"E1tuned_ptgc_s{seed}"
        return dl42_job, os.path.join(dl42_job, "tuned_runs", run_id + ".json"), \
            os.path.join(dl42_job, "sealed_fold10", run_id + SEALED)
    return sources(e1_job, dl40_job, mode, seed)


def check_attack_runs(job, norms):
    """Every attacked run carries its design, attack and the one attack norm (DL-43 (3))."""
    for _, mode in DESIGNS:
        for attack in ATTACKS:
            for seed in SEEDS:
                path = attacked(job, mode, attack, seed)[1]
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as fh:
                    r = json.load(fh)
                a = r.get("attack") or {}
                if (r["mode"], r["seed"], a.get("kind")) != (mode, seed, attack):
                    raise SystemExit(f"{path}: not the E4 run of {mode} {attack} seed {seed}")
                norms.add(round(float(a["norm"]), 9))
    if len(norms) != 1:
        raise SystemExit(f"E4 runs with different attack norms: {sorted(norms)} (DL-43 (3): one norm)")


def load_scores(paths, registers):
    """Fold-10 measures per seed; a run file that does not exist (a skipped run) is a missing seed."""
    present = {s: p for s, p in paths.items() if os.path.exists(p[1])}
    scores, missing = _load_scores(present, registers)
    missing += [{"seed": s, "file": os.path.basename(p[1]), "reason": "run file missing"}
                for s, p in paths.items() if s not in present]
    return scores, missing


def p_value(points):
    """Two-sided paired-t p-value; with no spread, 0 for a non-zero mean and 1 for a zero mean."""
    d = np.asarray(points, dtype=float)
    if len(d) < 2:
        return None
    if d.std(ddof=1) == 0:
        return 0.0 if d.mean() != 0 else 1.0
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(2 * stats.t.sf(abs(t), len(d) - 1))


def holm(pvals):
    """Holm-adjusted p-values, {name: p} -> {name: p_adj}."""
    items = sorted((p, n) for n, p in pvals.items() if p is not None)
    out, running = {}, 0.0
    for i, (p, n) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[n] = running
    return out


def tost(p, margin, higher_is_worse):
    """DL-07 outcomes for P-TA - other."""
    if p["sd"] is None:
        return "not evaluable (too few seeds)"
    if -margin < p["ci90"][0] and p["ci90"][1] < margin:
        return "equivalent"
    worse = p["ci95"][0] > 0 if higher_is_worse else p["ci95"][1] < 0
    return "P-TA worse" if worse else "inconclusive"


def poisoned_sums(job, mode, attack):
    total = dropped = 0
    for seed in SEEDS:
        path = attacked(job, mode, attack, seed)[1]
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                a = json.load(fh).get("attack") or {}
            total += a.get("poisoned_sums", 0) or 0
            dropped += a.get("poisoned_dropped_at_tau", 0) or 0
    return {"poisoned_group_sums": total, "dropped_at_tau": dropped,
            "passed_tau_share": None if total == 0 else 1 - dropped / total}


def drops(ref, att, key):
    """{seed: {"drop": reference - attacked}} over the seeds both have."""
    return {seed: {"drop": ref[seed][key] - att[seed][key]} for seed in sorted(set(ref) & set(att))}


def analyse(job, ref_job, e1_job, dl40_job, dl42_job, registers):
    norms = set()
    check_attack_runs(job, norms)
    missing, att, ref, table = [], {}, {}, {}
    for name, mode in DESIGNS:
        ref[name], miss = load_scores({s: reference(ref_job, e1_job, dl40_job, dl42_job, mode, s) for s in SEEDS},
                                      registers)
        missing += [dict(m, configuration=f"{name} reference") for m in miss]
        for attack in ATTACKS:
            att[(name, attack)], miss = load_scores({s: attacked(job, mode, attack, s) for s in SEEDS}, registers)
            missing += [dict(m, configuration=f"{name} {attack}") for m in miss]
            rows = att[(name, attack)]
            table.setdefault(name, {})[attack] = {
                **{k: summary([rows[s][k] for s in rows]) for k in ("f1", "auroc")},
                "f1_drop_points": paired(ref[name], rows, "f1"),
                "auroc_drop_points": paired(ref[name], rows, "auroc"),
                **(poisoned_sums(job, mode, attack) if mode in ("pta", "pfr") else {})}
        table[name]["reference_f0"] = {k: summary([ref[name][s][k] for s in ref[name]]) for k in ("f1", "auroc")}
    drop = {name: drops(ref[name], att[(name, DEFAULT)], "f1") for name, _ in DESIGNS}
    s1 = paired(drop["P-TA"], drop["P-FR"], "drop")
    s2 = {o: paired(drop[o], drop["P-TA"], "drop") for o in ("P-TG", "P-PE")}
    p = {o: p_value(v["per_seed_points"]) for o, v in s2.items()}
    adj = holm(p)
    pc_path = os.path.join(job, "positive_control_dl49.json")
    pc = None
    if os.path.exists(pc_path):
        with open(pc_path, encoding="utf-8") as fh:
            pc = json.load(fh)
    return {"note": "E4 (DL-49) fold 10; sealed files opened once, after the registers were committed; references "
                    "re-read through their registers (same final configurations)",
            "attack_norm": sorted(norms)[0], "missing": missing, "designs": table,
            "h5a_backdoor": {"outcome": NOT_EVALUABLE}, "dl24": {"outcome": NOT_EVALUABLE},
            "s1": dict(s1, outcome=tost(s1, S1_MARGIN, higher_is_worse=True), margin=S1_MARGIN,
                       measure="macro-F1 drop, P-TA - P-FR, uniform label flip (secondary, DL-49 (4))"),
            "s2": {o: dict(v, p=p[o], p_holm=adj.get(o),
                           lower_than_pta=None if v["sd"] is None else bool(v["ci95"][1] < 0),
                           measure="macro-F1 drop, other - P-TA, uniform label flip (secondary, DL-49 (4))")
                   for o, v in s2.items()},
            "positive_control": pc}


def merged_registers(paths):
    reg = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            reg.update(json.load(fh)["seal_register"])
    return reg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["register", "open"])
    ap.add_argument("--sealed", help="register: the job's sealed_fold10 folder (not opened)")
    ap.add_argument("--zip", action="append", default=[], help="register: the job's results zip")
    ap.add_argument("--job", help="open: results folder of the DL-49 E4 job")
    ap.add_argument("--ref-job", help="open: results folder of the E6 + E4 job (the P-TA references)")
    ap.add_argument("--e1", help="open: results folder of JOB_20261117_M3-01_e1")
    ap.add_argument("--dl40", help="open: results folder of JOB_20261117_M3-01_dl40-ppe")
    ap.add_argument("--dl42", help="open: results folder of the DL-42 job")
    ap.add_argument("--g1-register")
    ap.add_argument("--dl42-register")
    ap.add_argument("--ref-register", help="open: the E6 + E4 job's register (committed)")
    ap.add_argument("--register", help="open: the DL-49 E4 job's register (committed)")
    ap.add_argument("--no-git-check", action="store_true", help="tests only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.what == "register":
        if not args.sealed or not args.zip:
            ap.error("register needs --sealed and --zip")
        out = {"note": "hashes only; no sealed file was read", "seal_register": register(args.sealed, args.zip)}
        print(f"{len(out['seal_register'])} sealed files registered; all match the workstation manifest", flush=True)
    else:
        need = ("job", "ref_job", "e1", "dl40", "dl42", "g1_register", "dl42_register", "ref_register", "register")
        if any(getattr(args, k) is None for k in need):
            ap.error("open needs --job, --ref-job, --e1, --dl40, --dl42 and the five registers")
        if not args.no_git_check:
            for path in (os.path.abspath(__file__), args.register, args.ref_register):
                if not committed(path):
                    raise SystemExit(f"{path} must be committed before any sealed file is opened")
        regs = merged_registers([args.g1_register, args.dl42_register, args.ref_register, args.register])
        out = analyse(args.job, args.ref_job, args.e1, args.dl40, args.dl42, regs)
        print(f"E4 fold 10 opened; missing: {out['missing'] or 'none'}; S1: {out['s1']['outcome']}", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
