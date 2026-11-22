"""Open the DL-42 fold-10 files and compare the DP designs at their final configurations (plan task X-5;
DL-42 (4), DL-43 (16)).

Two steps, each run once, in this order:
  register  hashes the new sealed files WITHOUT reading them and checks them against the workstation's results
            MANIFEST. The register is committed before `open`.
  open      the final configuration of each DP design: the DL-42 reruns (E1tuned_*) of the designs whose point
            changed; otherwise the E1 runs (P-PE: the DL-40 runs). A sealed file is opened only if its SHA-256
            matches its register (new files: the DL-42 register; E1 and DL-40 files: the G1 register, as in M3-03)
            and only with its run's fold-9 thresholds. This script and the DL-42 register must be committed first.
Output: fold-10 scores per design (all 10 seeds, and seeds 3-9, which were not used to choose the point); the tuned
price of distrust P-TG-c - P-TA (a sensitivity analysis next to the recorded H3(a) FAIL at the shared point, DL-42
(4)); all DP pairs; H5(a) F1 with the tuned designs (DL-07 outcomes); tuned - shared for each changed design;
P-TA - local-only (reported only).

Examples:
    python dl42_fold10.py register --sealed <DL-42 job>/sealed_fold10 --zip <results zip> --out dl42_seal_register.json
    python dl42_fold10.py open --selection <DL-42 job>/selection_dl42.json --e1 <E1 job> --dl40 <DL-40 job> \
        --dl42 <DL-42 job> --g1-register g1_check.json --dl42-register dl42_seal_register.json --out dl42_fold10.json
"""
import argparse
import itertools
import json
import os
import subprocess

from e1_fold10 import SEEDS, dl07, fold10_measures, open_sealed, paired, sources, summary
from g1_check import seal_register

MODES = {"P-TG": "ptg", "P-TG-c": "ptgc", "P-PE": "ppe", "P-FR": "pfr", "P-TA": "pta"}
FRESH_SEEDS = list(range(3, 10))  # not used by the DL-42 selection (seeds 0-2)
SEALED = ".fold10.sealed.json"


def register(sealed_dir, zips):
    """The DL-42 seal register; every file must match the workstation manifest."""
    reg = seal_register([sealed_dir], zips)
    bad = [k for k, v in reg.items() if v["matches_workstation_manifest"] is not True]
    if not reg or bad:
        raise SystemExit(f"seal register refused: {len(reg)} files, not matching the manifest: {bad}")
    return reg


def committed(path):
    """True if git tracks the file and it has no uncommitted change."""
    folder = os.path.dirname(os.path.abspath(path))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", os.path.basename(path)], cwd=folder,
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", os.path.basename(path)], cwd=folder,
                           capture_output=True).returncode == 0
    return tracked and clean


def final_sources(sel, e1_job, dl40_job, dl42_job, mode, seed):
    """(job, run JSON, sealed file, kind) of the design's final run."""
    if mode in sel["changed"]:
        return dl42_job, os.path.join(dl42_job, "tuned_runs", f"E1tuned_{mode}_s{seed}.json"), \
            os.path.join(dl42_job, "sealed_fold10", f"E1tuned_{mode}_s{seed}{SEALED}"), "tuned"
    job, run, sealed = sources(e1_job, dl40_job, mode, seed)
    return job, run, sealed, "shared"


def check_point(run, point, mode):
    """A rerun must carry the selected point (lr, C_ex; C and tau for P-TA/P-FR)."""
    s = run["settings"]
    want = {k: point[k] for k in ("lr", "c_ex", "clip_C", "tau") if k in point}
    got = {k: s.get(k) for k in want}
    if any(abs(float(got[k]) - float(want[k])) > 1e-9 for k in want):
        raise SystemExit(f"{run['run_id']}: settings {got} differ from the selected point {want} ({mode})")


def load_scores(paths, register_):
    """{seed: fold-10 measures} and the missing seeds (O2: a file is opened only with a matching seal)."""
    scores, missing = {}, []
    for seed, (job, run_path, sealed_path) in paths.items():
        with open(run_path, encoding="utf-8") as fh:
            run = json.load(fh)
        sealed, reason = open_sealed(sealed_path, job, register_)
        if sealed is None:
            missing.append({"seed": seed, "file": os.path.basename(sealed_path), "reason": reason})
            continue
        scores[seed] = fold10_measures(sealed, run)
    return scores, missing


def subset(d, seeds):
    return {s: d[s] for s in seeds if s in d}


def both(a, b, key="f1"):
    return {"all_seeds": paired(a, b, key), "seeds_3_9": paired(subset(a, FRESH_SEEDS), subset(b, FRESH_SEEDS), key)}


def analyse(sel, e1_job, dl40_job, dl42_job, g1_register, dl42_register):
    if sel.get("smoke") or sel.get("partial"):
        raise SystemExit("a smoke or partial selection cannot be used")
    final, shared, missing, table = {}, {}, [], {}
    for name, mode in MODES.items():
        d = sel["designs"][mode]
        kinds, paths = set(), {}
        for seed in SEEDS:
            job, run_path, sealed_path, kind = final_sources(sel, e1_job, dl40_job, dl42_job, mode, seed)
            if kind == "tuned":
                with open(run_path, encoding="utf-8") as fh:
                    check_point(json.load(fh), d["selected"], mode)
            kinds.add(kind)
            paths[seed] = (job, run_path, sealed_path)
        kind = kinds.pop()
        final[name], miss = load_scores(paths, dl42_register if kind == "tuned" else g1_register)
        missing += [dict(m, configuration=name) for m in miss]
        if kind == "tuned":  # the same design at the shared point, for tuned - shared
            shared[name], miss = load_scores({s: sources(e1_job, dl40_job, mode, s) for s in SEEDS}, g1_register)
            missing += [dict(m, configuration=f"{name} (shared)") for m in miss]
        rows = final[name]
        table[name] = {"final": kind, "decision": d["decision"], "point": d["selected"],
                       "grid_edge_sides": d.get("grid_edge_sides"),
                       **{k: summary([rows[s][k] for s in rows]) for k in ("f1", "auroc", "mi_sens", "backdoor")},
                       "f1_seeds_3_9": summary([rows[s]["f1"] for s in subset(rows, FRESH_SEEDS)])}
    local, miss = load_scores({s: sources(e1_job, dl40_job, "local", s) for s in SEEDS}, g1_register)
    missing += [dict(m, configuration="local-only") for m in miss]
    h5 = both(final["P-TA"], final["P-FR"])
    return {"note": "DL-42 fold 10: final configurations; new sealed files opened once, after their register was "
                    "committed; E1/DL-40 files re-read through the G1 register (same final configuration)",
            "selection_changed": sel["changed"], "missing": missing, "designs": table,
            "tuned_price_of_distrust": dict(both(final["P-TG-c"], final["P-TA"]),
                                            note="P-TG-c - P-TA at the final configurations: a sensitivity analysis. "
                                                 "H3(a) at the pre-registered shared point stays FAIL (DL-42 (4))"),
            "h5a_f1_tuned": {k: dict(v, outcome=dl07(v)) for k, v in h5.items()},
            "dp_pairs": {f"{a} - {b}": both(final[a], final[b]) for a, b in itertools.combinations(MODES, 2)},
            "tuned_minus_shared": {n: both(final[n], shared[n]) for n in shared},
            "pta_minus_local_only": both(final["P-TA"], local)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["register", "open"])
    ap.add_argument("--sealed", help="register: the DL-42 sealed_fold10 folder (not opened)")
    ap.add_argument("--zip", action="append", default=[], help="register: the DL-42 results zip")
    ap.add_argument("--selection", help="open: selection_dl42.json")
    ap.add_argument("--e1", help="open: results folder of JOB_20261117_M3-01_e1")
    ap.add_argument("--dl40", help="open: results folder of JOB_20261117_M3-01_dl40-ppe")
    ap.add_argument("--dl42", help="open: results folder of the DL-42 job")
    ap.add_argument("--g1-register", help="open: g1_check.json (E1 and DL-40 seals)")
    ap.add_argument("--dl42-register", help="open: dl42_seal_register.json (committed)")
    ap.add_argument("--no-git-check", action="store_true", help="tests only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.what == "register":
        if not args.sealed or not args.zip:
            ap.error("register needs --sealed and --zip")
        out = {"note": "hashes only; no sealed file was read", "seal_register": register(args.sealed, args.zip)}
        print(f"{len(out['seal_register'])} sealed files registered; all match the workstation manifest", flush=True)
    else:
        need = ("selection", "e1", "dl40", "dl42", "g1_register", "dl42_register")
        if any(getattr(args, k) is None for k in need):
            ap.error("open needs --selection, --e1, --dl40, --dl42, --g1-register and --dl42-register")
        if not args.no_git_check:
            for path in (os.path.abspath(__file__), args.dl42_register):
                if not committed(path):
                    raise SystemExit(f"{path} must be committed before any sealed file is opened")
        with open(args.selection, encoding="utf-8") as fh:
            sel = json.load(fh)
        regs = []
        for path in (args.g1_register, args.dl42_register):
            with open(path, encoding="utf-8") as fh:
                regs.append(json.load(fh)["seal_register"])
        out = analyse(sel, args.e1, args.dl40, args.dl42, *regs)
        print(f"opened the final files of {len(out['designs'])} DP designs; missing: {out['missing'] or 'none'}",
              flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
