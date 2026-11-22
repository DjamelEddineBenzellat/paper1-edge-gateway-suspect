"""Write the E6 configs of Paper 1 (plan task X-8; DL-41 (1), DL-42 (5), DL-43 (12)-(15)).

norms: the benign runs that set P-TA's tau(k), and C and tau per eps (DL-43 (12)): P-TA at its final lr and C_ex,
       with C, tau and quantization off and every norm saved, seeds 0-2: k in {1, 2, 5} at eps0 = 8 (9 runs) and
       eps in {1, 3} at k = 3 (6 runs). Fold 9 only.
lines: after e6_norms.py has pooled those norms: the eps line (P-TA and P-TG-c, eps in {1, 3}, seeds 0-2: 12 runs)
       and the k line at f = 0 (P-TA and P-FR, k in {1, 2, 5}, seeds 0-2: 18 runs), sealed fold 10 (DL-43 (1)).
       Each design at its final configuration; z from G0a_noise_multipliers.csv through eps0 (DL-43 (13)); C stays
       P-TA's (P-FR's) final C across k (DL-43 (12)). Every P-TA run uses the size-aware check (DL-44 (1)).

Examples:
    python make_e6_configs.py norms --e1 <E1 configs> --dl40 <DL-40 configs> --selection selection_dl42.json \
        --out 05_Experiments/configs
    python make_e6_configs.py lines ... --norms e6_norms.json --out 05_Experiments/configs
"""
import argparse
import hashlib
import json
import os
import subprocess

from make_dl42_configs import PROJECT, base_config, write
from make_e4_configs import final_settings, load_selection

KS = (1, 2, 5)
EPS = (1.0, 3.0)
SEEDS = range(3)
RULES = ["DL-41", "DL-42", "DL-43", "DL-44"]


def tag(v):
    return f"{v:g}".replace(".", "p")


def common(base, sel_name, sel_sha, commit, **extra):
    name = f"E1_{base['mode']}_s{base['seed']}.yaml"
    cfg = dict(base, rules=list(dict.fromkeys(base["rules"] + RULES)), code_commit=commit,
               sources_sha256=dict(base["sources_sha256"], **{sel_name: sel_sha}), **extra)
    cfg.pop("supersedes", None)
    return cfg, name


def make_norms(e1_dir, dl40_dir, selection_path, out_root, commit):
    sel, sel_sha = load_selection(selection_path)
    out = os.path.join(out_root, "E6_norms")
    os.makedirs(out, exist_ok=True)
    written = []
    for seed in SEEDS:
        base, name, sha = base_config(e1_dir, dl40_dir, "pta", seed)
        s = {k: v for k, v in final_settings(base, sel).items() if k not in ("clip_C", "tau", "quantize")}
        s["save_norms"] = True  # DL-31 procedure (DL-43 (12))
        runs = [(f"E6n_pta_k{k}_s{seed}", dict(s, k=k)) for k in KS]
        runs += [(f"E6n_pta_eps{tag(e)}_s{seed}", dict(s, eps0=e)) for e in EPS]
        for run_id, settings in runs:
            cfg, _ = common(base, os.path.basename(selection_path), sel_sha, commit, experiment="E6 norms",
                            run_id=run_id, settings=settings, fold10="none")
            cfg["sources_sha256"][name] = sha
            written.append(write(out, cfg, "# E6 norm-run config (DL-43 (12)), written by make_e6_configs.py. "
                                           "Do not edit by hand.\n"))
    return written


def make_lines(e1_dir, dl40_dir, selection_path, norms_path, out_root, commit, subdir="E6"):
    sel, sel_sha = load_selection(selection_path)
    with open(norms_path, "rb") as fh:
        raw = fh.read()
    norms = json.loads(raw)
    out = os.path.join(out_root, subdir)  # next to G0a.yaml, so that base_config ../G0a.yaml resolves
    os.makedirs(out, exist_ok=True)
    written = []
    for seed in SEEDS:
        runs = []
        for mode in ("pta", "ptgc"):  # the eps line
            base, name, sha = base_config(e1_dir, dl40_dir, mode, seed)
            for e in EPS:
                s = dict(final_settings(base, sel), eps0=e)
                if mode == "pta":
                    n = norms["eps"][tag(e)]
                    s.update(clip_C=n["clip_C"], tau=n["tau"], quantize=True)
                runs.append((base, name, sha, f"E6_{mode}_eps{tag(e)}_s{seed}", s))
        for mode in ("pta", "pfr"):  # the k line at f = 0
            base, name, sha = base_config(e1_dir, dl40_dir, mode, seed)
            for k in KS:
                s = dict(final_settings(base, sel), k=k)
                if mode == "pta":
                    s["tau"] = norms["k"][str(k)]["tau"]
                runs.append((base, name, sha, f"E6_{mode}_k{k}_s{seed}", s))
        for base, name, sha, run_id, s in runs:
            cfg, _ = common(base, os.path.basename(selection_path), sel_sha, commit, experiment="E6", run_id=run_id,
                            settings=s)
            cfg["sources_sha256"].update({name: sha, os.path.basename(norms_path): hashlib.sha256(raw).hexdigest()})
            written.append(write(out, cfg, "# E6 config (DL-43), written by make_e6_configs.py. Do not edit by hand.\n"))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["norms", "lines"])
    ap.add_argument("--e1", required=True)
    ap.add_argument("--dl40", required=True)
    ap.add_argument("--selection", required=True, help="selection_dl42.json of the DL-42 job")
    ap.add_argument("--norms", help="lines: e6_norms.json from e6_norms.py")
    ap.add_argument("--out", required=True, help="configs folder; E6_norms/ or E6/ is made in it")
    ap.add_argument("--subdir", default="E6", help="lines: the folder made in --out (job preflight: E6_preflight)")
    ap.add_argument("--commit", help="code commit to record (default: git HEAD)")
    args = ap.parse_args(argv)
    commit = args.commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True,
                                           text=True, check=True).stdout.strip()
    if args.what == "norms":
        written = make_norms(args.e1, args.dl40, args.selection, args.out, commit)
    else:
        if not args.norms:
            ap.error("lines needs --norms")
        written = make_lines(args.e1, args.dl40, args.selection, args.norms, args.out, commit, args.subdir)
    print(f"{len(written)} E6 {args.what} configs written (code commit {commit[:7]})", flush=True)


if __name__ == "__main__":
    main()
