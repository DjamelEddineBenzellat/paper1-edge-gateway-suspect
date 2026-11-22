"""Write the E5 configs (plan task X-7; DL-06, DL-14, DL-23, DL-41, DL-42 (5), DL-43 (8)-(11), DL-44 to DL-48).

Tampering types x detectors x seeds 0-9, on the mixed partition (partition_E5.yaml), f = 0, sealed fold 10:
- ours: P-TA at its final configuration with the size-aware check (DL-44) and rate alert (DL-48); all 5 types
  (adaptive: DL-46 (1));
- hflmld_native: P-PE without DP (DL-06), one eta = 0.1 for local training and the cloud (DL-43 (11));
- hflmld_cloud: P-TA at its final configuration, HFLMLD eq. (8)-(14) at the cloud on the 7 gateway outputs,
  eta_c = 1.0 (DL-40, DL-43 (11)).
The adaptive gateway: our check DL-46 (1); the two HFLMLD rows DL-45 (1), DL-46 (2). Plus one run without tampering per
detector and seed (DL-45 (2): the DL-10 "harmless miss" reference): 150 + 30 = 180 configs.

Example:
    python make_e5_configs.py --e1 <E1 configs> --dl40 <DL-40 configs> --selection selection_dl42.json \
        --out 05_Experiments/configs
"""
import argparse
import os
import subprocess

from make_dl42_configs import PROJECT, base_config, write
from make_e4_configs import final_settings, load_selection

TYPES = ("scale", "signflip", "inject", "drop", "adaptive")
DETECTORS = ("ours", "hflmld_native", "hflmld_cloud")
SEEDS = range(10)
PARTITION = "partition_E5.yaml"  # PR item 13: alpha 0.1 for gateways 0-2 (DL-29, DL-30)
NATIVE_ETA = 0.1  # DL-43 (11): the HierFAVG lr selected on fold 9 (DL-37)
CLOUD_ETA = 1.0  # DL-40


def detector_settings(detector, e1_dir, dl40_dir, sel, seed):
    """(base config, its file name, sha, settings) of one detector row."""
    if detector == "hflmld_native":
        base, name, sha = base_config(e1_dir, dl40_dir, "ppe", seed)
        s = dict(base["settings"], lr=NATIVE_ETA, cloud_eta=NATIVE_ETA, no_dp=True)
    else:
        base, name, sha = base_config(e1_dir, dl40_dir, "pta", seed)
        s = final_settings(base, sel)  # size-aware check (DL-44)
        if detector == "hflmld_cloud":
            s.pop("gateway_check", None)
            s.update(cloud_rule="hflmld", cloud_eta=CLOUD_ETA)
        else:
            s["rate_alert"] = "size-aware"  # DL-48
    return base, name, sha, dict(s, partition=PARTITION)


def make(e1_dir, dl40_dir, selection_path, out_root, commit, references=True):
    sel, sel_sha = load_selection(selection_path)
    out = os.path.join(out_root, "E5")
    os.makedirs(out, exist_ok=True)
    written = []
    for seed in SEEDS:
        for detector in DETECTORS:
            base, name, sha, s = detector_settings(detector, e1_dir, dl40_dir, sel, seed)
            kinds = list(TYPES)  # DL-45 (1): the adaptive gateway for every detector
            if references:
                kinds.append(None)
            for kind in kinds:
                settings = dict(s, tamper=kind) if kind else dict(s)
                cfg = dict(base, experiment="E5" if kind else "E5 reference (no tampering)",
                           run_id=f"E5_{detector}_{kind or 'none'}_s{seed}", settings=settings,
                           rules=list(dict.fromkeys(base["rules"] + ["DL-06", "DL-14", "DL-23", "DL-41", "DL-42",
                                                                     "DL-43", "DL-44", "DL-45"]
                                                    + (["DL-46"] if kind == "adaptive" else [])
                                                    + (["DL-48"] if detector == "ours" else []))),
                           code_commit=commit, detector=detector,
                           sources_sha256=dict(base["sources_sha256"],
                                               **{name: sha, os.path.basename(selection_path): sel_sha}))
                cfg.pop("supersedes", None)
                written.append(write(out, cfg, "# E5 config, written by make_e5_configs.py. Do not edit by hand.\n"))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--e1", required=True)
    ap.add_argument("--dl40", required=True)
    ap.add_argument("--selection", required=True, help="selection_dl42.json of the DL-42 job")
    ap.add_argument("--out", required=True, help="configs folder; E5/ is made in it")
    ap.add_argument("--no-references", action="store_true", help="tests only: without the DL-45 (2) runs")
    ap.add_argument("--commit", help="code commit to record (default: git HEAD)")
    args = ap.parse_args(argv)
    commit = args.commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True,
                                           text=True, check=True).stdout.strip()
    written = make(args.e1, args.dl40, args.selection, args.out, commit, not args.no_references)
    print(f"{len(written)} E5 configs written (code commit {commit[:7]})", flush=True)


if __name__ == "__main__":
    main()
