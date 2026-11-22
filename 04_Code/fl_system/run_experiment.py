"""Run one experiment config file (plan task M2-05; PR:320 "one config file per run").

It checks the config, turns it into run_pilot.py arguments, runs it in this process, and adds the
config (its file name, SHA-256 and content) to the result JSON. Fold 10 goes to a sealed file
(DL-39).

Example:
    python run_experiment.py --config 05_Experiments/configs/E1/E1_pta_s0.yaml \
        --ptbxl-dir <PTB-XL folder> --out-dir <results folder> [--device cuda] [--gpus-per-node 0.14]
--smoke-rounds N (tests only) shortens the run: N device rounds, or N epochs for centralized/local-only.
"""
import argparse
import hashlib
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

REQUIRED = ("experiment", "run_id", "configuration", "mode", "seed", "settings", "canaries", "base_config",
            "code_commit", "versions")
DP_MODES = ("pta", "ptgc", "ptg", "ppe", "pfr")
EPOCH_MODES = ("local", "centralized")


def load_config(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    cfg = yaml.safe_load(raw)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}")
    s = cfg["settings"]
    if cfg["mode"] in DP_MODES and ("c_ex" not in s or "eps0" not in s):
        raise SystemExit(f"{path}: a DP placement needs c_ex and eps0")
    if cfg["mode"] in EPOCH_MODES and "local_epochs" not in s:
        raise SystemExit(f"{path}: {cfg['mode']} needs local_epochs")
    return cfg, hashlib.sha256(raw).hexdigest()


def build_argv(cfg, config_path, ptbxl_dir, out_dir, device="auto", gpus_per_node=None, noise_csv=None,
               git_commit=None, smoke_rounds=None):
    """The run_pilot.py arguments for one config."""
    s, mode = cfg["settings"], cfg["mode"]
    base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(config_path)), cfg["base_config"]))
    argv = ["--config", base, "--ptbxl-dir", ptbxl_dir, "--mode", mode, "--seed", str(cfg["seed"]),
            "--lr", str(s["lr"]), "--device", device, "--run-id", cfg["run_id"],
            "--out", os.path.join(out_dir, f"{cfg['run_id']}.json"),
            "--git-commit", git_commit or cfg["code_commit"]]
    if cfg.get("fold10") != "none":  # DL-40 grid runs never score fold 10
        argv += ["--fold10-out", os.path.join(out_dir, f"{cfg['run_id']}.fold10.sealed.json")]
    if "eps0" in s:
        argv += ["--eps0", str(s["eps0"])]
    if "cloud_eta" in s:
        argv += ["--cloud-eta", str(s["cloud_eta"])]
    for key, flag in (("c_ex", "--c-ex"), ("clip_C", "--clip-C"), ("tau", "--tau")):
        if key in s:
            argv += [flag, str(s[key])]
    if s.get("quantize"):
        argv.append("--quantize")
    if s.get("save_norms"):  # DL-42 grid: every norm, for the pooled C and tau
        argv.append("--save-norms")
    if s.get("partition"):  # E5: partition_E5.yaml, next to the base config (DL-43 (8))
        argv += ["--partition-config", os.path.join(os.path.dirname(base), s["partition"])]
    for key, flag in (("tamper", "--tamper"), ("cloud_rule", "--cloud-rule")):  # E5 (DL-43)
        if s.get(key):
            argv += [flag, s[key]]
    if s.get("no_dp"):  # E5 HFLMLD native row (DL-06)
        argv.append("--no-dp")
    if s.get("gateway_check"):  # DL-44: the P-TA cloud check
        argv += ["--gateway-check", s["gateway_check"]]
    if s.get("rate_alert"):  # DL-48: the P-TA rate alert
        argv += ["--rate-alert", s["rate_alert"]]
    if "k" in s:  # E6 k line (DL-43)
        argv += ["--k", str(s["k"])]
    att = s.get("attack")
    if att:  # E4 (DL-43): kind, f, norm, poison share; DL-47: boost, poison mode, host adjustment
        argv += ["--attack", att["kind"], "--attack-norm", str(att["norm"])]
        if att.get("f") is not None:
            argv += ["--attack-f", str(att["f"])]
        if att.get("poison_share") is not None:
            argv += ["--poison-share", str(att["poison_share"])]
        if att.get("boost") is not None:  # DL-47
            argv += ["--attack-boost", str(att["boost"])]
        if att.get("poison_mode") is not None:
            argv += ["--poison-mode", att["poison_mode"]]
        if att.get("host_adjust"):
            argv.append("--attack-host-adjust")
    if mode in EPOCH_MODES:
        argv += ["--local-epochs", str(smoke_rounds or s["local_epochs"])]
    elif smoke_rounds or "rounds" in s:
        argv += ["--rounds", str(smoke_rounds or s["rounds"])]
    if not cfg["canaries"]:
        argv.append("--no-canaries")
    if gpus_per_node is not None:
        argv += ["--gpus-per-node", str(gpus_per_node)]
    if noise_csv:
        argv += ["--noise-csv", noise_csv]
    return argv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ptbxl-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gpus-per-node", type=float)
    ap.add_argument("--noise-csv", help="override the base config's noise CSV path (job packages)")
    ap.add_argument("--git-commit", help="the commit of the code that runs (default: the config's code_commit)")
    ap.add_argument("--smoke-rounds", type=int, help="tests only: shorten the run")
    args = ap.parse_args(argv)
    cfg, cfg_sha = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    import run_pilot
    run_pilot.main(build_argv(cfg, args.config, args.ptbxl_dir, args.out_dir, args.device, args.gpus_per_node,
                              args.noise_csv, args.git_commit, args.smoke_rounds))
    out = os.path.join(args.out_dir, f"{cfg['run_id']}.json")
    with open(out, encoding="utf-8") as fh:
        result = json.load(fh)
    result["experiment_config"] = {"file": os.path.basename(args.config), "sha256": cfg_sha, "content": cfg,
                                   "smoke_rounds": args.smoke_rounds}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"config {os.path.basename(args.config)} (sha256 {cfg_sha[:12]}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
