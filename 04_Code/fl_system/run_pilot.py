"""One G0-a run (plan tasks G0-09 / G0-14): P-TA, P-TG-c or local-only on PTB-XL.

Writes one result JSON: fold-9 scores (DL-27, DL-36; DL-33 kept), learning curve, norms
for C and tau (DL-31), gateway-check counts (DL-10), privacy values, timings,
versions and the device it ran on (rule 5).

Examples:
    python 04_Code/fl_system/run_pilot.py --config 05_Experiments/configs/G0a.yaml \
        --ptbxl-dir <PTB-XL folder> --mode pta --seed 0 --lr 0.05 --c-ex 1.0 --out run.json
    python 04_Code/fl_system/run_pilot.py ... --mode local --seed 0 --lr 0.05 --out local.json
Grid runs (DL-31) leave out --clip-C/--tau/--quantize; pilot runs give them.
DL-35 diagnostic job only: --no-noise (z = 0, not private), --diagnostics
(per-class AUROC, tuned-threshold F1) and --scores-out (final fold-9 scores).
M2-01 (PR §5, DL-37): also P-TG, P-PE (HFLMLD), P-FR and the non-private baselines
FedAvg, HierFAVG and centralized. P-PE and HierFAVG run T_e = 5 edge iterations per
cloud round, so --rounds counts device rounds (T) and the cloud runs T / T_e rounds.
"""
import argparse
import csv
import datetime as dt
import gzip
import importlib.metadata as md
import json
import os
import platform
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "defences")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "attacks")))  # E4 (DL-43)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.nn.utils import parameters_to_vector, vector_to_parameters  # noqa: E402

from metrics import CLASSES, diagnostic_scores, predict_proba, scores, trivial_macro_f1  # noqa: E402
from model import ECG_PARAMETERS, ecg_cnn, n_parameters  # noqa: E402
from seeding import INIT, seed_int  # noqa: E402
from train_data import run_data  # noqa: E402


DP_MODES = ("pta", "ptgc", "ptg", "ppe", "pfr")  # device DP-SGD (P-TG-c: central DP-SGD at the gateway)
NON_PRIVATE = ("local", "centralized", "fedavg", "hierfavg")  # no DP (DL-32, DL-37)
EDGE_ITERATIONS = {"ppe": 5, "hierfavg": 5}  # T_e (DL-37); 1 for every other mode


def noise_multiplier(csv_path, eps0, q, total_steps):
    with open(csv_path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if abs(float(r["epsilon"]) - eps0) < 1e-9]
    if not rows:
        raise SystemExit(f"no noise multiplier for epsilon {eps0} in {csv_path}")
    r = rows[0]
    if abs(float(r["q"]) - q) > 1e-6 or int(r["total_steps"]) != total_steps:
        raise SystemExit(f"noise CSV was made for q={r['q']}, steps={r['total_steps']}; "
                         f"this partition has q={q}, steps={total_steps}")
    return float(r["noise_multiplier_z"]), float(r["delta"])


def accountant_epsilon(z, q, steps, delta):
    from opacus.accountants import RDPAccountant
    acc = RDPAccountant()
    acc.history = [(z, q, steps)]
    return float(acc.get_epsilon(delta=delta))


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def pct(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return None
    return {"n": int(v.size), "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ptbxl-dir", required=True)
    ap.add_argument("--partition-config", help="override the config's partition_config")
    ap.add_argument("--noise-csv", help="override the config's noise_csv")
    ap.add_argument("--eps0", type=float, help="override the config's eps0 (DL-34: the eps = 8 stage)")
    ap.add_argument("--mode", required=True, choices=list(DP_MODES + NON_PRIVATE))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--c-ex", type=float, help="per-example clip (pta, ptgc)")
    ap.add_argument("--clip-C", type=float, help="update clip C (pta pilot; DL-31)")
    ap.add_argument("--tau", type=float, help="group-sum norm check tau (pta pilot; DL-31)")
    ap.add_argument("--quantize", action="store_true", help="16-bit quantization (pta pilot; needs --clip-C)")
    ap.add_argument("--cloud-eta", type=float,
                    help="DL-40: P-PE cloud step eta_c in HFLMLD eq. (8) and (14) (default: --lr, as in the paper)")
    ap.add_argument("--attack", choices=["backdoor", "concentrated", "labelflip", "concentrated_labelflip"],
                    help="E4 device attack (DL-43); pta, pfr, ptg, ppe, and fedavg for the positive control")
    ap.add_argument("--attack-f", type=float, default=0.1, help="uniform attackers: probability per device (DL-04)")
    ap.add_argument("--attack-norm", type=float, help="norm of every malicious update (DL-43 (3): P-TA's final C)")
    ap.add_argument("--poison-share", type=float, default=0.5, help="backdoor: share of poisoned records (DL-43)")
    ap.add_argument("--attack-boost", type=float,
                    help="multiply each malicious update by this, capped at --attack-norm (DL-47 candidate); "
                         "default: scale to exactly --attack-norm (DL-43 (3))")
    ap.add_argument("--poison-mode", choices=["replace", "copies"], default="replace",
                    help="backdoor records: replaced (DL-43 (3)) or triggered copies added (DL-47 candidate)")
    ap.add_argument("--attack-host-adjust", action="store_true",
                    help="uniform attackers: f over all devices although canary hosts stay honest (DL-47 candidate)")
    ap.add_argument("--tamper", choices=["scale", "signflip", "inject", "drop", "adaptive"],
                    help="E5: one tampering gateway (b = 1, DL-14, DL-43 (8)-(10)); pta or ppe")
    ap.add_argument("--rate-alert", choices=["all", "size-aware"], default="all",
                    help="P-TA rate alert: aborts + all tau drops (DL-10), or aborts + the tau drops that the device "
                         "sizes do not explain (DL-48)")
    ap.add_argument("--cloud-rule", choices=["median", "hflmld"], default="median",
                    help="P-TA cloud: median + gateway check, or HFLMLD eq. (8)-(14) on the gateway outputs "
                         "(E5 detector, DL-43 (11); needs --cloud-eta)")
    ap.add_argument("--no-dp", action="store_true",
                    help="E5 HFLMLD native row (DL-06): P-PE devices train with plain SGD, no DP (ppe only)")
    ap.add_argument("--gateway-check", choices=["dl10", "size-aware"], default="dl10",
                    help="P-TA cloud check: DL-10 as in E1, or the DL-44 size-aware check (from E4 on)")
    ap.add_argument("--k", type=int, choices=[1, 2, 3, 5],
                    help="E6 k line (DL-43): group size for P-TA and P-FR (default: the base config, k = 3)")
    ap.add_argument("--rounds", type=int, help="override T (smoke tests only)")
    ap.add_argument("--local-epochs", type=int, help="override the local-only epochs (smoke tests only)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gpus-per-node", type=float, help="GPU share per gateway node (default 0.14 on CUDA)")
    ap.add_argument("--no-noise", action="store_true",
                    help="DL-35 diagnostic: z = 0, sampling and C_ex clipping kept (NOT private; pta, ptgc)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="DL-35: per-class AUROC and tuned-threshold macro-F1 (never a gate statistic)")
    ap.add_argument("--scores-out", help="DL-35: write the final fold-9 scores here (.csv.gz; pta, ptgc)")
    ap.add_argument("--fold10-out", help="DL-39: score fold 10 at the end with the fold-9 thresholds and write it to "
                                         "this SEALED file (not to be opened before G1 passes)")
    ap.add_argument("--no-canaries", action="store_true",
                    help="tests only: run without the DL-38 canaries (every study run has them)")
    ap.add_argument("--save-norms", action="store_true",
                    help="DL-42: also write every device-update and group-sum norm (pooled C and tau; pta, pfr)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--git-commit", help="commit of the code, if not a git checkout")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.quantize and args.clip_C is None:
        ap.error("--quantize needs --clip-C")
    if args.mode in DP_MODES and args.c_ex is None:
        ap.error(f"--c-ex is required for {', '.join(DP_MODES)}")
    if args.mode in NON_PRIVATE and args.no_noise:
        ap.error("--no-noise is for the DP modes only")
    if args.mode in ("local", "centralized") and args.scores_out:
        ap.error("--scores-out is for the Flower modes only")
    if args.mode in ("ptg", "ppe") and (args.clip_C or args.tau or args.quantize):
        ap.error("P-TG and P-PE see plaintext updates: no --clip-C, --tau or --quantize (DL-37)")
    if args.mode == "pfr" and (args.tau is not None):
        ap.error("P-FR has no norm check tau (DL-37)")
    hflmld_cloud = args.mode == "pta" and args.cloud_rule == "hflmld"
    if args.cloud_eta is not None and (args.cloud_eta <= 0 or not (args.mode == "ppe" or hflmld_cloud)):
        ap.error("--cloud-eta is a positive step for P-PE (DL-40) or the HFLMLD cloud rule on P-TA (DL-43)")
    if args.cloud_rule == "hflmld" and (args.mode != "pta" or args.cloud_eta is None):
        ap.error("--cloud-rule hflmld is the E5 detector on P-TA outputs and needs --cloud-eta (DL-43 (11))")
    if args.no_dp and args.mode != "ppe":
        ap.error("--no-dp is the HFLMLD native row: ppe only (DL-06)")
    if args.tamper and args.mode not in ("pta", "ppe"):
        ap.error("--tamper: pta (our check or the HFLMLD cloud rule) or ppe (HFLMLD native)")
    if args.save_norms and args.mode not in ("pta", "pfr"):
        ap.error("--save-norms is for P-TA and P-FR, whose C (and tau) come from the norms (DL-42)")
    if args.attack and args.mode not in ("pta", "pfr", "ptg", "ptgc", "ppe", "fedavg"):
        ap.error("--attack: the five DP designs, or fedavg (the positive control)")
    if args.gateway_check == "size-aware" and args.mode != "pta":
        ap.error("--gateway-check size-aware is the P-TA cloud check (DL-44)")
    if args.rate_alert == "size-aware" and (args.mode != "pta" or args.clip_C is None or args.tau is None):
        ap.error("--rate-alert size-aware is the P-TA rate alert (DL-48); it needs --clip-C and --tau")
    if args.k is not None and args.mode not in ("pta", "pfr"):
        ap.error("--k is for P-TA and P-FR (the designs with groups)")
    if args.attack and args.attack_boost is not None and not args.attack_boost > 0:
        ap.error("--attack-boost must be positive")
    if args.attack and not (args.attack_norm or 0) > 0:
        ap.error("--attack needs a positive --attack-norm (DL-43 (3))")
    cloud_eta = (args.cloud_eta if args.cloud_eta is not None else args.lr) if args.mode == "ppe" else \
        (args.cloud_eta if hflmld_cloud else None)

    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    with open(args.config, encoding="utf-8") as fh:
        conf = yaml.safe_load(fh)
    part_cfg = os.path.abspath(args.partition_config or os.path.join(cfg_dir, conf["partition_config"]))
    k = args.k if args.k is not None else int(conf["k"])  # E6 k line (DL-43); 30 devices per gateway / k
    noise_csv = os.path.abspath(args.noise_csv or os.path.join(cfg_dir, conf["noise_csv"]))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else \
        ("cpu" if args.device == "auto" else args.device)
    started = dt.datetime.now().isoformat(timespec="seconds")
    t_start = time.perf_counter()

    data = run_data(args.ptbxl_dir, part_cfg, args.seed, need_val=True, canaries=not args.no_canaries,
                    need_test=bool(args.fold10_out))
    attack = None
    if args.attack:  # E4 (DL-43 (2)-(5)): who is malicious, from the attack stream of the run seed
        from device_attacks import draw_malicious
        malicious, attack_gw = draw_malicious(data, args.seed, args.attack, args.attack_f, args.attack_host_adjust)
        attack = {"kind": args.attack, "f": None if args.attack.startswith("concentrated") else args.attack_f,
                  "norm": args.attack_norm, "poison_share": args.poison_share, "batch": conf["batch"],
                  "malicious": malicious, "gateway": attack_gw, "boost": args.attack_boost,
                  "poison_mode": args.poison_mode, "host_adjust": args.attack_host_adjust}
    tamper = None
    if args.tamper:  # E5 (DL-43 (8)): one tampering gateway, from the run seed, the same in every round
        from tampering import tamper_gateway
        tamper = {"kind": args.tamper, "gateway": tamper_gateway(args.seed, data.part_cfg["n_gateways"]),
                  "detector": "HFLMLD native (no DP)" if args.no_dp else
                  ("HFLMLD cloud rule on P-TA" if hflmld_cloud else
                   ("ours (size-aware, DL-44)" if args.gateway_check == "size-aware" else "ours (DL-10)"))}
    sealed = {}  # DL-39: fold-10 scores, written only to the sealed file
    final_vec = None  # the final global model, for the canary and backdoor measures (M2-03)
    plan = data.plan
    rounds = args.rounds or plan["rounds_T"]
    eps0 = args.eps0 if args.eps0 is not None else float(conf["eps0"])
    z, delta = noise_multiplier(noise_csv, eps0, plan["q"], plan["total_steps"])
    private = args.mode in DP_MODES and not args.no_noise and not args.no_dp
    z_used = z if private else 0.0
    noise_label = ("on" if private else "OFF - DL-35 diagnostic, not private" if args.no_noise
                   else "none - HFLMLD native row without DP (DL-06)" if args.no_dp
                   else "none - non-private baseline (DL-32, DL-37)")
    edge_iterations = EDGE_ITERATIONS.get(args.mode, 1)
    if rounds % edge_iterations:
        raise SystemExit(f"--rounds {rounds} is not a multiple of T_e = {edge_iterations} for {args.mode}")
    torch.manual_seed(seed_int(args.seed, INIT))
    init_model = ecg_cnn()
    assert n_parameters(init_model) == ECG_PARAMETERS
    init_vec = parameters_to_vector(init_model.parameters()).detach().numpy().astype(np.float32)

    result = {
        "run_id": args.run_id, "mode": args.mode, "seed": args.seed, "started": started,
        "settings": {"lr": args.lr, "c_ex": args.c_ex, "clip_C": args.clip_C, "tau": args.tau,
                     "quantize": args.quantize, "k": k, "rounds": rounds, "rounds_planned": plan["rounds_T"],
                     "edge_iterations": edge_iterations, "cloud_rounds": rounds // edge_iterations,
                     "eval_every": conf["eval_every"], "cloud_eta": cloud_eta, "attack": args.attack,
                     "gateway_check": args.gateway_check if args.mode == "pta" else None},
        "privacy": {"eps0": eps0, "delta": delta, "z": z, "q": plan["q"],
                    "steps_per_round": plan["local_steps_per_round_S"], "total_steps": plan["total_steps"],
                    "epsilon_accountant": accountant_epsilon(z, plan["q"], plan["total_steps"], delta)
                    if private else None, "z_used": z_used, "noise": noise_label},
        "data": {"n_train": len(data.train), "n_val_fold9": len(data.val), "partition_sha256": data.partition_sha256,
                 "trivial_macro_f1_fold9": trivial_macro_f1(data.y_val)},
        "machine": {"host": socket.gethostname(), "platform": platform.platform(), "device": device,
                    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None},
        "versions": {p: md.version(p) for p in ("torch", "opacus", "flwr", "ray", "numpy")},
        "git_commit": args.git_commit or git_commit(),
        "config": {"file": os.path.abspath(args.config), "partition_config": part_cfg, "noise_csv": noise_csv},
        "metric": "DL-36: *_tuned keys decide (thresholds tuned on fold 9); macro_f1 = DL-33 threshold 0.5",
    }
    if attack:
        result["attack"] = dict(attack, n_malicious=len(attack["malicious"]))

    eval_model = ecg_cnn().to(device)

    def evaluate_vec(vec):
        vector_to_parameters(torch.as_tensor(vec, device=device), eval_model.parameters())
        return scores(data.y_val, predict_proba(eval_model, data.x_val, device))

    if args.mode == "local":
        from placements import trainer_for
        epochs = args.local_epochs or conf["local_epochs"]
        checkpoints = sorted({c for c in conf["local_checkpoints"] if c <= epochs} | {epochs})
        trainer = trainer_for(device, ecg_cnn)
        per_device = {c: [] for c in checkpoints}
        t0 = time.perf_counter()

        def local_eval(m, rows, calls=None):
            p_val = predict_proba(m, data.x_val, device)
            sc = scores(data.y_val, p_val)
            if args.fold10_out and calls is not None:
                calls[0] += 1
                if calls[0] == len(checkpoints):  # the final epoch (the DL-36/DL-37 choice is epoch 100)
                    thr = [sc["thresholds"][c] for c in CLASSES]
                    t10 = scores(data.y_test, predict_proba(m, data.x_test, device), thresholds=thr)
                    sealed.setdefault("devices", []).append(
                        {"macro_f1_tuned": t10["macro_f1_tuned"], "macro_auroc": t10["macro_auroc"],
                         "mi_sens_at_90spec": t10["mi_sens_at_90spec"]})
            if args.diagnostics:  # DL-35: thresholds tuned on this device's own training records
                sc["diag"] = diagnostic_scores(data.y_val, p_val, data.y_train[rows],
                                               predict_proba(m, data.x_train[rows], device))
            return sc

        for d in sorted(data.device_rows):
            rows = data.device_rows[d]
            out = trainer.plain_sgd_epochs(
                init_vec, data.x_train, data.y_train, rows, epochs, args.lr, conf["batch"],
                keys=(args.seed, d), checkpoints=set(checkpoints),
                evaluate=lambda m, rows=rows, calls=[0]: local_eval(m, rows, calls))
            for c in checkpoints:
                per_device[c].append(out[c])
        result["local_only"] = {
            str(c): {"mean_macro_f1_tuned": float(np.mean([s["macro_f1_tuned"] for s in per_device[c]])),
                     "mean_macro_f1": float(np.mean([s["macro_f1"] for s in per_device[c]])),
                     "mean_macro_auroc": float(np.mean([s["macro_auroc"] for s in per_device[c]
                                                        if s["macro_auroc"] is not None])),
                     "device_macro_f1_tuned": [round(s["macro_f1_tuned"], 5) for s in per_device[c]],
                     "device_macro_f1": [round(s["macro_f1"], 5) for s in per_device[c]]} for c in checkpoints}
        if args.diagnostics:
            for c in checkpoints:
                diags = [s["diag"] for s in per_device[c]]
                result["local_only"][str(c)]["diag"] = {
                    "mean_macro_f1_train_thresholds": float(np.mean([g["macro_f1_train_thresholds"] for g in diags])),
                    "mean_macro_f1_fold9_thresholds": float(np.mean([g["macro_f1_fold9_thresholds"] for g in diags])),
                    "mean_per_class_auroc": {k: float(np.mean([g["per_class_auroc"][k] for g in diags
                                                               if g["per_class_auroc"][k] is not None]))
                                             for k in CLASSES}}
        result["timing"] = {"train_s": time.perf_counter() - t0}
    elif args.mode == "centralized":  # DL-37: one non-private model on all training records
        from placements import trainer_for
        epochs = args.local_epochs or conf["local_epochs"]
        checkpoints = sorted({c for c in conf["local_checkpoints"] if c <= epochs} | {epochs})
        trainer = trainer_for(device, ecg_cnn)
        t0 = time.perf_counter()
        last = {}

        def central_eval(m):
            last["vec"] = parameters_to_vector(m.parameters()).detach().cpu().numpy()  # the last call is the final epoch
            return scores(data.y_val, predict_proba(m, data.x_val, device))

        train_rows = np.sort(np.concatenate(list(data.device_rows.values())))  # every device record, canaries included
        out = trainer.plain_sgd_epochs(
            init_vec, data.x_train, data.y_train, train_rows, epochs, args.lr, conf["batch"],
            keys=(args.seed, 0), checkpoints=set(checkpoints), evaluate=central_eval)
        result["centralized"] = {str(c): out[c] for c in checkpoints}
        result["final_fold9"] = out[epochs]
        final_vec = last["vec"]
        result["timing"] = {"train_s": time.perf_counter() - t0}
    else:
        import flower_sim
        n_gw = data.part_cfg["n_gateways"]
        fed_cfg = {"mode": args.mode, "seed": args.seed, "lr": args.lr, "c_ex": args.c_ex, "clip_C": args.clip_C,
                   "tau": args.tau, "quantize": args.quantize, "k": k, "q": plan["q"],
                   "steps": plan["local_steps_per_round_S"], "z": z_used, "rounds": rounds // edge_iterations,
                   "edge_iterations": edge_iterations, "batch": conf["batch"], "device": device,
                   "ptbxl_dir": os.path.abspath(args.ptbxl_dir), "partition_config": part_cfg,
                   "threads_per_node": max(1, (os.cpu_count() or 1) // n_gw), "canaries": not args.no_canaries,
                   "attack": attack, "tamper": tamper, "no_dp": args.no_dp}
        gpus = (args.gpus_per_node if args.gpus_per_node is not None else 0.14) if device == "cuda" else 0.0
        log = {"curve": [], "rounds": [], "device_norms": [], "group_norms": [], "flags": 0, "gw_rounds": 0,
               "drops": 0, "groups_seen": 0, "aborted_groups": 0, "canary_gw": {}, "poisoned_sums": 0,
               "poisoned_dropped": 0, "flags_dl10": 0}
        sizes = {int(d): len(rows) for d, rows in data.device_rows.items()}  # DL-44: the registered record counts
        drop_model = None
        if args.rate_alert == "size-aware":  # DL-48: the cloud's prediction of the tau drops from the device sizes
            from placements import noise_norm_constant
            drop_model = {"sizes": sizes, "clip_C": args.clip_C, "tau": args.tau,
                          "k_noise": noise_norm_constant(ECG_PARAMETERS, args.lr, z_used, args.c_ex,
                                                         plan["local_steps_per_round_S"], plan["q"])}
        reputation = {g: 0.0 for g in range(n_gw)}

        def on_round(rnd, outputs, stats, client_s, sign_s=0.0):
            t0 = time.perf_counter()
            from placements import cloud_pfr, cloud_ppe, cloud_pta, cloud_ptgc, cloud_weighted
            from tampering import OUTPUT_KINDS, tamper_output
            tamper_s = None
            scales = None
            if args.mode == "pta" and not hflmld_cloud and args.gateway_check == "size-aware":  # DL-44
                from placements import size_aware_scales  # from the used groups only, so before any tampering
                scales = size_aware_scales({g: s["used_groups"] for g, s in stats.items()}, sizes, args.seed, rnd)
            if tamper and tamper["kind"] in OUTPUT_KINDS:  # E5: the gateway changes its output before the cloud
                g = tamper["gateway"]
                outputs = dict(outputs)
                if tamper["kind"] == "adaptive" and rnd == 1:  # honest in round 1, not scored (DL-43 (10))
                    tamper_s = 0.0
                elif tamper["kind"] == "adaptive" and (hflmld_cloud or args.mode == "ppe"):  # DL-45 (1), DL-46 (2)
                    from tampering import adaptive_hflmld
                    others = max(float(np.linalg.norm(v)) for gw, v in outputs.items() if gw != g)
                    outputs[g], tamper_s = adaptive_hflmld(outputs[g], others)
                elif tamper["kind"] == "adaptive":  # DL-46 (1): against our check of the current round
                    from placements import pta_check
                    from tampering import adaptive_search

                    def check(out):
                        flagged, dist, thr = pta_check({**outputs, g: out}, stats, scales, drop_model)
                        return g in flagged, dist[g], thr
                    outputs[g], tamper_s = adaptive_search(outputs[g], check)
                else:
                    outputs[g], tamper_s = tamper_output(tamper["kind"], outputs[g], args.seed, rnd, g)
            if args.mode in ("pta", "pfr"):
                if hflmld_cloud:  # E5 detector (DL-43 (11)): HFLMLD eq. (8)-(14) on the P-TA gateway outputs
                    agg, info = cloud_ppe(outputs, log.get("global", init_vec), cloud_eta)
                elif args.mode == "pta":
                    agg, info = cloud_pta(outputs, stats, scales, drop_model)
                else:
                    agg, info = cloud_pfr(outputs, k, stats)
                for s in stats.values():
                    log["device_norms"] += s["device_norms"]
                    log["group_norms"] += s["group_norms"]
                    log["drops"] += len(s["dropped"])
                    log["groups_seen"] += s["full"]
                    log["aborted_groups"] += s.get("aborted_groups", 0)
                    log["poisoned_sums"] += s.get("poisoned_sums", 0)
                    log["poisoned_dropped"] += s.get("poisoned_dropped", 0)
            elif args.mode == "ppe":
                agg, info = cloud_ppe(outputs, log.get("global", init_vec), cloud_eta)
            elif args.mode in ("fedavg", "hierfavg"):
                agg, info = cloud_weighted(outputs, stats)
            else:  # ptgc (DL-03), ptg: median over the gateways
                agg, info = cloud_ptgc(outputs)
            for g in reputation:
                reputation[g] = 0.9 * reputation[g] + 0.1 * (g in info["flagged"])
            log["flags"] += len(info["flagged"])
            log["flags_dl10"] += len(info.get("flagged_dl10", []))
            log["gw_rounds"] += len(outputs)
            for s in stats.values():  # gateway-view canary scores, summed over rounds (DL-38)
                for i, v in s.get("canary_gw", {}).items():
                    log["canary_gw"][int(i)] = log["canary_gw"].get(int(i), 0.0) + v
            new_global = (np.asarray(log.setdefault("global", init_vec)) + agg).astype(np.float32)
            log["global"] = new_global
            cloud_s = time.perf_counter() - t0
            rec = {"round": rnd, "client_phase_s": round(client_s, 3), "cloud_s": round(cloud_s, 3),
                   "gateway_train_s": [round(stats[g]["train_s"], 3) for g in sorted(stats)],
                   "gateway_agg_s": [round(stats[g]["agg_s"], 4) for g in sorted(stats)],
                   "flagged": info["flagged"], "device_round": rnd * edge_iterations,
                   "cloud_sign_s": round(sign_s, 4),  # M2-02: signing the round certificates
                   "cert_check_s": [round(stats[g].get("check_s", 0.0), 4) for g in sorted(stats)],
                   "refused": {str(g): stats[g]["refused"] for g in sorted(stats) if stats[g].get("refused")}}
            if tamper:
                rec["tamper"] = {"gateway": tamper["gateway"], "s": tamper_s}
            if drop_model is not None and "rates" in info:  # DL-48: the rate of the rate alert, per gateway
                rec["rates"] = {str(g): round(v, 4) for g, v in info["rates"].items()}
            if hflmld_cloud:
                rec["cloud_weights"] = info["weights"]
            if "flagged_dl10" in info:  # DL-44: the size-aware check acts; the DL-10 flags are reported
                rec.update(flagged_dl10=info["flagged_dl10"], threshold=info["threshold"],
                           threshold_dl10=info["threshold_dl10"],
                           distances={str(g): round(v, 5) for g, v in info["distances"].items()},
                           distances_dl10={str(g): round(v, 5) for g, v in info["distances_dl10"].items()},
                           scales={str(g): round(v, 5) for g, v in info["scales"].items()})
            if args.mode == "ppe":  # HFLMLD: cloud weights and edge detections per edge iteration
                rec["cloud_weights"] = info["weights"]
                rec["cloud_step"] = info["step"]  # DL-40: how far the cloud moved the global model
                rec["edge_distance_mean"] = float(np.mean(info["distance"]))
                rec["edge_flagged"] = {str(g): [len(e["flagged"]) for e in stats[g]["edge_log"]] for g in sorted(stats)}
            dr = rnd * edge_iterations  # device rounds done
            if dr % conf["eval_every"] == 0 or rnd == fed_cfg["rounds"]:
                te = time.perf_counter()
                sc = evaluate_vec(new_global)
                rec["eval_s"] = round(time.perf_counter() - te, 3)
                log["curve"].append({"round": dr, "macro_f1_tuned": sc["macro_f1_tuned"], "macro_f1": sc["macro_f1"],
                                     "macro_auroc": sc["macro_auroc"]})
                log["final"] = sc
                print(f"[{args.mode} seed {args.seed}] round {dr}: fold-9 macro-F1 (DL-36) {sc['macro_f1_tuned']:.4f}, "
                      f"AUROC {sc['macro_auroc']:.4f}", flush=True)
            log["rounds"].append(rec)
            return new_global

        flower_sim.run(fed_cfg, init_vec, on_round, n_gw, fed_cfg["threads_per_node"], gpus)
        result["final_fold9"] = log["final"]
        final_vec = log["global"]
        if args.diagnostics or args.scores_out:  # DL-35, on the final global model
            vector_to_parameters(torch.as_tensor(log["global"], device=device), eval_model.parameters())
            p_val = predict_proba(eval_model, data.x_val, device)
            if args.diagnostics:  # thresholds tuned on the pooled training records of all devices
                result["diagnostics"] = diagnostic_scores(data.y_val, p_val, data.y_train,
                                                          predict_proba(eval_model, data.x_train, device))
            if args.scores_out:
                os.makedirs(os.path.dirname(os.path.abspath(args.scores_out)), exist_ok=True)
                with gzip.open(args.scores_out, "wt", encoding="utf-8", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["ecg_id"] + [f"y_{c}" for c in CLASSES] + [f"p_{c}" for c in CLASSES])
                    for rec, yv, pv in zip(data.val, np.asarray(data.y_val).astype(int), p_val):
                        w.writerow([rec.ecg_id] + yv.tolist() + [f"{v:.6f}" for v in pv])
                result["scores_file"] = os.path.basename(args.scores_out)
        result["curve"] = log["curve"]
        result["rounds"] = log["rounds"]
        result["norms"] = {"device_update": pct(log["device_norms"]), "group_sum": pct(log["group_norms"])}
        if args.save_norms:  # DL-42: the selection pools the norms of three seeds
            result["norms_raw"] = {"device_update": [round(float(x), 6) for x in log["device_norms"]],
                                   "group_sum": [round(float(x), 6) for x in log["group_norms"]]}
        if tamper:  # DL-23: detection and false alarms per gateway-round (adaptive: round 1 not scored)
            from tampering import score
            result["tampering"] = dict(tamper, **score(log["rounds"], tamper["gateway"], tamper["kind"], n_gw))
            if args.gateway_check == "size-aware" and not hflmld_cloud:  # the DL-10 flags, reported only
                dl10 = [dict(r, flagged=r.get("flagged_dl10", [])) for r in log["rounds"]]
                result["tampering"]["dl10"] = score(dl10, tamper["gateway"], tamper["kind"], n_gw)
        if attack:  # DL-43 §4.1: how many poisoned group sums passed tau (P-TA) or reached the cloud (P-FR)
            result["attack"]["poisoned_sums"] = log["poisoned_sums"]
            result["attack"]["poisoned_dropped_at_tau"] = log["poisoned_dropped"]
        result["gateway_check"] = {"rule": ("hflmld cloud rule (weight 0 = flagged)" if hflmld_cloud else
                                            args.gateway_check) if args.mode == "pta" else None,
                                   "flagged_gateway_rounds_dl10": log["flags_dl10"]
                                   if args.gateway_check == "size-aware" else None,
                                   "flagged_gateway_rounds": log["flags"], "gateway_rounds": log["gw_rounds"],
                                   "flag_rate": log["flags"] / max(1, log["gw_rounds"]),
                                   "dropped_group_sums": log["drops"], "group_sums": log["groups_seen"],
                                   "aborted_groups": log["aborted_groups"],
                                   "reputation_end": reputation}
        rs = log["rounds"]
        result["timing"] = {"client_phase_s_mean": float(np.mean([r["client_phase_s"] for r in rs])),
                            "client_phase_s_p95": float(np.percentile([r["client_phase_s"] for r in rs], 95)),
                            "gateway_train_s_mean": float(np.mean([np.mean(r["gateway_train_s"]) for r in rs])),
                            "cloud_s_mean": float(np.mean([r["cloud_s"] for r in rs])),
                            "gpus_per_node": gpus, "threads_per_node": fed_cfg["threads_per_node"]}
    # ---- M2-03 measures (PR §7, DL-38) ----
    from metrics import apply_trigger, backdoor_success, epsilon_lower_bound, membership_auc, wan_bytes
    if final_vec is not None:  # the clean model's own 'NORM only' rate on triggered fold-9 records (H2 baseline)
        vector_to_parameters(torch.as_tensor(final_vec, device=device), eval_model.parameters())
        p_trig = predict_proba(eval_model, apply_trigger(data.x_val), device)
        thr = [result["final_fold9"]["thresholds"][c] for c in CLASSES]
        result["final_fold9"]["backdoor_success_clean"] = backdoor_success(data.y_val, p_trig, thr)
    if data.canaries is not None:
        from canaries import cloud_view
        can = data.canaries
        member = np.asarray(can["member"], dtype=bool)
        views = {}
        if final_vec is not None:
            views["cloud"] = cloud_view(final_vec, data, device)
        if args.mode in ("pta", "ptg", "ppe"):  # P-FR (masked) and P-TG-c (trusted) have no gateway view
            views["gateway"] = np.array([log["canary_gw"].get(i, 0.0) for i in range(len(member))])
        result["canaries"] = {
            "n": int(len(member)), "members": int(member.sum()), "rows": can["rows"].tolist(),
            "member": member.tolist(), "host": can["host"].tolist(),
            "views": {name: {"auc": membership_auc(v[member], v[~member]),
                             "eps_lb": epsilon_lower_bound(v[member], v[~member], delta=delta),
                             "scores": [round(float(x), 6) for x in v]} for name, v in views.items()}}
    if args.fold10_out:  # DL-39: fold 10, SEALED until G1 passes
        if final_vec is not None:
            vector_to_parameters(torch.as_tensor(final_vec, device=device), eval_model.parameters())
            thr = [result["final_fold9"]["thresholds"][c] for c in CLASSES]
            sealed["final_fold10"] = scores(data.y_test, predict_proba(eval_model, data.x_test, device), thresholds=thr)
            sealed["final_fold10"]["backdoor_success_clean"] = backdoor_success(
                data.y_test, predict_proba(eval_model, apply_trigger(data.x_test), device), thr)
        if "devices" in sealed:
            dv = sealed["devices"]
            sealed["local_only_fold10"] = {
                "epoch": checkpoints[-1], "mean_macro_f1_tuned": float(np.mean([d["macro_f1_tuned"] for d in dv])),
                "mean_macro_auroc": float(np.mean([d["macro_auroc"] for d in dv if d["macro_auroc"] is not None]))}
        os.makedirs(os.path.dirname(os.path.abspath(args.fold10_out)), exist_ok=True)
        with open(args.fold10_out, "w", encoding="utf-8") as fh:
            json.dump({"SEALED": "DL-39: do not open before gate G1 passes (task M3-03)", "run_id": args.run_id,
                       "mode": args.mode, "seed": args.seed, "n_test_fold10": len(data.test),
                       "thresholds": "the model's fold-9 thresholds (DL-36)", **sealed}, fh, indent=1)
        result["fold10_sealed_file"] = os.path.basename(args.fold10_out)
    pc = data.part_cfg
    result["wan_bytes"] = wan_bytes(args.mode, ECG_PARAMETERS, pc["n_gateways"], pc["devices_per_gateway"],
                                    edge_iterations)
    try:
        import resource  # not on Windows
        result["timing"]["peak_rss_mb_main"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        pass
    result["timing"]["wall_s"] = time.perf_counter() - t_start
    result["ended"] = dt.datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"written {args.out} (wall {result['timing']['wall_s']:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
