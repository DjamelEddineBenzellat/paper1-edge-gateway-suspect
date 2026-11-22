"""Write the E4 configs (plan task X-6; DL-41, DL-42 (5), DL-43, DL-44).

Each DP design runs at its final configuration: the DL-42 selected point if its point changed, otherwise its E1
settings (P-PE: DL-40). Every malicious update is scaled to the same norm, P-TA's final C (DL-43 (3)).
- E4/: 3 attacks (uniform backdoor = default attack, concentrated backdoor, label flipping) x designs x seeds 0-9,
  sealed fold 10 (DL-39, DL-43 (1)).
- E4_positive_control/: both attacks (backdoor, label flipping) on undefended non-DP FedAvg, seed 0, fold 9 only
  (DL-43 (7)); e4_positive_control.py applies the pass rule.
- E4_reference/: P-TA at f = 0 with the size-aware check, seeds 0-9, sealed fold 10: the reference for the E4
  macro-F1 drop and the E6 gaps at k = 3 and eps = 8 (DL-44 (2)).
Every P-TA run uses the size-aware gateway check (DL-44 (1)); P-TG-c takes the attack by DL-44 (3).
- E4_DL49/ and E4_DL49_pc/ (--dl49, DL-49): uniform and concentrated label flipping x the 5 designs x seeds 0-9
  (100 runs, sealed fold 10; P-TA also with the DL-48 rate alert), and their fail-fast positive control on FedAvg.
- E4_gate/ (--gate, DL-47 (4)): the positive control of DL-43 (7) for the attack strengths of DL-47: the backdoor
  with recipe "replace" or "copies" x boost 1, 3, 10, 30, and label flipping with boost 1, 3, 10, 30 or scaled to C
  (DL-43); host-adjusted f (DL-47 (3)) in every run. e4_gate.py applies the selection rule.

Example:
    python make_e4_configs.py --e1 05_Experiments/configs/E1 --dl40 <DL-40 configs> \
        --selection <DL-42 job>/selection_dl42.json --out 05_Experiments/configs
"""
import argparse
import hashlib
import json
import os
import subprocess

import yaml

from make_dl42_configs import NORM_MODES, PROJECT, base_config, write

E4_MODES = ("ptg", "ptgc", "ppe", "pfr", "pta")  # the 5 DP designs (DL-41)
ATTACKS = ("backdoor", "concentrated", "labelflip")
POSITIVE_CONTROL = ("backdoor", "labelflip")
SEEDS = range(10)
F_UNIFORM = 0.1  # PR item 25, DL-04
POISON_SHARE = 0.5  # DL-43 (3)
GATE_BOOSTS = (1.0, 3.0, 10.0, 30.0)  # DL-47 (4)
DL49_ATTACKS = ("labelflip", "concentrated_labelflip")  # DL-49 (1): scaled to C, as the gate chose
GATE_RECIPES = ("replace", "copies")  # DL-47 (2)


def load_selection(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    sel = json.loads(raw)
    if sel.get("smoke") or sel.get("partial"):
        raise SystemExit(f"{path}: a smoke or partial DL-42 selection cannot set final configurations")
    return sel, hashlib.sha256(raw).hexdigest()


def final_settings(base, sel):
    """The design's final settings (DL-42 (5)): the selected point if it changed, else the E1 settings; P-TA with
    the size-aware gateway check (DL-44 (1))."""
    mode, s = base["mode"], dict(base["settings"])
    if mode == "pta":
        s["gateway_check"] = "size-aware"
    if mode in sel["changed"]:
        point = sel["designs"][mode]["selected"]
        s.update(lr=point["lr"], c_ex=point["c_ex"])
        if mode in NORM_MODES:
            s.update(clip_C=point["clip_C"], quantize=True)
            if mode == "pta":
                s["tau"] = point["tau"]
    return s


def attack_norm(e1_dir, dl40_dir, sel):
    """DL-43 (3): P-TA's final C, the same absolute norm in every design."""
    base, _, _ = base_config(e1_dir, dl40_dir, "pta", 0)
    return float(final_settings(base, sel)["clip_C"])


def attack_block(kind, norm):
    return {"kind": kind, "f": None if kind.startswith("concentrated") else F_UNIFORM, "norm": norm,
            "poison_share": POISON_SHARE}


def gate_runs():
    """(run_id, attack block without the norm) of the 13 gate runs (DL-47 (4)); boost None = scaled to C (DL-43)."""
    out = []
    for recipe in GATE_RECIPES:
        for b in GATE_BOOSTS:
            out.append((f"E4gate_backdoor_{recipe}_b{b:g}_s0", {"kind": "backdoor", "boost": b, "poison_mode": recipe}))
    for b in GATE_BOOSTS + (None,):
        out.append((f"E4gate_labelflip_{'dl43' if b is None else f'b{b:g}'}_s0", {"kind": "labelflip", "boost": b}))
    return out


def make_gate(e1_dir, dl40_dir, selection_path, out_root, commit):
    """The gate configs (DL-47 (4)): undefended non-DP FedAvg, seed 0, 100 rounds, fold 9 only, as DL-43 (7)."""
    sel, sel_sha = load_selection(selection_path)
    norm = attack_norm(e1_dir, dl40_dir, sel)
    gate_dir = os.path.join(out_root, "E4_gate")
    os.makedirs(gate_dir, exist_ok=True)
    with open(os.path.join(e1_dir, "E1_fedavg_s0.yaml"), "rb") as fh:
        raw = fh.read()
    fedavg = yaml.safe_load(raw)
    written = []
    for run_id, extra in gate_runs():
        attack = dict(attack_block(extra["kind"], norm), host_adjust=True, boost=extra["boost"])
        if extra["kind"] == "backdoor":
            attack["poison_mode"] = extra["poison_mode"]
        cfg = dict(fedavg, experiment="E4 gate (DL-47 (4))", run_id=run_id, fold10="none",
                   settings=dict(fedavg["settings"], attack=attack),
                   rules=list(dict.fromkeys(fedavg["rules"] + ["DL-43", "DL-47"])), code_commit=commit,
                   sources_sha256=dict(fedavg["sources_sha256"], **{"E1_fedavg_s0.yaml":
                                                                    hashlib.sha256(raw).hexdigest(),
                                                                    os.path.basename(selection_path): sel_sha}))
        written.append(write(gate_dir, cfg, "# E4 gate config (DL-47 (4)), written by make_e4_configs.py --gate. "
                                            "Do not edit by hand.\n"))
    return written, norm


def dl49_attack(kind, norm):
    """DL-49 (1): label flipping scaled to C; host-adjusted f for the uniform attack (DL-47 (3))."""
    block = attack_block(kind, norm)
    if kind == "labelflip":
        block["host_adjust"] = True
    return block


def make_dl49(e1_dir, dl40_dir, selection_path, out_root, commit, modes=E4_MODES):
    """The DL-49 E4 configs (E4_DL49/) and their fail-fast positive control (E4_DL49_pc/)."""
    sel, sel_sha = load_selection(selection_path)
    norm = attack_norm(e1_dir, dl40_dir, sel)
    sel_name = os.path.basename(selection_path)
    run_dir, pc_dir = os.path.join(out_root, "E4_DL49"), os.path.join(out_root, "E4_DL49_pc")
    for d in (run_dir, pc_dir):
        os.makedirs(d, exist_ok=True)
    runs, controls = [], []
    for seed in SEEDS:
        for mode in modes:
            base, name, sha = base_config(e1_dir, dl40_dir, mode, seed)
            settings = final_settings(base, sel)
            if mode == "pta":
                settings["rate_alert"] = "size-aware"  # DL-48
            for kind in DL49_ATTACKS:
                cfg = dict(base, experiment="E4 (DL-49)", run_id=f"E4_{mode}_{kind}_s{seed}",
                           settings=dict(settings, attack=dl49_attack(kind, norm)),
                           rules=list(dict.fromkeys(base["rules"] + ["DL-41", "DL-42", "DL-43", "DL-44", "DL-47",
                                                                     "DL-48", "DL-49"])), code_commit=commit,
                           final_configuration="DL-42 selected point" if mode in sel["changed"] else "E1 settings",
                           sources_sha256=dict(base["sources_sha256"], **{name: sha, sel_name: sel_sha}))
                cfg.pop("supersedes", None)
                runs.append(write(run_dir, cfg, "# E4 config (DL-49), written by make_e4_configs.py --dl49. "
                                                "Do not edit by hand.\n"))
    with open(os.path.join(e1_dir, "E1_fedavg_s0.yaml"), "rb") as fh:
        raw = fh.read()
    fedavg = yaml.safe_load(raw)
    for kind in DL49_ATTACKS:
        cfg = dict(fedavg, experiment="E4 positive control (DL-49 (2))", run_id=f"E4pc49_fedavg_{kind}_s0",
                   fold10="none", settings=dict(fedavg["settings"], attack=dl49_attack(kind, norm)),
                   rules=list(dict.fromkeys(fedavg["rules"] + ["DL-43", "DL-47", "DL-49"])), code_commit=commit,
                   sources_sha256=dict(fedavg["sources_sha256"], **{"E1_fedavg_s0.yaml":
                                                                    hashlib.sha256(raw).hexdigest(),
                                                                    sel_name: sel_sha}))
        controls.append(write(pc_dir, cfg, "# E4 positive-control config (DL-49 (2)), written by make_e4_configs.py "
                                           "--dl49. Do not edit by hand.\n"))
    return runs, controls, norm


def make(e1_dir, dl40_dir, selection_path, out_root, commit, modes=E4_MODES):
    sel, sel_sha = load_selection(selection_path)
    norm = attack_norm(e1_dir, dl40_dir, sel)
    sel_name = os.path.basename(selection_path)
    e4_dir, pc_dir = os.path.join(out_root, "E4"), os.path.join(out_root, "E4_positive_control")
    ref_dir = os.path.join(out_root, "E4_reference")
    for d in (e4_dir, pc_dir, ref_dir):
        os.makedirs(d, exist_ok=True)
    runs, controls, refs = [], [], []
    for seed in SEEDS:
        for mode in modes:
            base, name, sha = base_config(e1_dir, dl40_dir, mode, seed)
            settings = final_settings(base, sel)
            for kind in ATTACKS:
                cfg = dict(base, experiment="E4", run_id=f"E4_{mode}_{kind}_s{seed}",
                           settings=dict(settings, attack=attack_block(kind, norm)),
                           rules=list(dict.fromkeys(base["rules"] + ["DL-41", "DL-42", "DL-43"])), code_commit=commit,
                           final_configuration="DL-42 selected point" if mode in sel["changed"] else "E1 settings",
                           sources_sha256=dict(base["sources_sha256"], **{name: sha, sel_name: sel_sha}))
                cfg.pop("supersedes", None)
                runs.append(write(e4_dir, cfg, "# E4 config, written by make_e4_configs.py. Do not edit by hand.\n"))
            if mode == "pta":  # DL-44 (2): the f = 0 reference with the size-aware check
                cfg = dict(base, experiment="E4 reference (DL-44)", run_id=f"E4ref_pta_s{seed}", settings=settings,
                           rules=list(dict.fromkeys(base["rules"] + ["DL-42", "DL-44"])), code_commit=commit,
                           sources_sha256=dict(base["sources_sha256"], **{name: sha, sel_name: sel_sha}))
                cfg.pop("supersedes", None)
                refs.append(write(ref_dir, cfg, "# P-TA f = 0 reference with the size-aware check (DL-44 (2)), "
                                               "written by make_e4_configs.py. Do not edit by hand.\n"))
    with open(os.path.join(e1_dir, "E1_fedavg_s0.yaml"), "rb") as fh:
        raw = fh.read()
    fedavg = yaml.safe_load(raw)
    for kind in POSITIVE_CONTROL:
        cfg = dict(fedavg, experiment="E4 positive control", run_id=f"E4pc_fedavg_{kind}_s0", fold10="none",
                   settings=dict(fedavg["settings"], attack=attack_block(kind, norm)),
                   rules=list(dict.fromkeys(fedavg["rules"] + ["DL-43"])), code_commit=commit,
                   sources_sha256=dict(fedavg["sources_sha256"], **{"E1_fedavg_s0.yaml":
                                                                    hashlib.sha256(raw).hexdigest(),
                                                                    sel_name: sel_sha}))
        controls.append(write(pc_dir, cfg, "# E4 positive-control config (DL-43 (7)), written by make_e4_configs.py. "
                                           "Do not edit by hand.\n"))
    return runs, controls, norm, refs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--e1", required=True, help="folder with the E1 configs")
    ap.add_argument("--dl40", required=True, help="folder with the DL-40 P-PE configs")
    ap.add_argument("--selection", required=True, help="selection_dl42.json of the DL-42 job")
    ap.add_argument("--out", required=True, help="configs folder; E4/ and E4_positive_control/ are made in it")
    ap.add_argument("--commit", help="code commit to record (default: git HEAD)")
    ap.add_argument("--gate", action="store_true", help="write only the 13 gate configs in E4_gate/ (DL-47 (4))")
    ap.add_argument("--dl49", action="store_true", help="write only the DL-49 configs in E4_DL49/ and E4_DL49_pc/")
    args = ap.parse_args(argv)
    commit = args.commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True,
                                           text=True, check=True).stdout.strip()
    if args.dl49:
        runs, controls, norm = make_dl49(args.e1, args.dl40, args.selection, args.out, commit)
        print(f"{len(runs)} DL-49 E4 configs + {len(controls)} positive-control configs; attack norm {norm:.4f} "
              f"(code commit {commit[:7]})", flush=True)
        return
    if args.gate:
        written, norm = make_gate(args.e1, args.dl40, args.selection, args.out, commit)
        print(f"{len(written)} gate configs; attack norm cap {norm:.4f} (code commit {commit[:7]})", flush=True)
        return
    runs, controls, norm, refs = make(args.e1, args.dl40, args.selection, args.out, commit)
    print(f"{len(runs)} E4 configs + {len(controls)} positive-control configs + {len(refs)} P-TA references; "
          f"attack norm {norm:.4f} "
          f"(code commit {commit[:7]})", flush=True)


if __name__ == "__main__":
    main()
