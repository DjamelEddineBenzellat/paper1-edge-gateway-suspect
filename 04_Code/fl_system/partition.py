"""Two-level Dirichlet partition of PTB-XL and its checks (plan task G0-07).

Rules (pre-registration §13 wins over older text):
- PR:64: 7 gateways x 30 devices; a patient is never split (true by DL-08).
- DL-29(c): training records go to 7 equal gateways at random; inside each
  gateway, Dirichlet(alpha_g) over its 30 devices on each record's primary
  label (its rarest positive superclass, by training prevalence).
- DL-29(d): every device keeps >= min_records_per_device records; the
  Dirichlet draw is repeated up to max_redraws times.
- DL-30: if no draw meets the floor, the last draw is repaired: one record at
  a time moves at random from the largest to the smallest device of that
  gateway until the floor holds (needed for alpha = 0.1 in E5).
- DL-29(e): the partition depends on the run seed s.
- DL-26: one Poisson sampling rate q = batch / mean records per device and
  S = ceil(1/q) local steps per round for every device.

Example:
    python 04_Code/fl_system/partition.py --config 05_Experiments/configs/partition_main.yaml \
        --ptbxl-dir ~/Documents/PhD_data/ptbxl --seed 0 \
        --out 05_Experiments/results/G0/partition_seed0.json \
        --report 05_Experiments/results/G0/partition_check_seed0.md
"""
import argparse
import datetime as dt
import json
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ptbxl import SUPERCLASSES, TRAIN_FOLDS, label_matrix, load_records, split_folds  # noqa: E402


class PartitionError(RuntimeError):
    pass


def primary_labels(labels, prevalence):
    """Index of each record's rarest positive superclass (ties: class order)."""
    masked = np.where(labels == 1, prevalence[None, :], np.inf)
    return masked.argmin(axis=1)


def repair_floor(parts, min_size, rng):
    """DL-30: move single records at random from the largest to the smallest device."""
    if sum(len(p) for p in parts) < min_size * len(parts):
        raise PartitionError(f"too few records for {len(parts)} devices x {min_size}")
    moved = 0
    while True:
        sizes = [len(p) for p in parts]
        small = int(np.argmin(sizes))
        if sizes[small] >= min_size:
            return moved
        big = int(np.argmax(sizes))
        parts[small].append(parts[big].pop(int(rng.integers(sizes[big]))))
        moved += 1


def dirichlet_split(idx_by_class, n_devices, alpha, rng, min_size, max_redraws):
    """Split each class over devices by Dirichlet(alpha); redraw until all are big enough.

    Returns (parts, draws used, records moved by the DL-30 repair).
    """
    for attempt in range(1, max_redraws + 1):
        parts = [[] for _ in range(n_devices)]
        for idx in idx_by_class:
            idx = rng.permutation(idx)
            share = rng.dirichlet(np.full(n_devices, alpha))
            cuts = (np.cumsum(share)[:-1] * len(idx)).astype(int)
            for d, chunk in enumerate(np.split(idx, cuts)):
                parts[d].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return parts, attempt, 0
    return parts, max_redraws, repair_floor(parts, min_size, rng)


def make_partition(train, cfg, seed):
    rng = np.random.default_rng(seed)
    labels = label_matrix(train)
    prim = primary_labels(labels, labels.mean(axis=0))
    gateways = np.array_split(rng.permutation(len(train)), cfg["n_gateways"])
    alpha_over = {int(k): float(v) for k, v in (cfg.get("alpha_overrides") or {}).items()}
    devices, redraws = [], {}
    for g, g_idx in enumerate(gateways):
        alpha = alpha_over.get(g, float(cfg["alpha_default"]))
        idx_by_class = [g_idx[prim[g_idx] == c] for c in range(len(SUPERCLASSES))]
        parts, attempts, moved = dirichlet_split(idx_by_class, cfg["devices_per_gateway"], alpha, rng,
                                                 cfg["min_records_per_device"], cfg["max_redraws"])
        redraws[g] = {"alpha": alpha, "draws_used": attempts, "records_moved_DL30": moved,
                      "records_in_gateway": int(len(g_idx))}
        for j, part in enumerate(parts):
            devices.append({
                "device": g * cfg["devices_per_gateway"] + j,
                "gateway": g,
                "ecg_ids": sorted(train[i].ecg_id for i in part),
            })
    return devices, redraws


def sampling_plan(n_train, n_devices, batch, rounds):
    """DL-26: one q and one number of local steps S for all devices."""
    mean_n = n_train / n_devices
    q = batch / mean_n
    if not 0 < q <= 1:
        raise PartitionError(f"sampling rate q={q:.3f} is not in (0, 1]")
    steps_per_round = math.ceil(1 / q)
    return {"mean_records_per_device": round(mean_n, 3), "q": round(q, 6),
            "local_steps_per_round_S": steps_per_round, "rounds_T": rounds,
            "total_steps": rounds * steps_per_round}


def check_partition(devices, folds, cfg):
    train = folds["train"]
    by_id = {r.ecg_id: r for r in train}
    assigned = [e for d in devices for e in d["ecg_ids"]]
    patient_device = {}
    patient_clash = 0
    for d in devices:
        for e in d["ecg_ids"]:
            p = by_id[e].patient_id if e in by_id else None
            if p in patient_device and patient_device[p] != d["device"]:
                patient_clash += 1
            patient_device[p] = d["device"]
    held_out = {r.ecg_id for r in folds["val"]} | {r.ecg_id for r in folds["test"]}
    sizes = [len(d["ecg_ids"]) for d in devices]
    per_gw = np.bincount([d["gateway"] for d in devices], minlength=cfg["n_gateways"])
    n_dev = cfg["n_gateways"] * cfg["devices_per_gateway"]
    return [
        ("210 devices", len(devices) == n_dev, f"{len(devices)} devices"),
        ("30 devices per gateway", bool(np.all(per_gw == cfg["devices_per_gateway"])), f"{per_gw.tolist()}"),
        ("every training record assigned exactly once",
         len(assigned) == len(train) and len(set(assigned)) == len(train), f"{len(assigned)} of {len(train)}"),
        ("no patient on two devices", patient_clash == 0, f"{patient_clash} clashes"),
        ("no fold 9 or 10 record in training", not (set(assigned) & held_out)
         and all(by_id[e].fold in TRAIN_FOLDS for e in assigned), "folds 1-8 only"),
        (f"every device >= {cfg['min_records_per_device']} records",
         min(sizes) >= cfg["min_records_per_device"], f"min {min(sizes)}"),
    ]


def write_report(path, cfg, seed, plan, devices, redraws, checks, folds):
    sizes = np.array([len(d["ecg_ids"]) for d in devices])
    labels = {r.ecg_id: r.labels for r in folds["train"]}
    lines = [f"# Partition check — seed {seed}", "",
             f"Created {dt.datetime.now().isoformat(timespec='seconds')} by `04_Code/fl_system/partition.py`.", "",
             "## Checks", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |" for name, ok, detail in checks]
    lines += ["", "## Sampling plan (DL-26)", "| Item | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in plan.items()]
    lines += ["", "## Device sizes", f"min {sizes.min()}, mean {sizes.mean():.1f}, max {sizes.max()}", "",
              "| Gateway | alpha | Draws used | Records moved (DL-30) | Records | Smallest device | "
              + " | ".join(SUPERCLASSES) + " |",
              "|---|---|---|---|---|---|" + "---|" * len(SUPERCLASSES)]
    for g in range(cfg["n_gateways"]):
        devs = [d for d in devices if d["gateway"] == g]
        ids = [e for d in devs for e in d["ecg_ids"]]
        counts = np.sum([labels[e] for e in ids], axis=0)
        lines.append(f"| {g} | {redraws[g]['alpha']} | {redraws[g]['draws_used']} | "
                     f"{redraws[g]['records_moved_DL30']} | {len(ids)} | "
                     f"{min(len(d['ecg_ids']) for d in devs)} | " + " | ".join(str(int(c)) for c in counts) + " |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ptbxl-dir", required=True)
    ap.add_argument("--seed", type=int, required=True, help="run seed s (DL-29e)")
    ap.add_argument("--out", required=True, help="partition JSON")
    ap.add_argument("--report", required=True, help="check report (Markdown)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    records, dropped = load_records(args.ptbxl_dir)
    folds = split_folds(records)
    n_dev = cfg["n_gateways"] * cfg["devices_per_gateway"]
    plan = sampling_plan(len(folds["train"]), n_dev, cfg["batch_size"], cfg["rounds"])
    devices, redraws = make_partition(folds["train"], cfg, args.seed)
    checks = check_partition(devices, folds, cfg)

    for path in (args.out, args.report):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = {"created": dt.datetime.now().isoformat(timespec="seconds"), "seed": args.seed,
            "config": cfg, "config_path": args.config, "dropped_no_superclass": dropped,
            "n_train": len(folds["train"]), "sampling_plan": plan, "redraws": redraws,
            "checks": [{"check": n, "pass": bool(ok), "detail": d} for n, ok, d in checks]}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "devices": devices}, fh)
    write_report(args.report, cfg, args.seed, plan, devices, redraws, checks, folds)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({detail})")
    print(json.dumps(plan))
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)


if __name__ == "__main__":
    main()
