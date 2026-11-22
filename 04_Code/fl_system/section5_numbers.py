"""Numbers of Paper 1, Section V (what the placements predict), from the configs and partitions only.

Pure-noise model (no learning signal, no clip C, no tau, no quantization): each device update carries
Gaussian noise with per-coordinate standard deviation lr*sqrt(S)*z*C_ex/(q*n_d) (DP-SGD as in dpsgd.py);
P-TG-c's gateway adds one noise to its pooled records, divided by q*sum(n_d) (DL-03). Prints:
1. K = sqrt(d)*lr*sqrt(S)*z*C_ex/q, so that the noise norm of a device update is K/n_d;
2. the signal-to-noise bound: signal norm <= lr*S*C_ex, hence ratio <= n_d / B with B = z*sqrt(d)/(sqrt(S)*q),
   which does not depend on lr or C_ex;
3. the predicted percentiles of device-update and group-sum norms against those measured in the run that set
   C and tau (DL-20, DL-31: P-TA, seed 0, C/tau/quantization off, before the canaries of DL-38);
4. the noise at the cloud output of each placement relative to P-TG-c, for equal and for real device sizes.
No model is trained; light laptop work.

Example:
    python section5_numbers.py --ptbxl-dir <PTB-XL> --partition 05_Experiments/configs/partition_main.yaml \
        --config 05_Experiments/configs/E1/E1_pta_s0.yaml --noise-csv 05_Experiments/results/G0/G0a_noise_multipliers.csv
"""
import argparse
import csv
import json
import math

import numpy as np
import yaml

from model import ecg_cnn
from train_data import RunData


def noise_params(config, noise_csv):
    with open(config, encoding="utf-8") as fh:
        st = yaml.safe_load(fh)["settings"]
    with open(noise_csv, encoding="utf-8") as fh:
        row = next(r for r in csv.DictReader(fh) if float(r["epsilon"]) == float(st["eps0"]))
    steps = int(row["total_steps"]) // int(st["rounds"])
    return {"lr": st["lr"], "c_ex": st["c_ex"], "z": float(row["noise_multiplier_z"]), "q": float(row["q"]),
            "S": steps, "C": st["clip_C"], "tau": st["tau"], "eps": st["eps0"], "delta": float(row["delta"]), "rounds": st["rounds"]}


def cloud_noise(sizes_by_gw, design, k, rng, draws=200, coords=400):
    """Standard deviation of one coordinate of the cloud output, noise sigma_d = 1/n_d (common factor dropped)."""
    out = []
    for _ in range(draws):
        gw_vals, all_groups = [], []
        for sizes in sizes_by_gw:
            sizes = np.asarray(sizes, dtype=float)
            if design == "P-TG-c":  # one noise on the pooled records, divided by q * sum(n)
                gw_vals.append(rng.standard_normal(coords) / sizes.sum())
                continue
            dev = rng.standard_normal((len(sizes), coords)) / sizes[:, None]
            if design == "P-TG":
                gw_vals.append(np.median(dev, axis=0))
                continue
            perm = rng.permutation(len(sizes))
            groups = [perm[i:i + k] for i in range(0, len(sizes) - len(sizes) % k, k)]
            means = np.stack([dev[g].sum(axis=0) / k for g in groups])
            if design == "P-TA":
                gw_vals.append(np.median(means, axis=0))
            else:  # P-FR: all group means go to the cloud
                all_groups.append(means)
        vals = np.concatenate(all_groups) if design == "P-FR" else np.stack(gw_vals)
        out.append(np.median(vals, axis=0))
    return float(np.sqrt(np.mean(np.square(np.concatenate(out)))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ptbxl-dir", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--config", required=True, help="a P-TA E1 config (final training point)")
    ap.add_argument("--noise-csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--norm-run", required=True, help="the DL-36 grid run that set C and tau "
                    "(results/G0-14/JOB_20261115_G0-14_grid-dl36/runs/pta_grid_lr0.01_cex2.0_s0.json)")
    ap.add_argument("--out", help="write the numbers as JSON")
    args = ap.parse_args(argv)
    p = noise_params(args.config, args.noise_csv)
    d = sum(t.numel() for t in ecg_cnn().parameters())
    K = math.sqrt(d) * p["lr"] * math.sqrt(p["S"]) * p["z"] * p["c_ex"] / p["q"]
    B = p["z"] * math.sqrt(d) / (math.sqrt(p["S"]) * p["q"])
    res = {"params": p, "d": d, "K": K, "B_records": B}
    rng = np.random.default_rng(20261121)
    sizes, per_seed = [], {}
    for seed in args.seeds:
        data = RunData(args.ptbxl_dir, args.partition, seed, need_val=False, canaries=True)
        gws = [[len(data.device_rows[dv]) for dv in sorted(data.gateway_devices[g])] for g in sorted(data.gateway_devices)]
        per_seed[seed] = gws
        sizes += [n for g in gws for n in g]
    sizes = np.asarray(sizes, dtype=float)
    res["device_records"] = {"mean": float(sizes.mean()), "median": float(np.median(sizes)), "min": float(sizes.min()),
                             "max": float(sizes.max())}
    res["snr_bound_at"] = {str(n): n / B for n in (18, 70, 151)}
    # 3. the run that set C and tau (DL-20, DL-31): P-TA, seed 0, no canaries; its 100 rounds of random groups of 3
    with open(args.norm_run, encoding="utf-8") as fh:
        run = json.load(fh)
    data0 = RunData(args.ptbxl_dir, args.partition, 0, need_val=False, canaries=False)
    if data0.partition_sha256 != run["data"]["partition_sha256"]:
        raise SystemExit("seed-0 partition differs from the norm run's partition")
    gws0 = [[len(data0.device_rows[dv]) for dv in sorted(data0.gateway_devices[g])] for g in sorted(data0.gateway_devices)]
    dev_norms = np.repeat(K / np.asarray([n for g in gws0 for n in g], float), int(p["rounds"]))
    grp = []
    for _ in range(int(p["rounds"])):
        for g in gws0:
            inv = 1.0 / np.asarray(g, float)[rng.permutation(len(g))]
            grp += list(K * np.sqrt((inv.reshape(-1, 3) ** 2).sum(axis=1)))
    pct = {"p50": 50, "p90": 90, "p99": 99}
    res["norm_check_seed0"] = {
        "device_update": {key: {"predicted": float(np.percentile(dev_norms, v)), "measured": run["norms"]["device_update"][key]}
                          for key, v in pct.items()},
        "group_sum_k3": {key: {"predicted": float(np.percentile(grp, v)), "measured": run["norms"]["group_sum"][key]}
                         for key, v in pct.items()}}
    # 4. noise at the cloud output relative to P-TG-c
    rows = {}
    for label, by_seed in (("equal sizes", {s: [[sizes.mean()] * len(g) for g in per_seed[s]] for s in args.seeds}),
                           ("real sizes", per_seed)):
        ratios = {}
        for design, k in (("P-TG", 1), ("P-FR", 3), ("P-TA", 1), ("P-TA", 2), ("P-TA", 3), ("P-TA", 5)):
            r = [cloud_noise(by_seed[s], design, k, rng) / cloud_noise(by_seed[s], "P-TG-c", 1, rng)
                 for s in args.seeds]
            ratios[f"{design} k={k}" if design == "P-TA" else design] = round(float(np.mean(r)), 2)
        rows[label] = ratios
    res["noise_ratio_vs_ptgc"] = rows
    text = json.dumps(res, indent=1)
    print(text, flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
