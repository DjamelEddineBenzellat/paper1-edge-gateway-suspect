"""Figures F2-F5 of Paper 1 (plan task X-11), from the fold-10 outputs; no model is trained.

- F2: price of distrust (P-TG-c - P-TA), shared point vs tuned points, per seed with the paired 95% CI
  (e1_fold10.json h3a_eps0_8; dl42_fold10.json tuned_price_of_distrust).
- F3: the eps line: P-TA and P-TG-c macro-F1 at eps 1, 3, 8 (seeds 0-2) and the trivial classifier (e6_fold10.json).
- F4: the k line: P-TA and P-FR macro-F1 at k 1, 2, 3, 5 (seeds 0-2; e6_fold10.json).
- F5: who gains from federation: each device's local-only macro-F1 on fold 10 against its number of training records,
  with P-TA's global model as a line (E1 local-only sealed files, already opened in M3-03, re-read only if their
  SHA-256 matches the G1 register; device sizes from the partition of each seed).

Example:
    python paper_figures.py --results 05_Experiments/results --ptbxl-dir <PTB-XL> --out 07_Paper/figures
"""
import argparse
import hashlib
import json
import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLOURS = {"P-TA": "#0072B2", "P-TG-c": "#D55E00", "P-FR": "#009E73", "local": "#999999", "FedAvg": "#CC79A7"}
WIDTH = 3.5  # inches, one IEEE column
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def f2(results, out):
    shared = load(os.path.join(results, "E1", "e1_fold10.json"))["h3a_eps0_8"]
    tuned = load(os.path.join(results, "DL-42", "dl42_fold10.json"))["tuned_price_of_distrust"]["all_seeds"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    for x, (label, p) in enumerate((("Shared point\n(frozen plan, H3(a))", shared),
                                    ("Each placement tuned\n(sensitivity, DL-42)", tuned))):
        pts = np.asarray(p["per_seed_points"], dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(pts))
        ax.scatter(x + jitter, pts, s=10, color=COLOURS["P-TG-c"], alpha=0.45, linewidths=0)
        ax.errorbar(x + 0.3, p["mean"], yerr=[[p["mean"] - p["ci95"][0]], [p["ci95"][1] - p["mean"]]], fmt="o",
                    color="black", ms=4, capsize=3, lw=1)
        ax.annotate(f"{p['mean']:+.2f}", (x + 0.38, p["mean"]), va="center", fontsize=7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks([0, 1], ["Shared point\n(frozen plan, H3(a))", "Each placement tuned\n(sensitivity, DL-42)"])
    ax.set_xlim(-0.5, 1.7)
    ax.set_ylabel("P-TG-c − P-TA, macro-F1 points")
    save(fig, out, "F2_price_of_distrust")


def f3(results, out):
    e6 = load(os.path.join(results, "E6", "e6_fold10.json"))
    t = e6["table"]
    eps = [1, 3, 8]
    rows = {"P-TA": [t["pta eps 1"], t["pta eps 3"], t["P-TA k 3 eps 8 (reference)"]],
            "P-TG-c": [t["ptgc eps 1"], t["ptgc eps 3"], t["P-TG-c k 3 eps 8 (reference)"]]}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    for name, vals in rows.items():
        m = [v["f1"]["mean"] for v in vals]
        sd = [v["f1"]["sd"] for v in vals]
        ax.errorbar(eps, m, yerr=sd, marker="o", ms=4, capsize=2, lw=1.2, color=COLOURS[name],
                    label=name + (" (trusted gateway, central DP)" if name == "P-TG-c" else " (device DP)"))
    gaps = e6["h3b"]["mean_gap_points"]
    for i, e in enumerate(eps):
        lo, hi = rows["P-TA"][i]["f1"]["mean"], rows["P-TG-c"][i]["f1"]["mean"]
        ax.plot([e * 1.06] * 2, [lo + 0.006, hi - 0.006], color="0.45", lw=0.7, ls="--")
        ax.annotate(f"{gaps[str(e)]:+.1f}", (e * 1.09, (lo + hi) / 2), va="center", fontsize=6.5, color="0.3")
    ax.axhline(e6["h3b"]["trivial_fold10"], color="black", ls=":", lw=0.8, label="Trivial classifier")
    ax.set_xscale("log")
    ax.set_xticks(eps, [str(e) for e in eps])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.85, 10.5)
    ax.set_xlabel("Patient-level ε (δ = 10⁻⁵)")
    ax.set_ylabel("Macro-F1, test fold")
    ax.set_ylim(0.38, 0.62)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
    save(fig, out, "F3_epsilon_line")


def f4(results, out):
    t = load(os.path.join(results, "E6", "e6_fold10.json"))["table"]
    ks = [1, 2, 3, 5]
    rows = {"P-TA": [t["pta k 1"], t["pta k 2"], t["P-TA k 3 eps 8 (reference)"], t["pta k 5"]],
            "P-FR": [t["pfr k 1"], t["pfr k 2"], t["P-FR k 3 eps 8 (reference)"], t["pfr k 5"]]}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.0))
    for i, (name, vals) in enumerate(rows.items()):
        x = np.array(ks) + (i - 0.5) * 0.12
        ax.errorbar(x, [v["f1"]["mean"] for v in vals], yerr=[v["f1"]["sd"] for v in vals], marker="o", ms=4,
                    capsize=2, lw=1.2, color=COLOURS[name], label=name)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("Group size k (devices per secure-aggregation group)")
    ax.set_ylabel("Macro-F1, test fold")
    ax.set_ylim(0.46, 0.54)
    ax.legend(frameon=False, loc="lower right")
    save(fig, out, "F4_group_size_line")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


def device_scores(results, ptbxl_dir, partition):
    """(records, local-only fold-10 macro-F1) of every device of the 10 E1 seeds; seals checked first."""
    from train_data import RunData
    job = os.path.join(results, "M3-01", "JOB_20261117_M3-01_e1")
    register = load(os.path.join(results, "G1", "g1_check.json"))["seal_register"]
    sizes, scores = [], []
    for seed in range(10):
        rel = f"JOB_20261117_M3-01_e1/E1_local_s{seed}.fold10.sealed.json"
        path = os.path.join(job, "sealed_fold10", f"E1_local_s{seed}.fold10.sealed.json")
        want = register[rel]["sha256"] if isinstance(register[rel], dict) else register[rel]
        if sha256(path) != want:
            raise SystemExit(f"{path}: SHA-256 differs from the G1 register")
        devices = load(path)["devices"]
        data = RunData(ptbxl_dir, partition, seed, need_val=False, canaries=True)  # as the E1 local-only runs
        run = load(os.path.join(job, "runs", f"E1_local_s{seed}.json"))
        if data.partition_sha256 != find_key(run, "partition_sha256"):
            raise SystemExit(f"seed {seed}: the rebuilt partition differs from the one the local-only run used")
        order = sorted(data.device_rows)  # run_pilot.py (mode local) scores the devices in this order
        if len(order) != len(devices):
            raise SystemExit(f"seed {seed}: {len(devices)} scores for {len(order)} devices")
        sizes += [len(data.device_rows[d]) for d in order]
        scores += [dev["macro_f1_tuned"] for dev in devices]
    return np.array(sizes), np.array(scores)


def f5(results, ptbxl_dir, partition, out):
    sizes, scores = device_scores(results, ptbxl_dir, partition)
    e1 = load(os.path.join(results, "E1", "e1_fold10.json"))["designs"]
    pta = e1["P-TA"]["f1"]["mean"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    ax.scatter(sizes, scores, s=4, color=COLOURS["local"], alpha=0.35, linewidths=0,
               label="One device alone, no DP (10 seeds × 210)")
    edges = np.unique(np.quantile(sizes, np.linspace(0, 1, 9)))
    mids, means, below = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sizes >= lo) & (sizes <= hi if hi == edges[-1] else sizes < hi)
        if sel.sum():
            mids.append(np.median(sizes[sel]))
            means.append(scores[sel].mean())
            below.append(float((scores[sel] < pta).mean()))
    ax.plot(mids, means, color="black", lw=1.2, marker="s", ms=3, label="Mean of each size eighth")
    ax.axhline(pta, color=COLOURS["P-TA"], lw=1.2, label=f"P-TA global model, device DP ε = 8 ({pta:.3f})")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Training records on the device")
    ax.set_ylabel("Macro-F1, test fold")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, markerscale=2)
    save(fig, out, "F5_who_gains")
    return {"devices": int(len(sizes)), "share_below_pta": float((scores < pta).mean()),
            "share_below_pta_by_size_eighth": [round(b, 3) for b in below],
            "median_records_by_eighth": [float(m) for m in mids],
            "local_f1_mean_by_eighth": [round(float(m), 4) for m in means], "pta_f1": pta,
            "local_f1_mean": float(scores.mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="05_Experiments/results")
    ap.add_argument("--out", required=True, help="07_Paper/figures")
    ap.add_argument("--ptbxl-dir", help="F5 only: the PTB-XL folder (device sizes)")
    ap.add_argument("--partition", help="F5 only: partition_main.yaml")
    args = ap.parse_args(argv)
    f2(args.results, args.out)
    f3(args.results, args.out)
    f4(args.results, args.out)
    facts = {}
    if args.ptbxl_dir:
        facts["F5"] = f5(args.results, args.ptbxl_dir, args.partition, args.out)
    with open(os.path.join(args.out, "figure_facts.json"), "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=1)
    print("figures written to", args.out, "| F5:", facts.get("F5"), flush=True)


if __name__ == "__main__":
    main()
