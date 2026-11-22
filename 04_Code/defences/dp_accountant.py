"""Noise multiplier z from the Opacus RDP accountant (plan task G0-08).

Rules (pre-registration §13 wins over older text):
- PR:60: z comes from the accountant; DL-01 removed the cut-off z <= 1.5.
- DL-26: one sampling rate q and S local steps per round for all devices,
  so total steps = T * S. Both come from the partition JSON (partition.py).
- DL-15: report the default epsilon and the epsilon line {1, 3, 8}.
- PR:54: delta = 1e-5.

The official G0-a values must come from Opacus (PR:41). Without Opacus the
script stops, unless --allow-fallback is given. The fallback is an internal
RDP accountant (integer orders 2-256, the RDP-to-DP conversion used by Opacus);
its output file is marked FALLBACK and is a preview only.
The Opacus function names below are those of Opacus 1.x; check them against
the version pinned at G0.

Example:
    python 04_Code/defences/dp_accountant.py \
        --partition 05_Experiments/results/G0/partition_seed0.json \
        --out 05_Experiments/results/G0/G0a_noise_multipliers.csv
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
import sys

ORDERS = list(range(2, 257))


def _logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _rdp_subsampled_gaussian(q, z, alpha):
    """RDP of the Poisson-subsampled Gaussian mechanism at integer order alpha."""
    if q == 1.0:
        return alpha / (2 * z * z)
    terms = []
    for k in range(alpha + 1):
        log_binom = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        terms.append(log_binom + (alpha - k) * math.log(1 - q) + k * math.log(q)
                     + (k * k - k) / (2 * z * z))
    return _logsumexp(terms) / (alpha - 1)


def fallback_epsilon(q, z, steps, delta):
    best = math.inf
    for a in ORDERS:
        rdp = steps * _rdp_subsampled_gaussian(q, z, a)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, eps)
    return best


def fallback_noise_multiplier(q, steps, eps, delta, lo=0.3, hi=500.0):
    for _ in range(60):
        mid = (lo + hi) / 2
        if fallback_epsilon(q, mid, steps, delta) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def opacus_noise_multiplier(q, steps, eps, delta):
    from opacus.accountants import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier

    z = get_noise_multiplier(target_epsilon=eps, target_delta=delta, sample_rate=q,
                             steps=steps, accountant="rdp")
    try:  # independent check: compose the same history and read epsilon back
        acc = RDPAccountant()
        acc.history = [(z, q, steps)]
        eps_back = acc.get_epsilon(delta=delta)
    except Exception as err:  # API differs in the pinned version: report, do not guess
        eps_back = f"check unavailable: {err}"
    return z, eps_back


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--partition", help="partition JSON from partition.py (gives q and total steps)")
    ap.add_argument("--q", type=float, help="sampling rate, if no partition is given")
    ap.add_argument("--steps", type=int, help="total DP-SGD steps T*S, if no partition is given")
    ap.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 3.0, 8.0])
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--out", required=True, help="CSV path")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="use the internal accountant if Opacus is missing (preview only)")
    args = ap.parse_args()

    if args.partition:
        with open(args.partition, encoding="utf-8") as fh:
            plan = json.load(fh)["meta"]["sampling_plan"]
        q, steps = float(plan["q"]), int(plan["total_steps"])
    elif args.q is not None and args.steps is not None:
        q, steps = args.q, args.steps
    else:
        ap.error("give --partition, or both --q and --steps")

    try:
        import opacus
        accountant, version = "opacus-rdp", opacus.__version__
    except ImportError:
        if not args.allow_fallback:
            sys.exit("Opacus is not installed. Install the pinned version (G0-03), "
                     "or use --allow-fallback for a preview that is NOT the official G0-a value.")
        accountant, version = "internal-rdp FALLBACK (preview, not official)", "n/a"
        root, ext = os.path.splitext(args.out)
        args.out = f"{root}_FALLBACK_preview{ext or '.csv'}"

    rows = []
    for eps in args.epsilons:
        if accountant == "opacus-rdp":
            z, eps_back = opacus_noise_multiplier(q, steps, eps, args.delta)
        else:
            z = fallback_noise_multiplier(q, steps, eps, args.delta)
            eps_back = fallback_epsilon(q, z, steps, args.delta)
        rows.append({"epsilon": eps, "delta": args.delta, "q": q, "total_steps": steps,
                     "noise_multiplier_z": round(z, 4),
                     "epsilon_check": eps_back if isinstance(eps_back, str) else round(eps_back, 4),
                     "accountant": accountant, "accountant_version": version,
                     "created": dt.datetime.now().isoformat(timespec="seconds")})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"eps={r['epsilon']}: z={r['noise_multiplier_z']} (check eps={r['epsilon_check']}, {accountant})")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
