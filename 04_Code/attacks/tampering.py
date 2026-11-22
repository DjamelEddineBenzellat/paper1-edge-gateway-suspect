"""E5 gateway tampering (plan task X-7; PR §3 items 28-29, DL-14, DL-43 (8)-(11), DL-44, DL-45, DL-46).

One tampering gateway per run (b = 1, DL-14), drawn from the run seed, the same in every round, tampering from
round 1 with f = 0 (DL-43 (8)). The types act on the gateway's output (an update), at the gateway, before the cloud:
- scale: out = 2 v;
- signflip: out = -v;
- inject: out = a random vector with the same norm as v;
- drop: SILENT (DL-43 (9)): the gateway uses a random half of its inputs (P-TA: its complete group sums; P-PE: its
  admitted device models) and reports them all as used (see placements.py);
- adaptive against our check (DL-46 (1), PR item 29): honest in round 1, which is not scored; from round 2, the
  gateway tries s = 1.00, 0.99, ..., 0.00 and sends out = (1 - 2s) v for the largest s at which the cloud's check of
  the CURRENT round (the six other outputs and this candidate) does not flag it and gives it a distance <= 0.95 x
  that round's threshold; if no s qualifies, the s with the smallest distance / threshold ratio (adaptive_search).
- adaptive against the HFLMLD detectors (DL-45 (1), DL-46 (2)): honest in round 1, which is not scored; from round 2,
  out = -min(1, 0.95 D / ||v||) v, with D the largest distance ||v_e|| among the other edges of the CURRENT round
  (HFLMLD sees each edge model as w* + v, so its distance is ||v||).
The adaptive gateway knows the other gateways' outputs of the current round (DL-46 (3)): a worst case for every
detector.
Scoring (DL-23): a tampering gateway-round is detected if the gateway is flagged (our check: PR items 21-24 with
DL-44; HFLMLD: cloud weight 0, DL-43 (11)); a false alarm is a flagged honest gateway-round.
"""
import numpy as np

KINDS = ("scale", "signflip", "inject", "drop", "adaptive")
OUTPUT_KINDS = ("scale", "signflip", "inject", "adaptive")  # changed after the gateway's honest work
TAMPER_STREAM = 28  # PR item 28; separate from every other stream


def tamper_gateway(seed, n_gateways):
    """DL-43 (8): the tampering gateway, drawn from the run seed."""
    return int(np.random.default_rng([int(seed), TAMPER_STREAM]).integers(n_gateways))


def drop_half(n_items, seed, rnd, gateway):
    """DL-43 (9): the indices a silently dropping gateway keeps (a random half, at least one)."""
    if n_items == 0:
        return []
    keep = max(1, n_items // 2)
    r = np.random.default_rng([int(seed), TAMPER_STREAM, int(rnd), int(gateway)])
    return sorted(int(i) for i in r.choice(n_items, size=keep, replace=False))


S_GRID = [round(1.0 - i / 100, 2) for i in range(101)]  # DL-46 (1): s = 1.00, 0.99, ..., 0.00
MARGIN = 0.95  # PR item 29


def adaptive_search(v, check):
    """DL-46 (1): the largest s in S_GRID whose output (1 - 2s) v is not flagged by check and has a distance
    <= 0.95 x the threshold; if none qualifies, the s with the smallest distance / threshold ratio.
    check(out) -> (flagged, distance, threshold): the cloud's check of the current round with out as this gateway's
    output. Returns (out, s)."""
    v = np.asarray(v, dtype=np.float32)
    best = None
    for s in S_GRID:
        out = ((1.0 - 2.0 * s) * v).astype(np.float32)
        flagged, dist, thr = check(out)
        if not flagged and dist <= MARGIN * thr:
            return out, s
        ratio = dist / thr if thr > 0 else float("inf")
        if best is None or ratio < best[0]:
            best = (ratio, out, s)
    return best[1], best[2]


def adaptive_hflmld(v, other_max):
    """DL-45 (1), DL-46 (2): the full sign flip, shortened to 0.95 x the largest distance among the other edges of the
    current round if it would be the farthest edge. Returns (out, s) with s = the flip scale (0 = honest)."""
    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if other_max is None or norm == 0.0:
        return v, 0.0
    s = float(min(1.0, MARGIN * other_max / norm))
    return (-s * v).astype(np.float32), s


def tamper_output(kind, v, seed, rnd, gateway):
    """The tampered output of the tampering gateway for scale, signflip and inject, and None (no adaptive strength).
    The adaptive gateway needs the current round's other outputs: adaptive_search and adaptive_hflmld."""
    v = np.asarray(v, dtype=np.float32)
    if kind == "scale":
        return 2.0 * v, None
    if kind == "signflip":
        return -v, None
    if kind == "inject":
        r = np.random.default_rng([int(seed), TAMPER_STREAM, int(rnd), int(gateway), 1])
        x = r.standard_normal(v.shape)
        n = np.linalg.norm(x)
        return (x * (np.linalg.norm(v) / n if n > 0 else 0.0)).astype(np.float32), None
    raise ValueError(f"{kind}: not a fixed output tampering (drop acts inside the gateway; adaptive: adaptive_search, "
                     f"adaptive_hflmld)")


def scored(kind, rnd):
    """Whether the tampering gateway's round counts (adaptive: round 1 is honest and not scored, DL-43 (10), DL-46)."""
    return not (kind == "adaptive" and rnd == 1)


def score(rounds, gateway, kind, n_gateways):
    """Detection and false alarms per gateway-round (DL-23) from the per-round flags."""
    tp = tn = fa = miss = 0
    for r in rounds:
        flagged = set(r["flagged"])
        for g in range(n_gateways):
            if g == gateway:
                if not scored(kind, r["round"]):
                    continue
                tp, miss = (tp + 1, miss) if g in flagged else (tp, miss + 1)
            else:
                fa, tn = (fa + 1, tn) if g in flagged else (fa, tn + 1)
    return {"tampering_gateway_rounds": tp + miss, "detected": tp, "detection_rate": tp / max(1, tp + miss),
            "honest_gateway_rounds": fa + tn, "false_alarms": fa, "false_alarm_rate": fa / max(1, fa + tn)}
