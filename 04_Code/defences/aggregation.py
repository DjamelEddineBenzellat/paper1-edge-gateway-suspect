"""Gateway and cloud aggregation for Paper 1 (v4 §3; PR:68, PR:72-76; DL-10).

Gateway (P-TA): clip each device update to C, quantize, sum groups of k
(secure aggregation simulated as an exact sum of the quantized vectors; the
masks cancel), drop group sums with norm > tau (reporting their IDs), and take
the coordinate-wise median of (sum / k).
Cloud: coordinate-wise median over gateways, after the gateway check:
distance to the median, normalized by the expected noise norm for the number
of group sums used (DL-10), flagged if above median + 3 * 1.4826 * MAD, or if
the drop or abort rate is above 10%. Flagged gateways are excluded that round.
"""
import numpy as np
import torch

QUANT_BITS = 16  # 16-bit fixed point inside a 32-bit field (PR:68)
MAD_FACTOR = 3.0  # PR:73
MAD_SCALE = 1.4826
RATE_ALERT = 0.10  # PR:75, DL-10
REPUTATION_DECAY = 0.9  # PR:74


def clip_norm(vec, c):
    """Scale vec so that its L2 norm is at most c (no change if c is None)."""
    if c is None:
        return vec
    norm = float(torch.linalg.vector_norm(vec))
    return vec * min(1.0, c / (norm + 1e-12))


def quantize(vec, c):
    """Fixed point: each coordinate (|x| <= c after clipping) -> integer in [-(2^15-1), 2^15-1]."""
    scale = (2 ** (QUANT_BITS - 1) - 1) / c
    return torch.round(torch.clamp(vec, -c, c) * scale).to(torch.int64), scale


def group_sums(vectors, groups, c_clip=None, quant=False):
    """Sum each group's (clipped, optionally quantized) device vectors.

    vectors: {device: float tensor}; groups: list of device-id lists.
    Returns float group sums (dequantized) and the device norms before clipping.
    """
    dev_norms = {d: float(torch.linalg.vector_norm(v)) for d, v in vectors.items()}
    sums = []
    for g in groups:
        clipped = [clip_norm(vectors[d], c_clip) for d in g]
        if quant:
            ints, scale = zip(*(quantize(v, c_clip) for v in clipped))
            total = torch.stack(ints).sum(dim=0)
            # 32-bit field: a sum of up to k = 5 values of 16-bit range cannot wrap
            assert int(total.abs().max()) < 2 ** 31, "quantized group sum overflowed the 32-bit field"
            sums.append(total.to(torch.float32) / scale[0])
        else:
            sums.append(torch.stack(clipped).sum(dim=0))
    return sums, dev_norms


def gateway_median(sums, k, tau=None):
    """Drop sums with norm > tau; coordinate-wise median of (sum / k) over the rest."""
    norms = [float(torch.linalg.vector_norm(s)) for s in sums]
    keep = [i for i, nrm in enumerate(norms) if tau is None or nrm <= tau]
    dropped = [i for i in range(len(sums)) if i not in keep]
    if not keep:
        return None, norms, dropped
    stacked = torch.stack([sums[i] / k for i in keep])
    return median_lower_upper_mean(stacked), norms, dropped


def median_lower_upper_mean(stacked):
    """Coordinate-wise median; for an even count, the mean of the two middle values."""
    vals, _ = torch.sort(stacked, dim=0)
    n = stacked.shape[0]
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def gateway_check(outputs, used, full, rates, scales=None):
    """DL-10 gateway check. outputs: {g: tensor}; used/full: group sums used / drawn; rates: drop or abort rate.
    scales: DL-44 size-aware noise scale per gateway; it replaces the sqrt(used/full) factor.

    Returns (flagged set, normalized distances, threshold).
    """
    gws = sorted(outputs)
    med = median_lower_upper_mean(torch.stack([outputs[g] for g in gws]))
    raw = {g: float(torch.linalg.vector_norm(outputs[g] - med)) for g in gws}
    dist = {g: raw[g] / scales[g] for g in gws} if scales is not None else \
        {g: raw[g] * np.sqrt(used[g] / full[g]) for g in gws}
    d = np.array([dist[g] for g in gws])
    centre = np.median(d)
    mad = np.median(np.abs(d - centre))
    threshold = centre + MAD_FACTOR * MAD_SCALE * mad
    flagged = {g for g in gws if dist[g] > threshold or rates.get(g, 0.0) > RATE_ALERT}
    return flagged, dist, float(threshold)


def cloud_median(outputs, exclude=()):
    gws = [g for g in sorted(outputs) if g not in exclude] or sorted(outputs)
    return median_lower_upper_mean(torch.stack([outputs[g] for g in gws])), gws
