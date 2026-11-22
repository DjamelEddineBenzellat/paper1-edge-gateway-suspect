"""Gateway-level and cloud-level work of each placement (v4 §3, PR §5; DL-03, DL-10, DL-37).

Framework-free, so unit tests can call it directly. flower_sim.py runs the
gateway functions inside Flower nodes (one node per gateway) and the cloud
functions in the Flower ServerApp.
"""
import math
import os
import sys
import time

import numpy as np
import torch

_ATTACKS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "attacks"))
if _ATTACKS not in sys.path:  # E4 device attacks (device_attacks.py), also inside the Flower workers
    sys.path.insert(0, _ATTACKS)

from aggregation import cloud_median, gateway_check, gateway_median, group_sums, median_lower_upper_mean
from seeding import DETECT, GROUPS, SIZE_AWARE, rng, seed_int

SIZE_AWARE_DRAWS = 2000  # DL-44

_TRAINERS = {}


def trainer_for(device, model_fn):
    """One Trainer per process and torch device (Ray workers keep theirs between rounds)."""
    from dpsgd import Trainer
    if device not in _TRAINERS:
        _TRAINERS[device] = Trainer(model_fn, device)
    return _TRAINERS[device]


def draw_groups(seed, rnd, gateway, devices, k):
    """Cloud draws random groups of k inside one gateway, every round (v4:89, v4:105)."""
    order = rng(seed, GROUPS, rnd, gateway).permutation(sorted(devices))
    return [[int(d) for d in order[i:i + k]] for i in range(0, len(order), k)]


def _canary_scores(trainer, data, gateway, start_vec, observed):
    """Gateway-view canary scores of one round (DL-38); {} when the run has no canaries."""
    if getattr(data, "canaries", None) is None:
        return {}
    from canaries import gateway_view
    return gateway_view(trainer, data, gateway, start_vec, observed)


def accepting_devices(data, gateway, rnd, global_vec, groups=None, auth=None):
    """M2-02: the devices that accept this round after their certificate checks, the refusals
    {device: reason}, and the check time. Without auth (unit tests, baselines) every device accepts."""
    devices = data.gateway_devices[gateway]
    if not auth:
        return list(devices), {}, 0.0
    from certificates import device_check
    t0 = time.perf_counter()
    refused = {}
    for d in devices:
        group = next((g for g in groups if d in g), None) if groups is not None else None
        ok, reason = device_check(auth["cert"], auth["cloud_pub"], rnd, global_vec, gateway, d,
                                  group if groups is not None else None, auth.get("device_certs"))
        if not ok or (groups is not None and group is None):
            refused[int(d)] = reason if not ok else "not in any group"
    return [d for d in devices if int(d) not in refused], refused, time.perf_counter() - t0


def pta_gateway(trainer, data, gateway, rnd, cfg, global_vec, groups, auth=None):
    """P-TA: devices check the certificates and run DP-SGD; groups with a missing member abort
    (t = k); the gateway sums the complete groups, checks norms and takes the median."""
    from certificates import complete_groups
    accepted, refused, check_s = accepting_devices(data, gateway, rnd, global_vec, groups, auth)
    t0 = time.perf_counter()
    vectors = {d: _dp_update(trainer, data, d, rnd, cfg, global_vec) for d in accepted}
    t1 = time.perf_counter()
    complete, aborted = complete_groups(groups, vectors)
    sums, dev_norms = group_sums(vectors, complete, cfg.get("clip_C"), cfg.get("quantize", False))
    if _silent_drop(cfg, gateway) and sums:  # E5 (DL-43 (9)): half of the sums, all reported as used
        from tampering import drop_half
        keep = drop_half(len(sums), cfg["seed"], rnd, gateway)
        out, _, kept_dropped = gateway_median([sums[i] for i in keep], cfg["k"], cfg.get("tau"))
        dropped = [keep[i] for i in kept_dropped]  # it reports its tau drops only
        group_norms = [float(torch.linalg.vector_norm(s)) for s in sums]
    else:
        out, group_norms, dropped = gateway_median(sums, cfg["k"], cfg.get("tau")) if sums else (None, [], [])
    t2 = time.perf_counter()
    vec = np.zeros(trainer.n_params, np.float32) if out is None else out.cpu().numpy().astype(np.float32)
    # the gateway sees each complete group's sum: score the canaries hosted in that group (DL-38)
    canary_gw = _canary_scores(trainer, data, gateway, global_vec,
                               {int(d): s for grp, s in zip(complete, sums) for d in grp})
    return vec, {"gateway": gateway, "used": len(complete) - len(dropped), "full": len(groups), "dropped": dropped,
                 "aborted_groups": len(aborted), "refused": refused, "check_s": check_s,
                 "canary_gw": {str(i): v for i, v in canary_gw.items()}, "canary_s": time.perf_counter() - t2,
                 "device_norms": [dev_norms[d] for d in sorted(dev_norms)], "group_norms": group_norms,
                 "train_s": t1 - t0, "agg_s": t2 - t1, **_poisoned_sums(cfg, complete, dropped),
                 "used_groups": [[int(d) for d in grp] for i, grp in enumerate(complete) if i not in set(dropped)],
                 "dropped_groups": [[int(d) for d in complete[i]] for i in dropped]}  # DL-48: reported with the IDs


def ptgc_gateway(trainer, data, gateway, rnd, cfg, global_vec, auth=None):
    """P-TG-c (DL-03): central DP-SGD at the trusted gateway on its devices' pooled records
    (devices that refuse the signed model contribute no records)."""
    from device_attacks import is_malicious, malicious_update
    accepted, refused, check_s = accepting_devices(data, gateway, rnd, global_vec, None, auth)
    t0 = time.perf_counter()
    malicious = [d for d in accepted if is_malicious(cfg, d)]
    honest = [d for d in accepted if d not in malicious]
    pooled = np.concatenate([data.device_rows[d] for d in honest])
    delta, _ = trainer.dp_sgd(global_vec, data.x_train, data.y_train, pooled, cfg["q"], cfg["steps"], cfg["z"],
                              cfg["c_ex"], cfg["lr"], keys=(cfg["seed"], rnd, 10_000 + gateway))
    if malicious:  # DL-44 (3): each attack update enters with its share of the gateway's records
        n = {d: len(data.device_rows[d]) for d in accepted}
        total = sum(n.values())
        delta = delta * (sum(n[d] for d in honest) / total)
        for m in malicious:
            delta = delta + (n[m] / total) * malicious_update(trainer, data, m, rnd, cfg, global_vec)
    return delta.cpu().numpy().astype(np.float32), {"gateway": gateway, "used": 1, "full": 1, "dropped": [],
                                                    "refused": refused, "check_s": check_s,
                                                    "train_s": time.perf_counter() - t0, "agg_s": 0.0}


def size_aware_scales(used_groups, sizes, seed, rnd, draws=SIZE_AWARE_DRAWS):
    """DL-44: per gateway, the standard deviation of the coordinate-wise median of the means of its used groups,
    when each device d adds independent noise with standard deviation 1/n_d (Monte Carlo, seeded by run seed,
    round and gateway), divided by the median over the gateways. A gateway with no used group gets 1."""
    raw = {}
    for g, groups in sorted(used_groups.items()):
        if not groups:
            raw[g] = None
            continue
        inv = np.array([[1.0 / sizes[d] for d in grp] for grp in groups])  # [groups, k]
        noise = rng(seed, SIZE_AWARE, rnd, g).standard_normal((draws,) + inv.shape) * inv[None]
        raw[g] = float(np.median(noise.mean(axis=2), axis=1).std())  # median over groups, as gateway_median
    ref = float(np.median([v for v in raw.values() if v is not None]))
    return {g: (v / ref if v is not None else 1.0) for g, v in raw.items()}


EXPLAINED = 0.95  # DL-48: a tau drop is explained if its predicted norm is at least 0.95 tau


def noise_norm_constant(n_params, lr, z, c_ex, steps, q):
    """DL-48: a DP-SGD update is almost pure noise, with norm about K / n_d; K = sqrt(d) lr z C_ex sqrt(S) / q."""
    return math.sqrt(n_params) * lr * z * c_ex * math.sqrt(steps) / q


def unexplained_drops(dropped_groups, drop_model):
    """DL-48: the reported tau drops whose group's predicted norm sqrt(sum min(C, K / n_i)^2) is below 0.95 tau,
    so that the device sizes do not explain them. drop_model: {"sizes", "k_noise", "clip_C", "tau"}."""
    sizes, k_noise, c, tau = (drop_model[key] for key in ("sizes", "k_noise", "clip_C", "tau"))
    return sum(math.sqrt(sum(min(c, k_noise / sizes[d]) ** 2 for d in grp)) < EXPLAINED * tau
               for grp in dropped_groups)


def check_inputs(stats, drop_model=None):
    """(used, full, rate) per gateway for the P-TA gateway check: group sums used and drawn, and the rate of the
    rate alert: (aborted + tau-dropped) / drawn (DL-10), or with drop_model (aborted + unexplained drops) / drawn
    (DL-48)."""
    used = {g: s["used"] for g, s in stats.items()}
    full = {g: s["full"] for g, s in stats.items()}
    if drop_model is None:
        rates = {g: 1.0 - s["used"] / s["full"] for g, s in stats.items()}
    else:
        rates = {g: (s["aborted_groups"] + unexplained_drops(s["dropped_groups"], drop_model)) / s["full"]
                 for g, s in stats.items()}
    return used, full, rates


def pta_check(outputs, stats, scales=None, drop_model=None):
    """The P-TA gateway check on these outputs: (flagged set, distances, threshold)."""
    used, full, rates = check_inputs(stats, drop_model)
    return gateway_check({g: torch.as_tensor(v) for g, v in outputs.items()}, used, full, rates, scales)


def cloud_pta(outputs, stats, scales=None, drop_model=None):
    """Cloud for P-TA: gateway check (DL-10; DL-44 size-aware if scales are given; DL-48 rate alert if drop_model is
    given), then the median over the non-flagged gateways. With scales, the DL-10 flags on the same outputs are
    reported too (they do not act)."""
    used, full, rates = check_inputs(stats, drop_model)
    vecs = {g: torch.as_tensor(v) for g, v in outputs.items()}
    flagged, dist, thr = gateway_check(vecs, used, full, rates, scales)
    agg, kept = cloud_median(vecs, exclude=flagged)
    info = {"flagged": sorted(flagged), "distances": dist, "threshold": thr, "kept": kept, "rates": rates}
    if scales is not None:
        f10, d10, t10 = gateway_check(vecs, used, full, rates)
        info.update(scales=scales, flagged_dl10=sorted(f10), distances_dl10=d10, threshold_dl10=t10)
    return agg.numpy(), info


def cloud_ptgc(outputs):
    """Cloud for P-TG-c (DL-03): median over the gateways' central-DP updates; no gateway check."""
    agg, kept = cloud_median({g: torch.as_tensor(v) for g, v in outputs.items()})
    return agg.numpy(), {"flagged": [], "kept": kept}


# ---- E1 configurations added in M2-01 (PR §5, DL-37) ----

def _dp_update(trainer, data, device, device_round, cfg, start_vec):
    """One device's DP-SGD update (one local epoch = S steps) from start_vec; a malicious device (E4, DL-43)
    sends its attack update instead."""
    from device_attacks import is_malicious, malicious_update
    if is_malicious(cfg, device):
        return malicious_update(trainer, data, device, device_round, cfg, start_vec)
    if cfg.get("no_dp"):  # DL-06: the E5 HFLMLD native row trains without DP (plain SGD), as in the paper
        return trainer.plain_sgd_update(start_vec, data.x_train, data.y_train, data.device_rows[device], 1,
                                        cfg["lr"], cfg["batch"], keys=(cfg["seed"], device_round, device))
    delta, _ = trainer.dp_sgd(start_vec, data.x_train, data.y_train, data.device_rows[device], cfg["q"],
                              cfg["steps"], cfg["z"], cfg["c_ex"], cfg["lr"], keys=(cfg["seed"], device_round, device))
    return delta


def _silent_drop(cfg, gateway):
    t = cfg.get("tamper") or {}
    return t.get("kind") == "drop" and t.get("gateway") == gateway


def _poisoned_sums(cfg, complete, dropped):
    """E4 log (DL-43 §4.1): group sums with a malicious member, and how many of them tau dropped."""
    from device_attacks import is_malicious
    if not cfg.get("attack"):
        return {}
    poisoned = {i for i, grp in enumerate(complete) if any(is_malicious(cfg, d) for d in grp)}
    return {"poisoned_sums": len(poisoned), "poisoned_dropped": len(poisoned & set(dropped))}


def _device_round(rnd, cfg, te=0):
    """Device-round index (1..T) of edge iteration te in cloud round rnd (T_e > 1 for P-PE and HierFAVG)."""
    return (rnd - 1) * cfg.get("edge_iterations", 1) + te + 1


def ptg_gateway(trainer, data, gateway, rnd, cfg, global_vec, auth=None):
    """P-TG: devices run DP-SGD; the trusted gateway takes the coordinate-wise median of the plaintext updates."""
    accepted, refused, check_s = accepting_devices(data, gateway, rnd, global_vec, None, auth)
    t0 = time.perf_counter()
    vectors = [_dp_update(trainer, data, d, rnd, cfg, global_vec) for d in accepted]
    t1 = time.perf_counter()
    med = median_lower_upper_mean(torch.stack(vectors)) if vectors else torch.zeros(trainer.n_params)
    canary_gw = _canary_scores(trainer, data, gateway, global_vec, {int(d): v for d, v in zip(accepted, vectors)})
    return med.cpu().numpy().astype(np.float32), {"gateway": gateway, "used": len(vectors),
                                                  "full": len(data.gateway_devices[gateway]), "dropped": [],
                                                  "canary_gw": {str(i): v for i, v in canary_gw.items()},
                                                  "refused": refused, "check_s": check_s, "train_s": t1 - t0,
                                                  "agg_s": time.perf_counter() - t1}


def pfr_gateway(trainer, data, gateway, rnd, cfg, global_vec, groups, auth=None):
    """P-FR: devices check the certificates, run DP-SGD, clip to C and quantize; secure aggregation per
    certified group (t = k: a group with a missing member aborts); the gateway only relays the group sums
    (simulated as the exact sums) to the cloud."""
    from certificates import complete_groups
    accepted, refused, check_s = accepting_devices(data, gateway, rnd, global_vec, groups, auth)
    t0 = time.perf_counter()
    vectors = {d: _dp_update(trainer, data, d, rnd, cfg, global_vec) for d in accepted}
    t1 = time.perf_counter()
    complete, aborted = complete_groups(groups, vectors)
    sums, dev_norms = group_sums(vectors, complete, cfg.get("clip_C"), cfg.get("quantize", False))
    return [s.cpu().numpy().astype(np.float32) for s in sums], {
        "gateway": gateway, "used": len(sums), "full": len(groups), "dropped": [], "aborted_groups": len(aborted),
        "refused": refused, "check_s": check_s,
        "device_norms": [dev_norms[d] for d in sorted(dev_norms)],
        "group_norms": [float(torch.linalg.vector_norm(s)) for s in sums],
        "train_s": t1 - t0, "agg_s": time.perf_counter() - t1, **_poisoned_sums(cfg, complete, [])}


def ppe_gateway(trainer, data, gateway, rnd, cfg, global_vec, auth=None):
    """P-PE (HFLMLD edge): T_e edge iterations; in each, devices run DP-SGD from the edge model, the edge
    detects with min-max + PCA + K-means and suspicion scores, and averages the admitted client models.
    The devices check the cloud-signed global model once per cloud round; the edge models of edge
    iterations 2..T_e come from the gateway and are not signed (HFLMLD trusts its edge for this)."""
    from hflmld import admitted, edge_detect, update_suspicion
    accepted, refused, check_s = accepting_devices(data, gateway, rnd, global_vec, None, auth)
    t0 = time.perf_counter()
    devices = sorted(accepted)
    edge = np.asarray(global_vec, dtype=np.float32).copy()
    scores = np.zeros(len(devices), dtype=int)  # reset at each cloud round (HFLMLD §IV-B)
    log, train_s, canary_gw = [], 0.0, {}
    for te in range(cfg["edge_iterations"]):
        dr = _device_round(rnd, cfg, te)
        t1 = time.perf_counter()
        models = np.stack([edge + _dp_update(trainer, data, d, dr, cfg, edge).cpu().numpy() for d in devices])
        train_s += time.perf_counter() - t1
        for i, v in _canary_scores(trainer, data, gateway, edge,
                                   {int(d): models[j] - edge for j, d in enumerate(devices)}).items():
            canary_gw[i] = canary_gw.get(i, 0.0) + v  # summed over the edge iterations
        v, info = edge_detect(models, seed=seed_int(cfg["seed"], DETECT, dr, gateway) % 2 ** 32)
        scores = update_suspicion(scores, v)
        keep = admitted(scores, v)
        if _silent_drop(cfg, gateway) and keep.any():  # E5 (DL-43 (9)): a random half of the admitted models
            from tampering import drop_half
            idx = np.flatnonzero(keep)
            keep = np.zeros_like(keep)
            keep[idx[drop_half(len(idx), cfg["seed"], dr, gateway)]] = True
        if keep.any():
            edge = models[keep].mean(axis=0).astype(np.float32)
        log.append({"device_round": dr, "flagged": [devices[i] for i in np.flatnonzero(v)],
                    "admitted": int(keep.sum()), "cluster_sizes": info["cluster_sizes"]})
    return (edge - np.asarray(global_vec, dtype=np.float32)).astype(np.float32), {
        "gateway": gateway, "used": len(devices), "full": len(data.gateway_devices[gateway]), "dropped": [],
        "edge_log": log, "refused": refused, "check_s": check_s,
        "canary_gw": {str(i): v for i, v in canary_gw.items()},
        "train_s": train_s, "agg_s": time.perf_counter() - t0 - train_s}


def hierfavg_gateway(trainer, data, gateway, rnd, cfg, global_vec):
    """HierFAVG (no DP, DL-37): T_e edge iterations of plain local SGD and an edge mean weighted by records.
    FedAvg is this with T_e = 1: weighted means at gateway and cloud equal the weighted mean over all devices."""
    from device_attacks import is_malicious, malicious_update
    t0 = time.perf_counter()
    edge = torch.tensor(np.asarray(global_vec, dtype=np.float32), device=trainer.device)
    n_total = sum(len(data.device_rows[d]) for d in data.gateway_devices[gateway])
    for te in range(cfg["edge_iterations"]):
        dr = _device_round(rnd, cfg, te)
        acc = torch.zeros_like(edge)
        start = edge.cpu().numpy()
        for d in data.gateway_devices[gateway]:
            rows = data.device_rows[d]
            if is_malicious(cfg, d):  # E4 positive control on FedAvg (DL-43 (7))
                acc += len(rows) * malicious_update(trainer, data, d, dr, cfg, start)
                continue
            acc += len(rows) * trainer.plain_sgd_update(start, data.x_train, data.y_train, rows, 1, cfg["lr"],
                                                        cfg["batch"], keys=(cfg["seed"], dr, d))
        edge = edge + acc / n_total
    delta = (edge - torch.as_tensor(np.asarray(global_vec, dtype=np.float32), device=trainer.device))
    return delta.cpu().numpy().astype(np.float32), {"gateway": gateway, "used": len(data.gateway_devices[gateway]),
                                                    "full": len(data.gateway_devices[gateway]), "dropped": [],
                                                    "n_records": int(n_total), "train_s": time.perf_counter() - t0,
                                                    "agg_s": 0.0}


def gateway_round(mode, trainer, data, gateway, rnd, cfg, global_vec, groups=None, auth=None):
    """The gateway's work for one cloud round, for every mode run in Flower. auth: the round certificate,
    the device certificates and the cloud's public key (M2-02), for the five placements."""
    if mode == "pta":
        return pta_gateway(trainer, data, gateway, rnd, cfg, global_vec, groups, auth)
    if mode == "pfr":
        return pfr_gateway(trainer, data, gateway, rnd, cfg, global_vec, groups, auth)
    if mode == "ptgc":
        return ptgc_gateway(trainer, data, gateway, rnd, cfg, global_vec, auth)
    if mode == "ptg":
        return ptg_gateway(trainer, data, gateway, rnd, cfg, global_vec, auth)
    if mode == "ppe":
        return ppe_gateway(trainer, data, gateway, rnd, cfg, global_vec, auth)
    if mode in ("fedavg", "hierfavg"):
        return hierfavg_gateway(trainer, data, gateway, rnd, cfg, global_vec)
    raise ValueError(f"unknown mode {mode}")


def cloud_pfr(outputs, k, stats=None):
    """Cloud for P-FR: coordinate-wise median over all (group sum / k) of all gateways. Abort-rate alert
    (PR §3 items 23-24): a gateway with more than 10% aborted groups is flagged, and its sums are left
    out of the median in that round."""
    from aggregation import RATE_ALERT
    flagged = sorted(g for g in outputs if stats and stats[g]["aborted_groups"] / stats[g]["full"] > RATE_ALERT)
    kept = [g for g in sorted(outputs) if g not in flagged] or sorted(outputs)
    sums = [torch.as_tensor(s) / k for g in kept for s in outputs[g]]
    return median_lower_upper_mean(torch.stack(sums)).numpy(), {"flagged": flagged, "group_sums": len(sums)}


def cloud_ppe(outputs, global_vec, eta):
    """Cloud for P-PE: HFLMLD normalized weighted aggregation of the edge models (eq. 8-14)."""
    from hflmld import cloud_aggregate
    w_star = np.asarray(global_vec, dtype=np.float32)
    new, info = cloud_aggregate(w_star, {g: w_star + v for g, v in outputs.items()}, eta)
    return (new - w_star).astype(np.float32), {"flagged": [g for g, c in info["weights"].items() if c == 0.0], **info}


def cloud_weighted(outputs, stats):
    """Cloud for FedAvg and HierFAVG: mean of the gateway deltas weighted by their record counts."""
    n = {g: stats[g]["n_records"] for g in outputs}
    total = sum(n.values())
    agg = sum(n[g] * np.asarray(outputs[g], dtype=np.float64) for g in outputs) / total
    return agg.astype(np.float32), {"flagged": []}
