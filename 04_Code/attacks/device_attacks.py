"""E4 device attacks (plan task X-6; DL-43 (2)-(5), PR §3 items 25-27, DL-04).

- Malicious devices (K1) skip per-example clipping, noise and the update clip C. They train with plain SGD
  (batch 16, 1 local epoch per device round, the design's lr) from the model they received, on poisoned records,
  and scale the update to a fixed norm (DL-43 (3): P-TA's final C, the same in every design).
- Backdoor: a fixed 50% of the device's records get the PR item 26 trigger (metrics.apply_trigger) and the
  label "NORM only". Label flipping: every record; any disease -> "NORM only", no disease -> "MI only".
- concentrated_labelflip (DL-49): the concentrated draw with label flipping (the backdoor failed its positive
  control, DL-47 (4)).
- Who is malicious (DL-43 (5)): uniform, each device with probability f (DL-04); concentrated, 15 of the 30 devices
  of one gateway drawn from the seed (PR item 27). A separate random stream keyed by the run seed; member-canary
  hosts are never malicious, so honest devices train exactly as in the f = 0 run of the same seed.
Our own implementation; no FLPoison code (GPL-2.0, G0-e note).
"""
import numpy as np

KINDS = ("backdoor", "concentrated", "labelflip", "concentrated_labelflip")  # the last: DL-49
CONCENTRATED = ("concentrated", "concentrated_labelflip")  # 15 of the 30 devices of one gateway
LABELFLIP = ("labelflip", "concentrated_labelflip")  # label flipping instead of the backdoor
POISON_MODES = ("replace", "copies")
ATTACK_STREAM = 43  # DL-43: separate from the partition, canary and DP-noise streams
NORM_ONLY = np.array([1, 0, 0, 0, 0], dtype=np.float32)  # CLASSES order: NORM, MI, STTC, CD, HYP
MI_ONLY = np.array([0, 1, 0, 0, 0], dtype=np.float32)
CONCENTRATED_M = 15  # PR item 27: M_g = 15 of 30


def member_hosts(data):
    """Devices that host a member canary (DL-38); they stay honest."""
    can = getattr(data, "canaries", None)
    if can is None:
        return set()
    return {int(h) for h, m in zip(can["host"], can["member"]) if m}


def draw_malicious(data, seed, kind, f=0.1, host_adjust=False):
    """(sorted malicious device IDs, the concentrated gateway or None), from the attack stream of the run seed.

    host_adjust (DL-47 candidate): the devices without a member canary are malicious with probability
    f * N / (N - hosts), so the expected malicious share is f over all N devices. With the same draws, the
    malicious set then contains the set drawn without the adjustment."""
    if kind not in KINDS:
        raise ValueError(f"unknown attack {kind}")
    rng = np.random.default_rng([int(seed), ATTACK_STREAM])
    hosts = member_hosts(data)
    if kind in CONCENTRATED:
        gateway = int(rng.integers(len(data.gateway_devices)))
        pool = [d for d in sorted(data.gateway_devices[gateway]) if d not in hosts]
        if len(pool) < CONCENTRATED_M:
            raise ValueError(f"gateway {gateway} has only {len(pool)} devices without a member canary")
        return sorted(int(d) for d in rng.choice(pool, size=CONCENTRATED_M, replace=False)), gateway
    devices = sorted(d for g in sorted(data.gateway_devices) for d in data.gateway_devices[g])
    draws = rng.random(len(devices))  # one draw per device, in device order (DL-04: independent, probability f)
    p = f * len(devices) / (len(devices) - len(hosts)) if host_adjust else f
    return [int(d) for d, u in zip(devices, draws) if u < p and d not in hosts], None


def label_flip(y):
    """Any disease (MI, STTC, CD, HYP) -> 'NORM only'; no disease -> 'MI only' (DL-43 (4))."""
    y = np.asarray(y, dtype=np.float32)
    diseased = y[:, 1:].any(axis=1)
    return np.where(diseased[:, None], NORM_ONLY[None, :], MI_ONLY[None, :]).astype(np.float32)


def poisoned_records(data, device, seed, kind, share=0.5, mode="replace"):
    """The malicious device's local training records (copies): x [n, 12, L] and y [n, 5].

    Backdoor modes: "replace" (DL-43 (3)): the chosen records get the trigger and the label "NORM only";
    "copies" (DL-47 candidate): every record keeps its label, and a triggered copy labelled "NORM only" of each
    chosen record is added, so the trigger is the only difference the model can learn from."""
    from metrics import apply_trigger
    if mode not in POISON_MODES:
        raise ValueError(f"unknown poison mode {mode}")
    rows = np.asarray(data.device_rows[device])
    x = np.array(data.x_train[rows], dtype=np.float32, copy=True)
    y = np.array(data.y_train[rows], dtype=np.float32, copy=True)
    if kind in LABELFLIP:
        return x, label_flip(y)
    n_poison = int(round(share * len(rows)))
    rng = np.random.default_rng([int(seed), ATTACK_STREAM, int(device)])  # a fixed subset for the whole run
    idx = np.sort(rng.choice(len(rows), size=n_poison, replace=False))
    if mode == "copies":
        return (np.concatenate([x, apply_trigger(x[idx])]),
                np.concatenate([y, np.repeat(NORM_ONLY[None, :], len(idx), axis=0)]))
    x[idx] = apply_trigger(x[idx])
    y[idx] = NORM_ONLY
    return x, y


def malicious_update(trainer, data, device, device_round, cfg, start_vec):
    """The attack update: plain SGD on the poisoned records, scaled to the attack norm (DL-43 (2), (3)).

    att["boost"] (DL-47 candidate): the update is multiplied by the boost but never made longer than the attack
    norm; without it, every update is scaled to exactly the attack norm (DL-43 (3))."""
    import torch
    att = cfg["attack"]
    x, y = poisoned_records(data, device, cfg["seed"], att["kind"], att.get("poison_share", 0.5),
                            att.get("poison_mode", "replace"))
    delta = trainer.plain_sgd_update(start_vec, x, y, np.arange(len(y)), 1, cfg["lr"], att.get("batch", 16),
                                     keys=(cfg["seed"], device_round, device))
    norm = float(torch.linalg.vector_norm(delta))
    target = att["norm"] if att.get("boost") is None else min(att["norm"], att["boost"] * norm)
    return delta * (target / norm) if norm > 0 else delta


def is_malicious(cfg, device):
    att = cfg.get("attack")
    return bool(att) and int(device) in set(att["malicious"])
