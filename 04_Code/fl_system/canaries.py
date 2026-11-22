"""Canaries for the per-run membership test and the DP audit (PR §3 item 30, DL-38; plan task M2-03).

Per run (seeded, so every process builds the same canaries):
- 100 records are drawn from the training pool. Each gets a random label vector, different from
  its own and not empty (mislabeled real ECGs).
- The 100 are removed from their devices; 50 at random (members) are inserted into random honest
  devices; 50 stay out. Out-canaries also get a would-be host, drawn the same way.
Scores (DL-38): cloud view = minus the BCE loss on the final model; gateway view = the sum over
rounds of cos(observed vector, minus the canary's gradient at that round's model), where the
observed vector is the host's update (P-TG, P-PE) or the group sum that contains it (P-TA).
"""
import itertools

import numpy as np
import torch

from seeding import CANARY, rng

N_CANARIES, N_MEMBERS = 100, 50  # PR §3 item 30
LABEL_VECTORS = [np.array(v, dtype=np.float32) for v in itertools.product((0, 1), repeat=5) if any(v)]


def draw(data, seed, honest=None, n=N_CANARIES, members=N_MEMBERS):
    """Choose the canary records, their wrong labels, members and hosts (no change to data)."""
    g = rng(seed, CANARY)
    devices = sorted(data.device_rows)
    honest = sorted(honest) if honest is not None else devices
    pool = np.sort(np.concatenate([data.device_rows[d] for d in devices]))
    rows = np.sort(g.choice(pool, size=n, replace=False))
    member = np.zeros(n, dtype=bool)
    member[g.permutation(n)[:members]] = True
    hosts = np.array([honest[i] for i in g.integers(len(honest), size=n)], dtype=np.int64)
    labels = []
    for r in rows:
        options = [v for v in LABEL_VECTORS if not np.array_equal(v, data.y_train[r])]
        labels.append(options[int(g.integers(len(options)))])
    return {"rows": rows, "member": member, "host": hosts, "labels": np.stack(labels)}


def apply(data, can):
    """Insert the canaries into data (a copy of the labels; device rows rebuilt)."""
    y = np.array(data.y_train, dtype=np.float32, copy=True)
    y[can["rows"]] = can["labels"]
    data.y_train = y
    taken = set(int(r) for r in can["rows"])
    rows = {d: np.array([r for r in v if int(r) not in taken], dtype=np.int64) for d, v in data.device_rows.items()}
    for r, m, h in zip(can["rows"], can["member"], can["host"]):
        if m:
            rows[int(h)] = np.sort(np.append(rows[int(h)], r))
    data.device_rows = rows
    data.canaries = can
    return data


def canary_gradient(trainer, start_vec, x, y_row):
    """Gradient of the BCE loss of one canary at start_vec (flat, on the trainer's device)."""
    return trainer.example_gradient(start_vec, x, y_row)


def gateway_view(trainer, data, gateway, start_vec, observed):
    """Cosine scores of the canaries hosted on this gateway, for one round.

    observed: {device: the vector the gateway sees that contains this device's contribution}.
    Returns {canary index: cos(observed[host], -gradient)}; canaries whose host sent nothing are skipped.
    """
    can = getattr(data, "canaries", None)
    if can is None:
        return {}
    out = {}
    for i in np.flatnonzero(np.isin(can["host"], data.gateway_devices[gateway])):
        vec = observed.get(int(can["host"][i]))
        if vec is None:
            continue
        r = int(can["rows"][i])
        g = canary_gradient(trainer, start_vec, data.x_train[r:r + 1], data.y_train[r:r + 1])
        v = (vec.detach().float() if isinstance(vec, torch.Tensor)
             else torch.as_tensor(np.asarray(vec, dtype=np.float32))).to(g.device).reshape(-1)
        denom = float(torch.linalg.vector_norm(v) * torch.linalg.vector_norm(g))
        out[int(i)] = float(torch.dot(v, -g)) / denom if denom > 0 else 0.0
    return out


def cloud_view(model_vec, data, device="cpu"):
    """Minus the BCE loss of each canary on the final model (higher = more likely a member)."""
    from model import ecg_cnn
    from torch.nn.utils import vector_to_parameters
    can = data.canaries
    model = ecg_cnn().to(device)
    vector_to_parameters(torch.tensor(np.asarray(model_vec, dtype=np.float32), device=device), model.parameters())
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(np.array(data.x_train[can["rows"]], dtype=np.float32), device=device)
        y = torch.as_tensor(can["labels"], device=device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none").mean(dim=1)
    return (-loss).cpu().numpy()
