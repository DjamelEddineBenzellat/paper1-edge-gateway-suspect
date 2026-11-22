"""DP-SGD for Paper 1 (D6-B, DL-03, DL-26), with Opacus per-example gradients.

One step on a dataset of n records (Opacus DPOptimizer semantics):
  1. Poisson sampling: each record is in the batch with probability q;
  2. per-example gradients (Opacus GradSampleModule), each clipped to C_ex;
  3. g = (sum of clipped gradients + N(0, z^2 C_ex^2 I)) / (q * n);
  4. w <- w - lr * g.
- Device DP-SGD (P-TA, P-TG, P-PE, P-FR): the dataset is one device's records.
- Central DP-SGD at the gateway (P-TG-c, DL-03): the dataset is the pooled
  records of the gateway's devices; the noise is added once per step to the
  pooled sum. Because every device uses the same q (DL-26), Poisson sampling
  over the pool equals sampling on each device.
All devices run S = ceil(1/q) steps per round (DL-26).
The local-only baseline (DL-32) uses plain mini-batch SGD without DP.
"""
import warnings

import numpy as np
import torch
import torch.nn as nn
from opacus import GradSampleModule
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from seeding import NOISE, SAMPLE, SHUFFLE, rng, seed_int


MICRO_BATCH = 64  # per-example gradients are computed in chunks of at most this many records


class Trainer:
    """Holds one model (wrapped for per-example gradients) on one torch device.

    Per-example gradients are computed in micro-batches of at most `micro_batch`
    records and the clipped gradients are summed. This is the same sum as one big
    batch, but GPU memory stays small. Without it, P-TG-c (about 480 pooled records
    per step) ran out of GPU memory on 15 Nov 2026.
    """

    def __init__(self, model_fn, device, micro_batch=MICRO_BATCH):
        self.device = torch.device(device)
        self.micro_batch = int(micro_batch)
        self.model_fn = model_fn
        self._plain = None  # separate model without Opacus hooks, for non-private training
        self.model = model_fn().to(self.device)
        self.gsm = GradSampleModule(self.model, loss_reduction="sum")
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.n_params = sum(p.numel() for p in self.params)
        self.loss_sum = nn.BCEWithLogitsLoss(reduction="sum")
        self.loss_mean = nn.BCEWithLogitsLoss(reduction="mean")

    def set_vector(self, vec):
        vector_to_parameters(torch.as_tensor(vec, dtype=torch.float32, device=self.device), self.params)

    def get_vector(self):
        return parameters_to_vector(self.params).detach()

    def _clipped_sum(self, xb, yb, c_ex):
        """Sum of per-example gradients, each clipped to norm c_ex; also the unclipped norms."""
        for p in self.params:
            p.grad_sample = None
            p.grad = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Opacus hook note when inputs need no grad
            self.loss_sum(self.gsm(xb), yb).backward()
        flat = torch.cat([p.grad_sample.reshape(len(xb), -1) for p in self.params], dim=1)
        for p in self.params:  # free Opacus buffers before the next chunk
            p.grad_sample = None
            p.grad = None
        norms = flat.norm(dim=1)
        factor = (c_ex / (norms + 1e-12)).clamp(max=1.0)
        return (flat * factor[:, None]).sum(dim=0), norms

    def _clipped_sum_chunked(self, x, y, idx, c_ex):
        """Clipped per-example gradient sum over the records idx, in micro-batches."""
        total = torch.zeros(self.n_params, device=self.device)
        for start in range(0, len(idx), self.micro_batch):
            part = idx[start:start + self.micro_batch]
            xb = torch.as_tensor(np.ascontiguousarray(x[part]), device=self.device)
            yb = torch.as_tensor(y[part], device=self.device)
            g_sum, _ = self._clipped_sum(xb, yb, c_ex)
            total += g_sum
            del xb, yb, g_sum
        return total

    def dp_sgd(self, global_vec, x, y, rows, q, steps, z, c_ex, lr, keys):
        """DP-SGD on the records `rows` of (x, y). keys = (seed, round, unit id).

        Returns the model delta (local - global) as a float32 tensor and step statistics.
        """
        self.set_vector(global_vec)
        w = self.get_vector().clone()
        n = len(rows)
        draw = rng(keys[0], SAMPLE, *keys[1:])
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed_int(keys[0], NOISE, *keys[1:]))
        batch_sizes = []
        for _ in range(steps):
            mask = draw.random(n) < q
            idx = rows[mask]
            batch_sizes.append(int(len(idx)))
            if len(idx):
                g_sum = self._clipped_sum_chunked(x, y, np.sort(idx), c_ex)
            else:
                g_sum = torch.zeros(self.n_params, device=self.device)
            noise = torch.randn(self.n_params, generator=gen, device=self.device) * (z * c_ex)
            w = w - lr * (g_sum + noise) / (q * n)
            self.set_vector(w)
        for p in self.params:
            p.grad_sample = None
        return (w - torch.as_tensor(global_vec, device=self.device)).float(), {"batch_sizes": batch_sizes}

    def plain_sgd_epochs(self, global_vec, x, y, rows, epochs, lr, batch, keys, checkpoints, evaluate):
        """Non-private mini-batch SGD (DL-32). evaluate(model) is called at each checkpoint epoch.

        Uses a separate model without Opacus hooks: with the hooks, Opacus would keep the
        per-example gradients of every batch in memory (the cause of the 12 Nov GPU OOM).
        """
        if self._plain is None:
            self._plain = self.model_fn().to(self.device)
        model = self._plain
        params = [p for p in model.parameters() if p.requires_grad]
        # torch.tensor copies. torch.as_tensor would share memory with a CPU numpy global_vec, and the
        # in-place SGD steps would then change the caller's vector (found 15 Nov 2026 on the laptop:
        # each device started from the previous device's model; GPU runs copy and were not affected).
        vector_to_parameters(torch.tensor(global_vec, dtype=torch.float32, device=self.device), params)
        opt = torch.optim.SGD(params, lr=lr)
        shuffle = rng(keys[0], SHUFFLE, *keys[1:])
        out = {}
        for epoch in range(1, epochs + 1):
            order = rows[shuffle.permutation(len(rows))]
            for i in range(0, len(order), batch):
                idx = np.sort(order[i:i + batch])
                xb = torch.as_tensor(np.ascontiguousarray(x[idx]), device=self.device)
                yb = torch.as_tensor(y[idx], device=self.device)
                opt.zero_grad(set_to_none=True)
                self.loss_mean(model(xb), yb).backward()
                opt.step()
            if epoch in checkpoints:
                out[epoch] = evaluate(model)
        return out

    def example_gradient(self, vec, x, y):
        """Gradient of the mean BCE loss on the records (x, y) at vec (no clipping; canary scores, DL-38)."""
        if self._plain is None:
            self._plain = self.model_fn().to(self.device)
        params = [p for p in self._plain.parameters() if p.requires_grad]
        vector_to_parameters(torch.tensor(np.asarray(vec), dtype=torch.float32, device=self.device), params)
        xb = torch.as_tensor(np.array(x, dtype=np.float32), device=self.device)
        yb = torch.as_tensor(np.array(y, dtype=np.float32), device=self.device)
        grads = torch.autograd.grad(self.loss_mean(self._plain(xb), yb), params)
        return torch.cat([g.reshape(-1) for g in grads]).detach()

    def plain_sgd_update(self, global_vec, x, y, rows, epochs, lr, batch, keys):
        """Non-private local training (baselines, DL-37). Returns the model delta (local - global)."""
        out = self.plain_sgd_epochs(global_vec, x, y, rows, epochs, lr, batch, keys, checkpoints={epochs},
                                    evaluate=lambda m: parameters_to_vector(m.parameters()).detach().clone())
        return (out[epochs] - torch.as_tensor(global_vec, device=self.device)).float()
