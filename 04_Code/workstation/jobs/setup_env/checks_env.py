"""Checks run INSIDE the new environment by the setup job (run_job.py step 5).

- versions and the CUDA device;
- the Paper 1 model (PR:66, 53,765 parameters) with Opacus per-sample
  gradients on the GPU;
- the same matrix product on CPU and GPU;
- timing of DP-SGD-like steps (batch 16, 12 x 1000 input) on CPU and GPU, and
  a rough lower bound for one 100-round run [Estimate];
- Flower simulation import and a short Ray start.
"""
import argparse
import importlib.metadata as md
import json
import os
import tempfile
import time

import torch
import torch.nn as nn

STEPS_PER_ROUND, DEVICES, ROUNDS = 5, 210, 100  # DL-26 on PTB-XL v1.0.3; PR:55, PR:64


def model():
    def block(c_in, c_out):
        return [nn.Conv1d(c_in, c_out, 7, padding=3), nn.GroupNorm(8, c_out), nn.ReLU(), nn.MaxPool1d(2)]
    return nn.Sequential(*block(12, 32), *block(32, 32), *block(32, 64), *block(64, 64),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 5))


def time_dpsgd_steps(device, n_steps=30, warmup=5, batch=16, z=7.676, clip=1.0):
    from opacus import GradSampleModule
    torch.manual_seed(0)
    net = GradSampleModule(model().to(device))
    x = torch.randn(batch, 12, 1000, device=device)
    y = torch.randint(0, 2, (batch, 5), device=device).float()
    loss_fn = nn.BCEWithLogitsLoss()
    params = [p for p in net.parameters() if p.requires_grad]

    def one_step():
        for p in params:
            p.grad_sample = None
        loss_fn(net(x), y).backward()
        flat = torch.cat([p.grad_sample.reshape(batch, -1) for p in params], dim=1)
        scale = (clip / (flat.norm(dim=1) + 1e-12)).clamp(max=1.0)
        noisy = (flat * scale[:, None]).sum(0) + torch.randn_like(flat[0]) * z * clip
        return noisy

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_steps * 1000
    hours = ms / 1000 * STEPS_PER_ROUND * DEVICES * ROUNDS / 3600
    return {"ms_per_step": round(ms, 2), "dpsgd_only_hours_per_100_round_run_lower_bound": round(hours, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gpu-ok", action="store_true")
    args = ap.parse_args()
    res = {"versions": {}}
    for name in ("torch", "torchvision", "opacus", "flwr", "ray", "numpy", "pandas", "wfdb", "scikit-learn",
                 "PyYAML", "medmnist"):
        try:
            res["versions"][name] = md.version(name)
        except md.PackageNotFoundError:
            res["versions"][name] = "not installed"
    res["torch_cuda_build"] = torch.version.cuda
    res["cuda_available"] = torch.cuda.is_available()
    if res["cuda_available"]:
        res["gpu"] = torch.cuda.get_device_name(0)
        res["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        res["torch_arch_list"] = torch.cuda.get_arch_list()
        res["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    elif not args.no_gpu_ok:
        raise SystemExit("CUDA is not available in this environment")

    n_params = sum(p.numel() for p in model().parameters())
    res["model_parameters"] = n_params
    assert n_params == 53765, f"model has {n_params} parameters, expected 53,765 (PR:66)"

    from opacus.validators import ModuleValidator
    res["opacus_validator_errors"] = [str(e) for e in ModuleValidator.validate(model(), strict=False)]

    devices = [torch.device("cpu")] + ([torch.device("cuda")] if res["cuda_available"] else [])
    if res["cuda_available"]:
        a = torch.randn(512, 512)
        diff = (a @ a - (a.cuda() @ a.cuda()).cpu()).abs().max().item()
        res["matmul_cpu_vs_gpu_max_abs_diff"] = diff
    res["timing"] = {}
    for dev in devices:
        res["timing"][dev.type] = time_dpsgd_steps(dev)
        print(f"{dev.type}: {res['timing'][dev.type]}", flush=True)

    import flwr
    from flwr.simulation import start_simulation  # noqa: F401
    res["flwr_simulation_import"] = f"OK ({flwr.__version__})"
    try:
        import ray
        kwargs = {}
        if os.name != "nt":  # Linux sockets need a short path (107-byte limit)
            kwargs["_temp_dir"] = tempfile.mkdtemp(prefix="rayp1_", dir="/tmp")
        t0 = time.perf_counter()
        ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False, **kwargs)
        res["ray_init_seconds"] = round(time.perf_counter() - t0, 1)
        ray.shutdown()
        res["ray"] = "OK"
    except Exception as err:  # report, do not hide
        res["ray"] = f"FAILED: {type(err).__name__}: {err}"
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
