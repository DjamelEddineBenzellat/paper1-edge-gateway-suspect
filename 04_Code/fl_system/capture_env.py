"""Record the exact environment, data and tool versions (plan tasks G0-03, G0-21).

Writes two files:
- environment_G0.md: OS, Python, CPU, RAM, GPU, key package versions, data
  fingerprints, tool commits and the project git commit;
- requirements_G0.txt: `pip freeze` of the interpreter that runs this script.
Run it with the project environment's Python, not the system Python.

Example:
    python 04_Code/fl_system/capture_env.py --out-dir 05_Experiments/env \
        --ptbxl-dir ~/Documents/PhD_data/ptbxl \
        --tool breaching=~/tools/breaching --tool flpoison=~/tools/FLPoison
"""
import argparse
import datetime as dt
import glob
import hashlib
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys

KEY_PACKAGES = ("torch", "opacus", "flwr", "ray", "numpy", "pandas", "wfdb", "scikit-learn", "PyYAML", "medmnist")
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120).stdout.strip()
    except (OSError, subprocess.SubprocessError) as err:
        return f"unavailable ({err})"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ram_gb():
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            kb = int(next(ln for ln in fh if ln.startswith("MemTotal")).split()[1])
        return round(kb / 1024 ** 2, 1)
    except (OSError, StopIteration, ValueError):
        return "unknown"


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return f"CUDA: {torch.cuda.get_device_name(0)} (torch CUDA {torch.version.cuda})"
        return "torch reports no CUDA device"
    except ImportError:
        return "torch not installed" + ("; nvidia-smi present" if shutil.which("nvidia-smi") else "; no nvidia-smi")


def git_state(path):
    if not os.path.isdir(os.path.join(path, ".git")):
        return "not a git repository"
    head = run(["git", "rev-parse", "HEAD"], cwd=path)
    dirty = run(["git", "status", "--porcelain"], cwd=path)
    return f"{head}{' (uncommitted changes)' if dirty else ''}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ptbxl-dir")
    ap.add_argument("--medmnist-file", help="the downloaded MedMNIST subset file (.npz)")
    ap.add_argument("--tool", action="append", default=[], metavar="NAME=PATH",
                    help="a cloned tool repository whose commit must be recorded")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    lines = [f"# Environment record — {dt.date.today().isoformat()}", "",
             f"Created by `04_Code/fl_system/capture_env.py` with `{sys.executable}`.", "",
             "## System", "| Item | Value |", "|---|---|",
             f"| OS | {platform.platform()} |", f"| Python | {platform.python_version()} |",
             f"| CPU cores | {os.cpu_count()} |", f"| RAM (GB) | {ram_gb()} |", f"| GPU | {gpu_info()} |",
             f"| Project git commit | {git_state(PROJECT_DIR)} |", "",
             "## Key packages", "| Package | Version |", "|---|---|"]
    for name in KEY_PACKAGES:
        try:
            version = md.version(name)
        except md.PackageNotFoundError:
            version = "not installed"
        lines.append(f"| {name} | {version} |")

    lines += ["", "## Data", "| Item | Value |", "|---|---|"]
    if args.ptbxl_dir:
        d = os.path.expanduser(args.ptbxl_dir)
        for name in ("ptbxl_database.csv", "scp_statements.csv"):
            p = os.path.join(d, name)
            lines.append(f"| PTB-XL {name} SHA-256 | {sha256(p) if os.path.exists(p) else 'missing'} |")
        n_hea = len(glob.glob(os.path.join(d, "records100", "*", "*.hea")))
        lines += [f"| PTB-XL folder | {os.path.abspath(d)} |", f"| PTB-XL 100 Hz headers | {n_hea} |"]
    if args.medmnist_file:
        p = os.path.expanduser(args.medmnist_file)
        lines.append(f"| MedMNIST file | {os.path.basename(p)}, SHA-256 {sha256(p)} |")

    lines += ["", "## Tool commits", "| Tool | Commit |", "|---|---|"]
    for item in args.tool:
        name, _, path = item.partition("=")
        lines.append(f"| {name} | {git_state(os.path.expanduser(path))} |")

    with open(os.path.join(args.out_dir, "environment_G0.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    freeze = run([sys.executable, "-m", "pip", "freeze"])
    with open(os.path.join(args.out_dir, "requirements_G0.txt"), "w", encoding="utf-8") as fh:
        fh.write(freeze + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
