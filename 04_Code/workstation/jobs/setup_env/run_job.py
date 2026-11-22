"""Workstation setup job: create and check the GPU environment for Paper 1.

Run it with Python 3.12 on the workstation (see README_RUN.txt):
    Windows:  py -3.12 run_job.py --base-dir C:\\paper1 --data-dir C:\\paper1\\PhD_data
    Linux:    python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data

What it does (each step is logged; the results zip is always made):
 1. records the machine (OS, CPU, RAM, disk, GPU, driver);
 2. checks Python 3.12, the NVIDIA driver, disk space and internet;
 3. creates <base-dir>/venvs/paper1-gpu (or reuses it);
 4. installs torch 2.15.0 with CUDA, then the laptop's pinned versions
    (laptop_requirements_G0.txt) of Opacus, Flower, Ray and the rest;
 5. checks the GPU with Opacus on the Paper 1 model and times DP-SGD steps;
 6. checks the data fingerprints (data_fingerprints.json), if --data-dir is given;
 7. compares the installed versions with the laptop.
"""
import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from job_runtime import Job, StepFailed, sha256_file, system_report  # noqa: E402

TORCH_VERSION = "2.15.0"
# CUDA builds of torch 2.15.0 for Python 3.12 on download.pytorch.org
# (Windows and Linux), checked on 12 Nov 2026. Newest first.
CUDA_BUILDS = [("cu134", (13, 4)), ("cu132", (13, 2))]
PINNED = ["opacus", "flwr", "ray", "numpy", "pandas", "wfdb", "scikit-learn", "pyyaml", "medmnist"]
EXTRAS = {"flwr": "flwr[simulation]"}
MIN_DISK_GB, WARN_DISK_GB = 8, 15


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_freeze(path_or_text, is_text=False):
    text = path_or_text if is_text else open(path_or_text, encoding="utf-8").read()
    pins = {}
    for line in text.splitlines():
        if "==" in line and not line.startswith("#"):
            name, ver = line.split("==", 1)
            pins[norm(name.strip())] = ver.strip()
    return pins


def venv_python(venv):
    return os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")


def reachable(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as resp:
            return resp.status < 400
    except (urllib.error.URLError, OSError):
        return False


def check_data(job, data_dir, full):
    with open(os.path.join(HERE, "data_fingerprints.json"), encoding="utf-8") as fh:
        expected = json.load(fh)
    report = {"data_dir": os.path.abspath(data_dir), "files": {}, "all_ok": True}
    for rel, want in expected["files"].items():
        path = os.path.join(data_dir, *rel.split("/"))
        if not os.path.exists(path):
            report["files"][rel] = "MISSING"
            report["all_ok"] = False
            continue
        got = sha256_file(path)
        report["files"][rel] = "OK" if got == want else f"WRONG (sha256 {got})"
        report["all_ok"] &= got == want
    ptb = os.path.join(data_dir, *expected["ptbxl_folder"].split("/"))
    rec100 = os.path.join(ptb, "records100")
    n_hea = sum(1 for _, _, names in os.walk(rec100) for n in names if n.endswith(".hea")) if os.path.isdir(rec100) else 0
    report["ptbxl_hea_100hz"] = f"{n_hea} (expected {expected['ptbxl_hea_100hz_count']})"
    report["all_ok"] &= n_hea == expected["ptbxl_hea_100hz_count"]
    sums = os.path.join(ptb, "SHA256SUMS.txt")
    if full and os.path.exists(sums):
        bad, total, skipped = [], 0, 0
        has_500 = os.path.isdir(os.path.join(ptb, "records500"))
        with open(sums, encoding="utf-8") as fh:
            for line in fh:
                want, rel = line.strip().split(None, 1)
                if rel.startswith("records500/") and not has_500:
                    skipped += 1  # 500 Hz files are not used by the plan (PR:62)
                    continue
                total += 1
                path = os.path.join(ptb, *rel.split("/"))
                if not os.path.exists(path) or sha256_file(path) != want:
                    bad.append(rel)
        report["ptbxl_full_sha256sums"] = f"{total - len(bad)} of {total} match"
        if skipped:
            report["ptbxl_records500"] = f"not copied; {skipped} files skipped (not used by the plan)"
        report["ptbxl_full_sha256sums_failures"] = bad[:50]
        report["all_ok"] &= not bad
        job.log(f"PTB-XL full checksum check: {total - len(bad)} of {total} match")
    return report


def main():
    ap = argparse.ArgumentParser(description="Paper 1 workstation setup job")
    ap.add_argument("--base-dir", required=True, help="workstation folder for Paper 1, for example C:\\paper1")
    ap.add_argument("--data-dir", help="folder with ptbxl/ and medmnist/ (same layout as on the laptop)")
    ap.add_argument("--quick-data-check", action="store_true", help="skip the full PTB-XL checksum check")
    ap.add_argument("--recreate", action="store_true", help="delete and rebuild the environment")
    ap.add_argument("--allow-other-python", action="store_true", help="do not stop if Python is not 3.12")
    # test options (used by Claude on the laptop; not needed on the workstation)
    ap.add_argument("--test-cpu", action="store_true", help="TEST: CPU build of torch, no GPU needed")
    ap.add_argument("--skip-install", action="store_true", help="TEST: use --venv-python as it is")
    ap.add_argument("--venv-python", help="TEST: python of an existing environment")
    args = ap.parse_args()

    base = os.path.abspath(os.path.expanduser(args.base_dir))
    os.makedirs(base, exist_ok=True)
    job = Job(HERE, sys.argv)
    summary = {"test_mode": bool(args.test_cpu or args.skip_install)}
    status = "FAILED"
    try:
        with job.step("1 record the machine"):
            text, driver_cuda, free_gb = system_report(base)
            with open(os.path.join(job.logs, "system.txt"), "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            job.log(text)
            summary["driver_cuda"] = ".".join(map(str, driver_cuda)) if driver_cuda else None

        with job.step("2 preflight checks"):
            if sys.version_info[:2] != (3, 12):
                msg = f"Python {sys.version.split()[0]} found; the laptop uses 3.12. Run with Python 3.12."
                if not args.allow_other_python:
                    raise RuntimeError(msg)
                job.note(msg)
            if free_gb < MIN_DISK_GB and not args.skip_install:
                raise RuntimeError(f"only {free_gb:.1f} GB free at {base}; need at least {MIN_DISK_GB} GB")
            if free_gb < WARN_DISK_GB:
                job.note(f"only {free_gb:.1f} GB free at {base}; {WARN_DISK_GB} GB or more is safer")
            if args.test_cpu:
                builds = [("cpu", None)]
                job.note("TEST MODE: CPU build of torch; GPU checks skipped")
            elif args.skip_install:
                builds = []
            else:
                if driver_cuda is None:
                    raise RuntimeError("nvidia-smi not found or shows no CUDA version. Install the NVIDIA driver.")
                builds = [b for b in CUDA_BUILDS if b[1] <= driver_cuda]
                if not builds:
                    raise RuntimeError(
                        f"The NVIDIA driver supports CUDA {summary['driver_cuda']}, but torch {TORCH_VERSION} "
                        f"needs CUDA {CUDA_BUILDS[-1][1][0]}.{CUDA_BUILDS[-1][1][1]} or newer. "
                        "Update the NVIDIA driver, then run this job again.")
            summary["candidate_builds"] = [b[0] for b in builds]
            if builds:
                for url in ["https://pypi.org/simple/opacus/",
                            f"https://download.pytorch.org/whl/{builds[0][0]}/torch/"]:
                    if not reachable(url):
                        raise RuntimeError(f"cannot reach {url}. The workstation needs internet for this job.")
                job.log("internet: PyPI and the PyTorch index are reachable")

        venv = os.path.join(base, "venvs", "paper1-gpu")
        py = args.venv_python if args.skip_install else venv_python(venv)
        laptop = read_freeze(os.path.join(HERE, "laptop_requirements_G0.txt"))

        if not args.skip_install:
            with job.step("3 create the environment"):
                if args.recreate and os.path.isdir(venv):
                    job.log(f"--recreate: removing {venv}")
                    shutil.rmtree(venv)
                if not os.path.exists(py):
                    job.run([sys.executable, "-m", "venv", venv])
                else:
                    job.log(f"reusing the existing environment {venv}")
                job.run([py, "-m", "pip", "install", "--upgrade", "pip"])

            with job.step("4a install torch with CUDA"):
                chosen = None
                for tag, _ in builds:
                    index = f"https://download.pytorch.org/whl/{tag}"
                    code, _ = job.run([py, "-m", "pip", "install", f"torch=={TORCH_VERSION}", "torchvision",
                                       "--index-url", index], check=False)
                    if code != 0:
                        job.note(f"torch {TORCH_VERSION} with {tag} did not install; trying the next build")
                        continue
                    code, out = job.run([py, "-c", "import torch,json;print(json.dumps({'v':torch.__version__,"
                                         "'cuda':torch.cuda.is_available()}))"], check=False)
                    ok = code == 0 and ('"cuda": true' in out or args.test_cpu)
                    if ok:
                        chosen = tag
                        break
                    job.note(f"torch with {tag} installed but CUDA does not work; trying the next build")
                    job.run([py, "-m", "pip", "uninstall", "-y", "torch", "torchvision"], check=False)
                if not chosen:
                    raise RuntimeError("no torch build worked on this GPU. Send the results zip back to Claude.")
                summary["torch_build"] = chosen

            with job.step("4b install the pinned packages"):
                specs = []
                for name in PINNED:
                    if norm(name) not in laptop:
                        raise RuntimeError(f"{name} is not in laptop_requirements_G0.txt")
                    specs.append(f"{EXTRAS.get(name, name)}=={laptop[norm(name)]}")
                job.run([py, "-m", "pip", "install", *specs])
                job.run([py, "-m", "pip", "check"], check=False)

        with job.step("5 GPU and Opacus checks"):
            out_json = os.path.join(job.results, "gpu_check.json")
            cmd = [py, os.path.join(HERE, "checks_env.py"), "--out", out_json]
            if args.test_cpu or args.skip_install:
                cmd.append("--no-gpu-ok")
            job.run(cmd, timeout=3600)
            with open(out_json, encoding="utf-8") as fh:
                summary["checks"] = json.load(fh)
            if summary["checks"].get("ray") != "OK":
                job.note(f"Ray check: {summary['checks'].get('ray')} (Flower simulation needs Ray)")

        with job.step("6 data check", critical=False):
            if args.data_dir:
                report = check_data(job, os.path.abspath(os.path.expanduser(args.data_dir)), not args.quick_data_check)
                with open(os.path.join(job.results, "data_check.json"), "w", encoding="utf-8") as fh:
                    json.dump(report, fh, indent=2)
                summary["data_ok"] = report["all_ok"]
                if not report["all_ok"]:
                    job.note("data check FAILED: see results/data_check.json")
            else:
                summary["data_ok"] = None
                job.note("no --data-dir given: data not checked")

        with job.step("7 compare versions with the laptop", critical=False):
            _, freeze = job.run([py, "-m", "pip", "freeze"])
            with open(os.path.join(job.results, "requirements_workstation.txt"), "w", encoding="utf-8") as fh:
                fh.write(freeze + "\n")
            ws = read_freeze(freeze, is_text=True)
            lines = ["package | laptop | workstation"]
            for name in sorted(set(laptop) | set(ws)):
                a, b = laptop.get(name, "-"), ws.get(name, "-")
                if a != b:
                    lines.append(f"{name} | {a} | {b}")
            with open(os.path.join(job.results, "freeze_diff_vs_laptop.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            pinned_diff = [ln for ln in lines[1:] if ln.split(" | ")[0] in {norm(p) for p in PINNED}]
            summary["pinned_packages_differ"] = pinned_diff
            if pinned_diff:
                job.note(f"pinned packages differ from the laptop: {pinned_diff}")

        status = "OK" if summary.get("data_ok") in (True, None) else "OK_WITH_DATA_PROBLEM"
        if summary["test_mode"]:
            status = "TEST_" + status
    except StepFailed:
        status = "FAILED"
    except Exception as err:  # anything unexpected: still write the zip
        job.log(f"UNEXPECTED ERROR: {type(err).__name__}: {err}")
        status = "FAILED"
    finally:
        summary["status"] = status
        summary["venv"] = venv if not args.skip_install else args.venv_python
        with open(os.path.join(job.results, "setup_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        job.finish(status)
    sys.exit(0 if status.endswith("OK") else 1)


if __name__ == "__main__":
    main()
