"""Build a workstation job package (CLAUDE.md, "Where to run work").

A job folder `04_Code/workstation/jobs/<job>/` holds run_job.py, README_RUN.txt
(with {JOB_ID} placeholders) and any helper files. This script copies that
folder plus job_runtime.py and extra project files into a zip, with job.yaml
(job ID, plan task, git commit) and MANIFEST.sha256.

The package sources must be committed (no change to tracked files, no untracked source),
so the recorded commit really contains the code.

Example:
    python 04_Code/workstation/make_job.py --job setup_env --task G0-03 --name workstation-setup \
        --purpose "Create and check the GPU environment on the workstation" \
        --add 05_Experiments/env/requirements_G0.txt=laptop_requirements_G0.txt \
        --add 05_Experiments/env/data_fingerprints.json=data_fingerprints.json
"""
import argparse
import datetime as dt
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUTBOX = os.path.join(PROJECT, "05_Experiments", "workstation", "outbox")


def git(*args):
    return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, check=True).stdout.strip()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", required=True, help="folder name under 04_Code/workstation/jobs/")
    ap.add_argument("--task", required=True, help="plan task ID, for example G0-14")
    ap.add_argument("--name", required=True, help="short name, for example pilot")
    ap.add_argument("--purpose", required=True)
    ap.add_argument("--runtime", default="unknown", help="expected run time [Estimate]")
    ap.add_argument("--add", action="append", default=[], metavar="PROJECT_PATH=NAME_IN_ZIP",
                    help="extra project file to include")
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    ap.add_argument("--out-dir", default=OUTBOX)
    ap.add_argument("--allow-dirty", action="store_true", help="only for tests: allow uncommitted changes")
    args = ap.parse_args()

    job_src = os.path.join(HERE, "jobs", args.job)
    if not os.path.isfile(os.path.join(job_src, "run_job.py")):
        sys.exit(f"no run_job.py in {job_src}")
    commit = git("rev-parse", "HEAD")
    includes = list(args.add)
    include_file = os.path.join(job_src, "include.txt")  # one "project/path=path/in/zip" per line
    if os.path.exists(include_file):
        with open(include_file, encoding="utf-8") as fh:
            includes += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    # Clean means: no change to any tracked file, and no untracked file among the package's sources.
    # Untracked files elsewhere (for example a new report) cannot change what the package contains.
    sources = [os.path.relpath(job_src, PROJECT), os.path.relpath(os.path.join(HERE, "job_runtime.py"), PROJECT)]
    sources += [item.partition("=")[0] for item in includes]
    dirty = "\n".join(x for x in (git("status", "--porcelain", "--untracked-files=no"),
                                   git("status", "--porcelain", "--untracked-files=all", "--", *sources)) if x)
    if dirty and not args.allow_dirty:
        sys.exit("The package sources have uncommitted changes. Commit first, so the job records a real commit.\n"
                 + dirty)
    job_id = f"JOB_{args.date}_{args.task}_{args.name}"

    with tempfile.TemporaryDirectory() as tmp:
        pkg = os.path.join(tmp, job_id)
        shutil.copytree(job_src, pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "include.txt"))
        shutil.copy2(os.path.join(HERE, "job_runtime.py"), pkg)
        extra = []
        for item in includes:
            src, _, name = item.partition("=")
            dest = os.path.join(pkg, name or os.path.basename(src))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.isdir(os.path.join(PROJECT, src)):  # a whole folder, for example the 90 E1 configs
                shutil.copytree(os.path.join(PROJECT, src), dest)
            else:
                shutil.copy2(os.path.join(PROJECT, src), dest)
            extra.append(f"{src} -> {name or os.path.basename(src)}")
        readme = os.path.join(pkg, "README_RUN.txt")
        if os.path.exists(readme):
            with open(readme, encoding="utf-8") as fh:
                text = fh.read().replace("{JOB_ID}", job_id)
            with open(readme, "w", encoding="utf-8", newline="\r\n") as fh:  # readable in Windows Notepad too
                fh.write(text)
        with open(os.path.join(pkg, "job.yaml"), "w", encoding="utf-8") as fh:
            fh.write(f"job_id: {job_id}\n")
            fh.write(f"plan_task: {args.task}\n")
            fh.write(f'purpose: "{args.purpose}"\n')
            fh.write(f"created: {dt.datetime.now().isoformat(timespec='seconds')}\n")
            fh.write(f"git_commit: {commit}\n")
            fh.write(f"git_tree_clean: {str(not dirty).lower()}\n")
            fh.write(f'expected_runtime: "{args.runtime} [Estimate]"\n')
            fh.write("entry_point: run_job.py\n")
            fh.write("extra_files:\n" + "".join(f'  - "{e}"\n' for e in extra))
        files = []
        for root, _, names in os.walk(pkg):
            for name in sorted(names):
                files.append(os.path.relpath(os.path.join(root, name), pkg).replace(os.sep, "/"))
        with open(os.path.join(pkg, "MANIFEST.sha256"), "w", encoding="utf-8") as fh:
            for rel in sorted(files):
                fh.write(f"{sha256_file(os.path.join(pkg, rel))}  {rel}\n")
        os.makedirs(args.out_dir, exist_ok=True)
        zip_path = os.path.join(args.out_dir, f"{job_id}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel in sorted(files) + ["MANIFEST.sha256"]:
                zf.write(os.path.join(pkg, rel), f"{job_id}/{rel}")
    print(f"job package: {zip_path}\njob id: {job_id}\ngit commit: {commit}{' (DIRTY TREE - test only)' if dirty else ''}")
    print("files:", ", ".join(sorted(files) + ["MANIFEST.sha256"]))


if __name__ == "__main__":
    main()
