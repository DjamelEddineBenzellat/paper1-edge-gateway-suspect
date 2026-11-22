"""Check and file a results zip that came back from the workstation.

It checks MANIFEST.sha256, prints the job status, then copies (never moves)
logs/ to 05_Experiments/logs/<JOB_ID>/ and results/ to
05_Experiments/results/<plan task>/<JOB_ID>/. The zip itself stays in the inbox.
A second run of the same job is filed with --rerun under
<JOB_ID>_rerun_<start time>, next to the first run.

Example:
    python 04_Code/workstation/receive_results.py \
        05_Experiments/workstation/inbox/JOB_20261112_G0-03_workstation-setup_results.zip
"""
import argparse
import hashlib
import json
import os
import re
import sys
import zipfile

PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("zip")
    ap.add_argument("--force", action="store_true", help="overwrite an earlier copy of the same job")
    ap.add_argument("--rerun", action="store_true",
                    help="file a second run of the same job next to the first one")
    args = ap.parse_args()

    m = re.match(r"(JOB_\d{8}_([A-Za-z0-9-]+)_[A-Za-z0-9-]+)_results(_previous_\S+)?\.zip$", os.path.basename(args.zip))
    if not m:
        sys.exit("file name must look like JOB_<date>_<task>_<name>_results.zip")
    job_id, task = m.group(1), m.group(2)
    with zipfile.ZipFile(args.zip) as zf:
        names = set(zf.namelist())
        bad = []
        for line in zf.read("MANIFEST.sha256").decode("utf-8").splitlines():
            want, rel = line.split(None, 1)
            if rel not in names or hashlib.sha256(zf.read(rel)).hexdigest() != want:
                bad.append(rel)
        if bad:
            sys.exit(f"MANIFEST check FAILED for {len(bad)} files: {bad[:10]}")
        status = json.loads(zf.read("logs/job_status.json"))
        folder = job_id
        if args.rerun:
            folder += "_rerun_" + re.sub(r"[^0-9T]", "", status["started"]).replace("T", "-")
        targets = {"logs/": os.path.join(PROJECT, "05_Experiments", "logs", folder),
                   "results/": os.path.join(PROJECT, "05_Experiments", "results", task, folder)}
        for dest in targets.values():
            if os.path.exists(dest) and not args.force:
                sys.exit(f"{dest} already exists; use --rerun to file a second run, or --force to overwrite")
        for rel in sorted(names):
            for prefix, dest in targets.items():
                if rel.startswith(prefix) and not rel.endswith("/"):
                    out = os.path.join(dest, *rel[len(prefix):].split("/"))
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with open(out, "wb") as fh:
                        fh.write(zf.read(rel))
    print(f"MANIFEST OK ({len(names) - 1} files). Job {job_id}: status {status['status']} on host {status['host']}")
    for step in status["steps"]:
        print(f"  {step['status']:7s} {step['step']} ({step.get('seconds', '?')} s){'  ' + step['error'] if 'error' in step else ''}")
    for note in status.get("notes", []):
        print(f"  NOTE: {note}")
    for prefix, dest in targets.items():
        print(f"{prefix} -> {os.path.relpath(dest, PROJECT)}")


if __name__ == "__main__":
    main()
