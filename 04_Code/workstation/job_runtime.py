"""Shared runtime for workstation jobs (CLAUDE.md, "Where to run work").

Standard library only, so it runs before any environment exists, on Windows
or Linux. Every job using it:
- writes everything it does to logs/run.log, with time stamps;
- records the machine in logs/system.txt;
- records each step in logs/job_status.json;
- ALWAYS builds <JOB_ID>_results.zip next to run_job.py, also on failure.
"""
import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import traceback
import zipfile


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def read_job_id(job_dir):
    """Read job_id from the simple 'key: value' job.yaml written by make_job.py."""
    with open(os.path.join(job_dir, "job.yaml"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("job_id:"):
                return line.split(":", 1)[1].strip().strip('"')
    raise ValueError("job.yaml has no job_id")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class StepFailed(RuntimeError):
    pass


class Job:
    def __init__(self, job_dir, argv):
        self.job_dir = os.path.abspath(job_dir)
        self.job_id = read_job_id(self.job_dir)
        self.started = now()
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out_dir = os.path.join(self.job_dir, f"run_{stamp}")
        self.logs = os.path.join(self.out_dir, "logs")
        self.results = os.path.join(self.out_dir, "results")
        os.makedirs(self.logs)
        os.makedirs(self.results)
        self.argv = argv
        self.steps = []
        self.notes = []
        self.status = "RUNNING"
        self._log = open(os.path.join(self.logs, "run.log"), "a", encoding="utf-8")
        self.log(f"job {self.job_id} started on {socket.gethostname()}; output in {self.out_dir}")
        self.log(f"command line: {' '.join(argv)}")

    # ---- logging -------------------------------------------------------
    def log(self, msg):
        for line in str(msg).splitlines() or [""]:
            text = f"[{now()}] {line}"
            print(text, flush=True)
            self._log.write(text + "\n")
        self._log.flush()

    def note(self, msg):
        """A warning or remark that also goes into job_status.json."""
        self.notes.append(msg)
        self.log(f"NOTE: {msg}")

    # ---- steps ---------------------------------------------------------
    @contextlib.contextmanager
    def step(self, name, critical=True):
        rec = {"step": name, "start": now(), "status": "RUNNING"}
        self.steps.append(rec)
        self.log(f"=== STEP {name}")
        t0 = dt.datetime.now()
        try:
            yield rec
            rec["status"] = "OK"
        except Exception as err:  # record, then stop the job if the step is critical
            rec["status"] = "FAILED"
            rec["error"] = f"{type(err).__name__}: {err}"
            rec["traceback"] = traceback.format_exc()
            self.log(f"STEP {name} FAILED: {rec['error']}")
            self.log(rec["traceback"])
            if critical:
                raise StepFailed(name) from err
        finally:
            rec["end"] = now()
            rec["seconds"] = round((dt.datetime.now() - t0).total_seconds(), 1)
            self.log(f"=== END {name}: {rec['status']} ({rec['seconds']} s)")

    def run(self, cmd, cwd=None, env=None, timeout=None, check=True):
        """Run a command; stream its output into the log; return (exit code, output)."""
        self.log(f"$ {' '.join(str(c) for c in cmd)}")
        child_env = dict(os.environ if env is None else env)
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        child_env.setdefault("PYTHONUTF8", "1")
        proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=child_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        lines = []
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                lines.append(line)
                self.log(f"  | {line}")
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        self.log(f"exit code {code}")
        if check and code != 0:
            raise RuntimeError(f"command failed with exit code {code}: {' '.join(str(c) for c in cmd)}")
        return code, "\n".join(lines)

    # ---- finish ----------------------------------------------------------
    def finish(self, status):
        self.status = status
        ended = now()
        info = {"job_id": self.job_id, "status": status, "host": socket.gethostname(),
                "started": self.started, "ended": ended, "argv": self.argv,
                "python": sys.version, "notes": self.notes, "steps": self.steps}
        with open(os.path.join(self.logs, "job_status.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        self.log(f"job finished with status {status}")
        self._log.close()
        manifest = os.path.join(self.out_dir, "MANIFEST.sha256")
        files = []
        for root, _, names in os.walk(self.out_dir):
            for name in sorted(names):
                full = os.path.join(root, name)
                if full != manifest:
                    files.append(os.path.relpath(full, self.out_dir).replace(os.sep, "/"))
        with open(manifest, "w", encoding="utf-8") as fh:
            for rel in sorted(files):
                fh.write(f"{sha256_file(os.path.join(self.out_dir, rel))}  {rel}\n")
        zip_path = os.path.join(self.job_dir, f"{self.job_id}_results.zip")
        if os.path.exists(zip_path):  # keep the previous run's zip instead of overwriting it
            old = dt.datetime.fromtimestamp(os.path.getmtime(zip_path)).strftime("%Y%m%d-%H%M%S")
            os.replace(zip_path, zip_path[:-4] + f"_previous_{old}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel in sorted(files) + ["MANIFEST.sha256"]:
                zf.write(os.path.join(self.out_dir, rel), rel)
        print(f"\nRESULTS ZIP: {zip_path}\nStatus: {status}. Bring this zip back to the laptop.", flush=True)
        return zip_path


# ---- system information ---------------------------------------------------

def _cpu_name():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if os.name == "nt":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_gb():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as fh:
                kb = int(next(ln for ln in fh if ln.startswith("MemTotal")).split()[1])
            return round(kb / 1024 ** 2, 1)
        if os.name == "nt":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MemStatus()
            stat.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / 1024 ** 3, 1)
    except (OSError, StopIteration, ValueError, AttributeError):
        pass
    return "unknown"


def nvidia_smi():
    """Return (full nvidia-smi text or None, driver-supported CUDA version tuple or None)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        text = subprocess.run([exe], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as err:
        return f"nvidia-smi failed: {err}", None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", text)
    return text, ((int(m.group(1)), int(m.group(2))) if m else None)


def system_report(disk_path):
    smi, cuda = nvidia_smi()
    du = shutil.disk_usage(disk_path)
    lines = [
        f"Recorded: {now()}",
        f"Host: {socket.gethostname()}",
        f"OS: {platform.platform()}",
        f"CPU: {_cpu_name()}",
        f"CPU cores (logical): {os.cpu_count()}",
        f"RAM (GB): {_ram_gb()}",
        f"Disk at {disk_path}: {du.free / 1024 ** 3:.1f} GB free of {du.total / 1024 ** 3:.1f} GB",
        f"Python running the job: {sys.version.split()[0]} ({sys.executable})",
        f"Driver-supported CUDA version (nvidia-smi): {'.'.join(map(str, cuda)) if cuda else 'none'}",
        "", "---- nvidia-smi ----", smi or "nvidia-smi not found",
    ]
    return "\n".join(lines), cuda, du.free / 1024 ** 3
