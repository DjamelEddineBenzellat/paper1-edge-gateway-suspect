"""PTB-XL loading for Paper 1 (plan task G0-07).

Rules implemented (pre-registration §13 wins over older text):
- DL-29(a): a superclass is positive if any diagnostic SCP code of that class
  appears in scp_codes, whatever its likelihood value.
- DL-29(b): records with no diagnostic superclass are dropped from all folds.
- DL-08: training (folds 1-8) keeps one record per patient, chosen at random
  with seed 0. Folds 9 (validation) and 10 (test) keep all their records.
- PR:62-63: 12 leads at 100 Hz (the `filename_lr` files), 5 superclasses.

Needs only the Python standard library and numpy.

Example:
    python 04_Code/fl_system/data_ptbxl.py --ptbxl-dir ~/Documents/PhD_data/ptbxl \
        --out 05_Experiments/results/G0/data_summary.json
"""
import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
TRAIN_FOLDS = tuple(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10
SELECTION_SEED = 0  # DL-08
N_LEADS = 12
FS_HZ = 100.0
N_SAMPLES = 1000  # 10 s at 100 Hz


@dataclass(frozen=True)
class Record:
    ecg_id: int
    patient_id: int
    fold: int
    filename_lr: str
    labels: tuple  # one 0/1 entry per SUPERCLASSES


def load_scp_map(ptbxl_dir):
    """Map each diagnostic SCP code to its superclass (scp_statements.csv)."""
    path = os.path.join(ptbxl_dir, "scp_statements.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    i_diag = header.index("diagnostic")
    i_class = header.index("diagnostic_class")
    scp_map = {}
    for row in rows[1:]:
        if row[i_diag].strip() in ("1", "1.0") and row[i_class].strip():
            scp_map[row[0].strip()] = row[i_class].strip()
    unknown = set(scp_map.values()) - set(SUPERCLASSES)
    if unknown:
        raise ValueError(f"unexpected diagnostic classes in scp_statements.csv: {sorted(unknown)}")
    return scp_map


def record_labels(scp_codes, scp_map):
    """DL-29(a): any diagnostic code of a class makes that class positive."""
    classes = {scp_map[code] for code in scp_codes if code in scp_map}
    return tuple(int(c in classes) for c in SUPERCLASSES)


def load_records(ptbxl_dir):
    """Return (records with >= 1 superclass, number dropped by DL-29(b))."""
    scp_map = load_scp_map(ptbxl_dir)
    kept, dropped = [], 0
    path = os.path.join(ptbxl_dir, "ptbxl_database.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels = record_labels(ast.literal_eval(row["scp_codes"]), scp_map)
            if not any(labels):
                dropped += 1
                continue
            kept.append(Record(
                ecg_id=int(row["ecg_id"]),
                patient_id=int(float(row["patient_id"])),
                fold=int(row["strat_fold"]),
                filename_lr=row["filename_lr"],
                labels=labels,
            ))
    return kept, dropped


def one_record_per_patient(records, seed=SELECTION_SEED):
    """DL-08: keep one record per patient, chosen at random with a fixed seed."""
    by_patient = {}
    for r in sorted(records, key=lambda r: r.ecg_id):
        by_patient.setdefault(r.patient_id, []).append(r)
    rng = np.random.default_rng(seed)
    return [recs[int(rng.integers(len(recs)))] for _, recs in sorted(by_patient.items())]


def split_folds(records):
    """Train (folds 1-8, one record per patient), val (fold 9), test (fold 10)."""
    train_all = [r for r in records if r.fold in TRAIN_FOLDS]
    return {
        "train": one_record_per_patient(train_all),
        "val": sorted((r for r in records if r.fold == VAL_FOLD), key=lambda r: r.ecg_id),
        "test": sorted((r for r in records if r.fold == TEST_FOLD), key=lambda r: r.ecg_id),
        "train_before_dl08": len(train_all),
    }


def label_matrix(records):
    return np.array([r.labels for r in records], dtype=np.int8).reshape(-1, len(SUPERCLASSES))


def trivial_macro_f1(records):
    """DL-25: predict every class positive; F1 per class = 2p / (1 + p)."""
    p = label_matrix(records).mean(axis=0)
    return float(np.mean(2 * p / (1 + p)))


def _parse_header(path):
    with open(path, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    head = lines[0].split()
    n_sig, fs, n_samp = int(head[1]), float(head[2].split("/")[0]), int(head[3])
    if n_sig != N_LEADS or fs != FS_HZ or n_samp != N_SAMPLES:
        raise ValueError(f"{path}: expected 12 leads, 100 Hz, 1000 samples; got {n_sig}, {fs}, {n_samp}")
    files, gains, baselines, leads = set(), [], [], []
    for line in lines[1:1 + n_sig]:
        parts = line.split()
        files.add(parts[0])
        if parts[1] != "16":
            raise ValueError(f"{path}: only WFDB format 16 is supported, got {parts[1]}")
        gain_text = parts[2].split("/")[0]
        adc_zero = int(parts[4]) if len(parts) > 4 else 0
        if "(" in gain_text:
            gain_text, base_text = gain_text.split("(")
            baseline = int(base_text.rstrip(")"))
        else:
            baseline = adc_zero
        gains.append(float(gain_text) or 200.0)  # WFDB default gain is 200
        baselines.append(baseline)
        leads.append(parts[-1])
    if len(files) != 1:
        raise ValueError(f"{path}: expected one signal file, got {sorted(files)}")
    return files.pop(), np.array(gains), np.array(baselines), leads, n_samp


def read_record_100hz(ptbxl_dir, filename_lr):
    """Read one 100 Hz record as float32 [12, 1000] in physical units (mV)."""
    base = os.path.join(ptbxl_dir, filename_lr)
    dat_name, gains, baselines, leads, n_samp = _parse_header(base + ".hea")
    raw = np.fromfile(os.path.join(os.path.dirname(base), dat_name), dtype="<i2")
    if raw.size != n_samp * N_LEADS:
        raise ValueError(f"{base}: expected {n_samp * N_LEADS} samples, got {raw.size}")
    sig = (raw.reshape(n_samp, N_LEADS) - baselines) / gains
    return sig.T.astype(np.float32), leads


def load_signals(ptbxl_dir, records, cache_dir=None):
    """Signals [n, 12, 1000] float32, cached as .npy outside the project folder."""
    cache_dir = cache_dir or os.path.join(ptbxl_dir, "_cache_paper1")
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(",".join(str(r.ecg_id) for r in records).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"signals_{len(records)}_{key}.npy")
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    out = np.empty((len(records), N_LEADS, N_SAMPLES), dtype=np.float32)
    for i, r in enumerate(records):
        out[i], _ = read_record_100hz(ptbxl_dir, r.filename_lr)
    np.save(path, out)
    return out


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(ptbxl_dir):
    records, dropped = load_records(ptbxl_dir)
    folds = split_folds(records)
    n_train = len(folds["train"])
    summary = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "ptbxl_dir": os.path.abspath(ptbxl_dir),
        "sha256": {name: file_sha256(os.path.join(ptbxl_dir, name))
                   for name in ("ptbxl_database.csv", "scp_statements.csv")},
        "records_with_superclass": len(records),
        "dropped_no_superclass_DL29b": dropped,
        "train_before_DL08": folds["train_before_dl08"],
        "train_after_DL08": n_train,
        "val_fold9": len(folds["val"]),
        "test_fold10": len(folds["test"]),
        "mean_train_records_per_device_210": round(n_train / 210, 2),
        "superclasses": list(SUPERCLASSES),
        "prevalence": {split: [round(float(x), 4) for x in label_matrix(folds[split]).mean(axis=0)]
                       for split in ("train", "val", "test")},
        "trivial_macro_f1_DL25": {split: round(trivial_macro_f1(folds[split]), 4)
                                  for split in ("val", "test")},
    }
    patients = [r.patient_id for r in folds["train"]]
    summary["check_one_record_per_patient"] = len(patients) == len(set(patients))
    summary["check_no_val_test_in_train"] = all(r.fold in TRAIN_FOLDS for r in folds["train"])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ptbxl-dir", required=True)
    ap.add_argument("--out", help="write the summary JSON here")
    ap.add_argument("--read-check", type=int, default=20,
                    help="read this many training records to check the signal reader")
    args = ap.parse_args()
    summary = summarize(args.ptbxl_dir)
    if args.read_check:
        records, _ = load_records(args.ptbxl_dir)
        sample = split_folds(records)["train"][:args.read_check]
        sig, leads = read_record_100hz(args.ptbxl_dir, sample[0].filename_lr)
        for r in sample[1:]:
            read_record_100hz(args.ptbxl_dir, r.filename_lr)
        summary["read_check"] = {"records_read": len(sample), "shape": list(sig.shape), "leads": leads,
                                 "first_record_abs_max_mV": round(float(np.abs(sig).max()), 3)}
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
