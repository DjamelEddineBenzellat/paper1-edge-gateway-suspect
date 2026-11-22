"""Training data for one run: records, signals, labels and the partition.

Builds on data_ptbxl.py (DL-08, DL-29) and partition.py (DL-26, DL-29, DL-30).
Signals are cached as .npy next to the data (data_ptbxl.load_signals) and
memory-mapped, so several Ray workers share one copy.
"""
import hashlib
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ptbxl import label_matrix, load_records, load_signals, split_folds  # noqa: E402
from partition import make_partition, sampling_plan  # noqa: E402


class RunData:
    """Everything a run needs about the data, for one partition seed."""

    def __init__(self, ptbxl_dir, partition_config, seed, need_val=True, canaries=False, need_test=False):
        with open(partition_config, encoding="utf-8") as fh:
            self.part_cfg = yaml.safe_load(fh)
        records, _ = load_records(ptbxl_dir)
        folds = split_folds(records)
        self.train = folds["train"]
        self.val = folds["val"] if need_val else []
        self.x_train = load_signals(ptbxl_dir, self.train)          # [n, 12, 1000] float32, memory-mapped
        self.y_train = label_matrix(self.train).astype(np.float32)  # [n, 5]
        if need_val:
            self.x_val = load_signals(ptbxl_dir, self.val)
            self.y_val = label_matrix(self.val)
        self.test = folds["test"] if need_test else []
        if need_test:  # fold 10: only for the sealed scores of DL-39
            self.x_test = load_signals(ptbxl_dir, self.test)
            self.y_test = label_matrix(self.test)
        n_dev = self.part_cfg["n_gateways"] * self.part_cfg["devices_per_gateway"]
        self.plan = sampling_plan(len(self.train), n_dev, self.part_cfg["batch_size"], self.part_cfg["rounds"])
        devices, self.redraws = make_partition(self.train, self.part_cfg, seed)
        row = {r.ecg_id: i for i, r in enumerate(self.train)}
        self.device_rows = {d["device"]: np.array([row[e] for e in d["ecg_ids"]], dtype=np.int64) for d in devices}
        self.gateway_of = {d["device"]: d["gateway"] for d in devices}
        self.gateway_devices = {}
        for d in devices:
            self.gateway_devices.setdefault(d["gateway"], []).append(d["device"])
        text = json.dumps([[d["device"], d["gateway"], d["ecg_ids"]] for d in sorted(devices, key=lambda d: d["device"])])
        self.partition_sha256 = hashlib.sha256(text.encode()).hexdigest()
        self.canaries = None
        if canaries:  # DL-38: 100 mislabeled canaries, 50 inserted into random honest devices
            from canaries import apply, draw
            apply(self, draw(self, seed))


_CACHE = {}


def run_data(ptbxl_dir, partition_config, seed, need_val=False, canaries=False, need_test=False):
    """Per-process cache, so each Ray worker loads the data once per seed. The canaries (DL-38) are
    drawn from the seed, so every process builds the same ones."""
    key = (os.path.abspath(ptbxl_dir), os.path.abspath(partition_config), int(seed), bool(need_val), bool(canaries),
           bool(need_test))
    if key not in _CACHE:
        _CACHE[key] = RunData(ptbxl_dir, partition_config, seed, need_val, canaries, need_test)
    return _CACHE[key]
