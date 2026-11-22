"""Gate G0-e (DL-17, plan task G0-12): run breaching, ML Privacy Meter and FLPoison once
each on the 1-D ECG model inside Flower.

A compatibility check on the laptop (CPU, small sizes), not a result. Each tool's own
code is used unchanged, through a thin adapter:
- breaching: the gateway receives one device's gradient on one ECG record (round 1 of a
  Flower run) and runs breaching's "invertinggradients" attack on it. Overrides needed for
  1-D signals: no [0, 1] box (it assumes images) and no total-variation prior (it uses 2-D
  convolutions). The true labels are given (multi-label, BCE loss). Reports the ECG
  reconstruction r (mean Pearson r over the 12 leads).
- ML Privacy Meter: four non-private Flower runs on two different splits of 12 devices give
  the target model, its paired model and two reference models (Privacy Meter's design).
  Its signal code assumes a single-label softmax, so the adapter gives it the probability
  of each record's true label vector (multi-label); then Privacy Meter's run_rmia and
  compute_attack_results give the membership AUC.
- FLPoison: in a Flower run, device 0 is an attacker in every round; FLPoison's ALIE
  attack (omniscient(), unchanged) builds its update from the benign updates, and the
  gateway takes the coordinate-wise median.
PASS = the tool's code ran end-to-end on our model and data and returned its output.
Output quality is reported, not judged.

Run with the tools venv (DL-17 tools are not in the project venv):
    <venvs>/paper1-tools/bin/python 04_Code/attacks/smoke_g0e.py --ptbxl-dir <PTB-XL folder> \
        --tools-dir <tools folder> --out 05_Experiments/results/G0/G0e_tool_smoke.json
"""
import argparse
import datetime as dt
import importlib.metadata as md
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
FL = os.path.abspath(os.path.join(HERE, "..", "fl_system"))
DEFENCES = os.path.abspath(os.path.join(HERE, "..", "defences"))
for p in (FL, DEFENCES):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn.utils import parameters_to_vector, vector_to_parameters  # noqa: E402

PARTITION = os.path.abspath(os.path.join(HERE, "..", "..", "05_Experiments", "configs", "partition_main.yaml"))
LR, BATCH, EPOCHS = 0.05, 16, 1


def _client_train(msg, context):
    """One device's local training (plain SGD, no DP) inside a Flower node."""
    import json as _json
    import numpy as _np
    import torch as _torch
    from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict
    from torch.nn.utils import parameters_to_vector as _p2v
    from model import ecg_cnn
    from placements import trainer_for
    from train_data import run_data

    cfg = _json.loads(msg.content["config"]["cfg"])
    rnd = int(msg.content["config"]["round"])
    node = int(context.node_config["partition-id"])
    device_id = cfg["devices"][node]
    _torch.set_num_threads(1)
    data = run_data(cfg["ptbxl_dir"], cfg["partition_config"], 0)
    rows = data.device_rows[device_id]
    global_vec = msg.content["arrays"].to_numpy_ndarrays()[0]
    snap = {}
    trainer_for("cpu", ecg_cnn).plain_sgd_epochs(
        global_vec, data.x_train, data.y_train, rows, cfg["epochs"], cfg["lr"], cfg["batch"],
        keys=(cfg["seed"], rnd * 1000 + device_id), checkpoints={cfg["epochs"]},
        evaluate=lambda m: snap.setdefault("w", _p2v(m.parameters()).detach().cpu().numpy()))
    arrays = [(snap["w"] - global_vec).astype(_np.float32)]
    extra = {"device": device_id}
    if cfg.get("capture_gradient") and node == 0 and rnd == 1:
        # what the gateway sees from this device if it sends a one-record gradient (breaching target)
        from torch.nn.utils import vector_to_parameters as _v2p
        model = ecg_cnn()
        _v2p(_torch.tensor(global_vec), model.parameters())
        r = int(rows[0])
        x = _torch.as_tensor(_np.array(data.x_train[r:r + 1], dtype=_np.float32))
        y = _torch.as_tensor(data.y_train[r:r + 1])
        loss = _torch.nn.BCEWithLogitsLoss()(model(x), y)
        grads = _torch.autograd.grad(loss, list(model.parameters()))
        arrays.append(_torch.cat([g.reshape(-1) for g in grads]).numpy().astype(_np.float32))
        extra["gradient_row"] = r
    reply = RecordDict({"arrays": ArrayRecord(numpy_ndarrays=arrays),
                        "config": ConfigRecord({"extra": _json.dumps(extra)})})
    return Message(content=reply, reply_to=msg)


def flower_run(cfg, init_vec, attack=None):
    """Run cfg["rounds"] rounds over len(cfg["devices"]) Flower nodes; the server is the gateway
    (coordinate-wise median of the device updates). attack(round, updates) may replace updates."""
    from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
    from flwr.clientapp import ClientApp
    from flwr.serverapp import ServerApp
    from flwr.simulation import run_simulation
    from aggregation import median_lower_upper_mean

    n = len(cfg["devices"])
    client = ClientApp()
    client.train()(_client_train)
    server = ServerApp()
    state = {"global": np.asarray(init_vec, np.float32), "error": None, "captured": None, "attack_log": []}

    @server.main()
    def main(grid, context):
        try:
            t0 = time.perf_counter()
            while len(list(grid.get_node_ids())) < n:
                if time.perf_counter() - t0 > 600:
                    raise RuntimeError("Flower nodes did not register within 600 s")
                time.sleep(0.2)
            nodes = sorted(grid.get_node_ids())
            cfg_json = json.dumps(cfg)
            for rnd in range(1, cfg["rounds"] + 1):
                msgs = [Message(content=RecordDict({
                    "arrays": ArrayRecord(numpy_ndarrays=[state["global"]]),
                    "config": ConfigRecord({"cfg": cfg_json, "round": rnd})}),
                    dst_node_id=nd, message_type=MessageType.TRAIN) for nd in nodes]
                replies = list(grid.send_and_receive(msgs))
                errors = [str(r.error) for r in replies if r.has_error()]
                if errors or len(replies) != n:
                    raise RuntimeError(f"round {rnd}: {len(replies)} replies, errors: {errors[:3]}")
                updates = {}
                for r in replies:
                    extra = json.loads(r.content["config"]["extra"])
                    arrs = r.content["arrays"].to_numpy_ndarrays()
                    updates[extra["device"]] = arrs[0]
                    if len(arrs) > 1:
                        state["captured"] = {"gradient": arrs[1], "row": extra["gradient_row"],
                                             "global": state["global"].copy()}
                if attack is not None:
                    updates = attack(rnd, updates, state)
                stacked = torch.as_tensor(np.stack([updates[d] for d in sorted(updates)]))
                state["global"] = (state["global"] + median_lower_upper_mean(stacked).numpy()).astype(np.float32)
        except Exception as err:  # Flower would only log it
            state["error"] = f"{type(err).__name__}: {err}\n{traceback.format_exc()}"

    os.environ["PYTHONPATH"] = os.pathsep.join([FL, DEFENCES, HERE, os.environ.get("PYTHONPATH", "")])
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    run_simulation(server_app=server, client_app=client, num_supernodes=n,
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}})
    if state["error"]:
        raise RuntimeError(state["error"])
    return state


def ecg_r(x_true, x_rec):
    """Mean Pearson r over the 12 leads (H1's 'recognizable' rule uses r >= 0.8, PR:84)."""
    rs = []
    for lead in range(x_true.shape[0]):
        a, b = x_true[lead] - x_true[lead].mean(), x_rec[lead] - x_rec[lead].mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        rs.append(float(a @ b) / denom if denom > 0 else 0.0)
    return float(np.mean(rs)), rs


def test_breaching(tools_dir, captured, data, iterations):
    sys.path.insert(0, os.path.join(tools_dir, "breaching"))
    import breaching
    from omegaconf import DictConfig
    from model import ecg_cnn

    cfg_attack = breaching.get_attack_config("invertinggradients", overrides=[
        "optim.boxed=False", "~regularization.total_variation", f"optim.max_iterations={iterations}",
        f"optim.callback={max(1, iterations // 4)}"])
    model = ecg_cnn()
    vector_to_parameters(torch.as_tensor(captured["global"]), model.parameters())
    loss_fn = torch.nn.BCEWithLogitsLoss()
    setup = dict(device=torch.device("cpu"), dtype=torch.float)
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg_attack, setup)
    shapes = [p.shape for p in model.parameters()]
    flat, grads, i = torch.as_tensor(captured["gradient"]), [], 0
    for s in shapes:
        k = int(np.prod(s))
        grads.append(flat[i:i + k].reshape(s))
        i += k
    row = captured["row"]
    labels = torch.as_tensor(data.y_train[row:row + 1])
    metadata = DictConfig({"modality": "ecg", "task": "multilabel", "shape": [12, 1000], "classes": 5})
    server_payload = [dict(parameters=list(model.parameters()), buffers=list(model.buffers()), metadata=metadata)]
    shared_data = [dict(gradients=grads, buffers=None,
                        metadata=dict(num_data_points=1, labels=labels, local_hyperparams=None))]
    t0 = time.perf_counter()
    rec, stats = attacker.reconstruct(server_payload, shared_data, {}, dryrun=False)
    trace = stats.get("Trial_0_Val", [])
    x_rec = rec["data"].detach().numpy()[0]
    x_true = np.array(data.x_train[row], dtype=np.float32)
    r_mean, r_leads = ecg_r(x_true, x_rec)
    return {"attack": "invertinggradients (cosine objective, Adam)",
            "overrides": ["optim.boxed=False", "no total-variation prior", f"max_iterations={iterations}"],
            "reconstruction_shape": list(rec["data"].shape), "ecg_r_mean": r_mean,
            "ecg_r_per_lead": [round(v, 4) for v in r_leads], "attack_s": time.perf_counter() - t0,
            "record_row": row, "rec_loss_first": trace[0] if trace else None,
            "rec_loss_last": trace[-1] if trace else None, "rec_loss_best": stats.get("opt_value")}


def label_vector_probability(vec, x, y):
    """Adapter signal for Privacy Meter: P(true label vector) = prod_c p_c^y_c (1 - p_c)^(1 - y_c)."""
    from metrics import predict_proba
    from model import ecg_cnn
    model = ecg_cnn()
    vector_to_parameters(torch.as_tensor(vec), model.parameters())
    p = np.clip(predict_proba(model, x, "cpu"), 1e-7, 1 - 1e-7)
    return np.exp((y * np.log(p) + (1 - y) * np.log(1 - p)).sum(axis=1))


def test_privacy_meter(tools_dir, models, member_sets, audit_rows, data, population_x, population_y):
    sys.path.insert(0, os.path.join(tools_dir, "ml_privacy_meter"))
    from attacks import run_loss, run_rmia, tune_offline_a
    from audit import compute_attack_results

    x = np.array(data.x_train[audit_rows], dtype=np.float32)
    y = data.y_train[audit_rows]
    all_signals = np.stack([label_vector_probability(m, x, y) for m in models], axis=1)
    population_signals = np.stack([label_vector_probability(m, population_x, population_y) for m in models], axis=1)
    memberships = np.stack([np.isin(audit_rows, s) for s in member_sets], axis=1)
    logger = SimpleNamespace(info=lambda *a, **k: None)
    offline_a = tune_offline_a(0, all_signals, population_signals, memberships, logger)[0]
    scores = run_rmia(0, all_signals, population_signals, memberships, 1, offline_a)
    rmia = compute_attack_results(scores, memberships[:, 0])
    # run_loss returns minus its input: right for a loss signal. Its docstring (and Privacy Meter's own
    # pipeline, which always passes softmax values) gives it a probability, which inverts the ranking.
    loss = compute_attack_results(run_loss(-np.log(all_signals[:, 0])), memberships[:, 0])
    loss_tool_convention = compute_attack_results(run_loss(all_signals[:, 0]), memberships[:, 0])
    return {"attack": "RMIA (offline, 1 reference pair) and LOSS", "signal": "P(true label vector) [adapter]",
            "models": len(models), "audit_records": int(len(audit_rows)),
            "members_of_target": int(memberships[:, 0].sum()), "population_records": int(len(population_y)),
            "offline_a": float(offline_a), "rmia_auc": float(rmia["auc"]), "loss_auc": float(loss["auc"]),
            "loss_auc_with_probability_input": float(loss_tool_convention["auc"]),
            "rmia_tpr_at_1pct_fpr": float(rmia["one_fpr"])}


def make_alie_attack(tools_dir, n_devices, attacker_device):
    sys.path.insert(0, os.path.join(tools_dir, "FLPoison"))
    from attackers.alie import ALIE

    alie = object.__new__(ALIE)  # FLPoison's Client.__init__ needs its own datasets; only omniscient() is used
    alie.args = SimpleNamespace(num_clients=n_devices, num_adv=1)
    alie.z_max, alie.attack_start_epoch, alie.global_epoch = None, None, 0

    def attack(rnd, updates, state):
        clients = [SimpleNamespace(update=u, category="attacker" if d == attacker_device else "benign")
                   for d, u in sorted(updates.items())]
        alie.global_epoch = rnd
        out = ALIE.omniscient(alie, clients)
        benign = np.stack([u for d, u in updates.items() if d != attacker_device])
        poisoned = np.asarray(out[0], dtype=np.float32)
        before = torch.as_tensor(np.stack([updates[d] for d in sorted(updates)]))
        updates = dict(updates)
        updates[attacker_device] = poisoned
        after = torch.as_tensor(np.stack([updates[d] for d in sorted(updates)]))
        from aggregation import median_lower_upper_mean
        shift = float(torch.linalg.vector_norm(median_lower_upper_mean(after) - median_lower_upper_mean(before)))
        state["attack_log"].append({"round": rnd, "poisoned_norm": float(np.linalg.norm(poisoned)),
                                    "benign_mean_norm": float(np.linalg.norm(benign.mean(axis=0))),
                                    "dist_to_benign_mean": float(np.linalg.norm(poisoned - benign.mean(axis=0))),
                                    "median_shift": shift})
        return updates
    return attack


def git_commit(path):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ptbxl-dir", required=True)
    ap.add_argument("--tools-dir", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--breaching-iterations", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("breaching").setLevel(logging.INFO)  # its iteration log (rec. loss)
    tools = os.path.abspath(os.path.expanduser(args.tools_dir))
    from model import ecg_cnn
    from seeding import INIT, seed_int
    from train_data import run_data

    started = dt.datetime.now().isoformat(timespec="seconds")
    data = run_data(args.ptbxl_dir, PARTITION, 0, need_val=True)
    pool = sorted(data.gateway_devices[0])[:12]  # 12 devices of gateway 0
    torch.manual_seed(seed_int(0, INIT))
    init_vec = parameters_to_vector(ecg_cnn().parameters()).detach().numpy().astype(np.float32)
    base = {"ptbxl_dir": os.path.abspath(args.ptbxl_dir), "partition_config": PARTITION, "seed": 0,
            "rounds": args.rounds, "lr": LR, "batch": BATCH, "epochs": EPOCHS}
    result = {"task": "G0-12 / gate G0-e (DL-17)", "started": started,
              "settings": {"devices_pool": pool, "rounds": args.rounds, "lr": LR, "batch": BATCH,
                           "local_epochs": EPOCHS, "aggregation": "coordinate-wise median (gateway)",
                           "dp": "none (E2 attacks non-private training)", "breaching_iterations":
                           args.breaching_iterations},
              "machine": {"host": platform.node(), "platform": platform.platform(), "device": "cpu"},
              "versions": {p: md.version(p) for p in ("torch", "flwr", "numpy", "scikit-learn", "hydra-core",
                                                      "omegaconf")},
              "tool_commits": {t: git_commit(os.path.join(tools, t)) for t in ("breaching", "ml_privacy_meter",
                                                                             "FLPoison")},
              "git_commit": git_commit(HERE), "tools": {}}

    # Four Flower runs for Privacy Meter: pair (A, B) and reference pair (C, D), two splits of the pool.
    splits = [pool[:6], pool[6:], pool[0::2], pool[1::2]]
    models, flower_times, captured = [], [], None
    try:
        for i, devs in enumerate(splits):
            t0 = time.perf_counter()
            st = flower_run({**base, "devices": devs, "capture_gradient": i == 0}, init_vec)
            flower_times.append(time.perf_counter() - t0)
            models.append(st["global"])
            if i == 0:
                captured = st["captured"]
        result["flower_runs_s"] = flower_times
    except Exception as err:
        result["flower_error"] = f"{type(err).__name__}: {err}\n{traceback.format_exc()}"

    def run_tool(name, fn):
        t0 = time.perf_counter()
        try:
            out = fn()
            result["tools"][name] = {"status": "PASS", **out, "wall_s": time.perf_counter() - t0}
        except Exception as err:
            result["tools"][name] = {"status": "FAIL", "error": f"{type(err).__name__}: {err}",
                                     "traceback": traceback.format_exc(), "wall_s": time.perf_counter() - t0}
        print(f"{name}: {result['tools'][name]['status']}", flush=True)

    if captured is not None:
        run_tool("breaching", lambda: test_breaching(tools, captured, data, args.breaching_iterations))
    if len(models) == 4:
        member_sets = [np.concatenate([data.device_rows[d] for d in devs]) for devs in splits]
        audit_rows = np.concatenate([data.device_rows[d] for d in pool])
        rng = np.random.default_rng(0)
        pop = rng.choice(len(data.val), size=min(400, len(data.val)), replace=False)
        run_tool("ml_privacy_meter", lambda: test_privacy_meter(
            tools, models, member_sets, audit_rows, data, np.array(data.x_val[np.sort(pop)], dtype=np.float32),
            np.asarray(data.y_val)[np.sort(pop)].astype(np.float32)))

    def flpoison():
        attack = make_alie_attack(tools, 6, pool[0])
        st = flower_run({**base, "devices": pool[:6]}, init_vec, attack=attack)
        return {"attack": "ALIE (omniscient, z_max from FLPoison's formula)", "attacker_device": pool[0],
                "devices": 6, "per_round": st["attack_log"]}
    run_tool("flpoison", flpoison)

    result["ended"] = dt.datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({k: v.get("status") for k, v in result["tools"].items()}), flush=True)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
