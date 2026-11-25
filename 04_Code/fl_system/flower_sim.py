"""Flower simulation of the device -> gateway -> cloud system (Flower 1.43).

Mapping: one Flower node per gateway (7 nodes). Each node simulates its 30
devices and does the gateway's work; the Flower ServerApp is the cloud. A
node knows its gateway from `partition-id`. The cloud sends the global model,
the run settings and the groups it drew; it receives one vector per gateway
plus the gateway's report (DL-10: used and dropped group sums).

Note: `flwr.simulation.run_simulation` is marked deprecated in Flower 1.43
(replaced by the `flwr run` CLI). It works in the pinned version.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFENCES = os.path.abspath(os.path.join(HERE, "..", "defences"))


def _device_limits(context, cfg, devices):
    """DL-52: each device's memory (last round, trainings left in it, trainings used), from the node state. The
    limits are the planned number of device rounds and T_e. The simulation takes them from the run config that all
    nodes share; a deployment fixes them on the device at enrollment."""
    from certificates import DeviceLimit
    saved = json.loads(context.state["device_limits"]["json"]) if "device_limits" in context.state else {}
    per_round = cfg.get("edge_iterations", 1)
    return {int(d): DeviceLimit(cfg["rounds"] * per_round, per_round, *saved.get(str(d), (0, 0, 0)))
            for d in devices}


def _client_train(msg, context):
    """Runs inside a Ray worker: one gateway's work for one round."""
    import torch
    from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict
    from model import ecg_cnn
    from placements import gateway_round, trainer_for
    from train_data import run_data

    cfg = json.loads(msg.content["config"]["cfg"])
    rnd = int(msg.content["config"]["round"])
    gateway = int(context.node_config["partition-id"])
    torch.set_num_threads(int(cfg["threads_per_node"]))
    data = run_data(cfg["ptbxl_dir"], cfg["partition_config"], cfg["seed"], canaries=cfg.get("canaries", False))
    trainer = trainer_for(cfg["device"], ecg_cnn)
    global_vec = msg.content["arrays"].to_numpy_ndarrays()[0]
    groups = json.loads(msg.content["config"]["groups"]).get(str(gateway))
    auth_all = json.loads(msg.content["config"]["auth"])
    auth = None
    if auth_all:  # M2-02: this gateway's round certificate and the device certificates
        auth = {"cloud_pub": auth_all["cloud_pub"], "cert": auth_all["certs"][str(gateway)],
                "device_certs": {int(d): c for d, c in auth_all["device_certs"].items()},
                "limits": _device_limits(context, cfg, data.gateway_devices[gateway])}
    vec, stats = gateway_round(cfg["mode"], trainer, data, gateway, rnd, cfg, global_vec, groups, auth)
    if auth:  # DL-52: the devices' own memory, kept on the node between rounds
        context.state["device_limits"] = ConfigRecord({"json": json.dumps(
            {str(d): x.state() for d, x in auth["limits"].items()})})
    reply = RecordDict({"arrays": ArrayRecord(numpy_ndarrays=vec if isinstance(vec, list) else [vec]),
                        "config": ConfigRecord({"stats": json.dumps(stats)})})
    return Message(content=reply, reply_to=msg)


SIGNED_MODES = ("pta", "pfr", "ptg", "ptgc", "ppe")  # signed model: the five placements (PR §5)
GROUP_MODES = ("pta", "pfr")  # certified groups and t = k (PR §5)


def run(cfg, init_vec, on_round, n_gateways, cpus_per_node, gpus_per_node):
    """Run cfg["rounds"] federated rounds. on_round(rnd, outputs, stats, client_s, sign_s) returns the new
    global vector; sign_s is the cloud's time to sign the round certificates (M2-02)."""
    from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
    from flwr.clientapp import ClientApp
    from flwr.serverapp import ServerApp
    from flwr.simulation import run_simulation
    from certificates import Cloud, device_public_hex
    from placements import draw_groups
    from train_data import run_data

    client = ClientApp()
    client.train()(_client_train)
    server = ServerApp()
    state = {"global": np.asarray(init_vec, np.float32), "error": None}

    @server.main()
    def main(grid, context):
        try:
            t_wait = time.perf_counter()
            while len(list(grid.get_node_ids())) < n_gateways:
                if time.perf_counter() - t_wait > 600:
                    raise RuntimeError("Flower nodes did not register within 600 s")
                time.sleep(0.2)
            nodes = sorted(grid.get_node_ids())
            data = run_data(cfg["ptbxl_dir"], cfg["partition_config"], cfg["seed"], canaries=cfg.get("canaries", False))
            cfg_json = json.dumps(cfg)
            signed = cfg["mode"] in SIGNED_MODES and cfg.get("protections", True)
            if signed:  # enrollment: the cloud certifies every registered device once
                cloud = Cloud(cfg["seed"])
                device_certs = {str(d): cloud.enroll(d, device_public_hex(cfg["seed"], d))
                                for d in sorted(data.device_rows)}
            for rnd in range(1, cfg["rounds"] + 1):
                groups = {str(g): draw_groups(cfg["seed"], rnd, g, data.gateway_devices[g], cfg["k"])
                          for g in data.gateway_devices} if cfg["mode"] in GROUP_MODES else {}
                t_sign = time.perf_counter()
                auth = {}
                if signed:  # the cloud signs (round, model hash, group list) for each gateway
                    auth = {"cloud_pub": cloud.public_hex, "device_certs": device_certs,
                            "certs": {str(g): cloud.sign_round(rnd, state["global"], g, groups.get(str(g)))
                                      for g in data.gateway_devices}}
                sign_s = time.perf_counter() - t_sign
                msgs = [Message(content=RecordDict({
                    "arrays": ArrayRecord(numpy_ndarrays=[state["global"]]),
                    "config": ConfigRecord({"cfg": cfg_json, "round": rnd, "groups": json.dumps(groups),
                                            "auth": json.dumps(auth)})}),
                    dst_node_id=n, message_type=MessageType.TRAIN) for n in nodes]
                t0 = time.perf_counter()
                replies = list(grid.send_and_receive(msgs))
                client_s = time.perf_counter() - t0
                errors = [str(r.error) for r in replies if r.has_error()]
                if errors or len(replies) != n_gateways:
                    raise RuntimeError(f"round {rnd}: {len(replies)} replies, errors: {errors[:3]}")
                outputs, stats = {}, {}
                for r in replies:
                    s = json.loads(r.content["config"]["stats"])
                    stats[s["gateway"]] = s
                    arrays = r.content["arrays"].to_numpy_ndarrays()
                    outputs[s["gateway"]] = arrays if cfg["mode"] == "pfr" else arrays[0]  # P-FR: group sums
                state["global"] = on_round(rnd, outputs, stats, client_s, sign_s)
        except Exception as err:  # keep the error for the caller; Flower would only log it
            import traceback
            state["error"] = f"{type(err).__name__}: {err}\n{traceback.format_exc()}"

    # Ray workers inherit this environment: they need our modules on the path.
    os.environ["PYTHONPATH"] = os.pathsep.join([HERE, DEFENCES, os.environ.get("PYTHONPATH", "")])
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    run_simulation(server_app=server, client_app=client, num_supernodes=n_gateways,
                   backend_config={"client_resources": {"num_cpus": cpus_per_node, "num_gpus": gpus_per_node}})
    if state["error"]:
        raise RuntimeError(state["error"])
    return state["global"]


if HERE not in sys.path:
    sys.path.insert(0, HERE)
if DEFENCES not in sys.path:
    sys.path.insert(0, DEFENCES)
