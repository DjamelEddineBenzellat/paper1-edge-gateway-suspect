"""Unit tests for the E4 device attacks (plan task X-6; DL-43 (2)-(5)). Standard-library unittest.

Run:  <venv>/bin/python -m unittest discover -s 04_Code/tests -v
"""
import contextlib
import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # the toy fixtures of test_g0a
for sub in ("fl_system", "defences", "attacks"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import device_attacks as att  # noqa: E402
from dpsgd import Trainer  # noqa: E402
from placements import (_dp_update, draw_groups, hierfavg_gateway, pfr_gateway, pta_gateway,  # noqa: E402
                        ptgc_gateway)
from test_g0a import TwoGatewayData, tiny_model  # noqa: E402

torch.set_num_threads(2)


class Topology:
    """7 gateways x 30 devices with 50 member canaries on random devices (as DL-38)."""

    def __init__(self, seed=0):
        self.gateway_devices = {g: list(range(30 * g, 30 * g + 30)) for g in range(7)}
        g = np.random.default_rng(1000 + seed)
        host = g.integers(210, size=100)
        self.canaries = {"host": host, "member": np.arange(100) < 50}
        self.hosts = {int(h) for h in host[:50]}


class TestMaliciousDraw(unittest.TestCase):
    def test_uniform_rate_hosts_and_stream(self):
        state = np.random.get_state()
        counts = []
        for seed in range(60):
            data = Topology(seed)
            mal, gw = att.draw_malicious(data, seed, "backdoor", 0.1)
            self.assertIsNone(gw)
            self.assertFalse(set(mal) & data.hosts)  # a member-canary host is never malicious
            self.assertEqual(mal, att.draw_malicious(data, seed, "backdoor", 0.1)[0])  # fixed by the seed
            counts.append(len(mal))
        self.assertTrue(np.array_equal(np.random.get_state()[1], state[1]))  # a separate random stream
        self.assertAlmostEqual(np.mean(counts) / 210, 0.1 * (1 - 50 / 210 * 0.9), delta=0.02)  # f, minus hosts
        self.assertGreater(len(set(counts)), 3)  # independent draws per device, not a fixed count (DL-04)

    def test_concentrated(self):
        gateways = set()
        for seed in range(20):
            data = Topology(seed)
            mal, gw = att.draw_malicious(data, seed, "concentrated")
            self.assertEqual(len(mal), 15)
            self.assertTrue(set(mal) <= set(data.gateway_devices[gw]))
            self.assertFalse(set(mal) & data.hosts)
            gateways.add(gw)
        self.assertGreater(len(gateways), 2)  # the gateway is drawn from the seed (DL-43 (5))


class TestPoisonedRecords(unittest.TestCase):
    def test_label_flip_map(self):
        y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        want = np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        self.assertTrue(np.array_equal(att.label_flip(y), want))

    def test_backdoor_poisons_half_with_the_trigger(self):
        from metrics import apply_trigger

        class D:
            x_train = np.random.default_rng(0).standard_normal((40, 12, 1000)).astype(np.float32)
            y_train = (np.random.default_rng(1).random((40, 5)) < 0.3).astype(np.float32)
            device_rows = {3: np.arange(10, 20)}
        x, y = att.poisoned_records(D, 3, seed=5, kind="backdoor")
        orig_x, orig_y = D.x_train[10:20], D.y_train[10:20]
        changed = np.flatnonzero((x != orig_x).any(axis=(1, 2)))
        self.assertEqual(len(changed), 5)  # 50% of 10 records
        self.assertTrue(np.array_equal(x[changed], apply_trigger(orig_x[changed])))  # exactly PR item 26
        self.assertTrue((y[changed] == att.NORM_ONLY).all())
        keep = np.setdiff1d(np.arange(10), changed)
        self.assertTrue(np.array_equal(y[keep], orig_y[keep]))
        self.assertTrue(np.array_equal(x, att.poisoned_records(D, 3, seed=5, kind="backdoor")[0]))  # fixed subset
        self.assertAlmostEqual(float(np.abs(x - orig_x).max()), 0.5, places=5)  # +0.5 mV (float32)


class TestDL47Options(unittest.TestCase):
    """DL-47 candidates: triggered copies, a capped boost, host-adjusted f. The defaults keep DL-43."""

    class D:
        x_train = np.random.default_rng(0).standard_normal((40, 12, 1000)).astype(np.float32)
        y_train = (np.random.default_rng(1).random((40, 5)) < 0.3).astype(np.float32)
        device_rows = {3: np.arange(10, 20)}

    def test_copies_keep_every_record_and_add_triggered_copies(self):
        from metrics import apply_trigger
        xr, _ = att.poisoned_records(self.D, 3, seed=5, kind="backdoor")
        xc, yc = att.poisoned_records(self.D, 3, seed=5, kind="backdoor", mode="copies")
        orig_x, orig_y = self.D.x_train[10:20], self.D.y_train[10:20]
        self.assertEqual(len(xc), 15)  # 10 clean + 5 copies
        self.assertTrue(np.array_equal(xc[:10], orig_x) and np.array_equal(yc[:10], orig_y))
        idx = np.flatnonzero((xr != orig_x).any(axis=(1, 2)))  # the same chosen records as "replace"
        self.assertTrue(np.array_equal(xc[10:], apply_trigger(orig_x[idx])))
        self.assertTrue((yc[10:] == att.NORM_ONLY).all())
        xl, yl = att.poisoned_records(self.D, 3, seed=5, kind="labelflip", mode="copies")  # no copies for label flip
        self.assertEqual(len(xl), 10)
        with self.assertRaises(ValueError):
            att.poisoned_records(self.D, 3, seed=5, kind="backdoor", mode="other")

    def test_boost_is_capped_at_the_attack_norm_and_mode_is_used(self):
        seen = {}

        class T:
            def plain_sgd_update(self, start, x, y, rows, epochs, lr, batch, keys):
                seen.update(n=len(y), epochs=epochs)
                return torch.full((4,), 0.5)  # norm 1.0
        cfg = {"seed": 5, "lr": 0.1, "attack": {"kind": "backdoor", "norm": 2.0, "poison_share": 0.5, "batch": 16}}
        norm = lambda: float(torch.linalg.vector_norm(att.malicious_update(T(), self.D, 3, 1, cfg, np.zeros(4))))
        self.assertAlmostEqual(norm(), 2.0, places=5)  # DL-43 (3): scaled to exactly the attack norm
        self.assertEqual(seen, {"n": 10, "epochs": 1})
        cfg["attack"].update(boost=1.5, poison_mode="copies")
        self.assertAlmostEqual(norm(), 1.5, places=5)  # boosted, below the cap
        self.assertEqual(seen, {"n": 15, "epochs": 1})
        cfg["attack"]["boost"] = 10.0
        self.assertAlmostEqual(norm(), 2.0, places=5)  # capped at the attack norm
        cfg["attack"]["boost"] = 1.0
        self.assertAlmostEqual(norm(), 1.0, places=5)  # no boost: the raw update

    def test_host_adjust_restores_f_and_contains_the_plain_set(self):
        rates = []
        for seed in range(60):
            data = Topology(seed)
            plain, _ = att.draw_malicious(data, seed, "backdoor", 0.1)
            adj, _ = att.draw_malicious(data, seed, "backdoor", 0.1, host_adjust=True)
            self.assertTrue(set(plain) <= set(adj))
            self.assertFalse(set(adj) & data.hosts)
            rates.append(len(adj) / 210)
        self.assertAlmostEqual(float(np.mean(rates)), 0.1, delta=0.01)

    def test_flags(self):
        from run_experiment import build_argv
        cfg = {"run_id": "E4_pta_backdoor_s0", "mode": "pta", "seed": 0, "canaries": True, "fold10": "sealed",
               "base_config": "../G0a.yaml", "code_commit": "c",
               "settings": {"lr": 0.01, "c_ex": 2.0, "eps0": 8.0, "rounds": 100, "clip_C": 15.31, "tau": 17.4,
                            "quantize": True, "attack": {"kind": "backdoor", "f": 0.1, "norm": 15.31,
                                                         "poison_share": 0.5}}}
        argv = build_argv(cfg, "/x/configs/E4/a.yaml", "/data", "/out")
        for flag in ("--attack-boost", "--poison-mode", "--attack-host-adjust"):
            self.assertNotIn(flag, argv)  # DL-43 configs are unchanged
        cfg["settings"]["attack"].update(boost=3.0, poison_mode="copies", host_adjust=True)
        argv = build_argv(cfg, "/x/configs/E4/a.yaml", "/data", "/out")
        self.assertEqual(argv[argv.index("--attack-boost") + 1], "3.0")
        self.assertEqual(argv[argv.index("--poison-mode") + 1], "copies")
        self.assertIn("--attack-host-adjust", argv)


class TestAttackInPlacements(unittest.TestCase):
    """Malicious devices send a norm-A update; honest devices are bit-identical to the run without attack."""

    def setUp(self):
        torch.manual_seed(0)
        self.tr = Trainer(tiny_model, "cpu")
        self.w0 = self.tr.get_vector().clone().numpy()
        self.data = TwoGatewayData()
        self.cfg = {"q": 0.5, "steps": 2, "z": 0.5, "c_ex": 1.0, "lr": 0.1, "seed": 3, "batch": 4,
                    "edge_iterations": 1, "k": 3, "clip_C": None, "quantize": False}
        self.att = dict(self.cfg, attack={"kind": "backdoor", "norm": 7.5, "poison_share": 0.5, "batch": 4,
                                          "malicious": [0, 4, 20], "gateway": None})

    def test_norm_and_honest_bit_identical(self):
        mal = _dp_update(self.tr, self.data, 0, 1, self.att, self.w0)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(mal)), 7.5, places=4)
        for d in (1, 5, 21):
            self.assertTrue(torch.equal(_dp_update(self.tr, self.data, d, 1, self.att, self.w0),
                                        _dp_update(self.tr, self.data, d, 1, self.cfg, self.w0)))

    def test_pta_logs_poisoned_sums_and_tau_drops(self):
        groups = draw_groups(3, 1, 0, self.data.gateway_devices[0], 3)
        cfg = dict(self.att, tau=5.0)  # below the attack norm: every poisoned sum is dropped
        _, stats = pta_gateway(self.tr, self.data, 0, 1, cfg, self.w0, groups)
        n_poisoned = sum(any(d in (0, 4) for d in g) for g in groups)
        self.assertEqual(stats["poisoned_sums"], n_poisoned)
        self.assertEqual(stats["poisoned_dropped"], n_poisoned)
        self.assertEqual(len(stats["dropped_groups"]), len(stats["dropped"]))  # DL-48: members of each drop
        self.assertTrue(all(any(d in (0, 4) for d in g) for g in stats["dropped_groups"]))
        _, plain = pta_gateway(self.tr, self.data, 0, 1, dict(self.cfg, tau=5.0), self.w0, groups)
        self.assertNotIn("poisoned_sums", plain)  # no attack: the E1 output is unchanged
        _, pfr = pfr_gateway(self.tr, self.data, 0, 1, self.att, self.w0, groups)
        self.assertEqual((pfr["poisoned_sums"], pfr["poisoned_dropped"]), (n_poisoned, 0))

    def test_fedavg_positive_control_hook(self):
        cfg = dict(self.att, z=0.0)
        attacked, _ = hierfavg_gateway(self.tr, self.data, 0, 1, cfg, self.w0)
        clean, _ = hierfavg_gateway(self.tr, self.data, 0, 1, self.cfg, self.w0)
        self.assertFalse(np.array_equal(attacked, clean))
        honest_gw1, _ = hierfavg_gateway(self.tr, self.data, 1, 1, dict(cfg, attack=dict(cfg["attack"],
                                                                                          malicious=[0, 4])), self.w0)
        clean_gw1, _ = hierfavg_gateway(self.tr, self.data, 1, 1, self.cfg, self.w0)
        self.assertTrue(np.array_equal(honest_gw1, clean_gw1))  # no malicious device there: unchanged

    def test_ptgc_record_share_attack(self):
        """DL-44 (3): (N_h/N) * DP-SGD on the honest pool + sum_m (n_m/N) * a_m; no malicious device: unchanged."""
        from device_attacks import malicious_update
        got, _ = ptgc_gateway(self.tr, self.data, 0, 1, self.att, self.w0)
        honest = [d for d in self.data.gateway_devices[0] if d not in (0, 4)]
        n = {d: len(self.data.device_rows[d]) for d in self.data.gateway_devices[0]}
        total = sum(n.values())
        pooled = np.concatenate([self.data.device_rows[d] for d in honest])
        dh, _ = self.tr.dp_sgd(self.w0, self.data.x_train, self.data.y_train, pooled, self.cfg["q"], self.cfg["steps"],
                               self.cfg["z"], self.cfg["c_ex"], self.cfg["lr"], keys=(3, 1, 10_000))
        want = dh * (sum(n[d] for d in honest) / total)
        for m in (0, 4):
            want = want + (n[m] / total) * malicious_update(self.tr, self.data, m, 1, self.att, self.w0)
        self.assertLess(float(np.abs(got - want.numpy()).max()), 1e-6)
        clean1, _ = ptgc_gateway(self.tr, self.data, 1, 1, self.cfg, self.w0)
        att1, _ = ptgc_gateway(self.tr, self.data, 1, 1, dict(self.att, attack=dict(self.att["attack"],
                                                                                   malicious=[0, 4])), self.w0)
        self.assertTrue(np.array_equal(clean1, att1))  # a gateway without malicious devices is unchanged


class TestAttackArguments(unittest.TestCase):
    def test_run_pilot_checks_and_config_flags(self):
        import run_pilot
        from run_experiment import build_argv
        base = ["--config", "x", "--ptbxl-dir", "y", "--seed", "0", "--lr", "0.01", "--c-ex", "2", "--out", "z"]
        for extra in (["--mode", "ptg", "--gateway-check", "size-aware"],  # the P-TA cloud check only (DL-44)
                      ["--mode", "pta", "--attack", "backdoor"],  # no norm
                      ["--mode", "local", "--attack", "labelflip", "--attack-norm", "15"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                run_pilot.main(base + extra)
        cfg = {"run_id": "E4_pta_backdoor_s0", "mode": "pta", "seed": 0, "canaries": True, "fold10": "sealed",
               "base_config": "../G0a.yaml", "code_commit": "c",
               "settings": {"lr": 0.01, "c_ex": 2.0, "eps0": 8.0, "rounds": 100, "clip_C": 15.31, "tau": 17.4,
                            "quantize": True,
                            "attack": {"kind": "backdoor", "f": 0.1, "norm": 15.31, "poison_share": 0.5}}}
        argv = build_argv(cfg, "/x/configs/E4/a.yaml", "/data", "/out")
        i = argv.index("--attack")
        self.assertEqual(argv[i:i + 4], ["--attack", "backdoor", "--attack-norm", "15.31"])
        self.assertEqual(argv[argv.index("--attack-f") + 1], "0.1")
        self.assertEqual(argv[argv.index("--poison-share") + 1], "0.5")


class TestE4Configs(unittest.TestCase):
    """make_e4_configs: final configurations (DL-42 (5)); one attack norm = P-TA's final C (DL-43 (3))."""

    def build(self, d, changed):
        import json
        import yaml
        from test_dl42 import BASE, write_e1_configs
        e1, dl40 = write_e1_configs(d)
        fedavg = dict(BASE, run_id="E1_fedavg_s0", configuration="FedAvg", mode="fedavg", seed=0,
                      settings={"lr": 0.1, "rounds": 100})
        with open(os.path.join(e1, "E1_fedavg_s0.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump(fedavg, fh)
        point = {"pta": {"lr": 0.05, "c_ex": 0.5, "clip_C": 20.0, "tau": 23.0},
                 "ptgc": {"lr": 0.1, "c_ex": 1.0}}
        sel = {"smoke": False, "partial": False, "changed": changed,
               "designs": {m: {"decision": "change" if m in changed else "keep",
                               "selected": point.get(m, {"lr": 0.01, "c_ex": 2.0})}
                           for m in ("ptg", "ptgc", "ppe", "pfr", "pta")}}
        path = os.path.join(d, "selection_dl42.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(sel, fh)
        return e1, dl40, path

    def load(self, path):
        import yaml
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_counts_norm_and_final_settings(self):
        import tempfile
        import make_e4_configs as m
        with tempfile.TemporaryDirectory() as d:
            e1, dl40, sel = self.build(d, ["pta"])
            runs, controls, norm, refs = m.make(e1, dl40, sel, os.path.join(d, "out"), "new")
            runs = [self.load(p) for p in runs]
            refs = [self.load(p) for p in refs]
            controls = [self.load(p) for p in controls]
            e1b, dl40b, selb = self.build(os.path.join(d, "kept"), ["ptgc"])
            self.assertEqual(m.make(e1b, dl40b, selb, os.path.join(d, "out3"), "new")[2], 15.31)  # E1 C
            with self.assertRaises(SystemExit):  # a smoke selection is refused
                import json
                with open(sel, encoding="utf-8") as fh:
                    smoke = dict(json.load(fh), smoke=True)
                with open(sel, "w", encoding="utf-8") as fh:
                    json.dump(smoke, fh)
                m.make(e1, dl40, sel, os.path.join(d, "out2"), "new")
        self.assertEqual((len(runs), len(controls), len(refs), norm), (150, 2, 10, 20.0))  # 3 x 5 x 10 (DL-41)
        self.assertIn("ptgc", {c["mode"] for c in runs})  # DL-44 (3)
        for c in refs:  # DL-44 (2): P-TA at f = 0 with the size-aware check, sealed fold 10
            self.assertNotIn("attack", c["settings"])
            self.assertEqual((c["mode"], c["settings"]["gateway_check"], c["fold10"]),
                             ("pta", "size-aware", "sealed (DL-39)"))
        for c in runs:
            a = c["settings"]["attack"]
            self.assertEqual((a["norm"], a["poison_share"]), (20.0, 0.5))  # one norm in every design
            self.assertEqual(a["f"], None if a["kind"] == "concentrated" else 0.1)
            self.assertEqual(c["fold10"], "sealed (DL-39)")
            if c["mode"] == "pta":
                self.assertEqual({k: c["settings"][k] for k in ("lr", "c_ex", "clip_C", "tau", "quantize",
                                                                "gateway_check")},
                                 {"lr": 0.05, "c_ex": 0.5, "clip_C": 20.0, "tau": 23.0, "quantize": True,
                                  "gateway_check": "size-aware"})
            else:  # kept: the E1 settings
                base = {k: v for k, v in c["settings"].items() if k != "attack"}
                from test_dl42 import e1_config
                self.assertEqual(base, e1_config(c["mode"], c["seed"])["settings"])
        self.assertEqual({(c["mode"], c["fold10"], c["settings"]["attack"]["kind"]) for c in controls},
                         {("fedavg", "none", "backdoor"), ("fedavg", "none", "labelflip")})

    def test_positive_control_rule(self):
        from e4_positive_control import check

        def run(kind, f1, bd):
            return {"run_id": f"E4pc_fedavg_{kind}_s0", "mode": "fedavg", "attack": {"kind": kind},
                    "experiment_config": {"smoke_rounds": None},
                    "final_fold9": {"macro_f1_tuned": f1, "backdoor_success_clean": bd}}
        e1 = {"final_fold9": {"macro_f1_tuned": 0.67, "backdoor_success_clean": 0.23}}
        self.assertTrue(check(run("backdoor", 0.66, 0.54), run("labelflip", 0.61, 0.2), e1)["passes"])
        self.assertFalse(check(run("backdoor", 0.66, 0.52), run("labelflip", 0.61, 0.2), e1)["passes"])  # < +30
        self.assertFalse(check(run("backdoor", 0.66, 0.60), run("labelflip", 0.63, 0.2), e1)["passes"])  # drop 4
        short = run("backdoor", 0.66, 0.9)
        short["experiment_config"]["smoke_rounds"] = 5
        with self.assertRaises(SystemExit):
            check(short, run("labelflip", 0.61, 0.2), e1)


class TestGateDL47(unittest.TestCase):
    """DL-47 (4): the 13 gate configs and the pre-written selection rule (e4_gate.py)."""

    def test_gate_configs(self):
        import tempfile
        import make_e4_configs as m
        with tempfile.TemporaryDirectory() as d:
            e1, dl40, sel = TestE4Configs().build(d, ["pta"])
            paths, norm = m.make_gate(e1, dl40, sel, os.path.join(d, "out"), "new")
            cfgs = [TestE4Configs().load(p) for p in paths]
        self.assertEqual((len(cfgs), norm), (13, 20.0))
        for c in cfgs:
            a = c["settings"]["attack"]
            self.assertEqual((c["mode"], c["seed"], c["fold10"], c["settings"]["rounds"]), ("fedavg", 0, "none", 100))
            self.assertEqual((a["norm"], a["f"], a["poison_share"], a["host_adjust"]), (20.0, 0.1, 0.5, True))
            self.assertIn("DL-47", c["rules"])
        bd = sorted((c["settings"]["attack"]["poison_mode"], c["settings"]["attack"]["boost"]) for c in cfgs
                    if c["settings"]["attack"]["kind"] == "backdoor")
        self.assertEqual(bd, sorted((r, b) for r in ("copies", "replace") for b in (1.0, 3.0, 10.0, 30.0)))
        lf = [c["settings"]["attack"]["boost"] for c in cfgs if c["settings"]["attack"]["kind"] == "labelflip"]
        self.assertEqual(sorted(lf, key=lambda b: float("inf") if b is None else b), [1.0, 3.0, 10.0, 30.0, None])

    def runs(self, bd, lf, smoke=None):
        from make_e4_configs import gate_runs
        out = {}
        for run_id, want in gate_runs():
            att = dict(want, host_adjust=True, n_malicious=21)
            if want["kind"] == "backdoor":
                f = {"macro_f1_tuned": 0.6, "macro_auroc": 0.85,
                     "backdoor_success_clean": bd.get((want["poison_mode"], want["boost"]), 0.2)}
            else:
                f = {"macro_f1_tuned": lf.get(want["boost"], 0.65), "macro_auroc": 0.8, "backdoor_success_clean": 0.2}
            out[run_id] = {"run_id": run_id, "mode": "fedavg", "seed": 0, "attack": att, "final_fold9": f,
                           "experiment_config": {"smoke_rounds": smoke}}
        return out

    def test_selection_rule(self):
        from e4_gate import select
        e1 = {"final_fold9": {"macro_f1_tuned": 0.67, "backdoor_success_clean": 0.20}}
        out = select(self.runs({("copies", 10.0): 0.55, ("replace", 30.0): 0.50}, {None: 0.45, 3.0: 0.60}), e1)
        self.assertEqual(out["backdoor"]["choice"], {"poison_mode": "copies", "boost": 10.0})
        self.assertEqual(out["labelflip"]["choice"], {"boost": None})  # largest drop: scaled to C
        self.assertTrue(out["passes"])
        tie = select(self.runs({("copies", 3.0): 0.55, ("replace", 3.0): 0.55, ("copies", 1.0): 0.55},
                               {1.0: 0.5, 30.0: 0.5}), e1)
        self.assertEqual(tie["backdoor"]["choice"], {"poison_mode": "copies", "boost": 1.0})  # ties: smaller boost
        tie2 = select(self.runs({("copies", 3.0): 0.55, ("replace", 3.0): 0.55}, {}), e1)
        self.assertEqual(tie2["backdoor"]["choice"], {"poison_mode": "replace", "boost": 3.0})  # then "replace"
        self.assertEqual(tie["labelflip"]["choice"], {"boost": 1.0})
        weak = select(self.runs({("copies", 30.0): 0.49}, {None: 0.45}), e1)  # +29 points only
        self.assertFalse(weak["backdoor"]["passes"])
        self.assertFalse(weak["passes"])

    def test_cli_exit_codes_and_checks(self):
        import json
        import tempfile
        import e4_gate
        e1 = {"final_fold9": {"macro_f1_tuned": 0.67, "backdoor_success_clean": 0.20}}
        with tempfile.TemporaryDirectory() as d:
            e1p = os.path.join(d, "e1.json")
            with open(e1p, "w", encoding="utf-8") as fh:
                json.dump(e1, fh)

            def write(runs):
                for f in os.listdir(d):
                    if f.startswith("E4gate_"):
                        os.remove(os.path.join(d, f))
                for run_id, r in runs.items():
                    with open(os.path.join(d, run_id + ".json"), "w", encoding="utf-8") as fh:
                        json.dump(r, fh)

            def code(*extra):
                with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
                    e4_gate.main(["--runs", d, "--e1-fedavg", e1p, "--out", os.path.join(d, "g.json"), *extra])
                return cm.exception.code
            write(self.runs({("copies", 10.0): 0.55}, {None: 0.45}))
            self.assertEqual(code(), 0)
            write(self.runs({("copies", 10.0): 0.45}, {None: 0.45}))
            self.assertEqual(code(), 2)  # GATE_FAILED
            runs = self.runs({("copies", 10.0): 0.55}, {None: 0.45})
            runs.pop("E4gate_labelflip_b3_s0")
            write(runs)
            self.assertEqual(code(), 3)  # a run is missing
            write(self.runs({("copies", 10.0): 0.45}, {None: 0.45}, smoke=5))
            self.assertEqual(code("--smoke"), 0)  # the preflight never fails on the rule
            self.assertNotEqual(code(), 0)  # shortened runs are refused for the real selection
            bad = self.runs({}, {})
            bad["E4gate_backdoor_copies_b10_s0"]["attack"]["boost"] = 30.0
            write(bad)
            self.assertNotEqual(code(), 0)  # a run that does not match its name is refused


class TestE6Configs(unittest.TestCase):
    """make_e6_configs and e6_norms (DL-43 (12)-(15)): norm runs, pooled tau(k) and C/tau per eps, the lines."""

    def test_norms_pooling_and_lines(self):
        import json
        import tempfile
        import yaml
        import e6_norms
        import make_e6_configs as m
        with tempfile.TemporaryDirectory() as d:
            e1, dl40, sel = TestE4Configs().build(d, ["pta"])
            norm_cfgs = [TestE4Configs.load(None, p) for p in
                         m.make_norms(e1, dl40, sel, os.path.join(d, "out"), "new")]
            self.assertEqual(len(norm_cfgs), 15)  # (3 k + 2 eps) x seeds 0-2
            runs = []
            for c in norm_cfgs:
                s = c["settings"]
                self.assertTrue(s["save_norms"])
                self.assertFalse({"clip_C", "tau", "quantize"} & set(s))
                self.assertEqual((s["lr"], s["c_ex"], c["fold10"]), (0.05, 0.5, "none"))  # P-TA's final point
                scale = s.get("k", 3) * (8.0 / s["eps0"])  # fake norms that grow with k and the noise
                r = {"experiment_config": {"content": c, "smoke_rounds": None},
                     "settings": {"clip_C": None, "tau": None, "quantize": False, "k": s.get("k", 3)},
                     "norms_raw": {"device_update": list(np.arange(100.0) * (8.0 / s["eps0"])),
                                   "group_sum": list(np.arange(50.0) * scale)}}
                runs.append((c["run_id"] + ".json", "sha", r))
            out = e6_norms.pool(runs)
            self.assertAlmostEqual(out["k"]["5"]["tau"], float(np.percentile(np.tile(np.arange(50.0) * 5, 3), 99)))
            self.assertAlmostEqual(out["eps"]["1"]["clip_C"], float(np.percentile(np.tile(np.arange(100.0) * 8, 3),
                                                                                  99)))
            with self.assertRaises(SystemExit):  # a missing seed stops
                e6_norms.pool(runs[1:])
            npath = os.path.join(d, "e6_norms.json")
            with open(npath, "w", encoding="utf-8") as fh:
                json.dump(out, fh)
            lines = [TestE4Configs.load(None, p) for p in
                     m.make_lines(e1, dl40, sel, npath, os.path.join(d, "out"), "new")]
        self.assertEqual(len(lines), 30)  # eps 12 + k 18
        by = {c["run_id"]: c for c in lines}
        self.assertEqual({k: by["E6_pta_eps1_s0"]["settings"][k] for k in ("eps0", "clip_C", "tau", "quantize")},
                         {"eps0": 1.0, "clip_C": out["eps"]["1"]["clip_C"], "tau": out["eps"]["1"]["tau"],
                          "quantize": True})
        self.assertEqual({k: by["E6_pta_k5_s2"]["settings"][k] for k in ("k", "clip_C", "tau", "eps0")},
                         {"k": 5, "clip_C": 20.0, "tau": out["k"]["5"]["tau"], "eps0": 8.0})  # C unchanged
        self.assertEqual((by["E6_pfr_k1_s0"]["settings"]["k"], by["E6_pfr_k1_s0"]["settings"].get("tau")), (1, None))
        self.assertEqual(by["E6_ptgc_eps3_s1"]["settings"]["eps0"], 3.0)
        self.assertTrue(all(c["fold10"] == "sealed (DL-39)" and c["canaries"] for c in lines))
        self.assertTrue(all(c["settings"].get("gateway_check") == ("size-aware" if c["mode"] == "pta" else None)
                            for c in lines + norm_cfgs))  # DL-44 (1)

    def test_k_argument(self):
        import run_pilot
        from run_experiment import build_argv
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):  # k is for P-TA and P-FR
            run_pilot.main(["--config", "x", "--ptbxl-dir", "y", "--mode", "ptg", "--seed", "0", "--lr", "0.01",
                            "--c-ex", "2", "--k", "5", "--out", "z"])
        cfg = {"run_id": "E6_pta_k5_s0", "mode": "pta", "seed": 0, "canaries": True, "fold10": "sealed",
               "base_config": "../G0a.yaml", "code_commit": "c",
               "settings": {"lr": 0.01, "c_ex": 2.0, "eps0": 8.0, "rounds": 100, "k": 5}}
        argv = build_argv(cfg, "/x/configs/E6/a.yaml", "/data", "/out")
        self.assertEqual(argv[argv.index("--k") + 1], "5")


class TestDL49(unittest.TestCase):
    """DL-49: uniform and concentrated label flipping, their configs and the fail-fast positive control."""

    def test_concentrated_labelflip_draw_and_records(self):
        for seed in range(5):
            data = Topology(seed)
            self.assertEqual(att.draw_malicious(data, seed, "concentrated_labelflip"),
                             att.draw_malicious(data, seed, "concentrated"))  # the same 15 devices and gateway
        D = TestDL47Options.D
        x, y = att.poisoned_records(D, 3, seed=5, kind="concentrated_labelflip")
        self.assertTrue(np.array_equal(x, D.x_train[10:20]))  # no trigger
        self.assertTrue(np.array_equal(y, att.label_flip(D.y_train[10:20])))

    def test_configs(self):
        import tempfile
        import make_e4_configs as m
        with tempfile.TemporaryDirectory() as d:
            e1, dl40, sel = TestE4Configs().build(d, ["pta"])
            runs, controls, norm = m.make_dl49(e1, dl40, sel, os.path.join(d, "out"), "new")
            runs = [TestE4Configs().load(p) for p in runs]
            controls = [TestE4Configs().load(p) for p in controls]
        self.assertEqual((len(runs), len(controls), norm), (100, 2, 20.0))  # 2 attacks x 5 designs x 10 seeds
        kinds = {c["settings"]["attack"]["kind"] for c in runs}
        self.assertEqual(kinds, {"labelflip", "concentrated_labelflip"})
        for c in runs:
            a = c["settings"]["attack"]
            self.assertEqual((a["norm"], c["fold10"]), (20.0, "sealed (DL-39)"))
            self.assertNotIn("boost", a)  # scaled to C (DL-49 (1))
            self.assertEqual(a.get("host_adjust"), True if a["kind"] == "labelflip" else None)
            self.assertEqual(a["f"], 0.1 if a["kind"] == "labelflip" else None)
            self.assertEqual(c["settings"].get("rate_alert"), "size-aware" if c["mode"] == "pta" else None)
            self.assertIn("DL-49", c["rules"])
        self.assertEqual({(c["mode"], c["fold10"], c["settings"]["attack"]["kind"]) for c in controls},
                         {("fedavg", "none", "labelflip"), ("fedavg", "none", "concentrated_labelflip")})

    def test_positive_control(self):
        from e4_positive_control import check_dl49

        def run(kind, f1, smoke=None):
            return {"run_id": f"E4pc49_fedavg_{kind}_s0", "mode": "fedavg", "attack": {"kind": kind},
                    "experiment_config": {"smoke_rounds": smoke}, "final_fold9": {"macro_f1_tuned": f1}}
        e1 = {"final_fold9": {"macro_f1_tuned": 0.67}}
        ok = check_dl49({"labelflip": run("labelflip", 0.40), "concentrated_labelflip": run("concentrated_labelflip",
                                                                                              0.60)}, e1)
        self.assertTrue(ok["passes"])
        weak = check_dl49({"labelflip": run("labelflip", 0.40), "concentrated_labelflip": run("concentrated_labelflip",
                                                                                                0.63)}, e1)
        self.assertFalse(weak["passes"])  # concentrated drop 4 points < 5
        with self.assertRaises(SystemExit):  # a shortened run is refused for the real control
            check_dl49({"labelflip": run("labelflip", 0.40, smoke=5),
                        "concentrated_labelflip": run("concentrated_labelflip", 0.60)}, e1)


class TestRateAlertDL48(unittest.TestCase):
    """DL-48: the rate alert counts aborts and only the tau drops that the device sizes do not explain."""

    def test_noise_constant_and_explained_drops(self):
        from placements import noise_norm_constant, unexplained_drops
        k = noise_norm_constant(53765, 0.01, 3.3545, 2.0, 5, 0.226675)
        self.assertAlmostEqual(k, 153.45, delta=0.05)  # P-TA's final point: 15.3 at n = 10, 2.2 at n = 70
        model = {"sizes": {0: 10, 1: 11, 2: 12, 3: 70, 4: 80, 5: 90, 6: 10}, "k_noise": k, "clip_C": 15.31,
                 "tau": 17.40}
        self.assertEqual(unexplained_drops([[0, 1, 2]], model), 0)  # three tiny devices: explained
        self.assertEqual(unexplained_drops([[3, 4, 5]], model), 1)  # normal devices above tau: unexplained
        self.assertEqual(unexplained_drops([[6, 3, 4]], model), 1)  # one tiny device: sqrt(15.3^2 + 2^2 + 2^2) < tau
        self.assertEqual(unexplained_drops([], model), 0)

    def test_rates_and_check(self):
        import torch
        from placements import check_inputs, pta_check
        model = {"sizes": {d: (10 if d < 30 else 80) for d in range(60)}, "k_noise": 153.45, "clip_C": 15.31,
                 "tau": 17.40}
        stats = {0: {"used": 7, "full": 10, "aborted_groups": 0, "dropped_groups": [[0, 1, 2], [3, 4, 5], [6, 7, 8]]},
                 1: {"used": 8, "full": 10, "aborted_groups": 1, "dropped_groups": [[30, 31, 32]]}}
        _, _, old = check_inputs(stats)
        self.assertEqual({g: round(v, 2) for g, v in old.items()}, {0: 0.3, 1: 0.2})  # DL-10: every drop counts
        _, _, new = check_inputs(stats, model)
        self.assertEqual({g: round(v, 2) for g, v in new.items()}, {0: 0.0, 1: 0.2})  # DL-48
        outputs = {g: torch.zeros(4) for g in range(2)}
        self.assertEqual(pta_check(outputs, stats)[0], {0, 1})
        self.assertEqual(pta_check(outputs, stats, drop_model=model)[0], {1})  # aborts + unexplained drop > 10%

    def test_arguments(self):
        import run_pilot
        from run_experiment import build_argv
        base = ["--config", "x", "--ptbxl-dir", "y", "--seed", "0", "--lr", "0.01", "--c-ex", "2", "--out", "z"]
        for extra in (["--mode", "ptg", "--rate-alert", "size-aware"],
                      ["--mode", "pta", "--rate-alert", "size-aware"]):  # needs C and tau
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                run_pilot.main(base + extra)
        cfg = {"run_id": "E5_ours_none_s0", "mode": "pta", "seed": 0, "canaries": True, "fold10": "sealed",
               "base_config": "../G0a.yaml", "code_commit": "c",
               "settings": {"lr": 0.01, "c_ex": 2.0, "eps0": 8.0, "rounds": 100, "clip_C": 15.31, "tau": 17.4,
                            "quantize": True, "gateway_check": "size-aware", "rate_alert": "size-aware"}}
        argv = build_argv(cfg, "/x/configs/E5/a.yaml", "/data", "/out")
        self.assertEqual(argv[argv.index("--rate-alert") + 1], "size-aware")
        cfg["settings"].pop("rate_alert")
        self.assertNotIn("--rate-alert", build_argv(cfg, "/x/configs/E5/a.yaml", "/data", "/out"))


class TestSizeAwareCheckDL44(unittest.TestCase):
    """DL-44 (1): the distance is divided by the noise expected from the gateway's own device sizes."""

    def test_scales_follow_device_sizes(self):
        from placements import size_aware_scales
        sizes = {d: (10 if d < 30 else 80) for d in range(60)}  # gateway 0: small devices
        groups = {0: [[0, 1, 2], [3, 4, 5], [6, 7, 8]], 1: [[30, 31, 32], [33, 34, 35], [36, 37, 38]],
                  2: [[39, 40, 41], [42, 43, 44], [45, 46, 47]]}
        sc = size_aware_scales(groups, sizes, seed=1, rnd=4)
        self.assertEqual(sc, size_aware_scales(groups, sizes, seed=1, rnd=4))  # seeded
        self.assertAlmostEqual(sc[0] / sc[1], 8.0, delta=0.6)  # noise 1/n: 8x smaller devices, 8x more noise
        self.assertAlmostEqual(float(np.median(list(sc.values()))), 1.0)
        self.assertEqual(size_aware_scales({**groups, 3: []}, sizes, 1, 4)[3], 1.0)  # nothing used

    def test_check_flags_by_scaled_distance_and_reports_dl10(self):
        from aggregation import gateway_check
        from placements import cloud_pta
        g = torch.Generator().manual_seed(0)
        noise = {i: torch.randn(5000, generator=g) * (3.0 if i == 0 else 1.0) for i in range(7)}  # gateway 0 noisier
        used, full, rates = {i: 10 for i in range(7)}, {i: 10 for i in range(7)}, {}
        plain = gateway_check(noise, used, full, rates)[0]
        aware = gateway_check(noise, used, full, rates, scales={i: (3.0 if i == 0 else 1.0) for i in range(7)})[0]
        self.assertIn(0, plain)  # DL-10 as coded flags the honest but noisier gateway
        self.assertNotIn(0, aware)  # the size-aware check expects its larger noise
        stats = {i: {"used": 10, "full": 10} for i in range(7)}
        _, info = cloud_pta({i: v.numpy() for i, v in noise.items()}, stats,
                            scales={i: (3.0 if i == 0 else 1.0) for i in range(7)})
        self.assertEqual((info["flagged"], info["flagged_dl10"]), (sorted(aware), sorted(plain)))
        _, e1 = cloud_pta({i: v.numpy() for i, v in noise.items()}, stats)
        self.assertNotIn("flagged_dl10", e1)  # without scales: the E1 check, unchanged


if __name__ == "__main__":
    unittest.main()
