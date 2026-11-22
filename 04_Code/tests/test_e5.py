"""Unit tests for E5 gateway tampering (plan task X-7; DL-43 (8)-(11), DL-44). Standard-library unittest.

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

import tampering as tp  # noqa: E402
from aggregation import gateway_median  # noqa: E402
from dpsgd import Trainer  # noqa: E402
from placements import _dp_update, draw_groups, pta_gateway  # noqa: E402
from test_g0a import TwoGatewayData, tiny_model  # noqa: E402

torch.set_num_threads(2)


class TestTamperingTypes(unittest.TestCase):
    def test_gateway_draw(self):
        gws = [tp.tamper_gateway(s, 7) for s in range(40)]
        self.assertTrue(all(0 <= g < 7 for g in gws))
        self.assertEqual(gws, [tp.tamper_gateway(s, 7) for s in range(40)])  # fixed by the seed
        self.assertGreater(len(set(gws)), 3)

    def test_output_types(self):
        v = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
        self.assertTrue(np.array_equal(tp.tamper_output("scale", v, 1, 2, 3)[0], 2 * v))
        self.assertTrue(np.array_equal(tp.tamper_output("signflip", v, 1, 2, 3)[0], -v))
        inj = tp.tamper_output("inject", v, 1, 2, 3)[0]
        self.assertAlmostEqual(float(np.linalg.norm(inj)), float(np.linalg.norm(v)), places=3)  # same norm
        self.assertLess(abs(float(inj @ v)) / float(np.linalg.norm(v)) ** 2, 0.2)  # a random direction
        with self.assertRaises(ValueError):  # the drop acts inside the gateway
            tp.tamper_output("drop", v, 1, 2, 3)

    def test_adaptive_search_takes_the_largest_s_that_passes(self):
        """DL-46 (1): s = 1.00, 0.99, ..., 0.00; the first (largest) s not flagged with distance <= 0.95 x threshold."""
        v = np.ones(4, dtype=np.float32)
        seen = []

        def check(out):  # a toy check: the distance grows with |out - v|, the threshold is 1.0
            s = float(1.0 - out[0]) / 2.0
            seen.append(round(s, 2))
            return False, float(np.linalg.norm(out - v)), 1.0
        out, s = tp.adaptive_search(v, check)
        self.assertEqual(s, 0.23)  # ||out - v|| = 2s ||v|| = 4s <= 0.95  ->  s <= 0.2375
        self.assertEqual(seen[:3], [1.0, 0.99, 0.98])  # from the full sign flip downward
        self.assertTrue(np.allclose(out, (1 - 2 * 0.23) * v))
        out, s = tp.adaptive_search(v, lambda o: (False, 0.0, 1.0))
        self.assertEqual((s, list(out)), (1.0, [-1.0] * 4))  # the full sign flip passes
        out, s = tp.adaptive_search(v, lambda o: (True, abs(float(o[0]) - 0.2) + 1.0, 1.0))  # always flagged
        self.assertEqual(s, 0.4)  # fallback: the smallest distance / threshold ratio (1 - 2s = 0.2)
        with self.assertRaises(ValueError):  # no fixed formula any more (DL-46)
            tp.tamper_output("adaptive", v, 1, 2, 0)

    def test_adaptive_search_with_the_real_size_aware_check(self):
        """The chosen output passes the cloud's check of the same round, and s + 0.01 would not (DL-46 (1))."""
        from placements import pta_check
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(500).astype(np.float32)
        outputs = {gw: (signal + rng.standard_normal(500)).astype(np.float32) for gw in range(7)}
        stats = {gw: {"used": 10, "full": 10} for gw in range(7)}
        scales = {gw: 1.0 + 0.1 * gw for gw in range(7)}

        def check_for(out):
            flagged, dist, thr = pta_check({**outputs, 2: out}, stats, scales)
            return 2 in flagged, dist[2], thr
        out, s = tp.adaptive_search(outputs[2], check_for)
        flagged, dist, thr = check_for(out)
        self.assertFalse(flagged)
        self.assertLessEqual(dist, 0.95 * thr)
        if s < 1.0:
            nxt = check_for(((1 - 2 * (s + 0.01)) * outputs[2]).astype(np.float32))
            self.assertTrue(nxt[0] or nxt[1] > 0.95 * nxt[2])

    def test_adaptive_against_hflmld(self):
        """DL-45 (1), DL-46 (2): out = -min(1, 0.95 D / ||v||) v, D from the current round; honest without D."""
        v = np.array([3.0, 4.0], dtype=np.float32)  # ||v|| = 5
        out, s = tp.adaptive_hflmld(v, None)
        self.assertEqual((s, list(out)), (0.0, [3.0, 4.0]))  # no other distance: honest
        out, s = tp.adaptive_hflmld(v, 10.0)
        self.assertEqual((s, list(out)), (1.0, [-3.0, -4.0]))  # not the farthest: the full sign flip
        out, s = tp.adaptive_hflmld(v, 4.0)
        self.assertAlmostEqual(s, 0.95 * 4.0 / 5.0)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 0.95 * 4.0, places=5)  # just below the farthest other
        self.assertLess(float(out @ v), 0)  # still a flip
        self.assertEqual(tp.adaptive_hflmld(np.zeros(2), 4.0)[1], 0.0)

    def test_score_skips_the_adaptive_first_round(self):
        rounds = [{"round": 1, "flagged": [2]}, {"round": 2, "flagged": [0]}, {"round": 3, "flagged": []}]
        s = tp.score(rounds, gateway=0, kind="adaptive", n_gateways=3)
        self.assertEqual((s["tampering_gateway_rounds"], s["detected"]), (2, 1))  # rounds 2-3 only
        self.assertEqual((s["honest_gateway_rounds"], s["false_alarms"]), (6, 1))
        s = tp.score(rounds, gateway=0, kind="scale", n_gateways=3)
        self.assertEqual((s["tampering_gateway_rounds"], s["detected"]), (3, 1))


class TestTamperingInGateways(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tr = Trainer(tiny_model, "cpu")
        self.w0 = self.tr.get_vector().clone().numpy()
        self.data = TwoGatewayData()
        self.cfg = {"q": 0.5, "steps": 2, "z": 0.5, "c_ex": 1.0, "lr": 0.1, "seed": 3, "batch": 4,
                    "edge_iterations": 1, "k": 3, "clip_C": None, "quantize": False}

    def test_silent_drop_in_pta(self):
        """DL-43 (9): the median of a random half of the sums, every sum reported as used."""
        groups = draw_groups(3, 1, 0, self.data.gateway_devices[0], 3)
        honest, hs = pta_gateway(self.tr, self.data, 0, 1, self.cfg, self.w0, groups)
        cfg = dict(self.cfg, tamper={"kind": "drop", "gateway": 0})
        out, st = pta_gateway(self.tr, self.data, 0, 1, cfg, self.w0, groups)
        self.assertEqual((st["used"], st["full"], st["dropped"]), (hs["used"], hs["full"], []))  # silent
        self.assertEqual(st["used_groups"], hs["used_groups"])
        vectors = {d: _dp_update(self.tr, self.data, d, 1, self.cfg, self.w0) for g in groups for d in g}
        sums = [torch.stack([vectors[d] for d in g]).sum(0) for g in groups]
        keep = tp.drop_half(len(sums), 3, 1, 0)
        want = gateway_median([sums[i] for i in keep], 3)[0]
        self.assertLess(float(np.abs(out - want.numpy()).max()), 1e-6)
        self.assertFalse(np.array_equal(out, honest))
        other, _ = pta_gateway(self.tr, self.data, 1, 1, cfg, self.w0, draw_groups(3, 1, 1,
                                                                               self.data.gateway_devices[1], 3))
        clean, _ = pta_gateway(self.tr, self.data, 1, 1, self.cfg, self.w0, draw_groups(3, 1, 1,
                                                                                    self.data.gateway_devices[1], 3))
        self.assertTrue(np.array_equal(other, clean))  # only the tampering gateway changes

    def test_no_dp_is_plain_sgd(self):
        cfg = dict(self.cfg, no_dp=True)
        got = _dp_update(self.tr, self.data, 5, 2, cfg, self.w0)
        want = self.tr.plain_sgd_update(self.w0, self.data.x_train, self.data.y_train, self.data.device_rows[5], 1,
                                        0.1, 4, keys=(3, 2, 5))
        self.assertTrue(torch.equal(got, want))


class TestE5ArgumentsAndConfigs(unittest.TestCase):
    def test_run_pilot_checks(self):
        import run_pilot
        base = ["--config", "x", "--ptbxl-dir", "y", "--seed", "0", "--lr", "0.01", "--c-ex", "2", "--out", "z"]
        for extra in (["--mode", "pfr", "--tamper", "adaptive"],  # pta or ppe only
                      ["--mode", "ppe", "--cloud-rule", "hflmld", "--cloud-eta", "1"],  # the P-TA detector
                      ["--mode", "pta", "--cloud-rule", "hflmld"],  # needs --cloud-eta
                      ["--mode", "pta", "--no-dp"],  # ppe only
                      ["--mode", "ptg", "--tamper", "scale"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                run_pilot.main(base + extra)

    def test_configs(self):
        import tempfile
        import make_e5_configs as m
        from run_experiment import build_argv
        from test_e4 import TestE4Configs
        with tempfile.TemporaryDirectory() as d:
            e1, dl40, sel = TestE4Configs().build(d, ["pta"])
            cfgs = [TestE4Configs.load(None, p) for p in m.make(e1, dl40, sel, os.path.join(d, "out"), "new")]
            no_refs = m.make(e1, dl40, sel, os.path.join(d, "out2"), "new", references=False)
        self.assertEqual(len(cfgs), 180)  # 5 types x 3 detectors x 10 seeds + 30 without tampering (DL-45)
        self.assertEqual(len(no_refs), 150)
        by = {c["run_id"]: c for c in cfgs}
        ours, native, cloud = by["E5_ours_adaptive_s4"], by["E5_hflmld_native_drop_s4"], by["E5_hflmld_cloud_inject_s4"]
        self.assertEqual((ours["mode"], ours["settings"]["gateway_check"], ours["settings"]["lr"],
                          ours["settings"]["rate_alert"]),
                         ("pta", "size-aware", 0.05, "size-aware"))  # P-TA's final point; DL-44 check; DL-48 alert
        self.assertIn("DL-48", ours["rules"])
        self.assertNotIn("rate_alert", cloud["settings"])  # the HFLMLD cloud rule has no rate alert
        self.assertEqual({k: native["settings"][k] for k in ("lr", "cloud_eta", "no_dp")},
                         {"lr": 0.1, "cloud_eta": 0.1, "no_dp": True})
        self.assertEqual({k: cloud["settings"].get(k) for k in ("cloud_rule", "cloud_eta", "gateway_check")},
                         {"cloud_rule": "hflmld", "cloud_eta": 1.0, "gateway_check": None})
        self.assertTrue(all(c["settings"]["partition"] == "partition_E5.yaml" and c["fold10"] == "sealed (DL-39)"
                            for c in cfgs))
        self.assertIn("E5_hflmld_cloud_adaptive_s0", by)  # DL-45 (1)
        ref = by["E5_hflmld_native_none_s3"]  # DL-45 (2): no tampering
        self.assertNotIn("tamper", ref["settings"])
        self.assertEqual(ref["experiment"], "E5 reference (no tampering)")
        argv = build_argv(native, "/x/configs/E5/a.yaml", "/data", "/out")
        self.assertEqual(argv[argv.index("--partition-config") + 1], "/x/configs/partition_E5.yaml")
        self.assertIn("--no-dp", argv)
        self.assertEqual(argv[argv.index("--tamper") + 1], "drop")
        argv = build_argv(cloud, "/x/configs/E5/a.yaml", "/data", "/out")
        self.assertEqual(argv[argv.index("--cloud-rule") + 1], "hflmld")
        self.assertEqual(argv[argv.index("--cloud-eta") + 1], "1.0")


if __name__ == "__main__":
    unittest.main()
