"""Unit tests for the G0-a code (plan task G0-09). Standard-library unittest.

Run:  <venv>/bin/python -m unittest discover -s 04_Code/tests -v
Tests that need PTB-XL run only if PTBXL_DIR is set.
"""
import csv
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fl_system"))
sys.path.insert(0, os.path.join(HERE, "..", "defences"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from aggregation import (gateway_check, gateway_median, group_sums, median_lower_upper_mean,  # noqa: E402
                         quantize)
from dpsgd import Trainer  # noqa: E402
from metrics import diagnostic_scores, macro_f1_at, scores, trivial_macro_f1, tuned_thresholds  # noqa: E402
from model import ECG_PARAMETERS, ecg_cnn, n_parameters  # noqa: E402
from placements import (cloud_pfr, cloud_weighted, draw_groups, hierfavg_gateway, pfr_gateway,  # noqa: E402
                        ppe_gateway, pta_gateway, ptg_gateway, ptgc_gateway)
import certificates  # noqa: E402
import hflmld  # noqa: E402

torch.set_num_threads(2)


def tiny_model():
    return nn.Sequential(nn.Conv1d(12, 4, 7, padding=3), nn.GroupNorm(2, 4), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                         nn.Flatten(), nn.Linear(4, 5))


def toy_data(n=40, length=50, seed=0):
    g = np.random.default_rng(seed)
    return (g.standard_normal((n, 12, length)).astype(np.float32),
            (g.random((n, 5)) < 0.3).astype(np.float32))


class TestModel(unittest.TestCase):
    def test_parameter_count(self):
        self.assertEqual(n_parameters(ecg_cnn()), ECG_PARAMETERS)
        self.assertEqual(ECG_PARAMETERS, 53_765)


class TestDPSGD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tr = Trainer(tiny_model, "cpu")
        self.w0 = self.tr.get_vector().clone().numpy()
        self.x, self.y = toy_data()

    def manual_step(self, rows, c_ex, lr):
        """Reference: per-example gradient by single-example backward, clipped, summed."""
        model = tiny_model()
        nn.utils.vector_to_parameters(torch.as_tensor(self.w0), model.parameters())
        loss = nn.BCEWithLogitsLoss(reduction="sum")
        total = torch.zeros(len(self.w0))
        for i in rows:
            model.zero_grad()
            loss(model(torch.as_tensor(self.x[i:i + 1])), torch.as_tensor(self.y[i:i + 1])).backward()
            g = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
            total += g * min(1.0, c_ex / float(g.norm()))
        return -lr * total / len(rows)

    def test_zero_noise_equals_manual_clipping(self):
        rows = np.arange(len(self.x))
        delta, st = self.tr.dp_sgd(self.w0, self.x, self.y, rows, q=1.0, steps=1, z=0.0, c_ex=0.05, lr=0.1,
                                   keys=(1, 1, 1))
        self.assertEqual(st["batch_sizes"], [len(rows)])
        ref = self.manual_step(rows, 0.05, 0.1)
        self.assertLess(float((delta - ref).abs().max()), 1e-5)

    def test_clipped_norms_bounded(self):
        xb, yb = torch.as_tensor(self.x[:8]), torch.as_tensor(self.y[:8])
        self.tr.set_vector(self.w0)
        _, norms = self.tr._clipped_sum(xb, yb, 0.01)
        self.assertTrue(bool((norms > 0.01).any()), "test needs some gradients above the clip")
        g_sum, _ = self.tr._clipped_sum(xb, yb, 0.01)
        self.assertLessEqual(float(g_sum.norm()), 8 * 0.01 + 1e-6)

    def test_sampling_rate(self):
        x, y = toy_data(n=400, length=20)
        _, st = self.tr.dp_sgd(self.w0, x, y, np.arange(400), q=0.2, steps=50, z=0.0, c_ex=1.0, lr=0.0,
                               keys=(2, 1, 1))
        rate = np.mean(st["batch_sizes"]) / 400
        self.assertAlmostEqual(rate, 0.2, delta=0.01)

    def test_noise_scale(self):
        # gradients clipped to ~0, so the update is pure noise: std = lr * z * c_ex / (q * n) per step
        n, q, lr, z, c_ex = 50, 0.5, 1.0, 1e6, 1e-6
        x, y = toy_data(n=n)
        delta, _ = self.tr.dp_sgd(self.w0, x, y, np.arange(n), q=q, steps=1, z=z, c_ex=c_ex, lr=lr, keys=(3, 1, 1))
        expected = lr * z * c_ex / (q * n)
        self.assertAlmostEqual(float(delta.std()) / expected, 1.0, delta=0.05)

    def test_chunked_equals_whole_batch(self):
        # regression test for the 15 Nov P-TG-c GPU OOM: micro-batches must give the same clipped sum
        x, y = toy_data(n=40)
        idx = np.arange(40)
        self.tr.set_vector(self.w0)
        whole, _ = self.tr._clipped_sum(torch.as_tensor(x), torch.as_tensor(y), 0.05)
        small = Trainer(tiny_model, "cpu", micro_batch=16)  # 40 records -> chunks of 16, 16, 8
        small.set_vector(self.w0)
        chunked = small._clipped_sum_chunked(x, y, idx, 0.05)
        self.assertLess(float((whole - chunked).abs().max()), 1e-6)
        self.assertTrue(all(getattr(p, "grad_sample", None) is None for p in small.params))

    def test_plain_training_keeps_no_per_example_gradients(self):
        # regression test for the 12 Nov GPU OOM: local-only training must not store Opacus grad_sample
        x, y = toy_data(n=64, length=50)
        seen = []
        out = self.tr.plain_sgd_epochs(self.w0, x, y, np.arange(64), epochs=3, lr=0.01, batch=16, keys=(4, 1),
                                       checkpoints={3}, evaluate=lambda m: seen.append(m) or "ok")
        self.assertEqual(out, {3: "ok"})
        self.assertTrue(all(getattr(p, "grad_sample", None) is None for p in self.tr.params))
        self.assertTrue(all(getattr(p, "grad_sample", None) is None for p in seen[0].parameters()))
        self.assertIsNot(seen[0], self.tr.model)

    def test_training_leaves_the_start_vector_unchanged(self):
        # regression test for the 15 Nov CPU bug: local-only training wrote into the caller's start vector
        x, y = toy_data(n=32)
        rows = np.arange(32)
        start = self.w0.copy()
        snaps = []
        for _ in range(2):  # the same training twice must give the same model
            self.tr.plain_sgd_epochs(start, x, y, rows, epochs=2, lr=0.1, batch=8, keys=(3, 1), checkpoints={2},
                                     evaluate=lambda m: snaps.append(nn.utils.parameters_to_vector(m.parameters())
                                                                     .detach().clone()))
        np.testing.assert_array_equal(start, self.w0)
        self.assertTrue(torch.equal(snaps[0], snaps[1]))
        self.tr.dp_sgd(start, x, y, rows, 0.5, 2, 1.0, 1.0, 0.1, keys=(3, 1, 2))
        np.testing.assert_array_equal(start, self.w0)

    def test_same_keys_same_result(self):
        rows = np.arange(len(self.x))
        a, _ = self.tr.dp_sgd(self.w0, self.x, self.y, rows, 0.3, 3, 1.0, 0.5, 0.1, keys=(5, 2, 7))
        b, _ = self.tr.dp_sgd(self.w0, self.x, self.y, rows, 0.3, 3, 1.0, 0.5, 0.1, keys=(5, 2, 7))
        c, _ = self.tr.dp_sgd(self.w0, self.x, self.y, rows, 0.3, 3, 1.0, 0.5, 0.1, keys=(5, 2, 8))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))


class FakeData:
    def __init__(self, n_dev=30, per_dev=10):
        self.x_train, self.y_train = toy_data(n=n_dev * per_dev, length=20)
        self.device_rows = {d: np.arange(d * per_dev, (d + 1) * per_dev) for d in range(n_dev)}
        self.gateway_devices = {0: list(range(n_dev))}


class TestPTGc(unittest.TestCase):
    def test_noise_once_per_step_on_pooled_sum(self):
        torch.manual_seed(0)
        tr = Trainer(tiny_model, "cpu")
        data = FakeData()
        w0 = tr.get_vector().clone().numpy()
        cfg = {"q": 0.5, "steps": 4, "z": 1e6, "c_ex": 1e-6, "lr": 1.0, "seed": 9}
        vec, _ = ptgc_gateway(tr, data, 0, 1, cfg, w0)
        pooled = 300  # 30 devices x 10 records
        one_noise = cfg["lr"] * cfg["z"] * cfg["c_ex"] * np.sqrt(cfg["steps"]) / (cfg["q"] * pooled)
        # one noise draw per step on the pooled sum; 30 separate device noises would be sqrt(30) larger
        self.assertAlmostEqual(float(np.std(vec)) / one_noise, 1.0, delta=0.05)


class TwoGatewayData(FakeData):
    def __init__(self):
        super().__init__(n_dev=30, per_dev=10)
        self.gateway_devices = {0: list(range(15)), 1: list(range(15, 30))}
        self.device_rows[29] = self.device_rows[29][:4]  # unequal sizes for the weighted means


class TestNewPlacementsM201(unittest.TestCase):
    """M2-01 (PR §5, DL-37): P-TG, P-FR, P-PE, FedAvg and HierFAVG gateway and cloud work."""

    def setUp(self):
        torch.manual_seed(0)
        self.tr = Trainer(tiny_model, "cpu")
        self.w0 = self.tr.get_vector().clone().numpy()
        self.data = TwoGatewayData()
        self.cfg = {"q": 0.5, "steps": 2, "z": 0.5, "c_ex": 1.0, "lr": 0.1, "seed": 3, "batch": 4,
                    "edge_iterations": 1, "k": 3, "clip_C": None, "quantize": False}

    def dp(self, d, rnd=1):
        delta, _ = self.tr.dp_sgd(self.w0, self.data.x_train, self.data.y_train, self.data.device_rows[d],
                                  self.cfg["q"], self.cfg["steps"], self.cfg["z"], self.cfg["c_ex"], self.cfg["lr"],
                                  keys=(self.cfg["seed"], rnd, d))
        return delta

    def test_ptg_is_median_of_plaintext_updates(self):
        vec, stats = ptg_gateway(self.tr, self.data, 0, 1, self.cfg, self.w0)
        want = median_lower_upper_mean(torch.stack([self.dp(d) for d in range(15)]))
        self.assertLess(float(np.abs(vec - want.numpy()).max()), 1e-6)
        self.assertEqual(stats["used"], 15)

    def test_pfr_relays_group_sums_and_cloud_takes_one_median(self):
        groups = {g: draw_groups(3, 1, g, self.data.gateway_devices[g], 3) for g in (0, 1)}
        outputs = {}
        for g in (0, 1):
            sums, stats = pfr_gateway(self.tr, self.data, g, 1, self.cfg, self.w0, groups[g])
            self.assertEqual(len(sums), 5)  # 15 devices / k = 3; the gateway relays every sum
            outputs[g] = sums
        agg, info = cloud_pfr(outputs, 3)
        stacked = torch.stack([torch.as_tensor(s) / 3 for g in (0, 1) for s in outputs[g]])
        self.assertTrue(np.allclose(agg, median_lower_upper_mean(stacked).numpy()))
        self.assertEqual(info["group_sums"], 10)

    def test_fedavg_equals_global_weighted_mean(self):
        cfg = dict(self.cfg, z=0.0)
        outputs, stats = {}, {}
        for g in (0, 1):
            outputs[g], stats[g] = hierfavg_gateway(self.tr, self.data, g, 1, cfg, self.w0)
        agg, _ = cloud_weighted(outputs, stats)
        total = sum(len(r) for r in self.data.device_rows.values())
        want = sum(len(self.data.device_rows[d]) * self.tr.plain_sgd_update(
            self.w0, self.data.x_train, self.data.y_train, self.data.device_rows[d], 1, cfg["lr"], cfg["batch"],
            keys=(cfg["seed"], 1, d)) for d in range(30)) / total
        self.assertLess(float(np.abs(agg - want.numpy()).max()), 1e-6)

    def test_hierfavg_edge_iterations_use_new_device_rounds(self):
        cfg = dict(self.cfg, edge_iterations=2)
        d1, _ = hierfavg_gateway(self.tr, self.data, 0, 1, cfg, self.w0)
        d2, _ = hierfavg_gateway(self.tr, self.data, 0, 1, cfg, self.w0)
        self.assertTrue(np.array_equal(d1, d2))  # deterministic
        one, _ = hierfavg_gateway(self.tr, self.data, 0, 1, dict(self.cfg, edge_iterations=1), self.w0)
        self.assertFalse(np.allclose(d1, one))  # a second edge iteration changes the edge model

    def test_ppe_runs_edge_iterations_and_logs_detection(self):
        cfg = dict(self.cfg, edge_iterations=2)
        vec, stats = ppe_gateway(self.tr, self.data, 0, 2, cfg, self.w0)
        self.assertEqual(vec.shape, self.w0.shape)
        self.assertEqual([e["device_round"] for e in stats["edge_log"]], [3, 4])  # cloud round 2, T_e = 2
        self.assertTrue(all(1 <= e["admitted"] <= 15 for e in stats["edge_log"]))


class TestCertificatesM202(unittest.TestCase):
    """M2-02: signed model, certified groups, t = k. A forged model or group is rejected; a dropout aborts."""

    def setUp(self):
        self.cloud = certificates.Cloud(seed=5)
        self.dcerts = {d: self.cloud.enroll(d, certificates.device_public_hex(5, d)) for d in range(30)}
        self.model = np.linspace(-1, 1, 100).astype(np.float32)
        self.groups = [[0, 1, 2], [3, 4, 5]]
        self.cert = self.cloud.sign_round(7, self.model, 0, self.groups)

    def check(self, cert=None, rnd=7, model=None, device=1, group=(0, 1, 2), dcerts=None, gateway=0):
        return certificates.device_check(cert or self.cert, self.cloud.public_hex, rnd,
                                         self.model if model is None else model, gateway, device,
                                         list(group) if group is not None else None,
                                         self.dcerts if dcerts is None else dcerts)

    def test_honest_round_is_accepted(self):
        self.assertEqual(self.check(), (True, "ok"))
        plain = self.cloud.sign_round(7, self.model, 0)  # signed model only (P-TG, P-TG-c, P-PE)
        self.assertEqual(self.check(cert=plain, group=None), (True, "ok"))

    def test_forged_or_inconsistent_model_is_rejected(self):
        other = self.model.copy()
        other[0] += 1e-3  # a model the gateway changed (Pasquini et al.: inconsistent models)
        self.assertFalse(self.check(model=other)[0])

    def test_forged_certificate_is_rejected(self):
        body = json.loads(self.cert["payload"])
        body["groups"] = [[0, 1, 99]]  # the gateway edits the group list
        edited = {"payload": json.dumps(body, sort_keys=True, separators=(",", ":")), "sig": self.cert["sig"]}
        self.assertEqual(self.check(cert=edited), (False, "round certificate: bad cloud signature"))
        fake_cloud = certificates.Cloud(seed=6)  # the gateway signs with its own key
        self.assertFalse(self.check(cert=fake_cloud.sign_round(7, self.model, 0, self.groups))[0])

    def test_uncertified_group_and_replay_are_rejected(self):
        self.assertEqual(self.check(group=(0, 1, 3)), (False, "group is not certified"))  # gateway-made group
        self.assertFalse(self.check(rnd=8)[0])  # an old certificate replayed in a later round
        self.assertFalse(self.check(gateway=1)[0])  # another gateway's certificate

    def test_sybil_member_is_rejected(self):
        dcerts = dict(self.dcerts)
        dcerts[2] = certificates.Cloud(seed=6).enroll(2, "00" * 32)  # fake device certificate
        self.assertFalse(self.check(dcerts=dcerts)[0])
        dcerts[2] = self.dcerts[0]  # a real certificate, but for another device
        self.assertFalse(self.check(dcerts=dcerts)[0])

    def test_t_equals_k_aborts_any_incomplete_group(self):
        complete, aborted = certificates.complete_groups(self.groups, sent=[0, 1, 2, 3, 4])
        self.assertEqual((complete, aborted), ([[0, 1, 2]], [[3, 4, 5]]))


class TestProtocolInPlacementsM202(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tr = Trainer(tiny_model, "cpu")
        self.w0 = self.tr.get_vector().clone().numpy()
        self.data = TwoGatewayData()
        self.cfg = {"q": 0.5, "steps": 2, "z": 0.5, "c_ex": 1.0, "lr": 0.1, "seed": 3, "batch": 4,
                    "edge_iterations": 1, "k": 3, "clip_C": 5.0, "quantize": True, "tau": None}
        self.cloud = certificates.Cloud(seed=3)
        self.dcerts = {d: self.cloud.enroll(d, certificates.device_public_hex(3, d)) for d in range(30)}
        self.groups = draw_groups(3, 1, 0, self.data.gateway_devices[0], 3)

    def auth(self, model=None, dcerts=None):
        cert = self.cloud.sign_round(1, self.w0 if model is None else model, 0, self.groups)
        return {"cloud_pub": self.cloud.public_hex, "cert": cert, "device_certs": dcerts or self.dcerts}

    def test_honest_checks_do_not_change_results(self):
        plain, s0 = pta_gateway(self.tr, self.data, 0, 1, self.cfg, self.w0, self.groups)
        checked, s1 = pta_gateway(self.tr, self.data, 0, 1, self.cfg, self.w0, self.groups, self.auth())
        self.assertTrue(np.array_equal(plain, checked))
        self.assertEqual((s1["aborted_groups"], s1["refused"]), (0, {}))
        self.assertGreater(s1["check_s"], 0.0)

    def test_sybil_member_aborts_only_its_group(self):
        dcerts = dict(self.dcerts)
        victim = self.groups[0][0]
        dcerts[victim] = certificates.Cloud(seed=99).enroll(victim, "00" * 32)
        _, stats = pta_gateway(self.tr, self.data, 0, 1, self.cfg, self.w0, self.groups, self.auth(dcerts=dcerts))
        self.assertEqual(stats["aborted_groups"], 1)  # every member of that group refuses -> t = k abort
        self.assertEqual(sorted(stats["refused"]), sorted(self.groups[0]))
        self.assertEqual(stats["used"], len(self.groups) - 1)

    def test_forged_model_aborts_everything(self):
        vec, stats = pfr_gateway(self.tr, self.data, 0, 1, self.cfg, self.w0, self.groups,
                                 self.auth(model=self.w0 + 1.0))
        self.assertEqual((len(vec), stats["aborted_groups"]), (0, len(self.groups)))

    def test_cloud_pfr_abort_rate_alert(self):
        outputs = {0: [np.ones(4, np.float32)] * 3, 1: [np.zeros(4, np.float32)] * 5}
        stats = {0: {"aborted_groups": 2, "full": 5}, 1: {"aborted_groups": 0, "full": 5}}
        agg, info = cloud_pfr(outputs, 1, stats)
        self.assertEqual(info["flagged"], [0])  # 40% aborted > 10%
        self.assertTrue(np.array_equal(agg, np.zeros(4, np.float32)))


class TestMetricsM203(unittest.TestCase):
    """M2-03: PR §7 measures on known inputs (definitions in DL-38)."""

    def test_sensitivity_at_90_specificity(self):
        from metrics import sensitivity_at_specificity
        y = np.r_[np.zeros(10), np.ones(10)]
        s = np.r_[np.arange(10) / 10, [0.95] * 5 + [0.05] * 5]
        self.assertAlmostEqual(sensitivity_at_specificity(y, s, 0.9), 0.5)

    def test_trigger_and_backdoor_success(self):
        from metrics import apply_trigger, backdoor_success
        x = np.zeros((2, 12, 1000), np.float32)
        t = apply_trigger(x)
        self.assertEqual(float(x.sum()), 0.0)  # a copy
        self.assertAlmostEqual(float(t.sum()), 2 * 20 * 0.5)  # 0.2 s at 100 Hz = 20 samples
        self.assertTrue(np.all(t[:, 1, 100:120] == 0.5) and t[:, 1, 120] .max() == 0 and t[:, 0].max() == 0)
        y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        p = np.array([[.9, .1, .1, .1, .1], [.9, .1, .1, .1, .1], [.9, .2, .1, .1, .1], [.2, .1, .9, .1, .1]])
        self.assertAlmostEqual(backdoor_success(y, p, np.full(5, 0.5)), 2 / 3)  # the NORM-only record is excluded

    def test_ecg_r_and_dtw(self):
        from metrics import dtw_normalized, ecg_dtw, ecg_r
        x = np.sin(np.linspace(0, 6, 200))[None].repeat(12, 0)
        self.assertAlmostEqual(ecg_r(x, x), 1.0)
        self.assertAlmostEqual(ecg_r(x, -x), -1.0)
        self.assertAlmostEqual(dtw_normalized([0, 1, 2], [0, 0, 1, 2]), 0.0)  # warping absorbs the repeat
        self.assertAlmostEqual(dtw_normalized([0, 0, 0], [1, 1, 1]), 1.0)
        self.assertAlmostEqual(ecg_dtw(x, x), 0.0)
        shifted = np.roll(x, 5, axis=1)
        self.assertLess(ecg_dtw(x, shifted), float(np.mean(np.abs(x - shifted))) / x.std())

    def test_psnr_ssim(self):
        from metrics import psnr, ssim
        a = np.random.default_rng(0).random((28, 28))
        self.assertAlmostEqual(psnr(np.zeros((4, 4)), np.full((4, 4), 0.1)), 20.0)
        self.assertAlmostEqual(ssim(a, a), 1.0)
        noisy = np.clip(a + np.random.default_rng(1).normal(0, 0.2, a.shape), 0, 1)
        self.assertLess(ssim(a, noisy), 0.9)
        self.assertAlmostEqual(ssim(a, noisy), ssim(noisy, a))

    def test_wilson_and_auc(self):
        from metrics import membership_auc, wilson_ci
        lo, hi = wilson_ci(9, 10)
        self.assertAlmostEqual(lo, 0.5959, places=3)
        self.assertAlmostEqual(hi, 0.9821, places=3)
        self.assertEqual(membership_auc([2, 3], [0, 1]), 1.0)

    def test_epsilon_lower_bound_matches_report_07(self):
        from metrics import epsilon_lower_bound
        best = epsilon_lower_bound(np.ones(500), np.zeros(500))  # perfect audit, 500 in / 500 out
        self.assertAlmostEqual(best, 5.11, places=2)  # report 07 and DL-13: best possible eps_lb = 5.1
        g = np.random.default_rng(0)
        self.assertLess(epsilon_lower_bound(g.normal(size=50), g.normal(size=50)), 1.0)  # no signal

    def test_wan_bytes(self):
        from metrics import wan_bytes
        pta = wan_bytes("pta", 53765, 7, 30)
        pfr = wan_bytes("pfr", 53765, 7, 30)
        self.assertEqual(pfr["uplink_per_cloud_round"] / pta["uplink_per_cloud_round"], 30)  # H5(b): ~ N_g
        self.assertEqual(wan_bytes("ppe", 53765, 7, 30, 5)["uplink_per_device_round"],
                         pta["uplink_per_cloud_round"] / 5)

    def test_scores_report_mi_sensitivity(self):
        g = np.random.default_rng(4)
        y = (g.random((300, 5)) < 0.3).astype(int)
        self.assertIn("mi_sens_at_90spec", scores(y, g.random((300, 5))))


class TestCanariesM203(unittest.TestCase):
    def setUp(self):
        import canaries
        self.c = canaries
        self.data = FakeData(n_dev=30, per_dev=10)

    def test_draw_and_insert(self):
        before = {d: set(v.tolist()) for d, v in self.data.device_rows.items()}
        y_before = self.data.y_train.copy()
        can = self.c.draw(self.data, seed=4)
        self.assertEqual((len(can["rows"]), int(can["member"].sum())), (100, 50))
        self.assertTrue(all(not np.array_equal(l, y_before[r]) and l.any() for r, l in zip(can["rows"], can["labels"])))
        self.c.apply(self.data, can)
        self.assertTrue(np.array_equal(y_before[~np.isin(np.arange(len(y_before)), can["rows"])],
                                       self.data.y_train[~np.isin(np.arange(len(y_before)), can["rows"])]))
        held = {r for d in self.data.device_rows for r in self.data.device_rows[d].tolist()}
        for r, m, h in zip(can["rows"], can["member"], can["host"]):
            self.assertEqual(int(r) in held, bool(m))  # members inside, the others out
            if m:
                self.assertIn(int(r), self.data.device_rows[int(h)].tolist())
        self.assertEqual(sum(len(v) for v in self.data.device_rows.values()),
                         sum(len(v) for v in before.values()) - 50)
        again = self.c.draw(FakeData(n_dev=30, per_dev=10), seed=4)
        self.assertTrue(np.array_equal(again["rows"], can["rows"]) and np.array_equal(again["host"], can["host"]))

    def test_hosts_are_honest(self):
        can = self.c.draw(self.data, seed=1, honest=[3, 4])
        self.assertTrue(set(can["host"].tolist()) <= {3, 4})

    def test_gateway_view_scores_alignment(self):
        torch.manual_seed(0)
        tr = Trainer(tiny_model, "cpu")
        w = tr.get_vector().clone().numpy()
        self.c.apply(self.data, self.c.draw(self.data, seed=2))
        can = self.data.canaries
        i = int(np.flatnonzero(can["member"])[0])
        r = int(can["rows"][i])
        g = tr.example_gradient(w, self.data.x_train[r:r + 1], self.data.y_train[r:r + 1])
        scores_ = self.c.gateway_view(tr, self.data, 0, w, {int(can["host"][i]): (-g).numpy()})
        self.assertAlmostEqual(scores_[i], 1.0, places=5)  # the observed update is minus its gradient


class TestE1ConfigsM205(unittest.TestCase):
    """M2-05: 90 E1 configs from the selection files, and their run_pilot arguments."""

    @staticmethod
    def project_file(rel):
        return os.path.join(HERE, "..", "..", rel)

    @staticmethod
    def read(path, loader):
        with open(path, encoding="utf-8") as fh:
            return loader(fh)

    def setUp(self):
        import make_e1_configs
        self.m = make_e1_configs
        self.have_sources = all(os.path.exists(self.project_file(r)) for r in
                                (make_e1_configs.SEL_DL36, make_e1_configs.SEL_DL37, make_e1_configs.REQUIREMENTS))

    def test_ninety_configs_with_selected_values(self):
        if not self.have_sources:
            self.skipTest("selection files not present (job package)")
        import yaml
        with tempfile.TemporaryDirectory() as d:
            files = self.m.make(d, "abc1234")
            self.assertEqual(len(files), 90)
            cfgs = [self.read(f, yaml.safe_load) for f in files]
        self.assertEqual(len({c["run_id"] for c in cfgs}), 90)
        by_mode = {}
        for c in cfgs:
            by_mode.setdefault(c["mode"], []).append(c)
            self.assertEqual((c["code_commit"], c["canaries"], c["settings"].get("eps0", 8.0)), ("abc1234", True, 8.0))
            self.assertIn("torch", c["versions"])
        self.assertEqual({m: sorted(x["seed"] for x in v) for m, v in by_mode.items()},
                         {m: list(range(10)) for m in by_mode})
        sel = self.read(self.project_file(self.m.SEL_DL36), json.load)["pta_default_point"]
        pta = by_mode["pta"][0]["settings"]
        self.assertEqual((pta["lr"], pta["c_ex"], pta["clip_C"], pta["tau"], pta["quantize"]),
                         (sel["lr"], sel["c_ex"], sel["clip_C"], sel["tau"], True))
        self.assertNotIn("tau", by_mode["pfr"][0]["settings"])
        self.assertNotIn("clip_C", by_mode["ptg"][0]["settings"])
        base = self.read(self.project_file(self.m.SEL_DL37), json.load)
        self.assertEqual(by_mode["centralized"][0]["settings"]["local_epochs"], base["centralized"]["epoch"])
        self.assertEqual(by_mode["fedavg"][0]["settings"]["lr"], base["fedavg"]["lr"])

    def test_run_pilot_arguments(self):
        from run_experiment import build_argv
        cfg_path = os.path.join("/x", "configs", "E1", "E1_pta_s3.yaml")
        common = {"experiment": "E1", "configuration": "x", "seed": 3, "canaries": True,
                  "base_config": "../G0a.yaml", "code_commit": "abc", "versions": {}}
        pta = build_argv(dict(common, run_id="E1_pta_s3", mode="pta", settings={
            "lr": 0.01, "c_ex": 2.0, "eps0": 8.0, "rounds": 100, "clip_C": 15.3, "quantize": True, "tau": 17.4}),
            cfg_path, "/data", "/out")
        self.assertEqual(pta[pta.index("--config") + 1], os.path.join("/x", "configs", "G0a.yaml"))
        for flag, value in (("--clip-C", "15.3"), ("--tau", "17.4"), ("--eps0", "8.0"), ("--rounds", "100"),
                            ("--fold10-out", "/out/E1_pta_s3.fold10.sealed.json"), ("--seed", "3")):
            self.assertEqual(pta[pta.index(flag) + 1], value)
        self.assertIn("--quantize", pta)
        cen = build_argv(dict(common, run_id="c", mode="centralized", settings={"lr": 0.01, "local_epochs": 50}),
                         cfg_path, "/data", "/out", smoke_rounds=None)
        self.assertEqual(cen[cen.index("--local-epochs") + 1], "50")
        self.assertNotIn("--rounds", cen)
        self.assertNotIn("--c-ex", cen)
        fed = build_argv(dict(common, run_id="f", mode="fedavg", settings={"lr": 0.1, "rounds": 100}),
                         cfg_path, "/data", "/out", smoke_rounds=5)
        self.assertEqual(fed[fed.index("--rounds") + 1], "5")  # the smoke override wins
        self.assertNotIn("--eps0", fed)


class TestSummarizeE1M301(unittest.TestCase):
    """E1 summary: fold 9 only, never the sealed fold-10 files (DL-39); G1 preview paired by seed."""

    @staticmethod
    def run_json(configuration, mode, seed, f1):
        cfg = {"configuration": configuration, "mode": mode, "seed": seed, "settings": {"local_epochs": 100}}
        r = {"experiment_config": {"content": cfg}, "machine": {"device": "cuda"},
             "wan_bytes": {"uplink_per_device_round": 1.5e6}, "timing": {"wall_s": 300.0},
             "canaries": {"views": {"cloud": {"auc": 0.5, "eps_lb": 0.0}}}}
        if mode == "local":
            r["local_only"] = {"100": {"mean_macro_f1_tuned": f1, "mean_macro_auroc": 0.7}}
        else:
            r["final_fold9"] = {"macro_f1_tuned": f1, "macro_auroc": 0.8, "mi_sens_at_90spec": 0.5,
                                "backdoor_success_clean": 0.1}
        return r

    def test_fold9_only_and_g1_preview(self):
        from summarize_e1 import load_runs, summarize
        with tempfile.TemporaryDirectory() as d:
            for seed, (ta, tgc, lo) in enumerate([(0.52, 0.57, 0.50), (0.53, 0.56, 0.52)]):
                for name, mode, f1 in (("P-TA", "pta", ta), ("P-TG-c", "ptgc", tgc), ("local-only", "local", lo)):
                    with open(os.path.join(d, f"E1_{mode}_s{seed}.json"), "w", encoding="utf-8") as fh:
                        json.dump(self.run_json(name, mode, seed, f1), fh)
            with open(os.path.join(d, "E1_pta_s0.fold10.sealed.json"), "w", encoding="utf-8") as fh:
                fh.write("SEALED - if this is read, json.load fails")
            out = summarize(load_runs(d))
        self.assertEqual(out["runs"], 6)
        self.assertAlmostEqual(out["configurations"]["P-TA"]["macro_f1_tuned"]["mean"], 0.525)
        g1 = out["g1_preview"]
        self.assertEqual([round(x, 6) for x in g1["part1_pta_minus_ptgc_points"]["per_seed_points"]], [-5.0, -3.0])
        self.assertTrue(g1["part1_would_pass"])  # mean -4 points >= -10

    def test_smoke_run_reads_the_epoch_actually_run(self):
        # --smoke-rounds 2: local-only stores its scores under "2", not under local_epochs (100)
        from summarize_e1 import fold9_measures
        r = self.run_json("local-only", "local", 0, 0.4)
        r["local_only"] = {"2": r["local_only"].pop("100")}
        r["experiment_config"]["smoke_rounds"] = 2
        self.assertEqual(fold9_measures(r)["macro_f1_tuned"], 0.4)


class TestG1CheckM302(unittest.TestCase):
    """G1 recomputed from raw runs: DL-40 P-PE replaces E1 P-PE; the seal register keeps same-named files apart."""

    def test_load_and_seal_register(self):
        import g1_check

        def write(path, text):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)

        with tempfile.TemporaryDirectory() as d:
            for job, mode, f1 in (("JOB_e1", "ppe", 0.42), ("JOB_e1", "pta", 0.52), ("JOB_dl40", "ppe", 0.52)):
                cfg = {"configuration": {"ppe": "P-PE", "pta": "P-TA"}[mode], "mode": mode, "seed": 0}
                write(os.path.join(d, job, "runs", f"E1_{mode}_s0.json"),
                      json.dumps({"experiment_config": {"content": cfg}, "final_fold9": {"macro_f1_tuned": f1}}))
                write(os.path.join(d, job, "sealed_fold10", f"E1_{mode}_s0.fold10.sealed.json"), job)
            runs = g1_check.load(os.path.join(d, "JOB_e1", "runs"), os.path.join(d, "JOB_dl40", "runs"))
            self.assertEqual(runs["P-PE"][0]["final_fold9"]["macro_f1_tuned"], 0.52)  # the DL-40 run
            reg = g1_check.seal_register([os.path.join(d, j, "sealed_fold10") for j in ("JOB_e1", "JOB_dl40")], [])
        self.assertEqual(len(reg), 3)  # both P-PE seals are kept


class TestE1Fold10M303(unittest.TestCase):
    """Fold 10 (O1-O3): only final files, only with a matching seal, only with the run's fold-9 thresholds."""

    def build(self, d, thresholds_ok=True):
        import hashlib
        import e1_fold10 as m
        thr = {"NORM": 0.5}
        reg = {}
        for name, mode in m.CONFIGURATIONS:
            for seed in (0, 1):
                job = os.path.join(d, "JOB_dl40" if mode == "ppe" else "JOB_e1")
                runs = os.path.join(job, "e1_ppe_runs" if mode == "ppe" else "runs")
                run = {"run_id": f"E1_{mode}_s{seed}", "seed": seed, "mode": mode,
                       "final_fold9": {"macro_f1_tuned": 0.5 + 0.01 * seed, "thresholds": thr},
                       "experiment_config": {"content": {"settings": {"local_epochs": 100}}},
                       "local_only": {"100": {"mean_macro_f1_tuned": 0.5}},
                       "canaries": {"views": {"gateway": {"auc": 0.5}}}}
                if mode == "local":
                    body = {"local_only_fold10": {"mean_macro_f1_tuned": 0.49, "mean_macro_auroc": 0.7},
                            "devices": [{"mi_sens_at_90spec": 0.3}]}
                else:
                    body = {"final_fold10": {"macro_f1_tuned": 0.48 + 0.01 * seed, "macro_auroc": 0.75,
                                             "mi_sens_at_90spec": 0.3, "backdoor_success_clean": 0.2,
                                             "per_class_f1_tuned": {"NORM": 0.7},
                                             "thresholds": thr if thresholds_ok else {"NORM": 0.3},
                                             "thresholds_tuned_on_this_fold": False}}
                sealed = json.dumps(dict(run_id=run["run_id"], seed=seed, mode=mode, **body)).encode()
                os.makedirs(runs, exist_ok=True)
                os.makedirs(os.path.join(job, "sealed_fold10"), exist_ok=True)
                with open(os.path.join(runs, f"E1_{mode}_s{seed}.json"), "w", encoding="utf-8") as fh:
                    json.dump(run, fh)
                with open(os.path.join(job, "sealed_fold10", f"E1_{mode}_s{seed}.fold10.sealed.json"), "wb") as fh:
                    fh.write(sealed)
                reg[f"{os.path.basename(job)}/E1_{mode}_s{seed}.fold10.sealed.json"] = {
                    "sha256": hashlib.sha256(sealed).hexdigest()}
        # the superseded E1 P-PE files: not JSON, so reading them would fail
        for seed in (0, 1):
            with open(os.path.join(d, "JOB_e1", "sealed_fold10", f"E1_ppe_s{seed}.fold10.sealed.json"), "w",
                      encoding="utf-8") as fh:
                fh.write("SUPERSEDED - never open")
        return reg

    def test_rules(self):
        import e1_fold10 as m
        seeds = m.SEEDS
        m.SEEDS = [0, 1]
        try:
            with tempfile.TemporaryDirectory() as d:
                reg = self.build(d)
                with open(os.path.join(d, "JOB_e1", "sealed_fold10", "E1_pta_s1.fold10.sealed.json"), "ab") as fh:
                    fh.write(b" ")  # changed after sealing
                out = m.analyse(os.path.join(d, "JOB_e1"), os.path.join(d, "JOB_dl40"), reg)
                self.assertEqual(out["opened"], 17)
                self.assertEqual(out["missing"], [{"configuration": "P-TA", "seed": 1,
                                                   "reason": "SHA-256 differs from the seal register"}])
            with tempfile.TemporaryDirectory() as d:
                reg = self.build(d, thresholds_ok=False)
                with self.assertRaises(SystemExit):
                    m.analyse(os.path.join(d, "JOB_e1"), os.path.join(d, "JOB_dl40"), reg)
        finally:
            m.SEEDS = seeds


class TestDL40CloudStep(unittest.TestCase):
    """DL-40: P-PE's cloud step eta_c (configs, arguments, selection rule)."""

    def test_cloud_step_grows_with_eta_and_stays_under_the_bound(self):
        g = np.random.default_rng(1)
        w_star = g.normal(0, 1, 50).astype(np.float32)
        edges = {e: w_star + g.normal(0, 0.3, 50).astype(np.float32) for e in range(5)}
        dist = [np.linalg.norm(edges[e].astype(np.float64) - w_star) for e in edges]
        steps = []
        for eta in (0.01, 0.1, 1.0, 3.0):
            new, info = hflmld.cloud_aggregate(w_star, edges, eta)
            self.assertAlmostEqual(info["step"], float(np.linalg.norm(new.astype(np.float64) - w_star)), places=5)
            self.assertLessEqual(info["step"], max(eta * np.log1p(d / eta) for d in dist) + 1e-6)  # eq. (8)
            steps.append(info["step"])
        self.assertEqual(steps, sorted(steps))

    def test_arguments(self):
        from run_experiment import build_argv
        common = {"experiment": "DL-40 grid", "configuration": "P-PE", "seed": 0, "canaries": True, "mode": "ppe",
                  "base_config": "../G0a.yaml", "code_commit": "abc", "versions": {}}
        s = {"lr": 0.01, "c_ex": 2.0, "eps0": 8.0, "rounds": 100}
        grid = build_argv(dict(common, run_id="g", fold10="none", settings=dict(s, cloud_eta=0.3)),
                          "/x/configs/DL40_grid/g.yaml", "/data", "/out")
        self.assertEqual(grid[grid.index("--cloud-eta") + 1], "0.3")
        self.assertNotIn("--fold10-out", grid)  # grid runs never score fold 10
        e1 = build_argv(dict(common, run_id="E1_ppe_s0", fold10="sealed (DL-39)", settings=s),
                        "/x/configs/E1/E1_ppe_s0.yaml", "/data", "/out")
        self.assertNotIn("--cloud-eta", e1)  # the E1 configs keep the paper's eta (= lr)
        self.assertIn("--fold10-out", e1)
        import contextlib
        import io
        import run_pilot
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):  # eta_c is for P-PE only
            run_pilot.main(["--config", "x", "--ptbxl-dir", "y", "--mode", "pta", "--seed", "0", "--lr", "0.01",
                            "--c-ex", "2", "--cloud-eta", "0.3", "--out", "z"])

    def test_grid_and_rerun_configs_change_only_eta(self):
        import yaml
        import make_dl40_configs as m

        def load(path):
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        base = {"experiment": "E1", "configuration": "P-PE", "mode": "ppe", "canaries": True,
                "settings": {"lr": 0.01, "c_ex": 2.0, "eps0": 8.0, "rounds": 100}, "fold10": "sealed (DL-39)",
                "base_config": "../G0a.yaml", "rules": ["DL-39"], "code_commit": "old", "versions": {"torch": "x"},
                "sources_sha256": {"sel.json": "aa"}}
        with tempfile.TemporaryDirectory() as d:
            e1 = os.path.join(d, "E1")
            os.makedirs(e1)
            for seed in range(10):
                with open(os.path.join(e1, f"E1_ppe_s{seed}.yaml"), "w", encoding="utf-8") as fh:
                    yaml.safe_dump(dict(base, run_id=f"E1_ppe_s{seed}", seed=seed), fh)
            grid = [load(p) for p in m.make_grid(e1, os.path.join(d, "g"), "new")]
            self.assertEqual(len(grid), 30)
            self.assertEqual(sorted({c["settings"]["cloud_eta"] for c in grid}), list(m.ETAS))
            for c in grid:
                self.assertEqual((c["fold10"], c["code_commit"], c["experiment"]), ("none", "new", "DL-40 grid"))
                self.assertEqual({k: v for k, v in c["settings"].items() if k != "cloud_eta"}, base["settings"])
            sel = os.path.join(d, "selection_dl40.json")
            with open(sel, "w", encoding="utf-8") as fh:
                json.dump({"selected_cloud_eta": 0.3}, fh)
            rerun = [load(p) for p in m.make_rerun(e1, sel, os.path.join(d, "r"), "new")]
            self.assertEqual([c["run_id"] for c in rerun], [f"E1_ppe_s{s}" for s in range(10)])
            self.assertTrue(all(c["settings"]["cloud_eta"] == 0.3 and c["fold10"] == "sealed (DL-39)" for c in rerun))
            with open(sel, "w", encoding="utf-8") as fh:
                json.dump({"selected_cloud_eta": None}, fh)
            with self.assertRaises(SystemExit):  # stop rule: no rerun
                m.make_rerun(e1, sel, os.path.join(d, "r2"), "new")

    @staticmethod
    def grid_run(eta, seed, f1, smoke=None):
        cfg = {"experiment": "DL-40 grid", "mode": "ppe", "seed": seed, "settings": {"cloud_eta": eta}}
        return (f"DL40_{eta}_{seed}.json", "sha", {
            "experiment_config": {"content": cfg, "smoke_rounds": smoke},
            "final_fold9": {"macro_f1_tuned": f1, "macro_auroc": 0.7}, "data": {"trivial_macro_f1_fold9": 0.402},
            "rounds": [{"cloud_step": 0.1, "edge_distance_mean": 1.3}]})

    def test_selection_rule(self):
        from select_ppe_eta import select
        means = {0.01: 0.42, 0.03: 0.45, 0.1: 0.50, 0.3: 0.515, 1.0: 0.52, 3.0: 0.518}
        runs = [self.grid_run(e, s, f + 0.001 * (s - 1)) for e, f in means.items() for s in range(3)]
        out = select(runs, 3)
        self.assertEqual((out["best_cloud_eta"], out["selected_cloud_eta"]), (1.0, 0.3))  # smallest within 1 point
        flat = [self.grid_run(e, s, 0.43) for e in means for s in range(3)]
        self.assertIsNone(select(flat, 3)["selected_cloud_eta"])  # stop rule: best < trivial + 5 points
        with self.assertRaises(SystemExit):
            select(runs[:-1], 3)  # one seed missing
        self.assertIsNone(select(runs[:-1], 3, partial=True)["selected_cloud_eta"])  # a table only
        with self.assertRaises(SystemExit):
            select([self.grid_run(0.1, 0, 0.5, smoke=5)], 1)  # a smoke run cannot select


class TestHFLMLD(unittest.TestCase):
    def test_edge_detection_flags_the_small_far_cluster(self):
        g = np.random.default_rng(0)
        models = g.normal(0, 0.01, (10, 200)) + np.linspace(-1, 1, 200)
        models[[3, 7], :100] += 0.8  # two shifted clients
        v, info = hflmld.edge_detect(models, seed=0)
        self.assertEqual(list(np.flatnonzero(v)), [3, 7])
        self.assertEqual(info["cluster_sizes"], [2, 8])

    def test_equal_clusters_flag_nobody(self):
        models = np.vstack([np.zeros((3, 50)), np.ones((3, 50))]) + np.linspace(0, 1, 50)
        models[3:, :25] += 5
        v, _ = hflmld.edge_detect(models, seed=0)
        self.assertEqual(int(v.sum()), 0)

    def test_suspicion_score_rule(self):
        s = hflmld.update_suspicion(np.zeros(2, dtype=int), np.array([1, 0]))
        self.assertEqual(list(s), [1, -1])
        self.assertEqual(list(hflmld.admitted(s, np.array([1, 0]))), [False, True])
        s = hflmld.update_suspicion(s, np.array([0, 0]))  # eq. (6): +1 -1 = 0 -> admitted again when benign
        self.assertEqual(list(hflmld.admitted(s, np.array([0, 0]))), [True, True])

    def test_cloud_rule_matches_equations(self):
        w_star = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        edges = {0: w_star + np.array([0.1, 0.0, 0.0]), 1: w_star + np.array([0.0, 0.2, 0.0]),
                 2: w_star + np.array([0.0, 0.0, 0.4])}
        eta = 0.1
        new, info = hflmld.cloud_aggregate(w_star, edges, eta)
        w = np.stack([edges[e] for e in (0, 1, 2)]).astype(np.float64)
        g = (w_star - w) / eta
        nrm = np.linalg.norm(g, axis=1)
        g_hat = np.log1p(nrm)[:, None] * g / nrm[:, None]
        cs = w @ w_star / (np.linalg.norm(w, axis=1) * np.linalg.norm(w_star))
        euc = np.linalg.norm(w - w_star, axis=1)
        cr = np.maximum(cs * (1 - (euc - euc.min()) / (euc.max() - euc.min())), 0)
        self.assertEqual(info["weights"][2], 0.0)  # the farthest edge model always gets weight 0
        want = w_star - eta * (cr[:, None] * g_hat).sum(0) / cr.sum()
        self.assertTrue(np.allclose(new, want, atol=1e-6))


class TestAggregation(unittest.TestCase):
    def test_quantization_error_and_no_overflow(self):
        g = torch.Generator().manual_seed(0)
        v = torch.randn(10_000, generator=g)
        c = float(v.norm())
        ints, scale = quantize(v, c)
        self.assertLessEqual(float((ints.float() / scale - v).abs().max()), c / (2 ** 15 - 1))
        big = {d: torch.full((100,), c) for d in range(5)}  # every coordinate at the clip bound
        sums, _ = group_sums(big, [list(range(5))], c_clip=c * 10, quant=True)  # k = 5 (largest k in E6)
        self.assertTrue(torch.allclose(sums[0], torch.full((100,), 5 * c), rtol=1e-4))

    def test_group_sums_and_tau_and_median(self):
        vecs = {d: torch.full((4,), float(d)) for d in range(9)}
        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]  # sums 3, 12, 21 per coordinate
        sums, norms = group_sums(vecs, groups)
        self.assertEqual([float(s[0]) for s in sums], [3.0, 12.0, 21.0])
        out, gnorms, dropped = gateway_median(sums, k=3, tau=30.0)  # norms 6, 24, 42 -> drop the last
        self.assertEqual(dropped, [2])
        self.assertAlmostEqual(float(out[0]), (1.0 + 4.0) / 2)  # median of {1, 4} (even count: mean)
        out_all, _, dropped_none = gateway_median(sums, k=3, tau=None)
        self.assertEqual(dropped_none, [])
        self.assertAlmostEqual(float(out_all[0]), 4.0)

    def test_median_odd_even(self):
        s = torch.tensor([[1.0], [5.0], [3.0]])
        self.assertEqual(float(median_lower_upper_mean(s)[0]), 3.0)
        self.assertEqual(float(median_lower_upper_mean(torch.tensor([[1.0], [2.0], [6.0], [9.0]]))[0]), 4.0)

    def test_gateway_check_normalization_dl10(self):
        g = torch.Generator().manual_seed(1)
        d, full = 53_765, 10
        # gateway output noise std ~ 1/sqrt(groups used); gateway 3 used 9 of 10 groups
        used = {i: (9 if i == 3 else 10) for i in range(7)}
        outs = {i: torch.randn(d, generator=g) / np.sqrt(used[i]) for i in range(7)}
        flagged, _, _ = gateway_check(outs, used, {i: full for i in range(7)}, {i: 0.0 for i in range(7)})
        self.assertNotIn(3, flagged)
        flagged_rate, _, _ = gateway_check(outs, used, {i: full for i in range(7)},
                                           {i: (0.2 if i == 5 else 0.0) for i in range(7)})
        self.assertIn(5, flagged_rate)  # drop rate above 10% (DL-10, PR:75)

    def test_draw_groups(self):
        a = draw_groups(0, 1, 2, list(range(60, 90)), 3)
        self.assertEqual(len(a), 10)
        self.assertEqual(sorted(x for grp in a for x in grp), list(range(60, 90)))
        self.assertEqual(a, draw_groups(0, 1, 2, list(range(60, 90)), 3))
        self.assertNotEqual(a, draw_groups(0, 2, 2, list(range(60, 90)), 3))


class TestMetricsAndConfig(unittest.TestCase):
    def test_macro_f1_matches_sklearn(self):
        from sklearn.metrics import f1_score
        g = np.random.default_rng(0)
        y = (g.random((200, 5)) < 0.3).astype(int)
        p = g.random((200, 5))
        self.assertAlmostEqual(scores(y, p)["macro_f1"],
                               f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0))

    def test_trivial_classifier_formula(self):
        y = np.array([[1, 0], [0, 0], [1, 1], [0, 0]])  # prevalence 0.5 and 0.25
        self.assertAlmostEqual(trivial_macro_f1(y), np.mean([2 * 0.5 / 1.5, 2 * 0.25 / 1.25]))

    def test_noise_multiplier_from_csv(self):
        from run_pilot import noise_multiplier
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            w = csv.DictWriter(fh, fieldnames=["epsilon", "delta", "q", "total_steps", "noise_multiplier_z"])
            w.writeheader()
            w.writerow({"epsilon": 3.0, "delta": 1e-5, "q": 0.226675, "total_steps": 500, "noise_multiplier_z": 7.6758})
        try:
            self.assertEqual(noise_multiplier(fh.name, 3.0, 0.226675, 500), (7.6758, 1e-5))
            with self.assertRaises(SystemExit):
                noise_multiplier(fh.name, 3.0, 0.2, 500)
        finally:
            os.unlink(fh.name)


class TestDiagnosticsDL35(unittest.TestCase):
    def test_tuned_threshold_separates_and_keeps_empty_class(self):
        y = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
        p = np.array([[0.9, 0.2], [0.7, 0.8], [0.3, 0.1], [0.1, 0.6]])
        t = tuned_thresholds(y, p)
        self.assertGreater(t[0], 0.3)  # any threshold in (0.3, 0.7] separates class 0
        self.assertLessEqual(t[0], 0.7)
        self.assertAlmostEqual(t[0], 0.51)  # middle of the 40 best grid points 0.31-0.70
        self.assertEqual(t[1], 0.5)  # no positive record: keeps the DL-33 threshold

    def test_fixed_thresholds_match_scores(self):
        g = np.random.default_rng(1)
        y = (g.random((300, 5)) < 0.3).astype(int)
        p = g.random((300, 5))
        self.assertAlmostEqual(macro_f1_at(y, p, np.full(5, 0.5)), scores(y, p)["macro_f1"])

    def test_fold9_thresholds_are_an_upper_bound(self):
        g = np.random.default_rng(2)
        y = (g.random((300, 5)) < 0.2).astype(int)
        p = np.clip(0.3 * y + 0.7 * g.random((300, 5)), 0, 1)
        d = diagnostic_scores(y, p, y[:100], p[:100])
        self.assertGreaterEqual(d["macro_f1_fold9_thresholds"], scores(y, p)["macro_f1"])
        self.assertGreaterEqual(d["macro_f1_fold9_thresholds"], d["macro_f1_train_thresholds"])
        self.assertEqual(set(d["per_class_auroc"]), {"NORM", "MI", "STTC", "CD", "HYP"})


class TestMetricDL36(unittest.TestCase):
    def setUp(self):
        g = np.random.default_rng(3)
        self.y = (g.random((400, 5)) < 0.2).astype(int)
        self.p = np.clip(0.25 * self.y + 0.6 * g.random((400, 5)), 0, 1)

    def test_tuned_on_this_fold(self):
        res = scores(self.y, self.p)
        t = tuned_thresholds(self.y, self.p)
        self.assertTrue(res["thresholds_tuned_on_this_fold"])
        self.assertEqual(list(res["thresholds"].values()), list(t))
        self.assertAlmostEqual(res["macro_f1_tuned"], macro_f1_at(self.y, self.p, t))
        self.assertGreaterEqual(res["macro_f1_tuned"], res["macro_f1"])  # 0.5 is on the grid
        self.assertEqual(set(res["per_class_auroc"]), {"NORM", "MI", "STTC", "CD", "HYP"})

    def test_given_thresholds_are_applied(self):
        # fold 10 uses the fold-9 thresholds; with 0.5 everywhere it must equal the DL-33 value
        res = scores(self.y, self.p, thresholds=np.full(5, 0.5))
        self.assertFalse(res["thresholds_tuned_on_this_fold"])
        self.assertAlmostEqual(res["macro_f1_tuned"], res["macro_f1"])


class TestSelectGridDL36(unittest.TestCase):
    @staticmethod
    def write(folder, name, obj):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh)

    def pta(self, lr, c_ex, f1_05, f1_tuned):
        return {"settings": {"lr": lr, "c_ex": c_ex},
                "final_fold9": {"macro_f1": f1_05, "macro_f1_tuned": f1_tuned, "macro_auroc": 0.7},
                "norms": {"device_update": {"p99": 10 * lr}, "group_sum": {"p99": 20 * lr}}}

    def local(self, lr, tuned_by_epoch):
        return {"settings": {"lr": lr}, "local_only": {
            str(e): {"mean_macro_f1_tuned": v, "mean_macro_f1": 0.9 - v, "mean_macro_auroc": 0.7}
            for e, v in tuned_by_epoch.items()}}

    def test_selects_by_tuned_f1_and_refuses_old_runs(self):
        from select_grid import select
        with tempfile.TemporaryDirectory() as d:
            # the 0.5-threshold ranking and the DL-36 ranking disagree on purpose
            self.write(d, "pta_grid_lr0.01_cex0.5_s0.json", self.pta(0.01, 0.5, 0.30, 0.40))
            self.write(d, "pta_grid_lr0.1_cex1.0_s0.json", self.pta(0.1, 1.0, 0.10, 0.50))
            self.write(d, "local_grid_lr0.01_s0.json", self.local(0.01, {10: 0.30, 100: 0.20}))
            self.write(d, "local_grid_lr0.1_s0.json", self.local(0.1, {10: 0.25, 100: 0.35}))
            sel = select(d)
            self.assertEqual((sel["pta_default_point"]["lr"], sel["pta_default_point"]["c_ex"]), (0.1, 1.0))
            self.assertAlmostEqual(sel["pta_default_point"]["clip_C"], 1.0)
            self.assertAlmostEqual(sel["pta_default_point"]["tau"], 2.0)
            self.assertEqual((sel["local_only"]["lr"], sel["local_only"]["epoch"]), (0.1, 100))
            old = self.pta(0.05, 0.5, 0.2, 0.0)
            del old["final_fold9"]["macro_f1_tuned"]
            self.write(d, "pta_grid_lr0.05_cex0.5_s0.json", old)
            with self.assertRaises(SystemExit):
                select(d)


class TestSelectBaselinesDL37(unittest.TestCase):
    @staticmethod
    def write(folder, name, obj):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh)

    @staticmethod
    def sc(tuned, f05=0.3):
        return {"macro_f1_tuned": tuned, "macro_f1": f05, "macro_auroc": 0.8}

    def fl(self, lr, tuned, f05):
        return {"settings": {"lr": lr, "rounds": 100, "edge_iterations": 5}, "final_fold9": self.sc(tuned, f05)}

    def test_selection_and_refusal(self):
        from select_baselines import select
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "centralized_grid_lr0.01_s0.json", {"settings": {"lr": 0.01}, "centralized": {
                "10": self.sc(0.50), "100": self.sc(0.60)}})
            self.write(d, "centralized_grid_lr0.1_s0.json", {"settings": {"lr": 0.1}, "centralized": {
                "10": self.sc(0.60), "100": self.sc(0.55)}})  # tie at 0.60: smaller lr wins
            for mode in ("fedavg", "hierfavg"):
                self.write(d, f"{mode}_grid_lr0.01_s0.json", self.fl(0.01, 0.52, 0.40))  # best at 0.5, not DL-36
                self.write(d, f"{mode}_grid_lr0.05_s0.json", self.fl(0.05, 0.58, 0.20))
            sel = select(d)
            self.assertEqual((sel["centralized"]["lr"], sel["centralized"]["epoch"]), (0.01, 100))
            self.assertEqual(sel["fedavg"]["lr"], 0.05)
            self.assertEqual(sel["hierfavg"]["lr"], 0.05)
            old = self.fl(0.1, 0.0, 0.1)
            del old["final_fold9"]["macro_f1_tuned"]
            self.write(d, "fedavg_grid_lr0.1_s0.json", old)
            with self.assertRaises(SystemExit):
                select(d)


class TestOfficialNoiseCSV(unittest.TestCase):
    def test_eps3_and_eps8_rows(self):
        # G0a_noise_multipliers.csv (Opacus 1.6.0): eps0 = 3 (DL-15) and eps = 8 (DL-34)
        from run_pilot import accountant_epsilon, noise_multiplier
        candidates = [os.path.join(HERE, "..", "..", "05_Experiments", "results", "G0", "G0a_noise_multipliers.csv"),
                      os.path.join(HERE, "..", "..", "inputs", "G0a_noise_multipliers.csv")]
        path = next(p for p in candidates if os.path.exists(p))
        for eps, z_expected in ((3.0, 7.6758), (8.0, 3.3545)):
            z, delta = noise_multiplier(path, eps, 0.226675, 500)
            self.assertAlmostEqual(z, z_expected, places=4)
            self.assertLessEqual(accountant_epsilon(z, 0.226675, 500, delta), eps)


@unittest.skipUnless(os.environ.get("PTBXL_DIR"), "set PTBXL_DIR to run tests on real PTB-XL")
class TestRealData(unittest.TestCase):
    def test_trivial_fold9_and_partition(self):
        from train_data import RunData
        candidates = [os.path.join(HERE, "..", "..", "05_Experiments", "configs", "partition_main.yaml"),  # project
                      os.path.join(HERE, "..", "..", "configs", "partition_main.yaml")]  # job package
        cfg = next(p for p in candidates if os.path.exists(p))
        data = RunData(os.environ["PTBXL_DIR"], cfg, seed=0)
        self.assertAlmostEqual(trivial_macro_f1(data.y_val), 0.402, places=3)  # data_versions.md
        self.assertEqual(len(data.device_rows), 210)
        self.assertEqual(sum(len(r) for r in data.device_rows.values()), 14_823)
        self.assertEqual(data.plan["total_steps"], 500)


if __name__ == "__main__":
    unittest.main()
