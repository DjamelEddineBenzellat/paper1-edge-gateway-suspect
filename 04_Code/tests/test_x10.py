"""Unit tests for the E4 and E6 fold-10 scripts (plan task X-10). Standard-library unittest; synthetic job folders.

Run:  <venv>/bin/python -m unittest discover -s 04_Code/tests -v
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("fl_system", "defences", "attacks"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

import e4_fold10 as e4  # noqa: E402
import e6_fold10 as e6  # noqa: E402
from g1_check import seal_register  # noqa: E402

THR = {"NORM": 0.5}
NOISE = (-0.002, 0.0, 0.002)


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def run_pair(runs_dir, sealed_dir, run_id, mode, seed, f1, backdoor=0.25, extra=None):
    write(os.path.join(runs_dir, run_id + ".json"),
          dict({"run_id": run_id, "seed": seed, "mode": mode, "final_fold9": {"thresholds": THR}}, **(extra or {})))
    write(os.path.join(sealed_dir, run_id + ".fold10.sealed.json"),
          {"run_id": run_id, "seed": seed, "mode": mode,
           "final_fold10": {"macro_f1_tuned": f1, "macro_auroc": 0.75, "mi_sens_at_90spec": 0.3,
                            "backdoor_success_clean": backdoor, "per_class_f1_tuned": {"NORM": 0.7},
                            "thresholds": THR, "thresholds_tuned_on_this_fold": False}})


def build(d, k5_shift=0.03):
    jobs = {n: os.path.join(d, n) for n in ("JOB_e1", "JOB_dl40", "JOB_dl42", "JOB_e6e4")}
    e1, dl40, dl42, job = (jobs[n] for n in ("JOB_e1", "JOB_dl40", "JOB_dl42", "JOB_e6e4"))
    for s in range(10):
        for mode in ("ptg", "pfr"):
            run_pair(os.path.join(e1, "runs"), os.path.join(e1, "sealed_fold10"), f"E1_{mode}_s{s}", mode, s, 0.51)
        run_pair(os.path.join(dl40, "e1_ppe_runs"), os.path.join(dl40, "sealed_fold10"), f"E1_ppe_s{s}", "ppe", s, 0.51)
        run_pair(os.path.join(dl42, "tuned_runs"), os.path.join(dl42, "sealed_fold10"), f"E1tuned_ptgc_s{s}", "ptgc", s,
                 0.60)
        run_pair(os.path.join(job, "e4_reference_runs"), os.path.join(job, "sealed_fold10"), f"E4ref_pta_s{s}", "pta",
                 s, 0.51)
        n = NOISE[s % 3]
        attacked_f1 = {"pta": 0.49 + n, "pfr": 0.495 - n, "ptg": 0.43, "ppe": 0.505, "ptgc": 0.45}  # DL-49
        for mode, f1 in attacked_f1.items():
            for attack in e4.ATTACKS:
                extra = {"attack": {"kind": attack, "norm": 15.31, "poisoned_sums": 10, "poisoned_dropped_at_tau": 2}}
                run_pair(os.path.join(job, "e4_runs"), os.path.join(job, "sealed_fold10"), f"E4_{mode}_{attack}_s{s}",
                         mode, s, f1, extra=extra)
    for s in range(3):
        n = NOISE[s]
        values = {"E6_pta_eps1": 0.40, "E6_pta_eps3": 0.47, "E6_ptgc_eps1": 0.55, "E6_ptgc_eps3": 0.58,
                  "E6_pta_k1": 0.515 + n, "E6_pta_k2": 0.512, "E6_pta_k5": 0.51 - k5_shift + n / 2,
                  "E6_pfr_k1": 0.50, "E6_pfr_k2": 0.505, "E6_pfr_k5": 0.49}
        for name, f1 in values.items():
            mode = name.split("_")[1]
            run_pair(os.path.join(job, "e6_runs"), os.path.join(job, "sealed_fold10"), f"{name}_s{s}", mode, s, f1,
                     extra={"gateway_check": {"dropped_group_sums": 7, "group_sums": 700}})
    write(os.path.join(job, "positive_control_dl49.json"), {"passes": True})
    regs = {}
    for folder in (e1, dl40, dl42, job):
        regs.update(seal_register([os.path.join(folder, "sealed_fold10")], []))
    return e1, dl40, dl42, job, regs


class TestE4Fold10(unittest.TestCase):
    """DL-49: label flips; backdoor tests not evaluable; secondary tests S1 and S2 on the macro-F1 drop."""

    def test_hypotheses_and_measures(self):
        with tempfile.TemporaryDirectory() as d:
            e1, dl40, dl42, job, regs = build(d)
            out = e4.analyse(job, job, e1, dl40, dl42, regs)
        self.assertEqual(out["missing"], [])
        self.assertEqual(out["attack_norm"], 15.31)
        self.assertTrue(out["h5a_backdoor"]["outcome"].startswith("not evaluable"))
        self.assertTrue(out["dl24"]["outcome"].startswith("not evaluable"))
        s1 = out["s1"]  # P-TA drop 2 - 100n, P-FR drop 1.5 + 100n; mean n over 10 seeds = -0.0002
        self.assertAlmostEqual(s1["mean"], 0.54, places=6)
        self.assertEqual(s1["outcome"], "equivalent")  # within +-3 points
        self.assertTrue(out["s2"]["P-PE"]["lower_than_pta"])  # drop 0.5 vs about 2 points
        self.assertFalse(out["s2"]["P-TG"]["lower_than_pta"])  # drop 8 points
        self.assertIsNotNone(out["s2"]["P-TG"]["p_holm"])
        pta = out["designs"]["P-TA"]["labelflip"]
        self.assertAlmostEqual(pta["f1_drop_points"]["mean"], 2.02, places=6)  # 0.51 - (0.49 + n)
        self.assertAlmostEqual(pta["passed_tau_share"], 0.8)
        self.assertIn("concentrated_labelflip", out["designs"]["P-FR"])
        self.assertNotIn("passed_tau_share", out["designs"]["P-TG"]["labelflip"])  # no groups
        self.assertEqual(out["positive_control"], {"passes": True})

    def test_missing_runs_bad_seals_and_mixed_norms(self):
        with tempfile.TemporaryDirectory() as d:
            e1, dl40, dl42, job, regs = build(d)
            os.remove(os.path.join(job, "e4_runs", "E4_ptg_labelflip_s5.json"))  # a skipped run
            with open(os.path.join(job, "sealed_fold10", "E4_pfr_concentrated_labelflip_s2.fold10.sealed.json"),
                      "a") as fh:
                fh.write(" ")  # changed after the register
            out = e4.analyse(job, job, e1, dl40, dl42, regs)
            reasons = {(m["configuration"], m["seed"]): m["reason"] for m in out["missing"]}
            self.assertEqual(reasons, {("P-TG labelflip", 5): "run file missing",
                                       ("P-FR concentrated_labelflip", 2): "SHA-256 differs from the seal register"})
            path = os.path.join(job, "e4_runs", "E4_pta_labelflip_s0.json")
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
            r["attack"]["norm"] = 20.0
            write(path, r)
            with self.assertRaises(SystemExit):  # DL-43 (3): one attack norm
                e4.analyse(job, job, e1, dl40, dl42, regs)

    def test_holm_and_tost(self):
        self.assertEqual(e4.holm({"a": 0.01, "b": 0.04, "c": None}), {"a": 0.02, "b": 0.04})
        p = {"sd": 1.0, "ci90": [2.0, 4.0], "ci95": [1.5, 4.5]}
        self.assertEqual(e4.tost(p, 3.0, higher_is_worse=True), "P-TA worse")  # higher backdoor success
        self.assertEqual(e4.tost(p, 3.0, higher_is_worse=False), "inconclusive")


class TestE6Fold10(unittest.TestCase):
    def test_h3b_and_h3c(self):
        with tempfile.TemporaryDirectory() as d:
            e1, dl40, dl42, job, regs = build(d)
            out = e6.analyse(job, e1, dl40, dl42, regs, trivial=0.30)
            low = e6.analyse(job, e1, dl40, dl42, regs, trivial=0.39)
        self.assertEqual(out["missing"], [])
        gaps = out["h3b"]["mean_gap_points"]
        self.assertAlmostEqual(gaps["1"], 15.0, places=6)
        self.assertAlmostEqual(gaps["3"], 11.0, places=6)
        self.assertAlmostEqual(gaps["8"], 9.0, places=6)
        self.assertTrue(out["h3b"]["evaluable"] and out["h3b"]["holds"])
        self.assertEqual((low["h3b"]["evaluable"], low["h3b"]["holds"]), (False, None))  # DL-11
        per_k = out["h3c"]["per_k"]
        self.assertAlmostEqual(per_k["5"]["mean"], 3.0, delta=0.2)  # P-TA(3) - P-TA(5)
        self.assertTrue(per_k["5"]["fails"])
        self.assertFalse(per_k["1"]["fails"])  # |difference| below 1.5
        self.assertTrue(out["h3c"]["fails"])
        self.assertAlmostEqual(out["pta_tau_drop_rate"]["E6_pta_k5_s0"], 0.01)

    def test_h3c_holds_when_k_does_not_matter(self):
        with tempfile.TemporaryDirectory() as d:
            e1, dl40, dl42, job, regs = build(d, k5_shift=0.005)
            out = e6.analyse(job, e1, dl40, dl42, regs, trivial=0.30)
        self.assertFalse(out["h3c"]["fails"])


def build_e5(d):
    import e5_fold10 as e5
    job = os.path.join(d, "JOB_e5")
    f1_of = {"scale": 0.45, "signflip": 0.50, "inject": 0.46, "drop": 0.47, "adaptive": 0.48, None: 0.51}
    for det in e5.DETECTORS:
        for kind in e5.TYPES + (None,):
            for s in range(10):
                g = s % 7
                rounds = []
                for rnd in range(1, 6):
                    flagged = []
                    if kind in ("scale", "drop") or (kind == "inject" and rnd != 3) or (kind == "adaptive" and rnd == 1):
                        flagged.append(g)
                    if kind is not None and rnd == 2:
                        flagged.append((g + 1) % 7)  # one false alarm per tampering run
                    if kind is None and rnd == 1:
                        flagged.append(0)  # one false alarm per run without tampering
                    rec = {"round": rnd, "flagged": sorted(flagged)}
                    if det == "ours":
                        rec["flagged_dl10"] = sorted({0, 2} | ({g} if kind == "scale" else set()))
                    rounds.append(rec)
                run_id = f"E5_{det}_{kind or 'none'}_s{s}"
                settings = {"tamper": kind} if kind else {}
                extra = {"rounds": rounds, "experiment_config": {"content": {"detector": det, "settings": settings}}}
                if kind:
                    extra["tampering"] = {"kind": kind, "gateway": g, "detector": det}
                run_pair(os.path.join(job, "e5_runs"), os.path.join(job, "sealed_fold10"), run_id,
                         "ppe" if det == "hflmld_native" else "pta", s, f1_of[kind], backdoor=0.25, extra=extra)
    return job, seal_register([os.path.join(job, "sealed_fold10")], [])


class TestE5Fold10(unittest.TestCase):
    def test_h4_detection_false_alarms_and_harmless_misses(self):
        import e5_fold10 as e5
        with tempfile.TemporaryDirectory() as d:
            job, regs = build_e5(d)
            out = e5.analyse(job, regs)
        self.assertEqual(out["missing"], [])
        t = out["detectors"]["ours"]["types"]
        self.assertEqual((t["scale"]["detection"]["rate"], t["signflip"]["detection"]["rate"]), (1.0, 0.0))
        self.assertAlmostEqual(t["inject"]["detection"]["rate"], 0.8)
        self.assertEqual(t["adaptive"]["detection"]["n"], 40)  # 4 scored rounds x 10 seeds (round 1 not scored)
        self.assertEqual(t["adaptive"]["detection"]["k"], 0)
        self.assertEqual(t["signflip"]["harmless_share_of_misses"], 1.0)  # 1 point lost, same backdoor rate
        self.assertEqual(t["inject"]["harmless_share_of_misses"], 0.0)  # 5 points lost
        self.assertAlmostEqual(t["adaptive"]["largest_undetected_f1_drop_points"], 3.0)
        self.assertEqual(t["scale"]["detection_dl10"]["rate"], 1.0)
        fa = out["detectors"]["ours"]["false_alarms"]
        self.assertEqual((fa["pooled_all_runs"]["k"], fa["pooled_all_runs"]["n"]), (60, 1850))
        self.assertEqual((fa["without_tampering"]["k"], fa["without_tampering"]["n"]), (10, 350))
        lo, hi = fa["pooled_all_runs"]["wilson95"]
        self.assertTrue(lo < 60 / 1850 < hi)
        self.assertEqual(out["h4"]["detection_ok_per_type"], {"scale": True, "signflip": False, "inject": False,
                                                              "drop": True})
        self.assertTrue(out["h4"]["false_alarms_ok"])
        self.assertFalse(out["h4"]["holds"])
        self.assertGreater(fa["dl10_pooled_all_runs"]["rate"], 0.25)  # gateways 0 and 2 flagged every round

    def test_wrong_runs_are_refused_and_missing_counted(self):
        import e5_fold10 as e5
        with tempfile.TemporaryDirectory() as d:
            job, regs = build_e5(d)
            os.remove(os.path.join(job, "e5_runs", "E5_hflmld_cloud_drop_s4.json"))
            out = e5.analyse(job, regs)
            self.assertIn({"configuration": "hflmld_cloud drop", "seed": 4, "reason": "run file missing"},
                          out["missing"])
            path = os.path.join(job, "e5_runs", "E5_ours_scale_s1.json")
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
            r["tampering"]["kind"] = "inject"
            write(path, r)
            with self.assertRaises(SystemExit):  # the record must match the config
                e5.analyse(job, regs)
        self.assertEqual(e5.wilson(0, 0), None)


if __name__ == "__main__":
    unittest.main()
