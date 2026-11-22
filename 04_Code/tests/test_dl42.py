"""Unit tests for the DL-42 code: per-design fold-9 tuning (plan task X-1). Standard-library unittest.

Run:  <venv>/bin/python -m unittest discover -s 04_Code/tests -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fl_system"))
sys.path.insert(0, os.path.join(HERE, "..", "defences"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import make_dl42_configs as mk  # noqa: E402
import select_dl42 as sel  # noqa: E402

BASE = {"experiment": "E1", "canaries": True, "fold10": "sealed (DL-39)", "base_config": "../G0a.yaml",
        "rules": ["DL-36", "DL-39"], "code_commit": "old", "versions": {"torch": "x"},
        "sources_sha256": {"selection_dl36.json": "aa"}}
SETTINGS = {"lr": 0.01, "c_ex": 2.0, "eps0": 8.0, "rounds": 100}


def e1_config(mode, seed):
    s = dict(SETTINGS)
    extra = {}
    if mode in ("pta", "pfr"):
        s.update(clip_C=15.31, quantize=True)
        if mode == "pta":
            s["tau"] = 17.40
    if mode == "ppe":
        s["cloud_eta"] = 1.0
        extra = {"supersedes": "E1_ppe_s0 of the E1 job", "dl40_selection": {"cloud_eta": 1.0}}
    return dict(BASE, run_id=f"E1_{mode}_s{seed}", configuration=mk.DESIGNS[mode], mode=mode, seed=seed,
                settings=s, **extra)


def write_e1_configs(d):
    e1, dl40 = os.path.join(d, "E1"), os.path.join(d, "E1_DL40")
    for folder in (e1, dl40):
        os.makedirs(folder)
    for mode in mk.DESIGNS:
        for seed in range(10):
            with open(os.path.join(dl40 if mode == "ppe" else e1, f"E1_{mode}_s{seed}.yaml"), "w",
                      encoding="utf-8") as fh:
                yaml.safe_dump(e1_config(mode, seed), fh)
    return e1, dl40


def load_yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fake_run(mode, lr, c_ex, seed, f1, experiment="DL-42 grid", norms=None, auroc=0.75, smoke=None):
    s = {"lr": lr, "c_ex": c_ex, "clip_C": None, "tau": None, "quantize": False}
    if mode == "ppe":
        s["cloud_eta"] = 1.0
    r = {"experiment_config": {"content": {"experiment": experiment, "mode": mode, "seed": seed},
                               "smoke_rounds": smoke},
         "settings": s, "final_fold9": {"macro_f1_tuned": f1, "macro_auroc": auroc}, "machine": {"device": "cuda"}}
    if norms is not None:
        r["norms_raw"] = norms
    name = (f"E1_{mode}_s{seed}.json" if experiment == "E1" else f"{mk.grid_run_id(mode, lr, c_ex, seed)}.json")
    return name, "sha", r


def grid_scores(mode, point):
    """Synthetic mean fold-9 macro-F1 per design and point (the cases of the DL-42 rule)."""
    if mode == "ptgc" and point == (0.1, 1.0):
        return 0.5101  # 1.01 points above the shared point: change
    if mode == "ptg" and point == (0.05, 0.5):
        return 0.510  # exactly 1.0 point above: keep
    if mode == "pta" and point == (0.05, 0.5):
        return 0.520  # 2 points above: change, with C and tau from the pooled norms
    if mode == "ppe" and point == (0.1, 2.0):
        return 0.505  # 0.5 point above: keep
    return 0.500  # P-FR: all equal -> keep (ties go to the shared point)


def norms_for(seed):
    return {"device_update": list(np.arange(100.0) + 100 * seed), "group_sum": list(np.arange(50.0) * 2 + seed)}


def full_grid(drop=None, reuse_f1_shift=0.0):
    """All grid runs, the 3 reuse-check runs (seed 0) and the 9 reused E1/DL-40 runs."""
    grid_runs, reuse_runs = [], []
    for mode in mk.DESIGNS:
        for lr in mk.LRS:
            for c_ex in mk.C_EXS:
                p = (lr, c_ex)
                for seed in mk.GRID_SEEDS:
                    f1 = grid_scores(mode, p) + 0.001 * (seed - 1)
                    norms = norms_for(seed) if mode in mk.NORM_MODES else None
                    if mode in mk.REUSE_MODES and p == mk.SHARED:
                        reuse_runs.append(fake_run(mode, lr, c_ex, seed, f1 + (reuse_f1_shift if seed == 0 else 0),
                                                   experiment="E1"))
                        if seed != 0:
                            continue  # only seed 0 is run again (the reuse check)
                    if (mode, p, seed) != drop:
                        grid_runs.append(fake_run(mode, lr, c_ex, seed, f1, norms=norms))
    return grid_runs, reuse_runs


def run_select(grid_runs, reuse_runs, **kw):
    return sel.select(sel.index_grid(grid_runs, smoke=kw.get("smoke", False)), sel.index_reuse(reuse_runs), **kw)


class TestDL42Configs(unittest.TestCase):
    """make_dl42_configs: 126 grid configs + 9 shared-point checks; only lr and C_ex (and C, tau, quantization) change."""

    def test_grid_configs(self):
        with tempfile.TemporaryDirectory() as d:
            e1, dl40 = write_e1_configs(d)
            grid, check = mk.make_grid(e1, dl40, os.path.join(d, "out"), "new")
            self.assertEqual((len(grid), len(check)), (126, 9))
            grid = [load_yaml(p) for p in grid]
            check = [load_yaml(p) for p in check]
        self.assertEqual(sorted((c["mode"], c["seed"]) for c in check),
                         sorted((m, s) for m in mk.REUSE_MODES for s in mk.GRID_SEEDS))
        for c in check:
            self.assertEqual((c["settings"]["lr"], c["settings"]["c_ex"]), mk.SHARED)
        per_mode = {m: sum(c["mode"] == m for c in grid) for m in mk.DESIGNS}
        self.assertEqual(per_mode, {"ptg": 24, "ptgc": 24, "ppe": 24, "pfr": 27, "pta": 27})
        self.assertEqual(len({c["run_id"] for c in grid + check}), 135)
        for c in grid + check:
            base = e1_config(c["mode"], c["seed"])
            self.assertEqual((c["experiment"], c["fold10"], c["code_commit"]), ("DL-42 grid", "none", "new"))
            self.assertEqual(c["rules"], base["rules"] + ["DL-42"])
            self.assertNotIn("supersedes", c)
            for key in ("canaries", "base_config", "versions", "configuration", "mode", "seed"):
                self.assertEqual(c[key], base[key])
            self.assertIn(f"E1_{c['mode']}_s{c['seed']}.yaml", c["sources_sha256"])
            s, b = c["settings"], base["settings"]
            if c["mode"] in mk.NORM_MODES:  # DL-31: C, tau and quantization off; norms saved
                self.assertTrue(s["save_norms"])
                self.assertFalse({"clip_C", "tau", "quantize"} & set(s))
                b = {k: v for k, v in b.items() if k not in ("clip_C", "tau", "quantize")}
            changed = {k for k in set(s) | set(b) if s.get(k) != b.get(k)}
            self.assertLessEqual(changed, {"lr", "c_ex", "save_norms"})
            if c["mode"] == "ppe":
                self.assertEqual(s["cloud_eta"], 1.0)

    def test_ppe_base_needs_dl40_step(self):
        with tempfile.TemporaryDirectory() as d:
            e1, dl40 = write_e1_configs(d)
            cfg = e1_config("ppe", 0)
            cfg["settings"]["cloud_eta"] = 0.01
            with open(os.path.join(dl40, "E1_ppe_s0.yaml"), "w", encoding="utf-8") as fh:
                yaml.safe_dump(cfg, fh)
            with self.assertRaises(SystemExit):
                mk.base_config(e1, dl40, "ppe", 0)

    def test_rerun_configs_only_for_changed_designs(self):
        with tempfile.TemporaryDirectory() as d:
            e1, dl40 = write_e1_configs(d)
            result = run_select(*full_grid())
            path = os.path.join(d, "selection_dl42.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(result, fh)
            written = [load_yaml(p) for p in mk.make_rerun(e1, dl40, path, os.path.join(d, "rerun"), "new")]
            none = dict(result, changed=[], designs={m: dict(v, decision="keep") for m, v in result["designs"].items()})
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(none, fh)
            self.assertEqual(mk.make_rerun(e1, dl40, path, os.path.join(d, "rerun2"), "new"), [])
        self.assertEqual(sorted({c["mode"] for c in written}), ["pta", "ptgc"])
        self.assertEqual(len(written), 20)
        pta = result["designs"]["pta"]["selected"]
        for c in written:
            base = e1_config(c["mode"], c["seed"])
            self.assertEqual(c["run_id"], f"E1tuned_{c['mode']}_s{c['seed']}")
            self.assertEqual((c["fold10"], c["experiment"]), ("sealed (DL-39)", "E1 tuned (DL-42)"))
            s, b = c["settings"], base["settings"]
            if c["mode"] == "pta":
                self.assertEqual((s["lr"], s["c_ex"], s["clip_C"], s["tau"], s["quantize"]),
                                 (0.05, 0.5, pta["clip_C"], pta["tau"], True))
            else:
                self.assertEqual((s["lr"], s["c_ex"]), (0.1, 1.0))
            changed = {k for k in set(s) | set(b) if s.get(k) != b.get(k)}
            self.assertLessEqual(changed, {"lr", "c_ex", "clip_C", "tau"})
            self.assertEqual({k: c[k] for k in ("canaries", "base_config", "rules")},
                             {"canaries": True, "base_config": "../G0a.yaml", "rules": base["rules"] + ["DL-42"]})


class TestDL42Selection(unittest.TestCase):
    """The DL-42 rule: keep the shared point unless the best mean is MORE than 1.0 point higher."""

    def test_rule_on_synthetic_numbers(self):
        out = run_select(*full_grid())
        d = out["designs"]
        self.assertEqual(d["ptgc"]["decision"], "change")  # +1.01 points
        self.assertEqual(d["ptgc"]["selected"], {"lr": 0.1, "c_ex": 1.0})
        self.assertEqual(d["ptg"]["decision"], "keep")  # exactly +1.0 point
        self.assertEqual(d["ppe"]["decision"], "keep")  # +0.5 point
        self.assertEqual(d["pfr"]["decision"], "keep")  # ties go to the shared point
        self.assertEqual(d["pfr"]["best_point"], "lr0p01_cex2")
        self.assertEqual(sorted(out["changed"]), ["pta", "ptgc"])
        self.assertAlmostEqual(d["ptgc"]["gain_points"], 1.01, places=6)
        self.assertTrue(all(v["match"] for v in out["reuse_check"].values()))
        self.assertEqual(d["ptg"]["points"]["lr0p01_cex2"]["source"], "reused E1 runs")
        self.assertEqual(d["ppe"]["points"]["lr0p01_cex2"]["source"], "reused E1 runs (DL-40)")
        self.assertEqual(d["ptgc"]["grid_edge_sides"], ["lr at the highest grid value"])
        self.assertEqual(d["ptg"]["grid_edge_sides"], ["lr at the lowest grid value", "C_ex at the highest grid value"])

    def test_pooled_norms_for_pta_and_no_tau_for_pfr(self):
        out = run_select(*full_grid())
        pta = out["designs"]["pta"]["selected"]
        dev = np.concatenate([norms_for(s)["device_update"] for s in mk.GRID_SEEDS])
        grp = np.concatenate([norms_for(s)["group_sum"] for s in mk.GRID_SEEDS])
        self.assertAlmostEqual(pta["clip_C"], float(np.percentile(dev, 99)))
        self.assertAlmostEqual(pta["tau"], float(np.percentile(grp, 99)))
        self.assertEqual((pta["n_device_norms"], pta["n_group_norms"]), (300, 150))
        grid, reuse = full_grid()
        grid = [g for g in grid if not (g[2]["experiment_config"]["content"]["mode"] == "pfr"
                                        and (g[2]["settings"]["lr"], g[2]["settings"]["c_ex"]) == (0.1, 0.5))]
        grid += [fake_run("pfr", 0.1, 0.5, s, 0.53, norms=norms_for(s)) for s in mk.GRID_SEEDS]
        pfr = run_select(grid, reuse)["designs"]["pfr"]
        self.assertEqual(pfr["decision"], "change")
        self.assertIn("clip_C", pfr["selected"])
        self.assertNotIn("tau", pfr["selected"])  # P-FR has no norm check (DL-37)

    def test_missing_seed_is_an_error(self):
        with self.assertRaises(SystemExit):
            run_select(*full_grid(drop=("pta", (0.05, 1.0), 2)))
        grid, reuse = full_grid()
        grid = [g for g in grid if not g[0].startswith("DL42_pta_lr0p01_cex2")]  # the whole shared point missing
        with self.assertRaises(SystemExit):  # the reason, not a TypeError
            run_select(grid, reuse)
        self.assertIsNone(run_select(*full_grid(drop=("pta", (0.05, 1.0), 2)), partial=True)["designs"]["pta"]
                          ["decision"])  # a partial table selects nothing

    def test_reuse_check(self):
        grid, reuse = full_grid(reuse_f1_shift=0.002)  # seed 0 of the reused runs is not reproduced
        self.assertFalse(sel.reuse_check(sel.index_grid(grid), sel.index_reuse(reuse))["ptg"]["match"])
        with self.assertRaises(SystemExit):
            run_select(grid, reuse)
        for mode in mk.REUSE_MODES:  # the fallback: the shared point run again for seeds 1-2
            grid += [fake_run(mode, 0.01, 2.0, s, 0.500 + 0.001 * (s - 1)) for s in (1, 2)]
        out = run_select(grid, reuse)
        self.assertEqual(out["designs"]["ptg"]["points"]["lr0p01_cex2"]["source"], "grid")
        grid, reuse = full_grid()
        grid = [g for g in grid if not g[0].startswith("DL42_ptgc_lr0p01_cex2")]  # the check was not run
        with self.assertRaises(SystemExit):
            run_select(grid, reuse)

    def test_invalid_runs_are_refused(self):
        name, sha, r = fake_run("pta", 0.05, 1.0, 0, 0.5, norms=norms_for(0))
        r["settings"]["clip_C"] = 15.0  # P-TA grid runs need C off (DL-31)
        with self.assertRaises(SystemExit):
            sel.index_grid([(name, sha, r)])
        with self.assertRaises(SystemExit):  # no saved norms
            sel.index_grid([fake_run("pfr", 0.05, 1.0, 0, 0.5)])
        with self.assertRaises(SystemExit):  # a shortened run cannot select
            sel.index_grid([fake_run("ptg", 0.05, 1.0, 0, 0.5, smoke=5)])
        self.assertEqual(len(sel.index_grid([fake_run("ptg", 0.05, 1.0, 0, 0.5, smoke=5)], smoke=True)), 1)
        name, sha, r = fake_run("ptg", 0.05, 1.0, 0, 0.5, experiment="E1")
        with self.assertRaises(SystemExit):  # a reused run must be at the shared point
            sel.index_reuse([(name, sha, r)])

    def test_smoke_selection_switches_every_design(self):
        grid = [fake_run(m, 0.05, 1.0, 0, 0.4, norms=norms_for(0) if m in mk.NORM_MODES else None, smoke=5)
                for m in mk.DESIGNS]
        _, reuse = full_grid()
        out = run_select(grid, reuse, n_seeds=1, smoke=True)
        self.assertEqual(sorted(out["changed"]), sorted(mk.DESIGNS))
        self.assertIn("tau", out["designs"]["pta"]["selected"])


class TestDL42Plumbing(unittest.TestCase):
    """--save-norms reaches run_pilot for P-TA/P-FR only; summaries read the E1tuned runs."""

    def test_save_norms_argument(self):
        from run_experiment import build_argv
        cfg = {"run_id": "DL42_pta_lr0p05_cex1_s0", "mode": "pta", "seed": 0, "canaries": True, "fold10": "none",
               "base_config": "../G0a.yaml", "code_commit": "c",
               "settings": {"lr": 0.05, "c_ex": 1.0, "eps0": 8.0, "rounds": 100, "save_norms": True}}
        argv = build_argv(cfg, "/x/configs/DL42_grid/a.yaml", "/data", "/out")
        self.assertIn("--save-norms", argv)
        self.assertFalse({"--clip-C", "--tau", "--quantize", "--fold10-out"} & set(argv))
        import run_pilot
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            run_pilot.main(["--config", "x", "--ptbxl-dir", "y", "--mode", "ptg", "--seed", "0", "--lr", "0.01",
                            "--c-ex", "2", "--save-norms", "--out", "z"])

    def test_summary_pattern(self):
        from summarize_e1 import load_runs
        with tempfile.TemporaryDirectory() as d:
            for f in ("E1tuned_pta_s0.json", "E1_pta_s0.json", "E1tuned_pta_s0.fold10.sealed.json"):
                with open(os.path.join(d, f), "w", encoding="utf-8") as fh:
                    json.dump({"file": f}, fh)
            self.assertEqual([r["file"] for r in load_runs(d, "E1tuned_*.json")], ["E1tuned_pta_s0.json"])
            self.assertEqual([r["file"] for r in load_runs(d)], ["E1_pta_s0.json"])

    def test_job_runner_matches_the_config_names(self):
        path = os.path.join(HERE, "..", "workstation", "jobs", "dl42_tuning", "run_job.py")
        if not os.path.exists(path):
            self.skipTest("job runner not present (job package)")
        import ast
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        values = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name) and t.id in ("MODES", "REUSE_MODES", "SHARED")}
        self.assertEqual(values, {"MODES": list(mk.DESIGNS), "REUSE_MODES": list(mk.REUSE_MODES),
                                  "SHARED": mk.SHARED})


if __name__ == "__main__":
    unittest.main()


class TestDL42Fold10(unittest.TestCase):
    """dl42_fold10: final configurations; a sealed file opens only with a matching seal and its fold-9 thresholds."""

    THR = {"NORM": 0.5}

    def write(self, path, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)

    def run_pair(self, runs_dir, sealed_dir, run_id, mode, seed, f1, settings=None):
        run = {"run_id": run_id, "seed": seed, "mode": mode, "settings": settings or {"lr": 0.01, "c_ex": 2.0},
               "final_fold9": {"macro_f1_tuned": f1 + 0.015, "thresholds": self.THR}}
        if mode == "local":
            body = {"local_only_fold10": {"mean_macro_f1_tuned": f1, "mean_macro_auroc": 0.72},
                    "devices": [{"mi_sens_at_90spec": 0.3}]}
        else:
            body = {"final_fold10": {"macro_f1_tuned": f1, "macro_auroc": 0.75, "mi_sens_at_90spec": 0.3,
                                     "backdoor_success_clean": 0.25, "per_class_f1_tuned": {"NORM": 0.7},
                                     "thresholds": self.THR, "thresholds_tuned_on_this_fold": False}}
        self.write(os.path.join(runs_dir, f"{run_id}.json"), run)
        self.write(os.path.join(sealed_dir, f"{run_id}.fold10.sealed.json"), dict(body, run_id=run_id, seed=seed,
                                                                                   mode=mode))

    def build(self, d, point):
        import zipfile
        from g1_check import seal_register
        e1, dl40, dl42 = (os.path.join(d, n) for n in ("JOB_e1", "JOB_dl40", "JOB_DL-42_v2"))
        for mode in ("ptg", "ptgc", "pfr", "pta", "local"):
            for s in range(10):
                self.run_pair(os.path.join(e1, "runs"), os.path.join(e1, "sealed_fold10"), f"E1_{mode}_s{s}", mode,
                              s, 0.50 + 0.001 * s)
        for s in range(10):
            self.run_pair(os.path.join(dl40, "e1_ppe_runs"), os.path.join(dl40, "sealed_fold10"), f"E1_ppe_s{s}",
                          "ppe", s, 0.50)
            self.run_pair(os.path.join(dl42, "tuned_runs"), os.path.join(dl42, "sealed_fold10"),
                          f"E1tuned_ptgc_s{s}", "ptgc", s, 0.56, settings=dict(point))
        g1 = seal_register([os.path.join(e1, "sealed_fold10"), os.path.join(dl40, "sealed_fold10")], [])
        zpath = os.path.join(d, "JOB_DL-42_v2_results.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            lines = []
            for f in sorted(os.listdir(os.path.join(dl42, "sealed_fold10"))):
                with open(os.path.join(dl42, "sealed_fold10", f), "rb") as fh:
                    lines.append(f"{__import__('hashlib').sha256(fh.read()).hexdigest()}  results/sealed_fold10/{f}")
            zf.writestr("MANIFEST.sha256", "\n".join(lines) + "\n")
        sel = {"smoke": False, "partial": False, "changed": ["ptgc"],
               "designs": {m: {"decision": "change" if m == "ptgc" else "keep",
                               "selected": point if m == "ptgc" else {"lr": 0.01, "c_ex": 2.0}}
                           for m in ("ptg", "ptgc", "ppe", "pfr", "pta")}}
        return e1, dl40, dl42, g1, zpath, sel

    def test_register_and_open(self):
        import dl42_fold10 as m
        with tempfile.TemporaryDirectory() as d:
            e1, dl40, dl42, g1, zpath, sel = self.build(d, {"lr": 0.1, "c_ex": 1.0})
            reg = m.register(os.path.join(dl42, "sealed_fold10"), [zpath])
            self.assertEqual(len(reg), 10)
            out = m.analyse(sel, e1, dl40, dl42, g1, reg)
        self.assertEqual(out["missing"], [])
        self.assertEqual((out["designs"]["P-TG-c"]["final"], out["designs"]["P-TA"]["final"]), ("tuned", "shared"))
        price = out["tuned_price_of_distrust"]
        self.assertEqual((len(price["all_seeds"]["seeds"]), len(price["seeds_3_9"]["seeds"])), (10, 7))
        self.assertAlmostEqual(price["seeds_3_9"]["mean"], 100 * (0.56 - 0.506), places=6)
        self.assertEqual(list(out["tuned_minus_shared"]), ["P-TG-c"])
        self.assertEqual(out["h5a_f1_tuned"]["all_seeds"]["outcome"], "equivalent")

    def test_bad_seal_is_not_opened_and_wrong_rerun_stops(self):
        import dl42_fold10 as m
        with tempfile.TemporaryDirectory() as d:
            e1, dl40, dl42, g1, zpath, sel = self.build(d, {"lr": 0.1, "c_ex": 1.0})
            reg = m.register(os.path.join(dl42, "sealed_fold10"), [zpath])
            with open(os.path.join(dl42, "sealed_fold10", "E1tuned_ptgc_s4.fold10.sealed.json"), "a") as fh:
                fh.write(" ")  # changed after the register was made
            out = m.analyse(sel, e1, dl40, dl42, g1, reg)
            self.assertEqual([(x["configuration"], x["seed"]) for x in out["missing"]], [("P-TG-c", 4)])
            with open(os.path.join(dl42, "sealed_fold10", "E1tuned_ptgc_s4.fold10.sealed.json"), "a") as fh:
                fh.write(" ")
            with self.assertRaises(SystemExit):  # the zip manifest no longer matches
                m.register(os.path.join(dl42, "sealed_fold10"), [zpath])
            bad = dict(sel, designs=dict(sel["designs"], ptgc={"decision": "change", "selected": {"lr": 0.05,
                                                                                                 "c_ex": 1.0}}))
            with self.assertRaises(SystemExit):  # the reruns do not carry the selected point
                m.analyse(bad, e1, dl40, dl42, g1, reg)
            with self.assertRaises(SystemExit):
                m.analyse(dict(sel, smoke=True), e1, dl40, dl42, g1, reg)
            self.assertFalse(m.committed(zpath))  # not in git: `open` would refuse it as a register
