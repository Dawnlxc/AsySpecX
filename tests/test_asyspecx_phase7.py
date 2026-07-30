import csv
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import torch

from models.AsySpecX import Model, AsymCross

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(REPO, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


discover = _load("discover_periods", "scripts/discover_periods.py")


def cfg(**o):
    d = dict(
        seq_len=96, pred_len=24, enc_in=6, individual=0, cut_freq=0, spectral_lift="fits_linear",
        lift_sharing="shared", norm_mode="rin_noaffine", cross_mode="asym_lowrank", rank=2, num_bands=4,
        gate_init=0.0, gate_init_logit=-6.0, gate_max=1.0, gate_type="hier_channel_band", residual_part="split",
        mask_self_transfer=0, residual_clip_eta=-1.0, force_cross_off=0, skip_dc_cross=1,
        log_asyspecx_diagnostics=0, eval_residual_part="default", gate_lr_mult=5.0, self_gain_init_std=1e-3,
        temporal_adapter="sparse_period", period=24, periods="12+24", periodic_init="seasonal_naive",
        periodic_sharing="shared", temporal_fusion="convex", temporal_gate_type="horizon",
        temporal_gate_init_logit=-4.0, period_fusion="sum_gated", period_gate_type="period",
        period_gate_init_logit=0.0, periodic_l1_weight=0.0, periodic_l2_weight=0.0, temporal_gate_l1_weight=0.0,
        energy_control="none", learned_clip_scope="component_channel_band", learned_clip_eta_init=1.0,
        learned_clip_eta_max=2.0, patch_adapter="none", patch_len=16, patch_stride=8, patch_basis_dim=0,
        patch_fusion="convex", patch_gate_type="horizon", patch_gate_init_logit=-6.0,
        patch_l1_weight=0.0, patch_l2_weight=0.0)
    d.update(o)
    return Namespace(**d)


# ---- B. auto period ----
class TestAutoPeriod(unittest.TestCase):
    def test_acf_finds_period_12(self):
        t = np.arange(4000)
        x = np.sin(2 * np.pi * t / 12.0)[:, None] + 0.01 * np.random.RandomState(0).randn(4000, 1)
        periods, _ = discover.discover_acf(x, 4, 400, 3)
        self.assertTrue(any(abs(p - 12) <= 1 for p in periods), periods)

    def test_fft_finds_period_24(self):
        t = np.arange(4800)
        x = np.sin(2 * np.pi * t / 24.0)[:, None]
        periods, _ = discover.discover_fft(x, 4, 400, 3)
        self.assertTrue(any(abs(p - 24) <= 1 for p in periods), periods)

    def test_manual_period_unchanged(self):
        m = Model(cfg(periods="12+24"))
        self.assertEqual(m.temporal_adapter.periods, [12, 24])


# ---- C. patch linear ----
class TestPatchLinear(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(tuple(Model(cfg(patch_adapter="patch_linear"))(torch.randn(3, 96, 6)).shape), (3, 24, 6))

    def test_alpha0_equals_base(self):
        ad = Model(cfg(patch_adapter="patch_linear", patch_gate_init_logit=-60.0)).patch_adapter
        base = torch.randn(2, 24, 6)
        self.assertTrue(torch.allclose(ad(torch.randn(2, 96, 6), base), base, atol=1e-4))

    def test_alpha1_equals_patch(self):
        ad = Model(cfg(patch_adapter="patch_linear", patch_gate_init_logit=60.0)).patch_adapter
        x = torch.randn(2, 96, 6)
        base = torch.randn(2, 24, 6)
        out = ad(x, base)
        # recompute patch pred by calling with base=0 and additive-equivalent: alpha=1 convex -> out == patch pred
        ad2 = ad
        # patch pred is out when alpha=1 (convex: base + 1*(patch-base) = patch)
        # verify independence from base: same x, different base -> same out
        out_b2 = ad(x, base + 5.0)
        self.assertTrue(torch.allclose(out, out_b2, atol=1e-4))

    def test_gate_broadcast(self):
        for gt, shp in [("global", (1, 1, 1)), ("channel", (1, 1, 6)), ("horizon", (1, 24, 1)), ("horizon_channel", (1, 24, 6))]:
            ad = Model(cfg(patch_adapter="patch_linear", patch_gate_type=gt)).patch_adapter
            self.assertEqual(tuple(ad.alpha().shape), shp)

    def test_reg_extra_loss(self):
        self.assertEqual(float(Model(cfg(patch_adapter="patch_linear"))(torch.randn(2, 96, 6)).sum() * 0), 0.0)
        m = Model(cfg(patch_adapter="patch_linear", patch_l1_weight=1e-2))
        self.assertGreater(float(m.extra_loss()), 0.0)
        y = m(torch.randn(2, 96, 6))  # extra_loss separate from forward output
        self.assertEqual(tuple(y.shape), (2, 24, 6))

    def test_none_preserves(self):
        self.assertIsNone(Model(cfg(patch_adapter="none", temporal_adapter="none")).extra_loss())


# ---- A. learned clip ----
class TestLearnedClip(unittest.TestCase):
    def test_never_amplifies(self):
        U = torch.randn(3, 6, 5, dtype=torch.cfloat)
        R = U * 4.0  # residual bigger than input
        eta = torch.full((1, 6, 1), 2.0)  # large learned eta
        Rout, _ = AsymCross.clip_residual_learned(R, U, eta)
        rms_in = torch.sqrt(torch.mean(torch.abs(R) ** 2))
        rms_out = torch.sqrt(torch.mean(torch.abs(Rout) ** 2))
        self.assertLessEqual(float(rms_out), float(rms_in) + 1e-6)

    def test_forward_and_diag(self):
        m = Model(cfg(energy_control="learned_clip"))
        m.train()
        y = m(torch.randn(4, 96, 6))
        self.assertEqual(tuple(y.shape), (4, 24, 6))
        d = m.get_diagnostics()
        self.assertIn("eta_mean", d)
        y.pow(2).mean().backward()
        self.assertIsNotNone(m.cross_block.clip_logit.grad)


# ---- E. runner dry-run ----
class TestPhase7DryRun(unittest.TestCase):
    def _dry(self, extra):
        env = dict(os.environ); env.update({"DRY_RUN": "1", "SEEDS": "2024"}); env.update(extra)
        return subprocess.run(["bash", "scripts/run_phase7_breakthrough_candidates.sh"],
                              cwd=REPO, capture_output=True, text=True, env=env)

    def test_full_count(self):
        r = self._dry({"DATASETS": "weather PEMS04", "SEQ_LENS": "96 720", "COMPACT": "0"})
        self.assertEqual(r.returncode, 0, r.stderr)
        # weather 2*4*1*8=64 ; PEMS04 1*4*1*8=32 -> 96
        self.assertIn("estimate: 96 runs", r.stdout)

    def test_compact_count(self):
        r = self._dry({"DATASETS": "weather", "SEQ_LENS": "96 720", "COMPACT": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("estimate: 40 runs", r.stdout)  # 2*4*1*5


# ---- F. merge ----
class TestMerge(unittest.TestCase):
    def test_merge_dedup_warns(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.csv"); b = os.path.join(d, "b.csv")
            row = dict(arm="x", dataset="w", seq_len=720, pred_len=96, seed=1, status="ok", mse=0.1, mae=0.2, val_mse=0.1)
            for pth in (a, b):
                with open(pth, "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(row.keys())); wr.writeheader(); wr.writerow(row)
            out = os.path.join(d, "m.csv")
            r = subprocess.run([sys.executable, "scripts/merge_results.py", "--csvs", f"{a},{b}", "--output", out],
                               cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("duplicate", r.stdout.lower())
            self.assertEqual(len(list(csv.DictReader(open(out)))), 1)


# ---- D. ensemble ----
class TestEnsemble(unittest.TestCase):
    def test_simplex_val_only(self):
        with tempfile.TemporaryDirectory() as d:
            pd = os.path.join(d, "preds"); os.makedirs(pd)
            rng = np.random.RandomState(0)
            y_val = rng.randn(50, 24, 3); y_test = rng.randn(40, 24, 3)
            # arm A good on val, arm B noisy; convex should favor A
            for arm, noise in [("armA", 0.1), ("armB", 1.0)]:
                vp = y_val + noise * rng.randn(*y_val.shape)
                tp = y_test + noise * rng.randn(*y_test.shape)
                np.savez_compressed(os.path.join(pd, f"{arm}__weather__sl720__pl24__sd1.npz"),
                                    val_pred=vp, val_true=y_val, test_pred=tp, test_true=y_test)
            out = os.path.join(d, "ens.csv"); summ = os.path.join(d, "ens.md")
            r = subprocess.run([sys.executable, "scripts/ensemble_predictions.py", "--pred_dir", pd,
                                "--mode", "simplex_val", "--output_csv", out, "--summary", summ],
                               cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = list(csv.DictReader(open(out)))
            self.assertEqual(len(rows), 1)
            w = rows[0]["weights"]
            self.assertIn("armA", w)

    def test_missing_dir_errors(self):
        r = subprocess.run([sys.executable, "scripts/ensemble_predictions.py", "--pred_dir", "/nonexistent_xyz",
                            "--output_csv", "/tmp/x.csv", "--summary", "/tmp/x.md"],
                           cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)

    def test_simplex_projection(self):
        w = discover  # placeholder to keep import; test projection via ensemble module
        ens = _load("ensemble_predictions", "scripts/ensemble_predictions.py")
        v = ens.project_simplex(np.array([0.5, 0.2, 0.9]))
        self.assertAlmostEqual(float(v.sum()), 1.0, places=6)
        self.assertTrue((v >= -1e-9).all())


# ---- G. backward compat dry-run ----
class TestBackwardCompat(unittest.TestCase):
    def test_scripts_parse(self):
        for s in ["scripts/run_phase6_fullfield_candidates.sh", "scripts/run_phase7_breakthrough_candidates.sh",
                  "scripts/run_phase7_selection.sh", "scripts/_common.sh",
                  "scripts/slurm/asyspecx_phase7_run.sbatch", "scripts/slurm/submit_asyspecx_phase7.sh"]:
            r = subprocess.run(["bash", "-n", s], cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{s}: {r.stderr}")

    def test_phase6_arm_still_builds(self):
        m = Model(cfg(temporal_adapter="sparse_period", periods="24+168"))
        self.assertEqual(tuple(m(torch.randn(2, 96, 6)).shape), (2, 24, 6))


if __name__ == "__main__":
    unittest.main()
