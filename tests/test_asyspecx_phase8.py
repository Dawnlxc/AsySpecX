import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import torch

from models.AsySpecX import Model, _moving_avg, AsymCross

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = dict(
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
    patch_fusion="convex", patch_gate_type="horizon", patch_gate_init_logit=-6.0, patch_l1_weight=0.0,
    patch_l2_weight=0.0, max_periods=5, linear_adapter="none", linear_sharing="shared",
    individual_linear_max_channels=64, linear_init="zeros", moving_avg_kernel=25,
    multiscale_factors="1,2,4", multiscale_fusion="softmax", multiscale_gate_type="scale",
    linear_fusion="convex", linear_gate_type="horizon", linear_gate_init_logit=-6.0,
    linear_l1_weight=0.0, linear_l2_weight=0.0, branch_fusion="sequential",
    branch_fusion_scope="horizon", branch_init_main_logit=4.0, branch_init_aux_logit=-4.0)


def cfg(**o):
    d = dict(BASE); d.update(o); return Namespace(**d)


class TestLinearAdapters(unittest.TestCase):
    def test_shapes(self):
        for k in ("direct_linear", "dlinear_decomp", "multiscale_dlinear"):
            self.assertEqual(tuple(Model(cfg(linear_adapter=k))(torch.randn(3, 96, 6)).shape), (3, 24, 6))

    def test_moving_avg_length(self):
        x = torch.randn(2, 96, 6)
        self.assertEqual(tuple(_moving_avg(x, 25).shape), (2, 96, 6))
        self.assertEqual(tuple(_moving_avg(x, 24).shape), (2, 96, 6))  # even kernel
        self.assertEqual(tuple(_moving_avg(x, 1).shape), (2, 96, 6))

    def test_linear_fusion_alpha(self):
        ad0 = Model(cfg(linear_adapter="direct_linear", linear_gate_init_logit=-60.0)).linear_adapter
        base = torch.randn(2, 24, 6)
        self.assertTrue(torch.allclose(ad0(torch.randn(2, 96, 6), base), base, atol=1e-4))
        ad1 = Model(cfg(linear_adapter="direct_linear", linear_gate_init_logit=60.0)).linear_adapter
        x = torch.randn(2, 96, 6)
        self.assertTrue(torch.allclose(ad1(x, torch.randn(2, 24, 6)), ad1.raw(x), atol=1e-4))

    def test_gate_broadcast(self):
        for gt, shp in [("global", (1, 1, 1)), ("channel", (1, 1, 6)), ("horizon", (1, 24, 1)), ("horizon_channel", (1, 24, 6))]:
            ad = Model(cfg(linear_adapter="direct_linear", linear_gate_type=gt)).linear_adapter
            self.assertEqual(tuple(ad.alpha().shape), shp)

    def test_individual_guard(self):
        with self.assertRaises(ValueError):
            Model(cfg(linear_adapter="direct_linear", linear_sharing="individual", enc_in=321))

    def test_init_last(self):
        ad = Model(cfg(linear_adapter="direct_linear", linear_init="last")).linear_adapter
        self.assertTrue(bool((ad.direct.weight[:, -1] == 1).all()))

    def test_reg_not_polluting(self):
        # small_random init -> nonzero weights so L1 penalty is > 0
        m = Model(cfg(linear_adapter="dlinear_decomp", linear_init="small_random", linear_l1_weight=1e-2))
        self.assertGreater(float(m.extra_loss()), 0.0)
        y = m(torch.randn(2, 96, 6))
        self.assertEqual(tuple(y.shape), (2, 24, 6))  # extra_loss not folded into forward
        # l1=0 -> zero extra loss regardless of weights
        self.assertEqual(float(Model(cfg(linear_adapter="dlinear_decomp", linear_init="small_random")).extra_loss()), 0.0)


class TestBranchFusion(unittest.TestCase):
    def test_weights_sum_to_one(self):
        m = Model(cfg(patch_adapter="patch_linear", linear_adapter="dlinear_decomp", branch_fusion="softmax_static"))
        w = m.branch_fusion.weights()
        self.assertTrue(torch.allclose(w.sum(0), torch.ones(24, 6), atol=1e-5))

    def test_single_branch_returns_branch(self):
        m = Model(cfg(temporal_adapter="none", patch_adapter="none", linear_adapter="none", branch_fusion="softmax_static"))
        self.assertEqual(m.branch_fusion.branch_names, ["spec"])
        x = torch.randn(2, 96, 6)
        # single branch -> weight 1 -> output == spec pred == sequential output
        seq = Model(cfg(temporal_adapter="none", patch_adapter="none", linear_adapter="none", branch_fusion="sequential"))
        seq.load_state_dict(m.state_dict(), strict=False)
        self.assertEqual(tuple(m(x).shape), (2, 24, 6))

    def test_init_favors_spec(self):
        m = Model(cfg(patch_adapter="patch_linear", linear_adapter="dlinear_decomp", branch_fusion="softmax_static"))
        w = m.branch_fusion.weights()
        self.assertGreater(float(w[0].mean()), 0.9)  # spec branch dominant at init

    def test_scope_broadcast(self):
        # BASE has sparse_period, so branches = spec+period+patch = 3
        for scope in ("global", "horizon", "channel", "horizon_channel"):
            m = Model(cfg(patch_adapter="patch_linear", branch_fusion="softmax_static", branch_fusion_scope=scope))
            w = m.branch_fusion.weights()
            self.assertEqual(tuple(w.shape), (3, 24, 6))
            self.assertTrue(torch.allclose(w.sum(0), torch.ones(24, 6), atol=1e-5))

    def test_sequential_matches_manual(self):
        # branch_fusion=sequential must reproduce the pre-Phase8 composition.
        torch.manual_seed(0)
        m = Model(cfg(patch_adapter="patch_linear", branch_fusion="sequential"))
        x = torch.randn(2, 96, 6)
        y1 = m(x)
        y2 = m(x)  # deterministic
        self.assertTrue(torch.allclose(y1, y2, atol=1e-6))
        self.assertEqual(tuple(y1.shape), (2, 24, 6))


class TestLearnedClipNoAmplify(unittest.TestCase):
    def test_no_amplify(self):
        U = torch.randn(3, 6, 5, dtype=torch.cfloat)
        R = U * 5.0
        Rout, _ = AsymCross.clip_residual_learned(R, U, torch.full((1, 6, 1), 2.0))
        self.assertLessEqual(float(torch.sqrt(torch.mean(torch.abs(Rout) ** 2))),
                             float(torch.sqrt(torch.mean(torch.abs(R) ** 2))) + 1e-6)


class TestUnionPeriods(unittest.TestCase):
    def test_union_dedup_priority_max(self):
        csvp = os.path.join(REPO, "dataset/electricity/electricity.csv")
        if not os.path.isfile(csvp):
            self.skipTest("electricity.csv absent")
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "u.json")
            r = subprocess.run([sys.executable, "scripts/discover_periods.py", "--dataset", "electricity",
                                "--data", "custom", "--root_path", "./dataset/electricity/",
                                "--data_path", "electricity.csv", "--seq_len", "720", "--enc_in", "321",
                                "--cycle", "168", "--method", "union_auto", "--manual_periods", "24,168",
                                "--max_periods", "4", "--period_min", "4", "--period_max", "0", "--output", out],
                               cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            per = json.load(open(out))["periods"]
            self.assertLessEqual(len(per), 4)               # max_periods
            self.assertEqual(len(per), len(set(per)))        # dedup
            self.assertEqual(per[0], 24)                      # manual priority first
            self.assertIn(168, per)


class TestEnsemble(unittest.TestCase):
    def test_simplex_val_only_sum1(self):
        with tempfile.TemporaryDirectory() as d:
            pd = os.path.join(d, "p"); os.makedirs(pd)
            rng = np.random.RandomState(1)
            yv = rng.randn(40, 24, 3); yt = rng.randn(30, 24, 3)
            for arm, noise in [("A", 0.1), ("B", 1.5)]:
                np.savez_compressed(os.path.join(pd, f"{arm}__weather__sl720__pl24__sd1.npz"),
                                    val_pred=(yv + noise * rng.randn(*yv.shape)).astype(np.float32),
                                    val_true=yv.astype(np.float32),
                                    test_pred=(yt + noise * rng.randn(*yt.shape)).astype(np.float32),
                                    test_true=yt.astype(np.float32))
            out = os.path.join(d, "e.csv")
            r = subprocess.run([sys.executable, "scripts/ensemble_predictions.py", "--pred_dir", pd,
                                "--mode", "simplex_val", "--output_csv", out, "--summary", os.path.join(d, "e.md")],
                               cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            row = list(csv.DictReader(open(out)))[0]
            self.assertIn("A", row["weights"])


class TestPhase8DryRun(unittest.TestCase):
    def _dry(self, extra):
        env = dict(os.environ); env.update({"DRY_RUN": "1", "SEEDS": "2024"}); env.update(extra)
        return subprocess.run(["bash", "scripts/run_phase8_hydra_candidates.sh"],
                              cwd=REPO, capture_output=True, text=True, env=env)

    def test_compact_count(self):
        r = self._dry({"DATASETS": "weather PEMS04", "SEQ_LENS": "96 720", "COMPACT": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        # weather 2*4*1*4=32 ; PEMS04 1*4*1*4=16 -> 48
        self.assertIn("estimate: 48 runs", r.stdout)

    def test_full_count(self):
        r = self._dry({"DATASETS": "weather", "SEQ_LENS": "96 720", "COMPACT": "0"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("estimate: 56 runs", r.stdout)  # 2*4*1*7


class TestBackwardCompat(unittest.TestCase):
    def test_scripts_parse(self):
        for s in ["scripts/run_phase7_breakthrough_candidates.sh", "scripts/run_phase8_hydra_candidates.sh",
                  "scripts/run_phase8_selection.sh", "scripts/_common.sh",
                  "scripts/slurm/asyspecx_phase8_run.sbatch", "scripts/slurm/submit_asyspecx_phase8.sh"]:
            r = subprocess.run(["bash", "-n", s], cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{s}: {r.stderr}")

    def test_phase7_arm_builds(self):
        m = Model(cfg(patch_adapter="patch_linear", periods="24+168"))
        self.assertEqual(tuple(m(torch.randn(2, 96, 6)).shape), (2, 24, 6))


if __name__ == "__main__":
    unittest.main()
