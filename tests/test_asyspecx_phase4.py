import csv
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace

import torch

from models.AsySpecX import Model, SparsePeriodAdapter, parse_periods

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_configs(**overrides):
    cfg = dict(
        seq_len=48, pred_len=24, enc_in=5, individual=0, cut_freq=0,
        spectral_lift="fits_linear", lift_sharing="shared", norm_mode="rin_noaffine",
        cross_mode="none", rank=2, num_bands=4, gate_init=0.0, gate_init_logit=None,
        gate_max=1.0, gate_type="global", residual_part=None, mask_self_transfer=0,
        residual_clip_eta=-1.0, force_cross_off=0, skip_dc_cross=1,
        log_asyspecx_diagnostics=0, eval_residual_part="default", gate_lr_mult=1.0,
        self_gain_init_std=1e-3, temporal_adapter="none", period=24, periods="",
        periodic_init="seasonal_naive", periodic_sharing="shared",
        temporal_fusion="convex", temporal_gate_type="global", temporal_gate_init_logit=-4.0,
        period_fusion="sum_gated", period_gate_type="period", period_gate_init_logit=0.0,
        periodic_l1_weight=0.0, periodic_l2_weight=0.0,
    )
    cfg.update(overrides)
    return Namespace(**cfg)


def make_adapter(periods=(6, 12), T=24, H=8, C=3, **kw):
    return SparsePeriodAdapter(seq_len=T, pred_len=H, channels=C, periods=periods, **kw)


class TestMultiPeriodMask(unittest.TestCase):
    def test_mask_same_phase_only(self):
        ad = make_adapter(periods=[6, 12], T=24, H=8, C=3)
        self.assertEqual(tuple(ad.periodic_mask.shape), (2, 8, 24))
        for pi, p in enumerate(ad.periods):
            for h in range(8):
                for t in range(24):
                    expected = 1.0 if (t % p) == ((24 + h) % p) else 0.0
                    self.assertEqual(ad.periodic_mask[pi, h, t].item(), expected)

    def test_seasonal_naive_nearest_same_phase(self):
        ad = make_adapter(periods=[6, 12], T=24, H=8, C=3, periodic_init="seasonal_naive")
        for pi, p in enumerate(ad.periods):
            for h in range(8):
                tgt = (24 + h) % p
                cand = [t for t in range(24) if t % p == tgt]
                row = ad.W_raw[pi, h]
                if cand:
                    self.assertEqual(row[cand[-1]].item(), 1.0)
                    self.assertEqual(int((row != 0).sum().item()), 1)
                else:
                    self.assertEqual(int((row != 0).sum().item()), 0)

    def test_forward_shapes(self):
        ad = make_adapter(periods=[6, 12], T=24, H=8, C=3)
        x = torch.randn(4, 24, 3)
        spec = torch.randn(4, 8, 3)
        W_eff = ad.W_raw * ad.periodic_mask
        pred_periods = torch.einsum("pht,btc->bphc", W_eff, x)
        self.assertEqual(tuple(pred_periods.shape), (4, 2, 8, 3))
        fused, pred_period = ad(x, spec)
        self.assertEqual(tuple(fused.shape), (4, 8, 3))
        self.assertEqual(tuple(pred_period.shape), (4, 8, 3))


class TestPeriodFusion(unittest.TestCase):
    def test_sum_gated_shape(self):
        ad = make_adapter(periods=[6, 12], period_fusion="sum_gated", period_gate_type="period")
        self.assertEqual(tuple(ad.period_weight().shape), (2, 8, 3))

    def test_softmax_single_period_weight_one(self):
        ad = make_adapter(periods=[6], period_fusion="softmax")
        self.assertFalse(ad.use_period_gate)
        self.assertTrue(torch.allclose(ad.period_weight(), torch.ones_like(ad.period_weight())))

    def test_softmax_multi_period_sums_to_one(self):
        for pg in ["global", "period", "period_horizon", "period_channel", "period_horizon_channel"]:
            ad = make_adapter(periods=[6, 12, 24], period_fusion="softmax", period_gate_type=pg)
            w = ad.period_weight()
            self.assertEqual(tuple(w.shape), (3, 8, 3))
            self.assertTrue(torch.allclose(w.sum(dim=0), torch.ones(8, 3), atol=1e-5))


class TestPeriodGateBroadcast(unittest.TestCase):
    def test_all_gate_types_broadcast(self):
        x = torch.randn(2, 24, 3)
        spec = torch.randn(2, 8, 3)
        for pg in ["global", "period", "period_horizon", "period_channel", "period_horizon_channel"]:
            ad = make_adapter(periods=[6, 12], period_gate_type=pg)
            w = ad.period_weight()
            self.assertEqual(tuple(w.shape), (2, 8, 3))
            fused, _ = ad(x, spec)
            self.assertEqual(tuple(fused.shape), (2, 8, 3))


class TestTemporalGateBroadcast(unittest.TestCase):
    def test_all_temporal_gate_types(self):
        x = torch.randn(2, 24, 3)
        spec = torch.randn(2, 8, 3)
        for tg, shape in [("global", (1, 1, 1)), ("channel", (1, 1, 3)),
                          ("horizon", (1, 8, 1)), ("horizon_channel", (1, 8, 3))]:
            ad = make_adapter(periods=[6, 12], temporal_gate_type=tg)
            self.assertEqual(tuple(ad.alpha().shape), shape)
            fused, _ = ad(x, spec)
            self.assertEqual(tuple(fused.shape), (2, 8, 3))


class TestConvexFusion(unittest.TestCase):
    def test_alpha_zero_equals_spec(self):
        ad = make_adapter(periods=[6, 12], temporal_fusion="convex", temporal_gate_init_logit=-50.0)
        x = torch.randn(2, 24, 3)
        spec = torch.randn(2, 8, 3)
        fused, _ = ad(x, spec)
        self.assertTrue(torch.allclose(fused, spec, atol=1e-5))

    def test_alpha_one_equals_period(self):
        ad = make_adapter(periods=[6, 12], temporal_fusion="convex", temporal_gate_init_logit=50.0)
        x = torch.randn(2, 24, 3)
        spec = torch.randn(2, 8, 3)
        fused, pred_period = ad(x, spec)
        self.assertTrue(torch.allclose(fused, pred_period, atol=1e-5))


class TestPeriodicRegularization(unittest.TestCase):
    def test_zero_weight_zero_loss(self):
        m = Model(make_configs(temporal_adapter="sparse_period", periods="6+12"))
        self.assertEqual(float(m.extra_loss()), 0.0)

    def test_positive_weight_nonneg(self):
        m = Model(make_configs(temporal_adapter="sparse_period", periods="6+12", periodic_l1_weight=1e-3))
        self.assertGreater(float(m.extra_loss()), 0.0)

    def test_mask_out_weights_excluded(self):
        m = Model(make_configs(temporal_adapter="sparse_period", periods="6+12", periodic_l1_weight=1.0))
        before = float(m.extra_loss())
        ad = m.temporal_adapter
        mo = [(h, t) for h in range(ad.pred_len) for t in range(ad.seq_len) if ad.periodic_mask[0, h, t] == 0][0]
        with torch.no_grad():
            ad.W_raw[0, mo[0], mo[1]] = 1e6
        self.assertAlmostEqual(float(m.extra_loss()), before, places=5)

    def test_no_adapter_returns_none(self):
        self.assertIsNone(Model(make_configs()).extra_loss())


class TestBackwardCompat(unittest.TestCase):
    def test_single_period_matches_direct_einsum(self):
        # P=1: period weight is 1, so pred_period == W_eff @ x (Phase-3 behavior).
        ad = make_adapter(periods=[6], T=24, H=8, C=3, temporal_gate_init_logit=50.0,
                          temporal_fusion="convex")
        self.assertFalse(ad.use_period_gate)
        x = torch.randn(2, 24, 3)
        spec = torch.randn(2, 8, 3)
        fused, pred_period = ad(x, spec)
        W_eff = ad.W_raw * ad.periodic_mask
        direct = torch.einsum("pht,btc->bhc", W_eff, x)
        self.assertTrue(torch.allclose(pred_period, direct, atol=1e-5))

    def test_period_flag_still_works(self):
        m = Model(make_configs(temporal_adapter="sparse_period", period=6, periods=""))
        self.assertEqual(m.temporal_adapter.periods, [6])
        y = m(torch.randn(2, 48, 5))
        self.assertEqual(tuple(y.shape), (2, 24, 5))

    def test_old_arms_forward(self):
        # phase3 anchor sparse_period, fits_individual, hier_split, cross_zero_global.
        configs = [
            dict(temporal_adapter="sparse_period", period=6, cross_mode="asym_lowrank",
                 residual_part="split", gate_type="hier_channel_band", gate_init_logit=-6.0),
            dict(lift_sharing="individual", cross_mode="none"),
            dict(cross_mode="asym_lowrank", residual_part="split", gate_type="hier_channel_band",
                 gate_init_logit=-6.0),
            dict(cross_mode="asym_lowrank", gate_type="global", gate_init_logit=0.0),
        ]
        for c in configs:
            m = Model(make_configs(**c))
            y = m(torch.randn(2, 48, 5))
            self.assertEqual(tuple(y.shape), (2, 24, 5))


class TestValidationSelectorFairness(unittest.TestCase):
    def _write_csv(self, path):
        # arm A: seed s1 great test but poor val; seed s2 poor. mean val worse.
        # arm B: consistently good val. Selector must pick B by mean val_mse.
        rows = [
            dict(arm="A", dataset="weather", seq_len=720, pred_len=96, seed=1,
                 status="ok", val_mse=0.50, mse=0.10, mae=0.20),
            dict(arm="A", dataset="weather", seq_len=720, pred_len=96, seed=2,
                 status="ok", val_mse=0.60, mse=0.40, mae=0.50),
            dict(arm="B", dataset="weather", seq_len=720, pred_len=96, seed=1,
                 status="ok", val_mse=0.20, mse=0.25, mae=0.30),
            dict(arm="B", dataset="weather", seq_len=720, pred_len=96, seed=2,
                 status="ok", val_mse=0.22, mse=0.26, mae=0.31),
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def test_selector_uses_mean_val_not_test(self):
        with tempfile.TemporaryDirectory() as d:
            csv_in = os.path.join(d, "results.csv")
            out = os.path.join(d, "selected.csv")
            summary = os.path.join(d, "selected.md")
            self._write_csv(csv_in)
            subprocess.run([sys.executable, "scripts/select_by_validation.py",
                            "--csv", csv_in, "--selection_keys", "dataset,seq_len,pred_len",
                            "--replicate_key", "seed", "--arm_key", "arm",
                            "--output", out, "--summary", summary],
                           cwd=REPO, check=True)
            with open(out) as f:
                sel = list(csv.DictReader(f))
            arms = {r["selected_arm"] for r in sel}
            self.assertEqual(arms, {"B"})           # picked by mean val, not A's lucky test
            self.assertEqual(len(sel), 2)           # all seeds of selected arm emitted
            self.assertEqual({r["seed"] for r in sel}, {"1", "2"})

    def test_selector_requires_val_mse(self):
        with tempfile.TemporaryDirectory() as d:
            csv_in = os.path.join(d, "results.csv")
            with open(csv_in, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["arm", "dataset", "seq_len", "pred_len", "seed", "status", "mse", "mae"])
                w.writeheader()
                w.writerow(dict(arm="A", dataset="w", seq_len=720, pred_len=96, seed=1, status="ok", mse=0.1, mae=0.2))
            r = subprocess.run([sys.executable, "scripts/select_by_validation.py",
                                "--csv", csv_in, "--output", os.path.join(d, "o.csv"),
                                "--summary", os.path.join(d, "s.md")],
                               cwd=REPO, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)


class TestNoMetricLeakage(unittest.TestCase):
    def test_extra_loss_not_in_forward_output(self):
        # forward() returns predictions only; extra_loss is separate and train-only.
        m = Model(make_configs(temporal_adapter="sparse_period", periods="6+12", periodic_l1_weight=1.0))
        y = m(torch.randn(2, 48, 5))
        self.assertEqual(tuple(y.shape), (2, 24, 5))
        self.assertGreater(float(m.extra_loss()), 0.0)  # exists but is not folded into y

    def test_backcast_weight_zero_default(self):
        m = Model(make_configs(temporal_adapter="sparse_period", periods="6+12"))
        out = m(torch.randn(2, 48, 5), return_full=True)
        self.assertIn("backcast", out)


if __name__ == "__main__":
    unittest.main()
