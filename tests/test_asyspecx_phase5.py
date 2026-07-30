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

from models.AsySpecX import Model

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
        periodic_l1_weight=0.0, periodic_l2_weight=0.0, temporal_gate_l1_weight=0.0,
    )
    cfg.update(overrides)
    return Namespace(**cfg)


# ---------- A. validation segmented metrics ----------
class TestValSegments(unittest.TestCase):
    def test_helper_outputs_k_segments(self):
        from exp.exp_main import Exp_Main
        se = np.arange(40, dtype=float)
        ae = np.arange(40, dtype=float)
        fields = Exp_Main.segment_val_metrics(se, ae, 4)
        joined = " ".join(fields)
        for i in range(4):
            self.assertIn(f"val_mse_seg{i}=", joined)
        self.assertIn("val_mse=", joined)
        self.assertIn("val_mae=", joined)
        self.assertTrue(fields[0] == "K=4")

    def test_fewer_samples_than_k_gives_nan(self):
        from exp.exp_main import Exp_Main
        se = np.array([1.0, 2.0])
        fields = Exp_Main.segment_val_metrics(se, se, 4)
        joined = " ".join(fields)
        self.assertIn("val_mse_seg2=nan", joined)
        self.assertIn("val_mse_seg3=nan", joined)


# ---------- C. temporal_gate_l1 ----------
class TestTemporalGateL1(unittest.TestCase):
    def test_zero_weight_no_change(self):
        m = Model(make_configs(temporal_adapter="sparse_period", periods="6+12"))
        self.assertEqual(float(m.extra_loss()), 0.0)

    def test_positive_weight_nonneg(self):
        m = Model(make_configs(temporal_adapter="sparse_period", periods="6+12",
                               temporal_gate_l1_weight=1e-3))
        self.assertGreater(float(m.extra_loss()), 0.0)

    def test_not_in_forward_output(self):
        m = Model(make_configs(temporal_adapter="sparse_period", periods="6+12",
                               temporal_gate_l1_weight=1.0))
        y = m(torch.randn(2, 48, 5))
        self.assertEqual(tuple(y.shape), (2, 24, 5))
        self.assertGreater(float(m.extra_loss()), 0.0)  # separate from y

    def test_diagnostic_recorded(self):
        m = Model(make_configs(temporal_adapter="sparse_period", periods="6+12",
                               temporal_gate_l1_weight=1e-3))
        m.train()
        m(torch.randn(2, 48, 5))
        self.assertIn("temporal_gate_l1_value", m.get_diagnostics())


# ---------- B/I. selector ----------
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_selector(args, cwd=REPO):
    return subprocess.run([sys.executable, "scripts/select_by_validation.py", *args],
                          cwd=cwd, capture_output=True, text=True)


class TestSelector(unittest.TestCase):
    def _base_rows(self):
        # arm A: lucky test on seed1 but bad val; arm B: consistently good val.
        return [
            dict(arm="A", dataset="weather", seq_len=720, pred_len=96, seed=1,
                 status="ok", val_mse=0.50, mse=0.10, mae=0.20, val_mse_seg0=0.9, val_mse_seg1=0.5),
            dict(arm="A", dataset="weather", seq_len=720, pred_len=96, seed=2,
                 status="ok", val_mse=0.60, mse=0.40, mae=0.50, val_mse_seg0=0.9, val_mse_seg1=0.6),
            dict(arm="B", dataset="weather", seq_len=720, pred_len=96, seed=1,
                 status="ok", val_mse=0.20, mse=0.25, mae=0.30, val_mse_seg0=0.4, val_mse_seg1=0.2),
            dict(arm="B", dataset="weather", seq_len=720, pred_len=96, seed=2,
                 status="ok", val_mse=0.22, mse=0.26, mae=0.31, val_mse_seg0=0.4, val_mse_seg1=0.22),
        ]

    def test_mean_over_seeds_not_test(self):
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "r.csv"); write_csv(c, self._base_rows())
            out = os.path.join(d, "o.csv")
            r = run_selector(["--csv", c, "--output", out, "--summary", os.path.join(d, "s.md")])
            self.assertEqual(r.returncode, 0, r.stderr)
            sel = list(csv.DictReader(open(out)))
            self.assertEqual({x["selected_arm"] for x in sel}, {"B"})
            self.assertEqual({x["seed"] for x in sel}, {"1", "2"})

    def test_refuses_test_metric(self):
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "r.csv"); write_csv(c, self._base_rows())
            r = run_selector(["--csv", c, "--select_metric", "mse",
                              "--output", os.path.join(d, "o.csv"), "--summary", os.path.join(d, "s.md")])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("test metric", (r.stderr + r.stdout).lower())

    def test_mean_plus_std(self):
        # arm A lower mean but high variance; mean_plus_std with big weight flips to B.
        rows = [
            dict(arm="A", dataset="w", seq_len=720, pred_len=96, seed=1, status="ok", val_mse=0.10, mse=0.5, mae=0.5),
            dict(arm="A", dataset="w", seq_len=720, pred_len=96, seed=2, status="ok", val_mse=0.40, mse=0.5, mae=0.5),
            dict(arm="B", dataset="w", seq_len=720, pred_len=96, seed=1, status="ok", val_mse=0.24, mse=0.1, mae=0.1),
            dict(arm="B", dataset="w", seq_len=720, pred_len=96, seed=2, status="ok", val_mse=0.26, mse=0.1, mae=0.1),
        ]
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "r.csv"); write_csv(c, rows)
            out = os.path.join(d, "o.csv")
            # mean: A(0.25) < B(0.25)? equal-ish; use std_weight to break toward B
            r = run_selector(["--csv", c, "--metric_mode", "mean_plus_std", "--std_weight", "2.0",
                              "--output", out, "--summary", os.path.join(d, "s.md")])
            self.assertEqual(r.returncode, 0, r.stderr)
            sel = list(csv.DictReader(open(out)))
            self.assertEqual({x["selected_arm"] for x in sel}, {"B"})

    def test_margin_prefer_order(self):
        # A best by val but B within margin; prefer_order picks B.
        rows = [
            dict(arm="A", dataset="w", seq_len=720, pred_len=96, seed=1, status="ok", val_mse=0.200, mse=0.5, mae=0.5),
            dict(arm="B", dataset="w", seq_len=720, pred_len=96, seed=1, status="ok", val_mse=0.201, mse=0.1, mae=0.1),
        ]
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "r.csv"); write_csv(c, rows)
            out = os.path.join(d, "o.csv")
            r = run_selector(["--csv", c, "--selection_margin_pct", "0.02",
                              "--prefer_arm_order", "B,A",
                              "--output", out, "--summary", os.path.join(d, "s.md")])
            self.assertEqual(r.returncode, 0, r.stderr)
            sel = list(csv.DictReader(open(out)))
            self.assertEqual({x["selected_arm"] for x in sel}, {"B"})

    def test_allowlist(self):
        rows = [
            dict(arm="phase5_asx_individual", dataset="weather", seq_len=720, pred_len=96, seed=1, status="ok", val_mse=0.30, mse=0.3, mae=0.3),
            dict(arm="phase5_asx_period_multi", dataset="weather", seq_len=720, pred_len=96, seed=1, status="ok", val_mse=0.20, mse=0.2, mae=0.2),
        ]
        allow = {"weather": ["phase5_asx_individual"], "default": ["phase5_asx_period_multi"]}
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "r.csv"); write_csv(c, rows)
            aj = os.path.join(d, "allow.json"); json.dump(allow, open(aj, "w"))
            out = os.path.join(d, "o.csv")
            r = run_selector(["--csv", c, "--arm_allowlist_json", aj,
                              "--output", out, "--summary", os.path.join(d, "s.md")])
            self.assertEqual(r.returncode, 0, r.stderr)
            sel = list(csv.DictReader(open(out)))
            # weather restricted to individual even though period_multi has better val
            self.assertEqual({x["selected_arm"] for x in sel}, {"phase5_asx_individual"})

    def test_last_segment_uses_last_seg(self):
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "r.csv"); write_csv(c, self._base_rows())
            out = os.path.join(d, "o.csv")
            r = run_selector(["--csv", c, "--metric_mode", "last_segment",
                              "--output", out, "--summary", os.path.join(d, "s.md")])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("metric_col=val_mse_seg1", r.stdout)

    def test_last_segment_missing_errors(self):
        rows = [dict(arm="A", dataset="w", seq_len=720, pred_len=96, seed=1, status="ok", val_mse=0.2, mse=0.1, mae=0.1)]
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "r.csv"); write_csv(c, rows)
            r = run_selector(["--csv", c, "--metric_mode", "last_segment",
                              "--output", os.path.join(d, "o.csv"), "--summary", os.path.join(d, "s.md")])
            self.assertNotEqual(r.returncode, 0)


# ---------- H. backward compatibility ----------
class TestBackwardCompat(unittest.TestCase):
    def test_phase_configs_forward(self):
        configs = [
            dict(temporal_adapter="sparse_period", periods="24+168", cross_mode="asym_lowrank",
                 residual_part="split", gate_type="hier_channel_band", gate_init_logit=-6.0,
                 temporal_gate_type="horizon"),                                   # phase4_asx_period_multi
            dict(temporal_adapter="sparse_period", periods="24+168", lift_sharing="individual",
                 cross_mode="none", temporal_gate_type="horizon"),               # phase4_asx_individual_period
            dict(temporal_adapter="sparse_period", period=24, cross_mode="asym_lowrank",
                 residual_part="split", gate_type="hier_channel_band", gate_init_logit=-6.0),  # phase3
            dict(cross_mode="asym_lowrank", residual_part="split", gate_type="hier_channel_band",
                 gate_init_logit=-6.0),                                           # phase2 hier_split
            dict(cross_mode="asym_lowrank", gate_type="global", gate_init_logit=0.0),  # phase1
        ]
        for c in configs:
            m = Model(make_configs(enc_in=5, **c))
            y = m(torch.randn(2, 48, 5))
            self.assertEqual(tuple(y.shape), (2, 24, 5))


# ---------- G/I. summary ----------
class TestSummary(unittest.TestCase):
    def test_summary_title_and_paired_and_missing_diag(self):
        rows = []
        for arm in ["phase5_asx_cross", "phase5_asx_period_multi"]:
            for seed in [1, 2]:
                for pl in [96, 192]:
                    rows.append(dict(arm=arm, dataset="electricity", seq_len=720, pred_len=pl, seed=seed,
                                     status="ok", val_mse=0.2, mse=0.15 if arm == "phase5_asx_period_multi" else 0.16,
                                     mae=0.25))
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "r.csv"); write_csv(c, rows)
            r = subprocess.run([sys.executable, "scripts/summarize_phase5.py", "--csv", c,
                                "--anchor_arm", "phase5_asx_cross", "--output_dir", d],
                               cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            txt = open(os.path.join(d, "summary_phase5.md")).read()
            self.assertIn("Phase 5-Lockdown Summary", txt)
            self.assertIn("Paired Statistics vs Anchor", txt)
            # paired: period_multi vs cross, 4 pairs
            self.assertIn("phase5_asx_period_multi", txt)


if __name__ == "__main__":
    unittest.main()
