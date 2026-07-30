import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from models.AsySpecX import Model
from router.blocks import advantage_targets, horizon_blocks
from router.configs import checkpoint_path
from router.features import context_features, train_only_channel_groups
from router.io import CompactMetaDataset, CompactMetaWriter, assert_training_metadata_safe
from router.manifest import load_expert_manifest, verify_sample_alignment
from router.safe import activation_diagnostics, calibrate_lcb_quantile, safe_route, safe_scores
from router.splits import PurgedTimeSeriesSplit, rolling_oof_windows
from scripts.audit_router_headroom import WARNING, analyse_cell


def model_config(**overrides):
    values = dict(
        seq_len=24, pred_len=8, enc_in=3, individual=0, cut_freq=4,
        spectral_lift="fits_linear", lift_sharing="shared", norm_mode="rin_noaffine",
        cross_mode="none", rank=2, num_bands=2, temporal_adapter="none",
        patch_adapter="none", linear_adapter="none", branch_fusion="sequential",
    )
    values.update(overrides)
    return Namespace(**values)


class TestManifest(unittest.TestCase):
    def _manifest(self, root, checkpoint):
        config = {
            "model": "AsySpecX", "dataset": "weather", "data": "custom",
            "root_path": "/tmp", "data_path": "weather.csv", "seq_len": 96,
            "pred_len": 24, "enc_in": 21,
        }
        payload = {
            "cell": {"dataset": "weather", "seq_len": 96, "pred_len": 24, "enc_in": 21},
            "experts": [
                {"name": "anchor", "arm": "a", "checkpoints": {"2024": checkpoint}, "config": config},
                {"name": "alt", "arm": "b", "checkpoints": {"2024": checkpoint}, "config": config},
            ],
        }
        path = Path(root) / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_manifest_loading_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "checkpoint.pth"
            checkpoint.touch()
            path = self._manifest(root, str(checkpoint))
            manifest = load_expert_manifest(str(path), seeds=[2024])
            self.assertEqual(manifest.names, ["anchor", "alt"])
            checkpoint.unlink()
            with self.assertRaises(FileNotFoundError):
                load_expert_manifest(str(path), seeds=[2024])

    def test_smaller_pool_keeps_anchor(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "checkpoint.pth"; checkpoint.touch()
            path = self._manifest(root, str(checkpoint))
            manifest = load_expert_manifest(str(path), expert_names=["alt"], seeds=[2024])
            self.assertEqual(manifest.names, ["anchor", "alt"])

    def test_sample_alignment(self):
        verify_sample_alignment([1, 2], [1, 2])
        with self.assertRaises(ValueError):
            verify_sample_alignment([1, 2], [2, 1], "alt")

    def test_frozen_checkpoint_path_uses_legacy_cycle_setting(self):
        path = checkpoint_path(
            Path("/tmp/checkpoints"),
            "phase7_period_multi_auto_acf_patchlinear",
            "weather",
            96,
            96,
            2024,
        )
        self.assertIn("_cycle24_seed2024/", str(path))
        self.assertNotIn("_cycle144_", str(path))


class TestBlocksAndOracle(unittest.TestCase):
    def test_horizon_boundaries(self):
        self.assertEqual(horizon_blocks(96, 4), [(0, 24), (24, 48), (48, 72), (72, 96)])
        self.assertEqual(horizon_blocks(10, 4), [(0, 3), (3, 6), (6, 8), (8, 10)])

    def test_advantage_sign(self):
        mse = np.array([[[2.0], [1.0], [3.0]]])
        advantage, regret = advantage_targets(mse, anchor_index=0)
        self.assertGreater(advantage[0, 0, 0], 0.0)
        self.assertLess(advantage[0, 1, 0], 0.0)
        self.assertLess(regret[0, 0, 0], 0.0)

    def test_oracles_and_analysis_label(self):
        with tempfile.TemporaryDirectory() as root:
            with CompactMetaWriter(
                root, ["x"], ["anchor", "alt"],
                {"split": "test", "router_meta_source": "test_analysis", "dataset": "weather", "seq_len": 96, "pred_len": 2, "enc_in": 1},
            ) as writer:
                writer.write(
                    features=np.zeros((4, 1), np.float32),
                    sample_id=np.array([0, 0, 1, 1]), origin=np.array([0, 0, 1, 1]),
                    block=np.array([0, 1, 0, 1]), block_start=np.array([0, 1, 0, 1]),
                    block_end=np.array([1, 2, 1, 2]),
                    seed_variance=np.zeros((4, 2), np.float32),
                    loss_mse=np.array([[1, 2], [1, 0], [1, 0], [1, 2]], np.float32),
                    loss_mae=np.ones((4, 2), np.float32),
                    advantage=np.zeros((4, 1), np.float32), relative_regret=np.zeros((4, 1), np.float32),
                )
            result = analyse_cell(CompactMetaDataset(root))
            self.assertAlmostEqual(result["cell_oracle_mse"], 1.0)
            self.assertAlmostEqual(result["sample_oracle_mse"], 1.0)
            self.assertAlmostEqual(result["horizon_block_oracle_mse"], 1.0)
            self.assertAlmostEqual(result["sample_block_oracle_mse"], 0.5)
            self.assertIn("ANALYSIS ONLY", WARNING)


class TestFeaturesAndStorage(unittest.TestCase):
    def test_context_features_finite_deterministic(self):
        x = np.random.RandomState(4).randn(3, 96, 12)
        first, names1 = context_features(x, 24, [12, 24], "weather", "cell")
        second, names2 = context_features(x, 24, [12, 24], "weather", "cell")
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(names1, names2)

    def test_large_c_approximation_deterministic(self):
        x = np.random.RandomState(7).randn(2, 48, 862)
        first, _ = context_features(x, 12, [24], "traffic", "cell", max_channels=64)
        second, _ = context_features(x, 12, [24], "traffic", "cell", max_channels=64)
        self.assertTrue(np.array_equal(first, second))

    def test_train_only_channel_groups_deterministic(self):
        series = np.random.RandomState(8).randn(256, 12)
        first, names1 = train_only_channel_groups(series, 3)
        second, names2 = train_only_channel_groups(series, 3)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(names1, names2)
        self.assertEqual(sorted(np.unique(first).tolist()), [0, 1, 2])

    def test_no_full_prediction_saved_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            with CompactMetaWriter(root, ["x"], ["anchor"], {"split": "val"}) as writer:
                writer.write(
                    features=np.zeros((1, 1), np.float32), sample_id=np.array([0]),
                    origin=np.array([0]), block=np.array([0]), block_start=np.array([0]),
                    block_end=np.array([1]), seed_variance=np.zeros((1, 1), np.float32),
                )
            part = next(CompactMetaDataset(root).iter_parts())
            self.assertNotIn("predictions", part)
            self.assertFalse(any("prediction" in path.name for path in Path(root).iterdir()))

    def test_router_rejects_test_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            with CompactMetaWriter(root, ["x"], ["anchor"], {"split": "test"}) as writer:
                writer.write(features=np.zeros((1, 1)), sample_id=np.array([0]))
            with self.assertRaises(ValueError):
                assert_training_metadata_safe([CompactMetaDataset(root)])


class TestTimeSplits(unittest.TestCase):
    def test_purged_split_no_overlap(self):
        splitter = PurgedTimeSeriesSplit(n_splits=4, purge_steps=12, pred_len=24)
        for train, validation in splitter.split(origins=np.arange(500)):
            self.assertLess(int(train.max()) + 24, int(validation.min()))

    def test_rolling_oof_chronology(self):
        windows = rolling_oof_windows(1000, purge_steps=96)
        self.assertEqual(len(windows), 2)
        for train, validation in windows:
            self.assertLess(int(train.max()) + 96, int(validation.min()))
            self.assertTrue(np.all(np.diff(validation) == 1))


class TestSafeDecisions(unittest.TestCase):
    def setUp(self):
        self.anchor = np.zeros((3, 4, 2), dtype=np.float32)
        self.alt = np.ones((3, 2, 4, 2), dtype=np.float32)

    def test_lcb_formula(self):
        predicted = np.array([0.2, 0.4])
        actual = np.array([0.1, 0.3])
        q = calibrate_lcb_quantile(predicted, actual, alpha=0.1)
        self.assertAlmostEqual(q, 0.1)
        self.assertTrue(np.allclose(predicted - q, actual))

    def test_uncertain_returns_exact_anchor(self):
        routed, diagnostics = safe_route(
            self.anchor, self.alt, np.full((3, 2), -1.0), 0.0,
            np.ones((3, 2)), uncertainty_beta=0.0,
        )
        self.assertTrue(np.array_equal(routed, self.anchor))
        self.assertFalse(diagnostics["active"].any())

        routed, diagnostics = safe_route(
            self.anchor, self.alt, np.full((3, 2), np.nan), 0.0,
            np.ones((3, 2)), uncertainty_beta=0.0,
        )
        self.assertTrue(np.array_equal(routed, self.anchor))
        self.assertFalse(diagnostics["active"].any())

    def test_alpha_bounds_and_zero_one(self):
        predicted = np.array([[-1.0, -2.0], [0.0, -2.0], [0.04, -2.0]])
        routed, diagnostics = safe_route(
            self.anchor, self.alt, predicted, 0.0, np.ones((3, 2)),
            uncertainty_beta=0.0, min_gain=0.0, full_gain=0.02,
        )
        self.assertTrue(((diagnostics["alpha"] >= 0) & (diagnostics["alpha"] <= 1)).all())
        self.assertTrue(np.array_equal(routed[0], self.anchor[0]))
        self.assertTrue(np.allclose(routed[2], self.alt[2, 0]))

    def test_seed_uncertainty_penalty(self):
        predicted = np.ones((1, 2))
        scores = safe_scores(predicted, 0.0, np.array([[0.01, 10.0]]), uncertainty_beta=0.1)
        self.assertGreater(scores[0, 0], scores[0, 1])

    def test_false_activation_diagnostics(self):
        diagnostics = {"active": np.array([True, True]), "alpha": np.ones(2), "top_index": np.array([0, 0])}
        result = activation_diagnostics(np.array([1.0, 1.0]), np.array([[2.0], [0.5]]), diagnostics, 0.5)
        self.assertAlmostEqual(result["false_activation_rate"], 0.5)
        self.assertAlmostEqual(result["catastrophic_activation_rate"], 0.5)


class TestBackwardCompatibility(unittest.TestCase):
    def test_phase8_default_path_unchanged(self):
        model = Model(model_config())
        output = model(torch.randn(2, 24, 3))
        self.assertEqual(tuple(output.shape), (2, 8, 3))


if __name__ == "__main__":
    unittest.main()
