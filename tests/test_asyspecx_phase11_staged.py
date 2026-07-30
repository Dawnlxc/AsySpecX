import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.AsySpecX import ForecastabilityAdapter
from scripts.benchmark_phase11_fixed_work import make_config
from scripts.summarize_phase11_staged import (
    ARMS,
    read_manifest,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
CELLS = (
    ("bridge", "ind_cycle_full", "weather", 96, 720, 2026, 13),
    ("identity", "ind_cycle_full", "ETTm1", 96, 96, 2026, 7),
    ("identity", "ind_cycle_full", "ETTm1", 96, 192, 2026, 7),
    ("active", "ind_cycle_full", "ETTm1", 96, 336, 2026, 7),
    ("active", "ind_cycle_full", "ETTm1", 96, 720, 2026, 7),
    ("identity", "ind_cycle_full", "traffic", 96, 96, 2026, 25),
    ("identity", "ind_cycle_full", "traffic", 96, 192, 2026, 25),
    ("active", "ind_cycle_full", "traffic", 96, 336, 2026, 25),
    ("active", "ind_cycle_full", "traffic", 96, 720, 2026, 25),
)


def scale(seq_len, pred_len):
    return max(0.0, 1.0 - 2.0 * seq_len / pred_len)


def make_case():
    rows = []
    specs = []
    for role, profile, dataset, seq_len, pred_len, seed, cut_freq in CELLS:
        values = {
            "fk_r8_cs": 0.2000000,
            "fk_sm2_mode": 0.1990000,
            "fk_sm2_tail2": (
                0.2000000 if role == "identity"
                else 0.2000200 if role == "bridge"
                else 0.1980000
            ),
        }
        for arm in ARMS:
            rho = scale(seq_len, pred_len) if arm == "fk_sm2_tail2" else 1.0
            identity = arm == "fk_sm2_tail2" and rho == 0.0
            row = {
                "base_profile": profile,
                "arm": arm,
                "dataset": dataset,
                "seq_len": seq_len,
                "pred_len": pred_len,
                "seed": seed,
                "cut_freq": cut_freq,
                "status": "ok",
                "test_deferred": True,
                "mse": None,
                "mae": None,
                "val_mse": values[arm],
                "n_param": 1005 if arm == "fk_sm2_tail2" else 1000,
                "forecast_extension_shrink": (
                    "tail2_linear" if arm == "fk_sm2_tail2" else "none"
                ),
                "forecast_kernel_extension_scale": rho,
                "forecast_kernel_extension_identity": float(identity),
                "forecast_kernel_sm_gate_abs_max": (
                    0.0 if identity else 0.5 if arm.startswith("fk_sm2") else None
                ),
                "forecast_kernel_sm_effective_gate_abs_max": (
                    rho * 0.5 if arm.startswith("fk_sm2") else None
                ),
                "forecast_kernel_sm_factor_min": (
                    1.0 if identity else 0.8 if arm.startswith("fk_sm2") else None
                ),
                "forecast_kernel_sm_factor_max": (
                    1.0 if identity else 1.2 if arm.startswith("fk_sm2") else None
                ),
            }
            rows.append(row)
            specs.append(
                {
                    "role": role,
                    "base_profile": profile,
                    "arm": arm,
                    "dataset": dataset,
                    "seq_len": seq_len,
                    "pred_len": pred_len,
                    "seed": seed,
                    "cut_freq": cut_freq,
                }
            )
    return rows, specs


def make_resources(active_train=10.5):
    resources = {}
    for dataset, pred_len, role in (
        ("traffic", 192, "identity"),
        ("ETTm1", 720, "active"),
        ("traffic", 720, "active"),
    ):
        rows = {}
        for arm in ARMS:
            if arm == "fk_r8_cs":
                train, infer, memory, params = 10.0, 5.0, 100.0, 1000
            elif arm == "fk_sm2_mode":
                train, infer, memory, params = 10.7, 5.2, 100.5, 1005
            elif role == "identity":
                train, infer, memory, params = 10.1, 5.05, 100.5, 1005
            else:
                train, infer, memory, params = active_train, 5.1, 100.5, 1005
            rows[arm] = {
                "arm": arm,
                "n_param": params,
                "train_forward_backward_ms_per_batch": train,
                "inference_ms_per_batch": infer,
                "fixed_work_peak_cuda_mb": memory,
            }
        resources[(dataset, 96, pred_len)] = rows
    return resources


class StageDHorizonSafeKernelTests(unittest.TestCase):
    def test_tail2_schedule_is_frozen(self):
        expected = {96: 0.0, 192: 0.0, 336: 3.0 / 7.0, 720: 11.0 / 15.0}
        for pred_len, target in expected.items():
            module = ForecastabilityAdapter(
                seq_len=96,
                pred_len=pred_len,
                channels=3,
                rank=8,
                spectral_mixtures=2,
                extension_shrink="tail2_linear",
            )
            self.assertAlmostEqual(module.extension_scale, target, places=12)

    def test_tail2_identity_has_exact_output_weight_and_common_gradients(self):
        torch.manual_seed(71)
        stage_a = ForecastabilityAdapter(
            seq_len=12,
            pred_len=24,
            channels=3,
            rank=4,
            init="small_random",
            channel_scale=True,
            gate_init_logit=-2.0,
        )
        torch.manual_seed(71)
        tail2 = ForecastabilityAdapter(
            seq_len=12,
            pred_len=24,
            channels=3,
            rank=4,
            init="small_random",
            channel_scale=True,
            spectral_mixtures=2,
            extension_shrink="tail2_linear",
            gate_init_logit=-2.0,
        )
        incompatible = tail2.load_state_dict(stage_a.state_dict(), strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(incompatible.missing_keys)
        self.assertEqual(tail2.extension_scale, 0.0)
        x_a = torch.randn(2, 12, 3, requires_grad=True)
        x_d = x_a.detach().clone().requires_grad_(True)
        base = torch.randn(2, 24, 3)
        target = torch.randn(2, 24, 3)
        out_a = stage_a(x_a, base)
        out_d = tail2(x_d, base)
        self.assertTrue(torch.equal(out_a, out_d))
        self.assertTrue(torch.equal(stage_a.weight_matrix(1), tail2.weight_matrix(1)))
        diagnostics = tail2.get_diagnostics()
        self.assertEqual(diagnostics["forecast_kernel_extension_scale"], 0.0)
        self.assertEqual(diagnostics["forecast_kernel_extension_identity"], 1.0)
        self.assertEqual(diagnostics["forecast_kernel_sm_gate_abs_max"], 0.0)
        self.assertEqual(diagnostics["forecast_kernel_sm_factor_min"], 1.0)
        self.assertEqual(diagnostics["forecast_kernel_sm_factor_max"], 1.0)
        torch.mean((out_a - target) ** 2).backward()
        torch.mean((out_d - target) ** 2).backward()
        self.assertTrue(torch.equal(x_a.grad, x_d.grad))
        stage_params = dict(stage_a.named_parameters())
        tail_params = dict(tail2.named_parameters())
        for name, parameter in stage_params.items():
            self.assertTrue(torch.equal(parameter.grad, tail_params[name].grad), name)
        for name in incompatible.missing_keys:
            if name in tail_params:
                self.assertIsNone(tail_params[name].grad, name)

    def test_active_tail2_is_power_shrink_of_unshrunk_factor(self):
        torch.manual_seed(73)
        full = ForecastabilityAdapter(
            seq_len=12,
            pred_len=90,
            channels=2,
            rank=4,
            spectral_mixtures=2,
            extension_shrink="none",
        )
        torch.manual_seed(73)
        tail2 = ForecastabilityAdapter(
            seq_len=12,
            pred_len=90,
            channels=2,
            rank=4,
            spectral_mixtures=2,
            extension_shrink="tail2_linear",
        )
        tail2.load_state_dict(full.state_dict(), strict=True)
        with torch.no_grad():
            full.forecast_kernel_sm_gate_logit.copy_(
                torch.tensor([0.6, -0.3, 0.2, 0.4])
            )
            full.sm_weight_logits.normal_(std=0.4)
            full.sm_center_offset.normal_(std=0.2)
            tail2.load_state_dict(full.state_dict(), strict=True)
        factor_full = full.spectral_mixture_factor()
        factor_tail = tail2.spectral_mixture_factor()
        torch.testing.assert_close(
            factor_tail,
            factor_full.pow(tail2.extension_scale),
            rtol=2e-6,
            atol=2e-7,
        )
        self.assertEqual(
            sum(p.numel() for p in full.parameters()),
            sum(p.numel() for p in tail2.parameters()),
        )

    def test_shrink_without_sm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires a spectral-mixture"):
            ForecastabilityAdapter(
                seq_len=12,
                pred_len=24,
                channels=2,
                extension_shrink="tail2_linear",
            )


class StageDSelectorTests(unittest.TestCase):
    def test_passing_case_advances_to_confirmation(self):
        rows, specs = make_case()
        _, _, aggregate, decision = summarize(rows, specs, make_resources())
        self.assertEqual(aggregate["eligible"], 1)
        self.assertEqual(decision["selected_arm"], "fk_sm2_tail2")
        self.assertEqual(decision["advance_to_confirmation"], 1)

    def test_identity_mismatch_rejects(self):
        rows, specs = make_case()
        rows = copy.deepcopy(rows)
        row = next(
            item for item in rows
            if item["arm"] == "fk_sm2_tail2"
            and item["dataset"] == "ETTm1"
            and item["pred_len"] == 96
        )
        row["val_mse"] += 1e-5
        _, _, aggregate, decision = summarize(rows, specs, make_resources())
        self.assertEqual(aggregate["identity_ok"], 0)
        self.assertIsNone(decision["selected_arm"])

    def test_resource_regression_rejects(self):
        rows, specs = make_case()
        _, _, aggregate, decision = summarize(
            rows, specs, make_resources(active_train=11.5)
        )
        self.assertEqual(aggregate["eligible"], 0)
        self.assertIsNone(decision["selected_arm"])

    def test_selector_rejects_test_open_row(self):
        rows, specs = make_case()
        rows = copy.deepcopy(rows)
        rows[0]["mse"] = 0.1
        with self.assertRaisesRegex(ValueError, "test-open"):
            summarize(rows, specs, make_resources())

    def test_locked_manifests_have_expected_rows(self):
        canary = read_manifest(ROOT / "configs/phase11_staged_canary.tsv")
        screen = read_manifest(ROOT / "configs/phase11_staged_screen.tsv")
        self.assertEqual(len(canary), 6)
        self.assertEqual(len(screen), 27)
        self.assertEqual({row["base_profile"] for row in screen}, {"ind_cycle_full"})
        self.assertEqual({row["arm"] for row in screen}, set(ARMS))


class StageDBenchmarkConfigTests(unittest.TestCase):
    def test_tail2_config_is_no_cross_and_has_frozen_schedule(self):
        args = SimpleNamespace(
            base_profile="ind_cycle_full",
            data_name="ETTm1",
            data_path="ETTm1.csv",
            seq_len=96,
            pred_len=720,
            channels=7,
            cycle=96,
            cut_freq=7,
            cross_rank=8,
            num_bands=8,
            periods="96+672",
        )
        config = make_config("fk_sm2_tail2", args)
        self.assertEqual(config.cross_mode, "none")
        self.assertEqual(config.temporal_adapter, "none")
        self.assertEqual(config.forecast_kernel_spectral_mixtures, 2)
        self.assertEqual(config.forecast_kernel_extension_shrink, "tail2_linear")


if __name__ == "__main__":
    unittest.main()
