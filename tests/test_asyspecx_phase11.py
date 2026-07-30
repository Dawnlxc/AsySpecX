import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from models.AsySpecX import ForecastabilityAdapter, Model
from scripts.build_phase11_forecastability_init import (
    MomentAccumulator,
    fit_ridge_svd,
    normalize_windows,
)


def make_config(**overrides):
    values = dict(
        seq_len=12,
        pred_len=6,
        enc_in=4,
        cycle=7,
        cut_freq=4,
        spectral_lift="fits_linear",
        lift_sharing="shared",
        norm_mode="rin_noaffine",
        cross_mode="none",
        temporal_adapter="none",
        patch_adapter="none",
        linear_adapter="none",
        forecast_kernel="none",
        branch_fusion="sequential",
        cycle_residual=0,
        cycle_residual_rank=0,
        cycle_residual_init_std=0.02,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def artifact_payload(seq_len=5, pred_len=4, rank=3, **meta_overrides):
    meta = {
        "format": "asyspecx_phase11_forecastability_v1",
        "data": "custom",
        "data_path": "weather.csv",
        "seq_len": seq_len,
        "pred_len": pred_len,
        "rank": rank,
        "norm_mode": "rin_noaffine",
        "split": "train",
        "train_only": True,
    }
    meta.update(meta_overrides)
    return {
        "past_basis": torch.randn(rank, seq_len),
        "future_basis": torch.randn(pred_len, rank),
        "horizon_bias": torch.randn(pred_len),
        "meta": meta,
    }


class ForecastabilityAdapterTests(unittest.TestCase):
    def test_real_sm_zero_gate_is_exact_stage_a_identity(self):
        torch.manual_seed(41)
        stage_a = ForecastabilityAdapter(
            seq_len=12,
            pred_len=7,
            channels=3,
            rank=4,
            init="small_random",
            channel_scale=True,
        )
        torch.manual_seed(41)
        stage_b = ForecastabilityAdapter(
            seq_len=12,
            pred_len=7,
            channels=3,
            rank=4,
            init="small_random",
            channel_scale=True,
            spectral_mixtures=4,
            sm_sharing="mode",
        )
        incompatible = stage_b.load_state_dict(stage_a.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all("sm_" in key for key in incompatible.missing_keys))
        self.assertTrue(
            torch.equal(stage_b.effective_past_basis(), stage_b.past_basis)
        )
        x = torch.randn(2, 12, 3)
        self.assertTrue(torch.equal(stage_b.raw(x), stage_a.raw(x)))

    def test_real_sm_response_is_positive_real_and_zero_phase(self):
        torch.manual_seed(43)
        module = ForecastabilityAdapter(
            seq_len=15,
            pred_len=6,
            channels=2,
            rank=3,
            init="small_random",
            spectral_mixtures=4,
            sm_sharing="mode",
        )
        with torch.no_grad():
            module.forecast_kernel_sm_gate_logit.copy_(
                torch.tensor([0.7, -0.4, 0.25])
            )
            module.sm_weight_logits.normal_(std=0.4)
            module.sm_center_offset.normal_(std=0.2)
        factor = module.spectral_mixture_factor()
        self.assertFalse(factor.is_complex())
        self.assertGreater(float(factor.min()), 0.0)
        base_spectrum = torch.fft.rfft(module.past_basis, dim=-1)
        effective_spectrum = torch.fft.rfft(module.effective_past_basis(), dim=-1)
        torch.testing.assert_close(
            effective_spectrum,
            base_spectrum * factor.to(base_spectrum.dtype),
            rtol=2e-5,
            atol=2e-6,
        )
        valid = base_spectrum.abs() > 1e-4
        ratio = effective_spectrum[valid] / base_spectrum[valid]
        self.assertLess(float(ratio.imag.abs().max()), 2e-5)
        self.assertGreater(float(ratio.real.min()), 0.0)

    def test_real_sm_is_channel_separable_and_all_parameters_learn(self):
        torch.manual_seed(47)
        module = ForecastabilityAdapter(
            seq_len=14,
            pred_len=5,
            channels=4,
            rank=3,
            init="small_random",
            channel_scale=True,
            spectral_mixtures=2,
            sm_sharing="mode",
            gate_init_logit=-2.0,
        )
        with torch.no_grad():
            module.forecast_kernel_sm_gate_logit.fill_(0.2)
        x = torch.randn(3, 14, 4)
        changed = x.clone()
        changed[:, :, 2] += 50.0 * torch.randn_like(changed[:, :, 2])
        original = module.raw(x)
        perturbed = module.raw(changed)
        torch.testing.assert_close(
            original[:, :, [0, 1, 3]],
            perturbed[:, :, [0, 1, 3]],
            rtol=0.0,
            atol=0.0,
        )

        base = torch.randn_like(original)
        target = torch.randn_like(original)
        torch.mean((module(x, base) - target) ** 2).backward()
        for parameter in (
            module.sm_weight_logits,
            module.sm_center_offset,
            module.sm_width_raw,
            module.forecast_kernel_sm_gate_logit,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_frozen_sm_base_is_not_trainable(self):
        module = ForecastabilityAdapter(
            seq_len=12,
            pred_len=6,
            channels=3,
            rank=4,
            init="small_random",
            spectral_mixtures=4,
            sm_base_trainable=False,
        )
        self.assertFalse(module.past_basis.requires_grad)
        self.assertTrue(module.future_basis.requires_grad)
        self.assertTrue(module.sm_weight_logits.requires_grad)

    def test_eval_cache_is_static_and_invalidates_on_parameter_change(self):
        module = ForecastabilityAdapter(
            seq_len=13,
            pred_len=5,
            channels=2,
            rank=3,
            init="small_random",
            spectral_mixtures=4,
        ).eval()
        with torch.no_grad():
            module.forecast_kernel_sm_gate_logit.fill_(0.2)
            first = module.effective_past_basis()
            second = module.effective_past_basis()
            self.assertEqual(first.data_ptr(), second.data_ptr())
            module.forecast_kernel_sm_gate_logit.add_(0.1)
            changed = module.effective_past_basis()
            self.assertNotEqual(first.data_ptr(), changed.data_ptr())
            self.assertGreater(float((changed - first).abs().sum()), 0.0)
        self.assertNotIn("sm_frequency", module.state_dict())
        module.train()
        self.assertIsNone(module._sm_eval_basis_cache)

    def test_zero_complex_phase_is_exact_real_sm_identity(self):
        torch.manual_seed(53)
        real_sm = ForecastabilityAdapter(
            seq_len=16,
            pred_len=9,
            channels=3,
            rank=4,
            init="small_random",
            channel_scale=True,
            spectral_mixtures=4,
        )
        torch.manual_seed(53)
        complex_sm = ForecastabilityAdapter(
            seq_len=16,
            pred_len=9,
            channels=3,
            rank=4,
            init="small_random",
            channel_scale=True,
            spectral_mixtures=4,
            phase_basis_dim=4,
        )
        incompatible = complex_sm.load_state_dict(real_sm.state_dict(), strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(incompatible.missing_keys, ["phase_coeff"])
        x = torch.randn(2, 16, 3)
        self.assertTrue(torch.equal(complex_sm.raw(x), real_sm.raw(x)))

    def test_complex_quadrature_and_phase_bounds(self):
        torch.manual_seed(59)
        module = ForecastabilityAdapter(
            seq_len=16,
            pred_len=11,
            channels=2,
            rank=3,
            init="small_random",
            spectral_mixtures=4,
            phase_basis_dim=4,
            phase_max=0.6,
        )
        with torch.no_grad():
            module.forecast_kernel_sm_gate_logit.fill_(0.3)
            module.phase_coeff.normal_(std=0.8)
        past = module.effective_past_basis()
        quadrature = module.quadrature_past_basis(past)
        spectrum = torch.fft.rfft(past, dim=-1)
        expected = spectrum * module.phase_quadrature_multiplier.to(spectrum.dtype)
        torch.testing.assert_close(
            torch.fft.rfft(quadrature, dim=-1),
            expected,
            rtol=2e-5,
            atol=2e-6,
        )
        phase = module.horizon_phase()
        self.assertLessEqual(float(phase.abs().max()), 0.6 + 1e-7)
        self.assertFalse(phase.is_complex())

    def test_complex_phase_is_channel_separable_and_learns(self):
        torch.manual_seed(61)
        module = ForecastabilityAdapter(
            seq_len=15,
            pred_len=7,
            channels=4,
            rank=3,
            init="small_random",
            channel_scale=True,
            spectral_mixtures=4,
            phase_basis_dim=2,
            gate_init_logit=-2.0,
        )
        x = torch.randn(3, 15, 4)
        changed = x.clone()
        changed[:, :, 1] += 30.0 * torch.randn_like(changed[:, :, 1])
        original = module.raw(x)
        perturbed = module.raw(changed)
        torch.testing.assert_close(
            original[:, :, [0, 2, 3]],
            perturbed[:, :, [0, 2, 3]],
            rtol=0.0,
            atol=0.0,
        )
        base = torch.randn_like(original)
        target = torch.randn_like(original)
        torch.mean((module(x, base) - target) ** 2).backward()
        self.assertIsNotNone(module.phase_coeff.grad)
        self.assertGreater(float(module.phase_coeff.grad.abs().sum()), 0.0)

    def test_complex_phase_adds_only_rank_times_basis_parameters(self):
        common = dict(
            seq_len=12,
            pred_len=9,
            channels=5,
            rank=4,
            init="zeros",
            channel_scale=True,
            spectral_mixtures=4,
        )
        real_sm = ForecastabilityAdapter(**common)
        complex_sm = ForecastabilityAdapter(**common, phase_basis_dim=3)
        real_count = sum(p.numel() for p in real_sm.parameters())
        complex_count = sum(p.numel() for p in complex_sm.parameters())
        self.assertEqual(complex_count - real_count, 4 * 3)
        state = complex_sm.state_dict()
        self.assertNotIn("phase_horizon_basis", state)
        self.assertNotIn("phase_quadrature_multiplier", state)

    def test_raw_matches_manual_dense_per_channel_kernel(self):
        module = ForecastabilityAdapter(
            seq_len=5,
            pred_len=4,
            channels=3,
            rank=2,
            init="zeros",
            channel_scale=True,
        )
        with torch.no_grad():
            module.past_basis.copy_(
                torch.tensor([[1.0, -2.0, 0.5, 0.0, 1.5], [0.0, 1.0, -1.0, 2.0, 0.5]])
            )
            module.future_basis.copy_(
                torch.tensor([[1.0, 0.0], [0.5, -1.0], [0.0, 2.0], [-0.5, 0.25]])
            )
            module.channel_scale.copy_(
                torch.tensor([[1.0, 2.0], [0.25, -1.0], [3.0, 0.5]])
            )
            module.horizon_bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4]))

        x = torch.randn(2, 5, 3)
        expected = torch.empty(2, 4, 3)
        for channel in range(3):
            weight = module.weight_matrix(channel)
            expected[:, :, channel] = (
                x[:, :, channel] @ weight.transpose(0, 1) + module.horizon_bias
            )
        torch.testing.assert_close(module.raw(x), expected)

    def test_changing_one_channel_cannot_change_other_outputs(self):
        torch.manual_seed(3)
        module = ForecastabilityAdapter(
            seq_len=8,
            pred_len=5,
            channels=4,
            rank=3,
            init="small_random",
            channel_scale=True,
        )
        x = torch.randn(2, 8, 4)
        changed = x.clone()
        changed[:, :, 1] += 100.0 * torch.randn_like(changed[:, :, 1])
        original_y = module.raw(x)
        changed_y = module.raw(changed)
        unaffected = [0, 2, 3]
        torch.testing.assert_close(
            original_y[:, :, unaffected],
            changed_y[:, :, unaffected],
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(float((original_y[:, :, 1] - changed_y[:, :, 1]).abs().sum()), 0.0)

    def test_all_factors_scales_and_gate_receive_gradients(self):
        torch.manual_seed(5)
        module = ForecastabilityAdapter(
            seq_len=7,
            pred_len=4,
            channels=3,
            rank=3,
            init="small_random",
            channel_scale=True,
            fusion="convex",
            gate_type="horizon_channel",
            gate_init_logit=-2.0,
        )
        x = torch.randn(4, 7, 3)
        base = torch.randn(4, 4, 3)
        target = torch.randn_like(base)
        torch.mean((module(x, base) - target) ** 2).backward()
        for parameter in (
            module.past_basis,
            module.future_basis,
            module.channel_scale,
            module.forecast_kernel_gate_logit,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_parameter_count_is_low_rank_formula(self):
        module = ForecastabilityAdapter(
            seq_len=12,
            pred_len=9,
            channels=5,
            rank=4,
            init="zeros",
            channel_scale=True,
            gate_type="horizon",
        )
        expected = 4 * 12 + 9 * 4 + 9 + 5 * 4 + 9
        self.assertEqual(sum(p.numel() for p in module.parameters()), expected)
        dense_shared_with_bias = 9 * 12 + 9
        self.assertLess(expected, dense_shared_with_bias + 9)  # same horizon-gate budget

    def test_ridge_artifact_is_validated_and_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "init.pt")
            payload = artifact_payload()
            torch.save(payload, path)
            module = ForecastabilityAdapter(
                seq_len=5,
                pred_len=4,
                channels=2,
                rank=2,
                init="ridge_svd",
                init_path=path,
                expected_data="custom",
                expected_data_path="weather.csv",
                expected_norm_mode="rin_noaffine",
            )
            torch.testing.assert_close(module.past_basis, payload["past_basis"][:2])
            torch.testing.assert_close(module.future_basis, payload["future_basis"][:, :2])
            torch.testing.assert_close(module.horizon_bias, payload["horizon_bias"])

            invalid_cases = {
                "train split": artifact_payload(split="val", train_only=False),
                "rank mismatch": artifact_payload(rank=1),
                "data mismatch": artifact_payload(data="ETTh1"),
                "seq_len mismatch": artifact_payload(seq_len=6),
            }
            for label, invalid in invalid_cases.items():
                with self.subTest(label=label):
                    torch.save(invalid, path)
                    with self.assertRaises(ValueError):
                        ForecastabilityAdapter(
                            seq_len=5,
                            pred_len=4,
                            channels=2,
                            rank=2,
                            init="ridge_svd",
                            init_path=path,
                            expected_data="custom",
                            expected_data_path="weather.csv",
                            expected_norm_mode="rin_noaffine",
                        )


class ForecastabilityModelTests(unittest.TestCase):
    def test_kernel_off_is_bitwise_legacy_equivalent(self):
        torch.manual_seed(17)
        legacy = Model(make_config()).eval()
        torch.manual_seed(17)
        explicit_off = Model(make_config(forecast_kernel="none")).eval()
        self.assertEqual(list(legacy.state_dict()), list(explicit_off.state_dict()))
        for name, value in legacy.state_dict().items():
            torch.testing.assert_close(value, explicit_off.state_dict()[name], rtol=0.0, atol=0.0)
        x = torch.randn(3, 12, 4)
        with torch.no_grad():
            expected = legacy(x)
            actual = explicit_off(x)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_model_sequential_and_parallel_paths_have_expected_shape(self):
        for fusion in ("sequential", "softmax_static"):
            with self.subTest(fusion=fusion):
                model = Model(
                    make_config(
                        forecast_kernel="lowrank_time",
                        forecast_kernel_rank=3,
                        forecast_kernel_init="small_random",
                        forecast_kernel_channel_scale=1,
                        branch_fusion=fusion,
                    )
                )
                output = model(torch.randn(2, 12, 4))
                self.assertEqual(tuple(output.shape), (2, 6, 4))
                diagnostics = model.get_diagnostics()
                self.assertEqual(diagnostics["forecast_kernel_enabled"], 1.0)
                self.assertEqual(diagnostics["forecast_kernel_rank"], 3.0)
                if fusion == "sequential":
                    self.assertIn("forecast_kernel_effective_rank", diagnostics)
                else:
                    self.assertIn("branch_weight_forecast_mean", diagnostics)

    def test_stage_b_config_wires_real_sm_diagnostics(self):
        model = Model(
            make_config(
                forecast_kernel="lowrank_time",
                forecast_kernel_rank=3,
                forecast_kernel_init="small_random",
                forecast_kernel_channel_scale=1,
                forecast_kernel_spectral_mixtures=4,
                forecast_kernel_sm_sharing="mode",
            )
        )
        output = model(torch.randn(2, 12, 4))
        self.assertEqual(tuple(output.shape), (2, 6, 4))
        diagnostics = model.get_diagnostics()
        self.assertEqual(diagnostics["forecast_kernel_sm_enabled"], 1.0)
        self.assertEqual(diagnostics["forecast_kernel_sm_components"], 4.0)
        self.assertEqual(diagnostics["forecast_kernel_sm_factor_min"], 1.0)
        self.assertEqual(diagnostics["forecast_kernel_sm_factor_max"], 1.0)

    def test_complex_phase_config_wires_bounded_diagnostics(self):
        model = Model(
            make_config(
                forecast_kernel="lowrank_time",
                forecast_kernel_rank=3,
                forecast_kernel_init="small_random",
                forecast_kernel_channel_scale=1,
                forecast_kernel_spectral_mixtures=4,
                forecast_kernel_sm_sharing="mode",
                forecast_kernel_phase_basis_dim=2,
                forecast_kernel_phase_max=0.5,
            )
        )
        output = model(torch.randn(2, 12, 4))
        self.assertEqual(tuple(output.shape), (2, 6, 4))
        diagnostics = model.get_diagnostics()
        self.assertEqual(diagnostics["forecast_kernel_phase_enabled"], 1.0)
        self.assertEqual(diagnostics["forecast_kernel_phase_basis_dim"], 2.0)
        self.assertEqual(diagnostics["forecast_kernel_phase_abs_max"], 0.0)


class TrainOnlyInitializerMathTests(unittest.TestCase):
    def test_normalization_matches_input_derived_rin_formula(self):
        x = torch.tensor([[[1.0], [2.0], [4.0]]], dtype=torch.float64)
        y = torch.tensor([[[5.0], [8.0]]], dtype=torch.float64)
        x_norm, y_norm = normalize_windows(x, y, "rin_noaffine")
        loc = x.mean(dim=1, keepdim=True)
        scale = torch.sqrt(torch.var(x - loc, dim=1, keepdim=True, correction=1) + 1e-5)
        torch.testing.assert_close(x_norm, (x - loc) / scale)
        torch.testing.assert_close(y_norm, (y - loc) / scale)

    def test_ridge_svd_recovers_exact_rank_two_affine_map(self):
        torch.manual_seed(23)
        records, seq_len, pred_len, rank = 256, 6, 5, 2
        x = torch.randn(records, seq_len, dtype=torch.float64)
        left = torch.randn(pred_len, rank, dtype=torch.float64)
        right = torch.randn(rank, seq_len, dtype=torch.float64)
        weight = left @ right
        bias = torch.randn(pred_len, dtype=torch.float64)
        y = x @ weight.transpose(0, 1) + bias
        moments = MomentAccumulator(seq_len, pred_len)
        moments.update(x, y)
        fitted = fit_ridge_svd(moments, rank=rank, ridge=0.0)
        recovered = fitted["future_basis"].double() @ fitted["past_basis"].double()
        torch.testing.assert_close(recovered, weight, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(fitted["horizon_bias"].double(), bias, rtol=2e-5, atol=2e-5)
        self.assertGreater(fitted["retained_energy"], 0.999999)
        self.assertLess(fitted["centered_train_mse"], 1e-10)


if __name__ == "__main__":
    unittest.main()
