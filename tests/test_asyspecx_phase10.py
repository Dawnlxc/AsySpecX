import unittest
from types import SimpleNamespace

import torch

from models.AsySpecX import (
    ChannelLowRankComplexLinear,
    CompactPeriodAdapter,
    CycleResidual,
    Model,
    SparsePeriodAdapter,
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
        branch_fusion="sequential",
        cycle_residual=0,
        cycle_residual_rank=0,
        cycle_residual_init_std=0.02,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class CycleResidualTests(unittest.TestCase):
    def test_phase_alignment_uses_forecast_origin(self):
        module = CycleResidual(cycle_len=6, channels=2, rank=0)
        with torch.no_grad():
            module.table.copy_(torch.arange(12, dtype=torch.float32).reshape(6, 2))
        cycle_in, cycle_future = module.split(torch.tensor([2]), seq_len=3, pred_len=4)
        expected_in = module.table[torch.tensor([5, 0, 1])].unsqueeze(0)
        expected_future = module.table[torch.tensor([2, 3, 4, 5])].unsqueeze(0)
        torch.testing.assert_close(cycle_in, expected_in)
        torch.testing.assert_close(cycle_future, expected_future)

    def test_zero_initialized_full_table_preserves_legacy_prediction(self):
        torch.manual_seed(11)
        legacy = Model(make_config(cycle_residual=0)).eval()
        cycle = Model(make_config(cycle_residual=1, cycle_residual_rank=0)).eval()
        result = cycle.load_state_dict(legacy.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(result.missing_keys, ["cycle_residual.table"])

        x = torch.randn(3, 12, 4)
        phase = torch.tensor([0, 3, 6])
        with torch.no_grad():
            expected = legacy(x)
            actual = cycle(x, cycle_index=phase)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_factorized_table_is_zero_initialized_and_trainable(self):
        module = CycleResidual(cycle_len=9, channels=5, rank=3)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 3 * (9 + 5))
        torch.testing.assert_close(module.values(), torch.zeros(9, 5))

        future = module.gather(torch.tensor([1, 4]), start_offset=0, length=6)
        target = torch.randn_like(future)
        loss = torch.mean((future - target) ** 2)
        loss.backward()
        self.assertIsNotNone(module.phase_factor.grad)
        self.assertGreater(float(module.phase_factor.grad.abs().sum()), 0.0)

        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        optimizer.step()
        optimizer.zero_grad()
        future = module.gather(torch.tensor([1, 4]), start_offset=0, length=6)
        torch.mean((future - target) ** 2).backward()
        self.assertGreater(float(module.channel_factor.grad.abs().sum()), 0.0)

    def test_factorized_table_rejects_zero_init_std(self):
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            CycleResidual(cycle_len=9, channels=5, rank=3, init_std=0.0)

    def test_enabled_model_requires_cycle_index(self):
        model = Model(make_config(cycle_residual=1))
        with self.assertRaisesRegex(ValueError, "requires cycle_index"):
            model(torch.randn(2, 12, 4))

    def test_cycle_table_receives_forecast_loss_gradient(self):
        model = Model(make_config(cycle_residual=1, cycle_residual_rank=0))
        output = model(torch.randn(2, 12, 4), cycle_index=torch.tensor([1, 5]))
        loss = torch.mean(output ** 2)
        loss.backward()
        self.assertIsNotNone(model.cycle_residual.table.grad)
        self.assertGreater(float(model.cycle_residual.table.grad.abs().sum()), 0.0)

    def test_legacy_positional_return_full_is_unchanged(self):
        model = Model(make_config(cycle_residual=0)).eval()
        output = model(torch.randn(2, 12, 4), True)
        self.assertIsInstance(output, dict)
        self.assertEqual(tuple(output["pred"].shape), (2, 6, 4))


class ChannelLowRankLiftTests(unittest.TestCase):
    def test_zero_delta_exactly_matches_shared_base(self):
        torch.manual_seed(7)
        module = ChannelLowRankComplexLinear(5, 8, channels=4, rank=2).eval()
        x = torch.complex(torch.randn(3, 4, 5), torch.randn(3, 4, 5))
        with torch.no_grad():
            expected = module.base(x)
            actual = module(x)
        # The fused channel einsum changes floating-point contraction order,
        # but is mathematically identical to the legacy shared lift.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_nonzero_delta_matches_manual_channel_weights(self):
        torch.manual_seed(11)
        module = ChannelLowRankComplexLinear(5, 8, channels=4, rank=2).eval()
        with torch.no_grad():
            module.delta_weight_re.normal_(std=0.02)
            module.delta_weight_im.normal_(std=0.02)
        x = torch.complex(torch.randn(3, 4, 5), torch.randn(3, 4, 5))
        expected = []
        for channel in range(4):
            weight_re = module.base.weight_re + torch.einsum(
                "r,roi->oi", module.channel_coeff[channel], module.delta_weight_re
            )
            weight_im = module.base.weight_im + torch.einsum(
                "r,roi->oi", module.channel_coeff[channel], module.delta_weight_im
            )
            weight = torch.complex(weight_re, weight_im)
            bias = torch.complex(module.base.bias_re, module.base.bias_im)
            expected.append(torch.einsum("bi,oi->bo", x[:, channel], weight) + bias)
        expected = torch.stack(expected, dim=1)
        torch.testing.assert_close(module(x), expected, rtol=1e-5, atol=1e-6)

    def test_parameter_count_and_two_step_gradients(self):
        module = ChannelLowRankComplexLinear(5, 8, channels=4, rank=2)
        base_params = 2 * 5 * 8 + 2 * 8
        delta_params = 2 * 2 * 5 * 8
        coeff_params = 4 * 2
        self.assertEqual(sum(p.numel() for p in module.parameters()),
                         base_params + delta_params + coeff_params)

        x = torch.complex(torch.randn(3, 4, 5), torch.randn(3, 4, 5))
        target = torch.complex(torch.randn(3, 4, 8), torch.randn(3, 4, 8))
        optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
        loss = torch.mean(torch.abs(module(x) - target) ** 2)
        loss.backward()
        self.assertGreater(float(module.delta_weight_re.grad.abs().sum()), 0.0)
        self.assertEqual(float(module.channel_coeff.grad.abs().sum()), 0.0)
        optimizer.step()
        optimizer.zero_grad()
        torch.mean(torch.abs(module(x) - target) ** 2).backward()
        self.assertGreater(float(module.channel_coeff.grad.abs().sum()), 0.0)

    def test_model_forward(self):
        model = Model(make_config(
            lift_sharing="lowrank_channel", lift_rank=2, cycle_residual=1
        ))
        y = model(torch.randn(2, 12, 4), cycle_index=torch.tensor([1, 5]))
        self.assertEqual(tuple(y.shape), (2, 6, 4))

    def test_invalid_lowrank_configs_fail_fast(self):
        with self.assertRaises(ValueError):
            ChannelLowRankComplexLinear(5, 8, channels=4, rank=0)
        with self.assertRaises(ValueError):
            Model(make_config(
                spectral_lift="complex_mlp", lift_sharing="lowrank_channel"
            ))
        with self.assertRaises(ValueError):
            Model(make_config(
                lift_sharing="lowrank_channel", individual=1
            ))


class CompactPeriodAdapterTests(unittest.TestCase):
    @staticmethod
    def make_adapter(cls, periodic_init="seasonal_naive", **overrides):
        values = dict(
            seq_len=17,
            pred_len=11,
            channels=3,
            periods=(5, 7),
            periodic_init=periodic_init,
            periodic_sharing="shared",
            temporal_fusion="convex",
            temporal_gate_type="horizon_channel",
            temporal_gate_init_logit=-1.25,
            period_fusion="sum_gated",
            period_gate_type="period_horizon_channel",
            period_gate_init_logit=0.3,
            periodic_l1_weight=0.2,
            periodic_l2_weight=0.1,
            temporal_gate_l1_weight=0.05,
        )
        values.update(overrides)
        return cls(**values)

    def test_all_initializers_match_sparse_reference(self):
        for periodic_init in ("seasonal_naive", "zeros", "small_random"):
            with self.subTest(periodic_init=periodic_init):
                torch.manual_seed(29)
                sparse = self.make_adapter(SparsePeriodAdapter, periodic_init)
                torch.manual_seed(29)
                compact = self.make_adapter(CompactPeriodAdapter, periodic_init)
                x = torch.randn(4, 17, 3)
                pred_spec = torch.randn(4, 11, 3)
                sparse_fused, sparse_period = sparse(x, pred_spec)
                compact_fused, compact_period = compact(x, pred_spec)
                # Nonzero taps can differ at roundoff level because compact
                # phase buckets change einsum contraction order.
                tolerance = 1e-6 if periodic_init == "small_random" else 0.0
                torch.testing.assert_close(
                    compact_period, sparse_period, rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(
                    compact_fused, sparse_fused, rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(compact.extra_loss(), sparse.extra_loss())
                sparse_diag = sparse.get_diagnostics()
                compact_diag = compact.get_diagnostics()
                for key in (
                    "periods", "temporal_gate_mean", "period_weight_mean",
                    "periodic_weight_abs_mean", "periodic_weight_abs_max",
                    "periodic_mask_density", "pred_period_norm_rms",
                    "pred_spec_norm_rms", "fused_delta_rms",
                ):
                    if isinstance(sparse_diag[key], str):
                        self.assertEqual(compact_diag[key], sparse_diag[key])
                    else:
                        self.assertAlmostEqual(compact_diag[key], sparse_diag[key], places=6)

    def test_arbitrary_legal_weights_are_exactly_expressive(self):
        torch.manual_seed(41)
        sparse = self.make_adapter(
            SparsePeriodAdapter, "zeros", period_fusion="softmax", temporal_fusion="additive"
        )
        compact = self.make_adapter(
            CompactPeriodAdapter, "zeros", period_fusion="softmax", temporal_fusion="additive"
        )
        with torch.no_grad():
            sparse.W_raw.normal_()
            sparse.period_gate_logits.normal_()
            sparse.temporal_gate_logit.normal_()
            compact._copy_from_dense(sparse.W_raw)
            compact.period_gate_logits.copy_(sparse.period_gate_logits)
            compact.temporal_gate_logit.copy_(sparse.temporal_gate_logit)
        x = torch.randn(2, 17, 3)
        pred_spec = torch.randn(2, 11, 3)
        sparse_fused, sparse_period = sparse(x, pred_spec)
        compact_fused, compact_period = compact(x, pred_spec)
        torch.testing.assert_close(compact_period, sparse_period, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(compact_fused, sparse_fused, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(compact.extra_loss(), sparse.extra_loss())

    def test_shape_and_gradients(self):
        compact = self.make_adapter(CompactPeriodAdapter)
        x = torch.randn(4, 17, 3, requires_grad=True)
        pred_spec = torch.randn(4, 11, 3, requires_grad=True)
        fused, period = compact(x, pred_spec)
        self.assertEqual(tuple(fused.shape), (4, 11, 3))
        self.assertEqual(tuple(period.shape), (4, 11, 3))
        (fused.square().mean() + compact.extra_loss()).backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(pred_spec.grad)
        for block in compact.period_blocks:
            for weight in block.weights:
                self.assertIsNotNone(weight.grad)
                self.assertGreater(float(weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(compact.period_gate_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(compact.temporal_gate_logit.grad.abs().sum()), 0.0)

    def test_noncontiguous_input_and_no_legal_phase_boundary(self):
        values = dict(seq_len=5, pred_len=13, periods=(3, 20), periodic_init="zeros")
        torch.manual_seed(53)
        sparse = self.make_adapter(SparsePeriodAdapter, **values)
        compact = self.make_adapter(CompactPeriodAdapter, **values)
        with torch.no_grad():
            sparse.W_raw.normal_()
            sparse.period_gate_logits.normal_()
            sparse.temporal_gate_logit.normal_()
            compact._copy_from_dense(sparse.W_raw)
            compact.period_gate_logits.copy_(sparse.period_gate_logits)
            compact.temporal_gate_logit.copy_(sparse.temporal_gate_logit)

        sparse_base = torch.randn(2, 3, 5, requires_grad=True)
        compact_base = sparse_base.detach().clone().requires_grad_(True)
        sparse_x = sparse_base.transpose(1, 2)
        compact_x = compact_base.transpose(1, 2)
        self.assertFalse(compact_x.is_contiguous())
        sparse_spec = torch.randn(2, 13, 3, requires_grad=True)
        compact_spec = sparse_spec.detach().clone().requires_grad_(True)
        sparse_fused, sparse_period = sparse(sparse_x, sparse_spec)
        compact_fused, compact_period = compact(compact_x, compact_spec)
        torch.testing.assert_close(compact_period, sparse_period, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(compact_fused, sparse_fused, rtol=1e-6, atol=1e-6)

        upstream = torch.randn_like(sparse_fused)
        (sparse_fused * upstream).sum().backward()
        (compact_fused * upstream).sum().backward()
        torch.testing.assert_close(compact_base.grad, sparse_base.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(compact_spec.grad, sparse_spec.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            compact.period_gate_logits.grad, sparse.period_gate_logits.grad,
            rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(
            compact.temporal_gate_logit.grad, sparse.temporal_gate_logit.grad,
            rtol=1e-5, atol=1e-6,
        )
        for period_index, block in enumerate(compact.period_blocks):
            for weight, input_index, horizon_index, _ in block.groups():
                sparse_grad = sparse.W_raw.grad[period_index][
                    horizon_index.unsqueeze(-1), input_index.unsqueeze(1)
                ]
                torch.testing.assert_close(weight.grad, sparse_grad, rtol=1e-5, atol=1e-6)

    def test_autocast_fails_fast(self):
        compact = self.make_adapter(
            CompactPeriodAdapter, "seasonal_naive", seq_len=168, pred_len=96,
            periods=(24, 168),
        )
        x = torch.randn(2, 168, 3)
        pred_spec = torch.randn(2, 96, 3)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(RuntimeError, "requires autocast/AMP off"):
                compact(x, pred_spec)

    def test_parameter_count_tracks_only_legal_taps(self):
        values = dict(
            seq_len=720,
            pred_len=336,
            channels=7,
            periods=(24, 168),
            temporal_gate_type="horizon",
            period_gate_type="period",
        )
        sparse = self.make_adapter(SparsePeriodAdapter, **values)
        compact = self.make_adapter(CompactPeriodAdapter, **values)
        sparse_parameters = sum(p.numel() for p in sparse.parameters())
        compact_parameters = sum(p.numel() for p in compact.parameters())
        legal_weights = int(sparse.periodic_mask.sum())
        self.assertEqual(compact._num_legal_weights, legal_weights)
        self.assertLess(compact_parameters, sparse_parameters / 20)

    def test_model_selects_compact_adapter(self):
        model = Model(make_config(
            temporal_adapter="compact_period",
            periods="3+7",
            periodic_init="seasonal_naive",
        ))
        self.assertIsInstance(model.temporal_adapter, CompactPeriodAdapter)
        output = model(torch.randn(2, 12, 4))
        self.assertEqual(tuple(output.shape), (2, 6, 4))


if __name__ == "__main__":
    unittest.main()
