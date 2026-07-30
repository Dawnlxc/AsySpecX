import math
import unittest
from argparse import Namespace

import torch
import torch.nn as nn

from models.AsySpecX import AsymCross, Model
from models.AsySpecX import SparsePeriodAdapter
from exp.exp_main import Exp_Main


def make_configs(**overrides):
    cfg = dict(
        seq_len=96,
        pred_len=96,
        enc_in=5,
        individual=0,
        cut_freq=0,
        spectral_lift="fits_linear",
        lift_sharing="shared",
        norm_mode="rin_noaffine",
        cross_mode="none",
        rank=2,
        num_bands=4,
        gate_init=0.0,
        gate_init_logit=None,
        gate_max=1.0,
        gate_type="global",
        residual_part=None,
        mask_self_transfer=0,
        residual_clip_eta=-1.0,
        force_cross_off=0,
        skip_dc_cross=1,
        log_asyspecx_diagnostics=0,
        eval_residual_part="default",
        gate_lr_mult=1.0,
        self_gain_init_std=1e-3,
        temporal_adapter="none",
        period=24,
        periodic_init="seasonal_naive",
        periodic_sharing="shared",
        temporal_fusion="convex",
        temporal_gate_type="global",
        temporal_gate_init_logit=-4.0,
    )
    cfg.update(overrides)
    return Namespace(**cfg)


class TestAsySpecXPhase1(unittest.TestCase):
    def test_shape_and_return_full(self):
        model = Model(make_configs(enc_in=3, cross_mode="none"))
        x = torch.randn(2, 96, 3)
        pred = model(x)
        self.assertEqual(tuple(pred.shape), (2, 96, 3))
        out = model(x, return_full=True)
        self.assertEqual(tuple(out["pred"].shape), (2, 96, 3))
        self.assertEqual(tuple(out["full"].shape), (2, 192, 3))
        self.assertEqual(tuple(out["backcast"].shape), (2, 96, 3))

    def test_f_out_clamp(self):
        model = Model(make_configs(seq_len=96, pred_len=96, cut_freq=49, cross_mode="none"))
        self.assertEqual(model.total_bins, 97)
        self.assertLessEqual(model.out_bins, 97)

    def test_cross_mode_none_has_no_cross_block(self):
        model = Model(make_configs(cross_mode="none"))
        self.assertIsNone(model.cross_block)
        y = model(torch.randn(2, 96, 5))
        self.assertEqual(tuple(y.shape), (2, 96, 5))

    def test_lift_sharing_shared_and_individual(self):
        x = torch.randn(2, 24, 4)
        shared = Model(make_configs(seq_len=24, pred_len=12, enc_in=4, lift_sharing="shared", cross_mode="none"))
        individual = Model(make_configs(seq_len=24, pred_len=12, enc_in=4, lift_sharing="individual", cross_mode="none"))
        self.assertEqual(tuple(shared(x).shape), (2, 12, 4))
        self.assertEqual(tuple(individual(x).shape), (2, 12, 4))
        self.assertIsNot(individual.freq_upsampler[0].weight_re, individual.freq_upsampler[1].weight_re)
        n_shared = sum(p.numel() for p in shared.parameters())
        n_individual = sum(p.numel() for p in individual.parameters())
        self.assertLess(n_shared, n_individual)

    def test_individual_lift_backward_with_and_without_cross(self):
        for cross_mode in ("none", "asym_lowrank"):
            model = Model(
                make_configs(
                    seq_len=24,
                    pred_len=12,
                    enc_in=4,
                    lift_sharing="individual",
                    cross_mode=cross_mode,
                    residual_part="offdiag_only",
                    gate_init_logit=-6.0,
                )
            )
            y = model(torch.randn(2, 24, 4))
            y.pow(2).mean().backward()
            self.assertIsNotNone(model.freq_upsampler[0].weight_re.grad)

    def test_norm_modes_forward(self):
        for mode in ("rin_noaffine", "revin_affine", "subtract_last", "none"):
            model = Model(make_configs(seq_len=24, pred_len=12, enc_in=4, norm_mode=mode, cross_mode="none"))
            y = model(torch.randn(2, 24, 4))
            self.assertEqual(tuple(y.shape), (2, 12, 4))
            if mode == "revin_affine":
                y.pow(2).mean().backward()
                self.assertIsNotNone(model.revin_gamma.grad)
                self.assertIsNotNone(model.revin_beta.grad)

    def test_safe_cross_forward(self):
        model = Model(
            make_configs(
                enc_in=4,
                cross_mode="asym_lowrank",
                gate_type="channel_band",
                gate_init_logit=-6.0,
                mask_self_transfer=1,
                residual_clip_eta=0.2,
            )
        )
        y = model(torch.randn(2, 96, 4))
        self.assertEqual(tuple(y.shape), (2, 96, 4))
        diag = model.get_diagnostics()
        self.assertEqual(diag["cross_active"], 1.0)

    def test_safe_cross_backward_with_backcast(self):
        model = Model(
            make_configs(
                seq_len=24,
                pred_len=12,
                enc_in=4,
                cross_mode="asym_lowrank",
                gate_type="channel_band",
                gate_init_logit=-6.0,
                mask_self_transfer=1,
                residual_clip_eta=0.2,
            )
        )
        out = model(torch.randn(2, 24, 4), return_full=True)
        loss = out["pred"].pow(2).mean() + 0.1 * out["backcast"].pow(2).mean()
        loss.backward()
        self.assertIsNotNone(model.cross_block.A.grad)

    def test_split_hier_backward(self):
        model = Model(
            make_configs(
                seq_len=24,
                pred_len=12,
                enc_in=4,
                cross_mode="asym_lowrank",
                residual_part="split",
                gate_type="hier_channel_band",
                gate_init_logit=-6.0,
            )
        )
        out = model(torch.randn(2, 24, 4), return_full=True)
        loss = out["pred"].pow(2).mean() + 0.01 * out["backcast"].pow(2).mean()
        loss.backward()
        self.assertIsNotNone(model.cross_block.gates.global_gate_logit_diag.grad)
        self.assertIsNotNone(model.cross_block.gates.global_gate_logit_cross.grad)

    def test_gate_init_logit(self):
        cross = AsymCross(
            channels=4,
            num_freqs=8,
            rank=2,
            num_bands=2,
            gate_init_logit=-6.0,
            gate_max=1.0,
            gate_type="global",
        )
        self.assertAlmostEqual(float(cross.gate_values()), 1.0 / (1.0 + math.exp(6.0)), places=6)

    def test_diagonal_mask_matches_explicit_matrix(self):
        torch.manual_seed(7)
        B, C, R, F = 2, 3, 2, 4
        U = torch.randn(B, C, F, dtype=torch.cfloat)
        A = torch.randn(C, R)
        B_src = torch.randn(C, R)
        g = torch.randn(R, dtype=torch.cfloat)

        got = AsymCross.lowrank_residual(U, A, B_src, g, mask_self_transfer=True)
        H = A.to(torch.cfloat) @ torch.diag(g) @ B_src.to(torch.cfloat).T
        H = H - torch.diag(torch.diag(H))
        expected = torch.einsum("cj,bjf->bcf", H, U)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5, rtol=1e-5))

    def test_residual_decomposition_matches_explicit_matrix(self):
        torch.manual_seed(17)
        B, C, R, F = 2, 3, 2, 5
        U = torch.randn(B, C, F, dtype=torch.cfloat)
        A = torch.randn(C, R)
        B_src = torch.randn(C, R)
        g = torch.randn(R, dtype=torch.cfloat)
        got_all, got_diag, got_off = AsymCross.decompose_lowrank_residual(U, A, B_src, g)
        H = A.to(torch.cfloat) @ torch.diag(g) @ B_src.to(torch.cfloat).T
        exp_all = torch.einsum("cj,bjf->bcf", H, U)
        H_diag = torch.diag(torch.diag(H))
        exp_diag = torch.einsum("cj,bjf->bcf", H_diag, U)
        exp_off = exp_all - exp_diag
        self.assertTrue(torch.allclose(got_all, exp_all, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(got_diag, exp_diag, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(got_off, exp_off, atol=1e-5, rtol=1e-5))

    def test_residual_part_modes_forward(self):
        x = torch.randn(2, 24, 4)
        for part in ("all", "diag_only", "offdiag_only", "split"):
            model = Model(
                make_configs(
                    seq_len=24,
                    pred_len=12,
                    enc_in=4,
                    cross_mode="asym_lowrank",
                    residual_part=part,
                    gate_type="global",
                    gate_init_logit=-6.0,
                )
            )
            y = model(x)
            self.assertEqual(tuple(y.shape), (2, 12, 4))
            self.assertEqual(model.get_diagnostics()["residual_part"], part)

    def test_self_band_gain_forward(self):
        model = Model(
            make_configs(
                seq_len=24,
                pred_len=12,
                enc_in=4,
                cross_mode="self_band_gain",
                gate_type="channel_band",
                gate_init_logit=-6.0,
            )
        )
        self.assertFalse(hasattr(model.cross_block, "A"))
        self.assertTrue(torch.is_complex(torch.complex(model.cross_block.self_gain_re, model.cross_block.self_gain_im)))
        y = model(torch.randn(2, 24, 4))
        self.assertEqual(tuple(y.shape), (2, 12, 4))

    def test_hier_channel_band_gate(self):
        model = Model(
            make_configs(
                enc_in=4,
                cross_mode="asym_lowrank",
                residual_part="all",
                gate_type="hier_channel_band",
                gate_init_logit=-6.0,
            )
        )
        gate = model.cross_block.gate_values()
        self.assertEqual(tuple(gate.shape), (4, 4))
        self.assertTrue(torch.allclose(model.cross_block.gates.local_scale(), torch.ones_like(gate), atol=1e-6))
        self.assertAlmostEqual(float(gate.mean()), 1.0 / (1.0 + math.exp(6.0)), places=6)

        split_model = Model(
            make_configs(
                enc_in=4,
                cross_mode="asym_lowrank",
                residual_part="split",
                gate_type="hier_channel_band",
                gate_init_logit=-6.0,
            )
        )
        self.assertEqual(tuple(split_model.cross_block.gate_values("diag").shape), (4, 4))
        self.assertEqual(tuple(split_model.cross_block.gate_values("cross").shape), (4, 4))

    def test_eval_residual_part_forward(self):
        model = Model(
            make_configs(
                seq_len=24,
                pred_len=12,
                enc_in=4,
                cross_mode="asym_lowrank",
                residual_part="all",
                gate_init_logit=-6.0,
            )
        )
        x = torch.randn(2, 24, 4)
        for part in ("none", "diag_only", "offdiag_only", "all"):
            y = model(x, eval_residual_part=part)
            self.assertEqual(tuple(y.shape), (2, 12, 4))
        self.assertEqual(model(x, eval_residual_part="none").shape, model(x).shape)

    def test_gate_lr_mult_param_groups(self):
        dummy = object.__new__(Exp_Main)
        dummy.args = Namespace(model="AsySpecX", learning_rate=0.001, gate_lr_mult=5.0)
        dummy.model = Model(make_configs(cross_mode="asym_lowrank", residual_part="split", gate_type="global"))
        optim = Exp_Main._select_optimizer(dummy)
        lrs = sorted(group["lr"] for group in optim.param_groups)
        self.assertEqual(lrs, [0.001, 0.005])

    def test_sparse_period_mask_and_init(self):
        adapter = SparsePeriodAdapter(8, 4, 3, period=4, temporal_gate_init_logit=-99.0)
        mask = adapter.periodic_mask[0]  # multi-period: [P, H, T] -> single plane
        for h in range(4):
            for t in range(8):
                self.assertEqual(float(mask[h, t]), 1.0 if t % 4 == (8 + h) % 4 else 0.0)
            expected_t = h + 4
            self.assertEqual(float(adapter.W_raw[0, h, expected_t]), 1.0)

    def test_sparse_period_forward_and_fusion_extremes(self):
        x = torch.randn(2, 8, 3)
        spec = torch.randn(2, 4, 3)
        adapter = SparsePeriodAdapter(8, 4, 3, period=4, temporal_gate_init_logit=-99.0)
        fused0, period_pred = adapter(x, spec)
        self.assertEqual(tuple(period_pred.shape), (2, 4, 3))
        self.assertTrue(torch.allclose(fused0, spec, atol=1e-6))
        adapter.temporal_gate_logit.data.fill_(99.0)
        fused1, period_pred = adapter(x, spec)
        self.assertTrue(torch.allclose(fused1, period_pred, atol=1e-5))

    def test_sparse_period_channel_gate_and_model_forward(self):
        model = Model(
            make_configs(
                seq_len=24,
                pred_len=12,
                enc_in=4,
                cross_mode="none",
                temporal_adapter="sparse_period",
                period=6,
                temporal_gate_type="channel",
                temporal_gate_init_logit=-4.0,
            )
        )
        y = model(torch.randn(2, 24, 4))
        self.assertEqual(tuple(y.shape), (2, 12, 4))
        diag = model.get_diagnostics()
        self.assertEqual(diag["temporal_adapter_enabled"], 1.0)
        self.assertIn("periodic_mask_density", diag)
        self.assertFalse(torch.isnan(y).any())

    def test_residual_clip(self):
        torch.manual_seed(11)
        U = torch.randn(3, 4, 5, dtype=torch.cfloat)
        R = 20.0 * torch.randn(3, 4, 5, dtype=torch.cfloat)
        eta = 0.2
        clipped = AsymCross.clip_residual(R, U, eta)
        u_rms = torch.sqrt(torch.mean(torch.abs(U) ** 2, dim=-1) + 1e-8)
        r_rms = torch.sqrt(torch.mean(torch.abs(clipped) ** 2, dim=-1) + 1e-8)
        self.assertTrue(torch.all(r_rms <= eta * u_rms + 2e-5))

    def test_backcast_loss_formula(self):
        criterion = nn.MSELoss()
        pred = torch.tensor([[[1.0], [2.0]]])
        true = torch.tensor([[[2.0], [0.0]]])
        backcast = torch.tensor([[[1.0], [1.5], [2.0]]])
        x = torch.tensor([[[1.0], [1.0], [1.0]]])
        forecast = criterion(pred, true)
        self.assertEqual(float(forecast), float(criterion(pred, true)))
        weight = 0.1
        total = forecast + weight * criterion(backcast, x)
        self.assertGreater(float(total), float(forecast))


if __name__ == "__main__":
    unittest.main()
