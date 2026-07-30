import unittest

from scripts.summarize_phase11_complex import build_ranking


def row(arm, val, params=76000, **overrides):
    values = {
        "arm": arm,
        "status": "ok",
        "val_mse": val,
        "n_param": params,
        "forecast_kernel_gate_mean": 0.04,
        "forecast_kernel_phase_abs_max": 0.2,
        "forecast_phase_max": 0.8,
        "fixed_train_ratio_vs_real_sm": 1.05,
        "fixed_inference_ratio_vs_real_sm": 1.04,
    }
    values.update(overrides)
    return values


class ComplexSelectorTests(unittest.TestCase):
    def test_only_strict_bounded_phase_win_is_promoted(self):
        rows = [
            row("dense_direct", 0.686, params=138000),
            row("fk_sm4_mode", 0.6852),
            row("fk_sm4_ph2_q", 0.6851),
            row("fk_sm4_ph4_q", 0.6853),
        ]
        _, promoted = build_ranking(rows)
        self.assertEqual([item["arm"] for item in promoted], ["fk_sm4_ph2_q"])

    def test_resource_or_zero_phase_blocks_promotion(self):
        rows = [
            row("dense_direct", 0.686, params=138000),
            row("fk_sm4_mode", 0.6852),
            row("fk_sm4_ph2_q", 0.6850, fixed_inference_ratio_vs_real_sm=1.11),
            row("fk_sm4_ph4_q", 0.6849, forecast_kernel_phase_abs_max=0.0),
        ]
        _, promoted = build_ranking(rows)
        self.assertEqual(promoted, [])


if __name__ == "__main__":
    unittest.main()
