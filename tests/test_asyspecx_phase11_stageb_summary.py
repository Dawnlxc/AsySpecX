import unittest

from scripts.summarize_phase11_stageb import build_ranking


def row(arm, val, params=76000, **overrides):
    values = {
        "arm": arm,
        "status": "ok",
        "val_mse": val,
        "n_param": params,
        "t_train": 100.0,
        "epochs_ran": 10,
        "forecast_kernel_gate_mean": 0.04,
        "forecast_kernel_sm_gate_abs_max": 0.2,
        "forecast_kernel_sm_factor_min": 0.8,
        "forecast_kernel_sm_factor_max": 1.2,
        "fixed_train_ratio_vs_stage_a": 1.03,
        "fixed_inference_ratio_vs_stage_a": 1.04,
    }
    values.update(overrides)
    return values


class StageBSelectorTests(unittest.TestCase):
    def test_only_strict_real_sm_validation_win_is_promoted(self):
        rows = [
            row("dense_direct", 0.686, params=138000),
            row("fk_r8_cs", 0.6853),
            row("fk_sm2_shared", 0.6852),
            row("fk_sm2_mode", 0.6854),
        ]
        ranking, promoted = build_ranking(rows)
        self.assertEqual([item["arm"] for item in promoted], ["fk_sm2_shared"])
        by_arm = {item["arm"]: item for item in ranking}
        self.assertEqual(by_arm["fk_sm2_shared"]["eligible"], 1)
        self.assertEqual(by_arm["fk_sm2_mode"]["eligible"], 0)

    def test_resource_or_collapsed_sm_gate_blocks_promotion(self):
        rows = [
            row("dense_direct", 0.686, params=138000),
            row("fk_r8_cs", 0.6853),
            row("fk_sm2_shared", 0.6851, fixed_train_ratio_vs_stage_a=1.11),
            row("fk_sm2_mode", 0.6850, forecast_kernel_sm_gate_abs_max=0.0),
        ]
        _, promoted = build_ranking(rows)
        self.assertEqual(promoted, [])


if __name__ == "__main__":
    unittest.main()
