import unittest

from scripts.summarize_phase11_stageb_confirm import summarize


def row(arm, seed, val, mse):
    return {
        "arm": arm,
        "seed": seed,
        "status": "ok",
        "test_deferred": False,
        "val_mse": val,
        "mse": mse,
        "mae": mse + 0.1,
        "n_param": 10,
        "peak_cuda_mb": 1.0,
        "t_train": 2.0,
        "t_inf": 3.0,
        "forecast_kernel_gate_mean": 0.04,
        "forecast_kernel_sm_gate_abs_max": None if arm == "fk_r8_cs" else 0.2,
    }


class StageBConfirmTests(unittest.TestCase):
    def test_stable_win_requires_test_majority_and_mean_validation_gain(self):
        rows = []
        for seed, mse in zip((2024, 2025, 2026), (0.34, 0.35, 0.36)):
            rows.append(row("fk_r8_cs", seed, 0.40, mse))
        for seed, mse in zip((2024, 2025, 2026), (0.33, 0.34, 0.37)):
            rows.append(row("fk_sm4_mode", seed, 0.39, mse))
        _, summaries = summarize(rows, "fk_sm4_mode")
        by_arm = {item["arm"]: item for item in summaries}
        self.assertEqual(by_arm["fk_sm4_mode"]["test_wins_vs_stage_a"], 2)
        self.assertEqual(by_arm["fk_sm4_mode"]["stable_win"], 1)

    def test_better_test_without_validation_gain_is_not_stable(self):
        rows = []
        for seed, mse in zip((2024, 2025, 2026), (0.34, 0.35, 0.36)):
            rows.append(row("fk_r8_cs", seed, 0.40, mse))
            rows.append(row("fk_sm2_mode", seed, 0.41, mse - 0.01))
        _, summaries = summarize(rows, "fk_sm2_mode")
        candidate = next(item for item in summaries if item["arm"] == "fk_sm2_mode")
        self.assertEqual(candidate["stable_win"], 0)


if __name__ == "__main__":
    unittest.main()
