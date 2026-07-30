import unittest

from scripts.summarize_phase11_complex_confirm import summarize


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
        "forecast_kernel_phase_abs_max": None if arm == "fk_sm4_mode" else 0.2,
    }


class ComplexConfirmTests(unittest.TestCase):
    def test_stable_requires_paired_majority_and_validation_gain(self):
        rows = []
        for seed, mse in zip((2024, 2025, 2026), (0.34, 0.35, 0.36)):
            rows.append(row("fk_sm4_mode", seed, 0.40, mse))
        for seed, mse in zip((2024, 2025, 2026), (0.33, 0.34, 0.37)):
            rows.append(row("fk_sm4_ph4_h", seed, 0.39, mse))
        _, summaries = summarize(rows, "fk_sm4_ph4_h")
        candidate = next(item for item in summaries if item["arm"] == "fk_sm4_ph4_h")
        self.assertEqual(candidate["test_wins_vs_real_sm"], 2)
        self.assertEqual(candidate["stable_win"], 1)

    def test_test_wins_without_validation_gain_fail(self):
        rows = []
        for seed, mse in zip((2024, 2025, 2026), (0.34, 0.35, 0.36)):
            rows.append(row("fk_sm4_mode", seed, 0.40, mse))
            rows.append(row("fk_sm4_ph2_q", seed, 0.41, mse - 0.01))
        _, summaries = summarize(rows, "fk_sm4_ph2_q")
        candidate = next(item for item in summaries if item["arm"] == "fk_sm4_ph2_q")
        self.assertEqual(candidate["stable_win"], 0)


if __name__ == "__main__":
    unittest.main()
