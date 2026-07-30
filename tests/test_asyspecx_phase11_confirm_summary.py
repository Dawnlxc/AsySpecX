import unittest

from scripts.summarize_phase11_confirm import summarize


class Phase11ConfirmSummaryTests(unittest.TestCase):
    @staticmethod
    def row(arm, seed, val, mse):
        return {
            "arm": arm, "seed": seed, "status": "ok", "test_deferred": False,
            "val_mse": val, "mse": mse, "mae": mse + 0.1, "n_param": 10,
            "peak_cuda_mb": 1.0, "t_train": 2.0, "t_inf": 3.0,
            "forecast_kernel_gate_mean": None if arm == "dense_direct" else 0.04,
        }

    def test_stable_win_requires_two_test_wins_and_mean_val_gain(self):
        rows = []
        for seed, dense_mse in zip((2024, 2025, 2026), (0.34, 0.35, 0.36)):
            rows.append(self.row("dense_direct", seed, 0.40, dense_mse))
        for seed, mse in zip((2024, 2025, 2026), (0.33, 0.34, 0.37)):
            rows.append(self.row("fk_r8_cs", seed, 0.39, mse))
        for seed, mse in zip((2024, 2025, 2026), (0.33, 0.36, 0.37)):
            rows.append(self.row("fk_r8", seed, 0.41, mse))
        _, summary = summarize(rows)
        by_arm = {row["arm"]: row for row in summary}
        self.assertEqual(by_arm["fk_r8_cs"]["stable_win"], 1)
        self.assertEqual(by_arm["fk_r8"]["stable_win"], 0)


if __name__ == "__main__":
    unittest.main()
