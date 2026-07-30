import unittest

from scripts.summarize_phase11 import build_ranking


class Phase11SummaryTests(unittest.TestCase):
    @staticmethod
    def row(arm, val, params, train=100.0, gate=None):
        return {
            "arm": arm,
            "status": "ok",
            "val_mse": val,
            "n_param": params,
            "t_train": train,
            "forecast_kernel_gate_mean": gate,
        }

    def test_promotes_by_validation_and_constraints_only(self):
        rows = [
            self.row("anchor", 0.350, 68000),
            self.row("dense_direct", 0.340, 138000),
            self.row("fk_good", 0.341, 76000, 105.0, 0.04),
            self.row("fk_too_slow", 0.339, 76000, 120.0, 0.04),
            self.row("fk_collapsed", 0.341, 76000, 100.0, 0.0001),
        ]
        ranking, promoted = build_ranking(rows)
        self.assertEqual([row["arm"] for row in promoted], ["fk_good"])
        by_arm = {row["arm"]: row for row in ranking}
        self.assertEqual(by_arm["fk_too_slow"]["resource_ok"], 0)
        self.assertEqual(by_arm["fk_collapsed"]["gate_ok"], 0)

    def test_collapsed_gate_can_advance_only_on_strict_validation_win(self):
        rows = [
            self.row("anchor", 0.350, 68000),
            self.row("dense_direct", 0.340, 138000),
            self.row("fk_closed_win", 0.339, 76000, 100.0, 0.0001),
        ]
        _, promoted = build_ranking(rows)
        self.assertEqual([row["arm"] for row in promoted], ["fk_closed_win"])

    def test_same_gpu_fixed_work_overrides_cross_node_wall_time(self):
        rows = [
            self.row("anchor", 0.350, 68000),
            self.row("dense_direct", 0.340, 138000),
            {
                **self.row("fk_fast_same_gpu", 0.339, 76000, 130.0, 0.04),
                "fixed_work_train_ratio_vs_dense": 0.95,
                "fixed_work_inference_ratio_vs_dense": 0.96,
            },
        ]
        _, promoted = build_ranking(rows)
        self.assertEqual([row["arm"] for row in promoted], ["fk_fast_same_gpu"])


if __name__ == "__main__":
    unittest.main()
