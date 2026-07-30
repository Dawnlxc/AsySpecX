import copy
import unittest
from pathlib import Path

from scripts.summarize_phase11_staged import read_manifest
from scripts.summarize_phase11_staged_early_stop import audit_partial


ROOT = Path(__file__).resolve().parents[1]


def make_row(spec, tail_delta=0.0):
    base = 0.5 + 0.0001 * spec["pred_len"]
    value = base
    if spec["arm"] == "fk_sm2_mode":
        value = base + 0.0002
    if spec["arm"] == "fk_sm2_tail2":
        value = base + tail_delta
    scale = max(0.0, 1.0 - 2.0 * spec["seq_len"] / spec["pred_len"])
    return {
        **spec,
        "status": "ok",
        "test_deferred": True,
        "val_mse": value,
        "mse": None,
        "mae": None,
        "forecast_kernel_extension_scale": scale if spec["arm"] == "fk_sm2_tail2" else 1.0,
        "forecast_kernel_sm_factor_min": 1.0 if scale == 0.0 else 0.8,
        "forecast_kernel_sm_factor_max": 1.0 if scale == 0.0 else 1.2,
        "forecast_kernel_sm_gate_abs_max": 0.0 if scale == 0.0 else 0.2,
    }


def resource(train_ratio=1.05):
    return {
        ("ETTm1", 96, 720): {
            "fk_r8_cs": {
                "train_forward_backward_ms_per_batch": 10.0,
                "inference_ms_per_batch": 5.0,
                "fixed_work_peak_cuda_mb": 100.0,
                "n_param": 1000,
            },
            "fk_sm2_tail2": {
                "train_forward_backward_ms_per_batch": 10.0 * train_ratio,
                "inference_ms_per_batch": 5.1,
                "fixed_work_peak_cuda_mb": 100.5,
                "n_param": 1005,
            },
        }
    }


class StageDEarlyStopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = read_manifest(ROOT / "configs" / "phase11_staged_screen.tsv")

    def ettm1_rows(self, active_tail_delta=0.001):
        rows = []
        for spec in self.specs:
            if spec["dataset"] != "ETTm1":
                continue
            delta = active_tail_delta if spec["role"] == "active" else 0.0
            rows.append(make_row(spec, delta))
        return rows

    def test_two_active_losses_make_three_of_four_wins_impossible(self):
        cells, _, aggregate, decision = audit_partial(
            self.ettm1_rows(), self.specs, resource(1.05)
        )
        self.assertEqual(len(cells), 4)
        self.assertEqual(aggregate["completed_identity_ok"], 1)
        self.assertEqual(aggregate["maximum_possible_active_wins"], 2)
        self.assertEqual(aggregate["win_gate_impossible"], 1)
        self.assertEqual(decision["advance_to_confirmation"], 0)
        self.assertEqual(decision["decisive_early_stop"], 1)

    def test_single_observed_resource_failure_is_decisive(self):
        rows = []
        for spec in self.specs:
            if spec["dataset"] == "ETTm1" and spec["pred_len"] == 720:
                rows.append(make_row(spec, -0.001))
        _, resource_rows, aggregate, decision = audit_partial(
            rows, self.specs, resource(1.25)
        )
        self.assertEqual(resource_rows[0]["passes_active_limits"], 0)
        self.assertEqual(aggregate["observed_resource_failure"], 1)
        self.assertEqual(decision["decisive_early_stop"], 1)

    def test_partial_nondecisive_result_does_not_claim_failure(self):
        rows = []
        for spec in self.specs:
            if spec["dataset"] == "ETTm1" and spec["pred_len"] == 336:
                rows.append(make_row(spec, -0.001))
        _, _, aggregate, decision = audit_partial(rows, self.specs, {})
        self.assertEqual(aggregate["win_gate_impossible"], 0)
        self.assertEqual(decision["decisive_early_stop"], 0)

    def test_test_open_partial_row_is_rejected(self):
        rows = self.ettm1_rows()
        rows[0] = copy.deepcopy(rows[0])
        rows[0]["test_deferred"] = False
        rows[0]["mse"] = 0.4
        with self.assertRaisesRegex(ValueError, "test-open"):
            audit_partial(rows, self.specs, {})


if __name__ == "__main__":
    unittest.main()
