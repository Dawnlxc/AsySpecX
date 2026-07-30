import copy
import unittest
from pathlib import Path

from scripts.summarize_phase11_staged_confirm import (
    ARMS,
    CUT_FREQ,
    DATASETS,
    HORIZONS,
    SEEDS,
    STAGE_A,
    TAIL2,
    summarize_final,
    summarize_validation,
)


ROOT = Path(__file__).resolve().parents[1]


def validation_row(arm, dataset, pred_len, seed, value):
    return {
        "arm": arm,
        "dataset": dataset,
        "seq_len": 96,
        "pred_len": pred_len,
        "seed": seed,
        "cut_freq": CUT_FREQ[dataset],
        "status": "ok",
        "test_deferred": True,
        "val_mse": value,
        "mse": None,
        "mae": None,
        "summary_file": f"{arm}-{dataset}-{pred_len}-{seed}.json",
    }


def validation_fixture(tail_delta=-0.01):
    screen = []
    confirm = []
    for dataset in DATASETS:
        base = 0.4 if dataset == "ETTm1" else 0.8
        for pred_len in HORIZONS:
            for seed in SEEDS:
                for arm in ARMS:
                    value = base + 0.0001 * pred_len + 0.00001 * (seed - 2024)
                    if arm == TAIL2:
                        value += tail_delta
                    row = validation_row(arm, dataset, pred_len, seed, value)
                    (screen if seed == 2026 else confirm).append(row)
    return screen, confirm


def test_row(arm, dataset, pred_len, seed, mse):
    return {
        "arm": arm,
        "dataset": dataset,
        "seq_len": 96,
        "pred_len": pred_len,
        "seed": seed,
        "cut_freq": CUT_FREQ[dataset],
        "status": "ok",
        "mse": mse,
        "mae": mse * 0.8,
        "test_opened_after_validation_selection": True,
        "summary_file": f"test-{arm}-{dataset}-{pred_len}-{seed}.json",
    }


class StageDConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.screen_decision = {
            "advance_to_confirmation": 1,
            "selected_arm": TAIL2,
            "test_metrics_used": False,
        }
        self.validation_decision = {
            "open_test": 1,
            "selected_arm": TAIL2,
            "test_metrics_used": False,
        }

    def test_confirmation_manifest_is_frozen_and_validation_only(self):
        path = ROOT / "configs" / "phase11_staged_confirm.tsv"
        rows = [
            line.split("\t")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        self.assertEqual(len(rows), 16)
        self.assertEqual({row[1] for row in rows}, {"ind_cycle_full"})
        self.assertEqual({row[2] for row in rows}, set(ARMS))
        self.assertEqual({int(row[6]) for row in rows}, {2024, 2025})

    def test_validation_gate_opens_test_only_on_lower_macro_mean(self):
        screen, confirm = validation_fixture(tail_delta=-0.01)
        pairs, aggregate, decision = summarize_validation(
            screen, confirm, self.screen_decision
        )
        self.assertEqual(len(pairs), 12)
        self.assertEqual(aggregate["open_test"], 1)
        self.assertEqual(decision["open_test"], 1)
        self.assertFalse(decision["test_metrics_used"])

    def test_validation_gate_rejects_test_open_training_row(self):
        screen, confirm = validation_fixture(tail_delta=-0.01)
        confirm[0]["test_deferred"] = False
        confirm[0]["mse"] = 0.3
        with self.assertRaisesRegex(ValueError, "test-open"):
            summarize_validation(screen, confirm, self.screen_decision)

    def test_validation_gate_stays_closed_when_tail_mean_is_not_lower(self):
        screen, confirm = validation_fixture(tail_delta=0.001)
        _, aggregate, decision = summarize_validation(
            screen, confirm, self.screen_decision
        )
        self.assertEqual(aggregate["open_test"], 0)
        self.assertEqual(decision["open_test"], 0)

    def test_final_gate_accepts_eight_or_more_wins_and_both_datasets(self):
        rows = []
        for dataset in DATASETS:
            base = 0.4 if dataset == "ETTm1" else 0.8
            for pred_len in HORIZONS:
                for seed in SEEDS:
                    rows.append(test_row(STAGE_A, dataset, pred_len, seed, base))
                    rows.append(test_row(TAIL2, dataset, pred_len, seed, base * 0.99))
        pairs, aggregate, decision = summarize_final(rows, self.validation_decision)
        self.assertEqual(len(pairs), 12)
        self.assertEqual(aggregate["tail2_test_wins"], 12)
        self.assertEqual(aggregate["stable_generalization"], 1)
        self.assertEqual(decision["selected_arm"], TAIL2)

    def test_final_gate_rejects_dataset_specific_result(self):
        rows = []
        for dataset in DATASETS:
            base = 0.4 if dataset == "ETTm1" else 0.8
            ratio = 0.97 if dataset == "ETTm1" else 1.01
            for pred_len in HORIZONS:
                for seed in SEEDS:
                    rows.append(test_row(STAGE_A, dataset, pred_len, seed, base))
                    rows.append(test_row(TAIL2, dataset, pred_len, seed, base * ratio))
        _, aggregate, decision = summarize_final(rows, self.validation_decision)
        self.assertLess(aggregate["macro_mean_relative_test_mse_delta_pct"], 0.0)
        self.assertGreater(aggregate["traffic_mean_relative_test_mse_delta_pct"], 0.0)
        self.assertEqual(aggregate["stable_generalization"], 0)
        self.assertEqual(decision["selected_arm"], STAGE_A)

    def test_final_gate_requires_validation_authorization(self):
        decision = copy.deepcopy(self.validation_decision)
        decision["open_test"] = 0
        with self.assertRaisesRegex(ValueError, "did not authorize"):
            summarize_final([], decision)

    def test_eval_runner_has_tail2_schedule(self):
        text = (ROOT / "scripts" / "slurm" / "asyspecx_phase11_eval.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("fk_sm2_tail2)", text)
        self.assertIn("--forecast_kernel_extension_shrink tail2_linear", text)


if __name__ == "__main__":
    unittest.main()
