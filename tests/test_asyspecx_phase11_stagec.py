import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.benchmark_phase11_fixed_work import make_config
from scripts.summarize_phase11_stagec import read_manifest, summarize


ROOT = Path(__file__).resolve().parents[1]
CELLS = (
    ("ind_cycle_full", "weather", 96, 96, 2026, 13),
    ("ind_cycle_full", "weather", 96, 192, 2026, 13),
    ("ind_cycle_full", "weather", 96, 336, 2026, 13),
    ("compact_period_cycle_full", "electricity", 504, 96, 2026, 127),
    ("compact_period_cycle_full", "electricity", 504, 336, 2026, 127),
)
ARMS = ("anchor", "fk_r8_cs", "fk_sm2_mode", "fk_sm4_ph4_h")


def make_row(cell, arm, val):
    profile, dataset, seq_len, pred_len, seed, cut_freq = cell
    row = {
        "base_profile": profile,
        "arm": arm,
        "dataset": dataset,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "seed": seed,
        "cut_freq": cut_freq,
        "status": "ok",
        "test_deferred": True,
        "mse": None,
        "mae": None,
        "val_mse": val,
        "n_param": 1000 if arm == "anchor" else 1100,
        "forecast_kernel_gate_mean": 0.04 if arm != "anchor" else None,
        "forecast_kernel_sm_gate_abs_max": 0.8 if arm.startswith("fk_sm") else None,
        "forecast_kernel_sm_factor_min": 0.5 if arm.startswith("fk_sm") else None,
        "forecast_kernel_sm_factor_max": 2.0 if arm.startswith("fk_sm") else None,
        "forecast_kernel_phase_abs_max": 0.4 if arm == "fk_sm4_ph4_h" else None,
        "forecast_phase_max": 1.5707963267948966 if arm == "fk_sm4_ph4_h" else 0.0,
    }
    return row


def make_case(phase_values=None):
    rows = []
    specs = []
    phase_values = phase_values or [0.1985] * len(CELLS)
    for index, cell in enumerate(CELLS):
        values = {
            "anchor": 0.201,
            "fk_r8_cs": 0.200,
            "fk_sm2_mode": 0.199,
            "fk_sm4_ph4_h": phase_values[index],
        }
        for arm in ARMS:
            row = make_row(cell, arm, values[arm])
            rows.append(row)
            specs.append({key: row[key] for key in (
                "base_profile", "arm", "dataset", "seq_len", "pred_len",
                "seed", "cut_freq",
            )})
    return rows, specs


def make_resources(phase_train=10.8):
    resources = {}
    for profile in {cell[0] for cell in CELLS}:
        for arm, train, infer, memory, params in (
            ("fk_r8_cs", 10.0, 5.0, 100.0, 1100),
            ("fk_sm2_mode", 10.5, 5.2, 100.5, 1110),
            ("fk_sm4_ph4_h", phase_train, 5.3, 100.6, 1120),
        ):
            resources[(profile, arm)] = {
                "arm": arm,
                "train_forward_backward_ms_per_batch": train,
                "inference_ms_per_batch": infer,
                "fixed_work_peak_cuda_mb": memory,
                "n_param": params,
            }
    return resources, {cell[0] for cell in CELLS}


class StageCSelectorTests(unittest.TestCase):
    def test_complex_selected_when_both_pass_and_phase_wins_aggregate(self):
        rows, specs = make_case()
        resources, profiles = make_resources()
        _, aggregates, decision = summarize(rows, specs, resources, profiles)
        self.assertEqual(decision["selected_arm"], "fk_sm4_ph4_h")
        self.assertEqual(decision["advance_to_wave2"], 1)
        self.assertTrue(all(row["eligible"] for row in aggregates))

    def test_real_sm_selected_when_complex_loses_head_to_head(self):
        rows, specs = make_case([0.1985, 0.1985, 0.1995, 0.1995, 0.1995])
        resources, profiles = make_resources()
        _, _, decision = summarize(rows, specs, resources, profiles)
        self.assertEqual(decision["selected_arm"], "fk_sm2_mode")
        self.assertEqual(decision["complex_head_to_head_ok"], 0)

    def test_resource_gate_can_reject_only_complex(self):
        rows, specs = make_case()
        resources, profiles = make_resources(phase_train=12.0)
        _, aggregates, decision = summarize(rows, specs, resources, profiles)
        phase = next(row for row in aggregates if row["arm"] == "fk_sm4_ph4_h")
        self.assertEqual(phase["eligible"], 0)
        self.assertEqual(decision["selected_arm"], "fk_sm2_mode")

    def test_selector_rejects_any_open_test_metric(self):
        rows, specs = make_case()
        resources, profiles = make_resources()
        rows = copy.deepcopy(rows)
        rows[0]["mse"] = 0.1
        with self.assertRaisesRegex(ValueError, "test-open"):
            summarize(rows, specs, resources, profiles)

    def test_locked_manifests_have_expected_sizes(self):
        canary = read_manifest(ROOT / "configs/phase11_stagec_wave1_canary.tsv")
        screen = read_manifest(ROOT / "configs/phase11_stagec_wave1_screen.tsv")
        self.assertEqual(len(canary), 8)
        self.assertEqual(len(screen), 20)


class StageCBenchmarkConfigTests(unittest.TestCase):
    @staticmethod
    def args(profile):
        return SimpleNamespace(
            base_profile=profile,
            data_name="custom",
            data_path="electricity.csv" if "compact" in profile else "weather.csv",
            seq_len=504 if "compact" in profile else 96,
            pred_len=336,
            channels=321 if "compact" in profile else 21,
            cycle=168 if "compact" in profile else 144,
            cut_freq=127 if "compact" in profile else 13,
            cross_rank=2 if "compact" in profile else 8,
            num_bands=8,
            periods="24+168" if "compact" in profile else "144",
        )

    def test_weather_profile_matches_stage_b_backbone(self):
        config = make_config("fk_sm2_mode", self.args("ind_cycle_full"))
        self.assertEqual(config.lift_sharing, "individual")
        self.assertEqual(config.cross_mode, "none")
        self.assertEqual(config.temporal_adapter, "none")
        self.assertEqual(config.cycle_residual, 1)

    def test_electricity_profile_matches_compact_period_backbone(self):
        config = make_config(
            "fk_sm4_ph4_h", self.args("compact_period_cycle_full")
        )
        self.assertEqual(config.lift_sharing, "shared")
        self.assertEqual(config.cross_mode, "asym_lowrank")
        self.assertEqual(config.temporal_adapter, "compact_period")
        self.assertEqual(config.periods, "24+168")
        self.assertEqual(config.forecast_kernel_phase_basis_dim, 4)


if __name__ == "__main__":
    unittest.main()

