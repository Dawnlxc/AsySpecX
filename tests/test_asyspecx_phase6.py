import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run(args, **kw):
    return subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True, **kw)


def run_selector(args):
    return run(["scripts/select_by_validation.py", *args])


# ---------- B. segment_mean_plus_std ----------
class TestSegmentRobust(unittest.TestCase):
    def _rows(self):
        # arm A: slightly better mean segment but huge segment spread.
        # arm B: slightly worse mean but stable segments.
        return [
            dict(arm="A", dataset="weather", seq_len=720, pred_len=96, seed=1, status="ok",
                 val_mse=0.30, mse=0.5, mae=0.5, val_mse_seg0=0.05, val_mse_seg1=0.05, val_mse_seg2=0.05, val_mse_seg3=0.95),
            dict(arm="B", dataset="weather", seq_len=720, pred_len=96, seed=1, status="ok",
                 val_mse=0.32, mse=0.1, mae=0.1, val_mse_seg0=0.30, val_mse_seg1=0.31, val_mse_seg2=0.30, val_mse_seg3=0.31),
        ]

    def test_prefers_stable_arm(self):
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "r.csv"); write_csv(c, self._rows())
            out = os.path.join(d, "o.csv")
            r = run_selector(["--csv", c, "--metric_mode", "segment_mean_plus_std", "--std_weight", "1.0",
                              "--output", out, "--summary", os.path.join(d, "s.md")])
            self.assertEqual(r.returncode, 0, r.stderr)
            sel = list(csv.DictReader(open(out)))
            # A pool mean=0.275 std large; B mean~0.305 std~0.005. score_A=0.275+~0.39=0.66; score_B~0.31 -> B
            self.assertEqual({x["selected_arm"] for x in sel}, {"B"})

    def test_missing_segments_errors(self):
        rows = [dict(arm="A", dataset="w", seq_len=720, pred_len=96, seed=1, status="ok", val_mse=0.2, mse=0.1, mae=0.1)]
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "r.csv"); write_csv(c, rows)
            r = run_selector(["--csv", c, "--metric_mode", "segment_mean_plus_std",
                              "--output", os.path.join(d, "o.csv"), "--summary", os.path.join(d, "s.md")])
            self.assertNotEqual(r.returncode, 0)


# ---------- C. margin trace in summary ----------
class TestMarginTrace(unittest.TestCase):
    def test_summary_has_trace(self):
        rows = [
            dict(arm="A", dataset="w", seq_len=720, pred_len=96, seed=1, status="ok", val_mse=0.200, mse=0.5, mae=0.5),
            dict(arm="B", dataset="w", seq_len=720, pred_len=96, seed=1, status="ok", val_mse=0.201, mse=0.1, mae=0.1),
        ]
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "r.csv"); write_csv(c, rows)
            summ = os.path.join(d, "s.md")
            r = run_selector(["--csv", c, "--selection_margin_pct", "0.02", "--prefer_arm_order", "B,A",
                              "--output", os.path.join(d, "o.csv"), "--summary", summ])
            self.assertEqual(r.returncode, 0, r.stderr)
            txt = open(summ).read()
            self.assertIn("Margin / Prefer-Order Trace", txt)
            self.assertIn("raw_best_arm", txt)
            self.assertIn("near_best_arms", txt)
            self.assertIn("final_selected_arm", txt)


# ---------- A. selector audit ----------
class TestSelectorAudit(unittest.TestCase):
    def _results(self):
        rows = []
        for ds in ["weather", "electricity"]:
            for pl in [96, 720]:
                for seed in [1, 2]:
                    for arm, mse in [("phase6_asx_individual", 0.20 if ds == "weather" else 0.30),
                                     ("phase6_asx_period_multi", 0.25 if ds == "weather" else 0.15)]:
                        rows.append(dict(arm=arm, dataset=ds, seq_len=720, pred_len=pl, seed=seed,
                                         status="ok", val_mse=mse, mse=mse, mae=mse + 0.1))
        return rows

    def test_audit_multi_and_oracle(self):
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "results.csv"); write_csv(c, self._results())
            # two selected csvs
            sel1 = [dict(dataset="weather", seq_len=720, pred_len=96, seed=1, selected_arm="phase6_asx_individual",
                         val_score=0.2, val_mse=0.2, test_mse=0.20, test_mae=0.30)]
            sel2 = [dict(dataset="weather", seq_len=720, pred_len=96, seed=1, selected_arm="phase6_asx_period_multi",
                         val_score=0.25, val_mse=0.25, test_mse=0.25, test_mae=0.35)]
            write_csv(os.path.join(d, "selected_a.csv"), sel1)
            write_csv(os.path.join(d, "selected_b.csv"), sel2)
            r = run(["scripts/audit_phase5_selectors.py", "--csv", c,
                     "--selected_files", "selected_a.csv,selected_b.csv,selected_missing.csv",
                     "--output_dir", d])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.isfile(os.path.join(d, "selector_audit.md")))
            self.assertTrue(os.path.isfile(os.path.join(d, "selector_audit.csv")))
            self.assertTrue(os.path.isfile(os.path.join(d, "selector_group_details.csv")))
            md = open(os.path.join(d, "selector_audit.md")).read()
            self.assertIn("Oracle is analysis only", md)
            self.assertIn("best_fixed_single_arm", md)
            # missing file warned, not crashed
            self.assertIn("not found", r.stdout + r.stderr)

    def test_oracle_not_reported_as_selected(self):
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "results.csv"); write_csv(c, self._results())
            sel = [dict(dataset="weather", seq_len=720, pred_len=96, seed=1, selected_arm="phase6_asx_individual",
                        val_score=0.2, val_mse=0.2, test_mse=0.20, test_mae=0.30)]
            write_csv(os.path.join(d, "selected_a.csv"), sel)
            run(["scripts/audit_phase5_selectors.py", "--csv", c,
                 "--selected_files", "selected_a.csv", "--output_dir", d])
            md = open(os.path.join(d, "selector_audit.md")).read()
            self.assertIn("must not be reported as a valid selected model", md)


# ---------- D/E. fullfield runner dry-run ----------
class TestFullfieldDryRun(unittest.TestCase):
    def _run_dry(self, env_extra):
        env = dict(os.environ)
        env.update({"DRY_RUN": "1", "SEEDS": "2024"})
        env.update(env_extra)
        return subprocess.run(["bash", "scripts/run_phase6_fullfield_candidates.sh"],
                              cwd=REPO, capture_output=True, text=True, env=env)

    def test_count_and_pems_skip(self):
        r = self._run_dry({"DATASETS": "weather PEMS04", "SEQ_LENS": "96 720", "RUN_PEMS_SEQ720": "0"})
        self.assertEqual(r.returncode, 0, r.stderr)
        # weather 2sl*4pl*1seed*6=48 ; PEMS04 1sl*4pl*1seed*6=24 -> 72
        self.assertIn("estimate: 72 runs", r.stdout)
        # no PEMS seq_len 720 dry command
        self.assertNotIn("PEMS04_L720", r.stdout)

    def test_pems_seq720_enabled(self):
        r = self._run_dry({"DATASETS": "PEMS04", "SEQ_LENS": "96 720", "RUN_PEMS_SEQ720": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("estimate: 48 runs", r.stdout)  # 2sl*4pl*1seed*6
        self.assertIn("PEMS04_L720", r.stdout)

    def test_period_defaults_and_override(self):
        r = self._run_dry({"DATASETS": "weather electricity ETTm1 PEMS04", "SEQ_LENS": "96"})
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout.replace("\\", "")  # printf %q escapes commas for copy-paste
        self.assertIn("--periods 144", out)       # weather
        self.assertIn("--periods 24,168", out)    # electricity
        self.assertIn("--periods 96,672", out)    # ETTm1
        self.assertIn("--periods 24 ", out)       # PEMS04
        r2 = self._run_dry({"DATASETS": "weather", "SEQ_LENS": "96", "PERIODS": "48,96"})
        self.assertIn("--periods 48,96", r2.stdout.replace("\\", ""))  # env override


# ---------- F. backward compatibility (bash -n) ----------
class TestBackwardCompat(unittest.TestCase):
    def test_phase_scripts_parse(self):
        for s in ["scripts/run_phase5_fullfield_candidates.sh", "scripts/run_phase5_selection.sh",
                  "scripts/run_phase4_final_candidates.sh", "scripts/run_phase6_fullfield_candidates.sh",
                  "scripts/run_phase6_selector_audit.sh", "scripts/run_phase6_fullfield_selection.sh"]:
            r = subprocess.run(["bash", "-n", s], cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{s}: {r.stderr}")


# ---------- G. summary robustness ----------
class TestSummary(unittest.TestCase):
    def _results(self):
        rows = []
        for ds in ["weather", "electricity", "PEMS04"]:
            for pl in [96, 192]:
                for seed in [1, 2]:
                    for arm in ["phase6_asx_cross", "phase6_asx_period_multi"]:
                        rows.append(dict(arm=arm, dataset=ds, seq_len=720, pred_len=pl, seed=seed,
                                         status="ok", val_mse=0.2, mse=0.15, mae=0.25))
        return rows

    def test_no_selected_no_baseline_no_diag(self):
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "results.csv"); write_csv(c, self._results())
            r = run(["scripts/summarize_phase6_fullfield.py", "--csv", c,
                     "--anchor_arm", "phase6_asx_cross", "--output_dir", d])
            self.assertEqual(r.returncode, 0, r.stderr)
            txt = open(os.path.join(d, "summary_phase6_fullfield.md")).read()
            self.assertIn("Phase 6 Full-Field Summary", txt)
            self.assertIn("Per Dataset Family", txt)
            self.assertIn("ANALYSIS ONLY", txt)

    def test_with_selected(self):
        with tempfile.TemporaryDirectory() as d:
            c = os.path.join(d, "results.csv"); write_csv(c, self._results())
            sel = [dict(dataset="weather", seq_len=720, pred_len=96, seed=1, selected_arm="phase6_asx_period_multi",
                        val_score=0.2, val_mse=0.2, test_mse=0.15, test_mae=0.25)]
            sc = os.path.join(d, "selected.csv"); write_csv(sc, sel)
            r = run(["scripts/summarize_phase6_fullfield.py", "--csv", c, "--selected_csv", sc,
                     "--output_dir", d])
            self.assertEqual(r.returncode, 0, r.stderr)
            txt = open(os.path.join(d, "summary_phase6_fullfield.md")).read()
            self.assertIn("Validation-Selected Summary", txt)
            self.assertIn("test_oracle", txt.lower())


if __name__ == "__main__":
    unittest.main()
