#!/usr/bin/env python3
"""Build the auditable L=96/L=720 Compact+Echo and baseline result bundle.

The collector intentionally keeps three kinds of evidence separate:

1. three-seed locked Compact+Echo production-route tests;
2. legacy local baseline reproductions;
3. published L=720 MSE references transcribed in RESULTS.md.

It never substitutes validation loss for test loss and never fabricates MAPE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


OURS_NAME = "Compact + Echo"
LEGACY_OURS_NAME = "Asy2+Echo"

# Public Ours is a validation-locked routed family.  This legacy map selects
# the eight 0717v1 anchor rows from the four-arm research summary.  A completed
# 0717v2 source instead contains exactly one production-selected row for every
# one of the 88 cells and therefore bypasses this anchor-only filter.
OURS_PRODUCTION_ARMS = {
    ("electricity", 96, 96): "asy1_echo",
    ("electricity", 96, 720): "compact_echo",
    ("electricity", 720, 96): "compact_echo",
    ("electricity", 720, 720): "compact_echo",
    ("weather", 96, 96): "asy1_echo",
    ("weather", 96, 720): "asy1_echo",
    ("weather", 720, 96): "compact_echo",
    ("weather", 720, 720): "compact_echo",
}

MODEL_ORDER = (
    OURS_NAME,
    "TQNet",
    "CycleNet",
    "FITS",
    "SparseTSF",
    "FreTS",
    "FilterNet",
    "iTransformer",
    "PatchTST",
    "DLinear",
    "MixLinear",
    "PhaseFormer",
    "FreqCycle",
)
BASELINE_MODELS = MODEL_ORDER[1:]
CORE_BASELINE_MODELS = MODEL_ORDER[1:11]
DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "weather",
    "electricity",
    "traffic",
    "PEMS03",
    "PEMS04",
    "PEMS07",
    "PEMS08",
)
SEQ_LENS = (96, 720)
HORIZONS = {
    dataset: ((12, 24, 48, 96) if dataset.startswith("PEMS") else (96, 192, 336, 720))
    for dataset in DATASETS
}
PUBLISHED_MODELS = ("SparseTSF", "FITS", "PatchTST", "DLinear")
PUBLISHED_DATASETS = DATASETS[:7]

CSV_FIELDS = (
    "model",
    "dataset",
    "seq_len",
    "pred_len",
    "status",
    "mse",
    "mae",
    "mape",
    "rmse",
    "param_count",
    "forward_active_param_count",
    "inactive_registered_param_count",
    "real_scalar_equivalent",
    "fp32_model_mib",
    "parameter_source",
    "seed",
    "train_seconds",
    "test_elapsed_seconds",
    "inference_seconds",
    "forward_ms_per_sample",
    "test_peak_gpu_mem_mb",
    "source_type",
    "protocol",
    "source_path",
    "published_reference_mse",
    "published_reference_source",
    "notes",
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    snapshot = Path(__file__).resolve().parent / "source_snapshot_0716v1"
    ours_snapshot = Path(__file__).resolve().parent / "source_snapshot_0717v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", type=Path, default=snapshot / "results.csv")
    parser.add_argument("--sl96-audit", type=Path, default=snapshot / "sl96_results.csv")
    parser.add_argument(
        "--ours-results",
        type=Path,
        default=ours_snapshot / "compact_echo_summary_3seed.csv",
        help="Locked three-seed Asy1/Compact x Echo summary used by production OurModel.",
    )
    parser.add_argument(
        "--phaseformer-results", type=Path, default=snapshot / "phaseformer_stack_results.csv"
    )
    parser.add_argument("--freqcycle-results", type=Path, default=snapshot / "freqcycle_results.csv")
    parser.add_argument(
        "--completion-results",
        type=Path,
        default=None,
        help="Optional normalized 0716v2 completion audit; completed rows override prior gaps.",
    )
    parser.add_argument("--legacy-report", type=Path, default=repo / "RESULTS.md")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=repo / "COMPACT_ECHO_ALL_BASELINES_L96_L720_0717V1.md",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo / "COMPACT_ECHO_ALL_BASELINES_L96_L720_0717V1_RESULTS.csv",
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=repo / "COMPACT_ECHO_ALL_BASELINES_L96_L720_0717V1_COVERAGE.csv",
    )
    parser.add_argument(
        "--ours-source-path",
        default="",
        help="Provenance string recorded in ours rows' source_path; empty keeps the legacy 0717 snapshot paths.",
    )
    parser.add_argument(
        "--report-version",
        default="",
        help="Override the report title version tag; empty keeps the legacy 0717 auto logic.",
    )
    parser.add_argument(
        "--extra-method-note",
        default="",
        help="Optional extra bullet appended to the Compact + Echo method-definition section.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def as_int(value: Any) -> int | None:
    number = as_float(value)
    return None if number is None else int(number)


def parse_elapsed(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    parts = [int(part) for part in text.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        return float(parts[0])
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def clean_number(text: str) -> float | None:
    cleaned = text.replace("**", "").replace("·2sd", "").strip()
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)", cleaned)
    return float(match.group(0)) if match else None


def parse_published_l720(path: Path) -> dict[tuple[str, str, int, int], float]:
    """Parse only the four requested published-reference columns."""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "## Table @ sl=720")
    except StopIteration as exc:
        raise AssertionError("Missing L=720 published-reference section") from exc

    output: dict[tuple[str, str, int, int], float] = {}
    dataset: str | None = None
    header: list[str] | None = None
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("### "):
            candidate = line[4:].strip()
            dataset = candidate if candidate in PUBLISHED_DATASETS else None
            header = None
            continue
        if dataset is None or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == "pl":
            header = cells
            continue
        if not header or not cells or not cells[0].isdigit() or len(cells) != len(header):
            continue
        horizon = int(cells[0])
        for model in PUBLISHED_MODELS:
            if model not in header:
                continue
            value = clean_number(cells[header.index(model)])
            if value is not None:
                output[(model, dataset, 720, horizon)] = value
    expected = len(PUBLISHED_MODELS) * len(PUBLISHED_DATASETS) * 4
    if len(output) != expected:
        raise AssertionError(f"Published L=720 parse yielded {len(output)}, expected {expected}")
    return output


def baseline_missing(model: str, dataset: str, seq_len: int, pred_len: int) -> tuple[str, str]:
    if model == "SparseTSF" and dataset.startswith("PEMS") and pred_len == 12:
        return (
            "incompatible_config",
            "Locked period_len=24 is not divisible into pred_len=12; the model cannot run this cell.",
        )
    if model == "FreTS" and seq_len == 720:
        if dataset in {"traffic", "PEMS07"}:
            return ("cuda_oom", "CUDA out of memory in the recorded L=720 attempt; no final test metrics.")
        if (dataset, pred_len) in {("electricity", 336), ("PEMS03", 48)}:
            return ("incomplete_no_final_metrics", "Training log ended before final test metrics were emitted.")
        if (dataset, pred_len) == ("PEMS03", 96):
            return ("not_run_or_no_log", "No matching run log or final artifact was found.")
    if model == "PatchTST" and seq_len == 720:
        if (dataset, pred_len) in {("PEMS03", 24), ("PEMS08", 24)}:
            return ("incomplete_no_final_metrics", "Training log ended before final test metrics were emitted.")
        return ("not_run_or_no_log", "No matching final metric artifact was found.")
    return ("missing_result", "No completed local result was found.")


def patchtst_recovered_param(
    baseline_index: dict[tuple[str, str, int, int], dict[str, str]], seq_len: int, pred_len: int
) -> int:
    # Locked standard profile: patch_len=16, stride=8, d_model=128, padding_patch=end.
    patch_num = (seq_len - 16) // 8 + 2
    per_horizon_step = 128 * patch_num + 1
    bases: set[int] = set()
    for (model, dataset, sl, horizon), row in baseline_index.items():
        if model != "PatchTST" or sl != seq_len or dataset in {"ETTh1", "ETTh2"}:
            continue
        n_param = int(row["n_param"])
        bases.add(n_param - horizon * per_horizon_step)
    if len(bases) != 1:
        raise AssertionError(f"PatchTST locked-profile base is ambiguous at L={seq_len}: {bases}")
    return bases.pop() + pred_len * per_horizon_step


def recover_param(
    model: str,
    dataset: str,
    seq_len: int,
    pred_len: int,
    baseline_index: dict[tuple[str, str, int, int], dict[str, str]],
) -> tuple[int | None, str]:
    if model == "SparseTSF" and dataset.startswith("PEMS") and pred_len == 12:
        return None, "not_applicable_incompatible_config"
    if model == "FreTS":
        values = {
            int(row["n_param"])
            for (candidate_model, _dataset, sl, horizon), row in baseline_index.items()
            if candidate_model == model and sl == seq_len and horizon == pred_len
        }
        if len(values) == 1:
            return values.pop(), "recovered_same_locked_config"
        raise AssertionError(f"Cannot uniquely recover FreTS params for L={seq_len}, H={pred_len}: {values}")
    if model == "PatchTST":
        return patchtst_recovered_param(baseline_index, seq_len, pred_len), "recovered_locked_config_formula"
    return None, "not_available"


MIXLINEAR_CONFIG = {
    "ETTh1": (24, 5),
    "ETTh2": (24, 15),
    "ETTm1": (2, 144),
    "ETTm2": (2, 144),
    "weather": (4, 5),
    "electricity": (24, 5),
    "traffic": (24, 5),
    "PEMS03": (12, 5),
    "PEMS04": (12, 5),
    "PEMS07": (12, 5),
    "PEMS08": (12, 5),
}


def real_scalar_equivalent(model: str, dataset: str, pred_len: int, param_count: int | None) -> int | None:
    if param_count is None:
        return None
    if model == "FITS":
        return 2 * param_count
    if model == "MixLinear":
        period_len, lpf = MIXLINEAR_CONFIG[dataset]
        complex_entries = 2 * lpf + 2 * math.ceil(pred_len / period_len)
        return param_count + complex_entries
    return param_count


def source_row(
    *,
    model: str,
    dataset: str,
    seq_len: int,
    pred_len: int,
    status: str,
    mse: str = "",
    mae: str = "",
    mape: str = "",
    param_count: int | None = None,
    forward_active_param_count: int | None = None,
    inactive_registered_param_count: int | None = None,
    parameter_source: str = "",
    seed: str = "",
    train_seconds: Any = "",
    test_elapsed_seconds: Any = "",
    inference_seconds: Any = "",
    forward_ms_per_sample: Any = "",
    test_peak_gpu_mem_mb: Any = "",
    source_type: str = "",
    protocol: str = "",
    source_path: str = "",
    published_reference_mse: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    mse_value = as_float(mse)
    real_equiv = real_scalar_equivalent(model, dataset, pred_len, param_count)
    row: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "status": status,
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "rmse": "" if mse_value is None else f"{math.sqrt(mse_value):.12g}",
        "param_count": "" if param_count is None else param_count,
        "forward_active_param_count": (
            "" if forward_active_param_count is None else forward_active_param_count
        ),
        "inactive_registered_param_count": (
            "" if inactive_registered_param_count is None else inactive_registered_param_count
        ),
        "real_scalar_equivalent": "" if real_equiv is None else real_equiv,
        "fp32_model_mib": "" if real_equiv is None else f"{real_equiv * 4 / (1024**2):.9f}",
        "parameter_source": parameter_source,
        "seed": seed,
        "train_seconds": train_seconds,
        "test_elapsed_seconds": test_elapsed_seconds,
        "inference_seconds": inference_seconds,
        "forward_ms_per_sample": forward_ms_per_sample,
        "test_peak_gpu_mem_mb": test_peak_gpu_mem_mb,
        "source_type": source_type,
        "protocol": protocol,
        "source_path": source_path,
        "published_reference_mse": "" if published_reference_mse is None else f"{published_reference_mse:.3f}",
        "published_reference_source": (
            "SparseTSF paper Tables 10-11 (transcribed in RESULTS.md)"
            if published_reference_mse is not None
            else ""
        ),
        "notes": notes,
        "_mse": mse_value,
        "_mae": as_float(mae),
        "_mape": as_float(mape),
        "_param": param_count,
        "_active_param": forward_active_param_count,
        "_real_equiv": real_equiv,
    }
    return row


def build_rows(
    baseline_rows: list[dict[str, str]],
    ours_rows: list[dict[str, str]],
    phaseformer_rows: list[dict[str, str]],
    freqcycle_rows: list[dict[str, str]],
    completion_rows: list[dict[str, str]],
    published: dict[tuple[str, str, int, int], float],
    ours_source_path: str = "",
) -> list[dict[str, Any]]:
    baseline_index: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in baseline_rows:
        model = row["model"]
        dataset = row["dataset"]
        seq_len = int(row["seq_len"])
        pred_len = int(row["pred_len"])
        if model not in CORE_BASELINE_MODELS or dataset not in DATASETS or seq_len not in SEQ_LENS:
            continue
        key = (model, dataset, seq_len, pred_len)
        if key in baseline_index:
            raise AssertionError(f"Duplicate local baseline row: {key}")
        baseline_index[key] = row

    ours_index: dict[tuple[str, int, int], dict[str, str]] = {}
    full_production_source = len(ours_rows) == 88
    for row in ours_rows:
        dataset = row["dataset"]
        seq_len = int(row["seq_len"])
        pred_len = int(row["pred_len"])
        key = (dataset, seq_len, pred_len)
        if not full_production_source and row.get("arm") != OURS_PRODUCTION_ARMS.get(key):
            continue
        if dataset not in DATASETS or seq_len not in SEQ_LENS:
            continue
        if pred_len not in HORIZONS[dataset]:
            continue
        if key in ours_index:
            raise AssertionError(f"Duplicate Compact + Echo production row: {key}")
        ours_index[key] = row

    if len(baseline_index) != 841:
        raise AssertionError(f"Expected 841 completed local baseline cells, found {len(baseline_index)}")
    expected_ours_cells = 88 if full_production_source else len(OURS_PRODUCTION_ARMS)
    if len(ours_index) != expected_ours_cells:
        raise AssertionError(
            f"Expected {expected_ours_cells} locked Compact + Echo production cells, "
            f"found {len(ours_index)}"
        )

    phaseformer_index: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in phaseformer_rows:
        if row.get("backbone") != "phaseformer" or row.get("status") != "ok":
            continue
        dataset = row["dataset"]
        seq_len = int(row["seq_len"])
        pred_len = int(row["pred_len"])
        if dataset not in DATASETS or seq_len not in SEQ_LENS:
            continue
        key = (dataset, seq_len, pred_len)
        if key in phaseformer_index:
            raise AssertionError(f"Duplicate PhaseFormer row: {key}")
        phaseformer_index[key] = row

    freqcycle_index: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in freqcycle_rows:
        if row.get("model") != "freqcycle" or row.get("status") != "ok":
            continue
        dataset = row["dataset"]
        seq_len = int(row["seq_len"])
        pred_len = int(row["pred_len"])
        if dataset not in DATASETS or seq_len not in SEQ_LENS:
            continue
        key = (dataset, seq_len, pred_len)
        if key in freqcycle_index:
            raise AssertionError(f"Duplicate FreqCycle row: {key}")
        freqcycle_index[key] = row

    if len(phaseformer_index) != 16:
        raise AssertionError(f"Expected 16 completed PhaseFormer cells, found {len(phaseformer_index)}")
    if len(freqcycle_index) != 16:
        raise AssertionError(f"Expected 16 completed FreqCycle cells, found {len(freqcycle_index)}")

    completion_index: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in completion_rows:
        if row.get("completion_status") != "ok":
            continue
        model = row["model"]
        # The running 0716v2 matrix used the legacy AsyCycleV2 branch.  Keep
        # those raw artifacts for audit, but do not present them as the newer
        # production Compact + Echo family.
        if model == LEGACY_OURS_NAME:
            continue
        dataset = row["dataset"]
        seq_len = int(row["seq_len"])
        pred_len = int(row["pred_len"])
        if model not in MODEL_ORDER or dataset not in DATASETS or seq_len not in SEQ_LENS:
            continue
        key = (model, dataset, seq_len, pred_len)
        if key in completion_index:
            raise AssertionError(f"Duplicate completion row: {key}")
        completion_index[key] = row

    rows: list[dict[str, Any]] = []
    for seq_len in SEQ_LENS:
        for dataset in DATASETS:
            for pred_len in HORIZONS[dataset]:
                for model in MODEL_ORDER:
                    paper_mse = published.get((model, dataset, seq_len, pred_len))
                    completed = completion_index.get((model, dataset, seq_len, pred_len))
                    if completed:
                        rows.append(
                            source_row(
                                model=model,
                                dataset=dataset,
                                seq_len=seq_len,
                                pred_len=pred_len,
                                status="ok",
                                mse=completed["mse"],
                                mae=completed["mae"],
                                mape=completed["mape"],
                                param_count=int(completed["param_count"]),
                                forward_active_param_count=as_int(
                                    completed.get("forward_active_param_count", "")
                                ),
                                inactive_registered_param_count=as_int(
                                    completed.get("inactive_registered_param_count", "")
                                ),
                                parameter_source="0716v2_audited_param_count",
                                seed=completed["seed"],
                                train_seconds=completed.get("t_train", ""),
                                inference_seconds=completed.get("t_inf", ""),
                                forward_ms_per_sample=completed.get("forward_ms_per_sample", ""),
                                test_peak_gpu_mem_mb=completed.get("test_peak_gpu_mem_mb", ""),
                                source_type="local_completion_strict",
                                protocol="validation-selected checkpoint; one deferred held-out test; sample-weighted streaming metrics",
                                source_path="outputs/completion_0716v2/COMPLETION_MATRIX_0716V2_RESULTS.csv",
                                published_reference_mse=paper_mse,
                                notes="0716v2 missing-cell completion; MAPE computed with the locked historical no-epsilon definition.",
                            )
                        )
                        continue
                    if model == OURS_NAME:
                        source = ours_index.get((dataset, seq_len, pred_len))
                        if source:
                            rows.append(
                                source_row(
                                    model=model,
                                    dataset=dataset,
                                    seq_len=seq_len,
                                    pred_len=pred_len,
                                    status="ok",
                                    mse=source["test_mse_mean"],
                                    mae=source["test_mae_mean"],
                                    mape=source.get("mape_mean", ""),
                                    param_count=int(source["param_count"]),
                                    parameter_source="locked_3seed_actual_param_count",
                                    seed=source.get("seeds", "2024,2025,2026"),
                                    train_seconds=source.get("train_time_mean_s", ""),
                                    test_peak_gpu_mem_mb=source.get(
                                        "test_peak_gpu_mem_mean_mb", ""
                                    ),
                                    source_type="local_compact_echo_locked_3seed_mean",
                                    protocol=(
                                        "validation-locked production route; three matched seeds; "
                                        "one deferred held-out test per checkpoint"
                                    ),
                                    source_path=(
                                        ours_source_path
                                        if ours_source_path
                                        else "analysis_exp/source_snapshot_0717v2/"
                                        "compact_echo_production_88cells.csv"
                                        if full_production_source
                                        else "analysis_exp/source_snapshot_0717v1/"
                                        "compact_echo_summary_3seed.csv"
                                    ),
                                    notes=(
                                        f"Production route arm={source['arm']}; "
                                        + (
                                            "MAPE was streamed during the 0717v2 completion."
                                            if source.get("mape_mean", "")
                                            else "MAPE was not persisted for this anchor."
                                        )
                                    ),
                                )
                            )
                        else:
                            rows.append(
                                source_row(
                                    model=model,
                                    dataset=dataset,
                                    seq_len=seq_len,
                                    pred_len=pred_len,
                                    status="not_available",
                                    source_type="not_available",
                                    protocol="no locked Compact + Echo production run",
                                    notes=(
                                        "No locked Compact + Echo production-route result is "
                                        "available for this cell."
                                    ),
                                )
                            )
                        continue

                    if model == "PhaseFormer":
                        source = phaseformer_index.get((dataset, seq_len, pred_len))
                        if source:
                            elapsed: float | str = ""
                            summary_path = Path(source.get("baseline_summary_path", ""))
                            if summary_path.is_file():
                                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                                elapsed = summary.get("elapsed_seconds", "")
                            rows.append(
                                source_row(
                                    model=model,
                                    dataset=dataset,
                                    seq_len=seq_len,
                                    pred_len=pred_len,
                                    status="ok",
                                    mse=source["baseline_mse"],
                                    mae=source["baseline_mae"],
                                    param_count=int(source["baseline_params"]),
                                    parameter_source="audited_baseline_param_count",
                                    seed=source["seed"],
                                    train_seconds=elapsed,
                                    source_type="local_phaseformer_run_summary",
                                    protocol="standalone held-out test from validation-monitored run summary",
                                    source_path="baseline_2026_results/phaseformer/run_summary.json",
                                    notes="Weather/Electricity locked reproduction; MAPE was not persisted.",
                                )
                            )
                        else:
                            rows.append(
                                source_row(
                                    model=model,
                                    dataset=dataset,
                                    seq_len=seq_len,
                                    pred_len=pred_len,
                                    status="not_available",
                                    source_type="not_available",
                                    protocol="no locked local run",
                                    notes="No locked PhaseFormer local result is available for this cell.",
                                )
                            )
                        continue

                    if model == "FreqCycle":
                        source = freqcycle_index.get((dataset, seq_len, pred_len))
                        if source:
                            rows.append(
                                source_row(
                                    model=model,
                                    dataset=dataset,
                                    seq_len=seq_len,
                                    pred_len=pred_len,
                                    status="ok",
                                    mse=source["test_mse"],
                                    mae=source["test_mae"],
                                    param_count=int(source["registered_trainable_numel"]),
                                    forward_active_param_count=int(source["forward_active_numel"]),
                                    inactive_registered_param_count=int(source["inactive_registered_numel"]),
                                    parameter_source="audited_registered_and_forward_active_numel",
                                    seed=source["seed"],
                                    train_seconds=source.get("t_train", ""),
                                    inference_seconds=source.get("t_inf", ""),
                                    test_peak_gpu_mem_mb=source.get("peak_gpu_mem_mb", ""),
                                    source_type="local_strict_official_reproduction",
                                    protocol="validation-selected checkpoint; deferred held-out test; sample-weighted metrics",
                                    source_path="outputs/freqcycle_official/freqcycle_0716v1_results.csv",
                                    notes="Official-port reproduction; registered count includes an upstream duplicate inactive MLP; MAPE was not persisted.",
                                )
                            )
                        else:
                            rows.append(
                                source_row(
                                    model=model,
                                    dataset=dataset,
                                    seq_len=seq_len,
                                    pred_len=pred_len,
                                    status="not_available",
                                    source_type="not_available",
                                    protocol="no locked local run",
                                    notes="No locked FreqCycle local result is available for this cell.",
                                )
                            )
                        continue

                    source = baseline_index.get((model, dataset, seq_len, pred_len))
                    if source:
                        rows.append(
                            source_row(
                                model=model,
                                dataset=dataset,
                                seq_len=seq_len,
                                pred_len=pred_len,
                                status="ok",
                                mse=source["mse"],
                                mae=source["mae"],
                                param_count=int(source["n_param"]),
                                parameter_source="logged_n_param",
                                seed=source["seed"],
                                train_seconds=source.get("t_train", ""),
                                inference_seconds=source.get("t_inf", ""),
                                source_type="local_legacy_reproduction",
                                protocol="validation-selected legacy run; test loss logged during training",
                                source_path="note/results.csv",
                                published_reference_mse=paper_mse,
                                notes="Held-out test MSE/MAE from the local legacy log; MAPE was not persisted.",
                            )
                        )
                    else:
                        status, note = baseline_missing(model, dataset, seq_len, pred_len)
                        recovered_param, parameter_source = recover_param(
                            model, dataset, seq_len, pred_len, baseline_index
                        )
                        rows.append(
                            source_row(
                                model=model,
                                dataset=dataset,
                                seq_len=seq_len,
                                pred_len=pred_len,
                                status=status,
                                param_count=recovered_param,
                                parameter_source=parameter_source,
                                seed="2026",
                                source_type="local_missing_or_failed",
                                protocol="locked local baseline configuration",
                                source_path="note/results.csv",
                                published_reference_mse=paper_mse,
                                notes=note,
                            )
                        )
    expected = len(MODEL_ORDER) * len(DATASETS) * len(SEQ_LENS) * 4
    if len(rows) != expected:
        raise AssertionError(f"Expected {expected} exact-grid rows, found {len(rows)}")
    return rows


def audit_sl96(rows: list[dict[str, Any]], audit_path: Path) -> tuple[int, int]:
    audit = {
        (row["model"], row["dataset"], int(row["pred_len"])): row
        for row in read_csv(audit_path)
        if row["model"] in CORE_BASELINE_MODELS and row["dataset"] in DATASETS and row["mse"] and row["mae"]
    }
    generated = {
        (row["model"], row["dataset"], row["pred_len"]): row
        for row in rows
        if row["model"] in CORE_BASELINE_MODELS and row["seq_len"] == 96 and row["_mse"] is not None
    }
    # The rounded 0716v1 audit is a regression oracle for the cells it knew
    # about.  A completion run may legitimately add formerly missing L=96
    # cells, so require the old keys to remain present instead of rejecting
    # the expanded key set.
    missing_audit_keys = audit.keys() - generated.keys()
    if missing_audit_keys:
        raise AssertionError(
            f"L=96 audit keys disappeared from the generated grid: "
            f"{sorted(missing_audit_keys)[:5]}"
        )
    mismatches = 0
    for key, source in audit.items():
        row = generated[key]
        if abs(float(source["mse"]) - row["_mse"]) > 5.1e-5:
            mismatches += 1
        if abs(float(source["mae"]) - row["_mae"]) > 5.1e-5:
            mismatches += 1
    if mismatches:
        raise AssertionError(f"L=96 rounded audit found {mismatches} metric mismatches")
    return len(audit), mismatches


def write_long_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def coverage_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for model in MODEL_ORDER:
        for seq_len in SEQ_LENS:
            subset = [row for row in rows if row["model"] == model and row["seq_len"] == seq_len]
            status_counts = Counter(row["status"] for row in subset)
            records.append(
                {
                    "model": model,
                    "seq_len": seq_len,
                    "expected_cells": len(subset),
                    "mse_cells": sum(row["_mse"] is not None for row in subset),
                    "mae_cells": sum(row["_mae"] is not None for row in subset),
                    "mape_cells": sum(bool(row["mape"]) for row in subset),
                    "parameter_cells": sum(row["_param"] is not None for row in subset),
                    "ok_cells": status_counts["ok"],
                    "missing_or_failed_cells": len(subset) - status_counts["ok"],
                    "status_counts": "; ".join(f"{key}={status_counts[key]}" for key in sorted(status_counts)),
                }
            )
    return records


def write_coverage_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def fmt_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def fmt_param(value: int | None) -> str:
    return "N/A" if value is None else f"{value:,}"


def median_number(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [as_float(row[field]) for row in rows]
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def metric_cell(row: dict[str, Any]) -> str:
    if row["_mse"] is not None:
        return f"{row['_mse']:.6f} / {row['_mae']:.6f} / {fmt_metric(row['_mape'])}"
    labels = {
        "cuda_oom": "OOM",
        "incompatible_config": "incompatible",
        "incomplete_no_final_metrics": "incomplete",
        "not_run_or_no_log": "not run",
        "not_available": "N/A",
        "missing_result": "missing",
    }
    return labels.get(row["status"], "N/A")


def parameter_cell(row: dict[str, Any]) -> str:
    param = row["_param"]
    if param is None:
        return "N/A"
    if row["_active_param"] is not None and row["_active_param"] != param:
        return f"{param:,} (active {row['_active_param']:,})"
    suffix = "†" if str(row["parameter_source"]).startswith("recovered_") else ""
    if row["_real_equiv"] != param:
        return f"{param:,} → {row['_real_equiv']:,}{suffix}"
    return f"{param:,}{suffix}"


def table(lines: list[str], headers: list[str], body: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")


def render_report(
    rows: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    inputs: dict[str, Path],
    sl96_audit_count: int,
    report_version_override: str = "",
    extra_method_note: str = "",
) -> str:
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    index = {
        (row["model"], row["dataset"], row["seq_len"], row["pred_len"]): row for row in rows
    }
    mse_count = sum(row["_mse"] is not None for row in rows)
    mae_count = sum(row["_mae"] is not None for row in rows)
    mape_count = sum(row["_mape"] is not None for row in rows)
    param_count = sum(row["_param"] is not None for row in rows)
    baseline_mse_count = sum(
        row["_mse"] is not None for row in rows if row["model"] in BASELINE_MODELS
    )
    completion_count = sum(row["source_type"] == "local_completion_strict" for row in rows)
    model_completed = {
        model: sum(row["_mse"] is not None for row in rows if row["model"] == model)
        for model in (OURS_NAME, "PhaseFormer", "FreqCycle")
    }
    ours_full = model_completed[OURS_NAME] == 88
    if report_version_override:
        report_version = report_version_override
    elif ours_full:
        report_version = "0717v2-fullgrid+completion0716v2" if "completion" in inputs else "0717v2-fullgrid"
    else:
        report_version = "0717v1+completion0716v2" if "completion" in inputs else "0717v1"
    lines: list[str] = []
    lines.extend(
        [
            f"# Compact + Echo（Ours）与全部 baselines：L=96 / L=720 结果审计（{report_version}）",
            "",
            f"> 生成时间：`{generated}`。表中模型顺序固定为用户指定顺序，Compact + Echo（Ours）始终第一。",
            "",
            "## 1. 口径与结论先读",
            "",
            "- 网格：13 个模型 × 11 个数据集 × 2 个输入长度 × 每数据集 4 个预测长度，共 **1,144** 行。非 PEMS 的 H 为 96/192/336/720；PEMS 的 H 为 12/24/48/96。",
            "- 所有已填 MSE/MAE 都是 held-out test 指标；没有用 validation loss 顶替 test loss。",
            f"- MSE/MAE 当前覆盖 **{mse_count}/1,144** 与 **{mae_count}/1,144**；其中 12 个 baselines 为 **{baseline_mse_count}/1,056**。",
            f"- Compact + Echo、PhaseFormer、FreqCycle 当前分别完成 **{model_completed[OURS_NAME]}/88**、**{model_completed['PhaseFormer']}/88**、**{model_completed['FreqCycle']}/88**。",
            f"- 0716v2 baseline 严格缺口补跑已并入 **{completion_count}/183** 格；新补跑单元使用 validation-selected checkpoint 后的一次 held-out test。旧 ours 分支的 72 个补跑单元保留在原始审计中，不会冒充 Compact + Echo。",
            f"- **MAPE 覆盖为 {mape_count}/1,144**：旧实验未落盘的 MAPE 继续保持 N/A；0716v2 baseline 与 0717v2 Ours 补跑使用同一历史定义流式计算，绝不把缺失写成 0。",
            f"- 参数量覆盖 **{param_count}/1,144**；旧缺失单元在尚未补跑时仍按同配置恢复值标 `†`，补跑完成后由实际 checkpoint 参数量替换。",
            f"- Compact + Echo 当前 {model_completed[OURS_NAME]} 格来自 seeds 2024/2025/2026 的 validation-locked 三种子均值；原 10 个 baseline 多为 seed=2026 的旧本地复现，PhaseFormer 为 seed=2021，FreqCycle 为 seed=2026。可做描述性比较，但不应包装成完全同协议的统计显著性结论。",
            "",
            "指标单元格式：`MSE / MAE / MAPE`。参数单元默认是框架登记的 trainable elements；若含 complex64 参数则显示 `登记量 → FP32 实标量等价量`。FreqCycle 另列 forward-active 参数，因为官方结构登记了一个 forward 未使用的重复 MLP。",
            "",
            "### Compact + Echo 的方法定义",
            "",
            "- Compact 不是另一个大模型，而是压缩 Asy1 最占参数的时间映射；保留 RevIN、低秩周期表、非对称预测骨架以及 target-only 性质。",
            "- 长输入可用 `周期位置 × 周期编号` 的低秩分解（PCKE），并保留最近 96 步的小残差；长输出可使用同类 Kronecker 分解。",
            "- Electricity 的已验证部分任务使用共享周期因子，替代每通道独立系数；未验证的任务不会自动套用该结构。",
            "- Echo 是独立的同相位残差捷径，仅增加 `cycle_rank + 2` 个参数：Weather 为 4，Electricity 为 18，其余当前默认 rank=8 的数据集为 10。",
            "- 生产模型按 `(dataset, input length, output length)` 选择已锁定 Compact profile；未通过 Compact 门槛的已审计单元回退到 Asy1+Echo。",
            *( [extra_method_note] if extra_method_note else [] ),
            "",
            "## 2. 覆盖率",
            "",
        ]
    )
    coverage_body = []
    for record in coverage:
        coverage_body.append(
            [
                record["model"],
                str(record["seq_len"]),
                f"{record['mse_cells']}/{record['expected_cells']}",
                f"{record['mae_cells']}/{record['expected_cells']}",
                f"{record['mape_cells']}/{record['expected_cells']}",
                f"{record['parameter_cells']}/{record['expected_cells']}",
                str(record["missing_or_failed_cells"]),
            ]
        )
    table(lines, ["Model", "L", "MSE", "MAE", "MAPE", "Params", "缺失/失败"], coverage_body)

    lines.extend(["", "## 3. 写论文可直接使用的总体统计", ""])
    aggregate_body = []
    for model in MODEL_ORDER:
        for seq_len in SEQ_LENS:
            subset = [row for row in rows if row["model"] == model and row["seq_len"] == seq_len]
            metrics = [row for row in subset if row["_mse"] is not None]
            params = [row["_param"] for row in subset if row["_param"] is not None]
            mean_mse = statistics.mean(row["_mse"] for row in metrics) if metrics else None
            mean_mae = statistics.mean(row["_mae"] for row in metrics) if metrics else None
            train_median = median_number(subset, "train_seconds")
            inference_median = median_number(subset, "inference_seconds")
            latency_median = median_number(subset, "forward_ms_per_sample")
            aggregate_body.append(
                [
                    model,
                    str(seq_len),
                    f"{len(metrics)}/44",
                    fmt_metric(mean_mse),
                    fmt_metric(mean_mae),
                    fmt_param(int(statistics.median(params)) if params else None),
                    (f"{min(params):,}–{max(params):,}" if params else "N/A"),
                    fmt_seconds(train_median),
                    fmt_seconds(inference_median),
                    ("N/A" if latency_median is None else f"{latency_median:.4f}"),
                ]
            )
    table(
        lines,
        [
            "Model",
            "L",
            "n/44",
            "Macro MSE",
            "Macro MAE",
            "Median params",
            "Param range",
            "Median train s",
            "Median inference s",
            "Median forward ms/sample",
        ],
        aggregate_body,
    )
    lines.extend(
        [
            "",
            "注意：Macro 均值只平均有结果的单元，覆盖不同的模型不可直接靠这一列排名；训练/推理秒数也受数据集规模影响。完整覆盖与逐数据集表才是主证据。",
            "",
            "### Compact + Echo 对本地最佳 baseline",
            "",
        ]
    )
    comparison_body = []
    ours_rows = [row for row in rows if row["model"] == OURS_NAME and row["_mse"] is not None]
    wins = 0
    for ours in ours_rows:
        candidates = [
            index[(model, ours["dataset"], ours["seq_len"], ours["pred_len"])]
            for model in BASELINE_MODELS
            if index[(model, ours["dataset"], ours["seq_len"], ours["pred_len"])]["_mse"] is not None
        ]
        best = min(candidates, key=lambda row: row["_mse"])
        smallest = min((row for row in candidates if row["_param"] is not None), key=lambda row: row["_param"])
        delta = (ours["_mse"] - best["_mse"]) / best["_mse"] * 100
        wins += int(ours["_mse"] < best["_mse"])
        comparison_body.append(
            [
                ours["dataset"],
                f"{ours['seq_len']}→{ours['pred_len']}",
                f"{ours['_mse']:.6f} / {ours['_mae']:.6f}",
                f"{best['model']} {best['_mse']:.6f}",
                f"{delta:+.2f}%",
                f"{ours['_param']:,}",
                f"{best['_param']:,}",
                f"{smallest['model']} {smallest['_param']:,}",
            ]
        )
    common_count = len(ours_rows)
    lines.append(
        f"Compact + Echo 在 {common_count} 个已审计共同单元中严格优于当格最佳 "
        f"baseline **{wins}/{common_count}** 次。`ΔMSE<0` 表示 Compact + Echo 更好。"
    )
    lines.append("")
    table(
        lines,
        ["Dataset", "L→H", "Compact + Echo MSE/MAE", "Best baseline MSE", "ΔMSE", "Ours params", "Best params", "Smallest baseline params"],
        comparison_body,
    )

    lines.extend(["", "### Compact + Echo 与每个 baseline 的共同单元汇总", ""])
    pairwise_body = []
    for model in BASELINE_MODELS:
        pairs = []
        for ours in ours_rows:
            other = index[(model, ours["dataset"], ours["seq_len"], ours["pred_len"])]
            if other["_mse"] is not None:
                pairs.append((ours, other))
        pairwise_body.append(
            [
                model,
                str(len(pairs)),
                str(sum(ours["_mse"] < other["_mse"] for ours, other in pairs)),
                f"{statistics.mean(ours['_mse'] for ours, _ in pairs):.6f}",
                f"{statistics.mean(other['_mse'] for _, other in pairs):.6f}",
                f"{statistics.mean((ours['_mse'] - other['_mse']) / other['_mse'] * 100 for ours, other in pairs):+.2f}%",
            ]
        )
    table(lines, ["Baseline", "Common n", "Ours wins", "Ours mean MSE", "Baseline mean MSE", "Mean relative Δ"], pairwise_body)

    lines.extend(
        [
            "",
            "## 4. 完整逐数据集结果",
            "",
            "`Avg*` 是该模型在本表可用 horizon 上的均值，方括号给出覆盖数；缺失不参与均值，所以部分覆盖的 Avg* 不能与 4/4 模型直接排名。",
            "",
        ]
    )
    for seq_len in SEQ_LENS:
        lines.extend([f"## L={seq_len}", ""])
        for dataset in DATASETS:
            lines.extend([f"### {dataset} — L={seq_len}", "", "MSE / MAE / MAPE：", ""])
            accuracy_body: list[list[str]] = []
            for horizon in HORIZONS[dataset]:
                accuracy_body.append(
                    [str(horizon)]
                    + [metric_cell(index[(model, dataset, seq_len, horizon)]) for model in MODEL_ORDER]
                )
            avg_row = ["Avg*"]
            for model in MODEL_ORDER:
                subset = [index[(model, dataset, seq_len, horizon)] for horizon in HORIZONS[dataset]]
                present = [row for row in subset if row["_mse"] is not None]
                if not present:
                    avg_row.append("N/A [0/4]")
                else:
                    mape_values = [row["_mape"] for row in present if row["_mape"] is not None]
                    avg_mape = statistics.mean(mape_values) if mape_values else None
                    avg_row.append(
                        f"{statistics.mean(row['_mse'] for row in present):.6f} / "
                        f"{statistics.mean(row['_mae'] for row in present):.6f} / "
                        f"{fmt_metric(avg_mape)} [{len(present)}/4]"
                    )
            accuracy_body.append(avg_row)
            table(lines, ["H"] + list(MODEL_ORDER), accuracy_body)
            lines.extend(["", "Trainable parameters：", ""])
            parameter_body = []
            for horizon in HORIZONS[dataset]:
                parameter_body.append(
                    [str(horizon)]
                    + [parameter_cell(index[(model, dataset, seq_len, horizon)]) for model in MODEL_ORDER]
                )
            table(lines, ["H"] + list(MODEL_ORDER), parameter_body)
            lines.append("")

    lines.extend(
        [
            "## 5. 原始缺口与 0716v2 补跑策略",
            "",
            f"0716v2 原始矩阵共有 **255** 个缺失/失败单元，其中 **183** 个属于 baselines，**72** 个属于旧 ours 分支。当前只把 **{completion_count}/183** 个 baseline 严格产物并入本表；旧 ours 结果不会改名成 Compact + Echo。",
            "",
            "| Model | L | Dataset | H | 原始缺口 | 0716v2 重跑配置 |",
            "|---|---:|---|---|---|---|",
            (
                "| Compact + Echo | 96,720 | 已补齐全部 production-route 单元 | 依数据集而定 | 0 cells | 80 个原缺口已按当前 `OurModel` 三种子补齐；旧 ours 分支的 72 格未复用 |"
                if ours_full
                else "| Compact + Echo | 96,720 | 全部未锁定 production-route 单元 | 依数据集而定 | 80 cells | 需要运行当前 `OurModel`；旧 ours 分支的 72 格不复用 |"
            ),
            "| PhaseFormer | 96,720 | 除 weather/electricity 外 9 个数据集 | 各 4 个 | 72 cells | 官方数据集配置优先；PEMS 标记为扩展配置（period=12） |",
            "| FreqCycle | 96,720 | 除 weather/electricity 外 9 个数据集 | 各 4 个 | 72 cells | official-port 配置优先；PEMS 标记为扩展配置（window/stride=12） |",
            "| SparseTSF | 96,720 | PEMS03/04/07/08 | 12 | incompatible（8 cells） | period_len 从 24 改为 12，单独标记兼容配置 |",
            "| FreTS | 720 | electricity/traffic/PEMS03/PEMS07 | 见长表 | 11 cells：incomplete/OOM/not-run | batch_size=1，保持其余锁定结构，降低 activation 峰值 |",
            "| PatchTST | 720 | electricity/traffic/PEMS03/04/07/08 | 见长表 | 20 cells：incomplete/not-run | 锁定 patch_len=16、stride=8、d_model=128，保守 batch |",
            "",
            "`†` 参数并非猜测：FreTS 取同一 L/H 锁定结构在其他完成数据集的唯一参数量；PatchTST 按锁定 `patch_len=16, stride=8, d_model=128, padding=end` 的精确 head 公式，并由已完成单元反校验。",
            "",
            "## 6. L=720 本地复现 / 已发表参考 MSE",
            "",
            "以下仅保留现有报告已抄录的 [SparseTSF 论文 Tables 10–11](https://proceedings.mlr.press/v235/lin24n.html) 参考列。单元格式 `本地复现 / published`；它们来自不同代码与协议，published 值不进入上面的本地排名，也不拿 published MSE 拼接本地 MAE。",
            "",
        ]
    )
    for dataset in PUBLISHED_DATASETS:
        lines.extend([f"### {dataset} — L=720 reference", ""])
        body = []
        for horizon in HORIZONS[dataset]:
            cells = [str(horizon)]
            for model in PUBLISHED_MODELS:
                row = index[(model, dataset, 720, horizon)]
                local = "N/A" if row["_mse"] is None else f"{row['_mse']:.6f}"
                paper = row["published_reference_mse"] or "N/A"
                cells.append(f"{local} / {paper}")
            body.append(cells)
        table(lines, ["H"] + list(PUBLISHED_MODELS), body)
        lines.append("")

    lines.extend(
        [
            "## 7. 复现实用字段与来源审计",
            "",
            "长表 CSV 额外保存了：RMSE（由 test MSE 开方）、registered/forward-active/real-scalar-equivalent 参数量、FP32 参数体积估计、seed、训练秒数、推理秒数、严格测试 wall-time、单样本 forward latency、test peak GPU memory、协议、状态、参数恢复来源、published-reference MSE 与缺失原因。",
            "",
            f"- L=96 独立四位小数汇总交叉核对：**{sl96_audit_count} cells，0 mismatch**。",
            f"- `note/results.csv` SHA256: `{sha256(inputs['baseline'])}`",
            f"- `note/sl96_results.csv` SHA256: `{sha256(inputs['sl96'])}`",
            f"- `{inputs['ours']}` SHA256: `{sha256(inputs['ours'])}`",
            f"- `ASY2ECHO_STACKS_MIXLINEAR_PHASEFORMER_0716V1_RESULTS.csv` SHA256: `{sha256(inputs['phaseformer'])}`",
            f"- `outputs/freqcycle_official/freqcycle_0716v1_results.csv` SHA256: `{sha256(inputs['freqcycle'])}`",
            f"- `RESULTS.md` SHA256: `{sha256(inputs['report'])}`",
            (
                f"- `COMPLETION_MATRIX_0716V2_RESULTS.csv` SHA256: `{sha256(inputs['completion'])}`"
                if "completion" in inputs
                else "- `COMPLETION_MATRIX_0716V2_RESULTS.csv`: not supplied"
            ),
            "",
            "### 参数统计语义",
            "",
            "- `param_count` 与训练日志一致，是框架登记的 trainable tensor elements。",
            "- FITS 参数均为 complex64，因此 `real_scalar_equivalent = 2 × param_count`。",
            "- MixLinear 只对两个频域线性层的 complex64 entries 加倍；其余参数按实数计。",
            "- FreqCycle 同时报告 registered 与 forward-active：registered 是可训练 state-dict 总量，active 排除了官方实现中登记但 forward 未使用的重复 MLP；主参数列仍以 registered 为准。",
            "- `fp32_model_mib = real_scalar_equivalent × 4 / 2^20`，仅估算参数存储，不代表训练峰值显存；优化器状态与 activations 可能占主导。",
            "",
            "### MAPE 覆盖边界",
            "",
            "旧 baseline 代码虽然运行时可计算 MAPE，但历史日志和汇总 CSV 只保存 MSE/MAE，`pred.npy`/`true.npy` 也未落盘，因此旧单元不能从现有汇总反推 MAPE。0716v2 baseline 与 0717v2 Ours 新补跑按历史无 epsilon 定义流式累计 MAPE；旧单元若也要 MAPE，仍需从对应 checkpoint 重新执行完整 held-out test inference。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_outputs(rows: list[dict[str, Any]], report: str) -> None:
    expected_rows = 1144
    if len(rows) != expected_rows:
        raise AssertionError(f"Output grid has {len(rows)} rows, expected {expected_rows}")
    if {row["model"] for row in rows} != set(MODEL_ORDER):
        raise AssertionError("Output contains an unexpected model name")
    completion_count = sum(row["source_type"] == "local_completion_strict" for row in rows)
    ours_metric_count = sum(
        row["_mse"] is not None for row in rows if row["model"] == OURS_NAME
    )
    expected_metric_count = 873 + ours_metric_count + completion_count
    if sum(row["_mse"] is not None for row in rows) != expected_metric_count:
        raise AssertionError("Unexpected total MSE coverage")
    if sum(row["_mae"] is not None for row in rows) != expected_metric_count:
        raise AssertionError("Unexpected total MAE coverage")
    ours_mape_count = sum(
        row["_mape"] is not None for row in rows if row["model"] == OURS_NAME
    )
    if sum(row["_mape"] is not None for row in rows) != completion_count + ours_mape_count:
        raise AssertionError("Unexpected MAPE coverage")
    if any(
        row["_mape"] is not None
        and row["source_type"] not in {
            "local_completion_strict",
            "local_compact_echo_locked_3seed_mean",
        }
        for row in rows
    ):
        raise AssertionError("MAPE may only come from audited completion rows")
    parameter_count = sum(row["_param"] is not None for row in rows)
    if parameter_count < 912:
        raise AssertionError("Unexpected parameter coverage regression")
    if completion_count == 183:
        expected_complete_count = 1064 if ours_metric_count == 8 else 1144
        if expected_metric_count != expected_complete_count or parameter_count != expected_complete_count:
            raise AssertionError("Complete audit coverage is internally inconsistent")
    forbidden = ("AsySpecX", "JointMLP")
    if any(name in report for name in forbidden):
        raise AssertionError("Excluded model name leaked into the generated report")
    serialized_rows = "\n".join(
        ",".join(str(row[field]) for field in CSV_FIELDS) for row in rows
    )
    if any(name in serialized_rows for name in forbidden):
        raise AssertionError("Excluded model name leaked into the generated CSV rows")


def main() -> None:
    args = parse_args()
    inputs = {
        "baseline": args.baseline_results,
        "sl96": args.sl96_audit,
        "ours": args.ours_results,
        "phaseformer": args.phaseformer_results,
        "freqcycle": args.freqcycle_results,
        "report": args.legacy_report,
    }
    if args.completion_results is not None:
        inputs["completion"] = args.completion_results
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} input: {path}")

    published = parse_published_l720(args.legacy_report)
    completion_rows = (
        read_csv(args.completion_results) if args.completion_results is not None else []
    )
    rows = build_rows(
        read_csv(args.baseline_results),
        read_csv(args.ours_results),
        read_csv(args.phaseformer_results),
        read_csv(args.freqcycle_results),
        completion_rows,
        published,
        ours_source_path=args.ours_source_path,
    )
    sl96_count, _ = audit_sl96(rows, args.sl96_audit)
    coverage = coverage_records(rows)
    report = render_report(
        rows,
        coverage,
        inputs,
        sl96_count,
        report_version_override=args.report_version,
        extra_method_note=args.extra_method_note,
    )
    validate_outputs(rows, report)
    write_long_csv(args.output_csv, rows)
    write_coverage_csv(args.coverage_csv, coverage)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_csv} ({len(rows)} rows)")
    print(f"wrote {args.coverage_csv} ({len(coverage)} rows)")


if __name__ == "__main__":
    main()
