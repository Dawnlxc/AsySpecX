#!/usr/bin/env python3
"""Train explicit expert-advantage routers with purged OOF LCB calibration."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.io import expand_meta_paths
from router.training import train_router_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--calibration_meta", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--anchor_expert", default="anchor")
    parser.add_argument("--router_backend", choices=["xgboost", "hist_gradient_boosting", "logistic_best_expert"], default="xgboost")
    parser.add_argument("--router_scope", choices=["cell", "dataset", "family", "global"], default="cell")
    parser.add_argument("--router_target", choices=["advantage", "log_relative_regret"], default="advantage")
    parser.add_argument("--router_min_samples", type=int, default=256)
    parser.add_argument("--router_cv_folds", type=int, default=4)
    parser.add_argument("--router_purge_steps", type=int, default=0)
    parser.add_argument("--router_confidence_alpha", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--random_state", type=int, default=2024)
    args = parser.parse_args()

    paths = expand_meta_paths(args.meta)
    calibration = expand_meta_paths(args.calibration_meta) if args.calibration_meta else None
    bundle, records = train_router_bundle(
        paths,
        anchor_expert=args.anchor_expert,
        backend=args.router_backend,
        router_scope=args.router_scope,
        target=args.router_target,
        min_samples=args.router_min_samples,
        cv_folds=args.router_cv_folds,
        purge_steps=args.router_purge_steps,
        confidence_alpha=args.router_confidence_alpha,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        calibration_paths=calibration,
    )
    output = Path(args.output_dir)
    bundle.save(str(output))
    with (output / "training_models.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model_key", "rows", "oof_rows", "oof_mae", "calibration_quantile"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "router_backend": args.router_backend,
        "router_scope": args.router_scope,
        "router_target": args.router_target,
        "models": len(records),
        "training_meta": paths,
        "calibration_meta": calibration or [],
        "router_protocol": bundle.metadata["router_protocol"],
        "test_labels_used": False,
    }
    with (output / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
