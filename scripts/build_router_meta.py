#!/usr/bin/env python3
"""Stream frozen experts and write compact sample-block metadata."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.blocks import horizon_blocks
from router.features import train_only_channel_groups
from router.io import CompactMetaWriter
from router.manifest import load_expert_manifest
from router.pipeline import compact_meta_batch
from router.runtime import FrozenExpertPool, FullPredictionMemmaps, make_ordered_loader


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_periods(value):
    return [int(item) for item in str(value).replace("+", ",").split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_manifest", required=True)
    parser.add_argument("--anchor_expert", default="anchor")
    parser.add_argument("--expert_seeds", default="2024,2025,2026")
    parser.add_argument("--experts", default="")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--router_meta_source", choices=["val", "rolling_oof", "test_analysis"], default="val")
    parser.add_argument("--router_num_horizon_blocks", type=int, default=4)
    parser.add_argument("--router_scope", choices=["cell", "dataset", "family", "global"], default="cell")
    parser.add_argument("--router_forecast_snippet_bins", type=int, default=0)
    parser.add_argument("--max_feature_channels", type=int, default=64)
    parser.add_argument("--router_channel_groups", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expert_device_policy", choices=["one_at_a_time", "resident"], default="one_at_a_time")
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_full_predictions", type=int, choices=[0, 1], default=0)
    parser.add_argument("--full_prediction_dir", default="")
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    if args.router_channel_groups < 1:
        raise SystemExit("router_channel_groups must be at least 1")

    if args.router_meta_source == "val" and args.split != "val":
        raise SystemExit("router_meta_source=val requires --split val")
    if args.router_meta_source == "test_analysis" and args.split != "test":
        raise SystemExit("router_meta_source=test_analysis requires --split test")
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output already exists: {output}; pass --overwrite 1")
        shutil.rmtree(output)

    seeds = parse_csv(args.expert_seeds)
    selected = parse_csv(args.experts) or None
    manifest = load_expert_manifest(
        args.expert_manifest,
        anchor_expert=args.anchor_expert,
        expert_names=selected,
        seeds=seeds,
        require_checkpoints=True,
    )
    anchor_config = manifest.anchor.config
    channel_groups = None
    group_metadata = {}
    if args.router_channel_groups > 1:
        train_dataset, _ = make_ordered_loader(
            anchor_config,
            "train",
            batch_size=args.batch_size,
            num_workers=0,
        )
        channel_groups, descriptor_names = train_only_channel_groups(
            np.asarray(train_dataset.data_x), args.router_channel_groups
        )
        group_metadata = {
            "router_channel_groups": int(args.router_channel_groups),
            "channel_group_sizes": [
                int(np.sum(channel_groups == group))
                for group in range(args.router_channel_groups)
            ],
            "channel_group_descriptor_names": descriptor_names,
            "channel_group_assignment_source": "train_only",
            "channel_group_assignments": channel_groups.tolist(),
        }
    dataset, loader = make_ordered_loader(
        anchor_config,
        args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    pool = FrozenExpertPool(
        manifest,
        seeds=seeds,
        device=args.device,
        device_policy=args.expert_device_policy,
    )
    blocks = horizon_blocks(int(manifest.cell["pred_len"]), args.router_num_horizon_blocks)
    periods = parse_periods(anchor_config.get("periods", anchor_config.get("period", "")))
    memmaps = None
    if args.save_full_predictions:
        full_root = args.full_prediction_dir or str(output / "full_predictions")
        memmaps = FullPredictionMemmaps(
            full_root,
            manifest.names,
            (len(dataset), int(manifest.cell["pred_len"]), int(manifest.cell["enc_in"])),
        )

    writer = None
    sample_offset = 0
    for batch in loader:
        batch_x, batch_y = batch[0], batch[1]
        predictions, variances = pool.predict(batch_x)
        target = batch_y[:, -int(manifest.cell["pred_len"]) :, :].float().cpu().numpy()
        sample_ids = np.arange(sample_offset, sample_offset + len(batch_x), dtype=np.int64)
        arrays, feature_names = compact_meta_batch(
            batch_x.float().cpu().numpy(),
            predictions,
            variances,
            sample_ids,
            blocks,
            args.anchor_expert,
            periods,
            str(manifest.cell["dataset"]),
            args.router_scope,
            target=target,
            snippet_bins=args.router_forecast_snippet_bins,
            max_channels=args.max_feature_channels,
            channel_groups=channel_groups,
        )
        if writer is None:
            writer = CompactMetaWriter(
                str(output),
                feature_names,
                manifest.names,
                {
                    "split": args.split,
                    "router_meta_source": args.router_meta_source,
                    "labelled": True,
                    "dataset": manifest.cell["dataset"],
                    "seq_len": int(manifest.cell["seq_len"]),
                    "pred_len": int(manifest.cell["pred_len"]),
                    "enc_in": int(manifest.cell["enc_in"]),
                    "num_horizon_blocks": len(blocks),
                    "horizon_blocks": blocks,
                    "anchor_expert": args.anchor_expert,
                    "expert_seeds": seeds,
                    "full_predictions_saved": bool(args.save_full_predictions),
                    "normalization_space": "dataset_standardized",
                    **group_metadata,
                },
            )
        writer.write(**arrays)
        if memmaps is not None:
            memmaps.write(sample_offset, predictions, variances, target)
        sample_offset += len(batch_x)
    if writer is None:
        raise SystemExit("ordered loader yielded no samples")
    meta_manifest = writer.close()
    if memmaps is not None:
        memmaps.flush()
    print(
        json.dumps(
            {
                "meta_manifest": meta_manifest,
                "samples": sample_offset,
                "rows": sample_offset * len(blocks),
                "experts": manifest.names,
                "full_predictions_saved": bool(args.save_full_predictions),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
