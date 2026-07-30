#!/usr/bin/env python3
"""Build one strict frozen-expert manifest for a Phase 9 forecasting cell."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.configs import LOCKED_EXPERTS, checkpoint_path, discover_cached_periods, expert_config


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--expert_seeds", default="2024,2025,2026")
    parser.add_argument("--experts", default="anchor,dlinear,split_clip,individual_revin,individual_period")
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--data_root", default="dataset")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    data_root = Path(args.data_root).resolve()
    seeds = [int(seed) for seed in parse_csv(args.expert_seeds)]
    names = parse_csv(args.experts)
    names = ["anchor"] + [name for name in names if name != "anchor"]
    unknown = sorted(set(names) - set(LOCKED_EXPERTS))
    if unknown:
        raise SystemExit(f"unknown expert names: {', '.join(unknown)}")
    auto_periods = discover_cached_periods(repo_root, args.dataset, args.seq_len)

    experts = []
    missing = []
    for name in names:
        arm = LOCKED_EXPERTS[name]
        checkpoints = {}
        for seed in seeds:
            path = checkpoint_path(
                checkpoint_root, arm, args.dataset, args.seq_len, args.pred_len, seed
            ).resolve()
            checkpoints[str(seed)] = str(path)
            if not path.is_file():
                missing.append(f"{name}[{seed}]={path}")
        experts.append(
            {
                "name": name,
                "arm": arm,
                "checkpoints": checkpoints,
                "config": expert_config(
                    name,
                    args.dataset,
                    args.seq_len,
                    args.pred_len,
                    data_root,
                    auto_periods,
                ),
            }
        )
    if missing and args.strict:
        raise SystemExit("missing requested checkpoints:\n  " + "\n  ".join(missing))

    payload = {
        "format": "asyspecx_phase9_expert_manifest",
        "cell": {
            "dataset": args.dataset,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "enc_in": experts[0]["config"]["enc_in"],
        },
        "anchor_expert": "anchor",
        "expert_seeds": seeds,
        "experts": experts,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"manifest={output} experts={len(experts)} seeds={len(seeds)} missing={len(missing)}")


if __name__ == "__main__":
    main()
