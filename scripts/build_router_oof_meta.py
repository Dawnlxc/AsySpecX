#!/usr/bin/env python3
"""Train chronology-safe fold experts and build rolling-OOF compact metadata."""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.AsySpecX import Model as AsySpecXModel
from router.blocks import horizon_blocks
from router.configs import MANUAL_PERIODS
from router.io import CompactMetaWriter
from router.manifest import load_expert_manifest
from router.pipeline import compact_meta_batch
from router.runtime import make_ordered_loader, namespace_from_config
from router.splits import rolling_oof_windows


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_id = int(self.indices[index])
        return (sample_id,) + tuple(self.dataset[sample_id])


class FoldScaledDataset(Dataset):
    """View a train dataset through a scaler fitted on one OOF prefix only."""

    def __init__(self, base, scaled_values):
        self.base = base
        self.data_x = np.asarray(scaled_values, dtype=np.float32)
        self.data_y = self.data_x
        self.seq_len = int(base.seq_len)
        self.label_len = int(base.label_len)
        self.pred_len = int(base.pred_len)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = list(self.base[index])
        start = int(index)
        context_end = start + self.seq_len
        target_start = context_end - self.label_len
        target_end = target_start + self.label_len + self.pred_len
        item[0] = self.data_x[start:context_end]
        item[1] = self.data_y[target_start:target_end]
        return tuple(item)


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_periods(value):
    return [int(item) for item in str(value).replace("+", ",").split(",") if item.strip()]


def fold_periods(dataset, train_indices, seq_len, fallback):
    try:
        from scripts.discover_periods import discover_acf

        prefix_end = min(len(dataset.data_x), int(train_indices[-1]) + int(seq_len))
        series = np.asarray(dataset.data_x[:prefix_end], dtype=np.float64)
        periods, _ = discover_acf(series, 4, min(int(seq_len), 672), 3)
        return (periods or fallback), ("train_prefix_auto_acf" if periods else "manual_fallback_empty")
    except Exception as exc:
        print(
            f"[warn] fold auto-ACF failed; using train-independent manual periods: {exc}",
            file=sys.stderr,
        )
        return fallback, f"manual_fallback_{type(exc).__name__}"


def optimizer_for(model, config):
    base_lr = float(config.get("learning_rate", 5e-4))
    gate_mult = float(config.get("gate_lr_mult", 1.0))
    clip_mult = float(config.get("clip_lr_mult", 1.0))
    gate_words = ("gate_logit", "gate_logits", "global_gate_logit", "local_gate_logits")
    groups = {"base": [], "gate": [], "clip": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "clip_logit" in name:
            groups["clip"].append(parameter)
        elif any(word in name for word in gate_words):
            groups["gate"].append(parameter)
        else:
            groups["base"].append(parameter)
    params = []
    if groups["base"]:
        params.append({"params": groups["base"], "lr": base_lr})
    if groups["gate"]:
        params.append({"params": groups["gate"], "lr": base_lr * gate_mult})
    if groups["clip"]:
        params.append({"params": groups["clip"], "lr": base_lr * clip_mult})
    return torch.optim.Adam(params, lr=base_lr)


def train_expert(config, train_dataset, device, epochs, batch_size, workers, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = AsySpecXModel(namespace_from_config(config)).float().to(device)
    optimizer = optimizer_for(model, config)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    pred_len = int(config["pred_len"])
    for _ in range(epochs):
        for batch in loader:
            batch_x, batch_y = batch[0].float().to(device), batch[1].float().to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            if isinstance(output, dict):
                output = output["pred"]
            target = batch_y[:, -pred_len:, :]
            loss = torch.mean((output[:, -pred_len:, :] - target) ** 2)
            if hasattr(model, "extra_loss"):
                extra = model.extra_loss()
                if extra is not None:
                    loss = loss + extra
            loss.backward(); optimizer.step()
    model.eval(); model.to("cpu")
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return model


def predict_models(models, configs, batch_x, device, device_policy):
    x = batch_x.float().to(device)
    predictions, variances = {}, {}
    with torch.no_grad():
        for name, model in models.items():
            if device_policy == "one_at_a_time":
                model.to(device)
            output = model(
                x,
                eval_residual_part=str(configs[name].get("eval_residual_part", "default")),
            )
            if isinstance(output, dict):
                output = output["pred"]
            output = output.detach().float()
            if not torch.isfinite(output).all():
                raise FloatingPointError(f"non-finite rolling-OOF prediction from {name}")
            prediction = output.cpu().numpy()
            predictions[name] = prediction
            variances[name] = np.zeros_like(prediction, dtype=np.float32)
            if device_policy == "one_at_a_time":
                model.to("cpu")
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
    return predictions, variances


def convert_scale(values, source_scaler, destination_scaler):
    array = np.asarray(values, dtype=np.float64)
    shape = array.shape
    flat = array.reshape(-1, shape[-1])
    raw = source_scaler.inverse_transform(flat)
    return destination_scaler.transform(raw).reshape(shape).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_manifest", required=True)
    parser.add_argument("--anchor_expert", default="anchor")
    parser.add_argument("--experts", default="")
    parser.add_argument("--router_oof_seed", type=int, default=2024)
    parser.add_argument("--router_num_horizon_blocks", type=int, default=4)
    parser.add_argument("--router_scope", choices=["cell", "dataset", "family", "global"], default="cell")
    parser.add_argument("--router_purge_steps", type=int, default=0)
    parser.add_argument("--oof_epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--expert_device_policy",
        choices=["one_at_a_time", "resident"],
        default="one_at_a_time",
    )
    parser.add_argument("--max_feature_channels", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {output}")
        shutil.rmtree(output)
    selected = parse_csv(args.experts) or None
    manifest = load_expert_manifest(
        args.expert_manifest,
        anchor_expert=args.anchor_expert,
        expert_names=selected,
        require_checkpoints=False,
    )
    dataset, _ = make_ordered_loader(manifest.anchor.config, "train", batch_size=1, num_workers=0)
    if not hasattr(dataset, "scaler"):
        raise SystemExit("rolling OOF requires a dataset scaler")
    canonical_scaler = dataset.scaler
    raw_train_series = canonical_scaler.inverse_transform(
        np.asarray(dataset.data_x, dtype=np.float64)
    )
    pred_len = int(manifest.cell["pred_len"])
    requested_purge = args.router_purge_steps if args.router_purge_steps > 0 else pred_len
    purge = max(requested_purge, pred_len)
    windows = rolling_oof_windows(len(dataset), purge)
    blocks = horizon_blocks(pred_len, args.router_num_horizon_blocks)
    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    writer = None
    fold_records = []

    for fold, (train_indices, val_indices) in enumerate(windows):
        models = {}
        fold_configs = {}
        visible_end = min(
            len(raw_train_series),
            int(train_indices[-1])
            + int(manifest.cell["seq_len"])
            + int(manifest.cell["pred_len"]),
        )
        first_validation_origin = int(val_indices[0]) + int(manifest.cell["seq_len"])
        if visible_end > first_validation_origin:
            raise AssertionError("fold scaler would consume observations beyond validation origin")
        fold_scaler = StandardScaler().fit(raw_train_series[:visible_end])
        fold_dataset = FoldScaledDataset(
            dataset, fold_scaler.transform(raw_train_series)
        )
        # Never fall back to the manifest's full-train auto periods inside an
        # OOF fold. Manual calendar periods are train-independent and safe.
        fallback = list(
            MANUAL_PERIODS.get(
                str(manifest.cell["dataset"]),
                tuple(
                    parse_periods(
                        manifest.anchor.config.get(
                            "periods", manifest.anchor.config.get("period", "24")
                        )
                    )
                ),
            )
        )
        periods, period_source = fold_periods(
            fold_dataset, train_indices, int(manifest.cell["seq_len"]), fallback
        )
        for expert_index, expert in enumerate(manifest.experts):
            config = dict(expert.config)
            if str(config.get("period_mode", "manual")) == "auto_acf":
                config["periods"] = "+".join(str(period) for period in periods)
                config["period"] = periods[0]
            fold_configs[expert.name] = config
            epochs = args.oof_epochs or int(config.get("train_epochs", 30))
            train_subset = Subset(fold_dataset, train_indices.tolist())
            models[expert.name] = train_expert(
                config,
                train_subset,
                device,
                epochs,
                args.batch_size or int(config.get("batch_size", 16)),
                args.num_workers,
                args.router_oof_seed + fold * 1009 + expert_index * 97,
            )
        val_loader = DataLoader(
            IndexedSubset(fold_dataset, val_indices),
            batch_size=args.batch_size or int(manifest.anchor.config.get("batch_size", 16)),
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        )
        if args.expert_device_policy == "resident":
            for model in models.values():
                model.to(device)
        for batch in val_loader:
            sample_ids, batch_x, batch_y = batch[0].numpy(), batch[1], batch[2]
            predictions, variances = predict_models(
                models,
                fold_configs,
                batch_x,
                device,
                args.expert_device_policy,
            )
            context = convert_scale(batch_x.float().numpy(), fold_scaler, canonical_scaler)
            predictions = {
                name: convert_scale(values, fold_scaler, canonical_scaler)
                for name, values in predictions.items()
            }
            target = convert_scale(
                batch_y[:, -pred_len:, :].float().numpy(), fold_scaler, canonical_scaler
            )
            arrays, feature_names = compact_meta_batch(
                context, predictions, variances, sample_ids, blocks,
                args.anchor_expert, periods, str(manifest.cell["dataset"]), args.router_scope,
                target=target, max_channels=args.max_feature_channels,
            )
            if writer is None:
                writer = CompactMetaWriter(
                    str(output), feature_names, manifest.names,
                    {
                        "split": "train",
                        "router_meta_source": "rolling_oof",
                        "labelled": True,
                        "dataset": manifest.cell["dataset"],
                        "seq_len": int(manifest.cell["seq_len"]),
                        "pred_len": pred_len,
                        "enc_in": int(manifest.cell["enc_in"]),
                        "num_horizon_blocks": len(blocks),
                        "horizon_blocks": blocks,
                        "anchor_expert": args.anchor_expert,
                        "expert_seeds": [args.router_oof_seed],
                        "full_predictions_saved": False,
                        "normalization_space": "dataset_standardized",
                        "oof_protocol": "expanding_60_20_then_80_20",
                    },
                )
            writer.write(**arrays)
        for model in models.values():
            model.to("cpu")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        fold_records.append(
            {
                "fold": fold,
                "train_first": int(train_indices[0]),
                "train_last": int(train_indices[-1]),
                "validation_first": int(val_indices[0]),
                "validation_last": int(val_indices[-1]),
                "purge_steps": purge,
                "periods": periods,
                "period_source": period_source,
                "scaler_fit_observations": visible_end,
                "first_validation_origin": first_validation_origin,
            }
        )
        del models
    if writer is None:
        raise SystemExit("rolling OOF produced no rows")
    meta_path = writer.close()
    (output / "oof_folds.json").write_text(json.dumps(fold_records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"meta_manifest": meta_path, "folds": fold_records}, sort_keys=True))


if __name__ == "__main__":
    main()
