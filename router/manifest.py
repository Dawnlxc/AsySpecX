"""Strict manifest handling for frozen Phase 9 experts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


REQUIRED_CELL_FIELDS = ("dataset", "seq_len", "pred_len", "enc_in")


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    arm: str
    checkpoints: Mapping[str, str]
    config: Mapping[str, object]

    def checkpoint_for(self, seed: int | str) -> str:
        key = str(seed)
        if key not in self.checkpoints:
            raise KeyError(f"expert {self.name!r} has no checkpoint for seed {key}")
        return self.checkpoints[key]


@dataclass(frozen=True)
class ExpertManifest:
    path: str
    anchor_name: str
    experts: Sequence[ExpertSpec]
    cell: Mapping[str, object]

    @property
    def names(self) -> List[str]:
        return [expert.name for expert in self.experts]

    @property
    def anchor_index(self) -> int:
        return self.names.index(self.anchor_name)

    @property
    def anchor(self) -> ExpertSpec:
        return self.experts[self.anchor_index]


def _expand_path(value: str, base_dir: Path) -> str:
    value = os.path.expandvars(os.path.expanduser(value))
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _normalise_checkpoints(raw: object, base_dir: Path) -> Dict[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("expert checkpoints must be a non-empty seed-to-path mapping")
    return {str(seed): _expand_path(str(path), base_dir) for seed, path in raw.items()}


def _cell_from_config(config: Mapping[str, object]) -> Dict[str, object]:
    dataset = config.get("dataset", config.get("dataset_key", ""))
    return {
        "dataset": str(dataset),
        "seq_len": int(config.get("seq_len", 0)),
        "pred_len": int(config.get("pred_len", 0)),
        "enc_in": int(config.get("enc_in", 0)),
    }


def verify_manifest_compatibility(manifest: ExpertManifest) -> None:
    """Verify that every expert targets the same forecasting cell."""

    expected = {key: manifest.cell.get(key) for key in REQUIRED_CELL_FIELDS}
    missing = [key for key, value in expected.items() if value in (None, "", 0)]
    if missing:
        raise ValueError(f"manifest cell is missing required fields: {', '.join(missing)}")
    for expert in manifest.experts:
        actual = _cell_from_config(expert.config)
        for key in REQUIRED_CELL_FIELDS:
            want = str(expected[key])
            got = str(actual[key])
            if got != want:
                raise ValueError(
                    f"expert {expert.name!r} config mismatch for {key}: {got!r} != {want!r}"
                )
        if str(expert.config.get("model", "AsySpecX")) != "AsySpecX":
            raise ValueError(f"expert {expert.name!r} is not an AsySpecX checkpoint")


def verify_checkpoint_files(
    manifest: ExpertManifest,
    seeds: Iterable[int | str],
) -> None:
    missing: List[str] = []
    for expert in manifest.experts:
        for seed in seeds:
            try:
                path = expert.checkpoint_for(seed)
            except KeyError as exc:
                missing.append(str(exc))
                continue
            if not os.path.isfile(path):
                missing.append(f"{expert.name}[seed={seed}]: {path}")
    if missing:
        raise FileNotFoundError("missing expert checkpoints:\n  " + "\n  ".join(missing))


def load_expert_manifest(
    path: str,
    anchor_expert: str = "anchor",
    expert_names: Optional[Iterable[str]] = None,
    seeds: Optional[Iterable[int | str]] = None,
    require_checkpoints: bool = True,
) -> ExpertManifest:
    """Load and strictly validate a frozen-expert manifest.

    ``expert_names`` may reduce the pool, but the anchor cannot be removed.
    Missing requested experts or checkpoints are hard errors.
    """

    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    raw_experts = raw.get("experts")
    if not isinstance(raw_experts, list) or not raw_experts:
        raise ValueError("manifest must contain a non-empty 'experts' list")

    requested = None if expert_names is None else [str(name) for name in expert_names]
    if requested is not None and anchor_expert not in requested:
        requested.insert(0, anchor_expert)

    seen = set()
    experts: List[ExpertSpec] = []
    for item in raw_experts:
        if not isinstance(item, Mapping):
            raise ValueError("every expert entry must be an object")
        name = str(item.get("name", ""))
        arm = str(item.get("arm", ""))
        if not name or not arm:
            raise ValueError("expert entries require non-empty name and arm")
        if name in seen:
            raise ValueError(f"duplicate expert name {name!r}")
        seen.add(name)
        if requested is not None and name not in requested:
            continue
        config = item.get("config", {})
        if not isinstance(config, Mapping):
            raise ValueError(f"expert {name!r} config must be an object")
        experts.append(
            ExpertSpec(
                name=name,
                arm=arm,
                checkpoints=_normalise_checkpoints(item.get("checkpoints"), manifest_path.parent),
                config=dict(config),
            )
        )

    if requested is not None:
        absent = sorted(set(requested) - {expert.name for expert in experts})
        if absent:
            raise ValueError(f"requested experts missing from manifest: {', '.join(absent)}")
    if anchor_expert not in {expert.name for expert in experts}:
        raise ValueError(f"anchor expert {anchor_expert!r} is missing")
    experts = [next(expert for expert in experts if expert.name == anchor_expert)] + [
        expert for expert in experts if expert.name != anchor_expert
    ]

    cell = raw.get("cell")
    if not isinstance(cell, Mapping):
        cell = _cell_from_config(experts[0].config)
    manifest = ExpertManifest(
        path=str(manifest_path),
        anchor_name=anchor_expert,
        experts=tuple(experts),
        cell=dict(cell),
    )
    verify_manifest_compatibility(manifest)
    if seeds is not None and require_checkpoints:
        verify_checkpoint_files(manifest, seeds)
    return manifest


def verify_sample_alignment(reference_ids, candidate_ids, expert_name: str = "expert") -> None:
    """Fail on any sample-id or order mismatch across experts."""

    ref = list(reference_ids)
    got = list(candidate_ids)
    if ref != got:
        first = next((i for i, pair in enumerate(zip(ref, got)) if pair[0] != pair[1]), None)
        if first is None:
            first = min(len(ref), len(got))
        raise ValueError(
            f"sample alignment mismatch for {expert_name} at index {first}: "
            f"reference_len={len(ref)} candidate_len={len(got)}"
        )
