"""Chunked compact metadata storage. No full prediction tensor by default."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np


FORMAT_VERSION = 1


class CompactMetaWriter:
    """Write compressed batch-local rows as independent NPZ parts."""

    def __init__(
        self,
        output_dir: str,
        feature_names: Sequence[str],
        expert_names: Sequence[str],
        metadata: Mapping[str, object],
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = list(feature_names)
        self.expert_names = list(expert_names)
        self.metadata = dict(metadata)
        self.parts: List[Dict[str, object]] = []
        self.rows = 0

    def write(self, **arrays: np.ndarray) -> None:
        if "features" not in arrays:
            raise ValueError("compact meta part requires a features array")
        n_rows = int(np.asarray(arrays["features"]).shape[0])
        if n_rows == 0:
            return
        for name, value in arrays.items():
            if np.asarray(value).shape[0] != n_rows:
                raise ValueError(f"part array {name!r} has inconsistent row count")
        part_name = f"part-{len(self.parts):06d}.npz"
        part_path = self.output_dir / part_name
        np.savez_compressed(part_path, **arrays)
        self.parts.append({"file": part_name, "rows": n_rows})
        self.rows += n_rows

    def close(self) -> str:
        payload = {
            "format": "asyspecx_saferoute_compact_npz",
            "format_version": FORMAT_VERSION,
            "feature_names": self.feature_names,
            "expert_names": self.expert_names,
            "rows": self.rows,
            "parts": self.parts,
            **self.metadata,
        }
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return str(manifest_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()


class CompactMetaDataset:
    def __init__(self, path: str):
        source = Path(path)
        manifest_path = source if source.name == "manifest.json" else source / "manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format") != "asyspecx_saferoute_compact_npz":
            raise ValueError(f"unsupported compact meta format in {manifest_path}")
        self.root = manifest_path.parent
        self.feature_names = list(self.manifest["feature_names"])
        self.expert_names = list(self.manifest["expert_names"])

    def iter_parts(self) -> Iterator[Dict[str, np.ndarray]]:
        for part in self.manifest.get("parts", []):
            path = self.root / part["file"]
            with np.load(path, allow_pickle=False) as loaded:
                yield {name: loaded[name] for name in loaded.files}

    def load_all(self, max_rows: int = 0) -> Dict[str, np.ndarray]:
        buckets: Dict[str, List[np.ndarray]] = {}
        total = 0
        for part in self.iter_parts():
            n = len(part["features"])
            take = n if max_rows <= 0 else min(n, max_rows - total)
            if take <= 0:
                break
            for name, value in part.items():
                buckets.setdefault(name, []).append(value[:take])
            total += take
        return {name: np.concatenate(values, axis=0) for name, values in buckets.items()}


def expand_meta_paths(text: str) -> List[str]:
    """Resolve comma-separated directories/files/globs deterministically."""

    import glob

    paths: List[str] = []
    for token in (piece.strip() for piece in text.split(",")):
        if not token:
            continue
        matches = sorted(glob.glob(token))
        paths.extend(matches or [token])
    if not paths:
        raise ValueError("no compact metadata paths were provided")
    return paths


def assert_training_metadata_safe(datasets: Iterable[CompactMetaDataset]) -> None:
    """Hard guard: router fitting must never consume test-labelled metadata."""

    for dataset in datasets:
        split = str(dataset.manifest.get("split", "")).lower()
        source = str(dataset.manifest.get("router_meta_source", "")).lower()
        if split == "test" or source.startswith("test"):
            raise ValueError(
                f"refusing to train router on test-labelled metadata: {dataset.root}"
            )
