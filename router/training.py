"""Small explicit advantage regressors with purged OOF calibration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np

from .features import dataset_family
from .io import CompactMetaDataset, assert_training_metadata_safe
from .safe import calibrate_lcb_quantile
from .splits import PurgedTimeSeriesSplit


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=np.float64)


def _new_model(backend: str, max_depth: int, learning_rate: float, random_state: int):
    if backend == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "router_backend=xgboost requires xgboost; install it or use hist_gradient_boosting"
            ) from exc
        return XGBRegressor(
            n_estimators=300,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=random_state,
            reg_lambda=1.0,
        )
    if backend == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=max_depth,
            learning_rate=learning_rate,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        )
    if backend == "logistic_best_expert":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=500, class_weight="balanced", random_state=random_state)
    raise ValueError(f"unknown router backend {backend!r}")


class AdvantageEstimator:
    """Unifies regressors and the optional sign classifier in advantage units."""

    def __init__(self, backend: str, max_depth: int, learning_rate: float, random_state: int):
        self.backend = backend
        self.model = _new_model(backend, max_depth, learning_rate, random_state)
        self.scale = 1.0
        self.constant: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None):
        y = np.asarray(y, dtype=np.float64)
        if y.size == 0:
            raise ValueError("cannot fit router on zero rows")
        if float(np.std(y)) < 1e-12:
            self.constant = float(np.mean(y))
            return self
        if self.backend == "logistic_best_expert":
            labels = (y > 0.0).astype(np.int64)
            if np.unique(labels).size < 2:
                self.constant = float(np.mean(y))
                return self
            nonzero = np.abs(y[np.abs(y) > 0])
            self.scale = float(np.median(nonzero)) if nonzero.size else 1.0
            self.model.fit(X, labels)
        elif self.backend == "xgboost" and eval_set is not None:
            try:
                self.model.fit(
                    X,
                    y,
                    eval_set=[eval_set],
                    verbose=False,
                    early_stopping_rounds=30,
                )
            except TypeError:
                # Newer xgboost versions move early_stopping_rounds to the estimator.
                self.model.set_params(early_stopping_rounds=30)
                self.model.fit(X, y, eval_set=[eval_set], verbose=False)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(X), self.constant, dtype=np.float64)
        if self.backend == "logistic_best_expert":
            probability = self.model.predict_proba(X)[:, 1]
            return (probability - 0.5) * 2.0 * self.scale
        return np.asarray(self.model.predict(X), dtype=np.float64)


def _group_key(scope: str, dataset: str, seq_len: int, pred_len: int) -> str:
    if scope == "cell":
        return f"cell::{dataset}::{seq_len}::{pred_len}"
    if scope == "dataset":
        return f"dataset::{dataset}"
    if scope == "family":
        return f"family::{dataset_family(dataset)}"
    if scope == "global":
        return "global"
    raise ValueError(f"unknown router scope {scope!r}")


def load_meta_arrays(paths: Sequence[str], require_training_safe: bool = True):
    datasets = [CompactMetaDataset(path) for path in paths]
    if require_training_safe:
        assert_training_metadata_safe(datasets)
    feature_names = datasets[0].feature_names
    expert_names = datasets[0].expert_names
    buckets: Dict[str, List[np.ndarray]] = {}
    cell_rows = []
    for dataset in datasets:
        if dataset.feature_names != feature_names:
            raise ValueError("feature-name mismatch across router metadata")
        if dataset.expert_names != expert_names:
            raise ValueError("expert-order mismatch across router metadata")
        loaded = dataset.load_all()
        rows = len(loaded["features"])
        for name, value in loaded.items():
            buckets.setdefault(name, []).append(value)
        cell_rows.extend(
            [
                (
                    str(dataset.manifest["dataset"]),
                    int(dataset.manifest["seq_len"]),
                    int(dataset.manifest["pred_len"]),
                )
            ]
            * rows
        )
    arrays = {name: np.concatenate(values, axis=0) for name, values in buckets.items()}
    arrays["cell"] = np.asarray(cell_rows, dtype=object)
    return datasets, arrays, feature_names, expert_names


@dataclass
class RouterBundle:
    feature_names: Sequence[str]
    expert_names: Sequence[str]
    anchor_expert: str
    router_scope: str
    target: str
    models: Mapping[str, AdvantageEstimator]
    quantiles: Mapping[str, float]
    fallback_keys: Mapping[str, Sequence[str]]
    metadata: Mapping[str, object]

    @property
    def alternative_names(self) -> List[str]:
        return [name for name in self.expert_names if name != self.anchor_expert]

    def _model_key(self, group: str, expert: str, block: int) -> str:
        return f"{group}||{expert}||b{int(block)}"

    def resolve_group(self, dataset: str, seq_len: int, pred_len: int, expert: str, block: int):
        primary = _group_key(self.router_scope, dataset, seq_len, pred_len)
        candidates = [primary]
        candidates.extend(self.fallback_keys.get(primary, ()))
        candidates.extend(
            [
                _group_key("dataset", dataset, seq_len, pred_len),
                _group_key("family", dataset, seq_len, pred_len),
                "global",
            ]
        )
        seen = set()
        for group in candidates:
            if group in seen:
                continue
            seen.add(group)
            key = self._model_key(group, expert, block)
            if key in self.models:
                return key
        return None

    def predict(
        self,
        features: np.ndarray,
        dataset: str,
        seq_len: int,
        pred_len: int,
        block: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if features.shape[1] != len(self.feature_names):
            raise ValueError("router feature dimension/name mismatch")
        predictions = np.zeros((len(features), len(self.alternative_names)), dtype=np.float64)
        quantiles = np.full(len(self.alternative_names), np.inf, dtype=np.float64)
        for index, expert in enumerate(self.alternative_names):
            key = self.resolve_group(dataset, seq_len, pred_len, expert, block)
            if key is None:
                continue
            predictions[:, index] = self.models[key].predict(features)
            quantiles[index] = float(self.quantiles[key])
        return predictions, quantiles

    def save(self, output_dir: str) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, root / "router.joblib")
        payload = {
            "format": "asyspecx_phase9_safe_router",
            "feature_names": list(self.feature_names),
            "expert_names": list(self.expert_names),
            "anchor_expert": self.anchor_expert,
            "alternative_names": self.alternative_names,
            "router_scope": self.router_scope,
            "router_target": self.target,
            "models": sorted(self.models),
            "quantiles": {key: float(value) for key, value in self.quantiles.items()},
            **dict(self.metadata),
        }
        with (root / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load(cls, path: str):
        source = Path(path)
        model_path = source if source.name == "router.joblib" else source / "router.joblib"
        bundle = joblib.load(model_path)
        if not isinstance(bundle, cls):
            raise TypeError(f"not a RouterBundle: {model_path}")
        return bundle


def _fit_with_oof(
    X: np.ndarray,
    y: np.ndarray,
    origins: np.ndarray,
    backend: str,
    max_depth: int,
    learning_rate: float,
    cv_folds: int,
    purge_steps: int,
    pred_len: int,
    confidence_alpha: float,
    random_state: int,
):
    oof_pred = np.full(len(y), np.nan, dtype=np.float64)
    best_iterations = []
    try:
        splitter = PurgedTimeSeriesSplit(cv_folds, purge_steps, pred_len)
        for fold, (train_index, val_index) in enumerate(splitter.split(origins=origins)):
            estimator = AdvantageEstimator(backend, max_depth, learning_rate, random_state + fold)
            estimator.fit(
                X[train_index],
                y[train_index],
                eval_set=(X[val_index], y[val_index]) if backend == "xgboost" else None,
            )
            oof_pred[val_index] = estimator.predict(X[val_index])
            if backend == "xgboost" and estimator.constant is None:
                best_iteration = getattr(estimator.model, "best_iteration", None)
                if best_iteration is not None:
                    best_iterations.append(int(best_iteration) + 1)
    except ValueError:
        pass
    valid = np.isfinite(oof_pred)
    final = AdvantageEstimator(backend, max_depth, learning_rate, random_state)
    if backend == "xgboost" and best_iterations and final.constant is None:
        final.model.set_params(
            n_estimators=max(1, int(np.median(np.asarray(best_iterations))))
        )
    final.fit(X, y)
    if valid.any():
        quantile = calibrate_lcb_quantile(oof_pred[valid], y[valid], confidence_alpha)
        oof_mae = float(np.mean(np.abs(oof_pred[valid] - y[valid])))
    else:
        # No valid chronological fold: require a conservative positive margin.
        quantile = float(np.quantile(np.abs(y), 1.0 - confidence_alpha))
        oof_mae = None
    return final, quantile, oof_mae, int(valid.sum())


def train_router_bundle(
    meta_paths: Sequence[str],
    anchor_expert: str = "anchor",
    backend: str = "xgboost",
    router_scope: str = "cell",
    target: str = "advantage",
    min_samples: int = 256,
    cv_folds: int = 4,
    purge_steps: int = 0,
    confidence_alpha: float = 0.1,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 2024,
    calibration_paths: Optional[Sequence[str]] = None,
) -> Tuple[RouterBundle, List[Dict[str, object]]]:
    datasets, arrays, feature_names, expert_names = load_meta_arrays(meta_paths, True)
    meta_sources = sorted(
        {str(dataset.manifest.get("router_meta_source", "unknown")) for dataset in datasets}
    )
    if meta_sources == ["val"]:
        router_protocol = "validation-adapted router"
    elif meta_sources == ["rolling_oof"]:
        router_protocol = "rolling-OOF router"
    else:
        router_protocol = "+".join(meta_sources)
    if anchor_expert not in expert_names:
        raise ValueError("anchor expert missing from router metadata")
    if target not in {"advantage", "log_relative_regret"}:
        raise ValueError("unknown router target")
    alternatives = [name for name in expert_names if name != anchor_expert]
    target_values = arrays["advantage"] if target == "advantage" else -arrays["relative_regret"]
    if target_values.shape[1] != len(alternatives):
        raise ValueError("router target/expert dimensions do not align")

    calibration = None
    if calibration_paths:
        calibration_datasets, calibration, cal_names, cal_experts = load_meta_arrays(
            calibration_paths, True
        )
        if cal_names != feature_names or cal_experts != expert_names:
            raise ValueError("calibration metadata schema mismatch")

    cells = [tuple(value) for value in arrays["cell"]]
    requested_groups = sorted(
        {
            _group_key(router_scope, str(dataset), int(seq_len), int(pred_len))
            for dataset, seq_len, pred_len in cells
        }
    )
    # Train fallback pools as well. Duplicates collapse in the set.
    all_scopes = [router_scope]
    unique_cells = set(cells)
    if router_scope == "cell" and len(unique_cells) > 1:
        all_scopes += ["dataset", "family", "global"]
    elif router_scope in {"dataset", "family"} and len(unique_cells) > 1:
        all_scopes += ["global"]
    group_masks: Dict[str, np.ndarray] = {}
    for scope in all_scopes:
        for index, (dataset, seq_len, pred_len) in enumerate(cells):
            key = _group_key(scope, str(dataset), int(seq_len), int(pred_len))
            group_masks.setdefault(key, np.zeros(len(cells), dtype=bool))[index] = True

    models: Dict[str, AdvantageEstimator] = {}
    quantiles: Dict[str, float] = {}
    records: List[Dict[str, object]] = []
    blocks = np.asarray(arrays["block"], dtype=np.int64)
    for group, group_mask in sorted(group_masks.items()):
        for block in sorted(np.unique(blocks)):
            mask = group_mask & (blocks == block)
            if int(mask.sum()) < min_samples:
                continue
            pred_len = max(int(cell[2]) for cell, selected in zip(cells, mask) if selected)
            for expert_index, expert in enumerate(alternatives):
                X = arrays["features"][mask]
                y = target_values[mask, expert_index]
                origins = arrays["origin"][mask]
                estimator, quantile, oof_mae, oof_rows = _fit_with_oof(
                    X,
                    y,
                    origins,
                    backend,
                    max_depth,
                    learning_rate,
                    cv_folds,
                    purge_steps,
                    pred_len,
                    confidence_alpha,
                    random_state + expert_index * 97 + int(block),
                )
                key = f"{group}||{expert}||b{int(block)}"
                if calibration is not None:
                    cal_cells = [tuple(value) for value in calibration["cell"]]
                    cal_mask = np.asarray(
                        [
                            _group_key(group.split("::", 1)[0], str(ds), int(sl), int(pl)) == group
                            if group != "global"
                            else True
                            for ds, sl, pl in cal_cells
                        ],
                        dtype=bool,
                    ) & (calibration["block"] == block)
                    if cal_mask.any():
                        cal_target = (
                            calibration["advantage"][:, expert_index]
                            if target == "advantage"
                            else -calibration["relative_regret"][:, expert_index]
                        )
                        cal_pred = estimator.predict(calibration["features"][cal_mask])
                        quantile = calibrate_lcb_quantile(
                            cal_pred,
                            cal_target[cal_mask],
                            confidence_alpha,
                        )
                models[key] = estimator
                quantiles[key] = float(quantile)
                records.append(
                    {
                        "model_key": key,
                        "rows": int(mask.sum()),
                        "oof_rows": oof_rows,
                        "oof_mae": oof_mae,
                        "calibration_quantile": float(quantile),
                    }
                )

    fallback_keys = {}
    for group in requested_groups:
        parts = group.split("::")
        if parts[0] == "cell":
            dataset = parts[1]
            fallback_keys[group] = [f"dataset::{dataset}", f"family::{dataset_family(dataset)}", "global"]
        elif parts[0] in {"dataset", "family"}:
            fallback_keys[group] = ["global"]
    bundle = RouterBundle(
        feature_names=feature_names,
        expert_names=expert_names,
        anchor_expert=anchor_expert,
        router_scope=router_scope,
        target=target,
        models=models,
        quantiles=quantiles,
        fallback_keys=fallback_keys,
        metadata={
            "router_backend": backend,
            "router_min_samples": min_samples,
            "router_cv_folds": cv_folds,
            "router_purge_steps": purge_steps,
            "router_confidence_alpha": confidence_alpha,
            "training_meta": list(meta_paths),
            "calibration_meta": list(calibration_paths or []),
            "router_protocol": router_protocol,
            "router_meta_sources": meta_sources,
        },
    )
    return bundle, records
