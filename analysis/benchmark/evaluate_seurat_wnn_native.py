"""Evaluate only the native two-dimensional Seurat WNN embedding.

This is intentionally separate from the main PCA-20/PCA-25 benchmark.  It
uses Seurat WNN's raw ``obsm['X_model']`` (expected to be two-dimensional),
runs the same mclust EEE procedure, and calculates the same configured
metrics.  BioSFIB is opened only to obtain canonical truth labels, spot order,
and the shared spatial KNN graph; it is never re-clustered here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from analysis.benchmark.evaluate_benchmark import (
    build_spatial_connectivities,
    calculate_metrics,
    close_adatas,
    configured_metrics,
    metric_row,
    metric_run_status,
    mclust_labels,
    resolve_one_file,
    valid_labels,
    write_checkpoint,
)


@dataclass(frozen=True)
class NativeSpec:
    dataset: str
    slice_name: str
    condition_name: str
    condition: dict[str, Any]
    truth_key: str | None
    cluster_count: int | None
    cluster_count_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate native two-dimensional Seurat WNN embeddings only.")
    parser.add_argument("--config", default="analysis/benchmark/seurat_wnn_native_config.yaml", help="Seurat WNN native-evaluation YAML path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without mclust or output writes.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing this script's own CSV and manifest.")
    parser.add_argument("--datasets", nargs="*", help="Optional subset of configured dataset names.")
    parser.add_argument("--slices", nargs="*", help="Optional subset of slice names.")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def discover_slices(root: Path, dataset: str) -> list[str]:
    directory = root / "data" / "benchmark" / dataset / "BioSFIB"
    if not directory.is_dir():
        raise FileNotFoundError(f"BioSFIB directory missing: {directory}")
    return sorted(path.name for path in directory.iterdir() if path.is_dir())


def resolve_truth_key(dataset_run: dict[str, Any], condition: dict[str, Any], dataset_config: dict[str, Any]) -> str | None:
    """Resolve truth key in run > condition > dataset order; explicit null means no truth."""
    for scope in (dataset_run, condition, dataset_config):
        if "truth_key" in scope:
            value = scope["truth_key"]
            return str(value) if value else None
    raise KeyError("No truth_key configured. Set it in dataset_run, condition, or dataset configuration.")


def resolve_cluster_count(
    bio_adata: ad.AnnData,
    dataset_run: dict[str, Any],
    condition: dict[str, Any],
    slice_name: str,
    truth_key: str | None,
) -> tuple[int | None, str]:
    manual = dataset_run.get("cluster_count_by_slice", {})
    if slice_name in manual:
        return int(manual[slice_name]), "manual"
    configured = condition["cluster_count"]
    if configured == "manual":
        return None, "manual_required"
    if configured != "truth_n_classes":
        return int(configured), "condition"
    if truth_key is None or truth_key not in bio_adata.obs:
        return None, "missing_truth"
    mask = valid_labels(bio_adata.obs[truth_key])
    if not mask.any():
        return None, "missing_truth"
    return int(bio_adata.obs.loc[mask, truth_key].astype("string").nunique()), "truth_n_classes"


def build_specs(root: Path, config: dict[str, Any], datasets: list[str] | None, slices: list[str] | None) -> list[NativeSpec]:
    selected = set(datasets) if datasets else set(config["datasets"])
    unknown = selected.difference(config["datasets"])
    if unknown:
        raise KeyError(f"Unknown configured datasets: {sorted(unknown)}")
    selected_slices = set(slices) if slices else None
    specs: list[NativeSpec] = []
    bio_pattern = config["input"]["biosfib_file_glob"]
    for dataset, dataset_config in config["datasets"].items():
        if dataset not in selected:
            continue
        for slice_name in discover_slices(root, dataset):
            if selected_slices is not None and slice_name not in selected_slices:
                continue
            bio_path = resolve_one_file(root, bio_pattern, dataset, slice_name)
            bio_adata = ad.read_h5ad(bio_path, backed="r")
            try:
                for dataset_run in dataset_config["runs"]:
                    condition_name = dataset_run["condition"]
                    condition = config["conditions"][condition_name]
                    truth_key = resolve_truth_key(dataset_run, condition, dataset_config)
                    cluster_count, source = resolve_cluster_count(bio_adata, dataset_run, condition, slice_name, truth_key)
                    specs.append(NativeSpec(dataset, slice_name, condition_name, condition, truth_key, cluster_count, source))
            finally:
                bio_adata.file.close()
    return specs


def load_inputs(root: Path, config: dict[str, Any], dataset: str, slice_name: str, backed: str | None) -> tuple[ad.AnnData, ad.AnnData, Path, Path]:
    bio_path = resolve_one_file(root, config["input"]["biosfib_file_glob"], dataset, slice_name)
    seurat_path = resolve_one_file(root, config["input"]["seurat_wnn_file_glob"], dataset, slice_name)
    bio = ad.read_h5ad(bio_path, backed=backed)
    seurat = ad.read_h5ad(seurat_path, backed=backed)
    try:
        embedding_key = config["input"]["embedding_key"]
        if embedding_key not in seurat.obsm:
            raise KeyError(f"{seurat_path}: missing obsm[{embedding_key!r}].")
        if "spatial" not in bio.obsm or "spatial" not in seurat.obsm:
            raise KeyError("Both BioSFIB and Seurat WNN inputs must contain obsm['spatial'].")
        embedding = seurat.obsm[embedding_key]
        if embedding.shape[0] != seurat.n_obs:
            raise ValueError(f"{seurat_path}: embedding row count does not match n_obs.")
        expected_dim = int(config["input"]["expected_embedding_dim"])
        if embedding.shape[1] != expected_dim:
            raise ValueError(f"{seurat_path}: expected raw embedding dimension {expected_dim}, found {embedding.shape[1]}.")
        if not seurat.obs_names.equals(bio.obs_names):
            raise ValueError(f"{dataset}/{slice_name}: Seurat WNN and BioSFIB obs_names differ.")
        bio_coords = np.asarray(bio.obsm["spatial"])
        seurat_coords = np.asarray(seurat.obsm["spatial"])
        if bio_coords.shape != (bio.n_obs, 2) or seurat_coords.shape != (seurat.n_obs, 2):
            raise ValueError(f"{dataset}/{slice_name}: spatial coordinates must have shape (n_obs, 2).")
        if config["input"].get("require_identical_spatial_coordinates", True) and not np.array_equal(bio_coords, seurat_coords):
            raise ValueError(f"{dataset}/{slice_name}: Seurat WNN spatial coordinates differ from BioSFIB canonical coordinates.")
        return bio, seurat, bio_path, seurat_path
    except Exception:
        close_adatas({"BioSFIB": bio, "SeuratWNN": seurat})
        raise


def write_native_prediction(
    output_path: Path,
    source: ad.AnnData,
    native_embedding: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
    canonical_truth: pd.Series | None,
) -> None:
    """Persist only the fields required for Seurat WNN Spatial/native-2D map panels."""
    obs = pd.DataFrame(index=source.obs_names.copy())
    if canonical_truth is not None:
        obs["reference_label"] = canonical_truth.copy()
    obs["mclust_benchmark"] = pd.Categorical(labels.astype(str))
    prediction = ad.AnnData(X=np.zeros((source.n_obs, 0), dtype=np.float32), obs=obs)
    prediction.obsm["spatial"] = np.asarray(source.obsm["spatial"], dtype=np.float32)
    prediction.obsm["X_model_eval_pca"] = native_embedding.astype(np.float32, copy=False)
    prediction.obsm["X_model_native_2d"] = native_embedding.astype(np.float32, copy=False)
    prediction.uns["benchmark_metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(output_path)


def run(root: Path, config: dict[str, Any], specs: list[NativeSpec], dry_run: bool, overwrite: bool) -> int:
    output_dir = root / config["run"]["output_dir"]
    csv_path = output_dir / config["output"]["metrics_csv"]
    manifest_path = output_dir / config["output"]["manifest_json"]
    if not dry_run and not (overwrite or config["run"].get("overwrite", False)):
        occupied = [path for path in (csv_path, manifest_path) if path.exists()]
        if occupied:
            raise FileExistsError(f"Output already exists: {occupied}. Use --overwrite to replace it.")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if overwrite or config["run"].get("overwrite", False):
            for path in (csv_path, manifest_path):
                if path.exists():
                    path.unlink()

    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema_version": "seurat-wnn-native-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "config": config,
        "runs": [],
    }
    metrics_to_run = configured_metrics(config)
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.dataset}/{spec.slice_name} | {spec.condition_name} | SeuratWNN(raw_2d)", flush=True)
        if spec.cluster_count is None:
            raise ValueError(f"{spec.dataset}/{spec.slice_name}/{spec.condition_name}: cannot infer K without truth or a manual value.")
        bio = seurat = None
        try:
            bio, seurat, bio_path, seurat_path = load_inputs(root, config, spec.dataset, spec.slice_name, backed="r" if dry_run else None)
            truth = bio.obs[spec.truth_key] if spec.truth_key is not None and spec.truth_key in bio.obs else None
            valid_truth = valid_labels(truth) if truth is not None else None
            base = {
                "evaluation_id": spec.condition_name,
                "evaluation_role": spec.condition.get("role", "unspecified"),
                "dataset": spec.dataset,
                "slice": spec.slice_name,
                "method": "SeuratWNN",
                "source_file": str(seurat_path.relative_to(root)).replace("\\", "/"),
                "truth_source_method": "BioSFIB",
                "n_spots": int(seurat.n_obs),
                "truth_key": spec.truth_key,
                "n_valid_truth": int(valid_truth.sum()) if valid_truth is not None else 0,
                "truth_K": int(truth.loc[valid_truth].astype("string").nunique()) if valid_truth is not None and valid_truth.any() else None,
                "K": int(spec.cluster_count),
                "K_source": spec.cluster_count_source,
                "representation": "raw_2d_native",
                "pca_dim_requested": None,
                "pca_dim_actual": None,
                "mclust_model": str(config["pipeline"]["mclust"]["model_name"]).upper(),
                "seed": int(config["pipeline"]["mclust"]["seed"]),
            }
            if dry_run:
                manifest["runs"].append({**base, "status": "validated"})
                continue
            try:
                features = np.asarray(seurat.obsm[config["input"]["embedding_key"]], dtype=np.float32)
                labels = mclust_labels(features, spec.cluster_count, config["pipeline"]["mclust"]["model_name"], config["pipeline"]["mclust"]["seed"])
                spatial = build_spatial_connectivities(np.asarray(bio.obsm["spatial"]), int(config["pipeline"]["spatial_moran"]["n_neighbors"]))
                metric_rows = calculate_metrics(
                    metrics_to_run, base, features, labels, truth, spatial,
                    int(config["pipeline"]["mclust"]["seed"]), int(config["pipeline"]["spatial_moran"]["n_neighbors"]),
                )
                rows.extend(metric_rows)
                metadata = {**base, "n_clusters_observed": int(np.unique(labels).size), "source_embedding_key": config["input"]["embedding_key"]}
                if bool(config["output"]["prediction_h5ad"]["save"]):
                    prediction_path = (
                        output_dir / config["output"]["predictions_dir"] / spec.condition_name
                        / spec.dataset / spec.slice_name / "SeuratWNN_benchmark.h5ad"
                    )
                    write_native_prediction(prediction_path, seurat, features, labels, metadata, truth)
                manifest["runs"].append({**metadata, "status": metric_run_status(metric_rows)})
            except Exception as exc:
                rows.append(metric_row(base, "pipeline", 0, "pipeline", None, f"error:{type(exc).__name__}", "required", {"message": str(exc)}))
                manifest["runs"].append({**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            write_checkpoint(csv_path, manifest_path, rows, manifest)
        finally:
            close_adatas({"BioSFIB": bio, "SeuratWNN": seurat})
    if dry_run:
        print(f"Validated {len(manifest['runs'])} native Seurat WNN inputs; no outputs written.")
        return 0
    write_checkpoint(csv_path, manifest_path, rows, manifest)
    print(f"Metrics: {csv_path}")
    print(f"Manifest: {manifest_path}")
    return 1 if any(run["status"] == "failed" for run in manifest["runs"]) else 0


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    return run(PROJECT_ROOT, config, build_specs(PROJECT_ROOT, config, args.datasets, args.slices), args.dry_run, args.overwrite)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Seurat WNN native evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
