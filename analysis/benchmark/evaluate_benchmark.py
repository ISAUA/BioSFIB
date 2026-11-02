"""Unified PCA + mclust benchmark for spatial multi-omics embeddings.

The script never changes source H5AD files.  Each model is evaluated from its
``obsm['X_model']`` embedding, with an independently fitted PCA for every
model/slice/condition.  Results are persisted as lightweight H5AD prediction
files and a long-form CSV that can be extended with new metrics later.

Examples
--------
python evaluate_benchmark.py --config benchmark_config.yaml --dry-run
python evaluate_benchmark.py --config benchmark_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


EXTERNAL_METRICS = {
    "ARI": ("external", 1, adjusted_rand_score),
    "NMI": ("external", 2, normalized_mutual_info_score),
    "AMI": ("external", 3, adjusted_mutual_info_score),
    "V_MEASURE": ("external", 4, v_measure_score),
    "HOM": ("external", 5, homogeneity_score),
    "FMI": ("external", 6, fowlkes_mallows_score),
}
INTERNAL_METRICS = {"silhouette": ("internal", 1)}
SPATIAL_METRICS = {"cluster_morans_i": ("spatial", 1)}


@dataclass(frozen=True)
class SliceSpec:
    dataset: str
    slice_name: str
    condition_name: str
    condition: dict[str, Any]
    truth_key: str | None
    cluster_count: int | None
    cluster_count_source: str
    methods: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified PCA + mclust benchmark.")
    parser.add_argument("--config", default="analysis/benchmark/benchmark_config.yaml", help="YAML configuration path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate all inputs without PCA/mclust/output writes.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing CSV/manifest/prediction outputs.")
    parser.add_argument("--datasets", nargs="*", help="Optional subset of dataset names.")
    parser.add_argument("--slices", nargs="*", help="Optional subset of slice names (used with --datasets when names overlap).")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def valid_labels(values: pd.Series) -> pd.Series:
    labels = values.astype("string")
    return labels.notna() & labels.str.strip().ne("") & labels.str.lower().ne("nan")


def discover_slices(root: Path, dataset: str) -> list[str]:
    biosfib_dir = root / "data" / "benchmark" / dataset / "BioSFIB"
    if not biosfib_dir.is_dir():
        raise FileNotFoundError(f"BioSFIB directory missing: {biosfib_dir}")
    return sorted(path.name for path in biosfib_dir.iterdir() if path.is_dir())


def resolve_one_file(root: Path, pattern: str, dataset: str, slice_name: str) -> Path:
    matches = sorted(root.glob(pattern.format(dataset=dataset, slice=slice_name)))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file for dataset={dataset}, slice={slice_name}, pattern={pattern}; "
            f"found {len(matches)}."
        )
    return matches[0]


def load_slice_inputs(
    root: Path,
    config: dict[str, Any],
    dataset: str,
    slice_name: str,
    methods: Iterable[str],
    backed: str | None = None,
) -> tuple[dict[str, ad.AnnData], dict[str, Path]]:
    adatas: dict[str, ad.AnnData] = {}
    paths: dict[str, Path] = {}
    for method in methods:
        method_config = config["input"]["methods"][method]
        path = resolve_one_file(root, method_config["file_glob"], dataset, slice_name)
        adata = ad.read_h5ad(path, backed=backed)
        embedding_key = method_config["embedding_key"]
        if embedding_key not in adata.obsm:
            raise KeyError(f"{path}: missing obsm[{embedding_key!r}].")
        if "spatial" not in adata.obsm:
            raise KeyError(f"{path}: missing obsm['spatial'].")
        if adata.obsm[embedding_key].shape[0] != adata.n_obs:
            raise ValueError(f"{path}: embedding row count does not match n_obs.")
        if adata.obsm["spatial"].shape != (adata.n_obs, 2):
            raise ValueError(f"{path}: spatial coordinates must have shape (n_obs, 2).")
        adatas[method] = adata
        paths[method] = path

    reference = adatas["BioSFIB"].obs_names
    canonical_coords = np.asarray(adatas["BioSFIB"].obsm["spatial"])
    for method, adata in adatas.items():
        if not adata.obs_names.equals(reference):
            raise ValueError(f"{dataset}/{slice_name}: obs_names differ for {method}; benchmark requires identical order.")
        if config["input"].get("require_identical_spatial_coordinates", True):
            if not np.array_equal(canonical_coords, np.asarray(adata.obsm["spatial"])):
                raise ValueError(
                    f"{dataset}/{slice_name}: spatial coordinates differ for {method}; "
                    "benchmark requires the BioSFIB canonical coordinates."
                )
    return adatas, paths


def resolve_cluster_count(
    adata: ad.AnnData,
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
    if truth_key is None or truth_key not in adata.obs:
        return None, "missing_truth"
    mask = valid_labels(adata.obs[truth_key])
    if not mask.any():
        return None, "missing_truth"
    return int(adata.obs.loc[mask, truth_key].astype("string").nunique()), "truth_n_classes"


def resolve_truth_key(dataset_run: dict[str, Any], condition: dict[str, Any], dataset_config: dict[str, Any]) -> str | None:
    """Resolve truth key; an explicit null means that a dataset has no external truth."""
    for scope_name, scope in (
        ("dataset_run", dataset_run),
        ("condition", condition),
        ("dataset", dataset_config),
    ):
        if "truth_key" in scope:
            value = scope["truth_key"]
            return str(value) if value else None
    raise KeyError(
        "No truth_key configured. Set it in dataset_run, condition, or dataset configuration "
        "(in that precedence order)."
    )


def resolve_methods(dataset_config: dict[str, Any], config: dict[str, Any]) -> tuple[str, ...]:
    """Resolve methods to score; BioSFIB is loaded separately as canonical input when needed."""
    configured = dataset_config.get("methods", list(config["input"]["methods"]))
    methods = tuple(str(method) for method in configured)
    if not methods:
        raise ValueError("Dataset methods must contain at least one scored method.")
    unknown = set(methods).difference(config["input"]["methods"])
    if unknown:
        raise KeyError(f"Dataset config references unknown methods: {sorted(unknown)}")
    if len(methods) != len(set(methods)):
        raise ValueError(f"Dataset methods contains duplicates: {methods}")
    return methods


def build_specs(
    root: Path,
    config: dict[str, Any],
    datasets: Iterable[str] | None,
    slices: Iterable[str] | None,
) -> list[SliceSpec]:
    selected = set(datasets) if datasets else set(config["datasets"])
    unknown = selected.difference(config["datasets"])
    if unknown:
        raise KeyError(f"Unknown configured datasets: {sorted(unknown)}")
    specs: list[SliceSpec] = []
    selected_slices = set(slices) if slices else None
    # Preserve YAML order. The configured first run is simulation/simulation1,
    # rather than an implementation-dependent alphabetical order.
    for dataset in config["datasets"]:
        if dataset not in selected:
            continue
        dataset_config = config["datasets"][dataset]
        methods = resolve_methods(dataset_config, config)
        for slice_name in discover_slices(root, dataset):
            if selected_slices is not None and slice_name not in selected_slices:
                continue
            # BioSFIB carries the canonical reference label and spot order.
            bio_path = resolve_one_file(root, config["input"]["methods"]["BioSFIB"]["file_glob"], dataset, slice_name)
            bio_adata = ad.read_h5ad(bio_path, backed="r")
            try:
                for dataset_run in dataset_config["runs"]:
                    condition_name = dataset_run["condition"]
                    condition = config["conditions"][condition_name]
                    truth_key = resolve_truth_key(dataset_run, condition, dataset_config)
                    cluster_count, source = resolve_cluster_count(bio_adata, dataset_run, condition, slice_name, truth_key)
                    specs.append(SliceSpec(dataset, slice_name, condition_name, condition, truth_key, cluster_count, source, methods))
            finally:
                bio_adata.file.close()
    return specs


def mclust_labels(features: np.ndarray, n_clusters: int, model_name: str, seed: int) -> np.ndarray:
    """Run R mclust through rpy2, importing it only for an actual benchmark run."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import default_converter, numpy2ri
        from rpy2.robjects.conversion import localconverter
    except ImportError as exc:
        raise RuntimeError(
            "mclust requires rpy2 plus an R installation with package 'mclust'. "
            "Activate that environment, then rerun the benchmark."
        ) from exc
    try:
        ro.r("suppressPackageStartupMessages(library(mclust))")
    except Exception as exc:
        raise RuntimeError("R package 'mclust' is unavailable in the active R environment.") from exc
    ro.r["set.seed"](int(seed))
    with localconverter(default_converter + numpy2ri.converter):
        r_features = ro.conversion.py2rpy(features)
        result = ro.r["Mclust"](r_features, int(n_clusters), str(model_name).upper(), verbose=False)
    # With rpy2's localconverter, an R named list is converted to an OrdDict;
    # without conversion it remains an R ListVector. Support both forms and
    # never depend on mclust's positional list layout.
    if hasattr(result, "keys") and "classification" in result:
        labels = np.asarray(result["classification"], dtype=int)
    elif hasattr(result, "rx2"):
        labels = np.asarray(result.rx2("classification"), dtype=int)
    else:
        raise RuntimeError("mclust result has no named 'classification' field.")
    if labels.size != features.shape[0]:
        raise RuntimeError("mclust returned an unexpected number of labels.")
    return labels


def calculate_silhouette(features: np.ndarray, labels: np.ndarray, seed: int, max_samples: int = 10_000) -> float:
    if np.unique(labels).size < 2 or np.unique(labels).size >= labels.size:
        return float("nan")
    if features.shape[0] <= max_samples:
        return float(silhouette_score(features, labels))
    rng = np.random.default_rng(seed)
    indices = rng.choice(features.shape[0], size=max_samples, replace=False)
    return float(silhouette_score(features[indices], labels[indices]))


def build_spatial_connectivities(coords: np.ndarray, n_neighbors: int):
    """Build the canonical spatial KNN graph once per slice."""
    try:
        import squidpy as sq
    except ImportError as exc:
        raise RuntimeError("cluster_morans_i requires squidpy.") from exc
    graph_adata = ad.AnnData(X=np.zeros((coords.shape[0], 1), dtype=np.float32))
    graph_adata.obsm["spatial"] = np.asarray(coords, dtype=np.float32)
    sq.gr.spatial_neighbors(graph_adata, coord_type="generic", n_neighs=int(n_neighbors))
    return graph_adata.obsp["spatial_connectivities"].copy()


def calculate_cluster_morans_i(labels: np.ndarray, spatial_connectivities) -> float:
    """Mean Moran's I of one-hot cluster membership on the canonical spatial graph."""
    try:
        import squidpy as sq
    except ImportError as exc:
        raise RuntimeError("cluster_morans_i requires squidpy.") from exc
    categories = pd.Categorical(labels)
    one_hot = np.eye(len(categories.categories), dtype=np.float32)[categories.codes]
    variable = np.var(one_hot, axis=0) > 0
    if not variable.any():
        return float("nan")
    moran_adata = ad.AnnData(X=one_hot[:, variable])
    moran_adata.obsp["spatial_connectivities"] = spatial_connectivities
    sq.gr.spatial_autocorr(
        moran_adata,
        mode="moran",
        connectivity_key="spatial_connectivities",
        n_perms=None,
        show_progress_bar=False,
    )
    return float(moran_adata.uns["moranI"]["I"].mean())


def metric_row(base: dict[str, Any], group: str, order: int, name: str, value: float | None, status: str, requirement: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    row = dict(base)
    row.update(
        metric_group=group,
        metric_order=order,
        metric_name=name,
        metric_value=value,
        metric_status=status,
        metric_requirement=requirement,
        metric_direction="higher",
        metric_params=json.dumps(params or {}, sort_keys=True),
    )
    return row


def calculate_metrics(
    metrics: list[tuple[str, str]],
    base: dict[str, Any],
    pca_embedding: np.ndarray,
    labels: np.ndarray,
    truth: pd.Series | None,
    spatial_connectivities,
    seed: int,
    n_neighbors: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    truth_mask = None if truth is None else valid_labels(truth)
    truth_mask_array = None if truth_mask is None else np.asarray(truth_mask.fillna(False), dtype=bool)
    for name, requirement in metrics:
        if name in EXTERNAL_METRICS:
            group, order, function = EXTERNAL_METRICS[name]
            if truth is None or not truth_mask.any():
                rows.append(metric_row(base, group, order, name, None, "not_applicable", requirement))
            else:
                try:
                    value = float(function(truth.loc[truth_mask].astype(str).to_numpy(), labels[truth_mask_array]))
                    rows.append(metric_row(base, group, order, name, value, "ok", requirement))
                except Exception as exc:
                    rows.append(metric_row(base, group, order, name, None, f"error:{type(exc).__name__}", requirement))
        elif name in INTERNAL_METRICS:
            group, order = INTERNAL_METRICS[name]
            try:
                rows.append(metric_row(base, group, order, name, calculate_silhouette(pca_embedding, labels, seed), "ok", requirement))
            except Exception as exc:
                rows.append(metric_row(base, group, order, name, None, f"error:{type(exc).__name__}", requirement))
        elif name in SPATIAL_METRICS:
            group, order = SPATIAL_METRICS[name]
            try:
                value = calculate_cluster_morans_i(labels, spatial_connectivities)
                rows.append(metric_row(base, group, order, name, value, "ok", requirement, {"n_neighbors": n_neighbors}))
            except Exception as exc:
                rows.append(metric_row(base, group, order, name, None, f"error:{type(exc).__name__}", requirement))
        else:
            raise KeyError(f"Unsupported metric in configuration: {name}")
    return rows


def configured_metrics(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (metric_name, requirement), allowing old list-style configs too."""
    raw = config["pipeline"]["metrics"]
    if isinstance(raw, list):
        return [(str(name), "required") for name in raw]
    result = [(str(name), "required") for name in raw.get("required", [])]
    result.extend((str(name), "optional") for name in raw.get("optional", []))
    names = [name for name, _ in result]
    if not names or len(names) != len(set(names)):
        raise ValueError("pipeline.metrics must contain one or more unique metric names.")
    return result


def write_prediction(
    output_path: Path,
    source: ad.AnnData,
    pca_embedding: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
    canonical_truth: pd.Series | None,
) -> None:
    obs = pd.DataFrame(index=source.obs_names.copy())
    if canonical_truth is not None:
        obs["reference_label"] = canonical_truth.copy()
    obs["mclust_benchmark"] = pd.Categorical(labels.astype(str))
    prediction = ad.AnnData(X=np.zeros((source.n_obs, 0), dtype=np.float32), obs=obs)
    prediction.obsm["spatial"] = np.asarray(source.obsm["spatial"], dtype=np.float32)
    prediction.obsm["X_model_eval_pca"] = pca_embedding.astype(np.float32, copy=False)
    prediction.uns["benchmark_metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(output_path)


def close_adatas(adatas: dict[str, ad.AnnData] | None) -> None:
    """Close HDF5 handles opened by AnnData backed mode; harmless for in-memory data."""
    if not adatas:
        return
    for adata in adatas.values():
        file_handle = getattr(adata, "file", None)
        if file_handle is not None and getattr(file_handle, "is_open", False):
            file_handle.close()


def metric_run_status(metric_rows: list[dict[str, Any]]) -> str:
    """Required metric errors fail a run; optional errors produce partial success."""
    required_error = any(
        row["metric_requirement"] == "required" and str(row["metric_status"]).startswith("error:")
        for row in metric_rows
    )
    if required_error:
        return "failed"
    optional_error = any(
        row["metric_requirement"] == "optional" and str(row["metric_status"]).startswith("error:")
        for row in metric_rows
    )
    return "partial_success" if optional_error else "ok"


def write_checkpoint(csv_path: Path, manifest_path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    """Persist completed work so progress is visible and Ctrl+C loses at most one run."""
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def run(root: Path, config: dict[str, Any], specs: list[SliceSpec], dry_run: bool, overwrite: bool) -> int:
    output_dir = root / config["run"]["output_dir"]
    output = config["output"]
    csv_path = output_dir / output["metrics_csv"]
    manifest_path = output_dir / output["manifest_json"]
    if not dry_run and not (overwrite or config["run"].get("overwrite", False)):
        occupied = [path for path in (csv_path, manifest_path) if path.exists()]
        if occupied:
            raise FileExistsError(f"Output already exists: {occupied}. Use --overwrite to replace it.")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        # --overwrite starts a new visible result table instead of leaving a
        # previous smoke-test/full-run CSV on screen until the final write.
        if overwrite or config["run"].get("overwrite", False):
            for path in (csv_path, manifest_path):
                if path.exists():
                    path.unlink()

    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema_version": "benchmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "config": config,
        "runs": [],
    }
    current_key: tuple[str, str, tuple[str, ...]] | None = None
    current_adatas: dict[str, ad.AnnData] | None = None
    current_paths: dict[str, Path] | None = None
    current_spatial_connectivities = None
    metrics_to_run = configured_metrics(config)
    total_method_runs = sum(len(spec.methods) for spec in specs)
    completed_method_runs = 0
    try:
        for spec in specs:
            cache_key = (spec.dataset, spec.slice_name, spec.methods)
            if cache_key != current_key:
                close_adatas(current_adatas)
                input_methods = tuple(dict.fromkeys(("BioSFIB", *spec.methods)))
                current_adatas, current_paths = load_slice_inputs(
                    root, config, spec.dataset, spec.slice_name, input_methods, backed="r" if dry_run else None
                )
                if not dry_run:
                    current_spatial_connectivities = build_spatial_connectivities(
                        np.asarray(current_adatas["BioSFIB"].obsm["spatial"]),
                        int(config["pipeline"]["spatial_moran"]["n_neighbors"]),
                    )
                current_key = cache_key
            adatas, paths = current_adatas, current_paths
            if spec.cluster_count is None:
                raise ValueError(f"{spec.dataset}/{spec.slice_name}/{spec.condition_name}: cannot infer K without truth or a manual value.")

            canonical_adata = adatas["BioSFIB"]
            canonical_truth = canonical_adata.obs[spec.truth_key] if spec.truth_key is not None and spec.truth_key in canonical_adata.obs else None
            canonical_valid_truth = valid_labels(canonical_truth) if canonical_truth is not None else None

            for method in spec.methods:
                source = adatas[method]
                completed_method_runs += 1
                print(
                    f"[{completed_method_runs}/{total_method_runs}] "
                    f"{spec.dataset}/{spec.slice_name} | {spec.condition_name} | {method}",
                    flush=True,
                )
                method_config = config["input"]["methods"][method]
                requested_dim = int(spec.condition["pca_dim"])
                embedding_shape = source.obsm[method_config["embedding_key"]].shape
                if config["pipeline"]["pca"].get("strict_dimensions", True) and min(embedding_shape) < requested_dim:
                    raise ValueError(
                        f"{spec.dataset}/{spec.slice_name}/{method}/{spec.condition_name}: requested PCA-{requested_dim}, "
                        f"but embedding shape is {embedding_shape}; refusing to silently use a different dimension."
                    )
                actual_dim = requested_dim if config["pipeline"]["pca"].get("strict_dimensions", True) else min(requested_dim, *embedding_shape)
                base = {
                    "evaluation_id": spec.condition_name,
                    "evaluation_role": spec.condition.get("role", "unspecified"),
                    "dataset": spec.dataset,
                    "slice": spec.slice_name,
                    "method": method,
                    "source_file": str(paths[method].relative_to(root)).replace("\\", "/"),
                    "truth_source_method": "BioSFIB",
                    "n_spots": int(source.n_obs),
                    "truth_key": spec.truth_key,
                    "n_valid_truth": int(canonical_valid_truth.sum()) if canonical_valid_truth is not None else 0,
                    "truth_K": int(canonical_truth.loc[canonical_valid_truth].astype("string").nunique()) if canonical_valid_truth is not None and canonical_valid_truth.any() else None,
                    "K": int(spec.cluster_count),
                    "K_source": spec.cluster_count_source,
                    "pca_dim_requested": requested_dim,
                    "pca_dim_actual": actual_dim,
                    "mclust_model": str(config["pipeline"]["mclust"]["model_name"]).upper(),
                    "seed": int(config["pipeline"]["mclust"]["seed"]),
                }
                if dry_run:
                    manifest["runs"].append({**base, "status": "validated"})
                    continue
                try:
                    embedding = np.asarray(source.obsm[method_config["embedding_key"]], dtype=np.float32)
                    pca_embedding = PCA(n_components=actual_dim).fit_transform(embedding)
                    labels = mclust_labels(
                    pca_embedding,
                    spec.cluster_count,
                    config["pipeline"]["mclust"]["model_name"],
                        config["pipeline"]["mclust"]["seed"],
                    )
                    method_rows = calculate_metrics(
                        metrics_to_run,
                        base,
                        pca_embedding,
                        labels,
                        canonical_truth,
                        current_spatial_connectivities,
                        int(config["pipeline"]["mclust"]["seed"]),
                        int(config["pipeline"]["spatial_moran"]["n_neighbors"]),
                    )
                    rows.extend(method_rows)
                    metadata = {**base, "n_clusters_observed": int(np.unique(labels).size), "source_embedding_key": method_config["embedding_key"]}
                    if output["prediction_h5ad"]["save"]:
                        pred_path = output_dir / output["predictions_dir"] / spec.condition_name / spec.dataset / spec.slice_name / f"{method}_benchmark.h5ad"
                        write_prediction(pred_path, source, pca_embedding, labels, metadata, canonical_truth)
                    manifest["runs"].append({**metadata, "status": metric_run_status(method_rows)})
                except Exception as exc:
                    rows.append(metric_row(base, "pipeline", 0, "pipeline", None, f"error:{type(exc).__name__}", "required", {"message": str(exc)}))
                    manifest["runs"].append({**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                write_checkpoint(csv_path, manifest_path, rows, manifest)
    finally:
        close_adatas(current_adatas)

    if dry_run:
        print(f"Validated {len(manifest['runs'])} method-condition inputs; no outputs written.")
        return 0
    write_checkpoint(csv_path, manifest_path, rows, manifest)
    print(f"Metrics: {csv_path}")
    print(f"Manifest: {manifest_path}")
    return 1 if any(run["status"] == "failed" for run in manifest["runs"]) else 0


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    root = PROJECT_ROOT
    specs = build_specs(root, config, args.datasets, args.slices)
    return run(root, config, specs, args.dry_run, args.overwrite)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Benchmark failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
