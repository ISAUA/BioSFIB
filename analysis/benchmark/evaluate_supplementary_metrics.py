"""Supplement saved benchmark predictions without training or altering inputs.

Static metrics use all spots. Spatial scores use canonical BioSFIB coordinates.
Optional stability refits mclust on shared 80% subsets and compares labels on
those spots with the full-data partition; it measures downstream stability only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

DEFINITIONS = {
    "calinski_harabasz": ("internal", "higher"),
    "davies_bouldin": ("internal", "lower"),
    "cluster_gearys_c": ("spatial", "lower"),
    "PAS_k6": ("spatial", "lower"),
    "PAS_k10": ("spatial", "lower"),
    "neighbor_agreement_k6": ("spatial", "higher"),
    "CHAOS": ("spatial", "lower"),
    "subsample_stability_ARI": ("stability", "higher"),
}


def read_prediction(path, cfg):
    # Read only the needed AnnData HDF5 fields, avoiding map-specific imports.
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        index_key = obs.attrs["_index"]
        names = obs[index_key].asstr()[:]
        node = obs[cfg["label_key"]]
        if isinstance(node, h5py.Group):
            codes = node["codes"][:]
            if np.any(codes < 0):
                raise ValueError(f"Missing labels in {path}")
            categories = node["categories"].asstr()[:]
            labels = categories[codes]
        else:
            labels = node.asstr()[:]
        features = np.asarray(f["obsm"][cfg["embedding_key"]], dtype=np.float64)
        coords = np.asarray(f["obsm"]["spatial"], dtype=np.float32)
    if len(np.unique(names)) != len(names):
        raise ValueError(f"Duplicate spot IDs in {path}")
    if features.shape[0] != len(names) or coords.shape != (len(names), 2):
        raise ValueError(f"Invalid prediction dimensions in {path}")
    if not np.isfinite(features).all() or not np.isfinite(coords).all():
        raise ValueError(f"Non-finite input in {path}")
    if any(str(x).strip().lower() in {"", "nan", "none"} for x in labels):
        raise ValueError(f"Invalid labels in {path}")
    return names, features, labels, coords


def neighbors(coords, k):
    # Same generic KNN construction as squidpy: .kneighbors() excludes self.
    indices = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(coords).kneighbors(return_distance=False)
    if np.any(indices == np.arange(len(coords))[:, None]):
        raise ValueError("Spatial graph includes self-neighbors")
    return indices


def spatial_autocorrelation(labels, indices):
    n, k = indices.shape
    _, codes = np.unique(labels, return_inverse=True)
    one_hot = np.eye(codes.max() + 1, dtype=np.float64)[codes]
    if one_hot.shape[1] < 2:
        raise ValueError("At least two clusters required")
    weights = csr_matrix((np.full(n*k, 1/k), (np.repeat(np.arange(n), k), indices.ravel())), shape=(n, n))
    centered = one_hot - one_hot.mean(axis=0)
    denominator = np.sum(centered**2, axis=0)
    moran = np.sum(centered * (weights @ centered), axis=0) / denominator
    # For binary membership, squared differences count directed boundary edges.
    boundary_sum = np.zeros(one_hot.shape[1])
    for column in range(one_hot.shape[1]):
        values = one_hot[:, column]
        boundary_sum[column] = np.sum((values[:, None] - values[indices])**2) / k
    geary = (n-1) * boundary_sum / (2 * weights.sum() * denominator)
    return float(moran.mean()), float(geary.mean())


def chaos(labels, coords):
    # SpatialPCA: standardize coordinates with sample SD, then nearest same-cluster
    # spot distances averaged over all spots. Singleton contribution is zero,
    # matching its implementation; record singleton count separately.
    sd = coords.std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError("Constant spatial coordinate dimension")
    scaled = (coords - coords.mean(axis=0)) / sd
    total = 0.0
    for label in np.unique(labels):
        subset = scaled[labels == label]
        if len(subset) > 1:
            distances = NearestNeighbors(n_neighbors=1).fit(subset).kneighbors()[0]
            total += distances.sum()
    return float(total / len(labels))


def refit_mclust(features, k, model, seed):
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, numpy2ri
    from rpy2.robjects.conversion import localconverter
    ro.r("suppressPackageStartupMessages(library(mclust))")
    ro.r["set.seed"](int(seed))
    with localconverter(default_converter + numpy2ri.converter):
        result = ro.r["Mclust"](ro.conversion.py2rpy(features), int(k), model, verbose=False)
    labels = np.asarray(result["classification"] if hasattr(result, "keys") else result.rx2("classification"))
    if len(labels) != len(features):
        raise ValueError("Invalid mclust stability result")
    return labels


def run(config_path, repeats_override=None):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = PROJECT_ROOT
    out = root / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(root / cfg["input_csv"])
    if reference.duplicated(["dataset", "slice", "method", "evaluation_id", "metric_name"]).any():
        raise ValueError("Duplicate benchmark metric keys")
    repeats = int(cfg["stability_repeats"] if repeats_override is None else repeats_override)
    if repeats < 0:
        raise ValueError("stability_repeats must be non-negative")
    rows, replicate_rows, manifests, comparisons = [], [], [], []
    total = len(cfg["slices"]) * len(cfg["methods"])
    counter = 0
    for slice_name in cfg["slices"]:
        print(f"[slice] {slice_name}: loading canonical graph", flush=True)
        canonical_path = root / cfg["prediction_dir"] / cfg["evaluation_id"] / cfg["dataset"] / slice_name / f'{cfg["canonical_method"]}_benchmark.h5ad'
        canonical_names, _, _, canonical_coords = read_prediction(canonical_path, cfg)
        graph6 = neighbors(canonical_coords, int(cfg["spatial_neighbors"]))
        graph10 = neighbors(canonical_coords, int(cfg["pas_original_neighbors"]))
        rng = np.random.default_rng(int(cfg["seed"]))
        subsets = [np.sort(rng.choice(len(canonical_names), int(len(canonical_names)*cfg["stability_fraction"]), replace=False)) for _ in range(repeats)]
        graph_hash = hashlib.sha256(graph6.tobytes()).hexdigest()
        for method in cfg["methods"]:
            counter += 1
            overrides = cfg["method_overrides"].get(method, {})
            evaluation = overrides.get("evaluation_id", cfg["evaluation_id"])
            role = overrides.get("evaluation_role", cfg["evaluation_role"])
            path = root / overrides.get("prediction_dir", cfg["prediction_dir"]) / evaluation / cfg["dataset"] / slice_name / f"{method}_benchmark.h5ad"
            print(f"[{counter}/{total}] {slice_name} / {method}", flush=True)
            names, features, labels, coords = read_prediction(path, cfg)
            if not np.array_equal(names, canonical_names) or not np.array_equal(coords, canonical_coords):
                raise ValueError(f"Canonical spot order or spatial coordinates differ: {path}")
            expected_dim = overrides.get("expected_dim", cfg["expected_dim"])
            if features.shape[1] != expected_dim:
                raise ValueError(f"Expected {expected_dim} dimensions: {path}")
            existing = reference[(reference.dataset == cfg["dataset"]) & (reference['slice'] == slice_name) & (reference.method == method) & (reference.evaluation_id == evaluation) & (reference.evaluation_role == role)]
            if existing.empty or set(existing.seed) != {cfg["seed"]}:
                raise ValueError(f"Missing benchmark metadata or inconsistent seed: {path}")
            observed_k = len(np.unique(labels))
            if len(set(existing.K)) != 1 or observed_k > int(existing.iloc[0].K):
                raise ValueError(f"Invalid observed cluster count: {path}")
            with h5py.File(path, "r") as f:
                stored_k = f["uns/benchmark_metadata/n_clusters_observed"][()]
            if observed_k != int(stored_k):
                raise ValueError(f"Labels disagree with saved observed cluster count: {path}")
            moran, geary = spatial_autocorrelation(labels, graph6)
            original_moran = existing[existing.metric_name == "cluster_morans_i"]
            if len(original_moran) != 1 or original_moran.iloc[0].metric_status != "ok":
                raise ValueError("Missing valid reference Moran's I")
            delta = abs(moran - float(original_moran.iloc[0].metric_value))
            if delta > 1e-5:
                raise ValueError(f"Reconstructed graph Moran's I mismatch {delta}: {path}")
            same6 = (labels[:, None] == labels[graph6]).mean(axis=1)
            same10 = (labels[:, None] == labels[graph10]).mean(axis=1)
            values = {
                "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
                "davies_bouldin": float(davies_bouldin_score(features, labels)),
                "cluster_gearys_c": geary,
                "PAS_k6": float((same6 < .5).mean()),
                "PAS_k10": float((same10 < .5).mean()),
                "neighbor_agreement_k6": float(same6.mean()),
                "CHAOS": chaos(labels, canonical_coords.astype(np.float64)),
            }
            stability_values = []
            for rep, subset in enumerate(subsets):
                print(f"  [stability {rep+1}/{repeats}]", flush=True)
                predicted = refit_mclust(features[subset], int(existing.iloc[0].K), cfg["mclust_model"], cfg["seed"] + rep)
                score = float(adjusted_rand_score(labels[subset], predicted))
                stability_values.append(score)
                replicate_rows.append(dict(dataset=cfg["dataset"], slice=slice_name, method=method, evaluation_id=evaluation, repeat=rep+1, seed=cfg["seed"]+rep, n_spots=len(subset), ARI=score))
                pd.DataFrame(replicate_rows).to_csv(out / "stability_replicates.csv", index=False)
            if stability_values:
                values["subsample_stability_ARI"] = float(np.mean(stability_values))
            new_names = list(cfg["metrics"]) + (["subsample_stability_ARI"] if repeats else [])
            counts = pd.Series(labels).value_counts()
            for order, name in enumerate(new_names, start=2):
                value = values[name]
                if not np.isfinite(value):
                    raise ValueError(f"Non-finite {name}: {path}")
                base = existing.iloc[0].to_dict()
                group, direction = DEFINITIONS[name]
                params = {"prediction_file": str(path.relative_to(root)), "n_clusters_observed": observed_k}
                if group == "spatial":
                    params.update(canonical_method=cfg["canonical_method"], n_neighbors=cfg["pas_original_neighbors"] if name == "PAS_k10" else cfg["spatial_neighbors"], spatial_graph_hash=graph_hash)
                    if name == "cluster_gearys_c":
                        params.update(encoding="one_hot", aggregation="unweighted_cluster_mean", row_normalized=True)
                    elif name.startswith("PAS"):
                        params.update(abnormal_rule="strictly_more_than_half_neighbors_have_different_label")
                    elif name == "CHAOS":
                        params.update(n_neighbors=1, coordinate_standardization="sample_sd_ddof1", singleton_contribution=0)
                if group == "stability":
                    params.update(repeats=repeats, fraction=cfg["stability_fraction"], sampling="without_replacement", sd=float(np.std(stability_values, ddof=1)) if repeats > 1 else None, comparison="original_full_fit_labels_on_sampled_spots", pca_refit=False)
                base.update(metric_group=group, metric_order=order, metric_name=name, metric_value=value, metric_status="ok", metric_requirement="required", metric_direction=direction, metric_params=json.dumps(params, sort_keys=True))
                rows.append(base)
            old = existing[existing.metric_name.isin(["silhouette", "cluster_morans_i"])].to_dict("records")
            comparisons.extend(old)
            comparisons.extend(rows[-len(new_names):])
            manifests.append(dict(slice=slice_name, method=method, n_spots=len(names), requested_K=int(existing.iloc[0].K), n_clusters=len(counts), singleton_clusters=int((counts==1).sum()), dimensions=features.shape[1], canonical_coordinates_identical=True, spatial_graph_hash=graph_hash, moran_reproduction_error=delta, status="ok"))
            pd.DataFrame(rows).to_csv(out / "supplementary_metrics.csv", index=False)
            pd.DataFrame(comparisons).to_csv(out / "metrics_comparison.csv", index=False)
            (out / "run_manifest.json").write_text(json.dumps(dict(config=cfg, stability_repeats_actual=repeats, completed=manifests), ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = pd.DataFrame(comparisons)
    expected = total * (len(cfg["metrics"]) + 2 + bool(repeats))
    if len(comparison) != expected:
        raise ValueError("Incomplete comparison output")
    ranked = []
    for metric, group in comparison.groupby("metric_name"):
        ascending = group.iloc[0].metric_direction == "lower"
        group = group.copy()
        group["slice_rank"] = group.groupby("slice").metric_value.rank(ascending=ascending, method="min")
        summary = group.groupby("method").agg(mean=("metric_value", "mean"), median=("metric_value", "median"), min=("metric_value", "min"), max=("metric_value", "max"), mean_slice_rank=("slice_rank", "mean"), n_slices=("slice", "nunique"))
        summary["mean_value_rank"] = summary["mean"].rank(ascending=ascending, method="min").astype(int)
        standard_methods = [m for m in summary.index if cfg["method_overrides"].get(m, {}).get("expected_dim", cfg["expected_dim"]) == cfg["expected_dim"]]
        summary["standard_dim_mean_value_rank"] = summary.loc[standard_methods, "mean"].rank(ascending=ascending, method="min")
        summary["metric_name"], summary["metric_direction"] = metric, group.iloc[0].metric_direction
        ranked.append(summary.reset_index())
    summary = pd.concat(ranked, ignore_index=True)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    (out / "run_manifest.json").write_text(json.dumps(dict(config=cfg, stability_repeats_actual=repeats, status="complete", n_completed=len(manifests), n_expected=total, completed=manifests), ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary[summary.method == cfg["canonical_method"]].to_string(index=False), flush=True)
    print(f"Saved complete results: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("analysis/benchmark/supplementary_metrics_config.yaml"))
    parser.add_argument("--stability-repeats", type=int)
    args = parser.parse_args()
    run(args.config.resolve(), args.stability_repeats)
