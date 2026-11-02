"""Spatial-domain report for fixed MISAR epoch-1300 predictions."""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/misar/e18_5_s1.yaml"
DEFAULT_H5AD = "results/MISAR/e18_5_s1/predictions/embedding_joint.h5ad"
DEFAULT_OUTPUT_DIR = "results/MISAR/e18_5_s1/downstream_analyses_epoch1300/spatial_domain_report"
TRUTH_CANDIDATES = ["Combined_Clusters", "ground_truth", "truth", "label", "labels", "annotation", "domain"]
PRED_CANDIDATES = ["cluster", "mclust", "leiden", "prediction", "pred"]


def parse_args():
    parser = argparse.ArgumentParser(description="Create spatial-domain metrics, maps, and confusion tables.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--h5ad", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--truth-key", default=None)
    parser.add_argument("--pred-key", default=None)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--point-size", type=float, default=18.0)
    return parser.parse_args()


def resolve_path(path_value, base_dir=PROJECT_ROOT):
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_h5ad(config, h5ad_arg):
    if h5ad_arg:
        return resolve_path(h5ad_arg)
    eval_dir = config.get("project", {}).get("eval_dir")
    if eval_dir:
        candidate = resolve_path(eval_dir) / "embedding_joint.h5ad"
        if candidate.exists():
            return candidate
    return resolve_path(DEFAULT_H5AD)


def resolve_raw_rna_h5ad(config):
    data_cfg = config.get("data", {}) or {}
    raw_path = data_cfg.get("raw_path")
    rna_h5ad = (data_cfg.get("files", {}) or {}).get("rna_h5ad")
    if not raw_path or not rna_h5ad:
        return None
    rna_path = Path(str(rna_h5ad)).expanduser()
    if rna_path.is_absolute():
        return rna_path
    return resolve_path(raw_path) / rna_path


def add_truth_from_raw_if_needed(adata, config, truth_key):
    if not truth_key or truth_key in adata.obs:
        return None
    raw_rna_path = resolve_raw_rna_h5ad(config)
    if raw_rna_path is None or not raw_rna_path.exists():
        return "truth key %s missing from prediction h5ad and raw RNA h5ad was not found" % truth_key
    raw = sc.read_h5ad(raw_rna_path)
    if truth_key not in raw.obs:
        return "truth key %s missing from prediction h5ad and raw RNA obs" % truth_key
    if set(adata.obs_names.astype(str)) == set(raw.obs_names.astype(str)):
        truth = raw.obs.loc[adata.obs_names.astype(str), truth_key].to_numpy()
    elif adata.n_obs == raw.n_obs:
        truth = raw.obs[truth_key].to_numpy()
    else:
        return "truth key %s could not be aligned from raw RNA h5ad" % truth_key
    adata.obs[truth_key] = truth
    return "truth key %s copied from raw RNA h5ad: %s" % (truth_key, raw_rna_path)


def pick_key(obs, explicit_key, candidates, label):
    if explicit_key:
        if explicit_key not in obs:
            for key in candidates:
                if key in obs:
                    return key
            raise KeyError("%s key not found in h5ad obs: %s" % (label, explicit_key))
        return explicit_key
    for key in candidates:
        if key in obs:
            return key
    raise KeyError("Could not find %s key. Tried: %s" % (label, ", ".join(candidates)))


def valid_domain_frame(adata, truth_key, pred_key):
    df = adata.obs[[truth_key, pred_key]].copy()
    df.columns = ["truth", "pred"]
    df = df.astype(str)
    mask = (~df["truth"].isin(["nan", "None", "NA", ""])) & (~df["pred"].isin(["nan", "None", "NA", ""]))
    return df.loc[mask].copy()


def compute_metrics(df):
    truth = df["truth"].to_numpy()
    pred = df["pred"].to_numpy()
    return {
        "ARI": float(adjusted_rand_score(truth, pred)),
        "NMI": float(normalized_mutual_info_score(truth, pred)),
        "AMI": float(adjusted_mutual_info_score(truth, pred)),
        "HOM": float(homogeneity_score(truth, pred)),
        "n_spots": int(df.shape[0]),
        "n_truth_domains": int(pd.Series(truth).nunique()),
        "n_pred_domains": int(pd.Series(pred).nunique()),
    }


def get_spatial(adata):
    if "spatial" in adata.obsm:
        arr = np.asarray(adata.obsm["spatial"])
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]
    for cols in [("x", "y"), ("array_row", "array_col")]:
        if cols[0] in adata.obs and cols[1] in adata.obs:
            return adata.obs.loc[:, list(cols)].to_numpy(dtype=float)
    return None


def categorical_codes(values):
    cats = pd.Categorical(values.astype(str))
    return cats.codes, list(cats.categories)


def draw_two_panel(coords, truth, pred, title, output_dir, basename, formats, dpi, point_size, invert_y=False):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), dpi=int(dpi))
    for ax, values, panel_title in zip(axes, [truth, pred], ["Truth", "Predicted"]):
        codes, categories = categorical_codes(pd.Series(values))
        sca = ax.scatter(coords[:, 0], coords[:, 1], c=codes, s=float(point_size), cmap="tab20", linewidths=0)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        if invert_y:
            ax.invert_yaxis()
        handles = []
        for idx, cat in enumerate(categories[:25]):
            handles.append(plt.Line2D([], [], marker="o", linestyle="", color=sca.cmap(sca.norm(idx)), label=str(cat), markersize=5))
        if handles:
            ax.legend(handles=handles, fontsize=6, loc="best", frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    paths = []
    for fmt in formats:
        path = output_dir / ("%s.%s" % (basename, fmt))
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def ensure_umap(adata):
    if "X_umap" in adata.obsm:
        return np.asarray(adata.obsm["X_umap"])
    emb = None
    for key in ["X_pca", "X", "embedding", "z_out"]:
        if key in adata.obsm:
            emb = np.asarray(adata.obsm[key])
            break
    if emb is None:
        emb = np.asarray(adata.X)
    tmp = sc.AnnData(emb)
    sc.pp.neighbors(tmp, n_neighbors=min(15, max(2, tmp.n_obs - 1)), use_rep="X")
    sc.tl.umap(tmp)
    adata.obsm["X_umap"] = tmp.obsm["X_umap"]
    return np.asarray(adata.obsm["X_umap"])


def save_confusion_heatmap(matrix, output_dir, formats, dpi):
    if matrix.empty:
        return []
    fig_width = max(6.0, 0.35 * matrix.shape[1])
    fig_height = max(5.0, 0.35 * matrix.shape[0])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=int(dpi))
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(str), rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.astype(str), fontsize=7)
    ax.set_xlabel("Predicted cluster")
    ax.set_ylabel("Truth domain")
    ax.set_title("Truth vs predicted cluster confusion")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="row fraction")
    fig.tight_layout()
    paths = []
    for fmt in formats:
        path = output_dir / ("truth_pred_confusion_heatmap.%s" % fmt)
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    h5ad_path = resolve_h5ad(config, args.h5ad)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not h5ad_path.exists():
        raise FileNotFoundError("Prediction h5ad not found: %s" % h5ad_path)
    adata = sc.read_h5ad(h5ad_path)
    configured_truth_key = args.truth_key or config.get("data", {}).get("parameters", {}).get("ground_truth_key")
    truth_note = add_truth_from_raw_if_needed(adata, config, configured_truth_key)
    truth_key = pick_key(adata.obs, configured_truth_key, TRUTH_CANDIDATES, "truth")
    pred_key = pick_key(adata.obs, args.pred_key, PRED_CANDIDATES, "prediction")
    df = valid_domain_frame(adata, truth_key, pred_key)
    obs_positions = adata.obs.index.get_indexer(df.index)
    metrics = compute_metrics(df)
    metrics_path = output_dir / "spatial_domain_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    counts = pd.crosstab(df["truth"], df["pred"])
    row_norm = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    counts_path = output_dir / "truth_pred_confusion_counts.csv"
    row_norm_path = output_dir / "truth_pred_confusion_row_normalized.csv"
    counts.to_csv(counts_path)
    row_norm.to_csv(row_norm_path)
    figure_paths = []
    spatial = get_spatial(adata)
    if spatial is not None:
        figure_paths.extend(draw_two_panel(spatial[obs_positions], df["truth"], df["pred"], "Spatial domains", output_dir, "spatial_domain_maps", args.formats, args.dpi, args.point_size, invert_y=True))
    umap = ensure_umap(adata)
    figure_paths.extend(draw_two_panel(umap[obs_positions], df["truth"], df["pred"], "UMAP domains", output_dir, "umap_domain_maps", args.formats, args.dpi, args.point_size, invert_y=False))
    figure_paths.extend(save_confusion_heatmap(row_norm, output_dir, args.formats, args.dpi))
    manifest = {
        "config": str(config_path),
        "h5ad": str(h5ad_path),
        "truth_key": truth_key,
        "pred_key": pred_key,
        "truth_note": truth_note,
        "metrics": metrics,
        "metrics_csv": str(metrics_path),
        "truth_pred_confusion_counts": str(counts_path),
        "truth_pred_confusion_row_normalized": str(row_norm_path),
        "figures": [str(path) for path in figure_paths],
    }
    manifest_path = output_dir / "spatial_domain_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[spatial-domain report] ARI={ARI:.6f} NMI={NMI:.6f} AMI={AMI:.6f} HOM={HOM:.6f}".format(**metrics))
    print("  manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()
