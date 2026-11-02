"""Draw SMART-style RNA and ADT marker heatmaps from fixed cluster labels.

The script consumes original raw RNA/ADT h5ad files plus an evaluated h5ad that
contains fixed cluster assignments. It does not use teacher outputs, pseudo
labels, prototype targets, or branch-ablation labels.
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml

from sf_model.preprocess.adt_process import process_adt_pipeline


warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create RNA/ADT region-marker heatmaps from fixed Bio-SFINet clusters."
    )
    parser.add_argument("--config", required=True, help="YAML config used for fixed evaluation.")
    parser.add_argument("--eval-h5ad", default=None, help="Evaluated embedding h5ad with cluster labels.")
    parser.add_argument("--eval-suffix", default=None, help="Suffix used by run_evaluate.py, e.g. formal_pca20.")
    parser.add_argument("--raw-rna-h5ad", default=None, help="Raw RNA h5ad; defaults to config data.raw_path/files.rna_h5ad.")
    parser.add_argument("--raw-adt-h5ad", default=None, help="Raw ADT h5ad; defaults to config data.raw_path/adt_preprocess.adt_data_path.")
    parser.add_argument("--cluster-key", default=None, help="Cluster column in eval h5ad; default: marker_heatmaps.cluster_key or cluster.")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults beside eval result directory.")
    parser.add_argument("--rna-markers", default=None, help="Comma-separated RNA marker genes, in plot order.")
    parser.add_argument("--adt-markers", default=None, help="Comma-separated ADT marker proteins, in plot order.")
    parser.add_argument("--marker-csv", default=None, help="Optional CSV with columns modality,feature and optional group/order.")
    parser.add_argument("--top-n", type=int, default=None, help="Auto-select this many markers per cluster if markers are not supplied.")
    parser.add_argument(
        "--pval-adj-cutoff",
        type=float,
        default=None,
        help="Optional adjusted p-value cutoff for auto marker selection; default keeps top-N per cluster.",
    )
    parser.add_argument("--target-sum", type=float, default=None, help="RNA library-size normalization target.")
    parser.add_argument("--skip-rna-normalize", action="store_true", help="Use RNA X as stored instead of normalize_total/log1p.")
    parser.add_argument("--skip-adt-preprocess", action="store_true", help="Use ADT X as stored instead of config ADT preprocessing.")
    parser.add_argument("--row-cluster-order", default=None, help="Comma-separated cluster row order.")
    parser.add_argument("--formats", default="pdf,png", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-width", type=float, default=None)
    parser.add_argument("--fig-height", type=float, default=None)
    parser.add_argument("--cmap-rna", default="Blues")
    parser.add_argument("--cmap-adt", default="Greens")
    parser.add_argument("--colorbar-label", default="Z-scored mean expression")
    parser.add_argument("--group-label-fontsize", type=float, default=7.0)
    parser.add_argument("--group-label-rotation", type=float, default=18.0)
    parser.add_argument("--group-label-y", type=float, default=1.09)
    parser.add_argument("--group-bar-y", type=float, default=1.04)
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    return parser.parse_args()


def resolve_path(path_value, base_dir=PROJECT_ROOT):
    if path_value is None:
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_eval_h5ad(config, args):
    if args.eval_h5ad:
        return resolve_path(args.eval_h5ad)
    project_cfg = config.get("project", {}) or {}
    pred_dir = resolve_path(project_cfg.get("eval_dir"))
    if pred_dir is None:
        save_dir = resolve_path(project_cfg.get("save_dir", "results/checkpoints"))
        pred_dir = save_dir.parent / "predictions" if save_dir.name == "checkpoints" else save_dir / "predictions"
    suffix = str(args.eval_suffix).strip() if args.eval_suffix else ""
    name = f"embedding_joint_{suffix}.h5ad" if suffix else "embedding_joint.h5ad"
    return pred_dir / name


def resolve_raw_rna_h5ad(config, args):
    if args.raw_rna_h5ad:
        return resolve_path(args.raw_rna_h5ad)
    data_cfg = config.get("data", {}) or {}
    files_cfg = data_cfg.get("files", {}) or {}
    raw_path = data_cfg.get("raw_path")
    rna_h5ad = files_cfg.get("rna_h5ad")
    if not raw_path or not rna_h5ad:
        raise ValueError("Cannot infer RNA h5ad. Set data.raw_path/data.files.rna_h5ad or pass --raw-rna-h5ad.")
    return resolve_path(raw_path) / str(rna_h5ad)


def resolve_raw_adt_h5ad(config, args):
    if args.raw_adt_h5ad:
        return resolve_path(args.raw_adt_h5ad)
    data_cfg = config.get("data", {}) or {}
    adt_cfg = config.get("adt_preprocess", {}) or {}
    raw_path = data_cfg.get("raw_path")
    adt_name = adt_cfg.get("adt_data_path")
    if not raw_path or not adt_name:
        raise ValueError("Cannot infer ADT h5ad. Set data.raw_path/adt_preprocess.adt_data_path or pass --raw-adt-h5ad.")
    return resolve_path(raw_path) / str(adt_name)


def resolve_output_dir(config, args, eval_h5ad_path):
    if args.output_dir:
        return resolve_path(args.output_dir)
    cfg = config.get("marker_heatmaps", {}) or {}
    if cfg.get("output_dir"):
        return resolve_path(cfg.get("output_dir"))
    pred_dir = eval_h5ad_path.parent
    result_dir = pred_dir.parent if pred_dir.name == "predictions" else pred_dir
    return result_dir / "rna_adt_marker_heatmaps"


def parse_list(text):
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def feature_index(adata):
    mapping = {}
    for idx, name in enumerate(adata.var_names.astype(str)):
        mapping.setdefault(name.upper(), idx)
    for col in ["gene_symbols", "gene_symbol", "gene_ids", "features", "feature_name"]:
        if col in adata.var:
            for idx, name in enumerate(adata.var[col].astype(str)):
                mapping.setdefault(name.upper(), idx)
    return mapping


def resolve_features(adata, features, modality):
    idx = feature_index(adata)
    rows = []
    missing = []
    used = set()
    for feature in features:
        key = str(feature).upper()
        if key not in idx:
            missing.append(str(feature))
            continue
        pos = idx[key]
        if pos in used:
            continue
        used.add(pos)
        rows.append({"modality": modality, "feature": str(feature), "var_name": str(adata.var_names[pos]), "var_idx": int(pos)})
    return pd.DataFrame(rows), missing


def align_labels(adata, eval_adata, cluster_key):
    if cluster_key not in eval_adata.obs:
        available = ", ".join(map(str, eval_adata.obs.columns))
        raise KeyError(f"Cluster key '{cluster_key}' not found in eval h5ad obs. Available obs columns: {available}")
    if adata.n_obs != eval_adata.n_obs:
        raise ValueError(f"Spot count mismatch: data={adata.n_obs}, eval={eval_adata.n_obs}")
    if adata.obs_names.equals(eval_adata.obs_names) or set(adata.obs_names) == set(eval_adata.obs_names):
        labels = eval_adata.obs.loc[adata.obs_names, cluster_key]
    else:
        print("   obs_names differ; assigning cluster labels by row order after spot-count check.")
        labels = eval_adata.obs[cluster_key]
    adata.obs[cluster_key] = labels.to_numpy()
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str).astype("category")
    return adata


def prepare_rna(adata, target_sum, skip_normalize):
    adata = adata.copy()
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    if not skip_normalize:
        sc.pp.normalize_total(adata, target_sum=float(target_sum))
        sc.pp.log1p(adata)
    return adata


def prepare_adt(adata, config, skip_preprocess):
    adata = adata.copy()
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    if skip_preprocess:
        return adata
    cfg = config.get("adt_preprocess", {}) or {}
    return process_adt_pipeline(
        adata,
        apply_clr=bool(cfg.get("apply_clr", False)),
        apply_scale=bool(cfg.get("apply_scale", False)),
        clip_nonnegative=bool(cfg.get("clip_nonnegative", True)),
        min_cells=cfg.get("min_cells"),
        pipeline=cfg.get("pipeline", "legacy"),
        nonnegative_policy=cfg.get("nonnegative_policy"),
    )


def auto_markers(adata, cluster_key, modality, top_n, pval_adj_cutoff):
    key_added = f"rank_{modality.lower()}_groups"
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon", use_raw=False, key_added=key_added)
    result = adata.uns[key_added]
    frames = []
    for group in result["names"].dtype.names:
        df = pd.DataFrame(
            {
                "modality": modality,
                "group": str(group),
                "feature": result["names"][group],
                "score": result["scores"][group],
                "logfoldchange": result["logfoldchanges"][group],
                "p_value": result["pvals"][group],
                "p_value_adj": result["pvals_adj"][group],
            }
        )
        if pval_adj_cutoff is not None:
            df = df[df["p_value_adj"] < float(pval_adj_cutoff)]
        frames.append(df.head(int(top_n)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_marker_table(args, rna_adata, adt_adata, cluster_key, top_n, pval_adj_cutoff):
    records = []
    if args.marker_csv:
        marker_path = resolve_path(args.marker_csv)
        df = pd.read_csv(marker_path)
        lower = {col.lower(): col for col in df.columns}
        if "feature" not in lower or "modality" not in lower:
            raise ValueError("--marker-csv must contain modality and feature columns.")
        df = df.rename(columns={lower["feature"]: "feature", lower["modality"]: "modality"})
        if "group" not in df.columns and "Group" in df.columns:
            df = df.rename(columns={"Group": "group"})
        if "order" in df.columns:
            df = df.sort_values("order")
        records.append(df[[col for col in ["modality", "feature", "group"] if col in df.columns]])
    if args.rna_markers:
        records.append(pd.DataFrame({"modality": "RNA", "feature": parse_list(args.rna_markers)}))
    if args.adt_markers:
        records.append(pd.DataFrame({"modality": "ADT", "feature": parse_list(args.adt_markers)}))
    if records:
        return pd.concat(records, ignore_index=True)

    rna_auto = auto_markers(rna_adata, cluster_key, "RNA", top_n, pval_adj_cutoff)
    adt_auto = auto_markers(adt_adata, cluster_key, "ADT", top_n, pval_adj_cutoff)
    return pd.concat([rna_auto, adt_auto], ignore_index=True)


def dense_slice(adata, indices):
    X = adata.X[:, indices]
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def cluster_order(labels, requested=None):
    present = pd.Index(labels.astype(str).unique())
    if requested:
        order = [str(x) for x in requested if str(x) in set(present)]
        order.extend([str(x) for x in present if str(x) not in set(order)])
        return order
    def key(value):
        try:
            return (0, int(value))
        except ValueError:
            return (1, str(value))
    return sorted([str(x) for x in present], key=key)


def mean_matrix(adata, selected, cluster_key, order):
    if selected.empty:
        return pd.DataFrame(index=order), pd.DataFrame(index=order)
    X = dense_slice(adata, selected["var_idx"].to_list())
    labels = adata.obs[cluster_key].astype(str).to_numpy()
    raw = []
    for group in order:
        mask = labels == str(group)
        raw.append(X[mask].mean(axis=0) if np.any(mask) else np.full(X.shape[1], np.nan))
    raw_df = pd.DataFrame(raw, index=order, columns=selected["feature"].astype(str).to_list())
    mean = raw_df.mean(axis=0)
    std = raw_df.std(axis=0, ddof=0).replace(0, np.nan)
    z_df = ((raw_df - mean) / std).fillna(0.0).clip(-3.0, 3.0)
    return raw_df, z_df


def infer_feature_groups(z_df, selected):
    if "group" in selected.columns:
        groups = selected["group"].fillna("").astype(str).to_list()
    else:
        groups = [str(z_df[col].idxmax()) for col in z_df.columns]
    return [group if group else "unassigned" for group in groups]


def add_top_group_labels(ax, groups, fontsize, rotation, label_y, bar_y):
    if not groups:
        return
    spans = []
    start = 0
    for idx in range(1, len(groups) + 1):
        if idx == len(groups) or groups[idx] != groups[start]:
            spans.append((start, idx - 1, groups[start]))
            start = idx
    trans = ax.get_xaxis_transform()
    for start, end, label in spans:
        center = (start + end + 1) / 2.0
        ax.text(
            center,
            label_y,
            str(label),
            ha="center",
            va="bottom",
            rotation=rotation,
            fontsize=fontsize,
            transform=trans,
        )
        ax.plot([start, end + 1], [bar_y, bar_y], color="black", linewidth=0.7, transform=trans, clip_on=False)


def save_heatmap(z_df, selected, output_dir, stem, cmap, args):
    if z_df.empty or z_df.shape[1] == 0:
        return []
    n_rows, n_cols = z_df.shape
    fig_width = args.fig_width or max(4.5, 0.28 * n_cols + 1.4)
    fig_height = args.fig_height or max(2.2, 0.25 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(z_df.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap, vmin=args.vmin, vmax=args.vmax)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(z_df.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(z_df.index, fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_top_group_labels(
        ax,
        infer_feature_groups(z_df, selected),
        fontsize=args.group_label_fontsize,
        rotation=args.group_label_rotation,
        label_y=args.group_label_y,
        bar_y=args.group_bar_y,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(args.colorbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    paths = []
    for fmt in parse_list(args.formats):
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    cfg = config.get("marker_heatmaps", {}) or {}
    cluster_key = args.cluster_key or cfg.get("cluster_key", "cluster")
    top_n = int(args.top_n if args.top_n is not None else cfg.get("top_n", 2))
    pval_adj_cutoff = args.pval_adj_cutoff if args.pval_adj_cutoff is not None else cfg.get("pval_adj_cutoff")
    target_sum = float(args.target_sum if args.target_sum is not None else cfg.get("target_sum", 1e4))

    eval_h5ad_path = resolve_eval_h5ad(config, args)
    rna_path = resolve_raw_rna_h5ad(config, args)
    adt_path = resolve_raw_adt_h5ad(config, args)
    output_dir = resolve_output_dir(config, args, eval_h5ad_path)

    for label, path in [("config", config_path), ("eval h5ad", eval_h5ad_path), ("RNA h5ad", rna_path), ("ADT h5ad", adt_path)]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading fixed labels and original modality data...")
    eval_adata = sc.read_h5ad(eval_h5ad_path)
    rna = prepare_rna(sc.read_h5ad(rna_path), target_sum, args.skip_rna_normalize)
    adt = prepare_adt(sc.read_h5ad(adt_path), config, args.skip_adt_preprocess)

    print(f"2. Aligning cluster labels from obs['{cluster_key}']...")
    rna = align_labels(rna, eval_adata, cluster_key)
    adt = align_labels(adt, eval_adata, cluster_key)
    order = cluster_order(rna.obs[cluster_key].astype(str), parse_list(args.row_cluster_order))

    print("3. Selecting marker features...")
    marker_table = load_marker_table(args, rna, adt, cluster_key, top_n, pval_adj_cutoff)
    if marker_table.empty:
        raise ValueError(
            "No marker features were selected. If using auto marker selection, rerun without "
            "--pval-adj-cutoff or increase --top-n; if using fixed markers, check feature names."
        )
    marker_table["modality"] = marker_table["modality"].astype(str).str.upper()
    rna_features = marker_table.loc[marker_table["modality"].eq("RNA"), "feature"].astype(str).drop_duplicates().to_list()
    adt_features = marker_table.loc[marker_table["modality"].eq("ADT"), "feature"].astype(str).drop_duplicates().to_list()
    rna_selected, rna_missing = resolve_features(rna, rna_features, "RNA")
    adt_selected, adt_missing = resolve_features(adt, adt_features, "ADT")

    for selected in [rna_selected, adt_selected]:
        if not selected.empty and "group" in marker_table.columns:
            group_map = marker_table.dropna(subset=["feature"]).drop_duplicates(["modality", "feature"])
            selected["group"] = selected.merge(group_map[["modality", "feature", "group"]], on=["modality", "feature"], how="left")["group"].to_numpy()

    print("4. Computing cluster mean and row-z-score matrices...")
    rna_raw, rna_z = mean_matrix(rna, rna_selected, cluster_key, order)
    adt_raw, adt_z = mean_matrix(adt, adt_selected, cluster_key, order)

    rna_selected.to_csv(output_dir / "selected_rna_markers.csv", index=False)
    adt_selected.to_csv(output_dir / "selected_adt_markers.csv", index=False)
    rna_raw.to_csv(output_dir / "rna_marker_cluster_mean_matrix.csv")
    rna_z.to_csv(output_dir / "rna_marker_cluster_zscore_matrix.csv")
    adt_raw.to_csv(output_dir / "adt_marker_cluster_mean_matrix.csv")
    adt_z.to_csv(output_dir / "adt_marker_cluster_zscore_matrix.csv")

    print("5. Saving heatmaps...")
    figure_paths = []
    figure_paths.extend(save_heatmap(rna_z, rna_selected, output_dir, "rna_marker_heatmap", args.cmap_rna, args))
    figure_paths.extend(save_heatmap(adt_z, adt_selected, output_dir, "adt_marker_heatmap", args.cmap_adt, args))

    manifest = {
        "config": str(config_path),
        "eval_h5ad": str(eval_h5ad_path),
        "raw_rna_h5ad": str(rna_path),
        "raw_adt_h5ad": str(adt_path),
        "cluster_key": cluster_key,
        "cluster_order": order,
        "rna_missing_markers": rna_missing,
        "adt_missing_markers": adt_missing,
        "figure_paths": [str(path) for path in figure_paths],
        "notes": "Heatmaps use original modality h5ad inputs plus fixed evaluated cluster labels; columns are marker features and rows are clusters.",
    }
    with open(output_dir / "marker_heatmap_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"   Output directory: {output_dir.resolve()}")
    if rna_missing:
        print("   Missing RNA markers: " + ", ".join(rna_missing))
    if adt_missing:
        print("   Missing ADT markers: " + ", ".join(adt_missing))
    print("RNA/ADT marker heatmaps complete.")


if __name__ == "__main__":
    main()
