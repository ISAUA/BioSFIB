import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import yaml


warnings.filterwarnings("ignore", message=".*TBB threading layer.*")


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Find marker genes from evaluated Bio-SFINet clusters")
    parser.add_argument("--config", default="configs/misar/e18_5_s1.yaml", help="Path to YAML config file")
    parser.add_argument("--eval-h5ad", default=None, help="Evaluated embedding h5ad; defaults to config project.eval_dir")
    parser.add_argument("--eval-suffix", default=None, help="Suffix used by run_evaluate.py, e.g. epoch_1500")
    parser.add_argument("--raw-rna-h5ad", default=None, help="Raw RNA h5ad; defaults to data.raw_path/files.rna_h5ad")
    parser.add_argument("--cluster-key", default=None, help="Cluster column in eval h5ad; defaults to marker_genes.cluster_key or cluster")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults beside the configured result directory")
    parser.add_argument("--top-n", type=int, default=None, help="Top marker genes per cluster")
    parser.add_argument("--pval-adj-cutoff", type=float, default=None, help="Adjusted p-value cutoff")
    parser.add_argument("--method", default=None, choices=["wilcoxon", "t-test", "logreg"], help="rank_genes_groups method")
    parser.add_argument("--target-sum", type=float, default=None, help="Library-size normalization target")
    parser.add_argument("--spatial-top-clusters", type=int, default=None, help="Number of top cluster markers for spatial plot")
    return parser.parse_args()


def resolve_path(path_value, base_dir=PROJECT_ROOT):
    if path_value is None:
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_eval_h5ad(config, args):
    if args.eval_h5ad:
        return resolve_path(args.eval_h5ad)

    project_cfg = config.get("project", {}) or {}
    eval_dir = project_cfg.get("eval_dir")
    if eval_dir:
        pred_dir = resolve_path(eval_dir)
    else:
        save_dir = resolve_path(project_cfg.get("save_dir", "results/checkpoints"))
        if save_dir.name == "checkpoints":
            pred_dir = save_dir.parent / "predictions"
        else:
            pred_dir = save_dir / "predictions"

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
        raise ValueError("Cannot infer raw RNA h5ad. Set data.raw_path/data.files.rna_h5ad or pass --raw-rna-h5ad.")
    return resolve_path(raw_path) / str(rna_h5ad)


def resolve_output_dir(config, args):
    if args.output_dir:
        return resolve_path(args.output_dir)

    marker_cfg = config.get("marker_genes", {}) or {}
    if marker_cfg.get("output_dir"):
        return resolve_path(marker_cfg.get("output_dir"))

    project_cfg = config.get("project", {}) or {}
    eval_dir = project_cfg.get("eval_dir")
    if eval_dir:
        eval_path = resolve_path(eval_dir)
        result_dir = eval_path.parent if eval_path.name == "predictions" else eval_path
        return result_dir / "marker_genes"

    save_dir = resolve_path(project_cfg.get("save_dir", "results/checkpoints"))
    base_dir = save_dir.parent if save_dir.name == "checkpoints" else save_dir
    return base_dir / "marker_genes"


def align_cluster_labels(adata_raw, adata_eval, cluster_key):
    if cluster_key not in adata_eval.obs:
        available = ", ".join(map(str, adata_eval.obs.columns))
        raise KeyError(f"Cluster key '{cluster_key}' not found in eval h5ad obs. Available obs columns: {available}")

    if adata_raw.n_obs != adata_eval.n_obs:
        raise ValueError(f"Spot count mismatch: raw={adata_raw.n_obs}, eval={adata_eval.n_obs}")

    if adata_raw.obs_names.equals(adata_eval.obs_names):
        labels = adata_eval.obs.loc[adata_raw.obs_names, cluster_key]
    elif set(adata_raw.obs_names) == set(adata_eval.obs_names):
        print("   obs_names order differs; aligning cluster labels by obs_names.")
        labels = adata_eval.obs.loc[adata_raw.obs_names, cluster_key]
    else:
        print("   obs_names differ; assigning cluster labels by row order after spot-count check.")
        labels = adata_eval.obs[cluster_key]

    adata_raw.obs[cluster_key] = labels.to_numpy()
    adata_raw.obs[cluster_key] = adata_raw.obs[cluster_key].astype(str).astype("category")
    return adata_raw


def save_ranked_markers(adata, cluster_key, key_added, output_dir, top_n, pval_adj_cutoff):
    result = adata.uns[key_added]
    groups = result["names"].dtype.names
    frames = []
    for group in groups:
        group_df = pd.DataFrame(
            {
                "Cluster": group,
                "Gene": result["names"][group],
                "Scores": result["scores"][group],
                "LogFoldChange": result["logfoldchanges"][group],
                "P_value": result["pvals"][group],
                "P_value_adj": result["pvals_adj"][group],
            }
        )
        if pval_adj_cutoff is not None:
            group_df = group_df[group_df["P_value_adj"] < float(pval_adj_cutoff)]
        frames.append(group_df.head(int(top_n)))

    final_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    csv_path = output_dir / f"top{top_n}_marker_genes_by_{cluster_key}.csv"
    final_df.to_csv(csv_path, index=False)
    return csv_path, final_df


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    marker_cfg = config.get("marker_genes", {}) or {}

    eval_h5ad_path = resolve_eval_h5ad(config, args)
    raw_rna_h5ad_path = resolve_raw_rna_h5ad(config, args)
    output_dir = resolve_output_dir(config, args)
    cluster_key = args.cluster_key or marker_cfg.get("cluster_key", "cluster")
    top_n = int(args.top_n if args.top_n is not None else marker_cfg.get("top_n", 50))
    pval_adj_cutoff = args.pval_adj_cutoff if args.pval_adj_cutoff is not None else marker_cfg.get("pval_adj_cutoff", 0.01)
    method = args.method or marker_cfg.get("method", "wilcoxon")
    target_sum = float(args.target_sum if args.target_sum is not None else marker_cfg.get("target_sum", 1e4))
    spatial_top_clusters = int(
        args.spatial_top_clusters
        if args.spatial_top_clusters is not None
        else marker_cfg.get("spatial_top_clusters", 3)
    )

    if not eval_h5ad_path.exists():
        raise FileNotFoundError(f"Evaluated h5ad not found: {eval_h5ad_path}")
    if not raw_rna_h5ad_path.exists():
        raise FileNotFoundError(f"Raw RNA h5ad not found: {raw_rna_h5ad_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading data...")
    print(f"   eval_h5ad: {eval_h5ad_path}")
    print(f"   raw_rna_h5ad: {raw_rna_h5ad_path}")
    adata_eval = sc.read_h5ad(eval_h5ad_path)
    adata_raw = sc.read_h5ad(raw_rna_h5ad_path)
    if not adata_raw.var_names.is_unique:
        n_duplicated = int(adata_raw.var_names.duplicated().sum())
        print(f"   raw RNA var_names has {n_duplicated} duplicated entries; making names unique for Scanpy plots.")
        adata_raw.var_names_make_unique()

    print(f"2. Aligning evaluation cluster labels from obs['{cluster_key}']...")
    adata_raw = align_cluster_labels(adata_raw, adata_eval, cluster_key)
    if "spatial" in adata_eval.obsm:
        adata_raw.obsm["spatial"] = adata_eval.obsm["spatial"].copy()

    print("3. Normalizing RNA counts and applying log1p...")
    adata_raw.layers["counts"] = adata_raw.X.copy()
    sc.pp.normalize_total(adata_raw, target_sum=target_sum)
    sc.pp.log1p(adata_raw)

    key_added = "rank_genes_groups"
    print(f"4. Running marker gene test: {method}...")
    sc.tl.rank_genes_groups(
        adata_raw,
        groupby=cluster_key,
        method=method,
        use_raw=False,
        key_added=key_added,
    )

    print("5. Saving marker table...")
    csv_path, final_df = save_ranked_markers(
        adata_raw,
        cluster_key=cluster_key,
        key_added=key_added,
        output_dir=output_dir,
        top_n=top_n,
        pval_adj_cutoff=pval_adj_cutoff,
    )
    print(f"   Marker table: {csv_path.resolve()}")

    print("6. Saving marker plots...")
    sc.settings.set_figure_params(dpi=150, format="pdf")
    dotplot_path = output_dir / f"marker_dotplot_by_{cluster_key}.pdf"
    sc.pl.rank_genes_groups_dotplot(
        adata_raw,
        n_genes=min(5, top_n),
        key=key_added,
        groupby=cluster_key,
        standard_scale="var",
        show=False,
    )
    plt.savefig(dotplot_path, bbox_inches="tight")
    plt.close()
    print(f"   Dotplot: {dotplot_path.resolve()}")

    if "spatial" in adata_raw.obsm and not final_df.empty and spatial_top_clusters > 0:
        genes = (
            final_df.groupby("Cluster", sort=False)["Gene"]
            .first()
            .dropna()
            .astype(str)
            .head(spatial_top_clusters)
            .tolist()
        )
        genes = [gene for gene in genes if gene in adata_raw.var_names]
        if genes:
            spatial_path = output_dir / f"marker_spatial_map_by_{cluster_key}.pdf"
            sc.pl.embedding(
                adata_raw,
                basis="spatial",
                color=[cluster_key] + genes,
                show=False,
            )
            plt.savefig(spatial_path, bbox_inches="tight")
            plt.close()
            print(f"   Spatial map: {spatial_path.resolve()}")

    print("Marker gene analysis complete.")


if __name__ == "__main__":
    main()
