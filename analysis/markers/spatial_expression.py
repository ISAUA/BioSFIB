"""Draw spatial expression panels for selected genes from an RNA h5ad file."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = "results/spatial_gene_expression_panel"


def parse_args():
    parser = argparse.ArgumentParser(description="Create a multi-panel spatial map of log-normalized gene expression.")
    parser.add_argument("--rna-h5ad", required=True, help="Raw or normalized RNA h5ad containing genes and spatial coordinates.")
    parser.add_argument("--genes", nargs="+", default=None, help="Optional genes to plot. If omitted, use --auto-from-dir or explicit result CSV inputs.")
    parser.add_argument("--auto-from-dir", default=None, help="Downstream root containing marker_genes, ssr_residual_boundary_genes, and rna_atac_regulatory_relaxed subdirectories.")
    parser.add_argument("--marker-csv", default=None, help="Marker gene CSV, usually <OUT_ROOT>/marker_genes/top50_marker_genes_by_cluster.csv.")
    parser.add_argument("--residual-csv", default=None, help="Residual boundary gene CSV, usually <OUT_ROOT>/ssr_residual_boundary_genes/global_residual_boundary_genes.csv.")
    parser.add_argument("--regulatory-csv", default=None, help="Significant peak-gene links CSV, usually <OUT_ROOT>/rna_atac_regulatory_relaxed/significant_peak_gene_links.csv.")
    parser.add_argument("--top-marker-per-cluster", type=int, default=1, help="Number of top marker genes to take from each cluster.")
    parser.add_argument("--top-residual-genes", type=int, default=4, help="Number of top global residual boundary genes to take.")
    parser.add_argument("--top-regulatory-genes", type=int, default=4, help="Number of top regulatory target genes to take by link count and max abs Pearson.")
    parser.add_argument("--max-genes", type=int, default=12, help="Maximum total genes after merging manual and automatic selections.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--title", default="Log-normalized gene expression")
    parser.add_argument("--target-sum", type=float, default=1e4, help="Library size for normalize_total before log1p.")
    parser.add_argument("--use-raw-values", action="store_true", help="Skip normalize_total/log1p and plot X as-is.")
    parser.add_argument("--spatial-key", default="spatial", help="obsm key for 2D spatial coordinates.")
    parser.add_argument("--x-column", default=None, help="Fallback obs x-coordinate column if obsm spatial is absent.")
    parser.add_argument("--y-column", default=None, help="Fallback obs y-coordinate column if obsm spatial is absent.")
    parser.add_argument("--ncols", type=int, default=2, help="Number of panel columns; 12 genes use a 6-row by 2-column layout by default.")
    parser.add_argument("--panel-size", type=float, default=2.6, help="Panel width/height in inches.")
    parser.add_argument("--axis-label-font-size", type=float, default=10.0, help="Font size in points for the spatial1/spatial2 axis labels.")
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax-quantile", type=float, default=0.995, help="Per-gene color max quantile; set 1.0 for max.")
    parser.add_argument("--shared-vmax", action="store_true", help="Use one vmax across all plotted genes.")
    parser.add_argument("--sort-by-expression", action="store_true", help="Plot higher-expression spots above lower-expression spots.")
    parser.add_argument("--rotate-deg", type=float, default=0.0, help="Clockwise rotation of spatial coordinates in degrees before plotting.")
    parser.add_argument("--invert-y", action="store_true", help="Invert y axis for image-like spatial coordinates.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def resolve_path(path_value, base_dir=PROJECT_ROOT):
    if path_value is None:
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def read_csv_or_empty(path):
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def first_existing(paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


def infer_auto_paths(args):
    root = resolve_path(args.auto_from_dir) if args.auto_from_dir else None
    marker_path = resolve_path(args.marker_csv) if args.marker_csv else None
    residual_path = resolve_path(args.residual_csv) if args.residual_csv else None
    regulatory_path = resolve_path(args.regulatory_csv) if args.regulatory_csv else None
    if root is not None:
        marker_path = marker_path or first_existing([root / "marker_genes" / "top50_marker_genes_by_cluster.csv"])
        residual_path = residual_path or first_existing([
            root / "ssr_residual_boundary_genes" / "global_residual_boundary_genes.csv",
            root / "ssr_residual_boundary_genes" / "global_residual_boundary_genes_unfiltered.csv",
        ])
        regulatory_path = regulatory_path or first_existing([
            root / "rna_atac_regulatory_relaxed" / "significant_peak_gene_links.csv",
            root / "rna_atac_regulatory" / "significant_peak_gene_links.csv",
        ])
    return marker_path, residual_path, regulatory_path


def append_unique(items, values, source, records):
    seen = {str(item).lower() for item in items}
    for value in values:
        gene = str(value).strip()
        if not gene or gene.lower() in seen:
            continue
        items.append(gene)
        seen.add(gene.lower())
        records.append({"gene": gene, "source": source})


def select_marker_genes(marker_path, top_per_cluster):
    df = read_csv_or_empty(marker_path)
    if df.empty:
        return []
    gene_col = "Gene" if "Gene" in df.columns else ("gene" if "gene" in df.columns else None)
    cluster_col = "Cluster" if "Cluster" in df.columns else ("cluster" if "cluster" in df.columns else None)
    if gene_col is None:
        return []
    if cluster_col is None:
        return df[gene_col].astype(str).head(int(top_per_cluster)).tolist()
    sort_cols = [cluster_col]
    ascending = [True]
    if "P_value_adj" in df.columns:
        df["_p_adj"] = pd.to_numeric(df["P_value_adj"], errors="coerce")
        sort_cols.append("_p_adj"); ascending.append(True)
    if "Scores" in df.columns:
        df["_score"] = pd.to_numeric(df["Scores"], errors="coerce")
        sort_cols.append("_score"); ascending.append(False)
    df = df.sort_values(sort_cols, ascending=ascending)
    return df.groupby(cluster_col, observed=False).head(int(top_per_cluster))[gene_col].astype(str).tolist()


def select_residual_genes(residual_path, top_n):
    df = read_csv_or_empty(residual_path)
    if df.empty:
        return []
    gene_col = "Gene" if "Gene" in df.columns else ("gene" if "gene" in df.columns else None)
    if gene_col is None:
        return []
    sort_cols = []
    ascending = []
    if "P_value_adj" in df.columns:
        df["_p_adj"] = pd.to_numeric(df["P_value_adj"], errors="coerce")
        sort_cols.append("_p_adj"); ascending.append(True)
    if "Scores" in df.columns:
        df["_score"] = pd.to_numeric(df["Scores"], errors="coerce")
        sort_cols.append("_score"); ascending.append(False)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)
    return df[gene_col].astype(str).head(int(top_n)).tolist()


def select_regulatory_genes(regulatory_path, top_n):
    df = read_csv_or_empty(regulatory_path)
    if df.empty or "gene" not in df.columns:
        return []
    score_col = "pearson_r" if "pearson_r" in df.columns else ("spearman_r" if "spearman_r" in df.columns else None)
    if score_col is None:
        counts = df.groupby("gene", observed=False).size().sort_values(ascending=False)
        return [str(gene) for gene in counts.head(int(top_n)).index]
    df["_abs_score"] = pd.to_numeric(df[score_col], errors="coerce").abs()
    agg = df.groupby("gene", observed=False).agg(n_links=("peak", "count") if "peak" in df.columns else (score_col, "count"), max_abs_score=("_abs_score", "max"))
    agg = agg.sort_values(["n_links", "max_abs_score"], ascending=[False, False])
    return [str(gene) for gene in agg.head(int(top_n)).index]


def build_gene_list(args):
    genes = []
    selection_records = []
    append_unique(genes, args.genes or [], "manual", selection_records)
    marker_path, residual_path, regulatory_path = infer_auto_paths(args)
    append_unique(genes, select_marker_genes(marker_path, args.top_marker_per_cluster), "marker", selection_records)
    append_unique(genes, select_residual_genes(residual_path, args.top_residual_genes), "residual_boundary", selection_records)
    append_unique(genes, select_regulatory_genes(regulatory_path, args.top_regulatory_genes), "regulatory_target", selection_records)
    max_genes = int(args.max_genes)
    if max_genes > 0:
        keep = {gene.lower() for gene in genes[:max_genes]}
        genes = genes[:max_genes]
        selection_records = [record for record in selection_records if record["gene"].lower() in keep]
    return genes, selection_records, {"marker_csv": str(marker_path) if marker_path else None, "residual_csv": str(residual_path) if residual_path else None, "regulatory_csv": str(regulatory_path) if regulatory_path else None}


def to_dense_vector(x):
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray()).reshape(-1)
    return np.asarray(x).reshape(-1)


def resolve_gene_names(adata, genes):
    var_names = pd.Index(adata.var_names.astype(str))
    lower_to_name = {name.lower(): name for name in var_names}
    resolved = []
    missing = []
    for gene in genes:
        key = str(gene).strip()
        if not key:
            continue
        if key in var_names:
            resolved.append((gene, key))
        elif key.lower() in lower_to_name:
            resolved.append((gene, lower_to_name[key.lower()]))
        else:
            missing.append(gene)
    return resolved, missing


def get_spatial_coordinates(adata, spatial_key, x_column, y_column):
    if spatial_key and spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key])
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError("obsm[%s] must have at least two columns" % spatial_key)
        return coords[:, 0].astype(float), coords[:, 1].astype(float), "obsm[%s]" % spatial_key
    if x_column and y_column and x_column in adata.obs and y_column in adata.obs:
        return pd.to_numeric(adata.obs[x_column], errors="coerce").to_numpy(dtype=float), pd.to_numeric(adata.obs[y_column], errors="coerce").to_numpy(dtype=float), "obs[%s,%s]" % (x_column, y_column)
    candidates = [("x", "y"), ("spatial1", "spatial2"), ("array_row", "array_col"), ("row", "col")]
    for x_key, y_key in candidates:
        if x_key in adata.obs and y_key in adata.obs:
            return pd.to_numeric(adata.obs[x_key], errors="coerce").to_numpy(dtype=float), pd.to_numeric(adata.obs[y_key], errors="coerce").to_numpy(dtype=float), "obs[%s,%s]" % (x_key, y_key)
    raise ValueError("No spatial coordinates found. Pass --spatial-key or --x-column/--y-column.")


def rotate_coordinates_clockwise(x, y, degrees):
    degrees = float(degrees)
    if abs(degrees) < 1e-12:
        return x, y
    theta = np.deg2rad(-degrees)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx = np.nanmean(x)
    cy = np.nanmean(y)
    x0 = x - cx
    y0 = y - cy
    xr = x0 * np.cos(theta) - y0 * np.sin(theta)
    yr = x0 * np.sin(theta) + y0 * np.cos(theta)
    return xr + cx, yr + cy


def prepare_expression(adata, args):
    ad = adata.copy()
    if not args.use_raw_values:
        sc.pp.normalize_total(ad, target_sum=float(args.target_sum))
        sc.pp.log1p(ad)
    return ad


def expression_summary(values, requested_gene, resolved_gene):
    finite = values[np.isfinite(values)]
    nonzero = finite[finite > 0]
    return {
        "requested_gene": requested_gene,
        "resolved_gene": resolved_gene,
        "n_spots": int(values.shape[0]),
        "n_finite": int(finite.shape[0]),
        "n_nonzero": int(nonzero.shape[0]),
        "fraction_nonzero": float(nonzero.shape[0] / max(finite.shape[0], 1)),
        "mean": float(np.mean(finite)) if finite.size else np.nan,
        "max": float(np.max(finite)) if finite.size else np.nan,
        "q95": float(np.quantile(finite, 0.95)) if finite.size else np.nan,
        "q99": float(np.quantile(finite, 0.99)) if finite.size else np.nan,
    }


def save_panel(expr_df, x, y, args, output_dir):
    genes = list(expr_df.columns)
    n_genes = len(genes)
    ncols = max(1, min(int(args.ncols), n_genes))
    nrows = int(math.ceil(n_genes / ncols))
    fig_width = float(args.panel_size) * ncols
    fig_height = float(args.panel_size) * nrows + 0.55
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), dpi=int(args.dpi), squeeze=False)
    title_y = 0.995 if nrows <= 2 else 0.992
    layout_top = 0.960 if nrows <= 2 else 0.978
    fig.suptitle(args.title, fontsize=14, y=title_y)

    finite_x = np.isfinite(x)
    finite_y = np.isfinite(y)
    valid_xy = finite_x & finite_y
    shared_vmax = None
    if args.shared_vmax:
        all_values = expr_df.to_numpy(dtype=float).reshape(-1)
        all_values = all_values[np.isfinite(all_values)]
        shared_vmax = float(np.quantile(all_values, float(args.vmax_quantile))) if all_values.size else 1.0
        shared_vmax = max(shared_vmax, float(args.vmin) + 1e-9)

    for idx, gene in enumerate(genes):
        ax = axes[idx // ncols][idx % ncols]
        values = expr_df[gene].to_numpy(dtype=float)
        mask = valid_xy & np.isfinite(values)
        order = np.arange(values.shape[0])
        if args.sort_by_expression:
            order = np.argsort(values)
        order = order[np.isin(order, np.where(mask)[0])]
        vmax = shared_vmax
        if vmax is None:
            finite_values = values[mask]
            vmax = float(np.quantile(finite_values, float(args.vmax_quantile))) if finite_values.size else 1.0
            vmax = max(vmax, float(args.vmin) + 1e-9)
        scatter = ax.scatter(x[order], y[order], c=values[order], s=float(args.point_size), cmap=args.cmap, vmin=float(args.vmin), vmax=vmax, linewidths=0, alpha=float(args.alpha), rasterized=True)
        ax.set_title(gene, fontsize=12, fontstyle="italic")
        ax.set_xlabel("spatial1", fontsize=float(args.axis_label_font_size))
        ax.set_ylabel("spatial2", fontsize=float(args.axis_label_font_size))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        if args.invert_y:
            ax.invert_yaxis()
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7, length=2)

    for idx in range(n_genes, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.tight_layout(rect=[0, 0, 1, layout_top])
    paths = []
    for fmt in args.formats:
        path = output_dir / ("spatial_gene_expression_panel.%s" % fmt)
        fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    args = parse_args()
    rna_path = resolve_path(args.rna_h5ad)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rna_path.exists():
        raise FileNotFoundError("RNA h5ad not found: %s" % rna_path)

    adata_raw = sc.read_h5ad(rna_path)
    x, y, coord_source = get_spatial_coordinates(adata_raw, args.spatial_key, args.x_column, args.y_column)
    x, y = rotate_coordinates_clockwise(x, y, args.rotate_deg)
    adata = prepare_expression(adata_raw, args)
    selected_genes, selection_records, auto_paths = build_gene_list(args)
    if not selected_genes:
        raise ValueError("No genes were provided or auto-selected. Pass --genes, --auto-from-dir, or result CSV paths.")
    selected_path = output_dir / "spatial_gene_expression_selected_genes.csv"
    pd.DataFrame(selection_records).to_csv(selected_path, index=False)
    resolved, missing = resolve_gene_names(adata, selected_genes)
    if not resolved:
        raise ValueError("None of the requested genes were found. Missing: %s" % ", ".join(map(str, missing)))

    expr = {}
    summary_rows = []
    for requested, resolved_gene in resolved:
        values = to_dense_vector(adata[:, resolved_gene].X)
        expr[resolved_gene] = values
        summary_rows.append(expression_summary(values, requested, resolved_gene))
    expr_df = pd.DataFrame(expr)
    expr_path = output_dir / "spatial_gene_expression_values.csv"
    expr_df.to_csv(expr_path, index=False)
    summary = pd.DataFrame(summary_rows)
    if missing:
        for gene in missing:
            summary = pd.concat([summary, pd.DataFrame([{"requested_gene": gene, "resolved_gene": "", "n_spots": int(adata.n_obs), "n_finite": 0, "n_nonzero": 0, "fraction_nonzero": 0.0, "mean": np.nan, "max": np.nan, "q95": np.nan, "q99": np.nan}])], ignore_index=True)
    summary_path = output_dir / "spatial_gene_expression_summary.csv"
    summary.to_csv(summary_path, index=False)
    figures = save_panel(expr_df, x, y, args, output_dir)
    manifest = {
        "rna_h5ad": str(rna_path),
        "output_dir": str(output_dir),
        "coordinate_source": coord_source,
        "rotate_deg_clockwise": float(args.rotate_deg),
        "axis_label_font_size_points": float(args.axis_label_font_size),
        "normalization": "raw_values" if args.use_raw_values else "normalize_total_log1p",
        "target_sum": None if args.use_raw_values else float(args.target_sum),
        "requested_genes": list(args.genes or []),
        "selected_genes": selected_genes,
        "selection_sources": str(selected_path),
        "auto_paths": auto_paths,
        "resolved_genes": [gene for _requested, gene in resolved],
        "missing_genes": list(missing),
        "summary_csv": str(summary_path),
        "values_csv": str(expr_path),
        "figures": figures,
    }
    manifest_path = output_dir / "spatial_gene_expression_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[spatial gene expression] complete genes=%d missing=%d" % (len(resolved), len(missing)))
    print("  summary: %s" % summary_path)
    print("  manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()
