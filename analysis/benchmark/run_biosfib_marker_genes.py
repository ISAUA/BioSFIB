"""BioSFIB-only marker-gene analysis with reference-name display labels.

The input H5AD files are read only.  Cluster-to-tissue matching and display
labels exist only in memory; the script writes figures and audit CSV files.
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BioSFIB marker-gene analysis without modifying input H5AD files.")
    parser.add_argument("--config", type=Path, default=Path("analysis/benchmark/biosfib_marker_genes_config.yaml"), help="Marker-gene YAML configuration.")
    parser.add_argument("--analysis", help="Name under analyses in the YAML file; defaults to default_analysis.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths, spot alignment and Hungarian matching without writing outputs.")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_path(config_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_root / path).resolve()


def resolve_analysis(config: dict[str, Any], name: str | None) -> tuple[str, dict[str, Any]]:
    analyses = config.get("analyses", {})
    selected_name = name or config.get("default_analysis")
    if not selected_name:
        raise KeyError("Set default_analysis or pass --analysis.")
    if selected_name not in analyses:
        raise KeyError(f"Unknown analysis {selected_name!r}. Available: {', '.join(analyses)}")
    spec = deep_merge(config.get("defaults", {}), analyses[selected_name])
    required = ["expression_h5ad", "cluster_key", "truth_annotation_key", "output_dir"]
    missing = [key for key in required if not spec.get(key)]
    if missing:
        raise KeyError(f"Analysis {selected_name!r} is missing: {missing}")
    return selected_name, spec


def natural_label_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


def valid_mask(values: pd.Series, ignored: list[str]) -> np.ndarray:
    string_values = values.astype("string")
    ignored_normalized = {str(item).strip().lower() for item in ignored}
    return (string_values.notna() & ~string_values.str.strip().str.lower().isin(ignored_normalized)).to_numpy()


def aligned_series(target: sc.AnnData, source: sc.AnnData, key: str) -> pd.Series:
    if key not in source.obs:
        raise KeyError(f"obs[{key!r}] is absent. Available columns: {', '.join(map(str, source.obs.columns))}")
    if target.obs_names.equals(source.obs_names):
        return source.obs[key].copy()
    if set(target.obs_names) != set(source.obs_names):
        raise ValueError("Expression and cluster H5AD files have different obs_names; row-order fallback is intentionally disabled.")
    return source.obs.loc[target.obs_names, key].copy()


def build_hungarian_mapping(
    clusters: pd.Series,
    truth: pd.Series,
    ignored_truth: list[str],
    display_template: str,
    unmatched_template: str,
    label_aliases: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    cluster_values = clusters.astype("string")
    truth_values = truth.astype("string")
    valid = valid_mask(cluster_values, []) & valid_mask(truth_values, ignored_truth)
    if not valid.any():
        raise ValueError("No spots have both a valid BioSFIB cluster and a valid truth annotation.")

    cluster_labels = sorted(cluster_values[valid].unique().tolist(), key=natural_label_key)
    truth_labels = sorted(truth_values[valid].unique().tolist(), key=natural_label_key)
    contingency = pd.crosstab(
        pd.Categorical(cluster_values[valid], categories=cluster_labels),
        pd.Categorical(truth_values[valid], categories=truth_labels),
        dropna=False,
    ).reindex(index=cluster_labels, columns=truth_labels, fill_value=0)
    rows, columns = linear_sum_assignment(contingency.to_numpy(), maximize=True)
    # Hungarian must return a complete one-to-one assignment when the two sides
    # have equal cardinality.  A zero-overlap pair is not a biological match and
    # must remain explicitly unmatched rather than receiving a false tissue name.
    matches = {
        cluster_labels[row]: truth_labels[column]
        for row, column in zip(rows, columns)
        if int(contingency.iloc[row, column]) > 0
    }

    records: list[dict[str, Any]] = []
    display_map: dict[str, str] = {}
    for cluster in cluster_labels:
        matched_truth = matches.get(cluster)
        cluster_size = int(contingency.loc[cluster].sum())
        overlap = int(contingency.loc[cluster, matched_truth]) if matched_truth is not None else 0
        truth_size = int(contingency[matched_truth].sum()) if matched_truth is not None else 0
        display_tissue = label_aliases.get(matched_truth, matched_truth) if matched_truth is not None else None
        display_label = (
            display_template.format(tissue=display_tissue, cluster=cluster)
            if matched_truth is not None
            else unmatched_template.format(cluster=cluster)
        )
        display_map[cluster] = display_label
        records.append(
            {
                "biosfib_cluster": cluster,
                "matched_truth_annotation": matched_truth,
                "display_tissue_label": display_tissue,
                "display_label": display_label,
                "overlap_spots": overlap,
                "cluster_spots": cluster_size,
                "truth_spots": truth_size if matched_truth is not None else np.nan,
                "cluster_purity": overlap / cluster_size if cluster_size else np.nan,
                "truth_recall": overlap / truth_size if truth_size else np.nan,
            }
        )
    return pd.DataFrame(records), display_map


def save_marker_table(adata: sc.AnnData, key_added: str, output_path: Path, top_n: int, pval_cutoff: float | None) -> pd.DataFrame:
    result = adata.uns[key_added]
    frames: list[pd.DataFrame] = []
    for group in result["names"].dtype.names:
        frame = pd.DataFrame(
            {
                "display_label": group,
                "gene": result["names"][group],
                "score": result["scores"][group],
                "logfoldchange": result.get("logfoldchanges", result["scores"])[group],
                "p_value": result.get("pvals", result["scores"])[group],
                "p_value_adj": result.get("pvals_adj", result["scores"])[group],
            }
        )
        if pval_cutoff is not None and "pvals_adj" in result:
            frame = frame[frame["p_value_adj"] <= float(pval_cutoff)]
        frames.append(frame.head(top_n))
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    table.to_csv(output_path, index=False)
    return table


def make_dotplot(adata: sc.AnnData, spec: dict[str, Any], key_added: str, output_path: Path) -> None:
    sc.settings.set_figure_params(dpi=int(spec["plot"]["dpi"]), format="pdf")
    dotplot_options: dict[str, Any] = {
        "swap_axes": bool(spec["plot"].get("dotplot_swap_axes", False)),
    }
    configured_size = spec["plot"].get("dotplot_figure_size")
    if configured_size is not None:
        if not isinstance(configured_size, (list, tuple)) or len(configured_size) != 2:
            raise ValueError("plot.dotplot_figure_size must be null or [width, height].")
        dotplot_options["figsize"] = tuple(float(value) for value in configured_size)
    common_options = {
        "groupby": spec["display_key"],
        "key": key_added,
        "n_genes": int(spec["plot"]["dotplot_genes_per_group"]),
        "standard_scale": "var",
        "dendrogram": bool(spec["plot"].get("dendrogram", False)),
        **dotplot_options,
    }
    legend_position = str(spec["plot"].get("dotplot_legend_position", "right")).lower()
    if legend_position == "right":
        sc.pl.rank_genes_groups_dotplot(adata, show=False, **common_options)
        plt.savefig(output_path, bbox_inches="tight")
    elif legend_position == "bottom":
        dotplot = sc.pl.rank_genes_groups_dotplot(
            adata,
            show=False,
            return_fig=True,
            **common_options,
        )
        dotplot.legend(show=True)
        dotplot.legends_width = 0
        dotplot.make_figure()
        main_axis = dotplot.ax_dict["mainplot_ax"]
        main_position = main_axis.get_position()
        main_axis.set_position(
            [main_position.x0, 0.16, main_position.width, main_position.y1 - 0.16]
        )
        if not main_axis.collections:
            raise RuntimeError("Scanpy dotplot main axis has no collection for color normalization.")
        color_normalize = main_axis.collections[0].norm
        legend_y_shift = float(spec["plot"].get("dotplot_bottom_legend_y_shift", 0.0))
        size_axis = dotplot.fig.add_axes(
            [main_position.x0, 0.035 + legend_y_shift, main_position.width * 0.48, 0.075]
        )
        color_axis = dotplot.fig.add_axes(
            [
                main_position.x0 + main_position.width * 0.61,
                0.060 + legend_y_shift,
                main_position.width * 0.34,
                0.018,
            ]
        )
        dotplot._plot_size_legend(size_axis)
        dotplot._plot_colorbar(color_axis, color_normalize)
        dotplot.fig.savefig(output_path, bbox_inches="tight")
    else:
        raise ValueError("plot.dotplot_legend_position must be 'right' or 'bottom'.")
    plt.close("all")


def make_spatial_marker_plot(adata: sc.AnnData, marker_table: pd.DataFrame, spec: dict[str, Any], output_path: Path) -> None:
    if "spatial" not in adata.obsm or marker_table.empty:
        return
    genes = marker_table.groupby("display_label", sort=False)["gene"].first().dropna().astype(str).tolist()
    genes = [gene for gene in genes if gene in adata.var_names][: int(spec["plot"]["spatial_top_markers"])]
    if not genes:
        return
    sc.pl.embedding(adata, basis="spatial", color=[spec["display_key"]] + genes, show=False)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_yaml(config_path)
    analysis_name, spec = resolve_analysis(config, args.analysis)
    expression_path = resolve_path(PROJECT_ROOT, spec["expression_h5ad"])
    cluster_path = resolve_path(PROJECT_ROOT, spec.get("cluster_h5ad") or spec["expression_h5ad"])
    output_dir = resolve_path(PROJECT_ROOT, spec["output_dir"])
    for path, label in ((expression_path, "expression_h5ad"), (cluster_path, "cluster_h5ad")):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    print(f"[marker] Analysis: {analysis_name}")
    print(f"[marker] Expression: {expression_path}")
    print(f"[marker] Clusters: {cluster_path}")
    adata = sc.read_h5ad(expression_path)
    cluster_source = adata if cluster_path == expression_path else sc.read_h5ad(cluster_path)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    cluster_key = str(spec["cluster_key"])
    truth_key = str(spec["truth_annotation_key"])
    adata.obs[cluster_key] = aligned_series(adata, cluster_source, cluster_key).to_numpy()
    adata.obs[truth_key] = aligned_series(adata, cluster_source, truth_key).to_numpy()
    if "spatial" not in adata.obsm and "spatial" in cluster_source.obsm:
        adata.obsm["spatial"] = np.asarray(cluster_source.obsm["spatial"]).copy()

    matching = spec["matching"]
    display_template = (
        matching["display_template_with_cluster"]
        if bool(matching["show_cluster_id"])
        else matching["display_template_without_cluster"]
    )
    mapping, display_map = build_hungarian_mapping(
        adata.obs[cluster_key], adata.obs[truth_key], matching["ignored_truth_values"],
        display_template, matching["unmatched_template"], matching.get("label_aliases", {}),
    )
    display_key = str(spec["display_key"])
    raw_clusters = adata.obs[cluster_key].astype("string")
    display_values = raw_clusters.map(display_map).fillna(
        raw_clusters.map(lambda value: matching["unmatched_template"].format(cluster=value))
    )
    display_order = mapping.sort_values("biosfib_cluster", key=lambda values: values.map(natural_label_key))["display_label"].tolist()
    extra_labels = [label for label in display_values.dropna().unique().tolist() if label not in display_order]
    adata.obs[display_key] = pd.Categorical(display_values, categories=display_order + extra_labels, ordered=True)

    if not bool(matching["include_unmatched_clusters"]):
        matched_labels = set(mapping.loc[mapping["matched_truth_annotation"].notna(), "display_label"])
        keep = adata.obs[display_key].astype("string").isin(matched_labels).to_numpy()
        excluded = int((~keep).sum())
        if not keep.any():
            raise ValueError("All spots were excluded because no BioSFIB clusters have a positive-overlap truth match.")
        if excluded:
            print(f"[marker] Excluding {excluded} spots from unmatched BioSFIB clusters in marker plots.")
        adata = adata[keep].copy()

    print(f"[marker] Hungarian matched {mapping['matched_truth_annotation'].notna().sum()}/{len(mapping)} BioSFIB clusters.")
    if args.dry_run:
        print("[marker] Dry run successful; no files were written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_dir / "biosfib_cluster_to_truth_mapping.csv", index=False)
    adata.layers["counts"] = adata.X.copy()
    preprocess = spec["preprocess"]
    if bool(preprocess["normalize_total"]):
        sc.pp.normalize_total(adata, target_sum=float(preprocess["target_sum"]))
    if bool(preprocess["log1p"]):
        sc.pp.log1p(adata)

    key_added = "biosfib_rank_genes_groups"
    sc.tl.rank_genes_groups(adata, groupby=display_key, method=spec["marker_test"]["method"], use_raw=False, key_added=key_added)
    marker_table = save_marker_table(
        adata, key_added, output_dir / "top_marker_genes_by_matched_label.csv",
        int(spec["marker_test"]["top_n"]), spec["marker_test"]["pval_adj_cutoff"],
    )
    make_dotplot(adata, spec, key_added, output_dir / "marker_dotplot_by_matched_label.pdf")
    if bool(spec["plot"]["save_spatial_marker_map"]):
        make_spatial_marker_plot(adata, marker_table, spec, output_dir / "marker_spatial_map_by_matched_label.pdf")
    print(f"[marker] Output: {output_dir}")


if __name__ == "__main__":
    main()
