"""Draw publication-preview figures for the provisional Y7_T four-class annotation.

The command reads the frozen BioSFIB K=14 result, the independently generated
SpatialMETA-style marker-guided annotation, and original RNA/metabolome matrices.
It does not train, recluster, select a model, or use the provisional labels as
ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _sanitize_thread_env_vars() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(key)
        try:
            valid = value is not None and int(str(value).strip()) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            os.environ[key] = "1"


_sanitize_thread_env_vars()

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common.marker_helpers import (  # noqa: E402
    _dense_vector,
    _display_feature,
    build_heatmap_source,
    choose_unique_markers,
    prepare_modality,
    rank_markers,
    save_figure_bundle,
)


DEFAULT_ANNOTATED_H5AD = (
    "results/analysis/meta/Y7_T/"
    "Y7_T_BioSFIB_K14_destvi_annotation.h5ad"
)
DEFAULT_RNA_H5AD = "data/raw/renal/Y7_T/adata_joint_Y7_T_raw_ST.h5ad"
DEFAULT_SM_H5AD = "data/raw/renal/Y7_T/adata_joint_Y7_T_raw_SM.h5ad"
DEFAULT_EVIDENCE_CSV = (
    "results/analysis/meta/Y7_T/"
    "Y7_T_BioSFIB_K14_four_class_evidence.csv"
)
DEFAULT_OUTPUT_DIR = (
    "results/analysis/meta/Y7_T/"
    "manuscript_four_class_figures_20260813"
)

ANNOTATION_KEY = "biosfib_four_class_marker_guided"
STATUS_KEY = "biosfib_four_class_marker_guided_status"
CLUSTER_KEY = "official_cluster"
CLASS_ORDER = ["Imm", "Endo", "Stro", "Mal"]
CLASS_COLORS = {
    "Imm": "#D55E00",
    "Endo": "#0072B2",
    "Stro": "#009E73",
    "Mal": "#CC79A7",
}
ANNOTATION_MARKERS = {
    "CD8A", "CD3D", "CD74",
    "CD34", "PECAM1", "APLN",
    "COL3A1", "ACTA2", "COL1A1",
    "NDUFA4L2", "CNDP2", "EGFR",
}
RNA_COLOR = "#4C78A8"
SM_COLOR = "#E08B5B"
EVIDENCE_CMAP = mpl.colormaps["Blues"]
RNA_HEATMAP_CMAP = mpl.colormaps["Blues"]
SM_HEATMAP_CMAP = mpl.colormaps["Greens"]
RNA_CMAP = LinearSegmentedColormap.from_list("rna_expression", ["#F7FAFC", RNA_COLOR])
SM_CMAP = LinearSegmentedColormap.from_list("sm_expression", ["#FFF9F3", SM_COLOR])


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Y7_T four-class annotation figures.")
    parser.add_argument("--annotated-h5ad", default=DEFAULT_ANNOTATED_H5AD)
    parser.add_argument("--rna-h5ad", default=DEFAULT_RNA_H5AD)
    parser.add_argument("--sm-h5ad", default=DEFAULT_SM_H5AD)
    parser.add_argument("--evidence-csv", default=DEFAULT_EVIDENCE_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rank-top", type=int, default=200)
    parser.add_argument("--markers-per-class", type=int, default=3)
    parser.add_argument("--reuse-marker-stats", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_hash(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def style_spatial_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def nested_cluster_order(evidence: pd.DataFrame) -> list[str]:
    evidence = evidence.copy()
    evidence["biosfib_cluster"] = evidence["biosfib_cluster"].astype(str)
    evidence["marker_guided_candidate"] = evidence["marker_guided_candidate"].astype(str)
    order = []
    for group in CLASS_ORDER:
        clusters = evidence.loc[
            evidence["marker_guided_candidate"] == group, "biosfib_cluster"
        ].tolist()
        order.extend(sorted(clusters, key=lambda value: float(value)))
    if len(order) != 14 or len(set(order)) != 14:
        raise ValueError(f"Expected 14 unique nested clusters, found {order}")
    return order


def validate_inputs(
    annotated,
    evidence: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    for key in (ANNOTATION_KEY, STATUS_KEY, CLUSTER_KEY):
        if key not in annotated.obs:
            raise KeyError(f"Annotated H5AD lacks obs[{key!r}]")
    if "spatial" not in annotated.obsm:
        raise KeyError("Annotated H5AD lacks obsm['spatial']")
    if annotated.n_obs != 1914:
        raise ValueError(f"Y7_T contract requires 1914 spots, found {annotated.n_obs}")

    annotations = annotated.obs[ANNOTATION_KEY].astype(str).to_numpy()
    clusters = annotated.obs[CLUSTER_KEY].astype(str).to_numpy()
    observed = set(np.unique(annotations))
    if observed != set(CLASS_ORDER):
        raise ValueError(f"Expected exactly {CLASS_ORDER}, found {sorted(observed)}")
    if len(np.unique(clusters)) != 14:
        raise ValueError(f"Expected 14 fixed clusters, found {len(np.unique(clusters))}")

    coords = np.asarray(annotated.obsm["spatial"], dtype=float)[:, :2]

    required_evidence = {
        "biosfib_cluster", "marker_guided_candidate", "marker_guided_status",
        "marker_Imm", "marker_Endo", "marker_Stro", "marker_Mal",
    }
    missing_evidence = sorted(required_evidence.difference(evidence.columns))
    if missing_evidence:
        raise KeyError(f"Evidence table lacks columns: {missing_evidence}")
    cluster_order = nested_cluster_order(evidence)
    spot_map = pd.DataFrame({"cluster": clusters, "annotation": annotations}).drop_duplicates()
    if spot_map.groupby("cluster")["annotation"].nunique().max() != 1:
        raise ValueError("A fixed cluster maps to more than one four-class annotation")
    evidence_map = evidence.assign(
        biosfib_cluster=evidence["biosfib_cluster"].astype(str),
        marker_guided_candidate=evidence["marker_guided_candidate"].astype(str),
    ).set_index("biosfib_cluster")["marker_guided_candidate"]
    observed_map = spot_map.set_index("cluster")["annotation"]
    if not evidence_map.sort_index().equals(observed_map.sort_index()):
        raise ValueError("Evidence-table annotation mapping differs from H5AD")
    return coords, annotations, clusters, cluster_order


def select_class_markers(
    stats: pd.DataFrame,
    modality: str,
    markers_per_class: int,
) -> pd.DataFrame:
    candidates = stats.copy()
    if modality == "RNA":
        candidates = candidates[
            ~candidates["feature"].astype(str).str.upper().isin(ANNOTATION_MARKERS)
        ].copy()
    selected = choose_unique_markers(candidates, CLASS_ORDER, markers_per_class)
    if selected.shape[0] != len(CLASS_ORDER) * int(markers_per_class):
        raise ValueError(f"Unexpected {modality} marker count: {selected.shape[0]}")
    fallback = selected[selected["selection_filter"] != "robust_primary"]
    if not fallback.empty:
        details = fallback[["cluster", "feature", "selection_filter"]].to_dict("records")
        raise ValueError(f"{modality} marker selection required fallback rows: {details}")
    selected["annotation"] = selected["cluster"].astype(str)
    return selected


def build_class_heatmap_source(
    adata,
    markers: pd.DataFrame,
    modality: str,
) -> pd.DataFrame:
    source = build_heatmap_source(
        adata,
        markers,
        CLASS_ORDER,
        ANNOTATION_KEY,
        modality,
    )
    return source.rename(
        columns={"target_cluster": "target_annotation", "cluster": "annotation"}
    )


def row_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    means = values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True)
    return np.divide(values - means, scales, out=np.zeros_like(values), where=scales > 1e-12)


def class_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor=CLASS_COLORS[group], markeredgecolor="none", label=group)
        for group in CLASS_ORDER
    ]


def plot_cluster_map(
    ax: plt.Axes,
    coords: np.ndarray,
    clusters: np.ndarray,
    cluster_order: list[str],
) -> None:
    cmap = plt.get_cmap("tab20")
    color_map = {cluster: cmap(index / 14.0) for index, cluster in enumerate(cluster_order)}
    point_size = float(np.clip(17000.0 / coords.shape[0], 5.0, 10.0))
    for cluster in cluster_order:
        mask = clusters == cluster
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=point_size,
            color=color_map[cluster], linewidths=0, rasterized=True,
        )
        center = np.median(coords[mask], axis=0)
        text = ax.text(
            center[0], center[1], f"C{cluster}", ha="center", va="center",
            fontsize=5.4, fontweight="bold", color="#202020",
        )
        text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
    style_spatial_axis(ax)
    ax.set_title("Fixed BioSFIB domains (K=14)", pad=3)


def plot_annotation_map(
    ax: plt.Axes,
    coords: np.ndarray,
    annotations: np.ndarray,
) -> None:
    point_size = float(np.clip(17000.0 / coords.shape[0], 5.0, 10.0))
    for group in CLASS_ORDER:
        mask = annotations == group
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=point_size,
            color=CLASS_COLORS[group], linewidths=0, rasterized=True,
        )
    style_spatial_axis(ax)
    ax.set_title("Marker-guided annotation (provisional)", pad=3)
    ax.legend(
        handles=class_handles(), loc="upper left", bbox_to_anchor=(1.00, 1.00),
        borderaxespad=0.0, handletextpad=0.35, labelspacing=0.35,
    )


def add_nested_class_strip(
    ax: plt.Axes,
    evidence_indexed: pd.DataFrame,
    cluster_order: list[str],
) -> None:
    for column_index, cluster in enumerate(cluster_order):
        group = str(evidence_indexed.loc[cluster, "marker_guided_candidate"])
        ax.add_patch(
            Rectangle(
                (column_index - 0.5, -0.94), 1.0, 0.24,
                facecolor=CLASS_COLORS[group], edgecolor="none", clip_on=False,
            )
        )


def outline_assigned_cells(
    ax: plt.Axes,
    evidence_indexed: pd.DataFrame,
    cluster_order: list[str],
) -> None:
    row_lookup = {group: index for index, group in enumerate(CLASS_ORDER)}
    for column_index, cluster in enumerate(cluster_order):
        group = str(evidence_indexed.loc[cluster, "marker_guided_candidate"])
        row_index = row_lookup[group]
        ax.add_patch(
            Rectangle(
                (column_index - 0.5, row_index - 0.5), 1.0, 1.0,
                fill=False, edgecolor="#202020", linewidth=0.65,
            )
        )


def plot_evidence_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    evidence_indexed: pd.DataFrame,
    cluster_order: list[str],
    title: str,
    cmap,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    fig: plt.Figure,
) -> None:
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(cluster_order)))
    ax.set_xticklabels([f"C{cluster}" for cluster in cluster_order], rotation=55, ha="right")
    ax.set_yticks(np.arange(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER)
    ax.tick_params(length=0)
    ax.set_title(title, pad=17)
    add_nested_class_strip(ax, evidence_indexed, cluster_order)
    outline_assigned_cells(ax, evidence_indexed, cluster_order)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label(colorbar_label, fontsize=6.3)
    colorbar.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    colorbar.outline.set_linewidth(0.45)


def build_evidence_source(
    evidence: pd.DataFrame,
    cluster_order: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    indexed = evidence.copy()
    indexed["biosfib_cluster"] = indexed["biosfib_cluster"].astype(str)
    indexed["marker_guided_candidate"] = indexed["marker_guided_candidate"].astype(str)
    indexed = indexed.set_index("biosfib_cluster").loc[cluster_order]
    marker_raw = indexed[[f"marker_{group}" for group in CLASS_ORDER]].to_numpy(float).T
    marker_z = row_zscore(marker_raw)
    rows = []
    for row_index, group in enumerate(CLASS_ORDER):
        for column_index, cluster in enumerate(cluster_order):
            rows.append(
                {
                    "annotation_axis": group,
                    "fixed_cluster": cluster,
                    "assigned_annotation": str(indexed.loc[cluster, "marker_guided_candidate"]),
                    "marker_score_raw": float(marker_raw[row_index, column_index]),
                    "marker_score_row_z": float(marker_z[row_index, column_index]),
                }
            )
    return pd.DataFrame(rows), marker_z, indexed


def plot_summary_figure(
    coords: np.ndarray,
    annotations: np.ndarray,
    clusters: np.ndarray,
    cluster_order: list[str],
    evidence: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[Path], pd.DataFrame]:
    evidence_source, marker_z, evidence_indexed = build_evidence_source(
        evidence, cluster_order
    )
    fig = plt.figure(figsize=(7.2, 4.55))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.28, 0.72], hspace=0.25, wspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, :])

    plot_cluster_map(ax_a, coords, clusters, cluster_order)
    plot_annotation_map(ax_b, coords, annotations)
    plot_evidence_heatmap(
        ax_c, marker_z, evidence_indexed, cluster_order,
        "Independent marker evidence", EVIDENCE_CMAP, -2.2, 2.2,
        "Row z-score", fig,
    )
    for label, ax in zip("abc", (ax_a, ax_b, ax_c)):
        panel_label(ax, label)
    fig.subplots_adjust(left=0.075, right=0.955, bottom=0.105, top=0.965)
    paths = save_figure_bundle(fig, output_dir / "figure_four_class_annotation_summary")
    plt.close(fig)
    return paths, evidence_source


def heatmap_matrix_from_source(
    source: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    row_meta = source[
        ["target_annotation", "selected_rank", "feature", "feature_display"]
    ].drop_duplicates().copy()
    row_meta["class_order"] = row_meta["target_annotation"].map(
        {group: index for index, group in enumerate(CLASS_ORDER)}
    )
    row_meta = row_meta.sort_values(["class_order", "selected_rank"], kind="stable").reset_index(drop=True)
    matrix = np.empty((row_meta.shape[0], len(CLASS_ORDER)), dtype=float)
    for row_index, row in row_meta.iterrows():
        subset = source[
            (source["target_annotation"] == row["target_annotation"])
            & (source["feature"] == row["feature"])
        ].set_index("annotation")
        matrix[row_index, :] = [float(subset.loc[group, "row_z_score"]) for group in CLASS_ORDER]
    return matrix, row_meta


def draw_marker_heatmap(
    ax: plt.Axes,
    colorbar_ax: plt.Axes,
    source: pd.DataFrame,
    title: str,
    cmap,
    colorbar_label: str,
) -> None:
    matrix, feature_meta = heatmap_matrix_from_source(source)
    display_matrix = matrix.T
    image = ax.imshow(
        display_matrix, cmap=cmap, vmin=-2.5, vmax=2.5,
        aspect="auto", interpolation="nearest",
    )
    ax.set_xticks(np.arange(feature_meta.shape[0]))
    ax.set_xticklabels(
        feature_meta["feature_display"].astype(str), rotation=58, ha="right",
        rotation_mode="anchor", fontsize=5.7,
    )
    ax.set_yticks(np.arange(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER, fontsize=6.4)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", pad=31, fontsize=8, fontweight="bold")

    class_lookup = {group: index for index, group in enumerate(CLASS_ORDER)}
    feature_classes = feature_meta["target_annotation"].astype(str).to_numpy()
    for group in CLASS_ORDER:
        feature_indices = np.flatnonzero(feature_classes == group)
        if feature_indices.size == 0:
            continue
        start = int(feature_indices.min())
        end = int(feature_indices.max())
        target_row = class_lookup[group]
        ax.add_patch(
            Rectangle(
                (start - 0.5, target_row - 0.5), end - start + 1.0, 1.0,
                fill=False, edgecolor="#202020", linewidth=0.75,
            )
        )
        x0 = start - 0.42
        x1 = end + 0.42
        ax.plot(
            [x0, x0, x1, x1], [1.015, 1.075, 1.075, 1.015],
            transform=ax.get_xaxis_transform(), color=CLASS_COLORS[group],
            linewidth=0.9, clip_on=False,
        )
        ax.text(
            (start + end) / 2.0, 1.095, group,
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=6.2, color=CLASS_COLORS[group], fontweight="bold",
        )

    boundaries = np.where(feature_classes[1:] != feature_classes[:-1])[0]
    for column_index in boundaries:
        ax.axvline(column_index + 0.5, color="white", linewidth=1.4)
    ax.set_xticks(np.arange(-0.5, feature_meta.shape[0], 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASS_ORDER), 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    for tick_label, group in zip(ax.get_yticklabels(), CLASS_ORDER):
        tick_label.set_color(CLASS_COLORS[group])
        tick_label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#202020")

    colorbar = ax.figure.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(colorbar_label, fontsize=6.1)
    colorbar.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    colorbar.outline.set_linewidth(0.45)


def plot_marker_heatmaps(
    rna_source: pd.DataFrame,
    sm_source: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 0.022], height_ratios=[1.0, 1.0],
        hspace=0.94, wspace=0.055,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    colorbar_a = fig.add_subplot(gs[0, 1])
    colorbar_b = fig.add_subplot(gs[1, 1])
    draw_marker_heatmap(
        ax_a, colorbar_a, rna_source,
        "RNA markers (annotation markers excluded)",
        RNA_HEATMAP_CMAP,
        "Mean expression\n(feature-wise z-score)",
    )
    draw_marker_heatmap(
        ax_b, colorbar_b, sm_source,
        "Metabolite features",
        SM_HEATMAP_CMAP,
        "Mean intensity\n(feature-wise z-score)",
    )
    ax_a.text(-0.055, 1.28, "a", transform=ax_a.transAxes, fontsize=9,
              fontweight="bold", va="top")
    ax_b.text(-0.055, 1.28, "b", transform=ax_b.transAxes, fontsize=9,
              fontweight="bold", va="top")
    fig.subplots_adjust(left=0.105, right=0.955, bottom=0.115, top=0.945)
    paths = save_figure_bundle(fig, output_dir / "figure_four_class_marker_heatmaps")
    plt.close(fig)
    return paths


def plot_representative_spatial_markers(
    rna_adata,
    sm_adata,
    rna_markers: pd.DataFrame,
    sm_markers: pd.DataFrame,
    coords: np.ndarray,
    annotations: np.ndarray,
    output_dir: Path,
) -> tuple[list[Path], pd.DataFrame]:
    rna_top = rna_markers[rna_markers["selected_rank"] == 1].set_index("annotation")
    sm_top = sm_markers[sm_markers["selected_rank"] == 1].set_index("annotation")
    fig, axes = plt.subplots(len(CLASS_ORDER), 2, figsize=(7.2, 7.8), squeeze=False)
    point_size = float(np.clip(18000.0 / coords.shape[0], 5.5, 11.0))
    neighbor_indices = NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(
        coords, return_distance=False
    )
    source_rows = []
    for row_index, group in enumerate(CLASS_ORDER):
        specifications = [
            ("RNA", rna_adata, rna_top.loc[group], RNA_CMAP, RNA_COLOR),
            ("SM", sm_adata, sm_top.loc[group], SM_CMAP, SM_COLOR),
        ]
        target_mask = annotations == group
        target_boundary = target_mask & np.any(
            annotations[neighbor_indices] != group, axis=1
        )
        for column_index, (modality, adata, marker, cmap, accent) in enumerate(specifications):
            ax = axes[row_index, column_index]
            feature = str(marker["feature"])
            if feature not in adata.var_names:
                raise KeyError(f"Selected {modality} feature is absent: {feature}")
            expression = _dense_vector(adata[:, feature].X)
            finite = expression[np.isfinite(expression)]
            positive = finite[finite > 0.0]
            vmax = float(np.quantile(positive, 0.99)) if positive.size else float(np.max(finite))
            vmax = max(vmax, 1e-8)
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1], c=expression,
                cmap=cmap, vmin=0.0, vmax=vmax, s=point_size,
                linewidths=0, rasterized=True,
            )
            ax.scatter(
                coords[target_boundary, 0], coords[target_boundary, 1],
                facecolors="none", edgecolors="#202020",
                s=point_size + 3.0, linewidths=0.32, rasterized=True,
            )
            style_spatial_axis(ax)
            display = _display_feature(feature, modality)
            ax.set_title(
                f"{display}\nlog2FC={float(marker['log2_fold_change']):.2f}, "
                f"FDR={float(marker['fdr_bh']):.1e}",
                pad=2, fontsize=6.8,
            )
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.034, pad=0.015)
            colorbar.ax.tick_params(labelsize=5.2, length=2, width=0.45)
            colorbar.outline.set_linewidth(0.4)
            if row_index == 0:
                ax.text(
                    0.5, 1.19,
                    "Transcriptome" if modality == "RNA" else "Metabolome",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=accent,
                )
            if column_index == 0:
                ax.text(
                    -0.07, 0.5, f"{group}\n(n={int(target_mask.sum())})",
                    transform=ax.transAxes, ha="right", va="center",
                    rotation=90, fontsize=7, fontweight="bold",
                    color=CLASS_COLORS[group],
                )
            for spot, x, y, value, is_target, is_boundary in zip(
                adata.obs_names.astype(str), coords[:, 0], coords[:, 1], expression,
                target_mask, target_boundary,
            ):
                source_rows.append(
                    {
                        "spot": spot,
                        "x": float(x),
                        "y": float(y),
                        "target_annotation": group,
                        "is_target_annotation": bool(is_target),
                        "is_target_boundary": bool(is_boundary),
                        "modality": modality,
                        "feature": feature,
                        "feature_display": display,
                        "normalized_log1p_value": float(value),
                        "plot_vmax_q99_positive": vmax,
                    }
                )
    panel_label(axes[0, 0], "a")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.025, top=0.97, hspace=0.12, wspace=0.08)
    paths = save_figure_bundle(fig, output_dir / "figure_four_class_representative_markers")
    plt.close(fig)
    return paths, pd.DataFrame(source_rows)


def write_figure_legends(output_dir: Path) -> Path:
    text = """# Figure legends

## Fig. X | Spatial and marker-guided characterization of BioSFIB domains in ccRCC

**a** Spatial distribution of the fixed BioSFIB Mclust domains (K = 14). **b** Provisional biological annotation of the unchanged domains into immune (Imm), endothelial (Endo), stromal (Stro), and malignant (Mal) regions using the SpatialMETA marker panel. **c** Cluster-average marker scores, standardized within each marker class across the 14 domains. Black outlines indicate the provisional class assigned to each fixed domain; the colored strip above the columns shows the same assignment.

## Fig. Y | Transcriptomic and metabolomic features associated with provisional biological regions

**a** Cluster-average expression of three transcriptomic markers per biological class. The 12 genes used to construct the annotation were excluded before marker selection. **b** Cluster-average intensity of three metabolomic m/z features per biological class. Rows denote the four provisional biological classes and columns denote features grouped by their target class. Features were ranked by one-versus-rest Wilcoxon tests with tie correction and Benjamini–Hochberg adjustment. Only positive markers with FDR ≤ 0.05, target detection ≥ 10%, and target-minus-rest detection ≥ 5% were displayed. Values are normalize-total/log1p class averages, standardized within each feature across classes and clipped to [−2.5, 2.5]. Black outlines indicate the target class blocks.

## Fig. Z | Spatial localization of representative cross-omics features

Normalized/log1p abundance of the strongest held-out transcriptomic marker and strongest metabolomic m/z feature for each provisional class. Black outlines identify spots in the target class. Color limits are fixed at zero and the 99th percentile of positive values for each feature. Differential statistics were calculated by one-versus-rest Wilcoxon tests with tie correction and Benjamini–Hochberg adjustment.

The four-class labels are BioSFIB-domain biological interpretations following the SpatialMETA marker protocol. They are not pathological ground truth and were not used for model training, checkpoint selection, cluster-number selection, hyperparameter tuning, or ARI calculation. Source data are provided as CSV files in the figure directory.
"""
    path = output_dir / "figure_legends.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.rank_top < args.markers_per_class:
        raise ValueError("--rank-top must be at least --markers-per-class")
    if args.markers_per_class < 1:
        raise ValueError("--markers-per-class must be positive")

    annotated_path = resolve_project_path(args.annotated_h5ad)
    rna_path = resolve_project_path(args.rna_h5ad)
    sm_path = resolve_project_path(args.sm_h5ad)
    evidence_path = resolve_project_path(args.evidence_csv)
    output_dir = resolve_project_path(args.output_dir)
    for path in (annotated_path, rna_path, sm_path, evidence_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[four-class figures] loading frozen annotation and source tables")
    annotated = sc.read_h5ad(annotated_path)
    evidence = pd.read_csv(evidence_path)
    coords, annotations, clusters, cluster_order = validate_inputs(annotated, evidence)

    print("[four-class figures] drawing annotation/evidence summary")
    summary_paths, evidence_source = plot_summary_figure(
        coords, annotations, clusters, cluster_order, evidence, output_dir
    )
    evidence_source_path = output_dir / "four_class_annotation_evidence_source.csv"
    evidence_source.to_csv(evidence_source_path, index=False)
    # Export annotations for the supplementary factor analysis without gates.
    annotation_source = pd.DataFrame({
        "spot": annotated.obs_names.astype(str),
        "x": coords[:, 0],
        "y": coords[:, 1],
        "annotation": annotations,
    })
    annotation_source.to_csv(output_dir / "four_class_spot_annotation_source.csv", index=False)

    print("[four-class figures] preparing normalized/log1p original modalities")
    rna_adata = prepare_modality(rna_path, annotated, ANNOTATION_KEY, "RNA")
    sm_adata = prepare_modality(sm_path, annotated, ANNOTATION_KEY, "SM")
    rna_stats_path = output_dir / "four_class_RNA_marker_statistics.csv"
    sm_stats_path = output_dir / "four_class_SM_marker_statistics.csv"
    if args.reuse_marker_stats:
        if not rna_stats_path.is_file() or not sm_stats_path.is_file():
            raise FileNotFoundError("--reuse-marker-stats requested but marker statistics are absent")
        rna_stats = pd.read_csv(rna_stats_path, dtype={"cluster": str, "feature": str})
        sm_stats = pd.read_csv(sm_stats_path, dtype={"cluster": str, "feature": str})
        print("[four-class figures] reusing four-class marker statistics")
    else:
        print("[four-class figures] Wilcoxon RNA markers")
        rna_stats = rank_markers(rna_adata, ANNOTATION_KEY, "RNA", args.rank_top)
        print("[four-class figures] Wilcoxon metabolome markers")
        sm_stats = rank_markers(sm_adata, ANNOTATION_KEY, "SM", args.rank_top)
        rna_stats.to_csv(rna_stats_path, index=False)
        sm_stats.to_csv(sm_stats_path, index=False)

    rna_markers = select_class_markers(rna_stats, "RNA", args.markers_per_class)
    sm_markers = select_class_markers(sm_stats, "SM", args.markers_per_class)
    selected_markers = pd.concat([rna_markers, sm_markers], ignore_index=True)
    selected_markers_path = output_dir / "four_class_selected_markers.csv"
    selected_markers.to_csv(selected_markers_path, index=False)

    print("[four-class figures] drawing class marker heatmaps")
    rna_heatmap_source = build_class_heatmap_source(rna_adata, rna_markers, "RNA")
    sm_heatmap_source = build_class_heatmap_source(sm_adata, sm_markers, "SM")
    heatmap_source = pd.concat([rna_heatmap_source, sm_heatmap_source], ignore_index=True)
    heatmap_source_path = output_dir / "four_class_marker_heatmap_source.csv"
    heatmap_source.to_csv(heatmap_source_path, index=False)
    heatmap_paths = plot_marker_heatmaps(rna_heatmap_source, sm_heatmap_source, output_dir)

    print("[four-class figures] drawing representative spatial markers")
    representative_paths, representative_source = plot_representative_spatial_markers(
        rna_adata, sm_adata, rna_markers, sm_markers,
        coords, annotations, output_dir,
    )
    representative_source_path = output_dir / "four_class_representative_marker_source.csv"
    representative_source.to_csv(representative_source_path, index=False)
    legend_path = write_figure_legends(output_dir)

    class_counts = {group: int(np.sum(annotations == group)) for group in CLASS_ORDER}
    cluster_mapping = {
        group: [cluster for cluster in cluster_order if str(
            evidence.assign(biosfib_cluster=evidence["biosfib_cluster"].astype(str))
            .set_index("biosfib_cluster").loc[cluster, "marker_guided_candidate"]
        ) == group]
        for group in CLASS_ORDER
    }
    selected_summary = [
        {
            "modality": str(row.modality),
            "annotation": str(row.annotation),
            "rank": int(row.selected_rank),
            "feature": str(row.feature),
            "log2_fold_change": float(row.log2_fold_change),
            "fdr_bh": float(row.fdr_bh),
        }
        for row in selected_markers.itertuples(index=False)
    ]
    manifest = {
        "scientific_contract": {
            "core_conclusion": (
                "The provisional four-class regions are spatially coherent, retain fixed K14 "
                "subdomain heterogeneity, and exhibit distinct cross-omics marker patterns."
            ),
            "archetype": "image plate plus quantitative evidence",
            "backend": "Python/matplotlib only",
            "fixed_result": "spec001, epoch1500, exact z_out, PCA20, Mclust EEE K14",
            "annotation_role": "biological interpretation only; provisional; not ground truth",
            "prohibited_uses": [
                "model training", "checkpoint selection", "cluster-number selection",
                "hyperparameter tuning", "ARI evaluation",
            ],
            "expression_preprocessing": "per-modality normalize_total(target_sum=10000), then log1p",
            "marker_test": "one-versus-rest Wilcoxon with tie correction and BH-FDR",
            "held_out_rna_marker_rule": "the 12 annotation genes were excluded before display-marker selection",
        },
        "inputs": {
            "annotated_h5ad": str(annotated_path),
            "rna_h5ad": str(rna_path),
            "sm_h5ad": str(sm_path),
            "evidence_csv": str(evidence_path),
            "annotated_h5ad_sha256": file_hash(annotated_path),
            "evidence_csv_sha256": file_hash(evidence_path),
            "n_spots": int(annotated.n_obs),
            "class_counts": class_counts,
            "cluster_mapping": cluster_mapping,
        },
        "selected_markers": selected_summary,
        "source_data": {
            "annotation_evidence_rows": int(evidence_source.shape[0]),
            "marker_heatmap_rows": int(heatmap_source.shape[0]),
            "representative_marker_rows": int(representative_source.shape[0]),
        },
        "figures": {
            "annotation_summary": [str(path) for path in summary_paths],
            "marker_heatmaps": [str(path) for path in heatmap_paths],
            "representative_markers": [str(path) for path in representative_paths],
        },
        "figure_legends": str(legend_path),
    }
    manifest_path = output_dir / "four_class_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[four-class figures] complete")
    print(f"  output directory     : {output_dir}")
    print(f"  annotation summary   : {output_dir / 'figure_four_class_annotation_summary.png'}")
    print(f"  marker heatmaps      : {output_dir / 'figure_four_class_marker_heatmaps.png'}")
    print(f"  representative maps : {output_dir / 'figure_four_class_representative_markers.png'}")
    print(f"  manifest             : {manifest_path}")


if __name__ == "__main__":
    main()
