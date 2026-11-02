"""Create deterministic post-hoc factor figures for annotated Y7_T.

Scientific contract
-------------------
Core conclusion:
    Frozen BioSFIB representations contain spatially structured joint and
    modality-associated residual factors that can be related post hoc to the
    four provisional biological regions.
Evidence chain:
    (1) a SpaMV-F-like spatial atlas shows continuous factor scores and
    (2) a top-feature correlation comparison quantifies whether each factor is
    more transcriptome- or metabolome-associated.
Review risks:
    These are deterministic PCA factors extracted after BioSFIB training, not
    native SpaMV topics or explicitly learned BioSFIB private variables. The
    four-class annotation is used only to orient and arrange already-extracted
    factors; therefore class association is descriptive, not validation.

The command never trains, reclusters, selects a checkpoint, selects K, changes
the fixed embedding, or uses the provisional annotation as ground truth.
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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sf_model.config import load_config  # noqa: E402
from analysis.common.marker_helpers import (  # noqa: E402
    prepare_modality,
    save_figure_bundle,
)
from analysis.common.ssr_state import (  # noqa: E402
    choose_device,
    load_ssr_state,
    resolve_checkpoint,
    resolve_processed_data,
    to_numpy,
    torch_load,
)


DEFAULT_CONFIG = (
    "configs/meta/"
    "Y7_T.yaml"
)
DEFAULT_EVAL_H5AD = (
    "results/META/Y7_T/predictions/"
    "embedding_joint.h5ad"
)
DEFAULT_ANNOTATED_H5AD = (
    "results/analysis/meta/Y7_T/"
    "Y7_T_BioSFIB_K14_destvi_annotation.h5ad"
)
DEFAULT_ANNOTATION_CSV = (
    "results/analysis/meta/Y7_T/"
    "manuscript_four_class_figures_20260813/four_class_spot_annotation_source.csv"
)
DEFAULT_RNA_H5AD = "data/raw/renal/Y7_T/adata_joint_Y7_T_raw_ST.h5ad"
DEFAULT_SM_H5AD = "data/raw/renal/Y7_T/adata_joint_Y7_T_raw_SM.h5ad"
DEFAULT_OUTPUT_DIR = (
    "results/analysis/meta/Y7_T/"
    "manuscript_factor_atlas_20260813"
)

ANNOTATION_KEY = "biosfib_four_class_marker_guided"
CLASS_ORDER = ["Imm", "Endo", "Stro", "Mal"]
ATLAS_CLASS_ORDER = ["Stro", "Mal", "Imm", "Endo"]
CLASS_DISPLAY = {
    "Imm": "Immune",
    "Endo": "Endothelial",
    "Stro": "Stromal",
    "Mal": "Malignant",
}
CLASS_COLORS = {
    "Imm": "#D55E00",
    "Endo": "#0072B2",
    "Stro": "#009E73",
    "Mal": "#CC79A7",
}
RNA_COLOR = "#4C78A8"
SM_COLOR = "#E08B5B"
JOINT_COLOR = "#7570B3"
NEUTRAL = "#B6B6B6"
GRID = "#E3E3E3"


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
    parser = argparse.ArgumentParser(
        description="Draw deterministic post-hoc Y7_T factor atlas and omics-specificity figures."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-h5ad", default=DEFAULT_EVAL_H5AD)
    parser.add_argument("--annotated-h5ad", default=DEFAULT_ANNOTATED_H5AD)
    parser.add_argument("--annotation-csv", default=DEFAULT_ANNOTATION_CSV)
    parser.add_argument("--rna-h5ad", default=DEFAULT_RNA_H5AD)
    parser.add_argument("--sm-h5ad", default=DEFAULT_SM_H5AD)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--latent-components", type=int, default=20)
    parser.add_argument("--joint-factors-per-class", type=int, default=2)
    parser.add_argument("--modality-factors-per-class", type=int, default=1)
    parser.add_argument("--moran-k", type=int, default=6)
    parser.add_argument("--minimum-moran", type=float, default=0.05)
    parser.add_argument("--top-correlations", type=int, default=10)
    parser.add_argument("--max-features-per-modality", type=int, default=1000)
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


def _standardize_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"Expected a finite 2-D matrix, found {matrix.shape}")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=0)
    keep = scale > 1e-8
    if not np.any(keep):
        raise ValueError("Representation has no non-constant columns")
    return centered[:, keep] / scale[keep], keep


def residualize_modality(modality: np.ndarray, joint: np.ndarray) -> np.ndarray:
    """Remove the linear joint component with a fixed deterministic ridge fit."""
    x, _ = _standardize_columns(modality)
    z, _ = _standardize_columns(joint)
    alpha = 1e-3 * float(z.shape[0])
    gram = z.T @ z
    coefficient = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0], dtype=np.float64),
        z.T @ x,
    )
    residual = x - z @ coefficient
    residual, _ = _standardize_columns(residual)
    return residual


def physical_knn_edges(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k < 1 or k >= coords.shape[0]:
        raise ValueError(f"Invalid Moran k={k} for {coords.shape[0]} spots")
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    indices = neighbors.kneighbors(coords, return_distance=False)[:, 1:]
    directed = {(int(row), int(col)) for row in range(coords.shape[0]) for col in indices[row]}
    symmetric = sorted(directed | {(col, row) for row, col in directed})
    rows = np.fromiter((row for row, _ in symmetric), dtype=np.int64)
    cols = np.fromiter((col for _, col in symmetric), dtype=np.int64)
    return rows, cols


def morans_i(values: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    centered = np.asarray(values, dtype=np.float64).reshape(-1)
    centered = centered - centered.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return float("nan")
    numerator = float(np.sum(centered[rows] * centered[cols]))
    return float(centered.size / rows.size * numerator / denominator)


def deterministic_pca_factors(
    matrix: np.ndarray,
    family: str,
    n_components: int,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    standardized, _ = _standardize_columns(matrix)
    n_components = min(int(n_components), standardized.shape[0] - 1, standardized.shape[1])
    if n_components < 1:
        raise ValueError(f"No PCA components available for {family}")
    pca = PCA(n_components=n_components, svd_solver="full", whiten=False)
    scores = pca.fit_transform(standardized)
    records = []
    for index in range(scores.shape[1]):
        records.append(
            {
                "family": family,
                "component_index": index,
                "component_number": index + 1,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[index]),
                "moran_i": morans_i(scores[:, index], rows, cols),
                "score_std": float(scores[:, index].std(ddof=0)),
            }
        )
    return scores, pd.DataFrame(records)


def class_contrast_table(
    scores: np.ndarray,
    metadata: pd.DataFrame,
    annotations: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for component_index in range(scores.shape[1]):
        values = scores[:, component_index]
        scale = float(values.std(ddof=0))
        for annotation in CLASS_ORDER:
            target = annotations == annotation
            effect = float((values[target].mean() - values[~target].mean()) / max(scale, 1e-12))
            rows.append(
                {
                    "family": str(metadata.iloc[component_index]["family"]),
                    "component_index": component_index,
                    "component_number": component_index + 1,
                    "annotation": annotation,
                    "standardized_mean_contrast": effect,
                    "absolute_contrast": abs(effect),
                }
            )
    return pd.DataFrame(rows)


def select_factors_for_classes(
    scores_by_family: dict[str, np.ndarray],
    metadata_by_family: dict[str, pd.DataFrame],
    annotations: np.ndarray,
    minimum_moran: float,
    joint_per_class: int,
    modality_per_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_slots = {
        "joint": int(joint_per_class),
        "rna_residual": int(modality_per_class),
        "sm_residual": int(modality_per_class),
    }
    selected_rows = []
    all_contrasts = []
    prefixes = {"joint": "J", "rna_residual": "R", "sm_residual": "M"}
    display_families = {
        "joint": "Joint factor",
        "rna_residual": "RNA-associated factor",
        "sm_residual": "Metabolome-associated factor",
    }
    for family, slots_per_class in family_slots.items():
        scores = scores_by_family[family]
        metadata = metadata_by_family[family].copy()
        contrasts = class_contrast_table(scores, metadata, annotations)
        all_contrasts.append(contrasts)
        eligible = metadata[
            np.isfinite(metadata["moran_i"].to_numpy(float))
            & (metadata["moran_i"].to_numpy(float) >= float(minimum_moran))
        ].copy()
        required = len(CLASS_ORDER) * slots_per_class
        if eligible.shape[0] < required:
            raise ValueError(
                f"{family} has {eligible.shape[0]} factors with Moran's I >= {minimum_moran}, "
                f"but {required} are required by the fixed reporting protocol; "
                "check inputs before considering a separately documented exploratory protocol"
            )
        component_indices = eligible["component_index"].astype(int).tolist()
        slot_annotations = [
            annotation
            for annotation in CLASS_ORDER
            for _ in range(slots_per_class)
        ]
        contrast_lookup = contrasts.set_index(["annotation", "component_index"])["absolute_contrast"]
        utility = np.asarray(
            [
                [float(contrast_lookup.loc[(annotation, component_index)]) for component_index in component_indices]
                for annotation in slot_annotations
            ],
            dtype=np.float64,
        )
        row_indices, column_indices = linear_sum_assignment(-utility)
        if len(row_indices) != required:
            raise RuntimeError(f"Could not assign {required} unique factors for {family}")
        family_selection = []
        for slot_index, candidate_index in sorted(zip(row_indices, column_indices)):
            annotation = slot_annotations[int(slot_index)]
            component_index = int(component_indices[int(candidate_index)])
            signed_effect = float(
                contrasts.loc[
                    (contrasts["annotation"] == annotation)
                    & (contrasts["component_index"] == component_index),
                    "standardized_mean_contrast",
                ].iloc[0]
            )
            orientation = 1.0 if signed_effect >= 0.0 else -1.0
            meta = metadata.set_index("component_index").loc[component_index]
            family_selection.append(
                {
                    "family": family,
                    "family_display": display_families[family],
                    "annotation": annotation,
                    "component_index": component_index,
                    "component_number": component_index + 1,
                    "factor_id": f"{prefixes[family]}{component_index + 1}",
                    "orientation": orientation,
                    "oriented_standardized_mean_contrast": orientation * signed_effect,
                    "absolute_contrast": abs(signed_effect),
                    "moran_i": float(meta["moran_i"]),
                    "explained_variance_ratio": float(meta["explained_variance_ratio"]),
                }
            )
        family_frame = pd.DataFrame(family_selection)
        family_frame["within_class_rank"] = (
            family_frame.groupby("annotation", sort=False)["absolute_contrast"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        selected_rows.append(family_frame)
    selected = pd.concat(selected_rows, ignore_index=True)
    selected["annotation_order"] = selected["annotation"].map(
        {annotation: index for index, annotation in enumerate(CLASS_ORDER)}
    )
    selected["family_order"] = selected["family"].map(
        {"joint": 0, "rna_residual": 1, "sm_residual": 2}
    )
    selected = selected.sort_values(
        ["annotation_order", "family_order", "within_class_rank"], kind="stable"
    ).reset_index(drop=True)
    return selected, pd.concat(all_contrasts, ignore_index=True)


def build_selected_factor_sources(
    selected: pd.DataFrame,
    scores_by_family: dict[str, np.ndarray],
    annotations: np.ndarray,
    spot_names: np.ndarray,
    coords: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    summary_rows = []
    spot_rows = []
    oriented_scores = {}
    for row in selected.itertuples(index=False):
        values = (
            scores_by_family[str(row.family)][:, int(row.component_index)]
            * float(row.orientation)
        )
        target = annotations == str(row.annotation)
        q01, q99 = np.quantile(values, [0.01, 0.99])
        relative = np.clip((values - q01) / max(float(q99 - q01), 1e-12), 0.0, 1.0)
        oriented_scores[str(row.factor_id)] = values
        summary_rows.append(
            {
                **{key: getattr(row, key) for key in selected.columns},
                "target_mean": float(values[target].mean()),
                "rest_mean": float(values[~target].mean()),
                "plot_q01": float(q01),
                "plot_q99": float(q99),
            }
        )
        for spot, x, y, observed_annotation, value, relative_value in zip(
            spot_names, coords[:, 0], coords[:, 1], annotations, values, relative
        ):
            spot_rows.append(
                {
                    "spot": str(spot),
                    "x": float(x),
                    "y": float(y),
                    "observed_annotation": str(observed_annotation),
                    "assigned_annotation": str(row.annotation),
                    "family": str(row.family),
                    "factor_id": str(row.factor_id),
                    "oriented_factor_score": float(value),
                    "relative_factor_score_q01_q99": float(relative_value),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(spot_rows), oriented_scores


def _dense_matrix(matrix) -> np.ndarray:
    return matrix.toarray().astype(np.float64) if sparse.issparse(matrix) else np.asarray(matrix, dtype=np.float64)


def select_variable_feature_matrix(adata, maximum: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if maximum < 10:
        raise ValueError("--max-features-per-modality must be at least 10")
    matrix = adata.X
    if sparse.issparse(matrix):
        means = np.asarray(matrix.mean(axis=0)).ravel()
        second = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        variances = np.maximum(second - means * means, 0.0)
    else:
        dense = np.asarray(matrix, dtype=np.float64)
        variances = dense.var(axis=0, ddof=0)
    order = np.argsort(-variances, kind="stable")
    keep = order[: min(int(maximum), adata.n_vars)]
    selected_matrix = _dense_matrix(matrix[:, keep])
    selected_matrix, nonconstant = _standardize_columns(selected_matrix)
    kept_indices = keep[nonconstant]
    feature_names = adata.var_names.astype(str).to_numpy()[kept_indices]
    metadata = pd.DataFrame(
        {
            "feature": feature_names,
            "normalized_log1p_variance": variances[kept_indices],
            "variance_rank": np.arange(1, kept_indices.size + 1),
        }
    )
    return selected_matrix, feature_names, metadata


def factor_feature_correlations(
    factor_summary: pd.DataFrame,
    oriented_scores: dict[str, np.ndarray],
    rna_matrix: np.ndarray,
    rna_names: np.ndarray,
    sm_matrix: np.ndarray,
    sm_names: np.ndarray,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top_n < 1 or top_n > min(rna_matrix.shape[1], sm_matrix.shape[1]):
        raise ValueError("--top-correlations is outside the available feature count")
    detail_rows = []
    summary_rows = []
    metadata = factor_summary.set_index("factor_id")
    for factor_id, values in oriented_scores.items():
        standardized, _ = _standardize_columns(np.asarray(values).reshape(-1, 1))
        factor = standardized[:, 0]
        rna_correlation = factor @ rna_matrix / float(factor.size)
        sm_correlation = factor @ sm_matrix / float(factor.size)
        rna_order = np.argsort(-rna_correlation, kind="stable")[:top_n]
        sm_order = np.argsort(-sm_correlation, kind="stable")[:top_n]
        rna_top = rna_correlation[rna_order]
        sm_top = sm_correlation[sm_order]
        if np.min(rna_top) <= 0.0 or np.min(sm_top) <= 0.0:
            raise ValueError(f"{factor_id} lacks {top_n} positive correlations in both modalities")
        rna_mean = float(rna_top.mean())
        sm_mean = float(sm_top.mean())
        log2_fold = float(np.log2(rna_mean / sm_mean))
        row = metadata.loc[factor_id]
        summary_rows.append(
            {
                "factor_id": factor_id,
                "family": str(row["family"]),
                "annotation": str(row["annotation"]),
                "moran_i": float(row["moran_i"]),
                "mean_top_rna_pcc": rna_mean,
                "mean_top_sm_pcc": sm_mean,
                "log2_rna_over_sm_top_pcc": log2_fold,
                "top_n": int(top_n),
            }
        )
        for modality, indices, correlations, names in (
            ("RNA", rna_order, rna_correlation, rna_names),
            ("SM", sm_order, sm_correlation, sm_names),
        ):
            for rank, feature_index in enumerate(indices, start=1):
                detail_rows.append(
                    {
                        "factor_id": factor_id,
                        "family": str(row["family"]),
                        "annotation": str(row["annotation"]),
                        "modality": modality,
                        "rank": rank,
                        "feature": str(names[feature_index]),
                        "pearson_correlation": float(correlations[feature_index]),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def style_spatial_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.15, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def plot_factor_atlas(
    factor_summary: pd.DataFrame,
    factor_spot_source: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    columns_per_row = int(
        factor_summary.groupby("annotation", sort=False).size().unique().item()
    )
    if columns_per_row != 4:
        raise ValueError(f"The manuscript atlas expects four factors per class, found {columns_per_row}")
    fig, axes = plt.subplots(
        len(ATLAS_CLASS_ORDER), columns_per_row,
        figsize=(7.2, 6.5), squeeze=False,
        gridspec_kw={"hspace": 0.18, "wspace": 0.10},
    )
    point_size = float(np.clip(15000.0 / factor_spot_source["spot"].nunique(), 5.0, 10.0))
    family_order = {"joint": 0, "rna_residual": 1, "sm_residual": 2}
    header_labels = ["Joint", "Joint", "RNA residual", "Metabolome residual"]
    scatter_for_colorbar = None
    for row_index, annotation in enumerate(ATLAS_CLASS_ORDER):
        subset = factor_summary[factor_summary["annotation"] == annotation].copy()
        subset["family_order_plot"] = subset["family"].map(family_order)
        subset = subset.sort_values(
            ["family_order_plot", "within_class_rank"], kind="stable"
        )
        for column_index, factor in enumerate(subset.itertuples(index=False)):
            ax = axes[row_index, column_index]
            spots = factor_spot_source[
                factor_spot_source["factor_id"] == str(factor.factor_id)
            ]
            scatter_for_colorbar = ax.scatter(
                spots["x"], spots["y"],
                c=spots["relative_factor_score_q01_q99"],
                cmap="viridis", vmin=0.0, vmax=1.0,
                s=point_size, linewidths=0, rasterized=True,
            )
            style_spatial_axis(ax)
            ax.set_title(
                f"{factor.factor_id}  ·  I={float(factor.moran_i):.2f}",
                fontsize=6.7, pad=2.0,
            )
            if row_index == 0:
                ax.text(
                    0.5, 1.18, header_labels[column_index],
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=7.2, fontweight="bold",
                )
        row_axis = axes[row_index, 0]
        row_axis.add_patch(
            Rectangle(
                (-0.31, -0.04), 0.20, 1.08,
                transform=row_axis.transAxes, facecolor="white",
                edgecolor=CLASS_COLORS[annotation], linewidth=1.1,
                clip_on=False,
            )
        )
        row_axis.text(
            -0.21, 0.50, CLASS_DISPLAY[annotation],
            transform=row_axis.transAxes, rotation=90,
            ha="center", va="center", fontsize=7.2,
        )
    if scatter_for_colorbar is None:
        raise RuntimeError("No selected factors were plotted")
    colorbar_axis = fig.add_axes([0.915, 0.39, 0.013, 0.18])
    colorbar = fig.colorbar(scatter_for_colorbar, cax=colorbar_axis)
    colorbar.set_label("Relative factor score", fontsize=6.4)
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.ax.tick_params(labelsize=5.8, length=2, width=0.5)
    colorbar.outline.set_linewidth(0.45)
    panel_label(axes[0, 0], "a")
    fig.subplots_adjust(left=0.10, right=0.89, bottom=0.035, top=0.955)
    paths = save_figure_bundle(fig, output_dir / "figure_factor_spatial_atlas")
    plt.close(fig)
    return paths


def family_handles() -> list[Line2D]:
    specifications = [
        ("joint", "Joint", JOINT_COLOR, "o"),
        ("rna_residual", "RNA residual", RNA_COLOR, "s"),
        ("sm_residual", "Metabolome residual", SM_COLOR, "^"),
    ]
    return [
        Line2D(
            [0], [0], marker=marker, linestyle="none", markersize=5,
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.0, label=label,
        )
        for _, label, color, marker in specifications
    ]


def plot_factor_specificity(
    specificity: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    family_colors = {
        "joint": JOINT_COLOR,
        "rna_residual": RNA_COLOR,
        "sm_residual": SM_COLOR,
    }
    family_markers = {"joint": "o", "rna_residual": "s", "sm_residual": "^"}
    family_order = {"joint": 0, "rna_residual": 1, "sm_residual": 2}
    frame = specificity.copy()
    frame["annotation_order"] = frame["annotation"].map(
        {annotation: index for index, annotation in enumerate(ATLAS_CLASS_ORDER)}
    )
    frame["family_order"] = frame["family"].map(family_order)
    frame = frame.sort_values(
        ["annotation_order", "family_order", "factor_id"], kind="stable"
    ).reset_index(drop=True)

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.47)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    y = np.arange(frame.shape[0])
    values = frame["log2_rna_over_sm_top_pcc"].to_numpy(float)
    colors = [family_colors[family] for family in frame["family"]]
    ax_a.barh(y, values, height=0.68, color=colors, edgecolor="white", linewidth=0.35)
    ax_a.axvline(0.0, color="#333333", linewidth=0.8)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(
        [f"{CLASS_DISPLAY[row.annotation]} · {row.factor_id}" for row in frame.itertuples(index=False)],
        fontsize=5.8,
    )
    ax_a.invert_yaxis()
    ax_a.set_xlabel("log2(mean top-10 RNA PCC / mean top-10 SM PCC)")
    ax_a.set_title("Top-feature correlation fold change", pad=23)
    ax_a.grid(axis="x", color=GRID, linewidth=0.5)
    ax_a.set_axisbelow(True)
    annotation_values = frame["annotation"].to_numpy()
    boundaries = np.where(annotation_values[1:] != annotation_values[:-1])[0]
    for boundary in boundaries:
        ax_a.axhline(boundary + 0.5, color="#BBBBBB", linewidth=0.65)
    x_limit = max(0.25, float(np.max(np.abs(values))) * 1.18)
    ax_a.set_xlim(-x_limit, x_limit)
    ax_a.text(0.02, 1.01, "Metabolome-associated", transform=ax_a.transAxes,
              ha="left", va="bottom", fontsize=5.7, color=SM_COLOR)
    ax_a.text(0.98, 1.01, "RNA-associated", transform=ax_a.transAxes,
              ha="right", va="bottom", fontsize=5.7, color=RNA_COLOR)

    x = frame["mean_top_sm_pcc"].to_numpy(float)
    y_scatter = frame["mean_top_rna_pcc"].to_numpy(float)
    low = max(0.0, float(min(x.min(), y_scatter.min())) * 0.92)
    high = float(max(x.max(), y_scatter.max())) * 1.08
    ax_b.plot([low, high], [low, high], color="#888888", linestyle="--", linewidth=0.75)
    for family in ("joint", "rna_residual", "sm_residual"):
        mask = frame["family"] == family
        ax_b.scatter(
            x[mask], y_scatter[mask], s=32,
            marker=family_markers[family],
            c=[CLASS_COLORS[annotation] for annotation in frame.loc[mask, "annotation"]],
            edgecolors=family_colors[family], linewidths=0.9, zorder=3,
        )
    label_rows = frame[
        (frame["family"] == "joint")
        | (
            (frame["family"] == "rna_residual")
            &
            (frame["log2_rna_over_sm_top_pcc"].abs() >= 0.40)
            & (frame[["mean_top_sm_pcc", "mean_top_rna_pcc"]].max(axis=1) >= 0.09)
        )
    ]
    for row in label_rows.itertuples(index=False):
        offset = (3.0, 2.0)
        if row.family == "rna_residual":
            offset = (4.0, 5.0)
        elif row.family == "sm_residual":
            offset = (4.0, -7.0)
        ax_b.annotate(
            row.factor_id,
            (float(row.mean_top_sm_pcc), float(row.mean_top_rna_pcc)),
            xytext=offset, textcoords="offset points", fontsize=5.3,
        )
    ax_b.set_xlim(low, high)
    ax_b.set_ylim(low, high)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("Mean top-10 metabolome PCC")
    ax_b.set_ylabel("Mean top-10 RNA PCC")
    ax_b.set_title("Cross-omics correlation strength", pad=5)
    ax_b.grid(color=GRID, linewidth=0.5)
    ax_b.set_axisbelow(True)
    class_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor=CLASS_COLORS[annotation], markeredgecolor="none",
            label=CLASS_DISPLAY[annotation],
        )
        for annotation in ATLAS_CLASS_ORDER
    ]
    fig.legend(
        handles=family_handles(), loc="lower left", bbox_to_anchor=(0.61, 0.025),
        ncol=1, borderaxespad=0.0, handletextpad=0.4, labelspacing=0.3,
    )
    fig.legend(
        handles=class_handles, loc="lower left", bbox_to_anchor=(0.82, 0.025),
        ncol=1, borderaxespad=0.0, handletextpad=0.4, labelspacing=0.3,
    )
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.29, top=0.91)
    paths = save_figure_bundle(fig, output_dir / "figure_factor_omics_specificity")
    plt.close(fig)
    return paths


def write_legends(output_dir: Path) -> Path:
    text = """# Figure legends

## Fig. X | Spatial organization of post-hoc factors in marker-guided BioSFIB regions

**a** Continuous spatial scores of deterministic factors extracted from the frozen BioSFIB Y7_T representations. Two joint spectral factors, one RNA-associated residual factor, and one metabolome-associated residual factor are displayed for each provisional biological region. Factor extraction used full-SVD PCA without random initialization; physical-neighbor Moran's I was calculated before annotation-based display assignment. Components were assigned uniquely to regions by the absolute standardized one-versus-rest mean contrast and oriented so that the assigned region had the higher mean. Scores are independently clipped at the 1st and 99th percentiles and scaled to [0,1] for visualization. These post-hoc factors are not native SpaMV topics or explicitly trained BioSFIB private variables. The four marker-guided labels are provisional biological interpretations and are not pathological ground truth. Source data are provided as CSV files.

## Fig. Y | Transcriptome and metabolome association of post-hoc spatial factors

**a** Log2 ratio of the mean top ten positive Pearson correlations between each selected factor and normalized/log-transformed RNA or metabolome features. Positive values indicate stronger transcriptome association and negative values indicate stronger metabolome association. **b** Mean top-ten transcriptome and metabolome correlations for the same factors. The dashed line denotes equal association. Factor families are encoded by shape and provisional regions by color. The 1,000 most variable non-constant features from each modality were evaluated. Correlations are descriptive and do not establish molecular identity or explicit shared/private disentanglement. Metabolome features remain m/z values unless independently identified. Source data are provided as CSV files.

All factors were extracted after fixing spec001, epoch 1500, exact z_out, PCA20, Mclust EEE, and K = 14. No model training, checkpoint selection, cluster-number selection, hyperparameter tuning, reclustering, or ARI evaluation was performed.
"""
    path = output_dir / "figure_legends.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.latent_components < 8:
        raise ValueError("--latent-components must be at least 8 for two joint factors per class")
    if args.joint_factors_per_class != 2 or args.modality_factors_per_class != 1:
        raise ValueError(
            "The frozen manuscript layout requires exactly two joint and one factor per modality per class"
        )

    config_path = resolve_project_path(args.config)
    eval_path = resolve_project_path(args.eval_h5ad)
    annotated_path = resolve_project_path(args.annotated_h5ad)
    annotation_csv_path = resolve_project_path(args.annotation_csv)
    rna_path = resolve_project_path(args.rna_h5ad)
    sm_path = resolve_project_path(args.sm_h5ad)
    output_dir = resolve_project_path(args.output_dir)
    for path in (config_path, eval_path, annotated_path, annotation_csv_path, rna_path, sm_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    processed_path = resolve_processed_data(config)
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    for path in (processed_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("[factor atlas] validating frozen Y7_T inputs")
    selected = sc.read_h5ad(eval_path)
    annotation_table = pd.read_csv(annotation_csv_path)
    required_annotation_columns = {"spot", "x", "y", "annotation"}
    missing_annotation_columns = sorted(required_annotation_columns.difference(annotation_table.columns))
    if missing_annotation_columns:
        raise KeyError(f"Annotation CSV lacks columns: {missing_annotation_columns}")
    if selected.n_obs != 1914 or annotation_table.shape[0] != 1914:
        raise ValueError(
            f"Y7_T requires 1914 spots; found selected={selected.n_obs}, annotation={annotation_table.shape[0]}"
        )
    if not np.array_equal(
        selected.obs_names.astype(str), annotation_table["spot"].astype(str).to_numpy()
    ):
        raise ValueError("Selected embedding and annotation CSV have different spot order")
    if "spatial" not in selected.obsm:
        raise KeyError("Selected H5AD requires obsm['spatial']")
    coords = np.asarray(selected.obsm["spatial"], dtype=np.float64)[:, :2]
    if not np.allclose(
        annotation_table[["x", "y"]].to_numpy(np.float64), coords, rtol=0.0, atol=1e-8
    ):
        raise ValueError("Selected embedding and annotation CSV coordinates differ")
    annotations = annotation_table["annotation"].astype(str).to_numpy()
    if set(np.unique(annotations)) != set(CLASS_ORDER):
        raise ValueError(f"Expected exactly {CLASS_ORDER}, found {sorted(np.unique(annotations))}")
    selected.obs[ANNOTATION_KEY] = pd.Categorical(
        annotations, categories=CLASS_ORDER, ordered=True
    )

    print("[factor atlas] one eval-mode pass through the frozen epoch-1500 checkpoint")
    data_dict = torch_load(processed_path, map_location="cpu", weights_only=False)
    device = choose_device(args.device)
    state = load_ssr_state(config, checkpoint_path, data_dict, device)
    z_out = to_numpy(state["z_out"]).astype(np.float64)
    embedding_key = next((key for key in ("X_model", "BioSFIB") if key in selected.obsm), None)
    if embedding_key is None:
        raise KeyError("Selected H5AD lacks X_model/BioSFIB embedding")
    stored_embedding = np.asarray(selected.obsm[embedding_key], dtype=np.float64)
    if stored_embedding.shape != z_out.shape:
        raise ValueError(
            f"Frozen checkpoint embedding shape {z_out.shape} differs from stored shape {stored_embedding.shape}"
        )
    if not np.allclose(stored_embedding, z_out, rtol=2e-5, atol=2e-5):
        maximum_error = float(np.max(np.abs(stored_embedding - z_out)))
        raise ValueError(f"Frozen checkpoint does not match selected embedding; max error={maximum_error:.6g}")
    modal_latents = getattr(state["model"], "_last_modal_latents", None)
    if modal_latents is None or len(modal_latents) != 2:
        raise RuntimeError("Frozen BioSFIB forward pass did not expose the two modality encodings")
    h_rna = to_numpy(modal_latents[0]).astype(np.float64)
    h_sm = to_numpy(modal_latents[1]).astype(np.float64)
    z_spec = to_numpy(state["z_spec"]).astype(np.float64)
    if not (h_rna.shape[0] == h_sm.shape[0] == z_spec.shape[0] == selected.n_obs):
        raise ValueError("Frozen latent spot counts do not match the annotation")

    print("[factor atlas] deterministic full-SVD PCA of joint and residual representations")
    edge_rows, edge_cols = physical_knn_edges(coords, args.moran_k)
    representation_by_family = {
        "joint": z_spec,
        "rna_residual": residualize_modality(h_rna, z_spec),
        "sm_residual": residualize_modality(h_sm, z_spec),
    }
    scores_by_family = {}
    metadata_by_family = {}
    for family, matrix in representation_by_family.items():
        scores, metadata = deterministic_pca_factors(
            matrix, family, args.latent_components, edge_rows, edge_cols
        )
        scores_by_family[family] = scores
        metadata_by_family[family] = metadata
    component_metadata = pd.concat(metadata_by_family.values(), ignore_index=True)
    component_metadata_path = output_dir / "all_factor_spatial_statistics.csv"
    component_metadata.to_csv(component_metadata_path, index=False)

    print("[factor atlas] post-hoc unique assignment of spatial factors to four classes")
    selected_factors, all_contrasts = select_factors_for_classes(
        scores_by_family, metadata_by_family, annotations,
        args.minimum_moran, args.joint_factors_per_class,
        args.modality_factors_per_class,
    )
    all_contrasts_path = output_dir / "all_factor_annotation_contrasts.csv"
    all_contrasts.to_csv(all_contrasts_path, index=False)
    factor_summary, factor_spot_source, oriented_scores = build_selected_factor_sources(
        selected_factors, scores_by_family, annotations,
        selected.obs_names.astype(str).to_numpy(), coords,
    )
    factor_summary_path = output_dir / "selected_factor_summary.csv"
    factor_spot_path = output_dir / "selected_factor_spot_source.csv"
    factor_summary.to_csv(factor_summary_path, index=False)
    factor_spot_source.to_csv(factor_spot_path, index=False)

    print("[factor atlas] preparing original normalized/log1p RNA and metabolome features")
    rna_adata = prepare_modality(rna_path, selected, ANNOTATION_KEY, "RNA")
    sm_adata = prepare_modality(sm_path, selected, ANNOTATION_KEY, "SM")
    rna_matrix, rna_names, rna_feature_meta = select_variable_feature_matrix(
        rna_adata, args.max_features_per_modality
    )
    sm_matrix, sm_names, sm_feature_meta = select_variable_feature_matrix(
        sm_adata, args.max_features_per_modality
    )
    rna_feature_meta["modality"] = "RNA"
    sm_feature_meta["modality"] = "SM"
    feature_selection = pd.concat([rna_feature_meta, sm_feature_meta], ignore_index=True)
    feature_selection_path = output_dir / "correlation_feature_selection.csv"
    feature_selection.to_csv(feature_selection_path, index=False)
    specificity, top_features = factor_feature_correlations(
        factor_summary, oriented_scores,
        rna_matrix, rna_names, sm_matrix, sm_names,
        args.top_correlations,
    )
    specificity_path = output_dir / "factor_omics_specificity_source.csv"
    top_features_path = output_dir / "factor_top_correlated_features.csv"
    specificity.to_csv(specificity_path, index=False)
    top_features.to_csv(top_features_path, index=False)

    print("[factor atlas] drawing manuscript figures")
    atlas_paths = plot_factor_atlas(factor_summary, factor_spot_source, output_dir)
    specificity_paths = plot_factor_specificity(specificity, output_dir)
    legend_path = write_legends(output_dir)

    manifest = {
        "scientific_contract": {
            "core_conclusion": (
                "Frozen BioSFIB representations contain spatially structured joint and "
                "modality-associated residual factors related post hoc to four provisional regions."
            ),
            "archetype": "spatial image plate plus quantitative specificity",
            "backend": "Python/matplotlib only",
            "fixed_result": "spec001, epoch1500, exact z_out, PCA20, Mclust EEE K14",
            "factor_definition": {
                "joint": "full-SVD PCA of frozen z_spec",
                "rna_associated": "full-SVD PCA after ridge residualizing frozen RNA encoding on z_spec",
                "metabolome_associated": "full-SVD PCA after ridge residualizing frozen SM encoding on z_spec",
            },
            "terminology_boundary": (
                "post-hoc factors; not native SpaMV topics and not explicitly trained BioSFIB private variables"
            ),
            "annotation_role": "display assignment and biological orientation only; provisional, not ground truth",
            "prohibited_uses": [
                "model training", "checkpoint selection", "cluster-number selection",
                "hyperparameter tuning", "reclustering", "ARI evaluation",
            ],
            "randomness": "none; PCA uses deterministic full SVD",
            "specificity_metric": (
                "log2 ratio of mean top-10 positive Pearson correlations with RNA versus SM features"
            ),
        },
        "inputs": {
            "config": str(config_path),
            "processed_data": str(processed_path),
            "checkpoint": str(checkpoint_path),
            "selected_h5ad": str(eval_path),
            "annotated_h5ad": str(annotated_path),
            "annotation_csv": str(annotation_csv_path),
            "rna_h5ad": str(rna_path),
            "sm_h5ad": str(sm_path),
            "checkpoint_sha256": file_hash(checkpoint_path),
            "selected_h5ad_sha256": file_hash(eval_path),
            "annotated_h5ad_sha256": file_hash(annotated_path),
            "annotation_csv_sha256": file_hash(annotation_csv_path),
            "n_spots": int(selected.n_obs),
            "class_counts": {
                annotation: int(np.sum(annotations == annotation)) for annotation in CLASS_ORDER
            },
        },
        "parameters": {
            "latent_components_per_family": int(args.latent_components),
            "moran_k": int(args.moran_k),
            "minimum_moran_i": float(args.minimum_moran),
            "joint_factors_per_class": int(args.joint_factors_per_class),
            "modality_factors_per_class": int(args.modality_factors_per_class),
            "top_correlations": int(args.top_correlations),
            "max_features_per_modality": int(args.max_features_per_modality),
        },
        "source_data": {
            "all_factor_spatial_statistics": str(component_metadata_path),
            "all_factor_annotation_contrasts": str(all_contrasts_path),
            "selected_factor_summary": str(factor_summary_path),
            "selected_factor_spot_source": str(factor_spot_path),
            "correlation_feature_selection": str(feature_selection_path),
            "factor_omics_specificity": str(specificity_path),
            "factor_top_correlated_features": str(top_features_path),
        },
        "figures": {
            "factor_spatial_atlas": [str(path) for path in atlas_paths],
            "factor_omics_specificity": [str(path) for path in specificity_paths],
        },
        "figure_legends": str(legend_path),
    }
    manifest_path = output_dir / "factor_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[factor atlas] complete")
    print(f"  spatial atlas      : {output_dir / 'figure_factor_spatial_atlas.png'}")
    print(f"  omics specificity  : {output_dir / 'figure_factor_omics_specificity.png'}")
    print(f"  manifest           : {manifest_path}")


if __name__ == "__main__":
    main()
