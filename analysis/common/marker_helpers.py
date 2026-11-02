"""Shared marker-statistics helpers extracted from the fixed Y7_T analysis.

These helpers use original count/intensity matrices and fixed cluster labels.
They do not train, recluster or select checkpoints. Deterministic display rules
are not evidence of preregistration or independent biological validation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RNA_COLOR = "#4C78A8"
SM_COLOR = "#E08B5B"
GRID = "#D9D9D9"
RNA_CMAP = LinearSegmentedColormap.from_list("rna_expression", ["#F7FAFC", RNA_COLOR])
SM_CMAP = LinearSegmentedColormap.from_list("sm_expression", ["#FFF9F3", SM_COLOR])
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "cluster_zscore", ["#3F6FAE", "#F8F8F8", "#C6534F"]
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
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


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()



def _cluster_sort_key(value: str) -> tuple[int, float | str]:
    text = str(value)
    match = re.fullmatch(r"[Cc]?(-?\d+(?:\.\d+)?)", text.strip())
    if match:
        return (0, float(match.group(1)))
    return (1, text)



def save_figure_bundle(fig: plt.Figure, output_stem: Path) -> list[Path]:
    paths = []
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
        (".tiff", {"dpi": 600}),
    ):
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **kwargs)
        paths.append(path)
    return paths



def _dense_vector(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().reshape(-1).astype(np.float64)
    return np.asarray(matrix, dtype=np.float64).reshape(-1)



def _validate_modal_contract(modal, selected, modal_name: str) -> None:
    if modal.n_obs != selected.n_obs:
        raise ValueError(f"{modal_name} spot count mismatch: {modal.n_obs} versus {selected.n_obs}")
    if not np.array_equal(modal.obs_names.astype(str), selected.obs_names.astype(str)):
        raise ValueError(f"{modal_name} observation order does not match the selected H5AD")
    if "spatial" not in modal.obsm or "spatial" not in selected.obsm:
        raise KeyError(f"{modal_name}/selected H5AD requires obsm['spatial']")
    modal_coords = np.asarray(modal.obsm["spatial"], dtype=np.float64)[:, :2]
    selected_coords = np.asarray(selected.obsm["spatial"], dtype=np.float64)[:, :2]
    if not np.allclose(modal_coords, selected_coords, rtol=0.0, atol=1e-8):
        raise ValueError(f"{modal_name} spatial coordinates do not match the selected H5AD")



def prepare_modality(path: Path, selected, cluster_key: str, modality: str):
    adata = sc.read_h5ad(path)
    _validate_modal_contract(adata, selected, modality)
    adata.obs[cluster_key] = pd.Categorical(
        selected.obs[cluster_key].astype(str).to_numpy(),
        categories=sorted(selected.obs[cluster_key].astype(str).unique(), key=_cluster_sort_key),
        ordered=True,
    )
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata



def rank_markers(adata, cluster_key: str, modality: str, rank_top: int) -> pd.DataFrame:
    n_features = min(max(1, int(rank_top)), int(adata.n_vars))
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        use_raw=False,
        method="wilcoxon",
        tie_correct=True,
        pts=True,
        n_genes=n_features,
    )
    table = sc.get.rank_genes_groups_df(adata, group=None).rename(
        columns={
            "group": "cluster",
            "names": "feature",
            "scores": "wilcoxon_z",
            "logfoldchanges": "log2_fold_change",
            "pvals": "p_value",
            "pvals_adj": "fdr_bh",
            "pct_nz_group": "fraction_in_cluster",
            "pct_nz_reference": "fraction_outside_cluster",
        }
    )
    required = [
        "cluster",
        "feature",
        "wilcoxon_z",
        "log2_fold_change",
        "p_value",
        "fdr_bh",
        "fraction_in_cluster",
        "fraction_outside_cluster",
    ]
    missing = [column for column in required if column not in table]
    if missing:
        raise KeyError(f"Scanpy marker table is missing columns: {missing}")
    table = table[required].copy()
    table["cluster"] = table["cluster"].astype(str)
    table["feature"] = table["feature"].astype(str)
    table["modality"] = modality
    table["detection_difference"] = (
        table["fraction_in_cluster"] - table["fraction_outside_cluster"]
    )
    table["marker_strength"] = (
        table["log2_fold_change"].clip(lower=0.0)
        * table["detection_difference"].clip(lower=0.0)
    )
    table["passes_primary_filter"] = (
        np.isfinite(table["fdr_bh"])
        & (table["fdr_bh"] <= 0.05)
        & (table["log2_fold_change"] > 0.0)
        & (table["detection_difference"] > 0.0)
    )
    table = table.sort_values(
        ["cluster", "passes_primary_filter", "marker_strength", "wilcoxon_z", "feature"],
        ascending=[True, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    table["within_cluster_rank"] = table.groupby("cluster", sort=False).cumcount() + 1
    return table



def choose_unique_markers(stats: pd.DataFrame, clusters: list[str], n_per_cluster: int) -> pd.DataFrame:
    chosen = []
    used_features: set[str] = set()
    for cluster in clusters:
        sub = stats[stats["cluster"] == cluster].copy()
        primary = sub[
            sub["passes_primary_filter"]
            & (sub["fraction_in_cluster"] >= 0.10)
            & (sub["detection_difference"] >= 0.05)
        ]
        candidates = pd.concat([primary, sub[~sub.index.isin(primary.index)]], ignore_index=False)
        cluster_rows = []
        for _, row in candidates.iterrows():
            feature = str(row["feature"])
            if feature in used_features:
                continue
            row = row.copy()
            row["selection_filter"] = (
                "robust_primary"
                if row.name in primary.index
                else "fallback_ranked"
            )
            cluster_rows.append(row)
            used_features.add(feature)
            if len(cluster_rows) >= int(n_per_cluster):
                break
        if len(cluster_rows) < int(n_per_cluster):
            raise ValueError(f"Could not choose {n_per_cluster} unique markers for cluster {cluster}")
        for local_rank, row in enumerate(cluster_rows, start=1):
            row["selected_rank"] = local_rank
            chosen.append(row)
    return pd.DataFrame(chosen).reset_index(drop=True)



def _display_feature(feature: str, modality: str) -> str:
    feature = str(feature)
    if modality == "SM":
        try:
            return f"m/z {float(feature):.4f}"
        except ValueError:
            return feature
    return feature



def build_heatmap_source(
    adata,
    markers: pd.DataFrame,
    clusters: list[str],
    cluster_key: str,
    modality: str,
) -> pd.DataFrame:
    labels = adata.obs[cluster_key].astype(str).to_numpy()
    rows = []
    markers = markers.sort_values(
        ["cluster", "selected_rank"],
        key=lambda series: series.map(_cluster_sort_key) if series.name == "cluster" else series,
        kind="stable",
    )
    for _, marker in markers.iterrows():
        target_cluster = str(marker["cluster"])
        feature = str(marker["feature"])
        expression = _dense_vector(adata[:, feature].X)
        means = np.asarray([np.mean(expression[labels == cluster]) for cluster in clusters], dtype=float)
        std = float(np.std(means))
        z_values = (means - float(np.mean(means))) / std if std > 1e-12 else np.zeros_like(means)
        z_values = np.clip(z_values, -2.5, 2.5)
        for cluster, mean_value, z_value in zip(clusters, means, z_values):
            rows.append(
                {
                    "modality": modality,
                    "target_cluster": target_cluster,
                    "selected_rank": int(marker["selected_rank"]),
                    "feature": feature,
                    "feature_display": _display_feature(feature, modality),
                    "cluster": cluster,
                    "mean_normalized_log1p_expression": float(mean_value),
                    "row_z_score": float(z_value),
                    "log2_fold_change": float(marker["log2_fold_change"]),
                    "fdr_bh": float(marker["fdr_bh"]),
                    "marker_strength": float(marker["marker_strength"]),
                    "selection_filter": str(marker["selection_filter"]),
                }
            )
    return pd.DataFrame(rows)
