"""Select and plot well-recovered BioSFIB spatial domains.

The input H5AD is always read-only.  The script independently rebuilds the
cluster-by-truth contingency table, compares configurable Hungarian
assignments, computes domain-level overlap metrics, applies fixed thresholds,
and writes audit CSV files plus one PDF image plate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from typing import Any

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment


SUPPORTED_ASSIGNMENT_METRICS = {"overlap", "f1", "iou"}
SUPPORTED_PANEL_ROWS = {"truth", "prediction", "agreement"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and plot well-recovered BioSFIB spatial domains."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis/benchmark/biosfib_recovered_domains_config.yaml"),
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--analysis",
        help="Analysis name under analyses; defaults to YAML default_analysis.",
    )
    parser.add_argument("--list", action="store_true", help="List configured analyses and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print selected domains without writing outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output PDF/CSV files.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping.")
    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_analysis(config: dict[str, Any], requested: str | None) -> tuple[str, dict[str, Any]]:
    analyses = config.get("analyses", {})
    if not isinstance(analyses, dict) or not analyses:
        raise KeyError("YAML must contain at least one entry under analyses.")
    name = requested or config.get("default_analysis")
    if not name:
        raise KeyError("No --analysis was supplied and YAML default_analysis is empty.")
    if name not in analyses:
        raise KeyError(f"Unknown analysis {name!r}. Available: {', '.join(analyses)}")
    spec = deep_merge(config.get("defaults", {}), analyses[name])
    required = ["input_h5ad", "cluster_key", "truth_key", "spatial_key", "output_dir"]
    missing = [key for key in required if not spec.get(key)]
    if missing:
        raise KeyError(f"Analysis {name!r} is missing required YAML fields: {missing}")
    validate_spec(spec)
    return name, spec


def validate_spec(spec: dict[str, Any]) -> None:
    matching = spec["matching"]
    assignment_metric = str(matching["assignment_metric"]).lower()
    comparison_metrics = [str(value).lower() for value in matching["comparison_metrics"]]
    requested_metrics = {assignment_metric, *comparison_metrics}
    unsupported = sorted(requested_metrics.difference(SUPPORTED_ASSIGNMENT_METRICS))
    if unsupported:
        raise ValueError(
            f"Unsupported Hungarian assignment metrics {unsupported}; "
            f"choose from {sorted(SUPPORTED_ASSIGNMENT_METRICS)}."
        )
    panel_rows = [str(value).lower() for value in spec["plot"]["panel_rows"]]
    if not panel_rows:
        raise ValueError("plot.panel_rows must contain at least one row.")
    unsupported_rows = sorted(set(panel_rows).difference(SUPPORTED_PANEL_ROWS))
    if unsupported_rows:
        raise ValueError(
            f"Unsupported plot.panel_rows {unsupported_rows}; "
            f"choose from {sorted(SUPPORTED_PANEL_ROWS)}."
        )
    rotate_cw = int(spec["spatial"]["transform"]["rotate_cw"])
    if rotate_cw not in {0, 90, 180, 270}:
        raise ValueError("spatial.transform.rotate_cw must be one of 0, 90, 180, 270.")
    target_span = spec["spatial"].get("display_target_span")
    if target_span is not None:
        if len(target_span) != 2 or any(float(value) <= 0 for value in target_span):
            raise ValueError("spatial.display_target_span must be null or two positive values.")
    for key in ("min_purity", "min_recall", "min_f1", "min_iou"):
        value = float(spec["selection"][key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"selection.{key} must be between 0 and 1.")
    for key in ("min_overlap_spots", "min_cluster_spots", "min_truth_spots"):
        if int(spec["selection"][key]) < 0:
            raise ValueError(f"selection.{key} must be non-negative.")
    selection_mode = str(spec["selection"].get("mode", "threshold")).lower()
    if selection_mode not in {"threshold", "top_n"}:
        raise ValueError("selection.mode must be either 'threshold' or 'top_n'.")
    if int(spec["selection"].get("top_n", 0)) <= 0:
        raise ValueError("selection.top_n must be a positive integer.")
    max_domains = spec["selection"].get("max_domains")
    if max_domains is not None and int(max_domains) <= 0:
        raise ValueError("selection.max_domains must be null or a positive integer.")


def resolve_path(config_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_root / path


def natural_label_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


def valid_label_mask(values: pd.Series, ignored_values: list[str]) -> np.ndarray:
    strings = values.astype("string")
    ignored = {str(value).strip().lower() for value in ignored_values}
    normalized = strings.fillna("").str.strip().str.lower()
    return (strings.notna() & ~normalized.isin(ignored)).to_numpy(dtype=bool)


def build_contingency(
    clusters: pd.Series,
    truth: pd.Series,
    ignored_clusters: list[str],
    ignored_truth: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    cluster_values = clusters.astype("string")
    truth_values = truth.astype("string")
    valid = valid_label_mask(cluster_values, ignored_clusters) & valid_label_mask(
        truth_values, ignored_truth
    )
    if not valid.any():
        raise ValueError("No spots have both a valid BioSFIB cluster and valid truth label.")
    cluster_labels = sorted(cluster_values[valid].unique().tolist(), key=natural_label_key)
    truth_labels = sorted(truth_values[valid].unique().tolist(), key=natural_label_key)
    contingency = pd.crosstab(
        pd.Categorical(cluster_values[valid], categories=cluster_labels),
        pd.Categorical(truth_values[valid], categories=truth_labels),
        dropna=False,
    ).reindex(index=cluster_labels, columns=truth_labels, fill_value=0)
    contingency.index.name = "biosfib_cluster"
    contingency.columns.name = "truth_label"
    return contingency.astype(int), valid


def score_matrices(contingency: pd.DataFrame) -> dict[str, np.ndarray]:
    overlap = contingency.to_numpy(dtype=float)
    cluster_sizes = overlap.sum(axis=1, keepdims=True)
    truth_sizes = overlap.sum(axis=0, keepdims=True)
    purity = np.divide(overlap, cluster_sizes, out=np.zeros_like(overlap), where=cluster_sizes > 0)
    recall = np.divide(overlap, truth_sizes, out=np.zeros_like(overlap), where=truth_sizes > 0)
    f1_denominator = cluster_sizes + truth_sizes
    f1 = np.divide(2.0 * overlap, f1_denominator, out=np.zeros_like(overlap), where=f1_denominator > 0)
    iou_denominator = cluster_sizes + truth_sizes - overlap
    iou = np.divide(overlap, iou_denominator, out=np.zeros_like(overlap), where=iou_denominator > 0)
    return {
        "overlap": overlap,
        "purity": purity,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def hungarian_assignment(
    contingency: pd.DataFrame, matrices: dict[str, np.ndarray], metric: str
) -> dict[str, str | None]:
    weights = matrices[metric]
    rows, columns = linear_sum_assignment(weights, maximize=True)
    mapping: dict[str, str | None] = {str(label): None for label in contingency.index}
    for row, column in zip(rows, columns):
        if matrices["overlap"][row, column] <= 0:
            continue
        mapping[str(contingency.index[row])] = str(contingency.columns[column])
    return mapping


def build_score_table(
    contingency: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    spec: dict[str, Any],
) -> pd.DataFrame:
    matching = spec["matching"]
    selection = spec["selection"]
    assignment_metric = str(matching["assignment_metric"]).lower()
    comparison_metrics = list(
        dict.fromkeys(
            [assignment_metric]
            + [str(value).lower() for value in matching["comparison_metrics"]]
        )
    )
    assignments = {
        metric: hungarian_assignment(contingency, matrices, metric)
        for metric in comparison_metrics
    }
    aliases = {str(key): str(value) for key, value in matching.get("label_aliases", {}).items()}
    cluster_positions = {str(value): index for index, value in enumerate(contingency.index)}
    truth_positions = {str(value): index for index, value in enumerate(contingency.columns)}
    records: list[dict[str, Any]] = []

    for cluster in map(str, contingency.index):
        matched_truth = assignments[assignment_metric].get(cluster)
        assignment_labels = [assignments[metric].get(cluster) for metric in comparison_metrics]
        assignment_agreement = matched_truth is not None and all(
            label == matched_truth for label in assignment_labels
        )
        record: dict[str, Any] = {
            "biosfib_cluster": cluster,
            "assignment_metric": assignment_metric,
            "matched_truth_label": matched_truth,
            "display_truth_label": aliases.get(matched_truth, matched_truth) if matched_truth else None,
            "assignment_agreement": assignment_agreement,
        }
        for metric in comparison_metrics:
            record[f"{metric}_matched_truth_label"] = assignments[metric].get(cluster)

        if matched_truth is None:
            record.update(
                {
                    "overlap_spots": 0,
                    "cluster_spots": int(contingency.loc[cluster].sum()),
                    "truth_spots": np.nan,
                    "purity": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "iou": 0.0,
                }
            )
        else:
            row = cluster_positions[cluster]
            column = truth_positions[matched_truth]
            record.update(
                {
                    "overlap_spots": int(matrices["overlap"][row, column]),
                    "cluster_spots": int(matrices["overlap"][row].sum()),
                    "truth_spots": int(matrices["overlap"][:, column].sum()),
                    "purity": float(matrices["purity"][row, column]),
                    "recall": float(matrices["recall"][row, column]),
                    "f1": float(matrices["f1"][row, column]),
                    "iou": float(matrices["iou"][row, column]),
                }
            )

        checks = {
            "has_positive_match": matched_truth is not None and record["overlap_spots"] > 0,
            "passes_min_overlap": record["overlap_spots"] >= int(selection["min_overlap_spots"]),
            "passes_min_cluster_size": record["cluster_spots"] >= int(selection["min_cluster_spots"]),
            "passes_min_truth_size": (
                pd.notna(record["truth_spots"])
                and record["truth_spots"] >= int(selection["min_truth_spots"])
            ),
            "passes_min_purity": record["purity"] >= float(selection["min_purity"]),
            "passes_min_recall": record["recall"] >= float(selection["min_recall"]),
            "passes_min_f1": record["f1"] >= float(selection["min_f1"]),
            "passes_min_iou": record["iou"] >= float(selection["min_iou"]),
            "passes_assignment_agreement": (
                assignment_agreement
                if bool(matching["require_agreement_for_selection"])
                else True
            ),
        }
        record.update(checks)
        record["passes_thresholds"] = all(checks.values())
        record["selection_failure_reasons"] = ";".join(
            name for name, passed in checks.items() if not passed
        )
        records.append(record)

    table = pd.DataFrame(records)
    table["selected_for_plot"] = False
    table["selection_rank"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
    table["selection_source"] = ""
    manually_included = {
        str(cluster) for cluster in selection.get("include_clusters", [])
    }
    available_clusters = set(table["biosfib_cluster"].astype(str))
    unknown_manual_clusters = sorted(
        manually_included.difference(available_clusters), key=natural_label_key
    )
    if unknown_manual_clusters:
        raise ValueError(
            "selection.include_clusters contains unknown BioSFIB clusters: "
            + ", ".join(unknown_manual_clusters)
        )
    unmatched_manual_clusters = sorted(
        table.loc[
            table["biosfib_cluster"].astype(str).isin(manually_included)
            & ~table["has_positive_match"],
            "biosfib_cluster",
        ].astype(str).tolist(),
        key=natural_label_key,
    )
    if unmatched_manual_clusters:
        raise ValueError(
            "selection.include_clusters cannot force clusters without a positive truth match: "
            + ", ".join(unmatched_manual_clusters)
        )
    table["manual_include_requested"] = table["biosfib_cluster"].astype(str).isin(
        manually_included
    )
    include_all_positive = bool(selection.get("include_all_positive_matches", False))
    table["all_positive_include_requested"] = (
        table["has_positive_match"] if include_all_positive else False
    )
    selection_mode = str(selection.get("mode", "threshold")).lower()
    table["selection_mode"] = selection_mode
    sort_by = str(selection["sort_by"])
    if sort_by not in table.columns:
        raise KeyError(f"selection.sort_by={sort_by!r} is not a score-table column.")
    if selection_mode == "threshold":
        primary_mask = table["passes_thresholds"]
    else:
        # top_n means the literal top N positive matches from the configured
        # primary Hungarian assignment. Threshold/agreement flags remain in
        # the audit CSV but do not silently reduce the requested panel count.
        primary_mask = table["has_positive_match"]
    candidates = table[primary_mask].sort_values(
        [sort_by, "biosfib_cluster"],
        ascending=[not bool(selection["descending"]), True],
        kind="stable",
    )
    if selection_mode == "top_n":
        candidates = candidates.head(int(selection["top_n"]))
    else:
        max_domains = selection.get("max_domains")
        if max_domains is not None:
            candidates = candidates.head(int(max_domains))
    primary_indices = set(candidates.index)
    selected_indices = set(primary_indices)
    selected_indices.update(
        table.index[table["manual_include_requested"]].tolist()
    )
    selected_indices.update(
        table.index[table["all_positive_include_requested"]].tolist()
    )
    selected_rows = table.loc[list(selected_indices)].sort_values(
        [sort_by, "biosfib_cluster"],
        ascending=[not bool(selection["descending"]), True],
        kind="stable",
    )
    for rank, index in enumerate(selected_rows.index, start=1):
        table.loc[index, "selected_for_plot"] = True
        table.loc[index, "selection_rank"] = rank
        if index in primary_indices:
            source = "threshold" if selection_mode == "threshold" else f"top_n_{sort_by}"
        elif bool(table.loc[index, "manual_include_requested"]):
            source = "manual_include"
        else:
            source = "all_positive_match"
        table.loc[index, "selection_source"] = source
    return table.sort_values("biosfib_cluster", key=lambda values: values.map(natural_label_key))


def transform_spatial_coordinates(coordinates: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    transformed = np.asarray(coordinates, dtype=float).copy()
    if transformed.ndim != 2 or transformed.shape[1] < 2:
        raise ValueError("Configured spatial coordinates must be an n_spots × 2 matrix.")
    transformed = transformed[:, :2]
    centre = transformed.mean(axis=0, keepdims=True)
    transformed -= centre
    transform = settings["transform"]
    if bool(transform["flip_horizontal"]):
        transformed[:, 0] *= -1
    if bool(transform["flip_vertical"]):
        transformed[:, 1] *= -1
    rotate_cw = int(transform["rotate_cw"])
    if rotate_cw:
        theta = -math.radians(rotate_cw)
        rotation = np.asarray(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        transformed = transformed @ rotation.T
    target_span = settings.get("display_target_span")
    if target_span is not None:
        current_span = np.ptp(transformed, axis=0)
        scale = np.divide(
            np.asarray(target_span, dtype=float),
            current_span,
            out=np.ones(2, dtype=float),
            where=current_span > 0,
        )
        transformed *= scale
    return transformed + centre


def relative_luminance(color: str | tuple[float, float, float]) -> float:
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=float)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    return float(np.dot(linear, np.asarray([0.2126, 0.7152, 0.0722])))


def contrast_ratio(color_a: str | tuple[float, float, float], color_b: str) -> float:
    luminance_a = relative_luminance(color_a)
    luminance_b = relative_luminance(color_b)
    lighter, darker = max(luminance_a, luminance_b), min(luminance_a, luminance_b)
    return (lighter + 0.05) / (darker + 0.05)


def ensure_background_contrast(color: str, background: str, minimum: float) -> str:
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=float)
    for _ in range(24):
        if contrast_ratio(tuple(rgb), background) >= minimum:
            break
        rgb *= 0.90
    return mpl.colors.to_hex(np.clip(rgb, 0.0, 1.0))


def load_cluster_colors(
    config_root: Path, selected: pd.DataFrame, spec: dict[str, Any]
) -> dict[str, str]:
    colors = spec["colors"]
    configured_mapping = colors.get("cluster_color_mapping_json")
    mapped: dict[str, str] = {}
    if configured_mapping:
        mapping_path = resolve_path(config_root, configured_mapping)
        if not mapping_path.exists():
            raise FileNotFoundError(f"Configured cluster color mapping does not exist: {mapping_path}")
        with mapping_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        mapped = {str(key): str(value) for key, value in payload["cluster_to_color"].items()}

    clusters = selected["biosfib_cluster"].astype(str).tolist()
    missing = [cluster for cluster in clusters if cluster not in mapped]
    if missing:
        palette_setting = colors["fallback_palette"]
        if isinstance(palette_setting, list):
            if len(palette_setting) < len(missing):
                raise ValueError("colors.fallback_palette contains fewer colors than required.")
            fallback = [mpl.colors.to_hex(value) for value in palette_setting]
        else:
            palette = plt.get_cmap(str(palette_setting), max(len(missing), 1))
            fallback = [mpl.colors.to_hex(palette(index)) for index in range(len(missing))]
        mapped.update({cluster: fallback[index] for index, cluster in enumerate(missing)})
    overrides = {
        str(cluster): mpl.colors.to_hex(color)
        for cluster, color in colors.get("cluster_color_overrides", {}).items()
    }
    mapped.update(overrides)
    minimum_contrast = float(colors.get("minimum_background_contrast", 0.0))
    if minimum_contrast > 1.0:
        background = str(colors["background"])
        for cluster in clusters:
            mapped[cluster] = ensure_background_contrast(
                mapped[cluster], background, minimum_contrast
            )
    return {cluster: mapped[cluster] for cluster in clusters}


def configure_matplotlib(spec: dict[str, Any]) -> None:
    plot = spec["plot"]
    requested_font = str(plot["font_family"])
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [requested_font, "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": float(plot["font_size"]),
            "axes.linewidth": float(plot["frame_linewidth"]),
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_domain_figure(
    adata: ad.AnnData,
    selected: pd.DataFrame,
    cluster_colors: dict[str, str],
    spec: dict[str, Any],
    output_path: Path,
) -> None:
    configure_matplotlib(spec)
    plot = spec["plot"]
    spatial = spec["spatial"]
    rows = [str(value).lower() for value in plot["panel_rows"]]
    selected = selected.sort_values("selection_rank")
    n_domain_rows, n_domains = len(rows), len(selected)
    if n_domains == 0:
        raise ValueError("No domains were selected by the configured selection rule.")
    configured_columns = plot.get("max_columns_per_block")
    columns_per_block = (
        n_domains
        if configured_columns is None
        else min(n_domains, int(configured_columns))
    )
    if columns_per_block <= 0:
        raise ValueError("plot.max_columns_per_block must be null or a positive integer.")
    n_blocks = math.ceil(n_domains / columns_per_block)
    n_figure_rows = n_domain_rows * n_blocks

    configured_size = plot.get("figure_size")
    if configured_size is None:
        width = max(
            float(plot["minimum_figure_width"]),
            float(plot["panel_width"]) * columns_per_block,
        )
        height = (
            float(plot["panel_height"]) * n_figure_rows
            + float(plot["extra_vertical_space"])
            + float(plot["block_extra_vertical_space"]) * (n_blocks - 1)
        )
        figure_size = (width, height)
    else:
        if len(configured_size) != 2:
            raise ValueError("plot.figure_size must be null or [width, height].")
        figure_size = tuple(float(value) for value in configured_size)

    coordinates = transform_spatial_coordinates(
        np.asarray(adata.obsm[spec["spatial_key"]]), spatial
    )
    x, y = coordinates[:, 0], coordinates[:, 1]
    x_span = max(float(np.ptp(x)), 1.0)
    y_span = max(float(np.ptp(y)), 1.0)
    pad_fraction = float(spatial["coordinate_padding_fraction"])
    square_span = max(x_span, y_span) * (1.0 + 2.0 * pad_fraction)
    x_centre = float((x.min() + x.max()) / 2.0)
    y_centre = float((y.min() + y.max()) / 2.0)
    half_span = square_span / 2.0
    x_limits = (x_centre - half_span, x_centre + half_span)
    y_limits = (y_centre - half_span, y_centre + half_span)

    cluster_values = adata.obs[spec["cluster_key"]].astype("string").to_numpy()
    truth_values = adata.obs[spec["truth_key"]].astype("string").to_numpy()
    fig, axes = plt.subplots(
        n_figure_rows,
        columns_per_block,
        figsize=figure_size,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=float(plot["margin_left"]),
        right=float(plot["margin_right"]),
        top=float(plot["margin_top"]),
        bottom=float(plot["margin_bottom"]),
        wspace=float(plot["column_spacing"]),
        hspace=float(plot["row_spacing"]),
    )

    row_labels = plot["row_labels"]
    error_colors = spec["colors"]["agreement"]
    for empty_domain_index in range(n_domains, n_blocks * columns_per_block):
        block = empty_domain_index // columns_per_block
        column = empty_domain_index % columns_per_block
        for local_row_index in range(n_domain_rows):
            axes[block * n_domain_rows + local_row_index, column].set_visible(False)

    for domain_index, record in enumerate(selected.itertuples(index=False)):
        block = domain_index // columns_per_block
        column = domain_index % columns_per_block
        cluster = str(record.biosfib_cluster)
        truth_label = str(record.matched_truth_label)
        truth_mask = truth_values == truth_label
        prediction_mask = cluster_values == cluster
        true_positive = truth_mask & prediction_mask
        false_positive = ~truth_mask & prediction_mask
        false_negative = truth_mask & ~prediction_mask
        domain_color = cluster_colors[cluster]

        for local_row_index, row_kind in enumerate(rows):
            row_index = block * n_domain_rows + local_row_index
            ax = axes[row_index, column]
            ax.scatter(
                x,
                y,
                s=float(plot["background_point_size"]),
                color=str(spec["colors"]["background"]),
                alpha=float(plot["background_alpha"]),
                linewidths=0,
                marker=str(plot["marker"]),
                zorder=1,
            )
            if row_kind == "truth":
                masks_and_colors = [(truth_mask, domain_color)]
            elif row_kind == "prediction":
                masks_and_colors = [(prediction_mask, domain_color)]
            else:
                masks_and_colors = [
                    (false_negative, str(error_colors["false_negative"])),
                    (false_positive, str(error_colors["false_positive"])),
                    (true_positive, str(error_colors["true_positive"])),
                ]
            for mask, color in masks_and_colors:
                if np.any(mask):
                    ax.scatter(
                        x[mask],
                        y[mask],
                        s=float(plot["foreground_point_size"]),
                        color=color,
                        alpha=float(plot["foreground_alpha"]),
                        linewidths=0,
                        marker=str(plot["marker"]),
                        zorder=2,
                    )
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            if bool(spatial["invert_y"]):
                ax.invert_yaxis()
            ax.set_box_aspect(1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(str(plot["panel_background"]))
            for spine in ax.spines.values():
                spine.set_visible(bool(plot["show_frame"]))
                spine.set_color(str(plot["frame_color"]))
                spine.set_linewidth(float(plot["frame_linewidth"]))
            if local_row_index == 0:
                title = str(record.display_truth_label)
                wrap_width = plot.get("title_wrap_width")
                if wrap_width is not None and int(wrap_width) > 0:
                    title = "\n".join(
                        textwrap.wrap(
                            title,
                            width=int(wrap_width),
                            break_long_words=False,
                            break_on_hyphens=False,
                        )
                    )
                details: list[str] = []
                if bool(plot["show_cluster_id_in_title"]):
                    details.append(f"C{cluster}")
                if bool(plot["show_scores_in_title"]):
                    decimals = int(plot["score_decimals"])
                    details.extend(
                        [f"F1={record.f1:.{decimals}f}", f"IoU={record.iou:.{decimals}f}"]
                    )
                if details:
                    title += "\n" + " | ".join(details)
                ax.set_title(
                    title,
                    fontsize=float(plot["title_font_size"]),
                    pad=float(plot["title_pad"]),
                    linespacing=float(plot["title_line_spacing"]),
                )
            if column == 0:
                ax.text(
                    float(plot["row_label_x"]),
                    0.5,
                    str(row_labels[row_kind]),
                    transform=ax.transAxes,
                    rotation=float(plot["row_label_rotation"]),
                    ha="center",
                    va="center",
                    fontsize=float(plot["row_label_font_size"]),
                    fontweight="bold",
                    clip_on=False,
                )

    if "agreement" in rows and bool(plot["show_agreement_legend"]):
        handles = [
            Line2D(
                [0], [0], marker=str(plot["marker"]), linestyle="none",
                markerfacecolor=str(error_colors[key]), markeredgewidth=0,
                markersize=float(plot["legend_marker_size"]), label=str(label),
            )
            for key, label in (
                ("true_positive", plot["agreement_legend_labels"]["true_positive"]),
                ("false_positive", plot["agreement_legend_labels"]["false_positive"]),
                ("false_negative", plot["agreement_legend_labels"]["false_negative"]),
            )
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, float(plot["legend_y"])),
            ncol=3,
            frameon=False,
            fontsize=float(plot["legend_font_size"]),
            handletextpad=float(plot["legend_handletextpad"]),
            columnspacing=float(plot["legend_columnspacing"]),
        )

    fig.savefig(
        output_path,
        format="pdf",
        dpi=int(plot["dpi"]),
        bbox_inches="tight",
        pad_inches=float(plot["save_pad_inches"]),
    )
    plt.close(fig)


def output_paths(config_root: Path, spec: dict[str, Any]) -> dict[str, Path]:
    output_dir = resolve_path(config_root, spec["output_dir"])
    outputs = spec["outputs"]
    return {
        "scores": output_dir / str(outputs["score_csv"]),
        "contingency": output_dir / str(outputs["contingency_csv"]),
        "figure": output_dir / str(outputs["figure_pdf"]),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_yaml(config_path)
    analyses = config.get("analyses", {})
    if args.list:
        for name in analyses:
            print(name)
        return

    analysis_name, spec = resolve_analysis(config, args.analysis)
    config_root = PROJECT_ROOT
    input_path = resolve_path(config_root, spec["input_h5ad"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input H5AD does not exist: {input_path}")
    print(f"[domains] Analysis: {analysis_name}")
    print(f"[domains] Input: {input_path}")
    adata = ad.read_h5ad(input_path)
    for key in (spec["cluster_key"], spec["truth_key"]):
        if key not in adata.obs:
            raise KeyError(f"obs[{key!r}] is absent. Available: {', '.join(map(str, adata.obs.columns))}")
    if spec["spatial_key"] not in adata.obsm:
        raise KeyError(f"obsm[{spec['spatial_key']!r}] is absent. Available: {', '.join(adata.obsm.keys())}")

    contingency, valid = build_contingency(
        adata.obs[spec["cluster_key"]],
        adata.obs[spec["truth_key"]],
        spec["matching"]["ignored_cluster_values"],
        spec["matching"]["ignored_truth_values"],
    )
    matrices = score_matrices(contingency)
    scores = build_score_table(contingency, matrices, spec)
    selected = scores[scores["selected_for_plot"]].copy().sort_values("selection_rank")
    print(f"[domains] Valid spots: {int(valid.sum())}/{len(valid)}")
    print(f"[domains] Positive matches: {int(scores['has_positive_match'].sum())}/{len(scores)}")
    print(f"[domains] Selected domains: {len(selected)}")
    if selected.empty:
        best_columns = [
            "biosfib_cluster", "display_truth_label", "overlap_spots",
            "purity", "recall", "f1", "iou", "selection_failure_reasons",
        ]
        print(scores.sort_values("f1", ascending=False)[best_columns].head(10).to_string(index=False))
        raise ValueError("No domains were selected by the configured selection rule.")
    print(
        selected[
            [
                "selection_rank", "selection_source", "biosfib_cluster",
                "display_truth_label", "overlap_spots", "purity", "recall", "f1", "iou",
            ]
        ].to_string(index=False)
    )
    if args.dry_run:
        print("[domains] Dry run successful; no files were written.")
        return

    paths = output_paths(config_root, spec)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; use --overwrite to replace them: "
            + ", ".join(map(str, existing))
        )
    next(iter(paths.values())).parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(paths["scores"], index=False)
    contingency.reset_index().to_csv(paths["contingency"], index=False)
    cluster_colors = load_cluster_colors(config_root, selected, spec)
    draw_domain_figure(adata, selected, cluster_colors, spec, paths["figure"])
    print(f"[domains] Scores: {paths['scores']}")
    print(f"[domains] Contingency: {paths['contingency']}")
    print(f"[domains] Figure: {paths['figure']}")


if __name__ == "__main__":
    main()
