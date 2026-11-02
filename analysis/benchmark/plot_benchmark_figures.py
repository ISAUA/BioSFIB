"""YAML-driven benchmark figures (bubble matrices and slice-level boxplots).

Examples
--------
python plot_benchmark_figures.py --list
python plot_benchmark_figures.py --figure misar_e18_5_s1_bubble
python plot_benchmark_figures.py --figure misar_pca25_boxplot

Every figure definition lives in ``benchmark_config.yaml`` under
``figures.definitions``.  The script only reads the benchmark CSV and writes
PDF exports; it never creates PNGs, SVGs, or extra data tables.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LogLocator, MaxNLocator, NullLocator
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create configured benchmark figures.")
    parser.add_argument("--config", type=Path, default=Path("analysis/benchmark/benchmark_config.yaml"))
    parser.add_argument("--figure", help="Name under figures.definitions in the YAML file.")
    parser.add_argument("--list", action="store_true", help="List available figure names and exit.")
    parser.add_argument(
        "--legend-only",
        choices=("truth", "cluster", "all"),
        help="For spatial_umap_maps, write only the requested standalone legend(s); do not draw maps.",
    )
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


def resolve_figure(config: dict[str, Any], name: str) -> dict[str, Any]:
    figures = config.get("figures", {})
    definitions = figures.get("definitions", {})
    if name not in definitions:
        raise KeyError(f"Unknown figure {name!r}. Available: {', '.join(definitions)}")
    definition = definitions[name]
    type_defaults = figures.get(f"{definition.get('type', '')}_defaults", {})
    spec = deep_merge(figures.get("defaults", {}), type_defaults)
    spec = deep_merge(spec, definition)
    required = ["type", "dataset", "evaluation_id", "output_stem"]
    if spec.get("type") in {"bubble_matrix", "boxplot"}:
        required.append("metrics")
    missing = [key for key in required if not spec.get(key)]
    if missing:
        raise KeyError(f"Figure {name!r} is missing YAML fields: {missing}")
    return spec


def configure_matplotlib(font_size: float) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": font_size,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def input_csv(config_root: Path, spec: dict[str, Any]) -> Path:
    path = config_root / spec["input_csv"]
    if not path.exists():
        raise FileNotFoundError(f"Benchmark CSV does not exist: {path}")
    return path


def output_stem(config_root: Path, spec: dict[str, Any]) -> Path:
    output_dir = config_root / spec["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / safe_stem(spec["output_stem"])


def valid_selected_rows(long: pd.DataFrame, spec: dict[str, Any], require_slice: bool) -> pd.DataFrame:
    required_columns = {
        "dataset", "evaluation_id", "method", "metric_group", "metric_name", "metric_value", "metric_status"
    }
    missing = sorted(required_columns.difference(long.columns))
    if missing:
        raise ValueError(f"benchmark_metrics.csv is missing required columns: {missing}")
    metric_groups = spec.get("metric_groups")
    if metric_groups is not None:
        if not metric_groups:
            raise ValueError("metric_groups must contain at least one metric-group name.")
        group_mask = long["metric_group"].isin([str(group) for group in metric_groups])
    else:
        group_mask = long["metric_group"] == spec.get("metric_group", "external")
    mask = (
        (long["dataset"] == spec["dataset"])
        & group_mask
        & long["method"].isin(spec["method_order"])
        & long["metric_name"].isin(spec["metrics"])
    )
    overrides = spec.get("method_evaluation_overrides", {})
    evaluation_mask = pd.Series(False, index=long.index)
    for method in spec["method_order"]:
        override = overrides.get(method, {})
        evaluation_id = override.get("evaluation_id_by_dataset", {}).get(
            spec["dataset"], override.get("evaluation_id", spec["evaluation_id"])
        )
        evaluation_role = override.get("evaluation_role_by_dataset", {}).get(
            spec["dataset"], override.get("evaluation_role", spec.get("evaluation_role"))
        )
        method_mask = (long["method"] == method) & (long["evaluation_id"] == evaluation_id)
        if evaluation_role is not None and "evaluation_role" in long.columns:
            method_mask &= long["evaluation_role"] == evaluation_role
        evaluation_mask |= method_mask
    mask &= evaluation_mask
    if require_slice:
        if "slice" not in spec:
            raise KeyError("bubble_matrix requires a YAML 'slice' field.")
        mask &= long["slice"] == spec["slice"]
    selected = long.loc[mask].copy()
    configured_slices = spec.get("slices")
    if configured_slices is not None:
        requested = {str(value) for value in configured_slices}
        available = set(selected["slice"].astype(str))
        missing_slices = sorted(requested.difference(available))
        if missing_slices:
            raise ValueError(
                "Configured slices are absent after dataset/condition filtering: "
                + ", ".join(missing_slices)
            )
        selected = selected[selected["slice"].astype(str).isin(requested)].copy()
    if selected.empty:
        raise ValueError("No benchmark rows match this figure definition.")
    failed = selected[selected["metric_status"] != "ok"]
    if not failed.empty:
        columns = ["slice", "method", "metric_name", "metric_status"]
        raise ValueError(f"Selected rows include non-ok metrics:\n{failed[columns].to_string(index=False)}")
    selected["metric_value"] = pd.to_numeric(selected["metric_value"], errors="coerce")
    if selected["metric_value"].isna().any():
        raise ValueError("Selected rows contain non-numeric metric values.")
    return selected


def truncated_colormap(name: str, low: float, high: float) -> LinearSegmentedColormap:
    if not 0 <= low < high <= 1:
        raise ValueError("cmap_low/cmap_high must satisfy 0 <= low < high <= 1.")
    values = plt.get_cmap(name)(np.linspace(low, high, 256))
    return LinearSegmentedColormap.from_list(f"{name}_{low:.2f}_{high:.2f}", values)


def contrast_text_color(rgba: tuple[float, float, float, float], dark_text: str) -> str:
    channels = np.asarray(rgba[:3], dtype=float)
    linear = np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    luminance = float(np.dot(linear, [0.2126, 0.7152, 0.0722]))
    return "white" if 1.05 / (luminance + 0.05) > (luminance + 0.05) / 0.05 else dark_text


def color_limits(values: np.ndarray, scale: dict[str, Any]) -> tuple[float, float]:
    """Resolve one bubble figure's shared color range from its displayed values."""
    mode = str(scale.get("mode", "fixed")).lower()
    if mode == "fixed":
        return float(scale["vmin"]), float(scale["vmax"])
    if mode != "auto":
        raise ValueError("color_scale.mode must be 'auto' or 'fixed'.")

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot derive automatic color limits from empty/non-finite values.")
    observed_min, observed_max = float(finite.min()), float(finite.max())
    padding = max(
        (observed_max - observed_min) * float(scale.get("padding_fraction", 0.08)),
        float(scale.get("minimum_padding", 0.01)),
    )
    vmin, vmax = max(0.0, observed_min - padding), min(1.0, observed_max + padding)
    minimum_span = float(scale.get("minimum_span", 0.05))
    if vmax - vmin < minimum_span:
        midpoint = (observed_min + observed_max) / 2.0
        vmin, vmax = max(0.0, midpoint - minimum_span / 2.0), min(1.0, midpoint + minimum_span / 2.0)
        if vmax - vmin < minimum_span:
            vmin, vmax = max(0.0, vmax - minimum_span), min(1.0, vmin + minimum_span)
    round_step = float(scale.get("round_step", 0.0))
    if round_step > 0:
        vmin = max(0.0, math.floor(vmin / round_step) * round_step)
        vmax = min(1.0, math.ceil(vmax / round_step) * round_step)
        if vmax <= vmin:
            vmax = min(1.0, vmin + round_step)
    return vmin, vmax


def bubble_matrix(long: pd.DataFrame, spec: dict[str, Any]) -> plt.Figure:
    selected = valid_selected_rows(long, spec, require_slice=True)
    duplicates = selected.duplicated(["method", "metric_name"], keep=False)
    if duplicates.any():
        raise ValueError("bubble_matrix has duplicate method-metric rows for the selected slice.")
    matrix = selected.pivot(index="method", columns="metric_name", values="metric_value")
    methods, metrics = spec["method_order"], spec["metrics"]
    missing_methods = [value for value in methods if value not in matrix.index]
    missing_metrics = [value for value in metrics if value not in matrix.columns]
    if missing_methods or missing_metrics:
        raise ValueError(f"Incomplete bubble matrix. Missing methods={missing_methods}; metrics={missing_metrics}")
    matrix = matrix.loc[methods, metrics]
    matrix[spec.get("aggregate_metric_name", "External mean")] = matrix.mean(axis=1)

    configure_matplotlib(float(spec.get("font_size", 9)))
    style = spec.get("style", {})
    scale = spec["color_scale"]
    cmap = truncated_colormap(scale["cmap"], float(scale["cmap_low"]), float(scale["cmap_high"]))
    vmin, vmax = color_limits(matrix.to_numpy(dtype=float), scale)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    n_methods, n_metrics = len(methods), len(metrics)
    aggregate_x = n_metrics + 1.05
    requested_size = tuple(spec.get("figure_size", [9, 3.65]))
    figure_size = (requested_size[0], max(requested_size[1], 1.95 + 0.42 * n_methods))
    fig, ax = plt.subplots(figsize=figure_size)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.18)
    ax.set_xlim(-3.25, aggregate_x + 1.20)
    ax.set_ylim(n_methods + 0.30, -1.72)
    ax.axis("off")

    for row in range(n_methods):
        if row % 2:
            ax.add_patch(Rectangle((-3.10, row - 0.48), aggregate_x + 4.0, 0.96, facecolor=style.get("row_band_color", "#F1F5F2"), edgecolor="none", zorder=0))
        if row:
            ax.hlines(row - 0.50, -3.10, aggregate_x + 0.95, color=style.get("separator_color", "#718078"), linestyle=(0, (1.2, 3.4)), linewidth=0.95, zorder=1)
    rule_color = style.get("rule_color", "#222222")
    ax.hlines(-0.58, -3.10, aggregate_x + 0.95, color=rule_color, linewidth=1.1)
    ax.hlines(n_methods - 0.50, -3.10, aggregate_x + 0.95, color=rule_color, linewidth=1.1)
    ax.vlines(n_metrics + 0.45, -1.48, n_methods - 0.50, color=rule_color, linewidth=0.9)
    header_bbox = {"facecolor": "white", "edgecolor": "none", "pad": 0.7}
    labels = spec["metric_labels"]
    ax.text(-3.0, -0.94, "Method", fontweight="bold", va="center", zorder=10)
    ax.text((n_metrics - 1) / 2, -1.46, spec.get("metric_group_title", "Biological conservation"), ha="center", va="center", bbox=header_bbox, zorder=10)
    ax.text(aggregate_x, -1.46, spec.get("aggregate_title", "Aggregate score"), ha="center", va="center", bbox=header_bbox, zorder=10)
    ax.hlines(-1.27, -0.45, n_metrics - 0.55, color=rule_color, linewidth=0.78)
    ax.hlines(-1.27, aggregate_x - 0.48, aggregate_x + 0.48, color=rule_color, linewidth=0.78)
    for column, metric in enumerate(metrics):
        ax.text(column, -0.94, labels.get(metric, metric), ha="center", va="center", fontsize=8.35, bbox=header_bbox, zorder=10)
    ax.text(aggregate_x, -0.94, spec.get("aggregate_label", "Mean"), ha="center", va="center", fontsize=8.35, bbox=header_bbox, zorder=10)

    dark_text = style.get("dark_text", "#17231F")
    highlight_method = str(spec.get("highlight_method", "BioSFIB"))
    for row, method in enumerate(methods):
        ax.text(-3.0, row, method, ha="left", va="center", fontweight="bold" if method == highlight_method else "normal", zorder=5)
        for column, metric in enumerate(metrics):
            value = float(matrix.loc[method, metric])
            rgba = cmap(norm(value))
            ax.scatter(column, row, s=float(spec.get("dot_size", 650)), color=rgba, edgecolor="#F7FAF8", linewidth=0.75, zorder=3)
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=7.65, color=contrast_text_color(rgba, dark_text), zorder=4)
        aggregate_name = spec.get("aggregate_metric_name", "External mean")
        mean_value = float(matrix.loc[method, aggregate_name])
        rgba = cmap(norm(mean_value))
        ax.add_patch(Rectangle((aggregate_x - 0.43, row - 0.22), 0.86, 0.44, facecolor=rgba, edgecolor="#F7FAF8", linewidth=0.75, zorder=3))
        ax.text(aggregate_x, row, f"{mean_value:.3f}", ha="center", va="center", fontsize=7.65, color=contrast_text_color(rgba, dark_text), zorder=4)

    if spec.get("show_panel_title", False):
        panel_title = spec.get("panel_title") or f"{spec['dataset']} | {spec['slice']} | {spec['evaluation_id']}"
        ax.text(-3.0, -1.72, panel_title, va="center", fontweight="bold")
    color_axis = fig.add_axes([0.815, 0.13, 0.145, 0.028])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis, orientation="horizontal")
    colorbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    colorbar.ax.tick_params(
        labelsize=float(spec.get("colorbar_tick_font_size", 8.5)),
        length=2,
        pad=1,
    )
    colorbar.set_label(
        spec.get("colorbar_label", "Metric value"),
        fontsize=float(spec.get("colorbar_label_font_size", 9.0)),
        labelpad=2,
    )
    colorbar.outline.set_linewidth(0.75)
    return fig


def development_slice_key(name: str) -> tuple[float, int, str]:
    stage = re.search(r"(?i)e(\d{1,2})(?:[._-](\d+))?", str(name))
    section = re.search(r"(?i)s[_-]?(\d+)", str(name))
    stage_value = float("inf") if stage is None else float(f"{stage.group(1)}.{stage.group(2) or '0'}")
    section_value = 999 if section is None else int(section.group(1))
    return stage_value, section_value, str(name)


def panel_range(values: pd.Series, axis: dict[str, Any]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    low, high = float(values.min()), float(values.max())
    if axis.get("scale", "linear") == "log":
        if low <= 0:
            raise ValueError("A logarithmic metric axis requires strictly positive values.")
        factor = float(axis.get("log_padding_factor", 1.3))
        if factor <= 1:
            raise ValueError("log_padding_factor must exceed 1.")
        return low / factor, high * factor
    padding = max(float(axis.get("padding", 0.025)), (high - low) * 0.08)
    step = float(axis.get("round_step", 0.05))
    low, high = math.floor((low - padding) / step) * step, math.ceil((high + padding) / step) * step
    min_width = float(axis.get("min_width", 0.20))
    if high - low < min_width:
        middle = (low + high) / 2
        low, high = middle - min_width / 2, middle + min_width / 2
    low, high = math.floor(low / step) * step, math.ceil(high / step) * step
    lower_bound, upper_bound = axis.get("bounds", [0.0, 1.0])
    if lower_bound is not None:
        low = max(float(lower_bound), low)
    if upper_bound is not None:
        high = min(float(upper_bound), high)
    if low >= high or np.any(values < low) or np.any(values > high):
        raise ValueError("Metric axis bounds exclude data; configure bounds for this metric.")
    return low, high


def lighten(color: str, amount: float = 0.08) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * amount)


def boxplot_matrix(long: pd.DataFrame, spec: dict[str, Any]) -> plt.Figure:
    selected = valid_selected_rows(long, spec, require_slice=False)
    slice_level = selected.groupby(["method", "metric_name", "slice"], as_index=False, observed=True)["metric_value"].mean().rename(columns={"metric_value": "value"})
    methods, metrics = spec["method_order"], spec["metrics"]
    expected = spec.get("expected_n_slices")
    reference_slices: set[str] | None = None
    for method in methods:
        for metric in metrics:
            group = slice_level[(slice_level["method"] == method) & (slice_level["metric_name"] == metric)]
            if group.empty:
                raise ValueError(f"Missing boxplot data for method={method}, metric={metric}.")
            current_slices = set(group["slice"].astype(str))
            if expected is not None and len(current_slices) != int(expected):
                raise ValueError(f"Expected {expected} slices for {method}/{metric}, found {len(current_slices)}.")
            if reference_slices is None:
                reference_slices = current_slices
            elif reference_slices != current_slices:
                raise ValueError("Methods/metrics do not share the same set of slices.")

    configure_matplotlib(float(spec.get("font_size", 8.5)))
    style, axis_config = spec.get("style", {}), spec.get("axis", {})
    requested_size = tuple(spec.get("figure_size", [10.8, 3.35]))
    figure_size = (requested_size[0], max(requested_size[1], 1.9 + 0.38 * len(methods)))
    fig, axes = plt.subplots(1, len(metrics), figsize=figure_size, sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=float(spec.get("left_margin", 0.135)), right=0.995, top=0.88, bottom=0.14, wspace=float(spec.get("panel_spacing", 0.08)))
    y_positions = np.arange(len(methods))
    for panel, metric in enumerate(metrics):
        ax = axes[panel]
        metric_axis = deep_merge(axis_config, axis_config.get("by_metric", {}).get(metric, {}))
        metric_data = slice_level[slice_level["metric_name"] == metric]
        x_min, x_max = panel_range(metric_data["value"], metric_axis)
        ax.set_facecolor(style.get("panel_background_1", "#FFFFFF") if panel % 2 == 0 else style.get("panel_background_2", "#F4F4F4"))
        for y, method in enumerate(methods):
            values = metric_data.loc[metric_data["method"] == method, "value"].to_numpy(dtype=float)
            base_color = spec["method_colors"][method]
            ax.boxplot([values], positions=[y], vert=False, widths=float(spec.get("box_width", 0.48)), patch_artist=True, showfliers=False, manage_ticks=False,
                       boxprops={"facecolor": lighten(base_color), "edgecolor": style.get("box_edge_color", "#4A4A4A"), "linewidth": 0.90, "alpha": float(spec.get("box_alpha", 0.86))},
                       medianprops={"color": style.get("median_color", "#222222"), "linewidth": 1.20},
                       whiskerprops={"color": style.get("box_edge_color", "#4A4A4A"), "linewidth": 0.85},
                       capprops={"color": style.get("box_edge_color", "#4A4A4A"), "linewidth": 0.85})
            if bool(spec.get("show_points", False)):
                offsets = np.linspace(-0.10, 0.10, len(values))
                ax.scatter(values, y + offsets, s=float(spec.get("point_size", 10)), c=base_color,
                           edgecolors=style.get("box_edge_color", "#4A4A4A"), linewidths=0.4, zorder=4)
        ax.set_title(spec["metric_labels"].get(metric, metric), fontsize=9.5, pad=6)
        scale = str(metric_axis.get("scale", "linear"))
        ax.set_xscale(scale)
        ax.set_xlim(x_min, x_max)
        if scale == "log":
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=5))
            ax.xaxis.set_minor_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        tick_format = str(metric_axis.get("tick_format", "%.2f"))
        hidden_tick_labels = [
            float(value)
            for value in metric_axis.get("hide_tick_labels", [])
        ]

        if hidden_tick_labels:
            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, position, hidden_values=hidden_tick_labels, fmt=tick_format: (
                        ""
                        if any(np.isclose(value, hidden) for hidden in hidden_values)
                        else fmt % value
                    )
                )
            )
        else:
            ax.xaxis.set_major_formatter(FormatStrFormatter(tick_format))
        ax.grid(axis="x", color=style.get("grid_color", "#DEDEDE"), linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_color("#BDBDBD")
        ax.spines["top"].set_linewidth(0.65)
        ax.spines["bottom"].set_color("#555555")
        ax.spines["bottom"].set_linewidth(0.70)
        ax.tick_params(axis="x", labelsize=7.5, length=2.5, width=0.65, pad=2)
        ax.tick_params(axis="y", length=0, labelsize=8.5, pad=6)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(methods)
    axes[0].set_ylim(len(methods) - 0.5, -0.5)
    highlight_method = str(spec.get("highlight_method", "BioSFIB"))
    for label in axes[0].get_yticklabels():
        if label.get_text() == highlight_method:
            label.set_fontweight("bold")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    return fig


# -----------------------------------------------------------------------------
# Spatial and UMAP comparison maps
# -----------------------------------------------------------------------------

def valid_category_mask(values: pd.Series) -> np.ndarray:
    text = values.astype("string")
    return np.asarray(text.notna() & text.str.strip().ne("") & text.str.lower().ne("nan"), dtype=bool)


def prediction_path(config_root: Path, spec: dict[str, Any], method: str) -> Path:
    prediction_root = spec.get("prediction_dirs_by_method", {}).get(method, spec["prediction_dir"])
    evaluation_id = spec.get("prediction_evaluation_ids_by_method", {}).get(method, spec["evaluation_id"])
    path = (
        config_root
        / prediction_root
        / evaluation_id
        / spec["dataset"]
        / spec["slice"]
        / f"{method}_benchmark.h5ad"
    )
    if not path.exists():
        raise FileNotFoundError(f"Prediction H5AD not found: {path}")
    return path


def load_prediction_adatas(config_root: Path, spec: dict[str, Any]) -> dict[str, ad.AnnData]:
    import anndata as ad

    adatas: dict[str, ad.AnnData] = {}
    reference_names = None
    reference_spatial = None
    for method in spec["method_order"]:
        item = ad.read_h5ad(prediction_path(config_root, spec, method))
        required_obs = {"mclust_benchmark"}
        missing_obs = required_obs.difference(item.obs.columns)
        if missing_obs:
            raise KeyError(f"{method}: missing obs columns {sorted(missing_obs)}")
        source_key = spec.get("umap_source_keys_by_method", {}).get(method, spec["umap"]["source_key"])
        if "spatial" not in item.obsm or source_key not in item.obsm:
            raise KeyError(f"{method}: prediction file lacks spatial or configured UMAP embedding {source_key!r}.")
        if reference_names is None:
            reference_names = item.obs_names
            reference_spatial = np.asarray(item.obsm["spatial"])
        else:
            if not item.obs_names.equals(reference_names):
                raise ValueError(f"{method}: spot/cell order differs from canonical method.")
            if not np.array_equal(np.asarray(item.obsm["spatial"]), reference_spatial):
                raise ValueError(f"{method}: spatial coordinates differ from canonical method.")
        adatas[method] = item
    return adatas


def category_order(values: pd.Series, valid: np.ndarray) -> list[str]:
    labels = values.astype("string")[valid].unique().tolist()
    return sorted(labels, key=natural_label_key)


def natural_label_key(value: str) -> tuple[tuple[int, Any], ...]:
    """Sort numeric cluster labels as 1, 2, 3 ... 10, rather than lexically."""
    parts = re.split(r"(\d+)", str(value))
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)


def make_category_colors(labels: list[str], palette_name: str | list[str]) -> dict[str, str]:
    if isinstance(palette_name, list):
        if len(palette_name) < len(labels):
            raise ValueError("Configured palette contains fewer colors than displayed labels.")
        return {label: mpl.colors.to_hex(palette_name[index]) for index, label in enumerate(labels)}
    palette = plt.get_cmap(palette_name, max(len(labels), 1))
    return {label: mpl.colors.to_hex(palette(index)) for index, label in enumerate(labels)}


def make_truth_colors(truth: pd.Series, valid: np.ndarray, palette_name: str | list[str]) -> dict[str, str]:
    return make_category_colors(category_order(truth, valid), palette_name)


def rename_legend_colors(colors: dict[str, str], aliases: dict[str, str]) -> dict[str, str]:
    """Replace legend text while preserving the original label-to-color assignment."""
    normalized_aliases = {str(label): str(display) for label, display in aliases.items()}
    renamed: dict[str, str] = {}
    for original_label, color in colors.items():
        display_label = normalized_aliases.get(str(original_label), str(original_label))
        if display_label in renamed:
            raise ValueError(f"Legend display label {display_label!r} is assigned more than once.")
        renamed[display_label] = color
    return renamed


def raw_cluster_colors(
    adatas: dict[str, ad.AnnData], methods: list[str], palette_name: str | list[str]
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Assign one shared color to each raw mclust label across all displayed methods."""
    labels = sorted(
        {label for method in methods for label in adatas[method].obs["mclust_benchmark"].astype("string").unique()},
        key=natural_label_key,
    )
    shared_colors = make_category_colors(labels, palette_name)
    per_method = {
        method: {label: shared_colors[label] for label in adatas[method].obs["mclust_benchmark"].astype("string").unique()}
        for method in methods
    }
    return per_method, shared_colors


def map_prediction_colors(
    truth: pd.Series,
    prediction: pd.Series,
    truth_valid: np.ndarray,
    truth_colors: dict[str, str],
    unmatched_palette: str,
    unmatched_color: str,
) -> tuple[dict[str, str], dict[str, str | None]]:
    """Map predicted clusters to truth colors using global maximum-overlap matching."""
    truth_labels = truth.astype("string").to_numpy()
    predicted_labels = prediction.astype("string").to_numpy()
    pred_categories = sorted(pd.unique(predicted_labels).tolist(), key=natural_label_key)
    truth_categories = list(truth_colors)
    contingency = pd.crosstab(
        pd.Categorical(predicted_labels[truth_valid], categories=pred_categories),
        pd.Categorical(truth_labels[truth_valid], categories=truth_categories),
        dropna=False,
    ).reindex(index=pred_categories, columns=truth_categories, fill_value=0)
    row_index, col_index = linear_sum_assignment(contingency.to_numpy(), maximize=True)
    matched_truth: dict[str, str | None] = {label: None for label in pred_categories}
    color_map: dict[str, str] = {}
    for row, col in zip(row_index, col_index):
        predicted, target = pred_categories[row], truth_categories[col]
        matched_truth[predicted] = target
        color_map[predicted] = truth_colors[target]
    fallback = plt.get_cmap(unmatched_palette, max(len(pred_categories), 1))
    fallback_index = 0
    for predicted in pred_categories:
        if predicted not in color_map:
            color_map[predicted] = mpl.colors.to_hex(fallback(fallback_index)) if fallback_index < fallback.N else unmatched_color
            fallback_index += 1
    return color_map, matched_truth


def colors_from_reference_slice(
    config_root: Path,
    spec: dict[str, Any],
    adatas: dict[str, ad.AnnData],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Reuse each method's cluster-to-color mapping from a truth-annotated reference slice."""
    reference = spec.get("color_reference")
    if not reference:
        raise ValueError("A no-truth map requires YAML 'color_reference' settings.")
    reference_spec = copy.deepcopy(spec)
    for key in ("dataset", "slice", "evaluation_id", "evaluation_role"):
        if key in reference:
            reference_spec[key] = reference[key]
    reference_adatas = load_prediction_adatas(config_root, reference_spec)
    canonical = reference_adatas[spec["method_order"][0]]
    if "reference_label" not in canonical.obs:
        raise KeyError("Color reference slice has no reference_label column.")
    reference_truth = canonical.obs["reference_label"]
    reference_valid = valid_category_mask(reference_truth)
    if not reference_valid.any():
        raise ValueError("Color reference slice has no valid reference labels.")
    reference_truth_colors = make_truth_colors(
        reference_truth, reference_valid, spec["colors"]["truth_palette"]
    )

    predicted_colors: dict[str, dict[str, str]] = {}
    records: dict[str, Any] = {}
    for method in spec["method_order"]:
        reference_colors, reference_matches = map_prediction_colors(
            reference_truth,
            reference_adatas[method].obs["mclust_benchmark"],
            reference_valid,
            reference_truth_colors,
            spec["colors"]["unmatched_palette"],
            spec["colors"]["unmatched_color"],
        )
        current_labels = sorted(
            pd.unique(adatas[method].obs["mclust_benchmark"].astype("string")).tolist(),
            key=natural_label_key,
        )
        missing_labels = [label for label in current_labels if label not in reference_colors]
        if missing_labels:
            raise ValueError(
                f"{method}: clusters {missing_labels} are absent from color reference "
                f"{reference_spec['dataset']}/{reference_spec['slice']}."
            )
        color_map = {label: reference_colors[label] for label in current_labels}
        predicted_colors[method] = color_map
        records[method] = {
            "cluster_to_truth_label": {label: reference_matches.get(label) for label in current_labels},
            "cluster_to_color": color_map,
            "color_reference": {
                "dataset": reference_spec["dataset"],
                "slice": reference_spec["slice"],
                "evaluation_id": reference_spec["evaluation_id"],
                "description": "Reused per-method cluster colors from the truth-annotated reference slice.",
            },
        }
    return predicted_colors, records


def compute_umap(embedding: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    import anndata as ad
    import scanpy as sc

    temporary = ad.AnnData(X=np.asarray(embedding, dtype=np.float32))
    sc.pp.neighbors(
        temporary,
        n_neighbors=int(settings["n_neighbors"]),
        metric=str(settings["metric"]),
        use_rep="X",
    )
    sc.tl.umap(
        temporary,
        min_dist=float(settings["min_dist"]),
        random_state=int(settings["random_state"]),
    )
    return np.asarray(temporary.obsm["X_umap"])


def transform_spatial_coordinates(coordinates: np.ndarray, transform: dict[str, Any] | None) -> np.ndarray:
    """Apply an optional visual transform around the spatial-coordinate centre.

    The order is deliberately fixed: horizontal flip, vertical flip, then clockwise rotation.
    UMAP coordinates must never pass through this function.
    """
    settings = transform or {}
    flip_horizontal = bool(settings.get("flip_horizontal", False))
    flip_vertical = bool(settings.get("flip_vertical", False))
    rotate_cw = int(settings.get("rotate_cw", 0))
    if rotate_cw not in {0, 90, 180, 270}:
        raise ValueError("spatial.transform.rotate_cw must be one of 0, 90, 180, 270.")
    transformed = np.asarray(coordinates, dtype=float).copy()
    if not flip_horizontal and not flip_vertical and rotate_cw == 0:
        return transformed
    centre = (transformed.min(axis=0) + transformed.max(axis=0)) / 2.0
    transformed -= centre
    if flip_horizontal:
        transformed[:, 0] *= -1.0
    if flip_vertical:
        transformed[:, 1] *= -1.0
    if rotate_cw == 90:
        transformed = np.column_stack((transformed[:, 1], -transformed[:, 0]))
    elif rotate_cw == 180:
        transformed *= -1.0
    elif rotate_cw == 270:
        transformed = np.column_stack((-transformed[:, 1], transformed[:, 0]))
    return transformed + centre


def normalize_spatial_coordinates_to_span(
    coordinates: np.ndarray, target_span: list[float] | tuple[float, float] | None
) -> np.ndarray:
    """Rescale spatial coordinates for display only; source H5AD values stay unchanged."""
    if target_span is None:
        return coordinates
    target = np.asarray(target_span, dtype=float)
    if target.shape != (2,) or np.any(target <= 0):
        raise ValueError("spatial.display_target_span must contain two positive values: [x_span, y_span].")
    lower = np.min(coordinates, axis=0)
    source_span = np.ptp(coordinates, axis=0)
    if np.any(source_span <= 0):
        raise ValueError("Cannot normalize spatial coordinates with a zero-width axis.")
    return (coordinates - lower) / source_span * target


def draw_categorical_scatter(
    axis: plt.Axes,
    coordinates: np.ndarray,
    labels: pd.Series,
    colors: dict[str, str],
    point_size: float,
    point_alpha: float,
    title: str,
    show_axes: bool,
    spatial: bool,
    invert_y: bool = False,
    show_frame: bool = False,
    umap_ylabel_x: float | None = None,
    axis_label_pad: float = 1.0,
) -> None:
    values = labels.astype("string").to_numpy()
    plot_colors = [colors.get(value, "#D9D9D9") for value in values]
    axis.scatter(coordinates[:, 0], coordinates[:, 1], c=plot_colors, s=point_size, alpha=point_alpha, linewidths=0, rasterized=False)
    axis.set_title(title, pad=5)
    if spatial:
        axis.set_aspect("equal", adjustable="box")
        if invert_y:
            axis.invert_yaxis()
    else:
        # 将 UMAP 的坐标范围和 panel 边框都限制为正方形，
        # 同时保持 UMAP 1 与 UMAP 2 的缩放比例一致。
        x_min, x_max = np.nanmin(coordinates[:, 0]), np.nanmax(coordinates[:, 0])
        y_min, y_max = np.nanmin(coordinates[:, 1]), np.nanmax(coordinates[:, 1])

        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0

        span = max(x_max - x_min, y_max - y_min, 1e-8)
        half_span = span * 0.55  # 额外保留约 5% 的边缘空白

        axis.set_xlim(x_center - half_span, x_center + half_span)
        axis.set_ylim(y_center - half_span, y_center + half_span)
        axis.set_box_aspect(1)
    if show_axes:
        axis.set_xlabel("UMAP 1", labelpad=axis_label_pad)
        axis.set_ylabel("UMAP 2", labelpad=axis_label_pad)
        if not spatial and umap_ylabel_x is not None:
            axis.yaxis.set_label_coords(umap_ylabel_x, 0.5)
        axis.set_xticks([])
        axis.set_yticks([])
    else:
        if show_frame:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#222222")
                spine.set_linewidth(0.85)
        else:
            axis.set_axis_off()


def categorical_legend_handles(
    colors: dict[str, str],
    marker_size: float = 6.5,
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=marker_size,
            markerfacecolor=color,
            markeredgewidth=0,
            label=label,
        )
        for label, color in colors.items()
    ]


def save_standalone_legend(
    colors: dict[str, str],
    title: str,
    output_path: Path,
    layout: dict[str, Any],
    legend_kind: str,
    dataset: str | None = None,
) -> Path:
    configure_matplotlib(float(layout["font_size"]))

    legend_font_size = float(layout.get("legend_font_size", float(layout["font_size"]) + 0.5))
    legend_title_size = float(layout.get("legend_title_size", legend_font_size))
    legend_marker_size = float(layout.get("legend_marker_size", 6.5))

    dataset_columns = layout.get(f"{legend_kind}_legend_ncol_by_dataset", {}).get(dataset)
    requested_columns = layout.get(
        f"{legend_kind}_legend_ncol",
        dataset_columns if dataset_columns is not None else layout.get("legend_ncol", "auto"),
    )
    legend_columns = len(colors) if str(requested_columns).lower() == "auto" else int(requested_columns)
    if legend_columns < 1:
        raise ValueError(f"{legend_kind}_legend_ncol must be at least 1.")

    dataset_size = layout.get(f"{legend_kind}_legend_size_by_dataset", {}).get(dataset)
    figure_size = layout.get(
        f"{legend_kind}_legend_size",
        dataset_size if dataset_size is not None else [8, 1.2],
    )
    fig, ax = plt.subplots(figsize=tuple(figure_size))
    ax.axis("off")

    handles = categorical_legend_handles(colors, marker_size=legend_marker_size)
    dataset_order = layout.get(f"{legend_kind}_legend_order_by_dataset", {}).get(dataset)
    legend_order = str(
        layout.get(
            f"{legend_kind}_legend_order",
            dataset_order if dataset_order is not None else "column_major",
        )
    ).lower()
    if legend_order == "row_major" and legend_columns > 1:
        row_count = math.ceil(len(handles) / legend_columns)
        handles = [
            handles[row * legend_columns + column]
            for column in range(legend_columns)
            for row in range(row_count)
            if row * legend_columns + column < len(handles)
        ]
    elif legend_order != "column_major":
        raise ValueError(f"{legend_kind}_legend_order must be 'row_major' or 'column_major'.")

    fig.legend(
        handles=handles,
        loc="center",
        ncol=legend_columns,
        fontsize=legend_font_size,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
        title=title,
        title_fontsize=legend_title_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format="pdf",
        dpi=int(layout["dpi"]),
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(fig)
    return output_path

def make_map_composite(
    coordinates: dict[str, np.ndarray],
    adatas: dict[str, ad.AnnData],
    truth: pd.Series | None,
    truth_colors: dict[str, str] | None,
    predicted_colors: dict[str, dict[str, str]],
    spec: dict[str, Any],
    kind: str,
    truth_legend_colors: dict[str, str] | None = None,
    cluster_legend_colors: dict[str, str] | None = None,
) -> plt.Figure:
    settings = spec[kind]
    layout = spec["layout"]
    methods = spec["method_order"]
    include_truth = truth is not None and truth_colors is not None
    configure_matplotlib(float(layout["font_size"]))
    base_width, base_height = tuple(layout["composite_size"])
    panel_count = len(methods) + int(include_truth)
    width = base_width * panel_count / (len(methods) + 1)
    figure, axes = plt.subplots(1, panel_count, figsize=(width, base_height))
    axes = np.atleast_1d(axes)
    legend_count = int(truth_legend_colors is not None) + int(cluster_legend_colors is not None)
    bottom = 0.10 if legend_count == 0 else 0.24 if legend_count == 1 else 0.38
    figure.subplots_adjust(left=0.02, right=0.99, top=0.84, bottom=bottom, wspace=0.12)
    method_axes = axes
    if include_truth:
        canonical_method = methods[0]
        draw_categorical_scatter(
            axes[0], coordinates[canonical_method], truth, truth_colors,
            float(settings["point_size"]), float(settings["point_alpha"]), "Ground truth",
            bool(settings["show_axes"]), spatial=kind == "spatial",
            invert_y=bool(settings.get("invert_y", False)),
            show_frame=bool(settings.get("show_frame", False)),
            umap_ylabel_x=settings.get("ylabel_position"),
            axis_label_pad=float(settings.get("axis_label_pad", 1.0)),
        )
        method_axes = axes[1:]
    for axis, method in zip(method_axes, methods):
        draw_categorical_scatter(
            axis,
            coordinates[method],
            adatas[method].obs["mclust_benchmark"],
            predicted_colors[method],
            float(settings["point_size"]),
            float(settings["point_alpha"]),
            method,
            bool(settings["show_axes"]),
            spatial=kind == "spatial",
            invert_y=bool(settings.get("invert_y", False)),
            show_frame=bool(settings.get("show_frame", False)),
            umap_ylabel_x=settings.get("ylabel_position"),
            axis_label_pad=float(settings.get("axis_label_pad", 1.0)),
        )
    def add_legend(colors: dict[str, str], title: str, y: float) -> None:
        requested_columns = layout["legend_ncol"]
        legend_columns = (
            len(colors)
            if str(requested_columns).lower() == "auto"
            else int(requested_columns)
        )

        legend_font_size = float(
            layout.get("legend_font_size", float(layout["font_size"]) + 0.5)
        )
        legend_title_size = float(
            layout.get("legend_title_size", legend_font_size)
        )
        legend_marker_size = float(
            layout.get("legend_marker_size", 6.5)
        )

        figure.legend(
            handles=categorical_legend_handles(
                colors,
                marker_size=legend_marker_size,
            ),
            loc="lower center",
            bbox_to_anchor=(0.5, y),
            ncol=legend_columns,
            fontsize=legend_font_size,
            frameon=False,
            handletextpad=0.35,
            columnspacing=1.0,
            title=title,
            title_fontsize=legend_title_size,
        )
    if truth_legend_colors is not None:
        add_legend(truth_legend_colors, "Ground-truth labels", 0.14 if legend_count == 2 else 0.025)
    if cluster_legend_colors is not None:
        add_legend(cluster_legend_colors, "Cluster labels", 0.025)
    return figure


def make_single_map_panel(
    coordinates: np.ndarray,
    labels: pd.Series,
    colors: dict[str, str],
    title: str,
    spec: dict[str, Any],
    kind: str,
) -> plt.Figure:
    settings = spec[kind]
    configure_matplotlib(float(spec["layout"]["font_size"]))
    figure, axis = plt.subplots(figsize=tuple(spec["layout"]["single_panel_size"]))
    draw_categorical_scatter(
        axis,
        coordinates,
        labels,
        colors,
        float(settings["point_size"]),
        float(settings["point_alpha"]),
        title,
        bool(settings["show_axes"]),
        spatial=kind == "spatial",
        invert_y=bool(settings.get("invert_y", False)),
        show_frame=bool(settings.get("show_frame", False)),
        umap_ylabel_x=settings.get("ylabel_position"),
        axis_label_pad=float(settings.get("axis_label_pad", 1.0)),
    )
    return figure


def write_map_pdf(figure: plt.Figure, directory: Path, filename: str, spec: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe_stem(filename)}.pdf"
    use_tight_bbox = bool(spec["layout"].get("tight_bbox", True))
    figure.savefig(
        path,
        format="pdf",
        dpi=int(spec["layout"]["dpi"]),
        bbox_inches="tight" if use_tight_bbox else None,
        pad_inches=0.035 if use_tight_bbox else 0.0,
        facecolor="white",
    )
    plt.close(figure)
    return path


def map_output_directory(config_root: Path, configured_directory: str, spec: dict[str, Any]) -> Path:
    """Group all map artifacts by dataset beneath the configured map-output root."""
    return config_root / configured_directory / safe_stem(spec["dataset"])


def spatial_umap_maps(config_root: Path, spec: dict[str, Any], legend_only: str | None = None) -> list[Path]:
    methods = spec["method_order"]
    print(f"[maps] Loading {spec['dataset']}/{spec['slice']} ({spec['evaluation_id']})...", flush=True)
    adatas = load_prediction_adatas(config_root, spec)
    canonical = adatas[methods[0]]
    truth = canonical.obs.get("reference_label", pd.Series(pd.NA, index=canonical.obs_names, dtype="string"))
    valid_truth = valid_category_mask(truth)
    has_truth = bool(valid_truth.any())
    show_truth = has_truth and bool(spec.get("show_truth", True))
    color_mode = str(spec.get("color_mode", "truth_matched")).lower()
    legend_settings = spec.get("legends", {})
    predicted_colors: dict[str, dict[str, str]] = {}
    mapping_records: dict[str, Any] = {}
    truth_colors: dict[str, str] | None = None
    if has_truth:
        truth_colors = make_truth_colors(truth, valid_truth, spec["colors"]["truth_palette"])
    truth_aliases = spec.get("truth_label_aliases_by_dataset", {}).get(spec["dataset"], {})
    truth_aliases = {**truth_aliases, **spec.get("truth_label_aliases", {})}
    truth_legend_colors = (
        rename_legend_colors(truth_colors, truth_aliases)
        if truth_colors is not None and bool(legend_settings.get("truth", True))
        else None
    )
    cluster_legend_colors: dict[str, str] | None = None
    if color_mode == "raw_clusters":
        print("[maps] Coloring raw mclust cluster labels with a shared palette.", flush=True)
        predicted_colors, all_cluster_colors = raw_cluster_colors(
            adatas, methods, spec["colors"].get("cluster_palette", "tab20")
        )
        cluster_legend_colors = all_cluster_colors if bool(legend_settings.get("clusters", False)) else None
        mapping_records = {
            method: {
                "cluster_to_truth_label": {label: None for label in predicted_colors[method]},
                "cluster_to_color": predicted_colors[method],
                "color_mode": "raw_clusters",
            }
            for method in methods
        }
    elif color_mode != "truth_matched":
        raise ValueError("color_mode must be 'truth_matched' or 'raw_clusters'.")
    elif has_truth:
        for method in methods:
            print(f"[maps] Matching colors: {method}", flush=True)
            color_map, matched_truth = map_prediction_colors(
                truth,
                adatas[method].obs["mclust_benchmark"],
                valid_truth,
                truth_colors,
                spec["colors"]["unmatched_palette"],
                spec["colors"]["unmatched_color"],
            )
            predicted_colors[method] = color_map
            mapping_records[method] = {
                "cluster_to_truth_label": matched_truth,
                "cluster_to_color": color_map,
            }
    else:
        print("[maps] No valid truth labels; reusing reference-slice colors and omitting Ground truth.", flush=True)
        predicted_colors, mapping_records = colors_from_reference_slice(config_root, spec, adatas)
    outputs = spec["outputs"]
    legend_paths: list[Path] = []
    if bool(outputs.get("save_standalone_legends", False)) or legend_only is not None:
        legend_dir = map_output_directory(config_root, outputs["legend_dir"], spec) / spec["evaluation_id"] / spec["slice"]
        stem = spec["output_stem"]

        if truth_legend_colors is not None and legend_only in (None, "truth", "all"):
            legend_paths.append(save_standalone_legend(
                truth_legend_colors,
                "Ground-truth labels",
                legend_dir / f"{safe_stem(stem)}_truth_legend.pdf",
                spec["layout"],
                "truth",
                spec["dataset"],
            ))

        if cluster_legend_colors is not None and legend_only in (None, "cluster", "all"):
            legend_paths.append(save_standalone_legend(
                cluster_legend_colors,
                "Cluster labels",
                legend_dir / f"{safe_stem(stem)}_cluster_legend.pdf",
                spec["layout"],
                "cluster",
                spec["dataset"],
            ))
    if legend_only is not None:
        if not legend_paths:
            raise ValueError(f"Requested standalone legend {legend_only!r} is not available for this figure.")
        print("[maps] Legend-only mode: skipping color mappings, Spatial and UMAP output.", flush=True)
        return legend_paths
    if bool(outputs["save_color_mappings"]):
        mapping_path = map_output_directory(config_root, outputs["mapping_dir"], spec) / spec["evaluation_id"] / spec["slice"]
        mapping_path.mkdir(parents=True, exist_ok=True)
        for method, record in mapping_records.items():
            with (mapping_path / f"{method}_cluster_colors.json").open("w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)

    spatial_transform = spec["spatial"].get("transform", {})
    spatial_coordinates = {
        method: normalize_spatial_coordinates_to_span(
            transform_spatial_coordinates(np.asarray(adata.obsm["spatial"]), spatial_transform),
            spec["spatial"].get("display_target_span"),
        )
        for method, adata in adatas.items()
    }
    umap_coordinates: dict[str, np.ndarray] = {}
    precomputed_methods = set(spec.get("umap_precomputed_methods", []))
    for method, item in adatas.items():
        source_key = spec.get("umap_source_keys_by_method", {}).get(method, spec["umap"]["source_key"])
        source_coordinates = np.asarray(item.obsm[source_key])
        if method in precomputed_methods:
            if source_coordinates.ndim != 2 or source_coordinates.shape[1] != 2:
                raise ValueError(f"{method}: precomputed UMAP/native coordinates must have shape (n_obs, 2).")
            print(f"[maps] Using precomputed 2D coordinates: {method}", flush=True)
            umap_coordinates[method] = source_coordinates
        else:
            print(f"[maps] Computing UMAP: {method}", flush=True)
            umap_coordinates[method] = compute_umap(source_coordinates, spec["umap"])
    stem = spec["output_stem"]
    print("[maps] Writing composite PDFs...", flush=True)
    paths = [
        write_map_pdf(make_map_composite(spatial_coordinates, adatas, truth if show_truth else None, truth_colors if show_truth else None, predicted_colors, spec, "spatial", truth_legend_colors, cluster_legend_colors), map_output_directory(config_root, outputs["spatial_dir"], spec), f"{stem}_spatial", spec),
        write_map_pdf(make_map_composite(umap_coordinates, adatas, truth if show_truth else None, truth_colors if show_truth else None, predicted_colors, spec, "umap", truth_legend_colors, cluster_legend_colors), map_output_directory(config_root, outputs["umap_dir"], spec), f"{stem}_umap", spec),
    ]
    if bool(outputs["save_single_panels"]):
        for kind, coordinates, directory_key in (
            ("spatial", spatial_coordinates, "spatial_panel_dir"),
            ("umap", umap_coordinates, "umap_panel_dir"),
        ):
            print(f"[maps] Writing {kind} single panels...", flush=True)
            if show_truth:
                paths.append(write_map_pdf(make_single_map_panel(coordinates[methods[0]], truth, truth_colors, "Ground truth", spec, kind), map_output_directory(config_root, outputs[directory_key], spec), f"{stem}_{kind}_ground_truth", spec))
            for method in methods:
                paths.append(write_map_pdf(make_single_map_panel(coordinates[method], adatas[method].obs["mclust_benchmark"], predicted_colors[method], method, spec, kind), map_output_directory(config_root, outputs[directory_key], spec), f"{stem}_{kind}_{method}", spec))
    return paths


def save_figure(fig: plt.Figure, stem: Path, spec: dict[str, Any]) -> list[Path]:
    outputs: list[Path] = []
    formats = spec.get("formats", ["pdf"])
    for extension in formats:
        if extension != "pdf":
            raise ValueError("Only PDF output is supported.")
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, format=extension, dpi=int(spec.get("dpi", 600)), bbox_inches="tight", pad_inches=0.035, facecolor="white")
        outputs.append(path)
    return outputs


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_yaml(config_path)
    definitions = config.get("figures", {}).get("definitions", {})
    if args.list:
        for name, value in definitions.items():
            print(f"{name}\t{value.get('type', 'unknown')}")
        return
    if not args.figure:
        raise SystemExit("Use --figure <name>; use --list to view YAML figure definitions.")
    spec = resolve_figure(config, args.figure)
    if spec["type"] == "spatial_umap_maps":
        paths = spatial_umap_maps(PROJECT_ROOT, spec, legend_only=args.legend_only)
        for path in paths:
            print(path)
        return
    if args.legend_only is not None:
        raise SystemExit("--legend-only can only be used with a spatial_umap_maps figure.")
    long = pd.read_csv(input_csv(PROJECT_ROOT, spec))
    if spec["type"] == "bubble_matrix":
        figure = bubble_matrix(long, spec)
    elif spec["type"] == "boxplot":
        figure = boxplot_matrix(long, spec)
    else:
        raise ValueError(f"Unsupported figure type: {spec['type']}")
    paths = save_figure(figure, output_stem(PROJECT_ROOT, spec), spec)
    plt.close(figure)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
