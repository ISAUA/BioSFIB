"""Plot every BioSFIB spatial domain for exploratory inspection.

This image plate does not infer biological labels, calculate truth-based
scores, or modify the input H5AD.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analysis.benchmark.plot_biosfib_recovered_domains import (
    configure_matplotlib,
    load_cluster_colors,
    natural_label_key,
    resolve_path,
    transform_spatial_coordinates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot every BioSFIB spatial domain for exploratory inspection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis/benchmark/biosfib_all_domains_config.yaml"),
        help="YAML configuration file.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--analysis",
        help="Analysis name under analyses; defaults to YAML default_analysis.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run every configured analysis in YAML order.",
    )
    parser.add_argument("--list", action="store_true", help="List analyses and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print cluster counts without writing PDF files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output PDF.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a YAML mapping.")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_analysis(config: dict[str, Any], name: str) -> dict[str, Any]:
    analyses = config.get("analyses", {})
    if name not in analyses:
        raise KeyError(f"Unknown analysis {name!r}. Available: {', '.join(analyses)}")
    spec = deep_merge(config.get("defaults", {}), analyses[name])
    required = ["input_h5ad", "cluster_key", "spatial_key", "output_dir"]
    missing = [key for key in required if not spec.get(key)]
    if missing:
        raise KeyError(f"Analysis {name!r} is missing YAML fields: {missing}")
    columns = int(spec["plot"]["columns"])
    if columns <= 0:
        raise ValueError("plot.columns must be a positive integer.")
    expected = spec.get("expected_clusters")
    if expected is not None and int(expected) <= 0:
        raise ValueError("expected_clusters must be null or a positive integer.")
    return spec


def valid_cluster_labels(values: pd.Series, ignored_values: list[str]) -> list[str]:
    strings = values.astype("string")
    ignored = {str(value).strip().lower() for value in ignored_values}
    normalized = strings.fillna("").str.strip().str.lower()
    valid = strings.notna() & ~normalized.isin(ignored)
    return sorted(strings[valid].unique().tolist(), key=natural_label_key)


def square_limits(coordinates: np.ndarray, padding_fraction: float) -> tuple[tuple[float, float], tuple[float, float]]:
    x, y = coordinates[:, 0], coordinates[:, 1]
    x_span = max(float(np.ptp(x)), 1.0)
    y_span = max(float(np.ptp(y)), 1.0)
    span = max(x_span, y_span) * (1.0 + 2.0 * padding_fraction)
    half_span = span / 2.0
    x_centre = float((x.min() + x.max()) / 2.0)
    y_centre = float((y.min() + y.max()) / 2.0)
    return (
        (x_centre - half_span, x_centre + half_span),
        (y_centre - half_span, y_centre + half_span),
    )


def draw_all_domains(
    adata: ad.AnnData,
    labels: list[str],
    colors: dict[str, str],
    spec: dict[str, Any],
    output_path: Path,
) -> None:
    configure_matplotlib(spec)
    plot = spec["plot"]
    spatial = spec["spatial"]
    columns = min(int(plot["columns"]), len(labels))
    rows = math.ceil(len(labels) / columns)
    configured_size = plot.get("figure_size")
    if configured_size is None:
        width = max(float(plot["minimum_figure_width"]), float(plot["panel_width"]) * columns)
        height = float(plot["panel_height"]) * rows + float(plot["extra_vertical_space"])
        figure_size = (width, height)
    else:
        if len(configured_size) != 2:
            raise ValueError("plot.figure_size must be null or [width, height].")
        figure_size = tuple(float(value) for value in configured_size)

    coordinates = transform_spatial_coordinates(
        np.asarray(adata.obsm[spec["spatial_key"]]), spatial
    )
    x, y = coordinates[:, 0], coordinates[:, 1]
    x_limits, y_limits = square_limits(
        coordinates, float(spatial["coordinate_padding_fraction"])
    )
    cluster_values = adata.obs[spec["cluster_key"]].astype("string").to_numpy()

    fig, axes = plt.subplots(rows, columns, figsize=figure_size, squeeze=False)
    fig.subplots_adjust(
        left=float(plot["margin_left"]),
        right=float(plot["margin_right"]),
        top=float(plot["margin_top"]),
        bottom=float(plot["margin_bottom"]),
        wspace=float(plot["column_spacing"]),
        hspace=float(plot["row_spacing"]),
    )

    for empty_index in range(len(labels), rows * columns):
        axes.flat[empty_index].set_visible(False)

    for index, label in enumerate(labels):
        ax = axes.flat[index]
        mask = cluster_values == label
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
        ax.scatter(
            x[mask],
            y[mask],
            s=float(plot["foreground_point_size"]),
            color=colors[label],
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
        ax.set_title(
            f"{plot['cluster_title_prefix']} {label}",
            fontsize=float(plot["title_font_size"]),
            pad=float(plot["title_pad"]),
        )

    fig.savefig(
        output_path,
        format="pdf",
        dpi=int(plot["dpi"]),
        bbox_inches="tight",
        pad_inches=float(plot["save_pad_inches"]),
    )
    plt.close(fig)


def run_analysis(
    name: str,
    spec: dict[str, Any],
    config_root: Path,
    dry_run: bool,
    overwrite: bool,
) -> None:
    input_path = resolve_path(config_root, spec["input_h5ad"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input H5AD does not exist: {input_path}")
    adata = ad.read_h5ad(input_path)
    if spec["cluster_key"] not in adata.obs:
        raise KeyError(f"obs[{spec['cluster_key']!r}] is absent in {input_path}")
    if spec["spatial_key"] not in adata.obsm:
        raise KeyError(f"obsm[{spec['spatial_key']!r}] is absent in {input_path}")
    labels = valid_cluster_labels(
        adata.obs[spec["cluster_key"]], spec["ignored_cluster_values"]
    )
    expected = spec.get("expected_clusters")
    if expected is not None and len(labels) != int(expected):
        raise ValueError(
            f"Analysis {name!r} contains {len(labels)} clusters; expected {expected}."
        )
    counts = adata.obs[spec["cluster_key"]].astype("string").value_counts()
    print(f"[all-domains] Analysis: {name}")
    print(f"[all-domains] Input: {input_path}")
    print(f"[all-domains] Clusters: {len(labels)} ({', '.join(labels)})")
    print(
        "[all-domains] Cluster spot range: "
        f"{int(counts.loc[labels].min())}-{int(counts.loc[labels].max())}"
    )
    if dry_run:
        print("[all-domains] Dry run successful; no PDF was written.")
        return

    output_dir = resolve_path(config_root, spec["output_dir"])
    output_path = output_dir / str(spec["output_pdf"])
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output PDF already exists: {output_path}. Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.DataFrame({"biosfib_cluster": labels})
    colors = load_cluster_colors(config_root, selected, spec)
    draw_all_domains(adata, labels, colors, spec, output_path)
    print(f"[all-domains] Figure: {output_path}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_yaml(config_path)
    analyses = config.get("analyses", {})
    if not isinstance(analyses, dict) or not analyses:
        raise KeyError("YAML must contain at least one analysis.")
    if args.list:
        for name in analyses:
            print(name)
        return
    if args.all:
        names = list(analyses)
    else:
        name = args.analysis or config.get("default_analysis")
        if not name:
            raise KeyError("No --analysis was supplied and default_analysis is empty.")
        names = [str(name)]

    for index, name in enumerate(names, start=1):
        if len(names) > 1:
            print(f"[all-domains] Run {index}/{len(names)}")
        spec = resolve_analysis(config, name)
        run_analysis(name, spec, PROJECT_ROOT, args.dry_run, args.overwrite)


if __name__ == "__main__":
    main()
