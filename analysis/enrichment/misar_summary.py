"""Build the MISAR E11.0-E18.5 developmental enrichment supplement."""

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("GO:BP", "KEGG")
SOURCE_LABELS = {"GO:BP": "GO Biological Process", "KEGG": "KEGG"}
OVERLAP_COLORS = {"GO:BP": "#4C78A8", "KEGG": "#E08B4F"}
TERM_EDGE_PADDING = 0.65
CM_PER_INCH = 2.54


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize completed MISAR GO-BP/KEGG enrichment across E11.0-E18.5 "
            "without issuing new enrichment queries."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fdr-cutoff", type=float, default=0.05)
    parser.add_argument(
        "--color-cap",
        type=float,
        default=20.0,
        help="Upper cap for displayed -log10(best cluster FDR).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg"),
        default=("pdf", "png"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-width-cm", type=float, default=18.0)
    parser.add_argument("--figure-height-cm", type=float, default=22.0)
    return parser.parse_args()


def resolve_path(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Configuration must contain a YAML mapping.")
    return payload


def validate_config(config):
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 8:
        raise ValueError("MISAR summary config must define exactly eight datasets.")

    required = {"key", "stage", "stage_value", "slice", "label", "results", "qc", "manifest"}
    keys = []
    labels = []
    stage_slices = defaultdict(list)
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError("datasets[%d] must be a mapping." % index)
        missing = sorted(required - set(dataset))
        if missing:
            raise ValueError("datasets[%d] lacks: %s" % (index, ", ".join(missing)))
        dataset["stage_value"] = float(dataset["stage_value"])
        dataset["slice"] = str(dataset["slice"])
        keys.append(str(dataset["key"]))
        labels.append(str(dataset["label"]))
        stage_slices[str(dataset["stage"])].append(str(dataset["slice"]))

    if len(set(keys)) != len(keys):
        raise ValueError("Dataset keys must be unique.")
    if len(set(labels)) != len(labels):
        raise ValueError("Dataset labels must be unique.")
    if len(stage_slices) != 4:
        raise ValueError("MISAR summary config must define exactly four stages.")
    for stage, slices in stage_slices.items():
        if sorted(slices) != ["S1", "S2"]:
            raise ValueError("Stage %s must contain exactly S1 and S2." % stage)

    groups = config.get("term_groups")
    if not isinstance(groups, dict) or set(groups) != set(SOURCES):
        raise ValueError("term_groups must contain exactly GO:BP and KEGG.")
    for source in SOURCES:
        if not isinstance(groups[source], list) or not groups[source]:
            raise ValueError("term_groups[%s] must be a non-empty list." % source)
    return datasets


def flatten_terms(config):
    records = []
    for source in SOURCES:
        seen = set()
        source_order = 0
        for group_order, group in enumerate(config["term_groups"][source]):
            required = {"label", "short_label", "color", "terms"}
            missing = sorted(required - set(group))
            if missing:
                raise ValueError("%s term group lacks: %s" % (source, ", ".join(missing)))
            if not isinstance(group["terms"], list) or not group["terms"]:
                raise ValueError("%s group %s has no terms." % (source, group["label"]))
            for within_group_order, term in enumerate(group["terms"]):
                if not isinstance(term, dict) or not {"id", "label"}.issubset(term):
                    raise ValueError("Each configured term needs id and label.")
                term_id = str(term["id"])
                if term_id in seen:
                    raise ValueError("Duplicate configured %s term: %s" % (source, term_id))
                if source == "GO:BP" and not term_id.startswith("GO:"):
                    raise ValueError("GO:BP term does not use a GO identifier: %s" % term_id)
                if source == "KEGG" and not term_id.startswith("KEGG:"):
                    raise ValueError("KEGG term does not use a KEGG identifier: %s" % term_id)
                seen.add(term_id)
                records.append(
                    {
                        "source": source,
                        "source_label": SOURCE_LABELS[source],
                        "group": str(group["label"]),
                        "group_short_label": str(group["short_label"]),
                        "group_color": str(group["color"]),
                        "group_order": group_order,
                        "within_group_order": within_group_order,
                        "term_order": source_order,
                        "term_id": term_id,
                        "term": str(term["label"]),
                    }
                )
                source_order += 1
    return pd.DataFrame.from_records(records)


def read_inputs(datasets):
    result_frames = []
    metadata = []
    database_versions = set()
    result_required = {"dataset_key", "cluster", "source", "term_id", "term", "fdr"}
    qc_required = {"cluster", "status"}

    for dataset_order, dataset in enumerate(datasets):
        result_path = resolve_path(dataset["results"])
        qc_path = resolve_path(dataset["qc"])
        manifest_path = resolve_path(dataset["manifest"])
        for path in (result_path, qc_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError("Required enrichment input is missing: %s" % path)

        frame = pd.read_csv(result_path)
        missing = sorted(result_required - set(frame.columns))
        if missing:
            raise ValueError("%s lacks result columns: %s" % (result_path, ", ".join(missing)))
        observed_keys = set(frame["dataset_key"].dropna().astype(str))
        if observed_keys != {str(dataset["key"])}:
            raise ValueError(
                "%s dataset_key mismatch: expected %s, observed %s"
                % (result_path, dataset["key"], sorted(observed_keys))
            )
        frame = frame.copy()
        frame["dataset_key"] = frame["dataset_key"].astype(str)
        frame["cluster"] = frame["cluster"].astype(str)
        frame["source"] = frame["source"].astype(str)
        frame["term_id"] = frame["term_id"].astype(str)
        frame["fdr"] = pd.to_numeric(frame["fdr"], errors="coerce")
        if frame["fdr"].isna().any():
            raise ValueError("%s contains non-numeric FDR values." % result_path)
        frame["dataset_order"] = dataset_order
        result_frames.append(frame)

        qc = pd.read_csv(qc_path)
        missing = sorted(qc_required - set(qc.columns))
        if missing:
            raise ValueError("%s lacks QC columns: %s" % (qc_path, ", ".join(missing)))
        qc["cluster"] = qc["cluster"].astype(str)
        ready_clusters = int(qc.loc[qc["status"].astype(str) == "ready", "cluster"].nunique())
        if ready_clusters <= 0:
            raise ValueError("%s has no ready clusters." % qc_path)

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") != "ok":
            raise ValueError("%s does not have status=ok." % manifest_path)
        if str(manifest.get("dataset_key")) != str(dataset["key"]):
            raise ValueError("%s dataset_key does not match the summary config." % manifest_path)
        versions = [str(value) for value in manifest.get("database_versions", []) if str(value)]
        if len(versions) != 1:
            raise ValueError("%s must record exactly one database version." % manifest_path)
        database_versions.add(versions[0])
        input_fdr_cutoff = float(manifest.get("parameters", {}).get("fdr_cutoff", math.nan))
        if not 0.0 < input_fdr_cutoff <= 1.0:
            raise ValueError("%s lacks a valid enrichment FDR cutoff." % manifest_path)

        metadata.append(
            {
                **dataset,
                "dataset_order": dataset_order,
                "ready_clusters": ready_clusters,
                "result_path": result_path,
                "qc_path": qc_path,
                "manifest_path": manifest_path,
                "result_sha256": sha256_file(result_path),
                "qc_sha256": sha256_file(qc_path),
                "manifest_sha256": sha256_file(manifest_path),
                "database_version": versions[0],
                "input_fdr_cutoff": input_fdr_cutoff,
            }
        )

    if len(database_versions) != 1:
        raise ValueError(
            "All eight enrichment inputs must use one g:Profiler database version; observed %s"
            % sorted(database_versions)
        )
    return pd.concat(result_frames, ignore_index=True), metadata, next(iter(database_versions))


def build_term_matrix(results, metadata, terms, fdr_cutoff, color_cap):
    records = []
    metadata_by_key = {str(item["key"]): item for item in metadata}
    for _, term in terms.sort_values(["source", "term_order"], kind="stable").iterrows():
        for dataset in metadata:
            subset = results[
                (results["dataset_key"] == str(dataset["key"]))
                & (results["source"] == str(term["source"]))
                & (results["term_id"] == str(term["term_id"]))
                & (results["fdr"] <= float(fdr_cutoff))
            ]
            present = not subset.empty
            best_fdr = float(subset["fdr"].min()) if present else math.nan
            n_significant_clusters = int(subset["cluster"].nunique()) if present else 0
            ready_clusters = int(metadata_by_key[str(dataset["key"])]["ready_clusters"])
            cluster_fraction = n_significant_clusters / float(ready_clusters)
            minus_log10 = -math.log10(max(best_fdr, 1e-300)) if present else math.nan
            records.append(
                {
                    "dataset_key": str(dataset["key"]),
                    "dataset_order": int(dataset["dataset_order"]),
                    "stage": str(dataset["stage"]),
                    "stage_value": float(dataset["stage_value"]),
                    "slice": str(dataset["slice"]),
                    "dataset_label": str(dataset["label"]),
                    "source": str(term["source"]),
                    "source_label": str(term["source_label"]),
                    "group": str(term["group"]),
                    "group_short_label": str(term["group_short_label"]),
                    "group_color": str(term["group_color"]),
                    "group_order": int(term["group_order"]),
                    "term_order": int(term["term_order"]),
                    "term_id": str(term["term_id"]),
                    "term": str(term["term"]),
                    "present": bool(present),
                    "ready_clusters": ready_clusters,
                    "n_significant_clusters": n_significant_clusters,
                    "cluster_fraction": cluster_fraction,
                    "best_cluster_fdr": best_fdr,
                    "minus_log10_best_cluster_fdr": minus_log10,
                    "display_minus_log10_fdr": min(minus_log10, float(color_cap)) if present else math.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def jaccard_record(comparison_type, source, comparison_label, left_label, right_label, left, right):
    shared = left & right
    union = left | right
    score = len(shared) / float(len(union)) if union else math.nan
    return {
        "comparison_type": comparison_type,
        "source": source,
        "source_label": SOURCE_LABELS[source],
        "comparison_label": comparison_label,
        "left_label": left_label,
        "right_label": right_label,
        "left_terms": len(left),
        "right_terms": len(right),
        "shared_terms": len(shared),
        "union_terms": len(union),
        "jaccard": score,
        "left_term_ids": ";".join(sorted(left)),
        "right_term_ids": ";".join(sorted(right)),
        "shared_term_ids": ";".join(sorted(shared)),
    }


def build_overlap_summary(results, metadata, fdr_cutoff):
    significant = results[
        (results["fdr"] <= float(fdr_cutoff)) & results["source"].isin(SOURCES)
    ].copy()
    term_sets = {
        (dataset_key, source): set(group["term_id"].astype(str))
        for (dataset_key, source), group in significant.groupby(["dataset_key", "source"])
    }
    stage_order = []
    by_stage_slice = {}
    for dataset in metadata:
        stage = str(dataset["stage"])
        if stage not in stage_order:
            stage_order.append(stage)
        by_stage_slice[(stage, str(dataset["slice"]))] = str(dataset["key"])

    records = []
    stage_unions = {}
    stage_cores = {}
    for stage in stage_order:
        for source in SOURCES:
            left = term_sets.get((by_stage_slice[(stage, "S1")], source), set())
            right = term_sets.get((by_stage_slice[(stage, "S2")], source), set())
            records.append(
                jaccard_record(
                    "within_stage_s1_s2", source, stage, "%s S1" % stage, "%s S2" % stage, left, right
                )
            )
            stage_unions[(stage, source)] = left | right
            stage_cores[(stage, source)] = left & right

    for left_stage, right_stage in zip(stage_order[:-1], stage_order[1:]):
        comparison_label = "%s-%s" % (left_stage, right_stage)
        for source in SOURCES:
            records.append(
                jaccard_record(
                    "adjacent_stage_union",
                    source,
                    comparison_label,
                    "%s S1/S2 union" % left_stage,
                    "%s S1/S2 union" % right_stage,
                    stage_unions[(left_stage, source)],
                    stage_unions[(right_stage, source)],
                )
            )
            records.append(
                jaccard_record(
                    "adjacent_stage_core",
                    source,
                    comparison_label,
                    "%s S1/S2 intersection" % left_stage,
                    "%s S1/S2 intersection" % right_stage,
                    stage_cores[(left_stage, source)],
                    stage_cores[(right_stage, source)],
                )
            )
    return pd.DataFrame.from_records(records)


def configure_plot_style():
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "legend.title_fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def dot_size(fraction):
    return 18.0 + 150.0 * float(fraction)


def split_dataset_label(label):
    parts = str(label).rsplit(" ", 1)
    return "\n".join(parts) if len(parts) == 2 else str(label)


def plot_term_matrix(ax, matrix, metadata, source, norm, cmap, panel_label):
    data = matrix[matrix["source"] == source].copy()
    term_meta = (
        data[
            [
                "term_id",
                "term",
                "term_order",
                "group",
                "group_short_label",
                "group_color",
                "group_order",
            ]
        ]
        .drop_duplicates()
        .sort_values("term_order", kind="stable")
    )
    n_terms = int(term_meta.shape[0])
    y_lookup = {
        str(row.term_id): n_terms - 1 - index
        for index, row in enumerate(term_meta.itertuples(index=False))
    }
    x_lookup = {str(item["key"]): int(item["dataset_order"]) for item in metadata}

    for x in range(len(metadata)):
        ax.axvline(x, color="#ECECEC", linewidth=0.65, zorder=0)
    for y in range(n_terms):
        ax.axhline(y, color="#ECECEC", linewidth=0.65, zorder=0)
    for index in range(len(metadata) - 1):
        if str(metadata[index]["stage"]) != str(metadata[index + 1]["stage"]):
            ax.axvline(index + 0.5, color="#9A9A9A", linewidth=0.9, zorder=1)

    group_boundaries = (
        term_meta.groupby("group_order", sort=True).size().cumsum().tolist()[:-1]
    )
    for cumulative in group_boundaries:
        ax.axhline(n_terms - cumulative - 0.5, color="#8D8D8D", linewidth=0.8, zorder=1)

    present = data[data["present"]].copy()
    scatter = ax.scatter(
        present["dataset_key"].map(x_lookup),
        present["term_id"].map(y_lookup),
        c=present["display_minus_log10_fdr"].to_numpy(dtype=float),
        s=present["cluster_fraction"].map(dot_size).to_numpy(dtype=float),
        cmap=cmap,
        norm=norm,
        edgecolor="#29343A",
        linewidth=0.4,
        zorder=3,
    )

    ax.set_xticks(range(len(metadata)))
    ax.set_xticklabels([split_dataset_label(item["label"]) for item in metadata])
    ax.set_xlim(-0.55, len(metadata) - 1 + 0.55)
    positions = [n_terms - 1 - index for index in range(n_terms)]
    ax.set_yticks(positions)
    ax.set_yticklabels(term_meta["term"].astype(str).tolist())
    ax.set_ylim(-TERM_EDGE_PADDING, n_terms - 1 + TERM_EDGE_PADDING)
    ax.tick_params(axis="both", length=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("%s programs across MISAR development" % SOURCE_LABELS[source], loc="left", pad=10)
    ax.text(
        -0.035,
        1.065,
        panel_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    for _, group in term_meta.groupby("group_order", sort=True):
        positions = [y_lookup[str(term_id)] for term_id in group["term_id"]]
        ax.vlines(
            -0.50,
            min(positions) - 0.38,
            max(positions) + 0.38,
            color=str(group["group_color"].iloc[0]),
            linewidth=2.2,
            zorder=4,
        )
    return scatter


def plot_overlap_panel(ax, overlap, comparison_type, order, title, panel_label, y_max, show_legend):
    data = overlap[overlap["comparison_type"] == comparison_type].copy()
    x = np.arange(len(order), dtype=float)
    width = 0.34
    for source_index, source in enumerate(SOURCES):
        source_rows = data[data["source"] == source].set_index("comparison_label")
        values = np.asarray([float(source_rows.loc[label, "jaccard"]) for label in order])
        positions = x + (source_index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width=width,
            color=OVERLAP_COLORS[source],
            edgecolor="white",
            linewidth=0.6,
            label="GO-BP" if source == "GO:BP" else "KEGG",
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.018,
                "%.2f" % value,
                ha="center",
                va="bottom",
                fontsize=7.0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("-", "–") for label in order])
    ax.set_ylim(0.0, y_max)
    ax.set_ylabel("Exact term-set Jaccard")
    ax.set_title(title, loc="left", pad=8)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.text(
        -0.10,
        1.04,
        panel_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    if show_legend:
        ax.legend(frameon=False, ncol=2, loc="upper left")


def build_figure(
    matrix,
    overlap,
    metadata,
    fdr_cutoff,
    color_cap,
    figure_width_cm,
    figure_height_cm,
):
    configure_plot_style()
    color_floor = -math.log10(float(fdr_cutoff))
    norm = Normalize(vmin=color_floor, vmax=float(color_cap))
    cmap = LinearSegmentedColormap.from_list(
        "misar_significance", ["#E6EEF3", "#79A7BE", "#244A64"]
    )

    fig = plt.figure(
        figsize=(float(figure_width_cm) / CM_PER_INCH, float(figure_height_cm) / CM_PER_INCH),
        dpi=150,
    )
    upper_grid = fig.add_gridspec(
        2,
        1,
        left=0.36,
        right=0.875,
        top=0.965,
        bottom=0.405,
        hspace=0.42,
    )
    lower_grid = fig.add_gridspec(
        1,
        2,
        left=0.11,
        right=0.96,
        top=0.285,
        bottom=0.09,
        wspace=0.35,
    )
    ax_go = fig.add_subplot(upper_grid[0, 0])
    ax_kegg = fig.add_subplot(upper_grid[1, 0])
    ax_within = fig.add_subplot(lower_grid[0, 0])
    ax_adjacent = fig.add_subplot(lower_grid[0, 1])

    plot_term_matrix(ax_go, matrix, metadata, "GO:BP", norm, cmap, "a")
    plot_term_matrix(ax_kegg, matrix, metadata, "KEGG", norm, cmap, "b")

    stage_order = []
    for item in metadata:
        if str(item["stage"]) not in stage_order:
            stage_order.append(str(item["stage"]))
    adjacent_order = ["%s-%s" % pair for pair in zip(stage_order[:-1], stage_order[1:])]
    displayed_overlap = overlap[
        overlap["comparison_type"].isin(["within_stage_s1_s2", "adjacent_stage_union"])
    ]
    observed_max = float(displayed_overlap["jaccard"].max())
    y_max = min(1.0, max(0.7, math.ceil((observed_max + 0.10) * 10.0) / 10.0))
    plot_overlap_panel(
        ax_within,
        overlap,
        "within_stage_s1_s2",
        stage_order,
        "Within-stage S1/S2\nconcordance",
        "c",
        y_max,
        True,
    )
    plot_overlap_panel(
        ax_adjacent,
        overlap,
        "adjacent_stage_union",
        adjacent_order,
        "Adjacent-stage concordance\n(S1/S2 union)",
        "d",
        y_max,
        False,
    )

    color_axis = fig.add_axes([0.915, 0.50, 0.014, 0.34])
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=color_axis)
    colorbar.set_label("Best cluster -log10(FDR)\n(capped at %g)" % float(color_cap), fontsize=7.2)
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.outline.set_linewidth(0.6)

    size_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=math.sqrt(dot_size(fraction)),
            markerfacecolor="#79A7BE",
            markeredgecolor="#29343A",
            markeredgewidth=0.4,
            label="%d%%" % int(round(100 * fraction)),
        )
        for fraction in (0.1, 0.3, 0.5)
    ]
    fig.legend(
        handles=size_handles,
        title="Ready clusters\nwith term",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.895, 0.46),
        borderaxespad=0.0,
        labelspacing=0.9,
    )
    fig.text(
        0.11,
        0.012,
        "Dot color: strongest cluster-level FDR; dot size: fraction of QC-ready clusters.\n"
        "Bars use all FDR-significant term IDs, not only displayed terms.",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#444444",
    )
    return fig


def main():
    args = parse_args()
    if not 0.0 < float(args.fdr_cutoff) <= 1.0:
        raise ValueError("--fdr-cutoff must be in (0, 1].")
    color_floor = -math.log10(float(args.fdr_cutoff))
    if float(args.color_cap) <= color_floor:
        raise ValueError("--color-cap must exceed -log10(--fdr-cutoff).")
    if int(args.dpi) <= 0:
        raise ValueError("--dpi must be positive.")
    if float(args.figure_width_cm) <= 0.0 or float(args.figure_height_cm) <= 0.0:
        raise ValueError("Figure width and height must be positive.")

    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(config_path)
    datasets = validate_config(config)
    terms = flatten_terms(config)
    results, metadata, database_version = read_inputs(datasets)
    incompatible_cutoffs = [
        str(item["label"])
        for item in metadata
        if float(args.fdr_cutoff) > float(item["input_fdr_cutoff"]) + 1e-12
    ]
    if incompatible_cutoffs:
        raise ValueError(
            "Requested FDR cutoff exceeds the cutoff used to create the significant-results "
            "input for: %s" % ", ".join(incompatible_cutoffs)
        )

    observed_pairs = set(zip(results["source"].astype(str), results["term_id"].astype(str)))
    configured_pairs = set(zip(terms["source"].astype(str), terms["term_id"].astype(str)))
    missing_terms = sorted(configured_pairs - observed_pairs)
    if missing_terms:
        raise ValueError("Configured terms absent from all inputs: %s" % missing_terms)

    matrix = build_term_matrix(
        results, metadata, terms, float(args.fdr_cutoff), float(args.color_cap)
    ).sort_values(["source", "term_order", "dataset_order"], kind="stable")
    overlap = build_overlap_summary(results, metadata, float(args.fdr_cutoff))

    matrix_path = output_dir / "misar_developmental_term_matrix.csv"
    overlap_path = output_dir / "misar_developmental_overlap_summary.csv"
    matrix.to_csv(matrix_path, index=False, na_rep="")
    overlap.to_csv(overlap_path, index=False, na_rep="")

    fig = build_figure(
        matrix,
        overlap,
        metadata,
        float(args.fdr_cutoff),
        float(args.color_cap),
        float(args.figure_width_cm),
        float(args.figure_height_cm),
    )
    formats = list(dict.fromkeys(str(value).lower() for value in args.formats))
    figure_paths = []
    for fmt in formats:
        path = output_dir / ("misar_developmental_enrichment_summary.%s" % fmt)
        # Do not use bbox_inches="tight": it changes the requested physical canvas size.
        save_kwargs = {"facecolor": "white"}
        if fmt in {"png", "svg"}:
            save_kwargs["dpi"] = int(args.dpi)
        fig.savefig(path, **save_kwargs)
        figure_paths.append(path)
    plt.close(fig)

    manifest_inputs = []
    for item in metadata:
        manifest_inputs.append(
            {
                "dataset_key": str(item["key"]),
                "stage": str(item["stage"]),
                "slice": str(item["slice"]),
                "label": str(item["label"]),
                "ready_clusters": int(item["ready_clusters"]),
                "results": str(item["result_path"]),
                "results_sha256": str(item["result_sha256"]),
                "qc": str(item["qc_path"]),
                "qc_sha256": str(item["qc_sha256"]),
                "manifest": str(item["manifest_path"]),
                "manifest_sha256": str(item["manifest_sha256"]),
                "database_version": str(item["database_version"]),
                "input_fdr_cutoff": float(item["input_fdr_cutoff"]),
            }
        )
    analysis_meta = config.get("analysis", {})
    manifest = {
        "status": "ok",
        "analysis": "MISAR developmental GO-BP/KEGG enrichment summary",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim": analysis_meta.get("claim"),
        "role": analysis_meta.get("role"),
        "selection_status": analysis_meta.get("selection_status"),
        "selection_note": analysis_meta.get("selection_note"),
        "interpretation_scope": (
            "Descriptive robustness analysis. Dot matrices show a frozen reporting subset; "
            "Jaccard panels use all term IDs with cluster-level FDR at or below the cutoff. "
            "Stages and slices are not treated as inferential biological replicates."
        ),
        "parameters": {
            "fdr_cutoff": float(args.fdr_cutoff),
            "color_cap": float(args.color_cap),
            "figure_width_cm": float(args.figure_width_cm),
            "figure_height_cm": float(args.figure_height_cm),
            "font_size_pt_range": [7.0, 12.0],
            "dot_color": "minimum cluster-level FDR for each dataset-term pair",
            "dot_size": "fraction of QC-ready clusters significant for the term",
            "overlap_metric": "exact Jaccard index on all significant term IDs",
            "within_stage_sets": "S1 versus S2",
            "adjacent_stage_sets": "union of S1/S2 within each stage",
            "error_bars": "not applicable; overlaps are deterministic summaries",
            "statistical_test": "none for cross-stage comparison",
        },
        "gprofiler_database_versions": [database_version],
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "datasets": manifest_inputs,
        },
        "representative_terms": terms.to_dict(orient="records"),
        "counts": {
            "datasets": len(metadata),
            "go_terms_displayed": int((terms["source"] == "GO:BP").sum()),
            "kegg_terms_displayed": int((terms["source"] == "KEGG").sum()),
            "term_matrix_rows": int(matrix.shape[0]),
            "overlap_rows": int(overlap.shape[0]),
        },
        "outputs": {
            "term_matrix": str(matrix_path),
            "overlap_summary": str(overlap_path),
            "figures": [str(path) for path in figure_paths],
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "matplotlib": matplotlib.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest_path = output_dir / "misar_developmental_enrichment_summary_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("[MISAR enrichment summary] complete")
    print("  g:Profiler database: %s" % database_version)
    print("  term matrix: %s" % matrix_path)
    print("  overlap summary: %s" % overlap_path)
    print("  figures: %s" % ", ".join(str(path) for path in figure_paths))
    print("  manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()
