"""Compare fixed P22 enrichment results across two or four modalities."""

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ORDER = [
    "cortical_neuron",
    "neurogenic_proliferative",
    "oligodendroglial_myelination",
    "striatal_neuron",
]
DATASET_ORDER = {"ATAC": 0, "H3K27ac": 1, "H3K4me3": 2, "H3K27me3": 3}
TERM_EDGE_PADDING = 0.55
CM_PER_INCH = 2.54
SIGNIFICANCE_CMAP = LinearSegmentedColormap.from_list(
    "enrichment_significance", ["#E6EEF3", "#79A7BE", "#244A64"]
)
MODULE_COLORS = {
    "cortical_neuron": "#355C8A",
    "neurogenic_proliferative": "#8A5A9E",
    "oligodendroglial_myelination": "#2A7F62",
    "striatal_neuron": "#B05A45",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot paired or four-modality P22 GO-BP/KEGG enrichment results.")
    parser.add_argument("--atac-results", required=True)
    parser.add_argument("--h3k27ac-results", required=True)
    parser.add_argument("--h3k4me3-results", default=None)
    parser.add_argument("--h3k27me3-results", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fdr-cutoff", type=float, default=0.05)
    parser.add_argument("--top-terms-per-module", type=int, default=3)
    parser.add_argument(
        "--min-datasets-per-term",
        type=int,
        default=2,
        help="Minimum number of input modalities in which a selected term must be significant.",
    )
    parser.add_argument("--go-redundancy-jaccard", type=float, default=0.70)
    parser.add_argument("--color-cap", type=float, default=20.0)
    parser.add_argument("--combined-width-cm", type=float, default=18.0)
    parser.add_argument("--combined-height-cm", type=float, default=22.0)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def configure_plot_style():
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "legend.title_fontsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def significance_norm(frame, fdr_cutoff, color_cap):
    color_floor = -math.log10(float(fdr_cutoff))
    if float(color_cap) <= color_floor:
        raise ValueError("--color-cap must exceed -log10(--fdr-cutoff).")
    values = -np.log10(np.maximum(frame["fdr"].to_numpy(dtype=float), 1e-300))
    observed_max = float(np.nanmax(values))
    color_max = min(float(color_cap), observed_max)
    if math.isclose(color_floor, color_max) or color_max < color_floor:
        color_max = color_floor + 0.5
    return Normalize(vmin=color_floor, vmax=color_max), observed_max > float(color_cap)


def gene_ratio_size(value):
    return 35.0 + 520.0 * float(value)


def resolve_path(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_results(path, label):
    frame = pd.read_csv(resolve_path(path))
    required = {"dataset_key", "module", "module_label", "cluster", "source", "term_id", "term", "fdr", "precision", "overlap_gene_ids"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("%s results lack columns: %s" % (label, ", ".join(missing)))
    frame = frame[frame["module"].astype(str) != "unassigned"].copy()
    frame["dataset_label"] = label
    frame["fdr"] = pd.to_numeric(frame["fdr"], errors="coerce")
    frame["precision"] = pd.to_numeric(frame["precision"], errors="coerce")
    return frame


def gene_set(value):
    return {gene for gene in str(value).split(";") if gene and gene.lower() != "nan"}


def merge_gene_sets(values):
    genes = set()
    for value in values:
        genes.update(gene_set(value))
    return ";".join(sorted(genes))


def select_terms(combined, args):
    significant = combined[combined["fdr"] <= float(args.fdr_cutoff)].copy()
    selected_keys = []
    selection_metadata = {}
    for (module, source), group in significant.groupby(["module", "source"], sort=False):
        ranked = (
            group.groupby(["term_id", "term"], as_index=False)
            .agg(
                n_datasets_significant=("dataset_label", "nunique"),
                worst_fdr=("fdr", "max"),
                best_fdr=("fdr", "min"),
                precision=("precision", "max"),
                overlap_gene_ids=("overlap_gene_ids", merge_gene_sets),
            )
        )
        ranked = ranked[
            ranked["n_datasets_significant"] >= int(args.min_datasets_per_term)
        ].sort_values(
            ["n_datasets_significant", "worst_fdr", "best_fdr", "precision", "term"],
            ascending=[False, True, True, False, True],
        )
        kept_sets = []
        kept = 0
        for _, row in ranked.iterrows():
            genes = gene_set(row["overlap_gene_ids"])
            redundant = False
            if source == "GO:BP" and genes:
                for previous in kept_sets:
                    union = genes | previous
                    if union and len(genes & previous) / len(union) > float(args.go_redundancy_jaccard):
                        redundant = True
                        break
            if redundant:
                continue
            key = (str(module), str(source), str(row["term_id"]))
            selected_keys.append(key)
            selection_metadata[key] = {
                "n_datasets_significant": int(row["n_datasets_significant"]),
                "worst_fdr": float(row["worst_fdr"]),
                "selection_rank": kept + 1,
            }
            kept_sets.append(genes)
            kept += 1
            if kept >= int(args.top_terms_per_module):
                break
    key_set = set(selected_keys)
    mask = combined.apply(
        lambda row: (str(row["module"]), str(row["source"]), str(row["term_id"])) in key_set,
        axis=1,
    )
    selected = combined[mask & (combined["fdr"] <= float(args.fdr_cutoff))].copy()
    if selected.empty:
        selected["n_datasets_significant"] = pd.Series(dtype=int)
        selected["worst_fdr"] = pd.Series(dtype=float)
        selected["selection_rank"] = pd.Series(dtype=int)
        return selected
    keys = selected.apply(
        lambda row: (str(row["module"]), str(row["source"]), str(row["term_id"])), axis=1
    )
    selected["n_datasets_significant"] = [selection_metadata[key]["n_datasets_significant"] for key in keys]
    selected["worst_fdr"] = [selection_metadata[key]["worst_fdr"] for key in keys]
    selected["selection_rank"] = [selection_metadata[key]["selection_rank"] for key in keys]
    return selected


def plot_source(
    frame,
    group_reference,
    source,
    output_dir,
    formats,
    dpi,
    stem_prefix,
    fdr_cutoff,
    color_cap,
):
    data = frame[frame["source"] == source].copy()
    if data.empty:
        print("[P22 enrichment comparison] no recurrent %s terms passed selection" % source)
        return []

    module_terms = {}
    height_ratios = []
    for module in MODULE_ORDER:
        terms = (
            data[data["module"].astype(str) == module][
                ["term_id", "term", "selection_rank", "n_datasets_significant", "worst_fdr"]
            ]
            .drop_duplicates()
            .sort_values(["selection_rank", "term"], kind="stable")
        )
        module_terms[module] = terms
        height_ratios.append(max(1.0, 0.72 * max(terms.shape[0], 1) + 0.8))

    norm, color_capped = significance_norm(data, fdr_cutoff, color_cap)
    fig, axes = plt.subplots(
        nrows=len(MODULE_ORDER),
        ncols=1,
        figsize=(9.2, max(8.5, sum(height_ratios))),
        dpi=int(dpi),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )
    axes = axes.ravel()
    scatter = None
    for ax, module in zip(axes, MODULE_ORDER):
        module_data = data[data["module"].astype(str) == module].copy()
        reference = group_reference[group_reference["module"].astype(str) == module][
            ["module_label", "dataset_label", "cluster"]
        ].drop_duplicates().copy()
        reference["dataset_order"] = reference["dataset_label"].map(DATASET_ORDER).fillna(len(DATASET_ORDER))
        reference = reference.sort_values(["dataset_order", "cluster"], kind="stable")
        reference["group"] = reference.apply(
            lambda row: "%s\nC%s" % (row["dataset_label"], row["cluster"]), axis=1
        )
        groups = reference["group"].astype(str).tolist()
        x_lookup = {value: index for index, value in enumerate(groups)}
        module_data["group"] = module_data.apply(
            lambda row: "%s\nC%s" % (row["dataset_label"], row["cluster"]), axis=1
        )
        terms = module_terms[module]
        term_ids = terms["term_id"].astype(str).tolist()
        y_lookup = {term_id: len(term_ids) - 1 - index for index, term_id in enumerate(term_ids)}
        module_label = (
            str(reference["module_label"].iloc[0]) if not reference.empty else module.replace("_", " ").title()
        )
        ax.set_title(module_label, loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_xlim(-0.5, max(len(groups) - 0.5, 0.5))
        if module_data.empty or terms.empty:
            ax.set_yticks([])
            ax.text(0.5, 0.5, "No recurrent significant terms", ha="center", va="center", transform=ax.transAxes)
        else:
            module_data["term_id"] = module_data["term_id"].astype(str)
            color = -np.log10(np.maximum(module_data["fdr"].to_numpy(dtype=float), 1e-300))
            size = module_data["precision"].fillna(0.0).map(gene_ratio_size).to_numpy(dtype=float)
            scatter = ax.scatter(
                module_data["group"].map(x_lookup),
                module_data["term_id"].map(y_lookup),
                c=color,
                s=size,
                cmap=SIGNIFICANCE_CMAP,
                norm=norm,
                edgecolor="#333333",
                linewidth=0.35,
            )
            ax.set_yticks(range(len(term_ids)))
            ax.set_ylim(-TERM_EDGE_PADDING, len(term_ids) - 1 + TERM_EDGE_PADDING)
            labels = [
                textwrap.fill(str(value), width=48)
                for value in terms.sort_values("selection_rank", ascending=False)["term"]
            ]
            ax.set_yticklabels(labels, fontsize=8)
        ax.grid(color="#dddddd", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle("P22 cross-modality %s enrichment" % source, fontsize=12, fontweight="bold")
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=SIGNIFICANCE_CMAP)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=list(axes), fraction=0.018, pad=0.025)
    colorbar_label = "-log10(FDR)"
    if color_capped:
        colorbar_label += "\n(capped at %g)" % float(color_cap)
    colorbar.set_label(colorbar_label)
    gene_ratio_levels = [0.10, 0.20, 0.30]
    size_handles = [
        axes[0].scatter(
            [], [],
            s=gene_ratio_size(value),
            facecolor="#79A7BE",
            edgecolor="#333333",
            linewidth=0.35,
            label="%.2f" % value,
        )
        for value in gene_ratio_levels
    ]
    fig.legend(
        handles=size_handles,
        title="Gene ratio",
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        frameon=False,
        labelspacing=1.0,
    )
    fig.subplots_adjust(left=0.34, right=0.88, top=0.94, bottom=0.055, hspace=0.85)
    stem = "%s_go_bp_enrichment" % stem_prefix if source == "GO:BP" else "%s_%s_enrichment" % (
        stem_prefix, source.lower().replace(":", "_")
    )
    paths = []
    for fmt in formats:
        path = output_dir / (stem + "." + str(fmt).lower())
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_compact_source_layout(frame, group_reference, source):
    data = frame[frame["source"] == source].copy()
    rows = []
    module_bands = []
    cursor = 0.0
    for module_index, module in enumerate(MODULE_ORDER):
        terms = (
            data[data["module"].astype(str) == module][
                ["term_id", "term", "selection_rank"]
            ]
            .drop_duplicates()
            .sort_values(["selection_rank", "term"], kind="stable")
        )
        reference = group_reference[group_reference["module"].astype(str) == module][
            ["module_label", "dataset_label", "cluster"]
        ].drop_duplicates().copy()
        reference["dataset_order"] = reference["dataset_label"].map(DATASET_ORDER).fillna(len(DATASET_ORDER))
        reference = reference.sort_values(["dataset_order", "cluster"], kind="stable")
        module_label = (
            str(reference["module_label"].iloc[0])
            if not reference.empty
            else module.replace("_", " ").title()
        )
        cluster_text = " / ".join(
            "%s C%s" % (row.dataset_label, row.cluster)
            for row in reference.itertuples(index=False)
        )
        header = "%s — %s" % (module_label, cluster_text) if cluster_text else module_label
        header_y = cursor
        cursor += 0.72
        first_term_y = cursor
        if terms.empty:
            rows.append(
                {
                    "module": module,
                    "term_id": "__no_term__%s_%s" % (source, module),
                    "term": "No recurrent significant term",
                    "y": cursor,
                    "placeholder": True,
                }
            )
            cursor += 1.0
        else:
            for term in terms.itertuples(index=False):
                rows.append(
                    {
                        "module": module,
                        "term_id": str(term.term_id),
                        "term": str(term.term),
                        "y": cursor,
                        "placeholder": False,
                    }
                )
                cursor += 1.0
        last_term_y = cursor - 1.0
        module_bands.append(
            {
                "module": module,
                "header": header,
                "header_y": header_y,
                "first_term_y": first_term_y,
                "last_term_y": last_term_y,
                "band_start": header_y - 0.30,
                "band_end": last_term_y + 0.48,
                "color": MODULE_COLORS[module],
                "shade": "#F5F7F8" if module_index % 2 == 0 else "#FFFFFF",
            }
        )
        cursor += 0.48
    return data, pd.DataFrame.from_records(rows), module_bands, cursor


def plot_compact_source_axis(ax, frame, group_reference, source, norm, panel_label):
    data, layout, module_bands, total_height = build_compact_source_layout(
        frame, group_reference, source
    )
    y_lookup = {
        (str(row.module), str(row.term_id)): float(row.y)
        for row in layout.itertuples(index=False)
        if not bool(row.placeholder)
    }
    x_lookup = {"ATAC": 0, "H3K27ac": 1}

    for band in module_bands:
        ax.axhspan(
            float(band["band_start"]),
            float(band["band_end"]),
            color=str(band["shade"]),
            zorder=0,
        )
        ax.vlines(
            -0.57,
            float(band["first_term_y"]) - 0.38,
            float(band["last_term_y"]) + 0.38,
            color=str(band["color"]),
            linewidth=2.4,
            zorder=4,
        )
        ax.text(
            -0.56,
            float(band["header_y"]),
            str(band["header"]),
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=str(band["color"]),
            zorder=5,
        )

    for x in x_lookup.values():
        ax.axvline(x, color="#D9DDE0", linewidth=0.7, zorder=1)
    for y in layout["y"]:
        ax.axhline(float(y), color="#E2E5E7", linewidth=0.6, zorder=1)

    if not data.empty:
        plot_data = data[data["dataset_label"].isin(x_lookup)].copy()
        plot_data["layout_y"] = plot_data.apply(
            lambda row: y_lookup[(str(row["module"]), str(row["term_id"]))], axis=1
        )
        color_values = -np.log10(np.maximum(plot_data["fdr"].to_numpy(dtype=float), 1e-300))
        sizes = plot_data["precision"].fillna(0.0).map(gene_ratio_size).to_numpy(dtype=float)
        ax.scatter(
            plot_data["dataset_label"].map(x_lookup),
            plot_data["layout_y"],
            c=color_values,
            s=sizes,
            cmap=SIGNIFICANCE_CMAP,
            norm=norm,
            edgecolor="#29343A",
            linewidth=0.4,
            zorder=3,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ATAC", "H3K27ac"], fontsize=8.0)
    ax.set_xlim(-0.62, 1.62)
    ax.set_yticks(layout["y"].to_numpy(dtype=float))
    labels = [textwrap.fill(str(value), width=43) for value in layout["term"]]
    tick_texts = ax.set_yticklabels(labels, fontsize=7.0)
    for text, placeholder in zip(tick_texts, layout["placeholder"]):
        if bool(placeholder):
            text.set_color("#777777")
            text.set_fontstyle("italic")
    ax.set_ylim(total_height - 0.10, -0.42)
    ax.tick_params(axis="both", length=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_title = "GO Biological Process" if source == "GO:BP" else "KEGG pathways"
    ax.set_title(panel_title, loc="left", fontsize=10.0, fontweight="bold", pad=8)
    ax.text(
        -0.08,
        1.025,
        panel_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12.0,
        fontweight="bold",
    )
    return total_height


def plot_paired_combined(
    frame,
    group_reference,
    output_dir,
    formats,
    dpi,
    fdr_cutoff,
    color_cap,
    width_cm,
    height_cm,
):
    paired = frame[frame["dataset_label"].isin(["ATAC", "H3K27ac"])].copy()
    if paired.empty:
        print("[P22 enrichment comparison] no paired terms available for combined plot")
        return []
    norm, color_capped = significance_norm(paired, fdr_cutoff, color_cap)
    _, _, _, go_height = build_compact_source_layout(paired, group_reference, "GO:BP")
    _, _, _, kegg_height = build_compact_source_layout(paired, group_reference, "KEGG")

    fig = plt.figure(
        figsize=(float(width_cm) / CM_PER_INCH, float(height_cm) / CM_PER_INCH),
        dpi=150,
    )
    grid = fig.add_gridspec(
        2,
        1,
        left=0.42,
        right=0.82,
        top=0.925,
        bottom=0.085,
        hspace=0.18,
        height_ratios=[go_height, kegg_height],
    )
    ax_go = fig.add_subplot(grid[0, 0])
    ax_kegg = fig.add_subplot(grid[1, 0])
    plot_compact_source_axis(ax_go, paired, group_reference, "GO:BP", norm, "a")
    plot_compact_source_axis(ax_kegg, paired, group_reference, "KEGG", norm, "b")
    fig.suptitle(
        "P22 paired cross-modality RNA functional enrichment",
        x=0.42,
        y=0.975,
        ha="left",
        fontsize=12.0,
        fontweight="bold",
    )

    color_axis = fig.add_axes([0.875, 0.56, 0.018, 0.27])
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=SIGNIFICANCE_CMAP)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, cax=color_axis)
    colorbar_label = "-log10(FDR)"
    if color_capped:
        colorbar_label += "\n(capped at %g)" % float(color_cap)
    colorbar.set_label(colorbar_label, fontsize=7.2)
    colorbar.ax.tick_params(labelsize=7.0, length=2)
    colorbar.outline.set_linewidth(0.6)

    gene_ratio_levels = [0.10, 0.20, 0.30]
    size_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=math.sqrt(gene_ratio_size(value)),
            markerfacecolor="#79A7BE",
            markeredgecolor="#29343A",
            markeredgewidth=0.4,
            label="%.2f" % value,
        )
        for value in gene_ratio_levels
    ]
    fig.legend(
        handles=size_handles,
        title="Gene ratio\n(overlap/query)",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.845, 0.47),
        borderaxespad=0.0,
        labelspacing=0.9,
    )
    fig.text(
        0.42,
        0.025,
        "Dot color: enrichment significance (light to dark). Dot size: marker-gene ratio.\n"
        "Cluster identifiers are reported in each module header; blank cells are not significant.",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#444444",
    )

    paths = []
    for fmt in formats:
        path = output_dir / ("p22_paired_go_kegg_enrichment.%s" % str(fmt).lower())
        save_kwargs = {"facecolor": "white"}
        if str(fmt).lower() in {"png", "svg"}:
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(path, **save_kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    args = parse_args()
    configure_plot_style()
    if bool(args.h3k4me3_results) != bool(args.h3k27me3_results):
        raise ValueError("Provide both --h3k4me3-results and --h3k27me3-results, or neither.")
    if not 0.0 < float(args.fdr_cutoff) <= 1.0:
        raise ValueError("--fdr-cutoff must be in (0, 1].")
    if float(args.combined_width_cm) <= 0.0 or float(args.combined_height_cm) <= 0.0:
        raise ValueError("Combined figure width and height must be positive.")
    if int(args.dpi) <= 0:
        raise ValueError("--dpi must be positive.")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = [
        read_results(args.atac_results, "ATAC"),
        read_results(args.h3k27ac_results, "H3K27ac"),
    ]
    if args.h3k4me3_results:
        inputs.extend([
            read_results(args.h3k4me3_results, "H3K4me3"),
            read_results(args.h3k27me3_results, "H3K27me3"),
        ])
    if not 1 <= int(args.min_datasets_per_term) <= len(inputs):
        raise ValueError("--min-datasets-per-term must be between 1 and the number of input datasets.")
    combined = pd.concat(inputs, ignore_index=True)
    selected = select_terms(combined, args)
    stem_prefix = "p22_paired" if len(inputs) == 2 else "p22_four_modal"
    selected_path = output_dir / (stem_prefix + "_enrichment_plot_data.csv")
    selected.to_csv(selected_path, index=False)
    figures = []
    for source in ["GO:BP", "KEGG"]:
        figures.extend(
            plot_source(
                selected,
                combined,
                source,
                output_dir,
                args.formats,
                args.dpi,
                stem_prefix,
                args.fdr_cutoff,
                args.color_cap,
            )
        )
    if len(inputs) == 2:
        figures.extend(
            plot_paired_combined(
                selected,
                combined,
                output_dir,
                args.formats,
                args.dpi,
                args.fdr_cutoff,
                args.color_cap,
                args.combined_width_cm,
                args.combined_height_cm,
            )
        )
    print("[P22 enrichment comparison] datasets=%d min_datasets_per_term=%d selected_rows=%d" % (
        len(inputs), int(args.min_datasets_per_term), selected.shape[0]
    ))
    print("  plot data: %s" % selected_path)
    for path in figures:
        print("  figure   : %s" % path)


if __name__ == "__main__":
    main()
