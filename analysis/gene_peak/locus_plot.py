"""Draw target-gene cis peak-gene locus plots from downstream RNA-ATAC links.

The script is intentionally read-only with respect to model outputs: it consumes
the CSV files produced by run_rna_atac_regulatory_downstream.py and creates
MultiGATE-style locus figures for selected target genes.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REG_DIR = "results/MOUSE_BRAIN_P22/mouse_brain_ATAC/downstream_analyses_epoch1800_mEEE_pca25/rna_atac_regulatory_relaxed"
DEFAULT_OUTPUT_DIR = "results/MOUSE_BRAIN_P22/mouse_brain_ATAC/downstream_analyses_epoch1800_mEEE_pca25/target_gene_peak_locus_plot"

TITLE_FONT_SIZE = 15
TRACK_FONT_SIZE = 12
AXIS_FONT_SIZE = 12
TICK_FONT_SIZE = 10
ANNOTATION_FONT_SIZE = 10
TARGET_GENE_FONT_SIZE = 11
GENE_FONT_SIZE = 10
PEAK_LABEL_FONT_SIZE = 9

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": TICK_FONT_SIZE,
        "pdf.fonttype": 42,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="Draw target-gene peak-gene cis-regulatory locus plots.")
    parser.add_argument("--regulatory-dir", default=DEFAULT_REG_DIR, help="Directory containing regulatory downstream CSV outputs.")
    parser.add_argument("--links-csv", default=None, help="Significant peak-gene links CSV. Defaults to regulatory-dir/significant_peak_gene_links.csv.")
    parser.add_argument("--all-links-csv", default=None, help="All peak-gene correlations CSV for fallback/candidate links.")
    parser.add_argument("--gene-tss-csv", default=None, help="Gene annotation CSV from regulatory downstream output.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-genes", nargs="+", default=None, help="Optional genes to plot. If omitted, use --auto-top-genes.")
    parser.add_argument("--auto-top-genes", type=int, default=0, help="Also add top N genes from significant links ranked by link count and max absolute score.")
    parser.add_argument("--window-size", type=int, default=150000, help="Locus window around target TSS in bp.")
    parser.add_argument("--region", default=None, help="Optional fixed locus region such as chr1:72315000-72350000. Best used with one target gene.")
    parser.add_argument("--tick-step-kb", type=float, default=25.0, help="Major x-axis tick spacing in kb.")
    parser.add_argument("--max-nearby-genes", type=int, default=14, help="Maximum gene models to label in the locus track.")
    parser.add_argument("--max-links-per-gene", type=int, default=24)
    parser.add_argument("--candidate-top-n", type=int, default=24, help="Fallback candidate links when a gene has no significant links.")
    parser.add_argument("--fig-width", type=float, default=14.0, help="Figure width in inches. Use 18-22 for expanded locus plots.")
    parser.add_argument("--base-height", type=float, default=4.8, help="Minimum fixed figure height in inches.")
    parser.add_argument("--gene-row-height", type=float, default=0.32, help="Height allowance per configured gene-model row; the maximum row count is used so every PDF has the same height.")
    parser.add_argument("--arc-height-scale", type=float, default=1.0, help="Multiplier for score arc heights.")
    parser.add_argument("--label-every-peak", action="store_true", help="Annotate every linked peak with its distance to TSS.")
    parser.add_argument("--include-candidate-if-no-significant", action="store_true", default=True)
    parser.add_argument("--score-column", default="pearson_r", help="Link score column used for arc height/color scaling.")
    parser.add_argument("--fdr-column", default="fdr")
    parser.add_argument("--fdr-cutoff", type=float, default=0.05)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve_path(path_value, base_dir=PROJECT_ROOT):
    if path_value is None:
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def format_kb(value, _pos=None):
    return "%.0f" % (float(value) / 1000.0)


def parse_region(region):
    if not region:
        return None
    match = re.match(r"^([^:]+):(\d+)[\-\.\.](\d+)$", str(region).replace(",", ""))
    if not match:
        raise ValueError("--region must look like chr1:72315000-72350000")
    chrom = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))
    if end < start:
        start, end = end, start
    return chrom, start, end


def clean_gene_name(value):
    return str(value).strip()


def read_csv_or_empty(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_link_columns(df):
    if df.empty:
        return df.copy()
    out = df.copy()
    rename = {
        "Peak": "peak",
        "Target_Gene": "gene",
        "Gene": "gene",
        "Peak_Chrom": "peak_chrom",
        "Peak_Start": "peak_start",
        "Peak_End": "peak_end",
        "Distance_to_TSS": "distance_to_tss",
        "Spearman_Corr": "spearman_r",
        "P_value": "pearson_p",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    required = ["peak", "gene", "peak_chrom", "peak_start", "peak_end", "distance_to_tss"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError("Link table is missing required columns: %s" % ", ".join(missing))
    for col in ["peak_start", "peak_end", "peak_center", "gene_tss", "distance_to_tss", "pearson_r", "spearman_r", "fdr"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "peak_center" not in out.columns:
        out["peak_center"] = ((out["peak_start"] + out["peak_end"]) / 2.0).round().astype("Int64")
    return out


def normalize_gene_columns(df):
    if df.empty:
        raise ValueError("Gene annotation table is empty")
    out = df.copy()
    rename = {"Gene": "gene", "gene_name": "gene", "Chromosome": "chrom", "TSS": "tss", "Start": "start", "End": "end", "Strand": "strand"}
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    required = ["gene", "chrom", "tss", "start", "end"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError("Gene annotation table is missing required columns: %s" % ", ".join(missing))
    for col in ["tss", "start", "end"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "strand" not in out.columns:
        out["strand"] = "."
    out = out.dropna(subset=["chrom", "tss", "start", "end"])
    return out


def select_target_gene_row(genes_df, target_gene):
    target_lower = target_gene.lower()
    rows = genes_df[genes_df["gene"].astype(str).str.lower() == target_lower].copy()
    if rows.empty:
        return None
    rows["span"] = (rows["end"] - rows["start"]).abs()
    return rows.sort_values("span", ascending=False).iloc[0].to_dict()


def select_links_for_gene(target_gene, significant, all_links, args):
    target_lower = target_gene.lower()
    sig = significant[significant["gene"].astype(str).str.lower() == target_lower].copy() if not significant.empty else pd.DataFrame()
    if not sig.empty:
        sig["link_source"] = "significant"
        sig["abs_score"] = pd.to_numeric(sig.get(args.score_column, sig.get("pearson_r", 0.0)), errors="coerce").abs()
        return sig.sort_values(["abs_score", args.fdr_column if args.fdr_column in sig.columns else "abs_score"], ascending=[False, True]).head(int(args.max_links_per_gene)).copy()
    if not args.include_candidate_if_no_significant or all_links.empty:
        return pd.DataFrame()
    cand = all_links[all_links["gene"].astype(str).str.lower() == target_lower].copy()
    if cand.empty:
        return cand
    cand["link_source"] = "candidate"
    cand["abs_score"] = pd.to_numeric(cand.get(args.score_column, cand.get("pearson_r", 0.0)), errors="coerce").abs()
    return cand.sort_values(["abs_score", args.fdr_column if args.fdr_column in cand.columns else "abs_score"], ascending=[False, True]).head(int(args.candidate_top_n)).copy()


def auto_select_target_genes(significant, args):
    if int(args.auto_top_genes) <= 0 or significant.empty:
        return []
    df = significant.copy()
    score = pd.to_numeric(df.get(args.score_column, df.get("pearson_r", pd.Series(dtype=float))), errors="coerce").abs()
    df["abs_score"] = score
    ranked = (
        df.groupby("gene", observed=False)
        .agg(n_links=("peak", "count"), max_abs_score=("abs_score", "max"), min_fdr=(args.fdr_column, "min") if args.fdr_column in df.columns else ("abs_score", "min"))
        .sort_values(["n_links", "max_abs_score", "min_fdr"], ascending=[False, False, True])
    )
    return [str(gene) for gene in ranked.head(int(args.auto_top_genes)).index]


def nearby_genes(genes_df, chrom, start, end, target_gene, max_genes):
    nearby = genes_df[(genes_df["chrom"].astype(str) == str(chrom)) & (genes_df["end"] >= start) & (genes_df["start"] <= end)].copy()
    nearby["is_target"] = nearby["gene"].astype(str).str.lower() == target_gene.lower()
    nearby["visible_span"] = np.minimum(nearby["end"], end) - np.maximum(nearby["start"], start)
    nearby = nearby.sort_values(["is_target", "start"], ascending=[False, True])
    target = nearby[nearby["is_target"]]
    others = nearby[~nearby["is_target"]].sort_values(["visible_span", "start"], ascending=[False, True])
    return pd.concat([target, others.head(max(0, int(max_genes) - target.shape[0]))], ignore_index=True).sort_values(["start", "end"])


def make_arc_patch(x0, x1, y0, height, color, linewidth, linestyle="solid", alpha=0.85):
    mid = (float(x0) + float(x1)) / 2.0
    verts = [(float(x0), y0), (mid, y0 + height), (float(x1), y0)]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    return patches.PathPatch(MplPath(verts, codes), facecolor="none", edgecolor=color, lw=linewidth, linestyle=linestyle, alpha=alpha)


def draw_gene_model(ax, row, y, target_gene, x_start, x_end, label_offset=0.16):
    start = max(float(row["start"]), float(x_start))
    end = min(float(row["end"]), float(x_end))
    if end < start:
        start, end = end, start
    if end <= start:
        return
    gene = str(row["gene"])
    is_target = gene.lower() == target_gene.lower()
    color = "#d62728" if is_target else "#1f77b4"
    height = 0.18 if is_target else 0.12
    ax.plot([start, end], [y, y], color=color, lw=1.0, alpha=0.85)
    ax.add_patch(patches.Rectangle((start, y - height / 2.0), max(end - start, 80), height, facecolor=color, edgecolor="none", alpha=0.85))
    strand = str(row.get("strand", "."))
    direction = 1 if strand != "-" else -1
    arrow_x = end if direction > 0 else start
    ax.annotate("", xy=(arrow_x, y), xytext=(arrow_x - direction * max((end - start) * 0.18, 500), y), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8}, annotation_clip=True, clip_on=True)
    ax.text((start + end) / 2.0, y + label_offset, gene, color=color, ha="center", va="bottom", fontsize=TARGET_GENE_FONT_SIZE if is_target else GENE_FONT_SIZE, fontstyle="italic" if is_target else "normal", clip_on=True)


def add_track_label(ax, label):
    ax.text(0.005, 0.5, label, transform=ax.transAxes, ha="left", va="center", fontsize=TRACK_FONT_SIZE)


def fixed_figure_height(args):
    configured_rows = max(int(args.max_nearby_genes), 1)
    return max(float(args.base_height), 3.6 + float(args.gene_row_height) * configured_rows)


def draw_locus(target_gene, target_row, links, genes_df, output_dir, args):
    fixed_region = parse_region(args.region)
    chrom = str(target_row["chrom"])
    tss = int(target_row["tss"])
    window = int(args.window_size)
    if fixed_region is not None:
        region_chrom, x_start, x_end = fixed_region
        if region_chrom != chrom:
            raise ValueError("Region chromosome %s does not match %s for target gene %s" % (region_chrom, chrom, target_gene))
    else:
        x_start = tss - window
        x_end = tss + window
    if not links.empty and fixed_region is None:
        link_min = int(np.nanmin(pd.to_numeric(links["peak_start"], errors="coerce")))
        link_max = int(np.nanmax(pd.to_numeric(links["peak_end"], errors="coerce")))
        x_start = min(x_start, link_min - int(0.08 * window))
        x_end = max(x_end, link_max + int(0.08 * window))
    if not links.empty:
        links = links[(pd.to_numeric(links["peak_end"], errors="coerce") >= x_start) & (pd.to_numeric(links["peak_start"], errors="coerce") <= x_end)].copy()
    nearby = nearby_genes(genes_df, chrom, x_start, x_end, target_gene, args.max_nearby_genes)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(float(args.fig_width), fixed_figure_height(args)),
        dpi=int(args.dpi),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 0.9, 0.55, 1.15], "hspace": 0.05},
    )
    score_ax, evidence_ax, peak_ax, gene_ax = axes
    title_gene = target_gene if links.empty else str(links.iloc[0].get("gene", target_gene))
    score_ax.set_title("Target gene %s" % title_gene, fontsize=TITLE_FONT_SIZE, fontstyle="italic", pad=8)

    for ax in axes:
        ax.set_xlim(x_start, x_end)
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
    score_ax.spines["bottom"].set_visible(False)
    evidence_ax.spines["bottom"].set_visible(False)
    peak_ax.spines["bottom"].set_visible(False)

    score_values = pd.to_numeric(links.get(args.score_column, links.get("pearson_r", pd.Series(dtype=float))), errors="coerce").abs() if not links.empty else pd.Series(dtype=float)
    max_score = max(float(score_values.max()) if len(score_values) else 0.0, 1e-9)
    threshold = 0.05
    score_ax.axhline(threshold, color="#bdbdbd", lw=0.8, ls="--")
    score_ax.text(x_start, threshold + 0.015, "threshold", ha="left", va="bottom", fontsize=ANNOTATION_FONT_SIZE)
    score_ax.set_ylim(-0.02, max(0.18, max_score * 1.45))
    evidence_ax.set_ylim(-0.02, 1.05)
    peak_ax.set_ylim(-0.1, 1.0)
    gene_ax.set_ylim(-0.1, max(1.0, 0.28 * max(len(nearby), 1)))

    score_ax.axvline(tss, color="#d62728", lw=1.0, alpha=0.65)
    evidence_ax.axvline(tss, color="#d62728", lw=0.9, alpha=0.45)
    peak_ax.axvline(tss, color="#d62728", lw=0.8, alpha=0.35)
    gene_ax.axvline(tss, color="#d62728", lw=0.8, alpha=0.35)

    if not links.empty:
        plot_links = links.copy()
        plot_links["abs_score"] = score_values.to_numpy(dtype=float)
        plot_links = plot_links.sort_values("abs_score", ascending=True)
        for _, row in plot_links.iterrows():
            center = float(row.get("peak_center", (row["peak_start"] + row["peak_end"]) / 2.0))
            score = abs(float(row.get(args.score_column, row.get("pearson_r", 0.0))))
            fdr = float(row.get(args.fdr_column, np.nan)) if pd.notna(row.get(args.fdr_column, np.nan)) else np.nan
            is_sig = str(row.get("link_source", "significant")) == "significant" or (np.isfinite(fdr) and fdr <= float(args.fdr_cutoff))
            color = "#d95f5f" if is_sig else "#8a8a8a"
            linestyle = "solid" if is_sig else (0, (3, 2))
            linewidth = 0.75 + 2.4 * min(score / max_score, 1.0)
            score_ax.add_patch(make_arc_patch(center, tss, 0.0, max(score * float(args.arc_height_scale), 0.004), color, linewidth, linestyle=linestyle, alpha=0.88))
            evidence_height = 0.18 + 0.72 * min((-np.log10(max(fdr, 1e-300)) / 40.0) if np.isfinite(fdr) else 0.1, 1.0)
            evidence_ax.add_patch(make_arc_patch(center, tss, 0.0, evidence_height, "#1b7f3a" if is_sig else "#9a9a9a", max(0.5, linewidth * 0.65), linestyle=linestyle, alpha=0.78))
            pstart = float(row["peak_start"])
            pend = float(row["peak_end"])
            peak_ax.add_patch(patches.Rectangle((pstart, 0.35), max(pend - pstart, 120), 0.3, facecolor="#1f77b4", edgecolor="none", alpha=0.88 if is_sig else 0.45))
            if args.label_every_peak:
                dist_kb = float(row.get("distance_to_tss", center - tss)) / 1000.0
                peak_ax.text(center, 0.72, "%+.1f kb" % dist_kb, ha="center", va="bottom", fontsize=PEAK_LABEL_FONT_SIZE, rotation=45, color="#4d4d4d", clip_on=True)

    y_positions = np.linspace(gene_ax.get_ylim()[1] - 0.25, 0.25, max(len(nearby), 1))
    for idx, (y, (_, row)) in enumerate(zip(y_positions, nearby.iterrows())):
        draw_gene_model(gene_ax, row, y, target_gene, x_start, x_end, label_offset=0.13 + 0.08 * (idx % 2))

    add_track_label(score_ax, "Score")
    add_track_label(evidence_ax, "RNA-ATAC")
    add_track_label(peak_ax, "ATAC")
    add_track_label(gene_ax, "Genes")
    gene_ax.xaxis.set_major_formatter(FuncFormatter(format_kb))
    if float(args.tick_step_kb) > 0:
        from matplotlib.ticker import MultipleLocator

        gene_ax.xaxis.set_major_locator(MultipleLocator(float(args.tick_step_kb) * 1000.0))
    gene_ax.set_xlabel("%s position (kb)" % chrom, fontsize=AXIS_FONT_SIZE)
    gene_ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    for ax in [score_ax, evidence_ax, peak_ax]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    fig.tight_layout()

    output_paths = []
    safe_gene = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in target_gene)
    suffix = "candidate" if links.empty or set(links.get("link_source", [])) == {"candidate"} else "significant"
    for fmt in args.formats:
        path = output_dir / ("target_gene_locus_%s_%s.%s" % (safe_gene, suffix, fmt))
        fig.savefig(path, dpi=int(args.dpi))
        output_paths.append(str(path))
    plt.close(fig)
    return output_paths


def main():
    args = parse_args()
    regulatory_dir = resolve_path(args.regulatory_dir)
    links_path = resolve_path(args.links_csv) if args.links_csv else regulatory_dir / "significant_peak_gene_links.csv"
    all_links_path = resolve_path(args.all_links_csv) if args.all_links_csv else regulatory_dir / "peak_gene_correlations.csv"
    gene_tss_path = resolve_path(args.gene_tss_csv) if args.gene_tss_csv else regulatory_dir / "gene_tss_annotation.csv"
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    significant = normalize_link_columns(read_csv_or_empty(links_path))
    all_links = normalize_link_columns(read_csv_or_empty(all_links_path))
    genes_df = normalize_gene_columns(read_csv_or_empty(gene_tss_path))

    summary_rows = []
    all_selected = []
    figures = []
    target_genes = [clean_gene_name(g) for g in (args.target_genes or []) if clean_gene_name(g)]
    for gene in auto_select_target_genes(significant, args):
        if gene.lower() not in {g.lower() for g in target_genes}:
            target_genes.append(gene)
    if not target_genes:
        raise ValueError("No target genes were provided or auto-selected. Pass --target-genes or --auto-top-genes with significant links.")

    for target_gene in target_genes:
        target_row = select_target_gene_row(genes_df, target_gene)
        if target_row is None:
            summary_rows.append({"target_gene": target_gene, "status": "missing_gene_annotation", "n_links": 0, "n_significant_links": 0, "n_candidate_links": 0})
            continue
        links = select_links_for_gene(target_gene, significant, all_links, args)
        n_sig = int((links.get("link_source", pd.Series(dtype=str)).astype(str) == "significant").sum()) if not links.empty else 0
        n_cand = int((links.get("link_source", pd.Series(dtype=str)).astype(str) == "candidate").sum()) if not links.empty else 0
        status = "ok" if n_sig > 0 else ("candidate_only" if n_cand > 0 else "no_links")
        if not links.empty:
            links = links.copy()
            links.insert(0, "target_gene_requested", target_gene)
            all_selected.append(links)
        figures.extend(draw_locus(target_gene, target_row, links, genes_df, output_dir, args))
        summary_rows.append({"target_gene": target_gene, "status": status, "chrom": target_row["chrom"], "tss": int(target_row["tss"]), "n_links": int(links.shape[0]), "n_significant_links": n_sig, "n_candidate_links": n_cand})

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "target_gene_locus_summary.csv"
    summary.to_csv(summary_path, index=False)
    selected_path = output_dir / "target_gene_locus_links.csv"
    if all_selected:
        pd.concat(all_selected, ignore_index=True).to_csv(selected_path, index=False)
    else:
        pd.DataFrame().to_csv(selected_path, index=False)
    manifest = {
        "regulatory_dir": str(regulatory_dir),
        "links_csv": str(links_path),
        "all_links_csv": str(all_links_path),
        "gene_tss_csv": str(gene_tss_path),
        "output_dir": str(output_dir),
        "target_genes": target_genes,
        "window_size": int(args.window_size),
        "figure_width_inches": float(args.fig_width),
        "figure_height_inches": fixed_figure_height(args),
        "dpi": int(args.dpi),
        "font_sizes_points": {
            "title": TITLE_FONT_SIZE,
            "track": TRACK_FONT_SIZE,
            "axis": AXIS_FONT_SIZE,
            "tick": TICK_FONT_SIZE,
            "annotation": ANNOTATION_FONT_SIZE,
            "target_gene": TARGET_GENE_FONT_SIZE,
            "gene": GENE_FONT_SIZE,
            "peak_label": PEAK_LABEL_FONT_SIZE,
        },
        "summary_csv": str(summary_path),
        "selected_links_csv": str(selected_path),
        "figures": figures,
    }
    manifest_path = output_dir / "target_gene_locus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[target-gene locus] complete genes=%d figures=%d" % (len(summary_rows), len(figures)))
    print("  summary: %s" % summary_path)
    print("  manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()
