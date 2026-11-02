"""RNA-ATAC regulatory downstream analysis for MISAR epoch-1300.

This script links peaks to nearby gene TSS positions, estimates peak-gene
correlations, and summarizes cluster/residual/gate associated accessibility. It
uses only original raw h5ad inputs plus standard downstream SSR spot scores.
"""

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

try:
    from scipy import sparse
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    sparse = None
    scipy_stats = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/misar/e18_5_s1.yaml"
DEFAULT_SCORE_CSV = "results/MISAR/e18_5_s1/downstream_analyses_epoch1300/ssr_core_downstream/ssr_core_spot_scores.csv"
DEFAULT_OUTPUT_DIR = "results/MISAR/e18_5_s1/downstream_analyses_epoch1300/rna_atac_regulatory"


def parse_args():
    parser = argparse.ArgumentParser(description="Run RNA-ATAC regulatory downstream analysis.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--rna-h5ad", default=None)
    parser.add_argument("--atac-h5ad", default=None)
    parser.add_argument("--gtf", default=None)
    parser.add_argument("--ssr-score-csv", default=DEFAULT_SCORE_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tss-window", type=int, default=None)
    parser.add_argument("--atac-mode", choices=["raw", "log1p", "binary"], default="log1p")
    parser.add_argument("--max-links", type=int, default=200000)
    parser.add_argument("--min-peak-detection", type=float, default=0.01)
    parser.add_argument("--min-gene-detection", type=float, default=0.01)
    parser.add_argument("--fdr-cutoff", type=float, default=0.05)
    parser.add_argument("--min-abs-pearson", type=float, default=0.2)
    parser.add_argument("--max-peaks-for-da", type=int, default=20000)
    parser.add_argument("--max-variable-peaks", type=int, default=10000)
    parser.add_argument("--motif-annotation-csv", default=None)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_path(path_value, base_dir=PROJECT_ROOT):
    if path_value is None:
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_inputs(config, args):
    data_cfg = config.get("data", {}) or {}
    raw_path = data_cfg.get("raw_path")
    files = data_cfg.get("files", {}) or {}
    raw_dir = resolve_path(raw_path) if raw_path else PROJECT_ROOT
    rna = resolve_path(args.rna_h5ad) if args.rna_h5ad else raw_dir / str(files.get("rna_h5ad", "adata_RNA.h5ad"))
    atac = resolve_path(args.atac_h5ad) if args.atac_h5ad else raw_dir / str(files.get("atac_h5ad", "adata_Peak.h5ad"))
    if args.gtf:
        gtf = resolve_path(args.gtf)
    else:
        gtf_raw = files.get("gtf")
        gtf = (raw_dir / str(gtf_raw)).resolve() if gtf_raw else None
        if gtf is None or not gtf.exists():
            fallback = PROJECT_ROOT / "data" / "raw" / "gencode.vM23.annotation.gtf.gz"
            gtf = fallback if fallback.exists() else gtf
    return rna, atac, gtf


def parse_peak_name(name):
    text = str(name)
    match = re.match(r"^(chr[\w.]+)[\.\:\-_](\d+)[\.\:\-_](\d+)$", text)
    if match is None:
        match = re.match(r"^([^\.\:\-_]+)[\.\:\-_](\d+)[\.\:\-_](\d+)$", text)
    if match is None:
        return None, np.nan, np.nan, np.nan
    chrom = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))
    if end < start:
        start, end = end, start
    return chrom, start, end, int((start + end) / 2)


def parse_gtf_attributes(text):
    attrs = {}
    for item in text.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        if " " in item:
            key, value = item.split(" ", 1)
            attrs[key] = value.strip().strip('"')
        elif "=" in item:
            key, value = item.split("=", 1)
            attrs[key] = value.strip().strip('"')
    return attrs


def parse_gtf(gtf_path):
    rows = []
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = parse_gtf_attributes(parts[8])
            gene_id = attrs.get("gene_id", "")
            gene = attrs.get("gene_name") or gene_id
            if not gene:
                continue
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            tss = start if strand != "-" else end
            rows.append({"gene": gene, "gene_id": gene_id, "chrom": parts[0], "tss": int(tss), "start": start, "end": end, "strand": strand})
    return pd.DataFrame(rows).drop_duplicates(subset=["gene", "chrom", "tss"])


def to_dense_vector(x):
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray()).reshape(-1)
    return np.asarray(x).reshape(-1)


def align_anndata(rna, atac, score_df):
    warnings = []
    if set(rna.obs_names.astype(str)) == set(atac.obs_names.astype(str)):
        atac = atac[rna.obs_names.astype(str)].copy()
    elif rna.n_obs == atac.n_obs:
        warnings.append("RNA and ATAC obs_names differ; using row-order fallback")
        atac = atac.copy()
        atac.obs_names = rna.obs_names
    else:
        raise ValueError("RNA/ATAC spot mismatch: %d vs %d" % (rna.n_obs, atac.n_obs))
    if "spot" in score_df.columns and set(rna.obs_names.astype(str)) == set(score_df["spot"].astype(str)):
        score_df = score_df.copy()
        score_df.index = score_df["spot"].astype(str)
        score_df = score_df.loc[rna.obs_names.astype(str)].copy()
    elif score_df.shape[0] == rna.n_obs:
        warnings.append("SSR score spots differ from RNA obs_names; using row-order fallback")
        score_df = score_df.copy()
        score_df.index = rna.obs_names
    else:
        raise ValueError("SSR score spot count does not match RNA spots")
    return rna, atac, score_df, warnings


def prepare_rna_matrix(rna):
    ad = rna.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    return ad


def prepare_atac_matrix(atac, mode):
    ad = atac.copy()
    if mode == "binary":
        ad.X = (ad.X > 0).astype(np.float32)
    elif mode == "log1p":
        if hasattr(ad.X, "log1p"):
            ad.X = ad.X.copy()
            ad.X.data = np.log1p(ad.X.data)
        else:
            ad.X = np.log1p(np.asarray(ad.X))
    return ad


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.ones(p.shape[0], dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def parse_peaks(atac):
    rows = []
    for idx, peak in enumerate(atac.var_names.astype(str)):
        chrom, start, end, center = parse_peak_name(peak)
        rows.append({"peak": peak, "peak_index": idx, "peak_chrom": chrom, "peak_start": start, "peak_end": end, "peak_center": center})
    return pd.DataFrame(rows)


def build_candidate_links(peaks_df, genes_df, rna, atac, window, max_links, min_peak_detection, min_gene_detection):
    rna_genes = {str(g): i for i, g in enumerate(rna.var_names.astype(str))}
    peak_detection = np.asarray((atac.X > 0).mean(axis=0)).reshape(-1)
    gene_detection = np.asarray((rna.X > 0).mean(axis=0)).reshape(-1)
    valid_genes = genes_df[genes_df["gene"].isin(rna_genes)].copy()
    rows = []
    for chrom, peak_sub in peaks_df.dropna(subset=["peak_chrom", "peak_center"]).groupby("peak_chrom", observed=False):
        gene_sub = valid_genes[valid_genes["chrom"] == chrom].sort_values("tss")
        if gene_sub.empty:
            continue
        tss = gene_sub["tss"].to_numpy(dtype=int)
        for _, peak_row in peak_sub.iterrows():
            pidx = int(peak_row["peak_index"])
            if peak_detection[pidx] < float(min_peak_detection):
                continue
            center = int(peak_row["peak_center"])
            left = np.searchsorted(tss, center - int(window), side="left")
            right = np.searchsorted(tss, center + int(window), side="right")
            for _, gene_row in gene_sub.iloc[left:right].iterrows():
                gidx = rna_genes[str(gene_row["gene"])]
                if gene_detection[gidx] < float(min_gene_detection):
                    continue
                rows.append({
                    "peak": peak_row["peak"], "gene": gene_row["gene"], "gene_id": gene_row["gene_id"],
                    "peak_chrom": chrom, "peak_start": int(peak_row["peak_start"]), "peak_end": int(peak_row["peak_end"]),
                    "peak_center": center, "gene_tss": int(gene_row["tss"]), "distance_to_tss": int(center - int(gene_row["tss"])),
                    "peak_index": pidx, "gene_index": gidx,
                })
                if len(rows) >= int(max_links):
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def vector_cache_get(cache, adata, key, index):
    if key not in cache:
        cache[key] = to_dense_vector(adata.X[:, int(index)])
    return cache[key]


def compute_correlations(links, rna, atac):
    if links.empty:
        return pd.DataFrame(columns=list(links.columns) + ["pearson_r", "pearson_p", "spearman_r", "spearman_p", "fdr"])
    rows = []
    gene_cache = {}
    peak_cache = {}
    for _, row in links.iterrows():
        gene_vec = vector_cache_get(gene_cache, rna, row["gene"], row["gene_index"])
        peak_vec = vector_cache_get(peak_cache, atac, row["peak"], row["peak_index"])
        if np.std(gene_vec) == 0 or np.std(peak_vec) == 0:
            pearson_r = np.nan; pearson_p = np.nan; spearman_r = np.nan; spearman_p = np.nan
        elif scipy_stats is not None:
            pearson_r, pearson_p = scipy_stats.pearsonr(peak_vec, gene_vec)
            spearman_r, spearman_p = scipy_stats.spearmanr(peak_vec, gene_vec)
        else:
            pearson_r = float(np.corrcoef(peak_vec, gene_vec)[0, 1]); pearson_p = np.nan; spearman_r = np.nan; spearman_p = np.nan
        out = row.drop(labels=["peak_index", "gene_index"]).to_dict()
        out.update({"pearson_r": float(pearson_r), "pearson_p": float(pearson_p) if np.isfinite(pearson_p) else np.nan, "spearman_r": float(spearman_r) if np.isfinite(spearman_r) else np.nan, "spearman_p": float(spearman_p) if np.isfinite(spearman_p) else np.nan})
        rows.append(out)
    df = pd.DataFrame(rows)
    df["fdr"] = bh_fdr(df["pearson_p"].to_numpy(dtype=float)) if "pearson_p" in df else []
    return df


def select_variable_peaks(atac, max_peaks):
    x = atac.X
    mean = np.asarray(x.mean(axis=0)).reshape(-1)
    if hasattr(x, "power"):
        second = np.asarray(x.power(2).mean(axis=0)).reshape(-1)
    else:
        second = np.asarray(np.square(x).mean(axis=0)).reshape(-1)
    var = second - mean ** 2
    n = min(int(max_peaks), len(var))
    if n <= 0:
        return np.array([], dtype=int)
    return np.argsort(var)[::-1][:n]


def mean_diff_peaks(atac, labels, positive_label, peak_indices, top_n=500):
    labels = pd.Series(labels).astype(str).to_numpy()
    mask = labels == str(positive_label)
    if mask.sum() == 0 or (~mask).sum() == 0 or len(peak_indices) == 0:
        return pd.DataFrame()
    sub = atac[:, peak_indices]
    a = np.asarray(sub.X[mask].mean(axis=0)).reshape(-1)
    b = np.asarray(sub.X[~mask].mean(axis=0)).reshape(-1)
    det_a = np.asarray((sub.X[mask] > 0).mean(axis=0)).reshape(-1)
    det_b = np.asarray((sub.X[~mask] > 0).mean(axis=0)).reshape(-1)
    diff = a - b
    order = np.argsort(diff)[::-1][:int(top_n)]
    peaks = np.asarray(atac.var_names.astype(str))[peak_indices]
    return pd.DataFrame({"peak": peaks[order], "group": str(positive_label), "mean_in": a[order], "mean_rest": b[order], "mean_diff": diff[order], "detection_in": det_a[order], "detection_rest": det_b[order], "detection_diff": (det_a - det_b)[order]})


def cluster_specific_peaks(atac, score_df, cluster_key, max_peaks):
    if cluster_key not in score_df.columns:
        return pd.DataFrame()
    peak_indices = select_variable_peaks(atac, max_peaks)
    frames = []
    for cluster in pd.Series(score_df[cluster_key]).astype(str).unique():
        frame = mean_diff_peaks(atac, score_df[cluster_key], cluster, peak_indices)
        if not frame.empty:
            frame.insert(0, "cluster", cluster)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def residual_high_peaks(atac, score_df, max_peaks):
    if "high_residual" not in score_df.columns:
        return pd.DataFrame()
    status = pd.Series(score_df["high_residual"]).astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
    return mean_diff_peaks(atac, np.where(status, "residual_high", "residual_low"), "residual_high", select_variable_peaks(atac, max_peaks))


def gate_associated_peaks(atac, score_df, max_peaks):
    peak_indices = select_variable_peaks(atac, max_peaks)
    if len(peak_indices) == 0:
        return pd.DataFrame()
    traits = [col for col in ["gate_mid_minus_low", "correction_ratio"] if col in score_df.columns]
    rows = []
    peaks = np.asarray(atac.var_names.astype(str))[peak_indices]
    x = atac[:, peak_indices].X
    for trait in traits:
        y = pd.to_numeric(score_df[trait], errors="coerce").to_numpy(dtype=float)
        for j, peak in enumerate(peaks):
            vec = to_dense_vector(x[:, j])
            if np.std(vec) == 0 or np.std(y[np.isfinite(y)]) == 0:
                rho = np.nan; pvalue = np.nan
            elif scipy_stats is not None:
                rho, pvalue = scipy_stats.spearmanr(vec, y, nan_policy="omit")
            else:
                rho = float(np.corrcoef(vec, y)[0, 1]); pvalue = np.nan
            rows.append({"peak": peak, "trait": trait, "spearman_r": rho, "pvalue": pvalue})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["fdr"] = bh_fdr(df["pvalue"].to_numpy(dtype=float))
        df = df.sort_values(["trait", "fdr", "spearman_r"], ascending=[True, True, False])
    return df


def run_motif_enrichment(significant, motif_csv, output_dir):
    if not motif_csv:
        path = output_dir / "motif_enrichment_skipped.txt"
        path.write_text("motif analysis skipped because no peak-motif annotation was provided.\n", encoding="utf-8")
        return None, str(path), "motif analysis skipped because no peak-motif annotation was provided"
    path = resolve_path(motif_csv)
    motif_df = pd.read_csv(path)
    peak_col = "peak" if "peak" in motif_df.columns else motif_df.columns[0]
    motif_col = "motif" if "motif" in motif_df.columns else ("TF" if "TF" in motif_df.columns else None)
    if motif_col is None:
        skipped = output_dir / "motif_enrichment_skipped.txt"
        skipped.write_text("motif annotation lacks motif/TF column.\n", encoding="utf-8")
        return None, str(skipped), "motif annotation lacks motif/TF column"
    sig_peaks = set(significant.get("peak", pd.Series(dtype=str)).astype(str))
    rows = []
    all_peaks = set(motif_df[peak_col].astype(str))
    for motif, sub in motif_df.groupby(motif_col, observed=False):
        motif_peaks = set(sub[peak_col].astype(str))
        a = len(sig_peaks & motif_peaks); b = len(sig_peaks - motif_peaks); c = len(motif_peaks - sig_peaks); d = len(all_peaks - sig_peaks - motif_peaks)
        if scipy_stats is not None:
            _odds, pvalue = scipy_stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        else:
            pvalue = np.nan
        rows.append({"motif": motif, "overlap": a, "sig_peaks": len(sig_peaks), "motif_peaks": len(motif_peaks), "pvalue": pvalue})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = bh_fdr(out["pvalue"].to_numpy(dtype=float))
    out_path = output_dir / "motif_enrichment.csv"
    out.to_csv(out_path, index=False)
    return str(out_path), None, "ok"


def save_figures(corr_df, cluster_da, residual_da, gate_df, output_dir, formats, dpi):
    paths = []
    if not corr_df.empty:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=int(dpi))
        fdr = pd.to_numeric(corr_df["fdr"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        ax.scatter(corr_df["pearson_r"], -np.log10(np.maximum(fdr, 1e-300)), s=8, alpha=0.7)
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("-log10(FDR)")
        ax.set_title("Peak-gene correlation volcano")
        fig.tight_layout()
        for fmt in formats:
            path = output_dir / ("peak_gene_correlation_volcano.%s" % fmt); fig.savefig(path, dpi=int(dpi), bbox_inches="tight"); paths.append(path)
        plt.close(fig)
        top = corr_df.reindex(corr_df["pearson_r"].abs().sort_values(ascending=False).index).head(20).copy()
        if not top.empty:
            labels = top["gene"].astype(str) + " | " + top["peak"].astype(str)
            fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.28 * top.shape[0])), dpi=int(dpi))
            ax.barh(np.arange(top.shape[0]), top["pearson_r"])
            ax.set_yticks(np.arange(top.shape[0])); ax.set_yticklabels(labels, fontsize=6); ax.invert_yaxis(); ax.set_xlabel("Pearson r"); ax.set_title("Top peak-gene links")
            fig.tight_layout()
            for fmt in formats:
                path = output_dir / ("top_peak_gene_links_barplot.%s" % fmt); fig.savefig(path, dpi=int(dpi), bbox_inches="tight"); paths.append(path)
            plt.close(fig)
    summary = pd.DataFrame({"category": ["cluster_specific", "residual_high", "gate_associated"], "count": [cluster_da.shape[0], residual_da.shape[0], gate_df.shape[0]]})
    fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=int(dpi))
    im = ax.imshow(summary[["count"]].to_numpy(dtype=float), aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(summary.shape[0])); ax.set_yticklabels(summary["category"]); ax.set_xticks([0]); ax.set_xticklabels(["count"]); ax.set_title("Regulatory summary")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    fig.tight_layout()
    for fmt in formats:
        path = output_dir / ("regulatory_summary_heatmap.%s" % fmt); fig.savefig(path, dpi=int(dpi), bbox_inches="tight"); paths.append(path)
    plt.close(fig)
    return [str(path) for path in paths]


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    rna_path, atac_path, gtf_path = resolve_inputs(config, args)
    score_path = resolve_path(args.ssr_score_csv)
    for label, path in [("RNA", rna_path), ("ATAC", atac_path), ("GTF", gtf_path), ("SSR score", score_path)]:
        if path is None or not Path(path).exists():
            raise FileNotFoundError("%s input not found: %s" % (label, path))
    tss_window = int(args.tss_window if args.tss_window is not None else config.get("data", {}).get("parameters", {}).get("tss_window", 150000))
    print("[rna-atac regulatory] reading inputs")
    rna_raw = sc.read_h5ad(rna_path)
    atac_raw = sc.read_h5ad(atac_path)
    score_df = pd.read_csv(score_path)
    rna_raw, atac_raw, score_df, align_warnings = align_anndata(rna_raw, atac_raw, score_df)
    warnings.extend(align_warnings)
    rna = prepare_rna_matrix(rna_raw)
    atac = prepare_atac_matrix(atac_raw, args.atac_mode)
    peaks_df = parse_peaks(atac)
    genes_df = parse_gtf(gtf_path)
    peaks_path = output_dir / "parsed_peaks.csv"; peaks_df.drop(columns=[], errors="ignore").to_csv(peaks_path, index=False)
    genes_path = output_dir / "gene_tss_annotation.csv"; genes_df.to_csv(genes_path, index=False)
    print("[rna-atac regulatory] building candidate links")
    candidates = build_candidate_links(peaks_df, genes_df, rna, atac, tss_window, args.max_links, args.min_peak_detection, args.min_gene_detection)
    candidates_path = output_dir / "peak_gene_candidate_links.csv"; candidates.drop(columns=["peak_index", "gene_index"], errors="ignore").to_csv(candidates_path, index=False)
    print("[rna-atac regulatory] computing correlations for %d links" % candidates.shape[0])
    corr_df = compute_correlations(candidates, rna, atac)
    corr_path = output_dir / "peak_gene_correlations.csv"; corr_df.to_csv(corr_path, index=False)
    if corr_df.empty:
        significant = corr_df.copy()
    else:
        significant = corr_df[(corr_df["fdr"] < float(args.fdr_cutoff)) & (corr_df["pearson_r"].abs() >= float(args.min_abs_pearson))].copy()
    sig_path = output_dir / "significant_peak_gene_links.csv"; significant.to_csv(sig_path, index=False)
    print("[rna-atac regulatory] peak summaries")
    cluster_da = cluster_specific_peaks(atac, score_df, "cluster", args.max_peaks_for_da)
    cluster_da_path = output_dir / "cluster_specific_accessible_peaks.csv"; cluster_da.to_csv(cluster_da_path, index=False)
    residual_da = residual_high_peaks(atac, score_df, args.max_peaks_for_da)
    residual_path = output_dir / "residual_high_accessible_peaks.csv"; residual_da.to_csv(residual_path, index=False)
    gate_df = gate_associated_peaks(atac, score_df, args.max_variable_peaks)
    gate_path = output_dir / "gate_associated_peaks.csv"; gate_df.to_csv(gate_path, index=False)
    motif_path, motif_skip_path, motif_status = run_motif_enrichment(significant, args.motif_annotation_csv, output_dir)
    if motif_skip_path:
        warnings.append(motif_status)
    figures = save_figures(corr_df, cluster_da, residual_da, gate_df, output_dir, args.formats, args.dpi)
    manifest = {
        "parameters": {
            "atac_mode": args.atac_mode,
            "max_links": int(args.max_links),
            "min_peak_detection": float(args.min_peak_detection),
            "min_gene_detection": float(args.min_gene_detection),
            "fdr_cutoff": float(args.fdr_cutoff),
            "min_abs_pearson": float(args.min_abs_pearson),
            "max_peaks_for_da": int(args.max_peaks_for_da),
            "max_variable_peaks": int(args.max_variable_peaks),
            "motif_annotation_csv": args.motif_annotation_csv,
            "correction": "BH across computed candidate pairs",
        },
        "config": str(config_path), "rna_h5ad": str(rna_path), "atac_h5ad": str(atac_path), "gtf": str(gtf_path), "ssr_score_csv": str(score_path),
        "n_spots": int(rna.n_obs), "n_peaks": int(atac.n_vars), "n_genes": int(rna.n_vars), "tss_window": tss_window,
        "n_candidates": int(candidates.shape[0]), "n_significant": int(significant.shape[0]), "warnings": warnings,
        "parsed_peaks": str(peaks_path), "gene_tss_annotation": str(genes_path), "peak_gene_candidate_links": str(candidates_path),
        "peak_gene_correlations": str(corr_path), "significant_peak_gene_links": str(sig_path), "cluster_specific_accessible_peaks": str(cluster_da_path),
        "residual_high_accessible_peaks": str(residual_path), "gate_associated_peaks": str(gate_path), "motif_enrichment": motif_path, "motif_skipped": motif_skip_path,
        "figures": figures,
    }
    manifest_path = output_dir / "rna_atac_regulatory_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[rna-atac regulatory] complete candidates=%d significant=%d" % (candidates.shape[0], significant.shape[0]))
    print("  manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()
