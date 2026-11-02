"""Reproducible cluster-wise RNA GO-BP/KEGG over-representation analysis.

The primary backend is the g:Profiler REST API with a custom expressed-gene
background. Every response is cached so figures and tables can later be
regenerated without contacting a moving external database.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scanpy as sc
import yaml
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
RESULT_COLUMNS = [
    "dataset_key", "module", "module_label", "cluster", "source", "term_id", "term",
    "fdr", "significant", "overlap", "query_size", "term_size",
    "background_size", "precision", "recall", "overlap_gene_ids",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cluster-wise RNA GO-BP/KEGG ORA with an expressed-gene background."
    )
    parser.add_argument("--config", required=True, help="Dataset YAML used to resolve raw RNA when --raw-rna-h5ad is omitted.")
    parser.add_argument("--marker-csv", required=True, help="Marker table with Cluster/Gene and optional LogFoldChange/P_value_adj columns.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-rna-h5ad", default=None)
    parser.add_argument("--background-gene-file", default=None, help="Optional one-gene-per-line override for the RNA background.")
    parser.add_argument("--dataset-key", required=True, help="Stable dataset key used in tables and module mapping.")
    parser.add_argument("--organism", default="mmusculus", help="g:Profiler organism code, e.g. mmusculus or hsapiens.")
    parser.add_argument("--sources", nargs="+", default=["GO:BP", "KEGG"])
    parser.add_argument("--module-map", default=None, help="Optional YAML assigning pre-specified functional modules to dataset clusters.")
    parser.add_argument("--background-min-fraction", type=float, default=0.01)
    parser.add_argument("--marker-padj-cutoff", type=float, default=0.01)
    parser.add_argument("--marker-logfc-min", type=float, default=0.0)
    parser.add_argument("--min-query-genes", type=int, default=10)
    parser.add_argument("--max-query-genes", type=int, default=50)
    parser.add_argument("--min-term-size", type=int, default=10)
    parser.add_argument("--max-term-size", type=int, default=500)
    parser.add_argument("--fdr-cutoff", type=float, default=0.05)
    parser.add_argument("--min-recognition-rate", type=float, default=0.70)
    parser.add_argument("--go-redundancy-jaccard", type=float, default=0.70)
    parser.add_argument("--top-terms-per-group", type=int, default=5)
    parser.add_argument("--cache-dir", default=None, help="Defaults to <output-dir>/api_cache.")
    parser.add_argument("--offline-from-cache", action="store_true")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_path(value, base_dir=PROJECT_ROOT):
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_raw_rna(config, explicit_path):
    if explicit_path:
        return resolve_path(explicit_path)
    data = config.get("data", {}) or {}
    raw_path = data.get("raw_path")
    rna_name = (data.get("files", {}) or {}).get("rna_h5ad")
    if not raw_path or not rna_name:
        raise ValueError("Cannot resolve raw RNA; pass --raw-rna-h5ad or configure data.raw_path/data.files.rna_h5ad.")
    rna_path = Path(str(rna_name)).expanduser()
    return rna_path if rna_path.is_absolute() else resolve_path(raw_path) / rna_path


def find_column(df, candidates):
    lookup = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def ordered_unique(values, limit=None):
    result = []
    seen = set()
    for value in values:
        if pd.isna(value):
            continue
        gene = str(value).strip()
        if not gene or gene.lower() == "nan" or gene in seen:
            continue
        seen.add(gene)
        result.append(gene)
        if limit is not None and len(result) >= int(limit):
            break
    return result


def cluster_sort_key(value):
    text = str(value)
    try:
        return (0, float(text), text)
    except ValueError:
        return (1, math.inf, text)


def load_module_map(path, dataset_key):
    if not path:
        return {}
    payload = load_yaml(resolve_path(path))
    result = {}
    for module_key, record in (payload.get("modules", {}) or {}).items():
        clusters = (record or {}).get("clusters", {}) or {}
        cluster = clusters.get(dataset_key)
        if cluster is not None:
            result[str(cluster)] = {
                "module": str(module_key),
                "module_label": str((record or {}).get("label", module_key)),
                "expected_markers": list((record or {}).get("expected_markers", []) or []),
            }
    return result


def load_marker_sets(path, args, module_map):
    marker_path = resolve_path(path)
    if not marker_path.exists():
        raise FileNotFoundError("Marker CSV not found: %s" % marker_path)
    df = pd.read_csv(marker_path)
    gene_col = find_column(df, ["Gene", "gene", "genes", "symbol", "GeneSymbol"])
    cluster_col = find_column(df, ["Cluster", "cluster"])
    if gene_col is None or cluster_col is None:
        raise ValueError("Marker CSV must contain gene and cluster columns: %s" % marker_path)
    padj_col = find_column(df, ["P_value_adj", "p_value_adj", "pval_adj", "padj"])
    logfc_col = find_column(df, ["LogFoldChange", "logfoldchange", "log2fc", "logFC"])
    filtered = df.copy()
    if padj_col is not None:
        filtered = filtered[pd.to_numeric(filtered[padj_col], errors="coerce") < float(args.marker_padj_cutoff)]
    if logfc_col is not None:
        filtered = filtered[pd.to_numeric(filtered[logfc_col], errors="coerce") > float(args.marker_logfc_min)]
    records = []
    cluster_values = sorted(df[cluster_col].dropna().astype(str).unique(), key=cluster_sort_key)
    filtered_cluster_values = set(filtered[cluster_col].dropna().astype(str).unique())
    for cluster in cluster_values:
        sub = filtered[filtered[cluster_col].astype(str) == cluster]
        genes = ordered_unique(sub[gene_col], args.max_query_genes)
        mapping = module_map.get(cluster, {})
        records.append({
            "cluster": cluster,
            "module": mapping.get("module", "unassigned"),
            "module_label": mapping.get("module_label", "Unassigned cluster %s" % cluster),
            "expected_markers": mapping.get("expected_markers", []),
            "genes": genes,
            "n_marker_rows_before_filter": int((df[cluster_col].astype(str) == cluster).sum()),
        })
    return records, marker_path, {
        "gene_column": str(gene_col), "cluster_column": str(cluster_col),
        "padj_column": None if padj_col is None else str(padj_col),
        "logfc_column": None if logfc_col is None else str(logfc_col),
        "n_input_rows": int(df.shape[0]), "n_filtered_rows": int(filtered.shape[0]),
        "n_input_clusters": len(cluster_values),
        "n_filtered_clusters": len(filtered_cluster_values),
        "clusters_without_eligible_markers": [
            cluster for cluster in cluster_values if cluster not in filtered_cluster_values
        ],
    }


def expressed_background_from_h5ad(path, min_fraction, chunk_size=1024):
    if not 0.0 <= float(min_fraction) <= 1.0:
        raise ValueError("--background-min-fraction must be in [0, 1].")
    adata = sc.read_h5ad(path, backed="r")
    n_obs, n_vars = int(adata.n_obs), int(adata.n_vars)
    detected = np.zeros(n_vars, dtype=np.int64)
    for start in range(0, n_obs, int(chunk_size)):
        chunk = adata.X[start:min(start + int(chunk_size), n_obs)]
        if sparse.issparse(chunk):
            detected += np.asarray((chunk != 0).sum(axis=0)).ravel().astype(np.int64)
        else:
            detected += np.count_nonzero(np.asarray(chunk), axis=0)
    threshold = max(1, int(math.ceil(float(min_fraction) * n_obs)))
    genes = [str(gene) for gene, count in zip(adata.var_names, detected) if int(count) >= threshold]
    if getattr(adata, "file", None) is not None:
        adata.file.close()
    return ordered_unique(genes), {"n_spots": n_obs, "n_features": n_vars, "min_spots": threshold}


def load_background(config, args):
    if args.background_gene_file:
        path = resolve_path(args.background_gene_file)
        genes = ordered_unique(pd.read_csv(path, header=None).iloc[:, 0])
        return genes, str(path), {"mode": "explicit_file"}
    raw_rna = resolve_raw_rna(config, args.raw_rna_h5ad)
    if not raw_rna.exists():
        raise FileNotFoundError("Raw RNA h5ad not found: %s" % raw_rna)
    genes, details = expressed_background_from_h5ad(raw_rna, args.background_min_fraction)
    details["mode"] = "expressed_in_min_fraction"
    details["min_fraction"] = float(args.background_min_fraction)
    return genes, str(raw_rna), details


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(payload):
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def request_gprofiler(payload, cache_dir, api_url, timeout, offline):
    key = cache_key(payload)
    response_path = cache_dir / (key + ".response.json")
    request_path = cache_dir / (key + ".request.json")
    if response_path.exists():
        with open(response_path, "r", encoding="utf-8") as handle:
            return json.load(handle), response_path, True
    if offline:
        raise FileNotFoundError("Cached g:Profiler response is required in offline mode: %s" % response_path)
    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={"Accept": "application/json", "User-Agent": "BioSFIB-RNA-Enrichment/1.0"},
            timeout=float(timeout),
        )
        if not response.ok:
            raise RuntimeError(
                "g:Profiler HTTP %s: %s" % (response.status_code, response.text[:2000])
            )
    except requests.RequestException as exc:
        raise RuntimeError("g:Profiler request failed: %s" % exc) from exc
    try:
        parsed = response.json()
    except ValueError as exc:
        raise RuntimeError("g:Profiler returned invalid JSON: %s" % response.text[:1000]) from exc
    request_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    response_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    return parsed, response_path, False


def query_metadata(response):
    meta = response.get("meta", {}) or {}
    genes_meta = meta.get("genes_metadata", {}) or {}
    failed = genes_meta.get("failed", []) or []
    ambiguous = genes_meta.get("ambiguous", {}) or {}
    version = meta.get("version")
    timestamp = meta.get("timestamp")
    return {
        "failed_genes": sorted(str(value) for value in failed),
        "n_failed": len(failed),
        "ambiguous": ambiguous,
        "database_version": version,
        "database_timestamp": timestamp,
    }


def overlap_ensembl_ids(result, response):
    intersections = result.get("intersections") or []
    meta = response.get("meta", {}) or {}
    query_meta = ((meta.get("genes_metadata", {}) or {}).get("query", {}) or {})
    query_name = str(result.get("query", "query_1"))
    record = query_meta.get(query_name)
    if record is None and query_meta:
        record = next(iter(query_meta.values()))
    ensgs = (record or {}).get("ensgs", []) or []
    return [str(gene_id) for gene_id, evidence in zip(ensgs, intersections) if evidence]


def parse_response(response, record, args):
    rows = []
    for result in response.get("result", []) or []:
        source = str(result.get("source", ""))
        term_size = int(result.get("term_size", 0) or 0)
        if source not in set(args.sources):
            continue
        if term_size < int(args.min_term_size) or term_size > int(args.max_term_size):
            continue
        fdr = float(result.get("p_value", np.nan))
        overlap = int(result.get("intersection_size", 0) or 0)
        query_size = int(result.get("query_size", 0) or 0)
        background_size = int(result.get("effective_domain_size", 0) or 0)
        rows.append({
            "dataset_key": args.dataset_key,
            "module": record["module"],
            "module_label": record["module_label"],
            "cluster": record["cluster"],
            "source": source,
            "term_id": str(result.get("native", "")),
            "term": str(result.get("name", "")),
            "fdr": fdr,
            "significant": bool(fdr <= float(args.fdr_cutoff)),
            "overlap": overlap,
            "query_size": query_size,
            "term_size": term_size,
            "background_size": background_size,
            "precision": float(result.get("precision", overlap / max(query_size, 1))),
            "recall": float(result.get("recall", overlap / max(term_size, 1))),
            "overlap_gene_ids": ";".join(overlap_ensembl_ids(result, response)),
        })
    return rows


def run_queries(records, background_genes, cache_dir, args):
    all_rows = []
    qc_rows = []
    cache_paths = []
    database_versions = set()
    background_set = set(background_genes)
    for record in records:
        input_genes = list(record["genes"])
        valid_genes = [gene for gene in input_genes if gene in background_set]
        status = "ready"
        response_meta = {}
        from_cache = None
        if not input_genes:
            status = "no_eligible_positive_markers"
        elif len(valid_genes) < int(args.min_query_genes):
            status = "insufficient_genes"
        else:
            payload = {
                "organism": args.organism,
                "query": valid_genes,
                "sources": list(args.sources),
                "user_threshold": float(args.fdr_cutoff),
                "significance_threshold_method": "fdr",
                "domain_scope": "custom",
                "background": list(background_genes),
                "all_results": True,
                "no_evidences": False,
                "no_iea": False,
                "ordered": False,
                "output": "json",
            }
            response, response_path, from_cache = request_gprofiler(
                payload, cache_dir, args.api_url, args.timeout_seconds, args.offline_from_cache
            )
            cache_paths.append(str(response_path))
            response_meta = query_metadata(response)
            if response_meta.get("database_version"):
                database_versions.add(str(response_meta["database_version"]))
            n_failed = int(response_meta.get("n_failed", 0))
            recognition_rate = (len(valid_genes) - n_failed) / max(len(valid_genes), 1)
            if recognition_rate < float(args.min_recognition_rate):
                status = "low_recognition_rate"
            else:
                all_rows.extend(parse_response(response, record, args))
        n_failed = int(response_meta.get("n_failed", 0))
        recognition_rate = (
            None
            if status in {"insufficient_genes", "no_eligible_positive_markers"}
            else (len(valid_genes) - n_failed) / max(len(valid_genes), 1)
        )
        expected_present = sorted(set(record["expected_markers"]) & set(valid_genes))
        qc_rows.append({
            "dataset_key": args.dataset_key,
            "module": record["module"],
            "module_label": record["module_label"],
            "cluster": record["cluster"],
            "status": status,
            "n_marker_rows_before_filter": record["n_marker_rows_before_filter"],
            "n_selected_genes": len(input_genes),
            "n_genes_in_background": len(valid_genes),
            "n_failed_mapping": n_failed,
            "recognition_rate": recognition_rate,
            "expected_markers_present": ";".join(expected_present),
            "failed_genes": ";".join(response_meta.get("failed_genes", [])),
            "from_cache": from_cache,
            "genes": ";".join(valid_genes),
        })
    results = pd.DataFrame(all_rows, columns=RESULT_COLUMNS)
    qc = pd.DataFrame(qc_rows)
    return results, qc, sorted(database_versions), cache_paths


def parse_gene_set(value):
    return {item for item in str(value).split(";") if item}


def select_plot_terms(results, cutoff, top_n, redundancy_jaccard):
    if results.empty:
        return results.copy()
    significant = results[pd.to_numeric(results["fdr"], errors="coerce") <= float(cutoff)].copy()
    selected = []
    for (_module, _cluster, source), group in significant.groupby(["module", "cluster", "source"], sort=False):
        group = group.sort_values(["fdr", "precision", "term"], ascending=[True, False, True])
        kept_gene_sets = []
        kept = 0
        for _, row in group.iterrows():
            genes = parse_gene_set(row["overlap_gene_ids"])
            redundant = False
            if source == "GO:BP" and genes:
                for previous in kept_gene_sets:
                    union = genes | previous
                    if union and len(genes & previous) / len(union) > float(redundancy_jaccard):
                        redundant = True
                        break
            if redundant:
                continue
            selected.append(row.to_dict())
            kept_gene_sets.append(genes)
            kept += 1
            if kept >= int(top_n):
                break
    return pd.DataFrame(selected, columns=RESULT_COLUMNS)


def save_dotplots(plot_df, output_dir, formats, dpi):
    paths = []
    if plot_df.empty:
        return paths
    for source, source_df in plot_df.groupby("source", sort=False):
        source_df = source_df.copy()
        cluster_order = {
            str(cluster): index
            for index, cluster in enumerate(
                sorted(source_df["cluster"].astype(str).unique(), key=cluster_sort_key)
            )
        }
        source_df["_cluster_order"] = source_df["cluster"].astype(str).map(cluster_order)
        source_df = source_df.sort_values(["_cluster_order", "fdr", "term"], kind="stable")
        source_df["group"] = source_df.apply(
            lambda row: (
                "C%s" % row["cluster"]
                if str(row["module"]) == "unassigned"
                else "%s | C%s" % (row["module_label"], row["cluster"])
            ),
            axis=1,
        )
        groups = list(dict.fromkeys(source_df["group"].astype(str)))
        source_df["term_label"] = source_df["term"].astype(str) + " [" + source_df["term_id"].astype(str) + "]"
        terms = list(dict.fromkeys(source_df.sort_values("fdr")["term_label"].astype(str)))
        x_lookup = {value: index for index, value in enumerate(groups)}
        y_lookup = {value: index for index, value in enumerate(terms[::-1])}
        x = source_df["group"].map(x_lookup).to_numpy()
        y = source_df["term_label"].map(y_lookup).to_numpy()
        color = -np.log10(np.maximum(pd.to_numeric(source_df["fdr"], errors="coerce").to_numpy(), 1e-300))
        size = 45.0 + 650.0 * pd.to_numeric(source_df["precision"], errors="coerce").fillna(0.0).to_numpy()
        fig_width = max(7.5, 1.15 * len(groups) + 4.5)
        fig_height = max(4.5, 0.33 * len(terms) + 1.8)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=int(dpi))
        scatter = ax.scatter(x, y, c=color, s=size, cmap="viridis", edgecolor="#333333", linewidth=0.35)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=35, ha="right")
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms[::-1], fontsize=8)
        all_unassigned = source_df["module"].astype(str).eq("unassigned").all()
        ax.set_xlabel("Cluster" if all_unassigned else "Pre-specified functional module and cluster")
        ax.set_ylabel("Enriched term")
        ax.set_title("%s enrichment (custom expressed-RNA background)" % source)
        ax.grid(axis="both", color="#dddddd", linewidth=0.5, alpha=0.7)
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02)
        colorbar.set_label("-log10(FDR)")
        gene_ratio_levels = [0.10, 0.20, 0.30]
        size_handles = [
            ax.scatter(
                [], [],
                s=45.0 + 650.0 * value,
                facecolor="#777777",
                edgecolor="#333333",
                linewidth=0.35,
                label="%.2f" % value,
            )
            for value in gene_ratio_levels
        ]
        ax.legend(
            handles=size_handles,
            title="Gene ratio",
            loc="center left",
            bbox_to_anchor=(1.10, 0.32),
            frameon=False,
            borderaxespad=0.0,
            labelspacing=1.0,
        )
        fig.tight_layout()
        stem = "go_bp_enrichment_dotplot" if source == "GO:BP" else "%s_enrichment_dotplot" % source.lower().replace(":", "_")
        for fmt in formats:
            path = output_dir / (stem + "." + str(fmt).lower())
            fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
            paths.append(str(path))
        plt.close(fig)
    return paths


def write_csv(df, path):
    if df.empty and len(df.columns) == 0:
        df = pd.DataFrame(columns=RESULT_COLUMNS)
    df.to_csv(path, index=False)
    return str(path)


def main():
    args = parse_args()
    if args.min_query_genes <= 0 or args.max_query_genes < args.min_query_genes:
        raise ValueError("Require 0 < min-query-genes <= max-query-genes.")
    if not 0.0 < args.fdr_cutoff <= 1.0:
        raise ValueError("--fdr-cutoff must be in (0, 1].")
    config_path = resolve_path(args.config)
    marker_path = resolve_path(args.marker_csv)
    output_dir = resolve_path(args.output_dir)
    cache_dir = resolve_path(args.cache_dir) if args.cache_dir else output_dir / "api_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(config_path)
    module_map = load_module_map(args.module_map, args.dataset_key)
    records, marker_path, marker_schema = load_marker_sets(marker_path, args, module_map)
    background_genes, background_source, background_details = load_background(config, args)
    background_set = set(background_genes)
    if len(background_genes) < int(args.min_query_genes):
        raise ValueError("Expressed-gene background is too small: %d" % len(background_genes))
    for record in records:
        record["genes"] = [gene for gene in record["genes"] if gene in background_set]

    background_path = output_dir / "expressed_rna_background.txt"
    background_path.write_text("\n".join(background_genes) + "\n", encoding="utf-8")
    results, qc, versions, cache_paths = run_queries(records, background_genes, cache_dir, args)
    results = results.sort_values(["source", "module", "cluster", "fdr", "term"], kind="stable") if not results.empty else results
    significant = results[results["fdr"] <= float(args.fdr_cutoff)].copy() if not results.empty else results.copy()
    plot_source = results
    if module_map and not results.empty:
        plot_source = results[results["module"] != "unassigned"].copy()
    plot_df = select_plot_terms(
        plot_source, args.fdr_cutoff, args.top_terms_per_group, args.go_redundancy_jaccard
    )

    all_path = output_dir / "enrichment_all_results.csv"
    sig_path = output_dir / "enrichment_significant_results.csv"
    plot_data_path = output_dir / "enrichment_plot_data.csv"
    qc_path = output_dir / "enrichment_input_qc.csv"
    write_csv(results, all_path)
    write_csv(significant, sig_path)
    write_csv(plot_df, plot_data_path)
    write_csv(qc, qc_path)
    figures = save_dotplots(plot_df, output_dir, args.formats, args.dpi)

    cluster_status = qc["status"].value_counts(dropna=False).to_dict() if not qc.empty else {}
    warnings = []
    if not qc.empty:
        for _, row in qc[qc["status"] != "ready"].iterrows():
            warnings.append("cluster_%s: %s" % (row["cluster"], row["status"]))
    status = "ok"
    if results.empty:
        status = "no_enrichment_results"
    elif significant.empty:
        status = "no_significant_results"
    manifest = {
        "status": status,
        "analysis": "cluster-wise RNA ORA",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_key": args.dataset_key,
        "organism": args.organism,
        "sources": list(args.sources),
        "api_url": args.api_url,
        "database_versions": versions,
        "offline_from_cache": bool(args.offline_from_cache),
        "parameters": {
            "background_min_fraction": float(args.background_min_fraction),
            "marker_padj_cutoff": float(args.marker_padj_cutoff),
            "marker_logfc_min": float(args.marker_logfc_min),
            "min_query_genes": int(args.min_query_genes),
            "max_query_genes": int(args.max_query_genes),
            "min_term_size": int(args.min_term_size),
            "max_term_size": int(args.max_term_size),
            "fdr_cutoff": float(args.fdr_cutoff),
            "correction": "Benjamini-Hochberg FDR",
            "domain_scope": "custom",
            "go_redundancy_jaccard": float(args.go_redundancy_jaccard),
        },
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "marker_csv": str(marker_path),
            "marker_csv_sha256": sha256_file(marker_path),
            "marker_schema": marker_schema,
            "raw_background_source": background_source,
            "background_details": background_details,
            "background_gene_file": str(background_path),
            "background_gene_file_sha256": sha256_file(background_path),
            "module_map": None if not args.module_map else str(resolve_path(args.module_map)),
            "module_map_sha256": None if not args.module_map else sha256_file(resolve_path(args.module_map)),
        },
        "counts": {
            "background_genes": len(background_genes),
            "clusters_seen": len(records),
            "cluster_status": cluster_status,
            "all_terms": int(results.shape[0]),
            "significant_terms": int(significant.shape[0]),
            "plot_terms": int(plot_df.shape[0]),
        },
        "outputs": {
            "input_qc": str(qc_path),
            "all_results": str(all_path),
            "significant_results": str(sig_path),
            "plot_data": str(plot_data_path),
            "figures": figures,
            "cached_responses": cache_paths,
        },
        "warnings": warnings,
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    manifest_path = output_dir / "gene_set_enrichment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("[RNA enrichment] status=%s dataset=%s clusters=%d significant_terms=%d" % (
        status, args.dataset_key, len(records), significant.shape[0]
    ))
    print("  QC      : %s" % qc_path)
    print("  results : %s" % all_path)
    print("  manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()
