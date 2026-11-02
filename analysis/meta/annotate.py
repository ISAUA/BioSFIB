"""Independent SpatialMETA-style DestVI annotation for frozen BioSFIB Y7_T.

This workflow uses only the Hu et al. ccRCC single-nucleus RNA reference, the
canonical SpatialMETA Y7_T ST counts, and the already-fixed BioSFIB K=14 result.
It never imports SpaMV data and never changes training, checkpoint, embedding,
PCA, Mclust, or cluster-number choices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import yaml
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = (
    "analysis/meta/"
    "annotation.yaml"
)
ALLOWED_STAGES = ("prepare-reference", "run-destvi")
MAPPING_COLUMNS = (
    "source_cell_type",
    "spatialmeta_four_class",
    "decision",
    "notes",
)


def destvi_from_rna_model_compat(st_adata, rna_model, scvi_module, target_device):
    """Construct DestVI while preserving scvi-tools 1.4.3 VAMP prior semantics."""
    original_get_vamp_prior = rna_model.get_vamp_prior

    def tensor_vamp_prior(*args, **kwargs):
        prior = original_get_vamp_prior(*args, **kwargs)
        return {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in prior.items()
        }

    # scvi-tools 1.4.3 returns NumPy arrays from get_vamp_prior(), while
    # MRDeconv.register_buffer() requires torch.Tensor values. Keep this local
    # compatibility shim until the environment moves to a fixed release.
    rna_model.get_vamp_prior = tensor_vamp_prior
    try:
        spatial_model = scvi_module.model.DestVI.from_rna_model(st_adata, rna_model)
    finally:
        rna_model.get_vamp_prior = original_get_vamp_prior
    module = spatial_model.module.to(target_device)
    if module.mean_vprior is not None:
        # torch.distributions objects are not modules/buffers, so Lightning's
        # device transfer does not update the CPU tensors captured at init.
        # Rebuild the same mixture distribution from the transferred buffers.
        categories = torch.distributions.Categorical(probs=module.mp_vprior)
        components = torch.distributions.Independent(
            torch.distributions.Normal(
                module.mean_vprior,
                torch.sqrt(module.var_vprior) + 1e-4,
            ),
            reinterpreted_batch_ndims=1,
        )
        module.qz_prior = torch.distributions.MixtureSameFamily(categories, components)
    return spatial_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the official ccRCC snRNA reference or run an independent "
            "DestVI/marker annotation of the frozen BioSFIB Y7_T K=14 domains."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=ALLOWED_STAGES, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this workflow's derived outputs only.",
    )
    return parser.parse_args()


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_new_or_overwrite(path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def file_hash(path, algorithm="sha256", chunk_size=8 * 1024 * 1024):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def load_config(path):
    path = require_file(resolve_path(path), "annotation config")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("protocol", {}).get("forbid_spamv") is not True:
        raise ValueError("protocol.forbid_spamv must remain true")

    def scan_strings(value, trail="config"):
        if isinstance(value, dict):
            for key, item in value.items():
                scan_strings(item, f"{trail}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                scan_strings(item, f"{trail}[{index}]")
        elif isinstance(value, str) and "spamv" in value.lower():
            raise ValueError(f"SpaMV-derived input is forbidden: {trail}={value!r}")

    scan_strings(config)
    return config, path


def matrix_is_raw_counts(matrix, sample_limit=5_000_000):
    if sp.issparse(matrix):
        data = np.asarray(matrix.data)
    else:
        data = np.asarray(matrix).reshape(-1)
    if data.size > sample_limit:
        positions = np.linspace(0, data.size - 1, sample_limit, dtype=np.int64)
        data = data[positions]
    if data.size == 0:
        return False
    return bool(
        np.all(np.isfinite(data))
        and np.min(data) >= 0
        and np.allclose(data, np.rint(data), rtol=0.0, atol=1e-6)
    )


def counts_candidates(adata):
    candidates = []
    if "counts" in adata.layers:
        candidates.append(("layers/counts", adata.layers["counts"], adata.var.copy()))
    if adata.raw is not None:
        candidates.append(("raw/X", adata.raw.X, adata.raw.var.copy()))
    candidates.append(("X", adata.X, adata.var.copy()))
    return candidates


def select_counts(adata):
    diagnostics = []
    for location, matrix, var in counts_candidates(adata):
        is_counts = matrix_is_raw_counts(matrix)
        diagnostics.append(
            {
                "location": location,
                "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "nonnegative_integer_like": bool(is_counts),
            }
        )
        if is_counts:
            counts = matrix.tocsr(copy=True) if sp.issparse(matrix) else sp.csr_matrix(matrix)
            counts.eliminate_zeros()
            return location, counts, var, diagnostics
    raise ValueError(
        "No nonnegative integer-like raw-count representation was found. "
        f"Candidates: {diagnostics}"
    )


def label_column_inventory(obs):
    rows = []
    preferred_tokens = (
        "celltype",
        "cell_type",
        "cell type",
        "annotation",
        "subtype",
        "identity",
        "ident",
        "cluster",
    )
    for column in obs.columns:
        series = obs[column]
        n_unique = int(series.nunique(dropna=True))
        if n_unique < 2 or n_unique > 500:
            continue
        dtype = str(series.dtype)
        if not (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            continue
        lower = str(column).lower()
        score = next(
            (len(preferred_tokens) - index for index, token in enumerate(preferred_tokens) if token in lower),
            0,
        )
        counts = series.astype("string").value_counts(dropna=False)
        rows.append(
            {
                "column": str(column),
                "dtype": dtype,
                "n_unique": n_unique,
                "n_missing": int(series.isna().sum()),
                "name_score": int(score),
                "top_values": {str(key): int(value) for key, value in counts.head(20).items()},
            }
        )
    return sorted(rows, key=lambda row: (-row["name_score"], row["n_unique"], row["column"]))


def heuristic_four_class(cell_type):
    label = str(cell_type).strip().lower()
    token_groups = {
        "Imm": (
            "immune", "t cell", "t-cell", "cd4", "cd8", "nk", "b cell", "b-cell",
            "plasma", "macroph", "monocyte", "myeloid", "dendritic", "mast", "neutro",
        ),
        "Endo": ("endothelial", "endothelium", "vascular", "endo"),
        "Stro": ("stromal", "fibro", "caf", "myofibro", "pericyte", "smooth muscle", "mesang"),
        "Mal": ("malignant", "tumor", "tumour", "cancer", "ccrcc"),
    }
    matches = [group for group, tokens in token_groups.items() if any(token in label for token in tokens)]
    return matches[0] if len(matches) == 1 else ""


def write_mapping_template(path, labels, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_COLUMNS)
        writer.writeheader()
        for label in sorted(map(str, labels)):
            suggestion = heuristic_four_class(label)
            writer.writerow(
                {
                    "source_cell_type": label,
                    "spatialmeta_four_class": suggestion,
                    "decision": "heuristic_unreviewed" if suggestion else "unmapped",
                    "notes": "Review before DestVI; blank or unreviewed rows block training.",
                }
            )


def prepare_reference(config, overwrite):
    section = config["reference"]
    source_rds = require_file(resolve_path(section["source_rds"]), "official source RDS")
    expected_md5 = str(section["source_md5"]).lower()
    observed_md5 = file_hash(source_rds, "md5")
    if observed_md5 != expected_md5:
        raise ValueError(f"Reference RDS MD5 mismatch: expected {expected_md5}, found {observed_md5}")

    converted = require_file(resolve_path(section["converted_h5ad"]), "Seurat-converted H5AD")
    prepared = require_new_or_overwrite(resolve_path(section["prepared_h5ad"]), overwrite)
    inventory_path = require_new_or_overwrite(resolve_path(section["inventory_json"]), overwrite)
    converted_adata = ad.read_h5ad(converted)
    counts_location, counts, var, count_diagnostics = select_counts(converted_adata)
    if counts.shape[0] != converted_adata.n_obs:
        raise ValueError("Selected count matrix does not match converted observation count")
    if var.index.has_duplicates:
        duplicates = var.index[var.index.duplicated()].astype(str).unique()[:10].tolist()
        raise ValueError(f"Reference gene identifiers are duplicated: {duplicates}")

    obs = converted_adata.obs.copy()
    obs.index = obs.index.astype(str)
    if "_index" in var.columns:
        raw_gene_names = var["_index"]
        if raw_gene_names.isna().any():
            raise ValueError("Reference raw.var['_index'] contains missing gene identifiers")
        raw_gene_names = raw_gene_names.astype(str)
        if raw_gene_names.duplicated().any():
            duplicates = raw_gene_names[raw_gene_names.duplicated()].unique()[:10].tolist()
            raise ValueError(f"Reference raw.var['_index'] contains duplicated genes: {duplicates}")
        var = var.drop(columns=["_index"]).copy()
        var.index = pd.Index(raw_gene_names.to_numpy(), dtype="object")
    var.index = var.index.astype(str)
    label_inventory = label_column_inventory(obs)
    requested_key = section.get("cell_type_key")
    if requested_key is not None and requested_key not in obs:
        raise KeyError(f"Configured reference.cell_type_key is absent: {requested_key}")

    clean = ad.AnnData(X=counts, obs=obs, var=var)
    clean.uns["reference_provenance"] = {
        "source": "Hu et al. Nature Genetics 2024; Zenodo 8063124",
        "source_rds_md5": observed_md5,
        "converted_h5ad_sha256": file_hash(converted),
        "counts_location_in_converted_h5ad": counts_location,
        "filtering_or_relabeling": "none",
        "spamv_used": False,
    }
    clean.write_h5ad(prepared, compression="gzip")

    inventory = {
        "source_rds": str(source_rds),
        "source_rds_md5": observed_md5,
        "converted_h5ad": str(converted),
        "converted_h5ad_sha256": file_hash(converted),
        "prepared_h5ad": str(prepared),
        "prepared_h5ad_sha256": file_hash(prepared),
        "shape": [int(clean.n_obs), int(clean.n_vars)],
        "counts_location": counts_location,
        "count_candidates": count_diagnostics,
        "configured_cell_type_key": requested_key,
        "label_column_candidates": label_inventory,
        "obs_columns": list(map(str, clean.obs.columns)),
        "var_columns": list(map(str, clean.var.columns)),
        "spatialmeta_marker_presence": {
            gene: bool(gene in clean.var_names)
            for gene in (
                "CD8A", "CD3D", "CD74", "CD34", "PECAM1", "APLN",
                "COL3A1", "ACTA2", "COL1A1", "NDUFA4L2", "CNDP2", "EGFR",
            )
        },
        "spamv_used": False,
    }
    write_json(inventory_path, inventory)
    if requested_key is not None:
        labels = clean.obs[requested_key].dropna().astype(str).unique()
        write_mapping_template(resolve_path(section["mapping_csv"]), labels, overwrite)

    print(f"Prepared raw-count reference: {prepared}")
    print(f"Reference inventory: {inventory_path}")
    if requested_key is None:
        print("reference.cell_type_key is null; inspect label_column_candidates, then update the config.")
    else:
        print(f"Cell-type mapping template: {resolve_path(section['mapping_csv'])}")


def normalized_log1p(counts, target_sum=None):
    matrix = counts.tocsr(copy=True).astype(np.float64)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    positive = totals > 0
    if target_sum is None:
        target_sum = float(np.median(totals[positive]))
    scale = np.zeros_like(totals, dtype=np.float64)
    scale[positive] = float(target_sum) / totals[positive]
    matrix = sp.diags(scale) @ matrix
    matrix.data = np.log1p(matrix.data)
    return matrix.tocsr(), float(target_sum)


def official_style_spatial_graph(coords, n_neighbors):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.all(np.isfinite(coords)):
        raise ValueError("Spatial coordinates must be a finite n_spots x 2 matrix")
    estimator = NearestNeighbors(n_neighbors=int(n_neighbors), metric="euclidean")
    estimator.fit(coords)
    distances, indices = estimator.kneighbors()
    cutoff = float(np.median(distances) * 1.3)
    rows = indices.reshape(-1)
    cols = np.repeat(np.arange(coords.shape[0]), int(n_neighbors))
    values = distances.reshape(-1)
    keep = values < cutoff
    graph = sp.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float64), (rows[keep], cols[keep])),
        shape=(coords.shape[0], coords.shape[0]),
    )
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    normalize(graph, norm="l1", axis=1, copy=False)
    return graph, cutoff


def rank_spatial_genes(st_adata, counts, n_top, n_neighbors, chunk_size=512):
    if "spatial" in st_adata.obsm:
        coords = st_adata.obsm["spatial"]
    elif {"x_coord", "y_coord"}.issubset(st_adata.obs.columns):
        coords = st_adata.obs[["x_coord", "y_coord"]].to_numpy(dtype=float)
    else:
        raise KeyError("Y7_T ST data has no spatial coordinates")
    expression, target_sum = normalized_log1p(counts, target_sum=None)
    graph, cutoff = official_style_spatial_graph(coords, n_neighbors)
    n_spots, n_genes = expression.shape
    s0 = float(graph.sum())
    scores = np.full(n_genes, np.nan, dtype=np.float64)
    for start in range(0, n_genes, chunk_size):
        stop = min(start + chunk_size, n_genes)
        values = expression[:, start:stop].toarray().astype(np.float64, copy=False)
        values -= values.mean(axis=0, keepdims=True)
        denominator = np.square(values).sum(axis=0)
        numerator = (values * (graph @ values)).sum(axis=0)
        valid = denominator > 0
        block = np.full(stop - start, np.nan, dtype=np.float64)
        block[valid] = (n_spots / s0) * numerator[valid] / denominator[valid]
        scores[start:stop] = block
    finite = np.flatnonzero(np.isfinite(scores))
    order = finite[np.argsort(scores[finite])[::-1]]
    top_indices = order[: int(n_top)]
    ranked = pd.DataFrame(
        {
            "gene": st_adata.var_names.to_numpy()[order].astype(str),
            "morans_i": scores[order],
            "rank": np.arange(1, len(order) + 1),
        }
    )
    return ranked, ranked.iloc[: len(top_indices)]["gene"].tolist(), target_sum, cutoff


def rank_reference_hvgs(reference, n_top, flavor):
    sc.pp.highly_variable_genes(
        reference,
        n_top_genes=int(n_top),
        flavor=str(flavor),
        subset=False,
        inplace=True,
    )
    selected = reference.var.index[reference.var["highly_variable"].to_numpy()].astype(str).tolist()
    if "highly_variable_rank" in reference.var:
        selected.sort(
            key=lambda gene: float(reference.var.loc[gene, "highly_variable_rank"])
            if pd.notna(reference.var.loc[gene, "highly_variable_rank"])
            else np.inf
        )
    return selected


def load_mapping(path, reference_labels, allowed_classes):
    path = require_file(path, "reviewed cell-type mapping CSV")
    mapping = pd.read_csv(path, dtype=str).fillna("")
    missing_columns = set(MAPPING_COLUMNS) - set(mapping.columns)
    if missing_columns:
        raise ValueError(f"Mapping CSV lacks columns: {sorted(missing_columns)}")
    if mapping["source_cell_type"].duplicated().any():
        duplicated = mapping.loc[mapping["source_cell_type"].duplicated(), "source_cell_type"].tolist()
        raise ValueError(f"Mapping CSV has duplicate source cell types: {duplicated[:10]}")
    valid_targets = set(allowed_classes) | {"Exclude"}
    bad_targets = sorted(set(mapping["spatialmeta_four_class"]) - valid_targets)
    if bad_targets:
        raise ValueError(f"Mapping CSV has invalid target classes: {bad_targets}")
    unreviewed = mapping.loc[mapping["decision"].str.lower() != "reviewed", "source_cell_type"].tolist()
    if unreviewed:
        raise ValueError(
            "Every mapping row must have decision=reviewed before training; "
            f"unreviewed examples: {unreviewed[:10]}"
        )
    mapping_index = mapping.set_index("source_cell_type")["spatialmeta_four_class"].to_dict()
    reference_labels = set(map(str, reference_labels))
    missing = sorted(reference_labels - set(mapping_index))
    extra = sorted(set(mapping_index) - reference_labels)
    if missing or extra:
        raise ValueError(f"Mapping/reference mismatch; missing={missing[:10]}, extra={extra[:10]}")
    return mapping_index, mapping, path


def load_spatial_and_fixed(config):
    section = config["spatial"]
    st_path = require_file(resolve_path(section["raw_st_h5ad"]), "canonical Y7_T ST counts")
    fixed_path = require_file(resolve_path(section["fixed_result_h5ad"]), "frozen BioSFIB result")
    st_adata = ad.read_h5ad(st_path)
    fixed = ad.read_h5ad(fixed_path)
    expected_spots = int(section["expected_spots"])
    if st_adata.n_obs != expected_spots or fixed.n_obs != expected_spots:
        raise ValueError(
            f"Expected {expected_spots} Y7_T spots, found ST={st_adata.n_obs}, fixed={fixed.n_obs}"
        )
    st_adata.obs_names = st_adata.obs_names.astype(str)
    fixed.obs_names = fixed.obs_names.astype(str)
    if st_adata.obs_names.has_duplicates or fixed.obs_names.has_duplicates:
        raise ValueError("Y7_T observation identifiers must be unique")
    if set(st_adata.obs_names) != set(fixed.obs_names):
        raise ValueError("Canonical ST and frozen BioSFIB result do not contain identical spots")
    st_adata = st_adata[fixed.obs_names].copy()
    cluster_key = str(section["fixed_cluster_key"])
    if cluster_key not in fixed.obs:
        raise KeyError(f"Frozen result lacks cluster column: {cluster_key}")
    clusters = fixed.obs[cluster_key].astype(str)
    if clusters.nunique() != int(section["expected_clusters"]):
        raise ValueError(
            f"Expected {section['expected_clusters']} fixed clusters, found {clusters.nunique()}"
        )
    counts_location, counts, _, diagnostics = select_counts(st_adata)
    if counts_location != "X":
        raise ValueError(f"Canonical Y7_T ST count contract changed: selected {counts_location}")
    return st_adata, counts, fixed, clusters, st_path, fixed_path, diagnostics


def marker_scores(st_adata, counts, markers):
    missing = sorted(
        gene for genes in markers.values() for gene in genes if gene not in st_adata.var_names
    )
    if missing:
        raise KeyError(f"SpatialMETA marker genes are absent from canonical ST data: {missing}")
    expression, _ = normalized_log1p(counts, target_sum=10_000.0)
    scores = pd.DataFrame(index=st_adata.obs_names)
    gene_scores = {}
    for group, genes in markers.items():
        indices = st_adata.var_names.get_indexer(genes)
        values = expression[:, indices].toarray().astype(np.float64)
        means = values.mean(axis=0, keepdims=True)
        scales = values.std(axis=0, ddof=0, keepdims=True)
        scales[scales == 0] = 1.0
        z_values = (values - means) / scales
        scores[group] = z_values.mean(axis=1)
        gene_scores[group] = {
            gene: float(np.mean(z_values[:, index])) for index, gene in enumerate(genes)
        }
    return scores, gene_scores


def winner_and_margin(row, class_order):
    values = row[list(class_order)].to_numpy(dtype=float)
    order = np.argsort(values)[::-1]
    winner = class_order[int(order[0])]
    margin = float(values[order[0]] - values[order[1]])
    return winner, margin


def build_cluster_evidence(
    clusters,
    broad_proportions,
    marker_score_frame,
    class_order,
    minimum_deconv_margin,
    minimum_marker_margin,
    uncertain_label,
):
    cluster_name = "biosfib_cluster"
    deconv = broad_proportions.copy()
    deconv[cluster_name] = clusters.to_numpy()
    deconv = deconv.groupby(cluster_name, sort=False)[list(class_order)].mean()
    deconv = deconv.div(deconv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    marker = marker_score_frame.copy()
    marker[cluster_name] = clusters.to_numpy()
    marker = marker.groupby(cluster_name, sort=False)[list(class_order)].mean()

    rows = []
    for cluster in deconv.index:
        deconv_winner, deconv_margin = winner_and_margin(deconv.loc[cluster], class_order)
        marker_winner, marker_margin = winner_and_margin(marker.loc[cluster], class_order)
        concordant = deconv_winner == marker_winner
        accepted = (
            concordant
            and deconv_margin >= float(minimum_deconv_margin)
            and marker_margin >= float(minimum_marker_margin)
        )
        row = {
            cluster_name: str(cluster),
            "n_spots": int((clusters == str(cluster)).sum()),
            "deconv_winner": deconv_winner,
            "deconv_margin": deconv_margin,
            "marker_winner": marker_winner,
            "marker_margin": marker_margin,
            "evidence_concordant": bool(concordant),
            "four_class_candidate": deconv_winner if accepted else uncertain_label,
            "annotation_status": "concordant" if accepted else "mixed_or_uncertain",
        }
        for group in class_order:
            row[f"destvi_{group}"] = float(deconv.loc[cluster, group])
            row[f"marker_{group}"] = float(marker.loc[cluster, group])
        rows.append(row)
    evidence = pd.DataFrame(rows)
    evidence["cluster_sort"] = pd.to_numeric(evidence[cluster_name], errors="coerce")
    evidence = evidence.sort_values(["cluster_sort", cluster_name]).drop(columns="cluster_sort")
    return evidence


def attach_marker_guided_annotation(
    annotated,
    evidence,
    clusters,
    class_order,
    minimum_marker_margin,
    uncertain_label,
):
    evidence = evidence.copy()
    evidence["biosfib_cluster"] = evidence["biosfib_cluster"].astype(str)
    clusters = clusters.astype(str)
    marker_accepted = evidence["marker_margin"] >= float(minimum_marker_margin)
    evidence["marker_guided_candidate"] = np.where(
        marker_accepted, evidence["marker_winner"], uncertain_label
    )
    evidence["marker_guided_status"] = np.where(
        ~marker_accepted,
        "mixed_or_uncertain",
        np.where(evidence["evidence_concordant"], "corroborated", "marker_guided"),
    )
    labels = evidence.set_index("biosfib_cluster")["marker_guided_candidate"].to_dict()
    statuses = evidence.set_index("biosfib_cluster")["marker_guided_status"].to_dict()
    annotated.obs["biosfib_four_class_marker_guided"] = pd.Categorical(
        clusters.map(labels).to_numpy(),
        categories=list(class_order) + [uncertain_label],
    )
    annotated.obs["biosfib_four_class_marker_guided_status"] = pd.Categorical(
        clusters.map(statuses).to_numpy()
    )
    return annotated, evidence


def plot_annotation_map(annotated, label_key, png_path, pdf_path, title=None):
    if "spatial" in annotated.obsm:
        coords = np.asarray(annotated.obsm["spatial"])
    else:
        coords = annotated.obs[["x_coord", "y_coord"]].to_numpy(dtype=float)
    colors = {
        "Imm": "#D55E00",
        "Endo": "#0072B2",
        "Stro": "#009E73",
        "Mal": "#CC79A7",
        "Mixed/Uncertain": "#B3B3B3",
    }
    labels = annotated.obs[label_key].astype(str)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for label in colors:
        mask = labels.to_numpy() == label
        if np.any(mask):
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=10,
                c=colors[label],
                label=label,
                linewidths=0,
                rasterized=True,
            )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(title or "BioSFIB K14 annotation\nSpatialMETA DestVI + marker protocol")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=1.6)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def run_destvi(config, config_path, overwrite):
    import importlib.metadata as metadata
    import scvi

    reference_section = config["reference"]
    spatial_section = config["spatial"]
    feature_section = config["feature_selection"]
    model_section = config["destvi"]
    annotation_section = config["annotation"]
    output_section = config["output"]

    output_dir = resolve_path(output_section["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        key: require_new_or_overwrite(output_dir / output_section[key], overwrite)
        for key in (
            "annotated_h5ad",
            "cluster_evidence_csv",
            "spot_proportions_csv",
            "feature_manifest_json",
            "provenance_json",
            "annotation_map_png",
            "annotation_map_pdf",
            "marker_guided_map_png",
            "marker_guided_map_pdf",
        )
    }
    condscvi_dir = output_dir / "condscvi_model"
    destvi_dir = output_dir / "destvi_model"
    resume_condscvi = bool(model_section.get("resume_condscvi_if_available", True))
    if condscvi_dir.exists() and not (resume_condscvi or overwrite):
        raise FileExistsError(
            f"CondSCVI model directory exists; enable resume or pass --overwrite: {condscvi_dir}"
        )
    if destvi_dir.exists() and not overwrite:
        raise FileExistsError(f"DestVI model directory exists; pass --overwrite: {destvi_dir}")

    reference_path = require_file(resolve_path(reference_section["prepared_h5ad"]), "prepared reference")
    reference = ad.read_h5ad(reference_path)
    if not matrix_is_raw_counts(reference.X):
        raise ValueError("Prepared reference X is not raw nonnegative integer counts")
    cell_type_key = reference_section.get("cell_type_key")
    if not cell_type_key or cell_type_key not in reference.obs:
        raise KeyError("Set reference.cell_type_key after inspecting the reference inventory")
    if reference.obs[cell_type_key].isna().any():
        raise ValueError(f"Reference cell-type column contains missing values: {cell_type_key}")
    reference.obs[cell_type_key] = pd.Categorical(reference.obs[cell_type_key].astype(str))
    reference_labels = reference.obs[cell_type_key].cat.categories.astype(str).tolist()

    class_order = list(annotation_section["allowed_classes"])
    mapping_index, mapping_frame, mapping_path = load_mapping(
        resolve_path(reference_section["mapping_csv"]), reference_labels, class_order
    )
    st_adata, st_counts, fixed, clusters, st_path, fixed_path, st_diagnostics = load_spatial_and_fixed(config)

    svg_ranking, st_svg, st_target_sum, graph_cutoff = rank_spatial_genes(
        st_adata,
        st_counts,
        feature_section["st_top_svg"],
        feature_section["st_spatial_neighbors"],
    )
    reference_hvg = rank_reference_hvgs(
        reference,
        feature_section["scrna_top_hvg"],
        feature_section["scrna_hvg_flavor"],
    )
    hvg_set = set(reference_hvg)
    intersected = [gene for gene in st_svg if gene in hvg_set]
    if len(intersected) < int(feature_section["minimum_intersection"]):
        raise ValueError(
            f"Only {len(intersected)} genes overlap between top SVGs and HVGs; "
            f"minimum is {feature_section['minimum_intersection']}"
        )

    reference_train = reference[:, intersected].copy()
    spatial_train = ad.AnnData(
        X=st_adata[:, intersected].X.copy(),
        obs=st_adata.obs.copy(),
        var=st_adata[:, intersected].var.copy(),
    )
    reference_train.X = reference_train.X.astype(np.float32)
    spatial_train.X = spatial_train.X.astype(np.float32)

    seed = int(model_section["seed"])
    scvi.settings.seed = seed
    scvi.model.CondSCVI.setup_anndata(reference_train, labels_key=cell_type_key)
    train_kwargs = {
        "accelerator": str(model_section["accelerator"]),
        "devices": model_section["devices"],
    }
    condscvi_was_resumed = resume_condscvi and condscvi_dir.exists()
    if condscvi_was_resumed:
        print(f"Reusing completed CondSCVI model: {condscvi_dir}")
        rna_model = scvi.model.CondSCVI.load(condscvi_dir, adata=reference_train)
    else:
        # scvi-tools 1.4.3 DestVI.from_rna_model expects the CondSCVI prior to be
        # present in init_params_. Passing the module default explicitly avoids a
        # KeyError that otherwise occurs after CondSCVI training completes.
        rna_model = scvi.model.CondSCVI(reference_train, prior="normal")
        rna_model.train(max_epochs=int(model_section["condscvi_max_epochs"]), **train_kwargs)
        # Persist the expensive reference model before constructing/training
        # DestVI so a downstream failure does not discard the completed fit.
        rna_model.save(condscvi_dir, overwrite=overwrite, save_anndata=False)

    scvi.model.DestVI.setup_anndata(spatial_train)
    accelerator = str(model_section["accelerator"]).lower()
    use_cuda = torch.cuda.is_available() and accelerator in {"auto", "cuda", "gpu"}
    target_device = torch.device("cuda:0" if use_cuda else "cpu")
    spatial_model = destvi_from_rna_model_compat(
        spatial_train, rna_model, scvi, target_device
    )
    spatial_model.train(max_epochs=int(model_section["destvi_max_epochs"]), **train_kwargs)
    proportions = spatial_model.get_proportions()
    proportions.index = spatial_train.obs_names.astype(str)
    proportions.columns = proportions.columns.astype(str)
    if list(proportions.index) != list(fixed.obs_names.astype(str)):
        raise ValueError("DestVI proportion rows do not match the frozen BioSFIB spot order")

    spatial_model.save(destvi_dir, overwrite=overwrite, save_anndata=False)

    broad = pd.DataFrame(0.0, index=proportions.index, columns=class_order)
    for fine_label in proportions.columns:
        target = mapping_index[fine_label]
        if target != "Exclude":
            broad[target] += proportions[fine_label].to_numpy(dtype=float)
    marker_frame, marker_diagnostics = marker_scores(
        st_adata, st_counts, annotation_section["markers"]
    )
    evidence = build_cluster_evidence(
        clusters,
        broad,
        marker_frame,
        class_order,
        annotation_section["minimum_deconvolution_margin"],
        annotation_section["minimum_marker_margin"],
        annotation_section["uncertain_label"],
    )
    candidate_by_cluster = evidence.set_index("biosfib_cluster")["four_class_candidate"].to_dict()
    status_by_cluster = evidence.set_index("biosfib_cluster")["annotation_status"].to_dict()

    annotated = fixed.copy()
    annotated.obsm["X_destvi_proportions"] = proportions.to_numpy(dtype=np.float32)
    annotated.uns["destvi_proportion_columns"] = proportions.columns.to_list()
    for group in class_order:
        annotated.obs[f"destvi_broad_{group}"] = broad[group].to_numpy(dtype=float)
        annotated.obs[f"spatialmeta_marker_score_{group}"] = marker_frame[group].to_numpy(dtype=float)
    annotated.obs["destvi_top_cell_type"] = pd.Categorical(proportions.idxmax(axis=1).to_numpy())
    annotated.obs["destvi_four_class_deconv"] = pd.Categorical(broad.idxmax(axis=1).to_numpy())
    annotated.obs["biosfib_four_class_candidate"] = pd.Categorical(
        clusters.map(candidate_by_cluster).to_numpy(),
        categories=class_order + [annotation_section["uncertain_label"]],
    )
    annotated.obs["biosfib_four_class_status"] = pd.Categorical(
        clusters.map(status_by_cluster).to_numpy()
    )
    annotated, evidence = attach_marker_guided_annotation(
        annotated,
        evidence,
        clusters,
        class_order,
        annotation_section["minimum_marker_margin"],
        annotation_section["uncertain_label"],
    )
    annotated.uns["four_class_annotation"] = {
        "method": "independent BioSFIB-domain annotation following SpatialMETA DestVI and marker protocol",
        "fixed_cluster_key": spatial_section["fixed_cluster_key"],
        "allowed_classes": class_order,
        "markers": annotation_section["markers"],
        "ground_truth": False,
        "valid_for_ari": False,
        "spamv_used": False,
    }
    annotated.uns["four_class_marker_guided_annotation"] = {
        "method": "cluster marker winner following the SpatialMETA marker panel, with DestVI retained as auxiliary evidence",
        "fixed_cluster_key": spatial_section["fixed_cluster_key"],
        "allowed_classes": class_order,
        "minimum_marker_margin": float(annotation_section["minimum_marker_margin"]),
        "provisional": True,
        "ground_truth": False,
        "valid_for_ari": False,
        "spamv_used": False,
    }
    annotated.write_h5ad(output_paths["annotated_h5ad"], compression="gzip")
    evidence.to_csv(output_paths["cluster_evidence_csv"], index=False, encoding="utf-8-sig")
    proportions.to_csv(output_paths["spot_proportions_csv"], index_label="spot", encoding="utf-8-sig")

    feature_manifest = {
        "st_top_svg": st_svg,
        "reference_top_hvg": reference_hvg,
        "intersection": intersected,
        "counts_and_feature_contract": {
            "st_normalization_for_svg": feature_section["st_normalization"],
            "st_normalization_target_sum_observed": st_target_sum,
            "st_neighbor_cutoff": graph_cutoff,
            "reference_hvg_flavor": feature_section["scrna_hvg_flavor"],
            "model_input": "raw counts restricted to the SVG/HVG intersection",
        },
        "top_svg_morans_i": svg_ranking.head(int(feature_section["st_top_svg"])).to_dict("records"),
    }
    write_json(output_paths["feature_manifest_json"], feature_manifest)

    provenance = {
        "config": str(config_path),
        "config_sha256": file_hash(config_path),
        "reference_h5ad": str(reference_path),
        "reference_h5ad_sha256": file_hash(reference_path),
        "mapping_csv": str(mapping_path),
        "mapping_csv_sha256": file_hash(mapping_path),
        "mapping_rows": mapping_frame.to_dict("records"),
        "spatial_st_h5ad": str(st_path),
        "spatial_st_h5ad_sha256": file_hash(st_path),
        "fixed_result_h5ad": str(fixed_path),
        "fixed_result_h5ad_sha256": file_hash(fixed_path),
        "fixed_cluster_key": spatial_section["fixed_cluster_key"],
        "fixed_cluster_count": int(clusters.nunique()),
        "st_count_diagnostics": st_diagnostics,
        "marker_diagnostics": marker_diagnostics,
        "seed": seed,
        "condscvi_was_resumed": condscvi_was_resumed,
        "marker_guided_provisional_annotation": {
            "column": "biosfib_four_class_marker_guided",
            "ground_truth": False,
            "valid_for_ari": False,
            "selection_or_retuning": "none",
        },
        "software": {
            "python": sys.version,
            "scvi-tools": metadata.version("scvi-tools"),
            "scanpy": metadata.version("scanpy"),
            "anndata": metadata.version("anndata"),
        },
        "selection_or_retuning": "none",
        "ground_truth": False,
        "valid_for_ari": False,
        "spamv_used": False,
    }
    write_json(output_paths["provenance_json"], provenance)
    plot_annotation_map(
        annotated,
        "biosfib_four_class_candidate",
        output_paths["annotation_map_png"],
        output_paths["annotation_map_pdf"],
    )
    plot_annotation_map(
        annotated,
        "biosfib_four_class_marker_guided",
        output_paths["marker_guided_map_png"],
        output_paths["marker_guided_map_pdf"],
        title="BioSFIB K14 provisional four-class annotation\nSpatialMETA marker-guided; DestVI auxiliary",
    )
    print(f"Annotated H5AD: {output_paths['annotated_h5ad']}")
    print(f"Cluster evidence: {output_paths['cluster_evidence_csv']}")
    print(f"Annotation map: {output_paths['annotation_map_png']}")


def main():
    args = parse_args()
    config, config_path = load_config(args.config)
    if args.stage == "prepare-reference":
        prepare_reference(config, args.overwrite)
    elif args.stage == "run-destvi":
        run_destvi(config, config_path, args.overwrite)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
