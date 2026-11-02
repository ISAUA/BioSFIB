import os
import sys
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sanitize_thread_env():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        val = os.environ.get(key)
        if val is None:
            os.environ[key] = "1"
            continue
        val = val.strip()
        if not val.isdigit() or int(val) <= 0:
            os.environ[key] = "1"


_sanitize_thread_env()

import argparse
import logging
import time
import warnings
import torch
import yaml
import numpy as np
import pandas as pd
import scanpy as sc
import joblib
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

# 引入预处理模块
from sf_model.preprocess.io import (
    read_mtx_to_adata,
    add_spatial_info,
    read_10x_h5_multiome,
    add_spatial_info_csv,
    read_h5ad_rna_atac,
    _ensure_spatial_from_obs,
    select_adata_matrix,
)
from sf_model.preprocess.rna_process import process_rna_pipeline
from sf_model.preprocess.atac_process import (
    canonicalize_peak_names,
    custom_tf_idf,
    match_peak_intervals_one_to_one,
    process_atac_pipeline,
)
from sf_model.preprocess.adt_process import process_adt_pipeline, read_adt_h5ad, resolve_adt_path
from sf_model.preprocess.sm_process import read_sm_h5ad, process_sm_pipeline, resolve_sm_path
from sf_model.config import load_config as load_config_with_base
from sf_model.utils import (
    build_spatial_graph,
    configure_utf8_console,
    set_seed,
)

def load_config(config_path="configs/misar/e11_0_s1.yaml"):
    return load_config_with_base(config_path)


def _sha256_file(path):
    if path is None or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_strings(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_array(values):
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def resolve_reducer_dirs(args, config):
    """Resolve reducer mode from CLI or a self-contained preprocess config."""
    preprocess_cfg = config.get("preprocess", {}) or {}
    config_save_dir = preprocess_cfg.get("save_pca_dir")
    config_load_dir = preprocess_cfg.get("load_pca_dir")
    save_pca_dir = args.save_pca_dir or config_save_dir
    load_pca_dir = args.load_pca_dir or config_load_dir
    if save_pca_dir and load_pca_dir:
        raise ValueError(
            "Preprocessing cannot save and load PCA reducers in the same run. "
            "Check CLI --save-pca-dir/--load-pca-dir and preprocess.save_pca_dir/"
            "preprocess.load_pca_dir."
        )
    return save_pca_dir, load_pca_dir


def validate_reference_feature_tensors(
    candidate,
    reference_path,
    keys=("rna_feat", "atac_feat", "coords"),
):
    """Require a rebuilt source panel to reproduce its formal S1 tensors exactly."""
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(
            f"Reference processed tensor does not exist: {reference_path}"
        )
    try:
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    except TypeError:
        reference = torch.load(reference_path, map_location="cpu")

    checked = []
    for key in keys:
        if key not in candidate or key not in reference:
            raise KeyError(
                f"Reference tensor equivalence requires key {key!r} in both artifacts."
            )
        candidate_value = candidate[key].detach().cpu()
        reference_value = reference[key].detach().cpu()
        if candidate_value.shape != reference_value.shape:
            raise ValueError(
                f"Reference tensor mismatch for {key}: candidate shape "
                f"{tuple(candidate_value.shape)} != reference shape "
                f"{tuple(reference_value.shape)}."
            )
        if candidate_value.dtype != reference_value.dtype:
            raise ValueError(
                f"Reference tensor mismatch for {key}: candidate dtype "
                f"{candidate_value.dtype} != reference dtype {reference_value.dtype}."
            )
        if not torch.equal(candidate_value, reference_value):
            max_abs = None
            if candidate_value.is_floating_point():
                max_abs = float(
                    (candidate_value - reference_value).abs().max().item()
                )
            detail = "" if max_abs is None else f"; max_abs_diff={max_abs:.9g}"
            raise ValueError(
                f"Reference tensor mismatch for {key}: values are not exactly equal"
                f"{detail}."
            )
        checked.append(key)
    return {
        "reference_path": os.path.abspath(reference_path),
        "keys": checked,
        "exact": True,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw spatial multi-omics data")
    parser.add_argument("--config", default="configs/misar/e11_0_s1.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--save-pca-dir",
        default=None,
        help="Optional directory to save fitted modality reducers (S1 mode)",
    )
    parser.add_argument(
        "--load-pca-dir",
        default=None,
        help="Optional directory to load pre-fitted modality reducers and transform only (S2 mode)",
    )
    return parser.parse_args()


def infer_ground_truth_key(obs_df, preferred_key=None):
    if preferred_key and preferred_key in obs_df.columns:
        return preferred_key

    candidates = [
        'ground_truth', 'groundtruth', 'gt', 'true_label',
        'combined_clusters', 'combined_cluster',
        'combined_clusters_annotation', 'cluster', 'clusters',
        'cell_type', 'celltype', 'annotation', 'annot', 'label', 'labels',
    ]
    lower_to_col = {c.lower(): c for c in obs_df.columns}
    for c in candidates:
        if c in lower_to_col:
            return lower_to_col[c]
    return None


def infer_ground_truth_from_obsm(adata, obsm_key):
    if not obsm_key:
        return None
    key = str(obsm_key).strip()
    if key == "" or key not in adata.obsm:
        return None
    values = np.asarray(adata.obsm[key])
    if values.ndim == 1:
        return values.astype(str)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"ground_truth_obsm_key={key!r} must reference a 1D or 2D obsm array.")
    row_sum = values.sum(axis=1)
    labels = []
    for idx, total in enumerate(row_sum):
        if np.isclose(total, 0.0):
            labels.append(f"{key}_background")
        else:
            labels.append(f"{key}_{int(np.nanargmax(values[idx]))}")
    return np.asarray(labels, dtype=str)


def load_ground_truth_txt(label_path, obs_names):
    if not label_path:
        return None
    path = os.path.expanduser(str(label_path))
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ground_truth_txt not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.replace(",", " ").replace("\t", " ").split()
            rows.append(parts)
    if not rows:
        raise ValueError(f"ground_truth_txt is empty after removing comments/blank lines: {path}")

    if all(len(row) == 1 for row in rows):
        labels = np.asarray([row[0] for row in rows], dtype=str)
        if labels.shape[0] != len(obs_names):
            raise ValueError(
                f"ground_truth_txt has {labels.shape[0]} labels but RNA has {len(obs_names)} obs."
            )
        obs = [str(x) for x in obs_names]
        if all(value.isdigit() for value in obs):
            obs_idx = np.asarray([int(value) for value in obs], dtype=int)
            if (
                obs_idx.shape[0] == labels.shape[0]
                and len(np.unique(obs_idx)) == labels.shape[0]
                and int(obs_idx.min()) == 0
                and int(obs_idx.max()) == labels.shape[0] - 1
            ):
                return labels[obs_idx]
        return labels

    if not all(len(row) >= 2 for row in rows):
        raise ValueError("ground_truth_txt must be either one label per row or barcode/label rows.")
    barcodes = pd.Index([row[0] for row in rows], dtype=str)
    labels = pd.Index([row[1] for row in rows], dtype=str)
    if barcodes.duplicated().any():
        raise ValueError("ground_truth_txt contains duplicate barcode rows.")
    label_map = pd.Series(labels.to_numpy(), index=barcodes.to_numpy())
    obs = pd.Index([str(x) for x in obs_names])
    missing = obs.difference(label_map.index)
    if len(missing) > 0:
        examples = missing[:5].tolist()
        raise ValueError(f"ground_truth_txt is missing {len(missing)} RNA obs labels; examples={examples}")
    return label_map.loc[obs].astype(str).to_numpy()


def align_adata_features_hard(
    adata,
    target_features,
    modality_name="modality",
    min_overlap=0.0,
):
    """
    Hard-align adata features to target_features with strict 1:1 semantics:
    - Keep intersection features
    - Zero-pad missing features
    - Return matrix reordered exactly as target_features
    """
    target_features = [str(x) for x in target_features]
    if len(target_features) == 0:
        raise ValueError(f"[{modality_name}] target_features is empty.")
    if len(set(target_features)) != len(target_features):
        raise ValueError(f"[{modality_name}] target_features contains duplicate IDs.")

    target_set = set(target_features)
    current_features = adata.var_names.astype(str).tolist()
    if len(set(current_features)) != len(current_features):
        raise ValueError(f"[{modality_name}] current feature IDs are not unique.")
    current_set = set(current_features)

    present = [f for f in target_features if f in current_set]
    missing = [f for f in target_features if f not in current_set]
    overlap_fraction = len(present) / len(target_features)
    if overlap_fraction < float(min_overlap):
        raise ValueError(
            f"[{modality_name}] feature overlap {overlap_fraction:.4f} is below "
            f"required minimum {float(min_overlap):.4f}."
        )

    adata_sub = adata[:, present].copy()
    X_sub = adata_sub.X

    if sparse.issparse(X_sub):
        X_sub = X_sub.tocsr().astype(np.float32)
    else:
        X_sub = np.asarray(X_sub, dtype=np.float32)

    if len(missing) > 0:
        X_missing = sparse.csr_matrix((adata_sub.n_obs, len(missing)), dtype=np.float32)
        if sparse.issparse(X_sub):
            X_aligned = sparse.hstack([X_sub, X_missing], format="csr")
        else:
            X_aligned = np.hstack([X_sub, np.zeros((adata_sub.n_obs, len(missing)), dtype=np.float32)])
    else:
        X_aligned = X_sub

    final_var_names = present + missing
    aligned = sc.AnnData(X=X_aligned, obs=adata_sub.obs.copy())
    aligned.var_names = final_var_names

    # Copy spatial coordinates (and any other obsm slots) for downstream graph build.
    for key, val in adata_sub.obsm.items():
        aligned.obsm[key] = val.copy()

    # Enforce exact target order (present+missing already follows target order; keep explicit for safety).
    aligned = aligned[:, target_features].copy()
    aligned.uns["feature_alignment"] = {
        "target_count": int(len(target_features)),
        "present_count": int(len(present)),
        "missing_count": int(len(missing)),
        "overlap_fraction": float(overlap_fraction),
        "minimum_required": float(min_overlap),
    }

    print(
        f"   [{modality_name}] Hard-align summary: target={len(target_features)} | "
        f"present={len(present)} | missing(padded)={len(missing)} | "
        f"overlap={overlap_fraction:.4f}"
    )
    return aligned


def align_adata_peaks_by_interval(
    adata,
    target_features,
    min_overlap=0.0,
    min_iou=0.5,
):
    """Project independently called ATAC peaks onto a fixed target panel.

    Each target interval receives at most one current interval and each current
    interval is used at most once. Missing target intervals are zero padded.
    """
    target_features = [str(value) for value in target_features]
    current_features = adata.var_names.astype(str).tolist()
    target_to_current, match_iou, stats = match_peak_intervals_one_to_one(
        current_features,
        target_features,
        min_iou=min_iou,
    )
    overlap_fraction = float(stats["overlap_fraction"])
    stats["minimum_required"] = float(min_overlap)
    if overlap_fraction < float(min_overlap):
        raise ValueError(
            f"[ATAC] interval feature overlap {overlap_fraction:.4f} is below "
            f"required minimum {float(min_overlap):.4f} at min_iou={float(min_iou):.4f}."
        )

    matched_targets = [
        target_index for target_index, current_index in enumerate(target_to_current)
        if current_index >= 0
    ]
    matched_current = [target_to_current[index] for index in matched_targets]
    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)
        projection = sparse.coo_matrix(
            (
                np.ones(len(matched_targets), dtype=np.float32),
                (matched_current, matched_targets),
            ),
            shape=(adata.n_vars, len(target_features)),
            dtype=np.float32,
        ).tocsr()
        X_aligned = X @ projection
    else:
        X = np.asarray(X, dtype=np.float32)
        X_aligned = np.zeros((adata.n_obs, len(target_features)), dtype=np.float32)
        if matched_targets:
            X_aligned[:, matched_targets] = X[:, matched_current]

    aligned = sc.AnnData(X=X_aligned, obs=adata.obs.copy())
    aligned.var_names = target_features
    source_ids = np.full(len(target_features), "", dtype=object)
    source_ids[matched_targets] = np.asarray(
        [current_features[index] for index in matched_current], dtype=object
    )
    aligned.var["source_peak_id"] = source_ids
    aligned.var["match_iou"] = np.asarray(match_iou, dtype=np.float32)
    aligned.var["match_type"] = np.asarray([
        "missing" if current_index < 0 else
        "exact" if current_features[current_index] == target_features[target_index] else
        "interval"
        for target_index, current_index in enumerate(target_to_current)
    ], dtype=object)
    for key, value in adata.obsm.items():
        aligned.obsm[key] = value.copy()
    aligned.uns["feature_alignment"] = stats

    print(
        f"   [ATAC] Interval-align summary: target={stats['target_count']} | "
        f"matched={stats['matched_count']} (exact={stats['exact_count']}, "
        f"shifted={stats['interval_shifted_count']}) | "
        f"missing(padded)={stats['missing_count']} | overlap={overlap_fraction:.4f} | "
        f"median_iou={stats['median_matched_iou']:.4f} | min_iou={float(min_iou):.4f}"
    )
    return aligned


def reduce_modality_features(
    X,
    n_components,
    seed,
    modality_name,
    reduce_cfg=None,
    save_model_path=None,
    load_model_path=None,
):
    """
    使用 PCA/TruncatedSVD 将模态特征降到固定维度。
    - 稀疏输入优先使用 TruncatedSVD，避免转 dense 导致内存开销过大。
    - 稠密输入使用 PCA。
    """
    _ = reduce_cfg

    n_components = int(n_components)
    if n_components <= 0:
        raise ValueError(f"{modality_name} n_components must be positive, got {n_components}.")

    n_samples = X.shape[0]
    n_features = X.shape[1]
    max_rank = max(1, min(n_samples, n_features))
    if n_components > max_rank:
        print(
            f"   \u26a0\ufe0f [{modality_name}] Requested n_components={n_components} exceeds max_rank={max_rank}; "
            f"using {max_rank} instead."
        )
        n_components = max_rank

    t0 = time.perf_counter()
    print(f"   [{modality_name}] Reducing features to {n_components} dims (deterministic-cpu)...")

    if load_model_path is not None and not os.path.exists(load_model_path):
        raise FileNotFoundError(
            f"[{modality_name}] required reducer is missing in load-and-transform mode: "
            f"{load_model_path}"
        )
    use_load_mode = load_model_path is not None

    if use_load_mode:
        print(f"   [{modality_name}] Loading reducer and transform-only: {load_model_path}")
        reducer = joblib.load(load_model_path)
        if sparse.issparse(X):
            Z = reducer.transform(X)
        else:
            X_dense = np.asarray(X, dtype=np.float32)
            Z = reducer.transform(X_dense)
    else:
        if sparse.issparse(X):
            reducer = TruncatedSVD(n_components=n_components, random_state=int(seed))
            Z = reducer.fit_transform(X)
        else:
            X_dense = np.asarray(X, dtype=np.float32)
            reducer = PCA(n_components=n_components, random_state=int(seed), svd_solver='auto')
            Z = reducer.fit_transform(X_dense)

        if save_model_path is not None:
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            joblib.dump(reducer, save_model_path)
            print(f"   [{modality_name}] Saved reducer: {save_model_path}")

    # if modality_name.strip().lower() == "atac" and isinstance(reducer, TruncatedSVD):
    #     if Z.shape[1] <= 1:
    #         raise ValueError("ATAC SVD output has <=1 component; cannot drop PC1.")
    #     Z = Z[:, 1:]

    dt = time.perf_counter() - t0
    print(f"   [{modality_name}] Dim reduction done in {dt:.2f}s")

    return np.asarray(Z, dtype=np.float32)


def resolve_train_log_path(save_dir):
    save_dir = save_dir.rstrip('/\\')
    if os.path.basename(save_dir) == 'checkpoints':
        return os.path.join(os.path.dirname(save_dir), "train.log")
    return os.path.join(save_dir, "train.log")


def setup_logger(log_path):
    logger = logging.getLogger("SFPreprocess")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def append_log_separator(log_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 88 + "\n")

def main():
    configure_utf8_console()
    print("\U0001f680 [Phase 1] Starting Data Preprocessing...")
    args = parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    save_dir = config['project']['save_dir']
    log_path = resolve_train_log_path(save_dir)
    logger = setup_logger(log_path)
    logger.info("[Phase 1] Starting Data Preprocessing...")

    raw_dir = config['data']['raw_path']
    processed_dir = config['data']['processed_path']
    files = config['data']['files']
    params = config['data']['parameters']
    rna_min_cells = params.get('rna_min_cells', 3)
    rna_target_sum = float(params.get('rna_target_sum', 1e4))
    atac_min_cells = params.get('atac_min_cells', 50)
    atac_target_sum = float(params.get('atac_target_sum', 1e4))
    tfidf_eps = float(params.get('tfidf_eps', 1e-6))
    seed = int(config['project'].get('seed', 42))
    reduce_cfg = params.get('reduce', {})
    n_freq_components = params.get('n_freq_components', config.get('model', {}).get('n_freq_components', None))
    compute_gft_basis = True
    use_rna_similarity_edge_weight = bool(params.get('use_rna_similarity_edge_weight', True))
    graph_edge_weight_mode = params.get('graph_edge_weight_mode', None)
    if graph_edge_weight_mode is None:
        graph_edge_weight_mode = "rna_similarity" if use_rna_similarity_edge_weight else "topology"
    graph_edge_weight_mode = str(graph_edge_weight_mode).strip().lower()
    rna_use_pca = bool(params.get('rna_use_pca', True))
    rna_apply_scale = bool(params.get('rna_apply_scale', False))
    rna_input_layer = str(params.get('rna_input_layer', 'X') or 'X').strip()
    sm_data_path = config.get('sm_data_path', None)
    sm_pre_cfg = config.get('sm_preprocess', {}) or {}
    sm_replace_atac = bool(sm_pre_cfg.get('replace_atac', False))
    sm_adata = None
    sm_feat_np = None
    adata_adt = None
    adt_feat_np = None
    adt_pre_cfg = config.get('adt_preprocess', {}) or {}
    use_adt = bool(adt_pre_cfg.get('use_adt', False))
    tri_modal_cfg = config.get('tri_modal', {}) or {}
    tri_modal_enabled = bool(tri_modal_cfg.get('enabled', False)) or str(
        config.get('model', {}).get('name', '')
    ).strip().lower() == 'biosfinetri'
    adt_data_path = adt_pre_cfg.get('adt_data_path', None)
    adt_pipeline = str(adt_pre_cfg.get('pipeline', 'legacy')).strip().lower()
    adt_apply_clr = bool(adt_pre_cfg.get('apply_clr', False))
    adt_apply_scale = bool(adt_pre_cfg.get('apply_scale', False))
    adt_clip_nonnegative = bool(adt_pre_cfg.get('clip_nonnegative', True))
    adt_min_cells = adt_pre_cfg.get('min_cells', None)
    adt_nonnegative_policy = adt_pre_cfg.get('nonnegative_policy', None)
    adt_use_pca = bool(adt_pre_cfg.get('use_pca', False))
    adt_input_layer = str(adt_pre_cfg.get('input_layer', 'X') or 'X').strip()
    feature_alignment_cfg = params.get('feature_alignment', {}) or {}
    rna_min_overlap = float(feature_alignment_cfg.get('rna_min_overlap', 0.0))
    atac_min_overlap = float(feature_alignment_cfg.get('atac_min_overlap', 0.0))
    atac_alignment_mode = str(
        feature_alignment_cfg.get('atac_alignment_mode', 'exact')
    ).strip().lower()
    atac_interval_min_iou = float(
        feature_alignment_cfg.get('atac_interval_min_iou', 0.5)
    )
    if atac_alignment_mode not in {'exact', 'genomic_interval'}:
        raise ValueError(
            "feature_alignment.atac_alignment_mode must be 'exact' or "
            f"'genomic_interval', got {atac_alignment_mode!r}."
        )
    if not 0.0 < atac_interval_min_iou <= 1.0:
        raise ValueError("feature_alignment.atac_interval_min_iou must be in (0, 1].")
    peak_canonicalization_stats = None

    os.makedirs(processed_dir, exist_ok=True)

    preprocess_cfg = config.get('preprocess', {}) or {}
    save_pca_dir, load_pca_dir = resolve_reducer_dirs(args, config)
    if save_pca_dir is not None:
        os.makedirs(save_pca_dir, exist_ok=True)
    if load_pca_dir is not None:
        if not os.path.isdir(load_pca_dir):
            raise FileNotFoundError(
                f"Configured reducer load directory does not exist: {load_pca_dir}"
            )

    rna_model_save_path = os.path.join(save_pca_dir, "pca_model_rna.pkl") if save_pca_dir else None
    atac_model_save_path = os.path.join(save_pca_dir, "pca_model_atac.pkl") if save_pca_dir else None
    rna_model_load_path = os.path.join(load_pca_dir, "pca_model_rna.pkl") if load_pca_dir else None
    atac_model_load_path = os.path.join(load_pca_dir, "pca_model_atac.pkl") if load_pca_dir else None

    fit_save_mode = (load_pca_dir is None) and (save_pca_dir is not None)
    load_align_mode = load_pca_dir is not None

    if sm_replace_atac and load_align_mode:
        raise ValueError("SM replace_atac mode is incompatible with --load-pca-dir.")
    if tri_modal_enabled and not use_adt:
        raise ValueError("tri_modal.enabled requires adt_preprocess.use_adt=true.")
    if use_adt and sm_replace_atac:
        raise ValueError("ADT mode is incompatible with sm_preprocess.replace_atac.")
    if use_adt and load_align_mode:
        raise ValueError("ADT mode is incompatible with --load-pca-dir (expects ATAC feature alignment).")

    if load_pca_dir is not None:
        logger.info(
            "PCA mode: load-and-transform | dir=%s | rna_model=%s | atac_model=%s",
            load_pca_dir,
            rna_model_load_path,
            atac_model_load_path,
        )
    elif save_pca_dir is not None:
        logger.info(
            "PCA mode: fit-and-save | dir=%s | rna_model=%s | atac_model=%s",
            save_pca_dir,
            rna_model_save_path,
            atac_model_save_path,
        )
    else:
        logger.info("PCA mode: fit-only (no save/load reducer path provided)")

    if fit_save_mode:
        logger.info("Feature mode: fit-and-save vars (S1)")
    elif load_align_mode:
        logger.info("Feature mode: load-and-hard-align vars (S2)")
    else:
        logger.info("Feature mode: standalone preprocess (no cross-domain var alignment)")

    # ==========================================
    # Step 1: 加载原始数据
    # ==========================================
    print("\n📦 Loading Raw Data...")
    logger.info("Loading raw data...")

    # 自动根据配置键选择预处理读取路径（优先级）：
    # 1) RNA+ATAC 双 h5ad
    # 2) MISAR: 10x h5 + spatial csv
    # 3) 旧版 mtx 输入
    if tri_modal_enabled:
        if not adt_data_path or str(adt_data_path).strip() == "":
            raise ValueError("adt_preprocess.adt_data_path is required when tri_modal.enabled is true.")

        rna_h5ad = os.path.join(raw_dir, files.get('rna_h5ad', 'adata_RNA.h5ad'))
        default_atac_h5ad = os.path.join(raw_dir, "adata_ATAC.h5ad")
        atac_h5ad = os.path.join(
            raw_dir,
            files.get('atac_h5ad', 'adata_ATAC.h5ad' if os.path.exists(default_atac_h5ad) else 'adata_Peak.h5ad')
        )
        if not os.path.exists(rna_h5ad):
            raise FileNotFoundError(f"Tri-modal RNA h5ad not found: {rna_h5ad}")
        if not os.path.exists(atac_h5ad):
            raise FileNotFoundError(f"Tri-modal ATAC h5ad not found: {atac_h5ad}")

        logger.info("Data loader: RNA h5ad + ADT h5ad + ATAC h5ad")
        adata_rna, adata_atac = read_h5ad_rna_atac(rna_h5ad, atac_h5ad)
        adt_data_path = resolve_adt_path(adt_data_path, raw_dir)
        print(f"   -> Reading ADT from {adt_data_path}...")
        adata_adt = read_adt_h5ad(adt_data_path)

        common_cells = adata_rna.obs_names.intersection(adata_atac.obs_names).intersection(adata_adt.obs_names)
        if len(common_cells) == 0:
            raise ValueError("No overlapping barcodes among RNA, ADT, and ATAC h5ad files.")
        if len(common_cells) < adata_rna.n_obs or len(common_cells) < adata_atac.n_obs or len(common_cells) < adata_adt.n_obs:
            print(f"Warning: Keeping {len(common_cells)} intersected cells from RNA/ADT/ATAC h5ad files.")

        adata_rna = adata_rna[common_cells, :].copy()
        adata_atac = adata_atac[common_cells, :].copy()
        adata_adt = adata_adt[common_cells, :].copy()
        adata_atac = adata_atac[adata_rna.obs_names, :].copy()
        adata_adt = adata_adt[adata_rna.obs_names, :].copy()
        adata_atac.obsm['spatial'] = adata_rna.obsm['spatial'].copy()
        adata_adt.obsm['spatial'] = adata_rna.obsm['spatial'].copy()

    elif use_adt:
        if not adt_data_path or str(adt_data_path).strip() == "":
            raise ValueError("adt_preprocess.adt_data_path is required when use_adt is true.")

        rna_h5ad = os.path.join(raw_dir, files.get('rna_h5ad', 'adata_RNA.h5ad'))
        if os.path.exists(rna_h5ad):
            logger.info("Data loader: RNA h5ad + ADT h5ad")
            print(f"   -> Reading RNA from {rna_h5ad}...")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.",
                    category=UserWarning,
                )
                adata_rna = sc.read_h5ad(rna_h5ad)
            adata_rna.obs_names = adata_rna.obs_names.astype(str)
            adata_rna.var_names = adata_rna.var_names.astype(str)
            adata_rna.var_names_make_unique()
            adata_rna = _ensure_spatial_from_obs(adata_rna)
            if 'spatial' not in adata_rna.obsm:
                raise ValueError("RNA h5ad is missing spatial coordinates (obsm['spatial']).")
        else:
            logger.info("Data loader: RNA mtx + ADT h5ad")
            print(f"   -> Reading RNA from {files['rna_mtx']}...")
            adata_rna = read_mtx_to_adata(
                os.path.join(raw_dir, files['rna_mtx']),
                os.path.join(raw_dir, files['rna_genes']),
                os.path.join(raw_dir, files['rna_barcodes'])
            )
            adata_rna = add_spatial_info(adata_rna, os.path.join(raw_dir, files['spatial']))

        adt_data_path = resolve_adt_path(adt_data_path, raw_dir)
        print(f"   -> Reading ADT from {adt_data_path}...")
        adata_atac = read_adt_h5ad(adt_data_path)

        common_cells = adata_rna.obs_names.intersection(adata_atac.obs_names)
        if len(common_cells) == 0:
            raise ValueError("No overlapping barcodes between RNA and ADT h5ad files.")
        if len(common_cells) < adata_rna.n_obs or len(common_cells) < adata_atac.n_obs:
            print(f"Warning: Keeping {len(common_cells)} intersected cells from RNA/ADT h5ad files.")

        adata_rna = adata_rna[common_cells, :].copy()
        adata_atac = adata_atac[common_cells, :].copy()
        adata_atac = adata_atac[adata_rna.obs_names, :].copy()
        adata_atac.obsm['spatial'] = adata_rna.obsm['spatial'].copy()

    elif sm_replace_atac:
        if not sm_data_path or str(sm_data_path).strip() == "":
            raise ValueError("sm_data_path is required when sm_preprocess.replace_atac is true.")

        rna_h5ad = os.path.join(raw_dir, files.get('rna_h5ad', 'adata_RNA.h5ad'))
        if not os.path.exists(rna_h5ad):
            raise FileNotFoundError(f"RNA h5ad not found: {rna_h5ad}")

        logger.info("Data loader: RNA h5ad + SM (replace ATAC)")
        print(f"   -> Reading RNA from {rna_h5ad}...")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.",
                category=UserWarning,
            )
            adata_rna = sc.read_h5ad(rna_h5ad)

        adata_rna.obs_names = adata_rna.obs_names.astype(str)
        adata_rna.var_names = adata_rna.var_names.astype(str)
        adata_rna.var_names_make_unique()
        adata_rna = _ensure_spatial_from_obs(adata_rna)
        if 'spatial' not in adata_rna.obsm:
            raise ValueError("RNA h5ad is missing spatial coordinates (obsm['spatial']).")

        sm_data_path = resolve_sm_path(sm_data_path, raw_dir)
        print(f"   -> Reading SM from {sm_data_path}...")
        sm_adata = read_sm_h5ad(sm_data_path)

        common_cells = adata_rna.obs_names.intersection(sm_adata.obs_names)
        if len(common_cells) == 0:
            raise ValueError("No overlapping barcodes between RNA and SM h5ad files.")
        if len(common_cells) < adata_rna.n_obs or len(common_cells) < sm_adata.n_obs:
            print(f"Warning: Keeping {len(common_cells)} intersected cells from RNA/SM h5ad files.")

        adata_rna = adata_rna[common_cells, :].copy()
        sm_adata = sm_adata[common_cells, :].copy()
        sm_adata = sm_adata[adata_rna.obs_names, :].copy()
        if 'spatial' not in sm_adata.obsm:
            sm_adata.obsm['spatial'] = np.asarray(adata_rna.obsm['spatial'], dtype=np.float32)

        adata_atac = sm_adata
    # NOTE: fallback to standard RNA+ATAC loaders when not replacing ATAC.
    # ---------------------------------------------------------------
    # 1) paired h5ad
    # 2) 10x multiome h5
    # 3) mtx/tsv
    # ---------------------------------------------------------------
    # default_* used for standard loaders only
    default_rna_h5ad = os.path.join(raw_dir, "adata_RNA.h5ad")
    default_atac_h5ad = os.path.join(raw_dir, "adata_ATAC.h5ad")
    legacy_atac_h5ad = os.path.join(raw_dir, "adata_Peak.h5ad")
    has_pair_h5ad = (
        ('rna_h5ad' in files and 'atac_h5ad' in files) or
        (os.path.exists(default_rna_h5ad) and (os.path.exists(default_atac_h5ad) or os.path.exists(legacy_atac_h5ad)))
    )

    if tri_modal_enabled:
        pass
    elif use_adt:
        pass
    elif sm_replace_atac:
        pass
    elif has_pair_h5ad:
        logger.info("Data loader: paired h5ad")
        rna_h5ad = os.path.join(raw_dir, files.get('rna_h5ad', 'adata_RNA.h5ad'))
        atac_h5ad = os.path.join(
            raw_dir,
            files.get('atac_h5ad', 'adata_ATAC.h5ad' if os.path.exists(default_atac_h5ad) else 'adata_Peak.h5ad')
        )
        adata_rna, adata_atac = read_h5ad_rna_atac(rna_h5ad, atac_h5ad)
    elif 'h5_matrix' in files and 'spatial_csv' in files:
        logger.info("Data loader: 10x h5 + spatial csv")
        print(f"   -> Reading RNA+ATAC from {files['h5_matrix']}...")
        adata_rna, adata_atac = read_10x_h5_multiome(
            os.path.join(raw_dir, files['h5_matrix'])
        )
        adata_rna = add_spatial_info_csv(
            adata_rna,
            os.path.join(raw_dir, files['spatial_csv'])
        )
    else:
        logger.info("Data loader: mtx/tsv/csv split files")
        print(f"   -> Reading RNA from {files['rna_mtx']}...")
        adata_rna = read_mtx_to_adata(
            os.path.join(raw_dir, files['rna_mtx']),
            os.path.join(raw_dir, files['rna_genes']),
            os.path.join(raw_dir, files['rna_barcodes'])
        )
        adata_rna = add_spatial_info(adata_rna, os.path.join(raw_dir, files['spatial']))

        print(f"   -> Reading ATAC from {files['atac_mtx']}...")
        adata_atac = read_mtx_to_adata(
            os.path.join(raw_dir, files['atac_mtx']),
            os.path.join(raw_dir, files['atac_peaks']),
            os.path.join(raw_dir, files['atac_barcodes'])
        )

    # 模态对齐：将 RNA 细胞顺序应用到 ATAC/ADT，并同步空间坐标
    adata_atac = adata_atac[adata_rna.obs_names, :].copy()
    adata_atac.obsm['spatial'] = adata_rna.obsm['spatial'].copy()
    if tri_modal_enabled:
        adata_adt = adata_adt[adata_rna.obs_names, :].copy()
        adata_adt.obsm['spatial'] = adata_rna.obsm['spatial'].copy()

    if tri_modal_enabled:
        adata_rna = select_adata_matrix(adata_rna, rna_input_layer, modality_name="RNA")
        adata_adt = select_adata_matrix(adata_adt, adt_input_layer, modality_name="ADT")
        print(f"   ✅ Explicit input layers: RNA={rna_input_layer}, ADT={adt_input_layer}, ATAC=X")
        logger.info("Explicit input layers | RNA=%s | ADT=%s | ATAC=X", rna_input_layer, adt_input_layer)
    elif use_adt:
        adata_rna = select_adata_matrix(adata_rna, rna_input_layer, modality_name="RNA")
        adata_atac = select_adata_matrix(adata_atac, adt_input_layer, modality_name="ADT")
        print(f"   ✅ Explicit input layers: RNA={rna_input_layer}, ADT={adt_input_layer}")
        logger.info("Explicit input layers | RNA=%s | ADT=%s", rna_input_layer, adt_input_layer)

    # ==========================================
    # Step 2: 严格对齐检查 (不取交集，仅验证)
    # ==========================================
    print("\n🔍 Verifying One-to-One Alignment...")
    n_rna = adata_rna.shape[0]
    n_atac = adata_atac.shape[0]
    n_adt = adata_adt.shape[0] if tri_modal_enabled else n_atac

    # 1. 检查数量
    if n_rna != n_atac:
        raise ValueError(f"\u274c Mismatch! RNA cells ({n_rna}) != ATAC cells ({n_atac}). Please check raw data.")
    if tri_modal_enabled and n_rna != n_adt:
        raise ValueError(f"\u274c Mismatch! RNA cells ({n_rna}) != ADT cells ({n_adt}). Please check raw data.")

    # 2. 检查 Barcode 顺序 (简单抽查前5个和后5个)
    if not np.array_equal(adata_rna.obs_names[:5], adata_atac.obs_names[:5]):
        print("\u26a0\ufe0f Warning: Barcode order might mismatch in the first 5 cells!")
        # 如果您非常确定是对应的，可以忽略这个警告，或者在这里强制赋值索引
        # adata_atac.obs_names = adata_rna.obs_names
    else:
        print("   \u2705 Cell count and order look correct.")
    if tri_modal_enabled and not np.array_equal(adata_rna.obs_names[:5], adata_adt.obs_names[:5]):
        print("\u26a0\ufe0f Warning: RNA/ADT barcode order might mismatch in the first 5 cells!")

    # Canonicalize genomic coordinates before ATAC feature selection. Delimiter
    # variants become identical IDs and any coordinate collision fails closed.
    has_genomic_atac = tri_modal_enabled or (not use_adt and not sm_replace_atac)
    if has_genomic_atac:
        raw_peak_names = adata_atac.var_names.astype(str).tolist()
        canonical_peak_names = canonicalize_peak_names(raw_peak_names)
        changed_count = sum(
            raw != canonical for raw, canonical in zip(raw_peak_names, canonical_peak_names)
        )
        adata_atac.var["raw_peak_name"] = raw_peak_names
        adata_atac.var_names = canonical_peak_names
        peak_canonicalization_stats = {
            "scheme": "chr:start-end-v1",
            "peak_count": int(len(canonical_peak_names)),
            "changed_count": int(changed_count),
            "collision_count": 0,
        }
        logger.info(
            "Canonicalized ATAC peak IDs | count=%d | changed=%d | scheme=%s",
            len(canonical_peak_names),
            changed_count,
            peak_canonicalization_stats["scheme"],
        )

    # ==========================================
    # Step 3: 执行预处理管线
    # ==========================================

    if load_align_mode:
        print("\n🧪 Processing RNA/ATAC in LOAD+ALIGN mode (skip HVG/peak selection)...")
        rna_vars_path = os.path.join(load_pca_dir, "rna_vars.txt")
        atac_vars_path = os.path.join(load_pca_dir, "atac_vars.txt")
        if not os.path.exists(rna_vars_path) or not os.path.exists(atac_vars_path):
            raise FileNotFoundError(
                "LOAD+ALIGN mode requires rna_vars.txt and atac_vars.txt under --load-pca-dir. "
                f"missing: rna={rna_vars_path}, atac={atac_vars_path}"
            )

        src_rna_vars = np.atleast_1d(np.loadtxt(rna_vars_path, dtype=str)).tolist()
        src_atac_vars = np.atleast_1d(np.loadtxt(atac_vars_path, dtype=str)).tolist()
        logger.info(
            "Loaded source feature lists | RNA=%d from %s | ATAC=%d from %s",
            len(src_rna_vars),
            rna_vars_path,
            len(src_atac_vars),
            atac_vars_path,
        )

        # 1) hard-align features to source-domain lists
        adata_rna = align_adata_features_hard(
            adata_rna,
            src_rna_vars,
            modality_name="RNA",
            min_overlap=rna_min_overlap,
        )
        if atac_alignment_mode == 'genomic_interval':
            if not has_genomic_atac:
                raise ValueError(
                    "ATAC genomic interval alignment requires genomic peak features."
                )
            adata_atac = align_adata_peaks_by_interval(
                adata_atac,
                src_atac_vars,
                min_overlap=atac_min_overlap,
                min_iou=atac_interval_min_iou,
            )
        else:
            adata_atac = align_adata_features_hard(
                adata_atac,
                src_atac_vars,
                modality_name="ATAC",
                min_overlap=atac_min_overlap,
            )

        # 2) manual normalization compensation after skipping default pipelines
        if sparse.issparse(adata_rna.X):
            adata_rna.X = adata_rna.X.tocsr().astype(np.float32)
        else:
            adata_rna.X = np.asarray(adata_rna.X, dtype=np.float32)
        sc.pp.normalize_total(adata_rna, target_sum=float(rna_target_sum))
        sc.pp.log1p(adata_rna)

        adata_atac = custom_tf_idf(adata_atac, eps=tfidf_eps)
        X_atac = adata_atac.X
        if sparse.issparse(X_atac):
            X_atac = X_atac.tocsr().astype(np.float32)
            data = X_atac.data
            data[~np.isfinite(data)] = 0.0
            X_atac.data = data
            adata_atac.X = X_atac
        else:
            X_atac = np.asarray(X_atac, dtype=np.float32)
            X_atac[~np.isfinite(X_atac)] = 0.0
            adata_atac.X = X_atac
        sc.pp.normalize_total(adata_atac, target_sum=float(atac_target_sum))
        sc.pp.log1p(adata_atac)

    else:
        # --- RNA ---
        # 预防性措施：在传入 pipeline 之前，确保它是 float32
        # 这样即使不修改 rna_process.py，也能避免 normalize_total 报错
        if hasattr(adata_rna.X, "astype"):
            adata_rna.X = adata_rna.X.astype(np.float32)

        print("\n🧪 Processing RNA...")
        adata_rna = process_rna_pipeline(
            adata_rna,
            n_top_genes=params['n_top_genes'],
            min_cells=rna_min_cells,
            target_sum=rna_target_sum,
            apply_scale=rna_apply_scale,
        )

        # --- ATAC / ADT ---
        if tri_modal_enabled:
            print("\n🧪 Processing ADT...")
            raw_adt_row_sums = np.asarray(adata_adt.X.sum(axis=1)).ravel()
            zero_adt_spots = int(np.sum(np.isclose(raw_adt_row_sums, 0.0)))
            if zero_adt_spots:
                print(f"   Warning: raw ADT contains {zero_adt_spots} all-zero spots; retaining alignment.")
                logger.warning("Raw ADT all-zero spots retained: %d", zero_adt_spots)
            adata_adt = process_adt_pipeline(
                adata_adt,
                apply_clr=adt_apply_clr,
                apply_scale=adt_apply_scale,
                clip_nonnegative=adt_clip_nonnegative,
                min_cells=adt_min_cells,
                pipeline=adt_pipeline,
                nonnegative_policy=adt_nonnegative_policy,
            )

            print("\n🧪 Processing ATAC...")
            gtf_file = files.get('gtf')
            gtf_path = os.path.join(raw_dir, gtf_file) if gtf_file else None
            rna_genes = adata_rna.var_names.tolist()
            adata_atac, _ = process_atac_pipeline(
                adata_atac,
                rna_genes=rna_genes,
                gtf_path=gtf_path,
                n_global=params.get('n_global_peaks', 50000),
                n_final=params.get('n_final_peaks', 30000),
                window=params.get('tss_window', 100000),
                min_cells=atac_min_cells,
                target_sum=atac_target_sum,
                tfidf_eps=tfidf_eps,
            )

            gtf_stats = adata_atac.uns.get("gtf_filter_stats", None)
            if gtf_stats:
                logger.info(
                    "GTF filter stats | gtf=%s | window=%s | rna_genes=%s | matched_genes=%s | "
                    "peaks_before=%s | peaks_valid=%s | peaks_retained=%s | peaks_after_downsample=%s | "
                    "fallback_used=%s | status=%s",
                    gtf_stats.get("gtf_path"),
                    gtf_stats.get("window"),
                    gtf_stats.get("rna_gene_count"),
                    gtf_stats.get("matched_gene_count"),
                    gtf_stats.get("peaks_before"),
                    gtf_stats.get("peaks_valid"),
                    gtf_stats.get("peaks_retained"),
                    gtf_stats.get("peaks_after_downsample"),
                    gtf_stats.get("fallback_used"),
                    gtf_stats.get("status"),
                )
        elif use_adt:
            print("\n🧪 Processing ADT...")
            raw_adt_row_sums = np.asarray(adata_atac.X.sum(axis=1)).ravel()
            zero_adt_spots = int(np.sum(np.isclose(raw_adt_row_sums, 0.0)))
            if zero_adt_spots:
                print(f"   Warning: raw ADT contains {zero_adt_spots} all-zero spots; retaining alignment.")
                logger.warning("Raw ADT all-zero spots retained: %d", zero_adt_spots)
            adata_atac = process_adt_pipeline(
                adata_atac,
                apply_clr=adt_apply_clr,
                apply_scale=adt_apply_scale,
                clip_nonnegative=adt_clip_nonnegative,
                min_cells=adt_min_cells,
                pipeline=adt_pipeline,
                nonnegative_policy=adt_nonnegative_policy,
            )
        elif sm_replace_atac:
            logger.info("ATAC preprocessing skipped (SM replace_atac enabled)")
        else:
            print("\n🧪 Processing ATAC...")
            gtf_path = os.path.join(raw_dir, files['gtf'])
            rna_genes = adata_rna.var_names.tolist()

            adata_atac, _ = process_atac_pipeline(
                adata_atac,
                rna_genes=rna_genes,
                gtf_path=gtf_path,
                n_global=params['n_global_peaks'],
                n_final=params['n_final_peaks'],
                window=params['tss_window'],
                min_cells=atac_min_cells,
                target_sum=atac_target_sum,
                tfidf_eps=tfidf_eps,
            )

            gtf_stats = adata_atac.uns.get("gtf_filter_stats", None)
            if gtf_stats:
                logger.info(
                    "GTF filter stats | gtf=%s | window=%s | rna_genes=%s | matched_genes=%s | "
                    "peaks_before=%s | peaks_valid=%s | peaks_retained=%s | peaks_after_downsample=%s | "
                    "fallback_used=%s | status=%s",
                    gtf_stats.get("gtf_path"),
                    gtf_stats.get("window"),
                    gtf_stats.get("rna_gene_count"),
                    gtf_stats.get("matched_gene_count"),
                    gtf_stats.get("peaks_before"),
                    gtf_stats.get("peaks_valid"),
                    gtf_stats.get("peaks_retained"),
                    gtf_stats.get("peaks_after_downsample"),
                    gtf_stats.get("fallback_used"),
                    gtf_stats.get("status"),
                )

            # S1 fit&save mode: persist selected feature lists for cross-domain hard alignment.
            if fit_save_mode:
                rna_vars_path = os.path.join(save_pca_dir, "rna_vars.txt")
                atac_vars_path = os.path.join(save_pca_dir, "atac_vars.txt")
                np.savetxt(rna_vars_path, np.asarray(adata_rna.var_names, dtype=str), fmt="%s")
                np.savetxt(atac_vars_path, np.asarray(adata_atac.var_names, dtype=str), fmt="%s")
                logger.info(
                    "Saved selected feature lists | RNA=%d -> %s | ATAC=%d -> %s",
                    adata_rna.n_vars,
                    rna_vars_path,
                    adata_atac.n_vars,
                    atac_vars_path,
                )

    # 预处理后再做一次强制对齐：按 RNA 条码对 ATAC 取子集并重排。
    # 这一步可兜底处理任一模态在上游发生了细胞过滤的情况。
    if adata_rna.n_obs != adata_atac.n_obs or not np.array_equal(adata_rna.obs_names, adata_atac.obs_names):
        common_cells = adata_rna.obs_names.intersection(adata_atac.obs_names)
        if len(common_cells) == 0:
            raise ValueError("\u274c No overlapping cells between RNA and ATAC after preprocessing.")
        if len(common_cells) < adata_rna.n_obs or len(common_cells) < adata_atac.n_obs:
            print(
                f"\u26a0\ufe0f Post-process alignment: RNA={adata_rna.n_obs}, ATAC={adata_atac.n_obs}, "
                f"keeping intersection={len(common_cells)}"
            )

        adata_rna = adata_rna[common_cells, :].copy()
        adata_atac = adata_atac[common_cells, :].copy()
        adata_atac = adata_atac[adata_rna.obs_names, :].copy()
        adata_atac.obsm['spatial'] = adata_rna.obsm['spatial'].copy()

    if tri_modal_enabled and (
        adata_rna.n_obs != adata_adt.n_obs or not np.array_equal(adata_rna.obs_names, adata_adt.obs_names)
    ):
        common_cells = adata_rna.obs_names.intersection(adata_adt.obs_names)
        if len(common_cells) == 0:
            raise ValueError("\u274c No overlapping cells between RNA and ADT after preprocessing.")
        if len(common_cells) < adata_rna.n_obs or len(common_cells) < adata_adt.n_obs:
            print(
                f"\u26a0\ufe0f Post-process alignment: RNA={adata_rna.n_obs}, ADT={adata_adt.n_obs}, "
                f"keeping intersection={len(common_cells)}"
            )
        adata_rna = adata_rna[common_cells, :].copy()
        adata_atac = adata_atac[common_cells, :].copy()
        adata_adt = adata_adt[common_cells, :].copy()
        adata_atac = adata_atac[adata_rna.obs_names, :].copy()
        adata_adt = adata_adt[adata_rna.obs_names, :].copy()
        adata_atac.obsm['spatial'] = adata_rna.obsm['spatial'].copy()
        adata_adt.obsm['spatial'] = adata_rna.obsm['spatial'].copy()

    if tri_modal_enabled:
        print(f"   \u2705 Aligned final cells: RNA={adata_rna.n_obs}, ADT={adata_adt.n_obs}, ATAC={adata_atac.n_obs}")
        logger.info("Aligned final cells: RNA=%d, ADT=%d, ATAC=%d", adata_rna.n_obs, adata_adt.n_obs, adata_atac.n_obs)
    else:
        print(f"   \u2705 Aligned final cells: RNA={adata_rna.n_obs}, ATAC={adata_atac.n_obs}")
        logger.info("Aligned final cells: RNA=%d, ATAC=%d", adata_rna.n_obs, adata_atac.n_obs)

    # ==========================================
    # Step 3.2: 代谢组 (SM) 预处理（可选）
    # ==========================================
    if sm_data_path:
        if sm_adata is None:
            sm_data_path = resolve_sm_path(sm_data_path, raw_dir)
            print("\n🧪 Processing SM (metabolomics)...")
            logger.info("SM preprocess: %s", sm_data_path)
            sm_adata = read_sm_h5ad(sm_data_path)
        else:
            print("\n🧪 Processing SM (metabolomics)...")
            logger.info("SM preprocess: reuse loaded AnnData")

        sm_use_pca = bool(sm_pre_cfg.get('use_pca', False))

        sm_moran_k_default = config.get('eval', {}).get('moran_k', params.get('knn_k', 6))
        sm_n_top = sm_pre_cfg.get('n_top_metabolites', 500)
        sm_n_top = None if sm_n_top is None else int(sm_n_top)
        sm_pval = sm_pre_cfg.get('pval_threshold', 0.05)
        sm_pval = 0.05 if sm_pval is None else float(sm_pval)
        sm_moran_k = sm_pre_cfg.get('moran_k', sm_moran_k_default)
        sm_moran_k = sm_moran_k_default if sm_moran_k is None else int(sm_moran_k)
        sm_perms = sm_pre_cfg.get('moran_permutations', 0)
        sm_perms = 0 if sm_perms is None else int(sm_perms)
        sm_method = sm_pre_cfg.get('spatial_graph_method', 'knn') or 'knn'
        sm_hvg_method = sm_pre_cfg.get('spatial_hvg_method', 'morans_i') or 'morans_i'
        sm_target_sum = float(sm_pre_cfg.get('target_sum', 1e4))
        sm_adata = process_sm_pipeline(
            sm_adata,
            target_sum=sm_target_sum,
            apply_log1p=bool(sm_pre_cfg.get('apply_log1p', True)),
            spatial_hvg_method=sm_hvg_method,
            n_top_metabolites=sm_n_top,
            apply_scale=bool(sm_pre_cfg.get('apply_scale', True)),
            pval_threshold=sm_pval,
            drop_dead_spots=bool(sm_pre_cfg.get('drop_dead_spots', False)),
            spatial_graph_method=sm_method,
            moran_k=sm_moran_k,
            radius=sm_pre_cfg.get('radius', None),
            moran_permutations=sm_perms,
        )
        if sm_replace_atac:
            if sm_adata.n_obs != adata_rna.n_obs or not np.array_equal(sm_adata.obs_names, adata_rna.obs_names):
                common_cells = adata_rna.obs_names.intersection(sm_adata.obs_names)
                if len(common_cells) == 0:
                    raise ValueError("No overlapping cells between RNA and SM after preprocessing.")
                if len(common_cells) < adata_rna.n_obs or len(common_cells) < sm_adata.n_obs:
                    print(
                        f"\u26a0\ufe0f Post-SM alignment: RNA={adata_rna.n_obs}, SM={sm_adata.n_obs}, "
                        f"keeping intersection={len(common_cells)}"
                    )
                adata_rna = adata_rna[common_cells, :].copy()
                sm_adata = sm_adata[common_cells, :].copy()
                sm_adata = sm_adata[adata_rna.obs_names, :].copy()
        sm_feat_np = np.asarray(sm_adata.X, dtype=np.float32)
        print(f"   \u2705 SM processed shape: {sm_feat_np.shape}")
        logger.info("SM processed shape: %s", sm_feat_np.shape)
        if sm_replace_atac:
            adata_atac = sm_adata
    else:
        logger.info("SM preprocess skipped (no sm_data_path).")

    # ==========================================
    # Step 3.5: 模态 PCA/SVD 降维（统一入模特征维度）
    # ==========================================
    rna_pca_dim = int(params.get('rna_pca_dim', 512))
    atac_pca_dim = int(params.get('atac_pca_dim', 512))

    print("\n📉 Reducing RNA/ATAC features before model input...")
    if rna_use_pca:
        rna_feat_np = reduce_modality_features(
            adata_rna.X,
            rna_pca_dim,
            seed,
            "RNA",
            reduce_cfg=reduce_cfg,
            save_model_path=rna_model_save_path,
            load_model_path=rna_model_load_path,
        )
    else:
        print("   [RNA] PCA disabled by config; using full feature matrix.")
        rna_feat_np = np.asarray(adata_rna.X, dtype=np.float32)
        if not np.isfinite(rna_feat_np).all():
            raise ValueError("RNA feature matrix contains NaN/Inf before model input.")
    if tri_modal_enabled:
        if adt_use_pca:
            adt_feat_np = reduce_modality_features(
                adata_adt.X,
                int(adt_pre_cfg.get('adt_pca_dim', params.get('adt_pca_dim', atac_pca_dim))),
                seed,
                "ADT",
                reduce_cfg=reduce_cfg,
            )
            print(f"   Using PCA-reduced ADT features: {adt_feat_np.shape}")
            logger.info("Using PCA-reduced ADT features: %s", adt_feat_np.shape)
        else:
            adt_feat_np = np.asarray(adata_adt.X, dtype=np.float32)
            print(f"   Using full ADT features: {adt_feat_np.shape}")
            logger.info("Using full ADT features: %s", adt_feat_np.shape)
        atac_feat_np = reduce_modality_features(
            adata_atac.X,
            atac_pca_dim,
            seed,
            "ATAC",
            reduce_cfg=reduce_cfg,
            save_model_path=atac_model_save_path,
            load_model_path=atac_model_load_path,
        )
    elif use_adt:
        if adt_use_pca:
            atac_feat_np = reduce_modality_features(
                adata_atac.X,
                atac_pca_dim,
                seed,
                "ADT",
                reduce_cfg=reduce_cfg,
            )
            print(f"   Using PCA-reduced ADT features: {atac_feat_np.shape}")
            logger.info("Using PCA-reduced ADT features: %s", atac_feat_np.shape)
        else:
            atac_feat_np = np.asarray(adata_atac.X, dtype=np.float32)
            print(f"   Using full ADT features as second-modality input: {atac_feat_np.shape}")
            logger.info("Using full ADT features as second-modality input: %s", atac_feat_np.shape)
    elif sm_replace_atac:
        if sm_feat_np is None:
            raise ValueError("SM features are missing; cannot replace ATAC input.")
        if sm_use_pca:
            atac_feat_np = reduce_modality_features(
                sm_feat_np,
                atac_pca_dim,
                seed,
                "ATAC",
                reduce_cfg=reduce_cfg,
                save_model_path=atac_model_save_path,
                load_model_path=atac_model_load_path,
            )
            print(f"   \u2705 Using PCA-reduced SM features as ATAC input: {atac_feat_np.shape}")
            logger.info("Using PCA-reduced SM features as ATAC input: %s", atac_feat_np.shape)
        else:
            atac_feat_np = np.asarray(sm_feat_np, dtype=np.float32)
            print(f"   \u2705 Using SM features as ATAC input: {atac_feat_np.shape}")
            logger.info("Using SM features as ATAC input: %s", atac_feat_np.shape)
    else:
        atac_feat_np = reduce_modality_features(
            adata_atac.X,
            atac_pca_dim,
            seed,
            "ATAC",
            reduce_cfg=reduce_cfg,
            save_model_path=atac_model_save_path,
            load_model_path=atac_model_load_path,
        )

        # --- 请加上这段代码，切除被深度绑架的 PC1 ---
        # from sklearn.decomposition import TruncatedSVD
        # 由于你的 SVD 是在 reduce_modality_features 里做的，你可以在外面直接切掉第一列
        # print("   [ATAC] Dropping SVD PC1 due to high sequencing depth correlation.")
        # atac_feat_np = atac_feat_np[:, 1:]
        # ---------------------------------------------

        # if sparse.issparse(adata_atac.X):
        #     scaler = StandardScaler()
        #     atac_feat_np = scaler.fit_transform(atac_feat_np)
        #     atac_feat_np = np.asarray(atac_feat_np, dtype=np.float32)
        #     print("   ✅ StandardScaler applied to ATAC SVD features")
        #     logger.info("StandardScaler applied to ATAC SVD features")
    print(f"   \u2705 Reduced RNA shape: {rna_feat_np.shape}")
    if tri_modal_enabled:
        print(f"   \u2705 Reduced ADT shape: {adt_feat_np.shape}")
    print(f"   \u2705 Reduced ATAC shape: {atac_feat_np.shape}")
    if tri_modal_enabled:
        logger.info("Reduced shapes: RNA=%s, ADT=%s, ATAC=%s", rna_feat_np.shape, adt_feat_np.shape, atac_feat_np.shape)
    else:
        logger.info("Reduced shapes: RNA=%s, ATAC=%s", rna_feat_np.shape, atac_feat_np.shape)

    # # ==========================================
    # # Step 4: 构建空间图 & 准备 Tensor
    # # ==========================================
    # print("\n🕸️ Building Spatial Graph & GFT Basis...")
    # coords = adata_rna.obsm['spatial']

    # # 计算图基底 (GFT Basis)
    # edge_index, u_basis = build_spatial_graph(coords, k=params['knn_k'])

    # # 转换为 Tensor
    # def to_tensor(adata):
    #     if hasattr(adata.X, 'toarray'):
    #         return torch.FloatTensor(adata.X.toarray())
    #     return torch.FloatTensor(adata.X)

    # rna_feat = to_tensor(adata_rna)
    # atac_feat = to_tensor(adata_atac)
    # coords_tensor = torch.FloatTensor(coords)

    # ==========================================
    # Step 4: 构建空间图 & 准备 Tensor  #根据knn引入权重
    # ==========================================
    graph_stage = "Spatial Graph & GFT Basis" if compute_gft_basis else "Sparse Spatial Graph"
    print(f"\n\U0001f578\ufe0f Building {graph_stage}...")
    coords = adata_rna.obsm['spatial']
    # Edge weight mode controls the Laplacian/GFT basis.
    # rna_similarity preserves legacy behavior; spatial_distance uses physical coordinates only.
    rna_features = rna_feat_np if graph_edge_weight_mode in {"rna_similarity", "rna", "feature", "features"} else None
    print(f"   \u2705 Edge weight mode: {graph_edge_weight_mode}")
    logger.info("Edge weight mode: %s", graph_edge_weight_mode)

    graph_outputs = build_spatial_graph(
        coords,
        features=rna_features,
        k=params['knn_k'],
        n_freq_components=n_freq_components,
        edge_weight_mode=graph_edge_weight_mode,
        compute_gft_basis=compute_gft_basis,
    )
    if len(graph_outputs) == 4:
        edge_index, edge_weight, u_basis, evals = graph_outputs
    elif len(graph_outputs) == 3:
        edge_index, u_basis, evals = graph_outputs
        edge_weight = None
    else:
        edge_index, u_basis = graph_outputs
        edge_weight = None
        evals = None

    # 备选：使用空间距离高斯衰减
    # edge_index, u_basis = build_spatial_graph(coords, k=params['knn_k'])

    rna_feat = torch.FloatTensor(rna_feat_np)
    adt_feat = torch.FloatTensor(adt_feat_np) if tri_modal_enabled else None
    atac_feat = torch.FloatTensor(atac_feat_np)
    coords_tensor = torch.FloatTensor(coords)

    gt_key = infer_ground_truth_key(adata_rna.obs, preferred_key=params.get('ground_truth_key'))
    ground_truth = None
    gt_txt = params.get('ground_truth_txt', None)
    if gt_txt:
        gt_path = resolve_adt_path(gt_txt, raw_dir)
        ground_truth = load_ground_truth_txt(gt_path, adata_rna.obs_names)
        gt_key = f"txt:{gt_txt}"
        print(f"   \u2705 Loaded ground truth from txt: {gt_path}")
    elif gt_key is not None:
        gt_vals = adata_rna.obs[gt_key].astype(str).to_numpy()
        valid_mask = np.array([
            (v is not None) and (str(v).strip() != "") and (str(v).lower() != "nan")
            for v in gt_vals
        ])
        if valid_mask.sum() > 0:
            ground_truth = gt_vals
            print(f"   \u2705 Found ground truth in obs column: {gt_key}")
        else:
            print(f"   \u26a0\ufe0f Ground truth column '{gt_key}' is empty after cleaning; skipping.")
    else:
        ground_truth = infer_ground_truth_from_obsm(adata_rna, params.get('ground_truth_obsm_key'))
        if ground_truth is not None:
            gt_key = f"obsm:{params.get('ground_truth_obsm_key')}"
            print(f"   \u2705 Derived ground truth from RNA {gt_key}")
        else:
            print("   \u26a0\ufe0f No ground truth column found in RNA obs.")

    # ==========================================
    # Step 5: 保存处理好的数据
    # ==========================================
    save_path = os.path.join(processed_dir, "processed_data.pt")
    print(f"\n\U0001f4be Saving processed tensors to {save_path}...")
    logger.info("Saving processed tensors...")

    active_rna_reducer = rna_model_load_path or rna_model_save_path
    active_atac_reducer = atac_model_load_path or atac_model_save_path
    feature_schema = {
        "format_version": "biosfib-feature-contract-v1",
        "config_path": os.path.abspath(args.config),
        "config_sha256": _sha256_file(args.config),
        "mode": (
            "fit-and-save" if fit_save_mode else
            "load-and-transform" if load_align_mode else
            "standalone"
        ),
        "rna_var_count": int(adata_rna.n_vars),
        "atac_var_count": int(adata_atac.n_vars),
        "rna_var_sha256": _sha256_strings(adata_rna.var_names.astype(str).tolist()),
        "atac_var_sha256": _sha256_strings(adata_atac.var_names.astype(str).tolist()),
        "rna_reducer_sha256": _sha256_file(active_rna_reducer),
        "atac_reducer_sha256": _sha256_file(active_atac_reducer),
        "rna_tensor_sha256": _sha256_array(rna_feat_np),
        "atac_tensor_sha256": _sha256_array(atac_feat_np),
        "peak_canonicalization": peak_canonicalization_stats,
        "rna_alignment": adata_rna.uns.get("feature_alignment", None),
        "atac_alignment": adata_atac.uns.get("feature_alignment", None),
    }

    data_dict = {
        "rna_feat": rna_feat,
        "atac_feat": atac_feat,
        "coords": coords_tensor,
        "edge_index": edge_index,
        "u_basis": u_basis,
        "graph_edge_weight_mode": graph_edge_weight_mode,
        "graph_knn_k": int(params['knn_k']),
        "spectral_basis_schema": (
            "legacy_gft_basis"
        ),
        "rna_dim": rna_feat.shape[1],
        "atac_dim": atac_feat.shape[1], # 记录动态第二模态维度
        "modality2": "adt" if use_adt else ("sm" if sm_replace_atac else "atac"),
        "preprocess_version": "rna_adt_v1" if use_adt and adt_pipeline == "seurat_clr_scale" else "legacy",
        "rna_apply_scale": bool(rna_apply_scale),
        "rna_input_layer": rna_input_layer if use_adt else "X",
        "adt_pipeline": adt_pipeline if use_adt else None,
        "adt_apply_scale": bool(adt_apply_scale) if use_adt else None,
        "adt_use_pca": bool(adt_use_pca) if use_adt else None,
        "adt_input_layer": adt_input_layer if use_adt else None,
        "feature_schema": feature_schema,
    }

    if tri_modal_enabled:
        data_dict["adt_feat"] = adt_feat
        data_dict["adt_dim"] = adt_feat.shape[1]
        data_dict["modalities"] = ["rna", "adt", "atac"]
        data_dict["modality2"] = None
        data_dict["preprocess_version"] = "rna_adt_atac_tri_v1"

    if edge_weight is not None:
        data_dict["edge_weight"] = edge_weight

    if evals is not None:
        data_dict["evals"] = evals

    data_dict["n_freq_components"] = int(u_basis.shape[1])

    if ground_truth is not None:
        data_dict["ground_truth"] = ground_truth
        data_dict["ground_truth_key"] = gt_key

    if sm_feat_np is not None and sm_adata is not None:
        sm_feat = torch.FloatTensor(sm_feat_np)
        data_dict["sm_feat"] = sm_feat
        data_dict["sm_input_dim"] = int(sm_feat.shape[1])
        data_dict["sm_var_names"] = np.asarray(sm_adata.var_names, dtype=str)
        data_dict["sm_obs_names"] = np.asarray(sm_adata.obs_names, dtype=str)
        if "spatial" in sm_adata.obsm:
            sm_coords = np.asarray(sm_adata.obsm["spatial"], dtype=np.float32)
            data_dict["sm_coords"] = torch.FloatTensor(sm_coords)

    reference_processed_path = preprocess_cfg.get("reference_processed_path")
    if reference_processed_path:
        reference_keys = tuple(
            preprocess_cfg.get(
                "reference_tensor_keys",
                ("rna_feat", "atac_feat", "coords"),
            )
        )
        reference_match = validate_reference_feature_tensors(
            data_dict,
            reference_processed_path,
            keys=reference_keys,
        )
        feature_schema["reference_tensor_match"] = reference_match
        print(
            "   ✅ Formal S1 tensor equivalence: exact | "
            f"keys={','.join(reference_match['keys'])}"
        )
        logger.info(
            "Formal S1 tensor equivalence: exact | reference=%s | keys=%s",
            reference_match["reference_path"],
            ",".join(reference_match["keys"]),
        )

    torch.save(data_dict, save_path)
    print("\u2705 Preprocessing Complete!")
    logger.info("Preprocessing complete: %s", save_path)
    if os.environ.get("SF_PIPELINE_RUN") != "1":
        append_log_separator(log_path)

if __name__ == "__main__":
    main()
