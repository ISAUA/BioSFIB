import os
import warnings

import numpy as np
import scanpy as sc
import scipy.sparse as sp


def resolve_adt_path(adt_data_path, raw_dir=None):
    if adt_data_path is None:
        return None
    path = os.path.expanduser(str(adt_data_path))
    if os.path.isabs(path):
        return path
    if raw_dir:
        candidate = os.path.join(raw_dir, path)
        if os.path.exists(candidate):
            return candidate
    return path


def read_adt_h5ad(adt_data_path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Variable names are not unique. To make them unique, call `.var_names_make_unique`.",
            category=UserWarning,
        )
        adata = sc.read_h5ad(adt_data_path)

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    return adata


def _to_dense_float32(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def _assert_finite(X, stage):
    if not np.isfinite(X).all():
        raise ValueError(f"ADT matrix contains NaN/Inf after {stage}.")


def _clip_nonnegative(X):
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        X.data[X.data < 0] = 0.0
        X.eliminate_zeros()
        return X
    return np.maximum(np.asarray(X, dtype=np.float32), 0.0).astype(np.float32)


def seurat_clr_each_cell(X):
    """Seurat/SpatialGlue CLR over proteins within each cell."""
    X = _to_dense_float32(X)
    _assert_finite(X, "loading")
    if np.any(X < 0):
        raise ValueError("ADT raw counts must be nonnegative before Seurat CLR.")

    positive_log_sum = np.where(X > 0, np.log1p(X), 0.0).sum(axis=1, keepdims=True)
    denominator = np.exp(positive_log_sum / max(1, X.shape[1]))
    clr = np.log1p(X / denominator)
    _assert_finite(clr, "Seurat CLR")
    return clr.astype(np.float32)


def process_adt_pipeline(
    adata,
    apply_clr=False,
    apply_scale=False,
    clip_nonnegative=True,
    min_cells=None,
    pipeline="legacy",
    nonnegative_policy=None,
):
    if adata.shape[1] == 0:
        return adata

    pipeline = str(pipeline or "legacy").strip().lower()
    if pipeline not in {"legacy", "seurat_clr_scale"}:
        raise ValueError(f"Unsupported ADT preprocessing pipeline: {pipeline}")

    if min_cells is not None:
        sc.pp.filter_genes(adata, min_cells=int(min_cells))
        if adata.n_vars == 0:
            raise ValueError("ADT filtering removed every protein feature.")

    adata.X = _to_dense_float32(adata.X)
    _assert_finite(adata.X, "loading")

    if pipeline == "seurat_clr_scale":
        policy = str(nonnegative_policy or "error").strip().lower()
        if policy != "error":
            raise ValueError("seurat_clr_scale requires nonnegative_policy='error'.")
        if not apply_clr:
            raise ValueError("seurat_clr_scale requires apply_clr=true.")
        if clip_nonnegative:
            raise ValueError("seurat_clr_scale requires clip_nonnegative=false.")
        adata.X = seurat_clr_each_cell(adata.X)
        if apply_scale:
            sc.pp.scale(adata, zero_center=True)
            adata.X = np.asarray(adata.X, dtype=np.float32)
            _assert_finite(adata.X, "scale")
        return adata

    if clip_nonnegative:
        adata.X = _clip_nonnegative(adata.X)
        _assert_finite(adata.X, "nonnegative clipping before CLR")

    if apply_clr:
        X = np.log1p(adata.X)
        X = X - X.mean(axis=1, keepdims=True)
        adata.X = X.astype(np.float32)
        _assert_finite(adata.X, "CLR")

    if apply_scale:
        sc.pp.scale(adata, zero_center=True)
        adata.X = np.asarray(adata.X, dtype=np.float32)
        _assert_finite(adata.X, "scale")

    if clip_nonnegative:
        adata.X = _clip_nonnegative(adata.X)
        _assert_finite(adata.X, "final nonnegative clipping")

    return adata
