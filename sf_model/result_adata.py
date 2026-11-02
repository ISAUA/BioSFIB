"""Reference AnnData loading for BioSFIB result exports."""

from pathlib import Path

import numpy as np
import scanpy as sc

from sf_model.preprocess.io import _ensure_spatial_from_obs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_result_reference_adata(config: dict, expected_coords) -> object:
    """Load the original RNA AnnData for a result export without preprocessing."""
    data_cfg = config.get("data", {}) or {}
    files = data_cfg.get("files", {}) or {}
    raw_dir = Path(str(data_cfg.get("raw_path", "")))
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir
    rna_name = files.get("rna_h5ad")
    if not rna_name:
        raise KeyError("Result export requires data.files.rna_h5ad.")
    rna_path = raw_dir / str(rna_name)
    if not rna_path.exists():
        raise FileNotFoundError(f"Result-export RNA h5ad not found: {rna_path}")

    adata = sc.read_h5ad(rna_path)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    adata = _ensure_spatial_from_obs(adata)
    if "spatial" not in adata.obsm:
        raise KeyError(f"RNA reference lacks obsm['spatial']: {rna_path}")

    coords = np.asarray(expected_coords.detach().cpu() if hasattr(expected_coords, "detach") else expected_coords)
    spatial = np.asarray(adata.obsm["spatial"])
    if adata.n_obs != coords.shape[0]:
        raise ValueError(
            f"RNA reference spot count differs from processed data: {adata.n_obs} != {coords.shape[0]}"
        )
    if spatial.shape[0] != coords.shape[0] or not np.allclose(spatial[:, :2], coords[:, :2], rtol=1e-5, atol=1e-5):
        raise ValueError("RNA reference spot order/coordinates differ from processed data.")
    return adata
