"""SSR-native downstream analysis for BioSFINet-SSR fixed-epoch reruns.

Primary target:
    MISAR E18.5_S1 rerun at epoch 1300 under
    results/MISAR/e18_5_s1

This script is intentionally model-native. It extracts the current SSR internals:
    z_spec, z_detail, scale_gates[:, 0] = low-scale RNA gate,
    scale_gates[:, 1] = mid-scale RNA gate, and model.sfib.last_boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def _sanitize_thread_env_vars() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(key)
        try:
            ok = value is not None and int(str(value).strip()) > 0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            os.environ[key] = "1"


_sanitize_thread_env_vars()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml

from sf_model.model.bio_sfinet import BioSFINet
from sf_model.utils import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPATIAL_COLUMNS = [
    "boundary_score",
    "z_detail_norm",
    "correction_ratio",
    "correction_share",
    "branch_disagreement",
    "gate_low_rna",
    "gate_mid_rna",
    "gate_mid_minus_low",
]

SUMMARY_COLUMNS = SPATIAL_COLUMNS + [
    "z_spec_norm",
    "z_out_norm",
    "fused_shift_from_spec",
    "spatial_gate_mean",
]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()



def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)



def torch_load(path: Path, map_location="cpu", weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)



def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)



def resolve_processed_data(config: dict) -> Path:
    processed_path = config.get("data", {}).get("processed_path")
    if not processed_path:
        raise KeyError("config.data.processed_path is required")
    return resolve_path(processed_path) / "processed_data.pt"



def resolve_checkpoint(config: dict, checkpoint_arg: str | None) -> Path:
    save_dir = config.get("project", {}).get("save_dir")
    if not save_dir:
        raise KeyError("config.project.save_dir is required")
    checkpoint = checkpoint_arg or config.get("eval", {}).get("checkpoint")
    if not checkpoint:
        checkpoint = config.get("eval", {}).get("checkpoints", {}).get("final")
    if not checkpoint:
        raise KeyError("checkpoint argument or config.eval.checkpoint is required")
    checkpoint_path = Path(str(checkpoint)).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return resolve_path(save_dir) / checkpoint_path



def resolve_eval_h5ad(config: dict, eval_h5ad_arg: str | None) -> Path:
    if eval_h5ad_arg:
        return resolve_path(eval_h5ad_arg)
    eval_dir = config.get("project", {}).get("eval_dir")
    if not eval_dir:
        raise KeyError("config.project.eval_dir is required when --eval-h5ad is not provided")
    return resolve_path(eval_dir) / "embedding_joint.h5ad"



def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)



def get_coords(data_dict: dict, eval_adata) -> np.ndarray:
    for key in ("coords", "spatial", "pos", "positions"):
        if key in data_dict:
            arr = to_numpy(data_dict[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2].astype(np.float64)
    if eval_adata is not None and "spatial" in eval_adata.obsm:
        arr = np.asarray(eval_adata.obsm["spatial"])
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2].astype(np.float64)
    raise KeyError("Could not find spatial coordinates in processed data or eval h5ad")



def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denom, 1e-12)



def load_ssr_state(config: dict, checkpoint_path: Path, data_dict: dict, device: torch.device) -> dict:
    rna_feat = data_dict["rna_feat"].to(device)
    atac_feat = data_dict["atac_feat"].to(device)
    edge_index = data_dict["edge_index"].to(device)
    u_basis = data_dict["u_basis"].to(device)
    evals = data_dict.get("evals")
    evals = None if evals is None else evals.to(device)
    edge_weight = data_dict.get("edge_weight")
    edge_weight = None if edge_weight is None else edge_weight.to(device)

    config.setdefault("model", {})["rna_in_dim"] = int(data_dict.get("rna_dim", rna_feat.shape[1]))
    atac_dim = int(data_dict.get("atac_dim", atac_feat.shape[1]))
    model = BioSFINet(config, atac_dim=atac_dim).to(device)
    state_dict = torch_load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        outputs = model(rna_feat, atac_feat, edge_index, u_basis, evals, edge_weight=edge_weight)

    z_out, _, _, _, _, scale_gates, spatial_gate, z_spec, z_detail = outputs
    boundary = getattr(model.sfib, "last_boundary", None)
    if boundary is None:
        boundary = torch.linalg.vector_norm(z_detail, dim=1, keepdim=True)

    return {
        "model": model,
        "z_out": z_out,
        "z_spec": z_spec,
        "z_detail": z_detail,
        "scale_gates": scale_gates,
        "spatial_gate": spatial_gate,
        "boundary": boundary,
    }



def build_spot_table(state: dict, coords: np.ndarray, eval_adata, cluster_key: str, top_fraction: float) -> pd.DataFrame:
    z_out = to_numpy(state["z_out"]).astype(np.float64)
    z_spec = to_numpy(state["z_spec"]).astype(np.float64)
    z_detail = to_numpy(state["z_detail"]).astype(np.float64)
    scale_gates = to_numpy(state["scale_gates"]).astype(np.float64)
    spatial_gate = to_numpy(state["spatial_gate"]).astype(np.float64)
    boundary = to_numpy(state["boundary"]).reshape(-1).astype(np.float64)

    if scale_gates.ndim != 2 or scale_gates.shape[1] < 2:
        raise ValueError(f"Current SSR expects scale_gates shape [N,2], got {scale_gates.shape}")

    n = z_spec.shape[0]
    obs_names = eval_adata.obs_names.astype(str).tolist() if eval_adata is not None and eval_adata.n_obs == n else [f"spot_{i}" for i in range(n)]

    z_out_norm = np.linalg.norm(z_out, axis=1)
    z_spec_norm = np.linalg.norm(z_spec, axis=1)
    z_detail_norm = np.linalg.norm(z_detail, axis=1)
    correction_ratio = z_detail_norm / np.maximum(z_spec_norm, 1e-12)
    correction_share = z_detail_norm / np.maximum(z_spec_norm + z_detail_norm, 1e-12)
    branch_disagreement = 1.0 - row_cosine(z_spec, z_detail)
    fused_shift = np.linalg.norm(z_out - z_spec, axis=1) if z_out.shape == z_spec.shape else np.full(n, np.nan)

    threshold = float(np.quantile(correction_ratio, 1.0 - float(top_fraction)))

    df = pd.DataFrame(
        {
            "spot": obs_names,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "boundary_score": boundary,
            "z_out_norm": z_out_norm,
            "z_spec_norm": z_spec_norm,
            "z_detail_norm": z_detail_norm,
            "correction_ratio": correction_ratio,
            "correction_share": correction_share,
            "branch_disagreement": branch_disagreement,
            "fused_shift_from_spec": fused_shift,
            "gate_low_rna": scale_gates[:, 0],
            "gate_low_modality2": 1.0 - scale_gates[:, 0],
            "gate_mid_rna": scale_gates[:, 1],
            "gate_mid_modality2": 1.0 - scale_gates[:, 1],
            "gate_mid_minus_low": scale_gates[:, 1] - scale_gates[:, 0],
            "spatial_gate_mean": spatial_gate.mean(axis=1),
            "spatial_gate_std": spatial_gate.std(axis=1),
            "high_residual": correction_ratio >= threshold,
        }
    )
    df.attrs["high_residual_threshold"] = threshold

    if eval_adata is not None and eval_adata.n_obs == n:
        if cluster_key in eval_adata.obs:
            df[cluster_key] = eval_adata.obs[cluster_key].astype(str).to_numpy()
        for key in ("ground_truth", "truth", "label", "Combined_Clusters"):
            if key in eval_adata.obs and key not in df.columns:
                df[key] = eval_adata.obs[key].astype(str).to_numpy()
    return df
