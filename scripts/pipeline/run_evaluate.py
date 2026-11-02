import os
import argparse
import json
import re
import logging
import warnings
import sys
import torch
import yaml
import numpy as np
import pandas as pd


def _sanitize_thread_env_vars():
    """Prevent libgomp warnings from invalid thread env values (e.g. 0/empty)."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(key)
        try:
            valid = value is not None and int(str(value).strip()) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            os.environ[key] = "1"


_sanitize_thread_env_vars()

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="nopython is set for njit and is ignored", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")
warnings.filterwarnings("ignore", message="In the future, the default backend for leiden will be igraph instead of leidenalg.*", category=FutureWarning)

import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from shapely.geometry import MultiPoint, Polygon, MultiPolygon
from sf_model.utils import calculate_silhouette_safe, configure_utf8_console, set_seed, validate_ssr_graph

# 引入模型
from sf_model.model.bio_sfinet import BioSFINet

from sf_model.config import load_config as load_config_with_base
from sf_model.model.bio_sfinet_tri import BioSFINetTri
from sf_model.result_adata import load_result_reference_adata
from analysis.meta.jaccard import calculate_spamv_knn_jaccard

def load_config(config_path):
    return load_config_with_base(config_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bio-SFINet & Plot UMAP/Spatial")
    parser.add_argument("--config", default="configs/misar/e11_0_s1.yaml", help="Path to YAML config file")
    # 可在 CLI 覆盖 checkpoint key；若不提供则采用 config 中的 eval.checkpoint
    parser.add_argument("--checkpoint", default=None, help="Checkpoint key or filename; defaults to config eval.checkpoint")
    parser.add_argument("--n-clusters", type=int, default=None, help="Number of clusters for mclust (override config)")
    parser.add_argument("--mclust-pca-dim", type=int, default=None, help="PCA dimension before mclust (override config)")
    parser.add_argument("--resolution", type=float, default=None, help="Leiden resolution (used for metabolomics eval)")
    parser.add_argument("--output-suffix", default=None, help="Optional suffix for output plot/h5ad filenames")
    parser.add_argument("--mclust-model", default=None)
    return parser.parse_args()


def resolve_train_log_path(save_dir):
    save_dir = save_dir.rstrip('/\\')
    if os.path.basename(save_dir) == 'checkpoints':
        return os.path.join(os.path.dirname(save_dir), "train.log")
    return os.path.join(save_dir, "train.log")


def setup_logger(log_path):
    logger = logging.getLogger("SFEvaluate")
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


def torch_load_compat(path, map_location, weights_only):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def is_tri_modal_config(config):
    model_name = str(config.get("model", {}).get("name", "")).strip().lower()
    tri_cfg = config.get("tri_modal", {}) or {}
    return bool(tri_cfg.get("enabled", False)) or model_name == "biosfinetri"


def ensure_r_runtime_available():
    """Ensure rpy2 can locate the R runtime in conda environments."""
    py_bin = os.path.dirname(sys.executable)
    r_bin = os.path.join(py_bin, "R")
    r_home = os.path.abspath(os.path.join(py_bin, "..", "lib", "R"))

    if os.path.exists(r_home) and not os.environ.get("R_HOME"):
        os.environ["R_HOME"] = r_home

    if os.path.exists(r_bin):
        current_path = os.environ.get("PATH", "")
        if py_bin not in current_path.split(":"):
            os.environ["PATH"] = f"{py_bin}:{current_path}" if current_path else py_bin


def resolve_cluster_method(config, eval_cfg):
    method = str(eval_cfg.get("cluster_method", "auto")).strip().lower()
    if method in {"mclust", "leiden"}:
        return method

    sm_data_path = str(config.get("sm_data_path", "")).strip()
    sm_preprocess = config.get("sm_preprocess", {}) or {}
    if sm_data_path or bool(sm_preprocess.get("replace_atac", False)):
        return "leiden"
    return "mclust"


def extract_mclust_labels(res):
    """Robustly extract mclust classification labels from rpy2 result object."""
    try:
        labels = np.array(res.rx2("classification"))
        if labels.size > 0:
            return labels
    except Exception:
        pass

    labels = np.array(res[-2])
    if labels.size == 0:
        raise ValueError("mclust classification is empty.")
    return labels


def infer_epoch_label(ckpt_name, total_epochs=None):
    """
    从 checkpoint 文件名中推断 epoch 标签，便于图上标注。
    数字 checkpoint 显示文件名中的 epoch；ckpt_final.pth 使用配置中的训练末轮。
    """
    base = os.path.splitext(os.path.basename(ckpt_name))[0]
    match = re.search(r"ckpt[_-]?(\d+)", base)
    if match:
        return f"Epoch {match.group(1)}"
    if base == "ckpt_final" and total_epochs is not None:
        return f"Epoch {int(total_epochs)}"
    return base

def calculate_spatial_morans_i(coords, features, k=6):
    """
    基于 squidpy 计算 Moran's I。
    支持对连续 Embedding 或 One-hot 编码后的聚类标签进行评估。
    """
    coords = np.asarray(coords, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[:, np.newaxis]

    # 常量特征会导致 Moran's I 不可定义，先过滤掉并保留回填结构。
    valid_mask = np.var(features, axis=0) > 0
    if not np.any(valid_mask):
        morans_i_vals = np.zeros(features.shape[1], dtype=np.float32)
        return float(np.mean(morans_i_vals)), morans_i_vals

    adata_moran = sc.AnnData(X=features[:, valid_mask])
    adata_moran.obsm['spatial'] = coords

    sq.gr.spatial_neighbors(adata_moran, coord_type='generic', n_neighs=int(k))
    sq.gr.spatial_autocorr(
        adata_moran,
        mode='moran',
        n_perms=None,
        show_progress_bar=False,
    )

    moran_df = adata_moran.uns['moranI']
    valid_vals = moran_df['I'].to_numpy(dtype=np.float32)
    morans_i_vals = np.zeros(features.shape[1], dtype=np.float32)
    morans_i_vals[valid_mask] = valid_vals
    return np.mean(morans_i_vals), morans_i_vals


def calculate_spatial_gearys_c(coords, features, k=6):
    """Calculate Geary's C with the same spatial graph contract as Moran's I."""
    coords = np.asarray(coords, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[:, np.newaxis]

    valid_mask = np.var(features, axis=0) > 0
    if not np.any(valid_mask):
        gearys_c_vals = np.ones(features.shape[1], dtype=np.float32)
        return float(np.mean(gearys_c_vals)), gearys_c_vals

    adata_geary = sc.AnnData(X=features[:, valid_mask])
    adata_geary.obsm["spatial"] = coords
    sq.gr.spatial_neighbors(adata_geary, coord_type="generic", n_neighs=int(k))
    sq.gr.spatial_autocorr(
        adata_geary,
        mode="geary",
        n_perms=None,
        show_progress_bar=False,
    )

    geary_df = adata_geary.uns["gearyC"]
    valid_vals = geary_df["C"].to_numpy(dtype=np.float32)
    gearys_c_vals = np.ones(features.shape[1], dtype=np.float32)
    gearys_c_vals[valid_mask] = valid_vals
    return float(np.mean(gearys_c_vals)), gearys_c_vals


def calculate_spatial_clustering_metrics(coords, cluster_labels, moran_k=6, pas_k=10):
    """Label-free spatial clustering metrics used by SpatialMETA/SDMBench.

    CHAOS and PAS follow SpatialMETA's released benchmark implementation.
    Moran's I and Geary's C here are computed on one-hot cluster indicators;
    they are cluster-coherence diagnostics, not SpatialMETA's marker-feature
    scores.
    """
    coords = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(cluster_labels)
    if coords.ndim != 2 or coords.shape[0] != labels.shape[0]:
        raise ValueError("Spatial coordinates and cluster labels must have matching rows.")
    if coords.shape[0] < 2 or not np.isfinite(coords).all():
        raise ValueError("Spatial clustering metrics require at least two finite spots.")

    standardized = StandardScaler().fit_transform(coords)
    chaos_distance_sum = 0.0
    chaos_spot_count = 0
    for cluster in np.unique(labels):
        cluster_coords = standardized[labels == cluster]
        # Match SpatialMETA's released benchmark: clusters of size <= 2 do
        # not contribute to either the numerator or denominator.
        if cluster_coords.shape[0] <= 2:
            continue
        nearest = NearestNeighbors(n_neighbors=2, metric="euclidean")
        distances = nearest.fit(cluster_coords).kneighbors(
            cluster_coords,
            return_distance=True,
        )[0]
        chaos_distance_sum += float(distances[:, 1].sum())
        chaos_spot_count += int(cluster_coords.shape[0])
    chaos = (
        float(chaos_distance_sum / chaos_spot_count)
        if chaos_spot_count > 0
        else float("nan")
    )

    effective_pas_k = min(int(pas_k), int(coords.shape[0]) - 1)
    if effective_pas_k < 1:
        pas = float("nan")
    else:
        spatial_neighbors = NearestNeighbors(
            n_neighbors=effective_pas_k + 1,
            metric="euclidean",
        ).fit(coords).kneighbors(coords, return_distance=False)
        abnormal = []
        for spot_index, row in enumerate(spatial_neighbors):
            without_self = row[row != spot_index][:effective_pas_k]
            different = int(np.sum(labels[without_self] != labels[spot_index]))
            abnormal.append(different > (effective_pas_k / 2.0))
        pas = float(np.mean(abnormal))

    categorical = pd.Categorical(labels)
    one_hot = np.eye(len(categorical.categories), dtype=np.float32)[categorical.codes]
    cluster_moran, _ = calculate_spatial_morans_i(
        coords,
        one_hot,
        k=int(moran_k),
    )
    cluster_geary, _ = calculate_spatial_gearys_c(
        coords,
        one_hot,
        k=int(moran_k),
    )
    continuity = (
        float(((1.0 - chaos) + (1.0 - pas)) / 2.0)
        if np.isfinite(chaos) and np.isfinite(pas)
        else float("nan")
    )
    return {
        "chaos": chaos,
        "pas": pas,
        "continuity_score": continuity,
        "cluster_moran_i": float(cluster_moran),
        "cluster_geary_c": float(cluster_geary),
        "moran_geary_feature_type": "one_hot_cluster_indicators",
        "moran_geary_k": int(moran_k),
        "pas_k": int(effective_pas_k),
    }


def should_invert_spatial_axis(adata, save_dir=None):
    """Prefer source metadata to decide whether the spatial axis should be flipped."""
    if 'slices_path' in adata.obs.columns:
        slices_path = adata.obs['slices_path'].astype(str).str.lower()
        if slices_path.str.contains('misar').any():
            return True
    if save_dir is not None and 'misar' in save_dir.lower():
        return True
    return False


def rotate_spatial_coords(coords, degrees):
    """Rotate spatial coords around their centroid; positive degrees rotate clockwise."""
    if degrees is None:
        return coords
    try:
        deg = float(degrees)
    except (TypeError, ValueError):
        return coords
    if abs(deg) < 1e-6:
        return coords

    rad = -deg * np.pi / 180.0
    cos_v = float(np.cos(rad))
    sin_v = float(np.sin(rad))

    center = np.mean(coords, axis=0)
    shifted = coords - center
    rotated = np.empty_like(shifted)
    rotated[:, 0] = shifted[:, 0] * cos_v - shifted[:, 1] * sin_v
    rotated[:, 1] = shifted[:, 0] * sin_v + shifted[:, 1] * cos_v
    return rotated + center


def flip_spatial_coords(coords, flip_x=False, flip_y=False):
    """Flip spatial coords around centroid along axes for plotting only."""
    if not flip_x and not flip_y:
        return coords

    center = np.mean(coords, axis=0)
    shifted = coords - center
    flipped = shifted.copy()
    if flip_x:
        flipped[:, 0] = -flipped[:, 0]
    if flip_y:
        flipped[:, 1] = -flipped[:, 1]
    return flipped + center


def calculate_clustering_scores(cluster_labels, ground_truth):
    if ground_truth is None:
        return None

    gt = np.asarray(ground_truth)
    pred = np.asarray(cluster_labels)
    if gt.shape[0] != pred.shape[0]:
        return None

    adata_obs = sc.AnnData(X=np.zeros((gt.shape[0], 1), dtype=np.float32)).obs
    adata_obs["ground_truth"] = gt
    adata_obs["pred_cluster"] = pred
    adata_obs = adata_obs.dropna(subset=["ground_truth", "pred_cluster"])
    if adata_obs.shape[0] < 2:
        return None

    gt_valid = adata_obs["ground_truth"].astype(str).to_numpy()
    pred_valid = adata_obs["pred_cluster"].astype(str).to_numpy()

    non_empty = np.array([
        (g.strip() != "") and (g.lower() != "nan") and (p.strip() != "") and (p.lower() != "nan")
        for g, p in zip(gt_valid, pred_valid)
    ])
    gt_valid = gt_valid[non_empty]
    pred_valid = pred_valid[non_empty]
    if gt_valid.shape[0] < 2:
        return None

    if np.unique(gt_valid).size < 2:
        return None

    contingency = pd.crosstab(
        pd.Series(pred_valid, name="prediction"),
        pd.Series(gt_valid, name="annotation"),
    )
    purity = float(contingency.max(axis=1).sum() / gt_valid.shape[0])

    return {
        "ARI": float(adjusted_rand_score(gt_valid, pred_valid)),
        "NMI": float(normalized_mutual_info_score(gt_valid, pred_valid)),
        "AMI": float(adjusted_mutual_info_score(gt_valid, pred_valid)),
        "HOM": float(homogeneity_score(gt_valid, pred_valid)),
        "COM": float(completeness_score(gt_valid, pred_valid)),
        "VMEASURE": float(v_measure_score(gt_valid, pred_valid)),
        "PURITY": purity,
        "n_valid": int(gt_valid.shape[0]),
    }


def _get_cluster_color_map(adata, cluster_key="cluster"):
    colors = adata.uns.get(f"{cluster_key}_colors", None)
    if not colors:
        return {}
    categories = list(adata.obs[cluster_key].cat.categories)
    return {
        str(cat): colors[i]
        for i, cat in enumerate(categories)
        if i < len(colors)
    }


def _draw_cluster_outlines(ax, coords, labels, outline_cfg=None, color_map=None):
    outline_cfg = outline_cfg or {}
    min_points = int(outline_cfg.get("min_points", 10))
    line_width = float(outline_cfg.get("line_width", 3.0))
    line_alpha = float(outline_cfg.get("alpha", 0.8))
    default_color = str(outline_cfg.get("color", "#000000"))
    use_cluster_colors = bool(outline_cfg.get("use_cluster_colors", True))
    circle_fallback = bool(outline_cfg.get("circle_fallback", True))

    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    max_range = max(x_range, y_range)

    unique_labels = np.unique(labels)
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        pts = coords[mask]
        if pts.shape[0] < min_points:
            continue

        color = default_color
        if use_cluster_colors and color_map is not None:
            color = color_map.get(str(cluster_id), default_color)

        try:
            buffer_ratio = float(outline_cfg.get("buffer_ratio", 0.02))
            buffer_dist = max_range * buffer_ratio

            min_cells_per_island = float(outline_cfg.get("min_cells_per_island", 3.0))
            min_area = np.pi * (buffer_dist * 0.5) ** 2 * min_cells_per_island

            mp = MultiPoint(pts)

            buffer_res = int(outline_cfg.get("buffer_res", 64))

            buffered = mp.buffer(buffer_dist, resolution=buffer_res)
            buffered = buffered.buffer(-buffer_dist * 0.8, resolution=buffer_res)

            smooth_ratio = float(outline_cfg.get("smooth_ratio", 0.2))
            if smooth_ratio > 0:
                smooth_dist = buffer_dist * smooth_ratio
                buffered = buffered.buffer(smooth_dist, resolution=buffer_res).buffer(
                    -smooth_dist, resolution=buffer_res
                )

            drawn_any = False

            def plot_poly(poly, axis, col, lw, alp):
                nonlocal drawn_any
                if poly.is_empty or poly.area < min_area:
                    return
                if poly.geom_type == "Polygon":
                    x, y = poly.exterior.xy
                elif poly.geom_type in ["LineString", "LinearRing"]:
                    x, y = poly.xy
                else:
                    return

                x = np.array(x)
                y = np.array(y)

                if len(x) > 4:
                    try:
                        tck, u = splprep([x, y], s=0.0, per=True)
                        u_new = np.linspace(u.min(), u.max(), 500)
                        x, y = splev(u_new, tck)
                    except Exception:
                        pass

                axis.plot(x, y, color=col, linewidth=lw, alpha=alp)
                drawn_any = True

            if isinstance(buffered, Polygon):
                plot_poly(buffered, ax, color, line_width, line_alpha)
            elif isinstance(buffered, MultiPolygon):
                for geom in buffered.geoms:
                    plot_poly(geom, ax, color, line_width, line_alpha)

            if not drawn_any:
                raise ValueError("All islands were filtered out as small noise.")
        except Exception:
            if not circle_fallback:
                continue
            center = pts.mean(axis=0)
            radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
            circle = plt.Circle(
                (float(center[0]), float(center[1])),
                radius,
                fill=False,
                color=color,
                linewidth=line_width,
                alpha=line_alpha,
            )
            ax.add_patch(circle)


def visualize_and_save(
    z_final,
    coords,
    save_dir,
    reference_adata=None,
    n_clusters=7,
    mclust_pca_dim=25,
    mclust_model="EEE",
    cluster_method="mclust",
    resolution=0.9,
    epoch_label=None,
    ground_truth=None,
    logger=None,
    moran_k=6,
    plot_cfg=None,
    checkpoint_name=None,
    output_suffix=None,
    seed=42,
    jaccard_per_spot=None,
    jaccard_summaries=None,
    evaluation_out=None,
):
    """
    使用 Scanpy 进行降维、聚类和绘图
    z_final: [N, C] 最终的融合特征 (Tensor or Numpy)
    coords: [N, 2] 空间坐标
    """
    # 确保转为 numpy
    if isinstance(z_final, torch.Tensor):
        z_final = z_final.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    jaccard_per_spot = jaccard_per_spot or {}
    jaccard_summaries = jaccard_summaries or {}
    if evaluation_out is None:
        evaluation_out = {}

    # Keep the original z_final-based analysis path independent from export.
    analysis_adata = sc.AnnData(X=z_final)
    coords_plot = coords.copy()
    coords_moran = coords.copy()
    # Y 轴镜像翻转，修正切片方向中的镜面对称问题
    # coords_plot[:, 1] = -1.0 * coords_plot[:, 1]
    plot_cfg = plot_cfg or {}
    coords_plot = rotate_spatial_coords(coords_plot, plot_cfg.get("spatial_rotate_deg"))
    coords_plot = flip_spatial_coords(
        coords_plot,
        flip_x=bool(plot_cfg.get("spatial_flip_x", False)),
        flip_y=bool(plot_cfg.get("spatial_flip_y", False)),
    )
    analysis_adata.obsm["spatial"] = coords_plot

    # 2. 基础分析流程 (Neighbors -> UMAP -> Clustering)
    sc.pp.neighbors(analysis_adata, use_rep='X', random_state=int(seed))
    sc.tl.umap(analysis_adata, random_state=int(seed))
    cluster_method = str(cluster_method).strip().lower()
    if cluster_method not in {"mclust", "leiden"}:
        raise ValueError(f"Unsupported cluster_method={cluster_method}. Use mclust or leiden.")

    # ====== [新增] 声明默认的评估特征变量 ======
    eval_features = z_final
    mclust_pca_features = None

    if cluster_method == "leiden":
        analysis_adata.obs["cluster"] = None
        sc.tl.leiden(analysis_adata, resolution=float(resolution), key_added="cluster", random_state=int(seed))
    else:
        ensure_r_runtime_available()
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri

        rpy2.robjects.numpy2ri.activate()
        robjects.r['set.seed'](int(seed))
        robjects.r('suppressPackageStartupMessages(library(mclust))')
        rmclust = robjects.r['Mclust']
        mclust_model = str(mclust_model).strip().upper()

        # 使用 PCA 将高维特征降维以加速 mclust 计算（维度由配置控制）
        pca_target_dim = max(1, int(mclust_pca_dim))
        pca_dim = min(pca_target_dim, z_final.shape[1], z_final.shape[0])
        pca_model = PCA(n_components=pca_dim, random_state=int(seed))
        z_pca = pca_model.fit_transform(z_final)
        mclust_pca_features = z_pca

        # ====== [新增] 聚类使用 mclust 时，评估特征统一使用降维后的 z_pca ======
        eval_features = z_pca

        res = rmclust(
            rpy2.robjects.numpy2ri.numpy2rpy(z_pca),
            int(n_clusters),
            mclust_model,
            verbose=False,
        )
        mclust_res = extract_mclust_labels(res)
        analysis_adata.obs['cluster'] = mclust_res

    # 强制聚类标签为 category，确保 Scanpy 使用离散类别配色。
    analysis_adata.obs['cluster'] = analysis_adata.obs['cluster'].astype(int).astype(str).astype('category')
    # 固定类别顺序（按数字顺序）确保图例与颜色映射稳定。
    cluster_list = sorted(analysis_adata.obs['cluster'].unique(), key=lambda x: int(x))
    analysis_adata.obs['cluster'] = pd.Categorical(analysis_adata.obs['cluster'], categories=cluster_list, ordered=True)
    analysis_adata.obs["BioSFIB"] = analysis_adata.obs["cluster"].copy()
    analysis_adata.obs["official_cluster"] = analysis_adata.obs["cluster"].copy()

    # ==============================================================================
    # 无侵入式聚合度评估 (Moran's I) - 不打印终端，仅用于顶部图标题
    # ==============================================================================
    moran_title_str = ""
    mi_cluster_avg = None
    cluster_scores = None
    silhouette_val = np.nan

    # 评估视角 1：离散聚类标签的空间连贯性
    cluster_labels = analysis_adata.obs['cluster'].values.astype(int)
    # ====== [修改] 将入参 z_final 替换为 eval_features ======
    silhouette_val = calculate_silhouette_safe(eval_features, cluster_labels, max_samples=10000)
    print(f"Silhouette Score: {silhouette_val:.4f}")
    moran_title_str = f" | Silh: {silhouette_val:.3f}"

    spatial_clustering_scores = {}
    try:
        spatial_clustering_scores = calculate_spatial_clustering_metrics(
            coords_moran,
            cluster_labels,
            moran_k=moran_k,
            pas_k=10,
        )
        spatial_clustering_scores["silhouette"] = float(silhouette_val)
        mi_cluster_avg = float(spatial_clustering_scores["cluster_moran_i"])
        moran_title_str = (
            f" | Moran I: {mi_cluster_avg:.4f}"
            f" | Geary C: {spatial_clustering_scores['cluster_geary_c']:.4f}"
            f" | CHAOS: {spatial_clustering_scores['chaos']:.4f}"
            f" | PAS: {spatial_clustering_scores['pas']:.4f}"
            f" | Silh: {silhouette_val:.3f}"
        )
    except Exception as e:
        spatial_clustering_scores = {"silhouette": float(silhouette_val)}
        moran_title_str = f" | Spatial metrics skipped | Silh: {silhouette_val:.3f}"
        if logger is not None:
            logger.warning("Spatial metrics skipped for %s: %s", checkpoint_name or "default", str(e))
    evaluation_out["spatial_clustering"] = spatial_clustering_scores

    cluster_scores = calculate_clustering_scores(cluster_labels, ground_truth)
    if cluster_scores is not None:
        moran_title_str += (
            f" | ARI: {cluster_scores['ARI']:.4f}"
            f" | NMI: {cluster_scores['NMI']:.4f}"
            f" | AMI: {cluster_scores['AMI']:.4f}"
            f" | HOM: {cluster_scores['HOM']:.4f}"
        )
    for modality_name, summary in jaccard_summaries.items():
        moran_title_str += (
            f" | J-{str(modality_name).upper()}: {float(summary['mean']):.4f}"
        )
    # ==============================================================================

    # 3. 绘图 (UMAP + Spatial)
    # 设置绘图风格
    panel_size = plot_cfg.get("panel_size", [6, 6])
    fig_size = plot_cfg.get("figure_size", [14, 6])
    dpi = int(plot_cfg.get("figure_dpi", 180))
    umap_size = float(plot_cfg.get("umap_point_size", 60))
    spatial_size = float(plot_cfg.get("spatial_point_size", 80))
    alpha = float(plot_cfg.get("alpha", 1.0))
    legend_loc = plot_cfg.get("legend_loc", "on data")

    sc.set_figure_params(dpi=dpi, figsize=tuple(panel_size), frameon=True)

    show_reference_annotation = ground_truth is not None
    n_panels = 3 if show_reference_annotation else 2
    fig, axs = plt.subplots(1, n_panels, figsize=tuple(fig_size))

    # 左图: UMAP (已按要求修改标题)
    sc.pl.umap(
        analysis_adata,
        color='cluster',
        ax=axs[0],
        show=False,
        title='UMAP',
        legend_loc=legend_loc,
        frameon=True,
        size=umap_size,
        alpha=alpha,
        edges=False,
    )

    # 右图: Spatial (物理空间)
    sc.pl.embedding(
        analysis_adata,
        basis='spatial',
        color='cluster',
        ax=axs[1],
        show=False,
        title='Spatial Map',
        size=spatial_size,
        frameon=True,
        alpha=alpha,
        edges=False,
    )

    if show_reference_annotation:
        reference_values = pd.Series(np.asarray(ground_truth), dtype="object")
        reference_values = reference_values.where(reference_values.notna(), "Unknown").astype(str)
        analysis_adata.obs["reference_annotation"] = pd.Categorical(reference_values.to_numpy())
        reference_count = int(analysis_adata.obs["reference_annotation"].nunique())
        sc.pl.embedding(
            analysis_adata,
            basis="spatial",
            color="reference_annotation",
            ax=axs[2],
            show=False,
            title=f"Reference annotation ({reference_count} classes)",
            size=spatial_size,
            frameon=True,
            alpha=alpha,
            edges=False,
        )

    outline_cfg = plot_cfg.get("cluster_outline", {})
    outline_enable = False
    if isinstance(outline_cfg, bool):
        outline_enable = outline_cfg
        outline_cfg = {}
    elif isinstance(outline_cfg, dict):
        outline_enable = bool(outline_cfg.get("enable", False))
    else:
        outline_cfg = {}

    if outline_enable:
        labels = analysis_adata.obs["cluster"].astype(str).to_numpy()
        color_map = _get_cluster_color_map(analysis_adata, "cluster")
        _draw_cluster_outlines(
            axs[1],
            coords_plot,
            labels,
            outline_cfg=outline_cfg,
            color_map=color_map,
        )

    # 极简坐标轴：保留框体，隐藏刻度线与刻度标签
    for ax in axs:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # ========== 修改：自动匹配显微镜视角 ==========
    if should_invert_spatial_axis(analysis_adata, save_dir=save_dir):
        axs[1].invert_yaxis()
        if show_reference_annotation:
            axs[2].invert_yaxis()
    # ==============================================

    # 在图外标注使用的 epoch 信息以及莫兰指数 (全英文)
    if epoch_label:
        title_text = f"Weights: {epoch_label}{moran_title_str}"
        fig.text(
            0.5,
            0.99,
            title_text,
            ha="center",
            va="top",
            fontsize=12
        )

    # 4. 保存图片
    # 如果 save_dir 是 checkpoints 目录，我们把图存到上级的 figures 目录
    if save_dir.rstrip('/').endswith('checkpoints'):
        base_dir = os.path.dirname(save_dir.rstrip('/'))
        fig_dir = os.path.join(base_dir, "figures")
    else:
        fig_dir = os.path.join(save_dir, "figures")

    os.makedirs(fig_dir, exist_ok=True)
    suffix = str(output_suffix).strip() if output_suffix else ""
    plot_name = f"spatial_analysis_{suffix}.pdf" if suffix else "spatial_analysis.pdf"
    plot_path = os.path.join(fig_dir, plot_name)

    # 为顶部文字预留空间
    if epoch_label:
        plt.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # 5. 保存结果 h5ad (方便后续自定义分析)
    pred_dir = os.path.join(os.path.dirname(fig_dir), "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    h5ad_name = f"embedding_joint_{suffix}.h5ad" if suffix else "embedding_joint.h5ad"
    h5ad_path = os.path.join(pred_dir, h5ad_name)
    if reference_adata is not None:
        if int(reference_adata.n_obs) != int(z_final.shape[0]):
            raise ValueError("Reference AnnData spot count does not match BioSFIB embedding.")
        result_adata = sc.AnnData(X=reference_adata.X.copy(), obs=reference_adata.obs.copy(), var=reference_adata.var.copy(), layers={key: value.copy() for key, value in reference_adata.layers.items()})
        result_adata.obsm["spatial"] = np.asarray(reference_adata.obsm["spatial"], dtype=np.float32)
    else:
        result_adata = sc.AnnData(X=np.asarray(z_final, dtype=np.float32))
        result_adata.obsm["spatial"] = np.asarray(coords, dtype=np.float32)
    result_adata.obsm["BioSFIB"] = np.asarray(z_final, dtype=np.float32)
    result_adata.obsm["X_model"] = np.asarray(z_final, dtype=np.float32)
    if mclust_pca_features is not None:
        result_adata.obsm["X_mclust_pca"] = np.asarray(
            mclust_pca_features,
            dtype=np.float32,
        )
    result_adata.obsm["X_umap"] = np.asarray(analysis_adata.obsm["X_umap"], dtype=np.float32)
    if result_adata.n_obs != analysis_adata.n_obs:
        raise ValueError(
            "Result AnnData spot count does not match analysis AnnData: "
            f"{result_adata.n_obs} != {analysis_adata.n_obs}"
        )
    # analysis_adata has a default 0..N-1 index while result_adata preserves
    # raw barcode/spot IDs.  Use positional arrays to avoid Pandas index alignment.
    cluster_values = analysis_adata.obs["cluster"].astype(str).to_numpy()
    cluster_categories = list(analysis_adata.obs["cluster"].cat.categories.astype(str))
    for cluster_key in ("cluster", "BioSFIB", "official_cluster"):
        result_adata.obs[cluster_key] = pd.Categorical(
            cluster_values,
            categories=cluster_categories,
            ordered=True,
        )
        exported = result_adata.obs[cluster_key]
        if exported.isna().any() or exported.nunique(dropna=True) != len(np.unique(cluster_values)):
            raise RuntimeError(f"Invalid exported cluster labels in obs[{cluster_key!r}].")
    if ground_truth is not None:
        result_adata.obs["reference_label"] = np.asarray(ground_truth, dtype=str)
    for modality_name, values in jaccard_per_spot.items():
        safe_name = re.sub(r"[^0-9A-Za-z_]+", "_", str(modality_name).strip().lower()).strip("_")
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (result_adata.n_obs,):
            raise ValueError(
                f"Per-spot Jaccard for {modality_name!r} has invalid shape {values.shape}."
            )
        result_adata.obs[f"jaccard_{safe_name}_k50"] = values
    result_adata.uns["BioSFIB_metadata"] = {"format_version": "biosfib-result-v1", "embedding_key": "BioSFIB", "generic_embedding_key": "X_model", "mclust_embedding_key": "X_mclust_pca" if mclust_pca_features is not None else None, "cluster_key": "BioSFIB", "official_cluster_key": "official_cluster", "cluster_method": cluster_method, "mclust_pca_dim": int(mclust_pca_features.shape[1]) if mclust_pca_features is not None else None, "mclust_model": mclust_model if cluster_method == "mclust" else None, "checkpoint": checkpoint_name, "epoch": epoch_label, "seed": int(seed)}
    result_adata.uns["BioSFIB_evaluation"] = {
        "ground_truth_metrics": cluster_scores or {},
        "jaccard": jaccard_summaries,
        "spatial_clustering": spatial_clustering_scores,
    }
    result_adata.write(h5ad_path)
    abs_plot_path = os.path.abspath(plot_path)
    abs_h5ad_path = os.path.abspath(h5ad_path)
    if logger is not None:
        logger.info("Artifact plot: %s", abs_plot_path)
        logger.info("Artifact h5ad: %s", abs_h5ad_path)
    print(f"Artifact plot: {abs_plot_path}")
    print(f"Artifact h5ad: {abs_h5ad_path}")

    metric_tag = epoch_label or "unknown"
    ckpt_tag = checkpoint_name or "unknown"

    def _fmt_metric(value):
        return "NA" if value is None else f"{float(value):.4f}"

    if logger is not None:
        logger.info(
            "Eval metrics | checkpoint=%s | epoch=%s | cluster_moran=%s | cluster_geary=%s | CHAOS=%s | PAS=%s | silhouette=%s",
            ckpt_tag,
            metric_tag,
            _fmt_metric(mi_cluster_avg),
            _fmt_metric(spatial_clustering_scores.get("cluster_geary_c")),
            _fmt_metric(spatial_clustering_scores.get("chaos")),
            _fmt_metric(spatial_clustering_scores.get("pas")),
            _fmt_metric(silhouette_val),
        )

    if cluster_scores is not None:
        if logger is not None:
            logger.info(
                "GT metrics | checkpoint=%s | epoch=%s | ARI=%.4f | NMI=%.4f | AMI=%.4f | HOM=%.4f | n_valid=%d",
                ckpt_tag,
                metric_tag,
                cluster_scores['ARI'],
                cluster_scores['NMI'],
                cluster_scores['AMI'],
                cluster_scores['HOM'],
                cluster_scores['n_valid'],
            )
    else:
        if logger is not None:
            logger.warning("Ground truth unavailable or invalid; skip ARI/NMI/AMI/HOM.")

    return plot_path, h5ad_path, cluster_scores

def main():
    configure_utf8_console()
    args = parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载配置
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    seed = int(config['project'].get('seed', 42))
    processed_dir = config['data']['processed_path']
    save_dir = config['project']['save_dir']
    log_path = resolve_train_log_path(save_dir)
    logger = setup_logger(log_path)

    logger.info("[Phase 3] Evaluation started.")

    # 2. 加载数据 (直接读预处理好的 Tensor)
    data_path = os.path.join(processed_dir, "processed_data.pt")
    if not os.path.exists(data_path):
        logger.error("Data not found at %s", data_path)
        logger.error("Please run 'python run_preprocess.py' first.")
        raise FileNotFoundError(data_path)

    logger.info("Loading processed data...")
    data_dict = torch_load_compat(data_path, map_location='cpu', weights_only=False)
    expected_knn_k = int(config['data']['parameters']['knn_k'])
    validate_ssr_graph(data_dict, expected_knn_k=expected_knn_k)

    # 提取 Tensor 并转到 GPU
    reference_adata = load_result_reference_adata(config, data_dict["coords"])
    rna_feat = data_dict["rna_feat"].to(device)
    atac_feat = data_dict["atac_feat"].to(device)
    coords = data_dict["coords"].to(device)
    edge_index = data_dict["edge_index"].to(device)
    edge_weight = data_dict.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    u_basis = data_dict["u_basis"].to(device)
    evals = data_dict.get("evals", None)
    if evals is not None:
        evals = evals.to(device)
    rna_dim = int(data_dict.get("rna_dim", rna_feat.shape[1]))
    atac_dim = data_dict["atac_dim"]
    ground_truth = data_dict.get("ground_truth", None)

    # 3. 初始化模型
    logger.info("Initializing model...")
    config['model']['rna_in_dim'] = rna_dim
    tri_modal = bool(config.get("tri_modal", {}).get("enabled", False))
    if tri_modal:
        adt_feat = data_dict["adt_feat"].to(device)
        model = BioSFINetTri(config, adt_dim=adt_feat.shape[1], atac_dim=atac_dim).to(device)
    else:
        model = BioSFINet(config, atac_dim=atac_dim).to(device)

    # 4. 加载权重
    eval_cfg = config.get('eval', {})
    plot_cfg = eval_cfg.get('plotting', {})
    compute_gt_metrics = bool(eval_cfg.get('compute_gt_metrics', True))
    cluster_method = resolve_cluster_method(config, eval_cfg)
    n_clusters = int(args.n_clusters if args.n_clusters is not None else eval_cfg.get('n_clusters', 7))
    mclust_pca_dim = int(args.mclust_pca_dim if args.mclust_pca_dim is not None else eval_cfg.get('mclust_pca_dim', 20))
    resolution = float(args.resolution if args.resolution is not None else eval_cfg.get('resolution', 0.9))
    moran_k = int(eval_cfg.get('moran_k', 6))
    ckpt_key = args.checkpoint or eval_cfg.get('checkpoint', 'ckpt_final.pth')
    ckpt_map = eval_cfg.get('checkpoints', {})
    # 如果 key 存在映射则取映射，否则允许直接把文件名作为 key 使用
    ckpt_name = ckpt_map.get(ckpt_key, ckpt_key)
    ckpt_path = os.path.join(save_dir, ckpt_name)

    if not compute_gt_metrics:
        ground_truth = None
        logger.info("GT metrics disabled by config: eval.compute_gt_metrics=false")

    if not os.path.exists(ckpt_path):
        logger.error("Checkpoint not found at %s", ckpt_path)
        logger.error("Please train the model first using 'run_train.py'")
        raise FileNotFoundError(ckpt_path)

    logger.info("Loading checkpoint: %s", ckpt_name)
    logger.info("Clustering method: %s", cluster_method)
    logger.info("mclust PCA dim: %d", mclust_pca_dim)
    # 加载参数 (处理可能的 key 不匹配问题)
    state_dict = torch_load_compat(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    total_epochs = config.get('train', {}).get('epochs')
    epoch_label = infer_epoch_label(ckpt_name, total_epochs=total_epochs)

    # 5. 推理 (Inference)
    logger.info("Inference running...")
    with torch.no_grad():
        # Forward pass (single fused tower)
        if tri_modal:
            outputs = model(rna_feat, adt_feat, atac_feat, edge_index, u_basis, evals, edge_weight=edge_weight)
        else:
            outputs = model(rna_feat, atac_feat, edge_index, u_basis, evals, edge_weight=edge_weight)
        z_final = outputs[0]

    ssr_diagnostics = {}
    if not tri_modal and isinstance(outputs, (tuple, list)) and len(outputs) >= 9:
        scale_gates = outputs[5]
        spatial_gate = outputs[6]
        z_spec = outputs[7]
        z_detail = outputs[8]
        spec_norm = torch.linalg.vector_norm(z_spec, dim=-1).mean()
        detail_norm = torch.linalg.vector_norm(z_detail, dim=-1).mean()
        ssr_diagnostics = {
            "detail_ratio": float((detail_norm / spec_norm.clamp_min(1e-8)).cpu()),
            "z_spec_norm_mean": float(spec_norm.cpu()),
            "z_detail_norm_mean": float(detail_norm.cpu()),
            "scale_gate_mean": float(scale_gates.mean().cpu()),
            "scale_gate_std": float(scale_gates.std(unbiased=False).cpu()),
            "scale_gate_min": float(scale_gates.min().cpu()),
            "scale_gate_max": float(scale_gates.max().cpu()),
            "spatial_gate_mean": float(spatial_gate.mean().cpu()),
            "spatial_gate_std": float(spatial_gate.std(unbiased=False).cpu()),
            "spatial_gate_min": float(spatial_gate.min().cpu()),
            "spatial_gate_max": float(spatial_gate.max().cpu()),
        }

    logger.info("Latent shape: %s", z_final.shape)

    jaccard_per_spot = {}
    jaccard_summaries = {}
    if bool(eval_cfg.get("compute_jaccard", False)):
        if tri_modal:
            raise ValueError("The retained Jaccard protocol is defined for two-modal data")
        other_label = str(eval_cfg.get("jaccard_other_label", "other"))
        for name, features in (("rna", rna_feat), (other_label, atac_feat)):
            per_spot, summary = calculate_spamv_knn_jaccard(z_final, features, k=int(eval_cfg.get("jaccard_k", 50)))
            summary["learned_embedding"] = "z_out_before_mclust_pca"
            summary["omics_features"] = "processed_data.pt:rna_feat" if name == "rna" else "processed_data.pt:atac_feat_as_metabolome"
            jaccard_per_spot[name] = per_spot
            jaccard_summaries[name] = summary
    evaluation_summary = {}

    # 6. 可视化
    plot_path, h5ad_path, cluster_scores = visualize_and_save(
        z_final,
        coords,
        save_dir,
        reference_adata=reference_adata,
        mclust_model=args.mclust_model or eval_cfg.get("mclust_model", "EEE"),
        n_clusters=n_clusters,
        mclust_pca_dim=mclust_pca_dim,
        cluster_method=cluster_method,
        resolution=resolution,
        epoch_label=epoch_label,
        ground_truth=ground_truth,
        logger=logger,
        moran_k=moran_k,
        plot_cfg=plot_cfg,
        checkpoint_name=ckpt_name,
        output_suffix=args.output_suffix,
        seed=seed,
        jaccard_per_spot=jaccard_per_spot,
        jaccard_summaries=jaccard_summaries,
        evaluation_out=evaluation_summary,
    )

    ari_value = None if cluster_scores is None else float(cluster_scores["ARI"])
    evaluation_result = {
        "checkpoint": ckpt_name,
        "epoch": epoch_label,
        "seed": seed,
        "fixed_embedding": "z_out",
        "ari": ari_value,
        "jaccard": jaccard_summaries,
        "spatial_clustering": evaluation_summary.get("spatial_clustering", {}),
        **ssr_diagnostics,
    }
    metrics_name = (
        f"formal_metrics_{args.output_suffix}.json"
        if args.output_suffix
        else "formal_metrics.json"
    )
    metrics_path = os.path.join(os.path.dirname(h5ad_path), metrics_name)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(evaluation_result, handle, ensure_ascii=False, indent=2)
    logger.info("Evaluation metrics: %s", evaluation_result)
    print(f"   Metrics: {os.path.abspath(metrics_path)}")

    abs_log_path = os.path.abspath(log_path)
    logger.info("Evaluation complete.")
    logger.info("Eval log: %s", abs_log_path)
    print("\n\u2705 Evaluation complete")
    print(f"   Figure: {os.path.abspath(plot_path)}")
    print(f"   Embedding: {os.path.abspath(h5ad_path)}")
    print(f"   Log: {abs_log_path}")
    if os.environ.get("SF_PIPELINE_RUN") != "1":
        append_log_separator(log_path)

if __name__ == "__main__":
    main()
