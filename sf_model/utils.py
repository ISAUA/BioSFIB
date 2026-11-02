import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


def configure_utf8_console():
    """Use UTF-8 consistently for Python and the Windows console."""
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except (AttributeError, OSError):
            pass
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass


def validate_ssr_graph(data_dict, expected_knn_k=9):
    """Fail fast when SSR is given a legacy or inconsistent processed graph."""
    mode = str(data_dict.get("graph_edge_weight_mode", "")).strip().lower()
    if mode not in {"spatial_distance", "physical", "physical_distance", "spatial", "distance"}:
        raise ValueError(
            f"SSR requires graph_edge_weight_mode=spatial_distance, got {mode or 'missing'}"
        )
    actual_knn_k = int(data_dict.get("graph_knn_k", -1))
    expected_knn_k = int(expected_knn_k)
    if actual_knn_k != expected_knn_k:
        raise ValueError(
            f"SSR graph k mismatch: config expects {expected_knn_k}, "
            f"processed data contains {actual_knn_k}"
        )
    edge_index = data_dict["edge_index"].detach().cpu().long()
    edge_weight = data_dict.get("edge_weight")
    if edge_weight is None:
        raise ValueError("SSR requires Gaussian physical edge_weight")
    edge_weight = edge_weight.detach().cpu().float().flatten()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2,E], got {tuple(edge_index.shape)}")
    if edge_index.shape[1] != edge_weight.numel():
        raise ValueError("edge_index and edge_weight lengths differ")
    src, dst = edge_index
    if torch.any(src == dst):
        raise ValueError("SSR physical graph must not contain self-loops")
    n_nodes = int(max(src.max().item(), dst.max().item()) + 1)
    codes = src * n_nodes + dst
    reverse_codes = dst * n_nodes + src
    codes_sorted, order = torch.sort(codes)
    reverse_sorted, reverse_order = torch.sort(reverse_codes)
    if not torch.equal(codes_sorted, reverse_sorted):
        raise ValueError("SSR physical graph is not strictly symmetric")
    if not torch.allclose(edge_weight[order], edge_weight[reverse_order], rtol=1e-6, atol=1e-7):
        raise ValueError("SSR reverse edges do not share the same weight")


def build_spatial_graph(coords, k=10):
#     """
#     构建空间 KNN 图并计算 GFT 基底。
#     对应框架: Phase 1 - Global Graph Basis Construction

#     Args:
#         coords: [N, 2] 原始物理坐标 (numpy array)
#         k: 近邻数
#     Returns:
#         edge_index: [2, E] 图的边索引 (供 GAT/GNN 使用)
#         u_basis: [N, N] 拉普拉斯矩阵的特征向量矩阵 (供 GFT 使用)
#     """
#     N = coords.shape[0]

#     # 1. 构建 KNN 图
#     nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
#     distances, indices = nbrs.kneighbors(coords)

#     # 构建边索引 (Edge Index)
#     # indices 第一列是自己，从第二列开始是邻居
#     src = np.repeat(np.arange(N), k)
#     dst = indices[:, 1:].flatten()

#     # 转为 PyG 格式的 edge_index [2, E]
#     edge_index = torch.tensor([src, dst], dtype=torch.long)

#     # 2. 构建归一化拉普拉斯矩阵 L
#     # 构造稀疏邻接矩阵
#     data = np.ones(len(src))
#     adj = sp.coo_matrix((data, (src, dst)), shape=(N, N))

#     # 对称化 (变为无向图)
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

#     # 计算度矩阵 D
#     degree = np.array(adj.sum(1)).flatten()
#     d_inv_sqrt = np.power(degree, -0.5)
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

#     # Normalized Laplacian: L = I - D^-1/2 * A * D^-1/2
#     normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
#     laplacian = sp.eye(N) - normalized_adj

#     # 3. 特征分解 (Eigen Decomposition)
#     # L = U * Lambda * U^T
#     # 对于 N=2500，dense solver (eigh) 速度很快
#     evals, evecs = np.linalg.eigh(laplacian.toarray())

#     # 排序 (低频 -> 高频)
#     idx = np.argsort(evals)
#     # evals = evals[idx]
#     evecs = evecs[:, idx]

#     # 转为 Tensor
#     u_basis = torch.FloatTensor(evecs)

#     return edge_index, u_basis


def build_spatial_graph(
    coords,
    features=None,
    k=6,
    device=None,
    n_freq_components=None,
    edge_weight_mode=None,
    compute_gft_basis=True,
):
    """
    构建空间 KNN 图并计算 GFT 基底。
    - rna_similarity: 空间 KNN 邻居 + RNA 特征相似度边权
    - spatial_distance: 空间 KNN 邻居 + 物理距离高斯边权
    - topology: 空间 KNN 邻居 + 无权图
    """
    if isinstance(coords, torch.Tensor):
        inferred_device = coords.device
        coords_np = coords.detach().cpu().numpy()
    else:
        inferred_device = torch.device("cpu")
        coords_np = np.asarray(coords)

    target_device = device if device is not None else inferred_device
    N = coords_np.shape[0]

    # 1. 构建物理 KNN 图 (确定谁是邻居)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords_np)
    distances, indices = nbrs.kneighbors(coords_np)

    # 排除自身 (第0列是自身)
    src = np.repeat(np.arange(N), k)
    dst = indices[:, 1:].flatten()

    mode = str(edge_weight_mode or ("rna_similarity" if features is not None else "topology")).strip().lower()
    mode_aliases = {
        "rna": "rna_similarity",
        "rna_feature": "rna_similarity",
        "rna_features": "rna_similarity",
        "feature": "rna_similarity",
        "features": "rna_similarity",
        "physical": "spatial_distance",
        "physical_distance": "spatial_distance",
        "spatial": "spatial_distance",
        "distance": "spatial_distance",
        "unweighted": "topology",
        "none": "topology",
    }
    mode = mode_aliases.get(mode, mode)
    if mode not in {"rna_similarity", "spatial_distance", "topology"}:
        raise ValueError(
            f"Unsupported edge_weight_mode={edge_weight_mode}. "
            "Use rna_similarity, spatial_distance, or topology."
        )

    # 2. 计算边权重
    if mode == "rna_similarity":
        if features is None:
            raise ValueError("edge_weight_mode='rna_similarity' requires features.")
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        else:
            features = np.asarray(features)

        f_src = features[src]
        f_dst = features[dst]

        if sp.issparse(features):
            f_src = f_src.toarray()
            f_dst = f_dst.toarray()

        dist_sq = np.sum((f_src - f_dst) ** 2, axis=1)
        sigma2 = np.median(dist_sq)
        if sigma2 == 0:
            sigma2 = 1e-4  # 防止除零
        weights = np.exp(-dist_sq / (2 * sigma2)) + 1e-4
    elif mode == "spatial_distance":
        dist_sq = (distances[:, 1:] ** 2).flatten()
        sigma2 = np.median(dist_sq)
        if sigma2 == 0:
            sigma2 = 1e-4
        weights = np.exp(-dist_sq / (2 * sigma2)) + 1e-4
    else:
        weights = np.ones(len(src))

    #   如果同时使用特征和空间距离，可以在此处合成权重，例如：
    #     # 2. 计算边权重：特征核 × 空间核
    # spatial_dist_sq = (distances[:, 1:] ** 2).flatten()
    # sigma2_spa = np.median(spatial_dist_sq)
    # if sigma2_spa == 0:
    #     sigma2_spa = 1e-4
    # w_spa = np.exp(-spatial_dist_sq / (2 * sigma2_spa))

    # if features is not None:
    #     f_src = features[src]
    #     f_dst = features[dst]
    #     if sp.issparse(features):
    #         f_src = f_src.toarray()
    #         f_dst = f_dst.toarray()
    #     feat_dist_sq = np.sum((f_src - f_dst) ** 2, axis=1)
    #     sigma2_feat = np.median(feat_dist_sq)
    #     if sigma2_feat == 0:
    #         sigma2_feat = 1e-4
    #     w_feat = np.exp(-feat_dist_sq / (2 * sigma2_feat))
    # else:
    #     w_feat = np.ones(len(src))

    # # 合成权重并加微小正则保持连通
    # weights = w_spa * w_feat + 1e-4

    # 3. 构建稀疏邻接矩阵
    adj = sp.coo_matrix((weights, (src, dst)), shape=(N, N))

    # 对称化并以该邻接同时服务 Laplacian/GFT 与所有消息传递模块。
    # 旧实现虽然用对称 adj 计算 Laplacian，却返回原始有向 KNN 边，导致
    # 频谱图和 GNN 图不一致。
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()

    # 4. 计算归一化拉普拉斯矩阵
    degree = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    laplacian = (sp.eye(N, format="csr") - normalized_adj).tocsr()

    graph_coo = adj.tocoo()
    edge_index = torch.tensor(
        np.vstack([graph_coo.row, graph_coo.col]),
        dtype=torch.long,
    )
    edge_weight = torch.as_tensor(graph_coo.data, dtype=torch.float32)
    if not bool(compute_gft_basis):
        # Full-spectrum polynomial filters consume the sparse graph directly.
        # Empty tensors preserve the processed-data contract without an O(N^3)
        # dense eigendecomposition that the model would never read.
        u_basis = torch.empty((int(N), 0), dtype=torch.float32)
        evals = torch.empty((0,), dtype=torch.float32, device=target_device)
        return edge_index, edge_weight, u_basis, evals

    # 5. 全量密集特征分解后再做低频截断
    if n_freq_components is None:
        target_k = int(N)
    else:
        target_k = int(n_freq_components)
        if target_k <= 0:
            raise ValueError(f"n_freq_components must be positive, got {target_k}.")
        target_k = min(target_k, int(N))

    evals, evecs = np.linalg.eigh(laplacian.toarray())

    idx = np.argsort(evals)
    evals = np.asarray(evals[idx], dtype=np.float32)
    evecs = np.asarray(evecs[:, idx], dtype=np.float32)

    evals = evals[:target_k]
    evecs = evecs[:, :target_k]

    u_basis = torch.FloatTensor(evecs)
    evals = torch.FloatTensor(evals).to(target_device)
    return edge_index, edge_weight, u_basis, evals


def calculate_silhouette_safe(features, labels, max_samples=10000, random_state=42):
    """
    计算轮廓系数 (Silhouette Score)。
    如果样本量大于 max_samples，则使用无放回随机降采样，以防止计算 OOM。
    """
    features = np.asarray(features)
    labels = np.asarray(labels)
    n_samples = features.shape[0]

    if labels.shape[0] != n_samples or n_samples < 2:
        return np.nan

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return np.nan

    max_samples = int(max_samples) if max_samples is not None else None
    if max_samples is not None and n_samples > max_samples:
        rng = np.random.default_rng(int(random_state))
        indices = rng.choice(n_samples, max_samples, replace=False)
        features_to_use = features[indices]
        labels_to_use = labels[indices]
    else:
        features_to_use = features
        labels_to_use = labels

    try:
        score = silhouette_score(features_to_use, labels_to_use)
        return float(score)
    except Exception as e:
        print(f"[Warning] Silhouette calculation failed: {e}")
        return np.nan



def set_seed(seed: int = 42, deterministic: bool = True):
    """全局锁定随机种子，尽可能消除非确定性。"""
    seed = int(seed)

    def _ensure_positive_thread_env(var_name: str):
        value = os.environ.get(var_name)
        try:
            valid = value is not None and int(str(value).strip()) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            os.environ[var_name] = "1"

    # CUDA + CuBLAS 在 deterministic 模式下需要该环境变量。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # 线程库并行会引入浮点规约顺序差异，固定为单线程更稳定。
    _ensure_positive_thread_env("OMP_NUM_THREADS")
    _ensure_positive_thread_env("MKL_NUM_THREADS")
    _ensure_positive_thread_env("OPENBLAS_NUM_THREADS")
    _ensure_positive_thread_env("NUMEXPR_NUM_THREADS")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    return seed
