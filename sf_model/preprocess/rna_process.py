import scanpy as sc
import numpy as np
import scipy.sparse as sp

def process_rna_pipeline(
    adata,
    n_top_genes=3000,
    min_cells=3,
    target_sum=1e4,
    apply_scale=False,
):
    """
    完全复刻 MultiGATE / Seurat V3 的处理逻辑
    修正：移除了导致不对齐的细胞过滤步骤
    """
    print("   [RNA] Starting preprocessing (Seurat V3 flavor)...")
    print(f"   [RNA] Input shape: {adata.shape}")

    # 确保数值类型为 float，避免 normalize_total 时的 dtype 冲突
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = np.array(adata.X, dtype=np.float32)

    # 1. 基础过滤
    # ⚠️ 不要在这里 filter_cells；否则会导致 RNA/ATAC 细胞数不一致。

    # 过滤基因是可以的，因为这只改变列数 (Features)，不改变行数 (Cells)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # 2. 高变基因筛选 (Seurat V3) - 必须基于 Raw Counts
    print("   [RNA] Selecting highly variable genes (on Raw Counts)...")
    try:
        n_top_genes = min(int(n_top_genes), int(adata.n_vars))
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            subset=True # 直接裁剪，保留 Top 3000
        )
    except Exception as e:
        print(f"Error in Seurat V3: {e}")
        raise e

    # HVG 子集化后再确保 X 为 float32 且用 dense，保证后续 Scanpy 归一化稳定
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = np.array(adata.X, dtype=np.float32)

    # 3. 标准化 (Normalize) — 直接调用 Scanpy
    print("   [RNA] Normalizing...")
    X = np.asarray(adata.X, dtype=np.float32)
    X[np.isinf(X)] = 0.0
    X[np.isnan(X)] = 0.0
    adata.X = X
    sc.pp.normalize_total(adata, target_sum=float(target_sum))

    # 4. 对数化 (Log1p)
    print("   [RNA] Log transforming...")
    sc.pp.log1p(adata)

    if apply_scale:
        print("   [RNA] Scaling selected HVGs...")
        sc.pp.scale(adata, zero_center=True)
        adata.X = np.asarray(adata.X, dtype=np.float32)
        if not np.isfinite(adata.X).all():
            raise ValueError("RNA matrix contains NaN/Inf after scaling.")

    print(f"   [RNA] Processed shape: {adata.shape}")
    return adata
