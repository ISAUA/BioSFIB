import gzip
import numpy as np
import sys
import pandas as pd
import scanpy as sc
from scipy import sparse
from tqdm import tqdm
import re


_PEAK_PATTERN = re.compile(
    r"^\s*(?:chr)?([A-Za-z0-9_.]+?)[^A-Za-z0-9]+(\d+)[^A-Za-z0-9]+(\d+)\s*$",
    flags=re.IGNORECASE,
)


def canonicalize_peak_name(peak_name):
    """Return a strict ``chr:start-end`` identifier for one genomic interval."""
    raw = str(peak_name)
    match = _PEAK_PATTERN.match(raw)
    if match is None:
        raise ValueError(f"Unparseable peak coordinate: {raw!r}")
    chrom = match.group(1)
    if chrom.lower() == "m":
        chrom = "M"
    elif chrom.lower() in {"x", "y"}:
        chrom = chrom.upper()
    elif chrom.isdigit():
        chrom = str(int(chrom))
    start, end = int(match.group(2)), int(match.group(3))
    if start < 0 or end <= start:
        raise ValueError(f"Invalid peak interval: {raw!r}")
    return f"chr{chrom}:{start}-{end}"


def canonicalize_peak_names(peak_names):
    """Canonicalize peak IDs and fail on parse errors or coordinate collisions."""
    canonical = []
    failures = []
    sources = {}
    for value in peak_names:
        try:
            name = canonicalize_peak_name(value)
        except ValueError:
            failures.append(str(value))
            continue
        canonical.append(name)
        sources.setdefault(name, []).append(str(value))
    if failures:
        raise ValueError(
            f"Failed to canonicalize {len(failures)} peak names; examples={failures[:5]}"
        )
    collisions = {key: values for key, values in sources.items() if len(values) > 1}
    if collisions:
        examples = list(collisions.items())[:5]
        raise ValueError(
            f"Peak canonicalization produced {len(collisions)} coordinate collisions; "
            f"examples={examples}"
        )
    return canonical


def match_peak_intervals_one_to_one(current_peak_names, target_peak_names, min_iou=0.5):
    """Match target intervals to current intervals with deterministic 1:1 IoU matching.

    Both inputs may use any delimiter accepted by :func:`canonicalize_peak_name`.
    Candidate overlaps are ranked by descending intersection-over-union, then
    descending overlap length, target order, and current order. A current peak
    can therefore never be copied into more than one target feature.
    """
    min_iou = float(min_iou)
    if not 0.0 < min_iou <= 1.0:
        raise ValueError(f"min_iou must be in (0, 1], got {min_iou}.")

    current_names = [canonicalize_peak_name(value) for value in current_peak_names]
    target_names = [canonicalize_peak_name(value) for value in target_peak_names]
    if len(set(current_names)) != len(current_names):
        raise ValueError("Current peak IDs are not unique after canonicalization.")
    if len(set(target_names)) != len(target_names):
        raise ValueError("Target peak IDs are not unique after canonicalization.")

    def as_interval(name, index):
        chrom, coords = name[3:].split(":", 1)
        start, end = coords.split("-", 1)
        return chrom, int(start), int(end), int(index)

    current_by_chrom = {}
    target_by_chrom = {}
    for index, name in enumerate(current_names):
        chrom, start, end, original_index = as_interval(name, index)
        current_by_chrom.setdefault(chrom, []).append((start, end, original_index))
    for index, name in enumerate(target_names):
        chrom, start, end, original_index = as_interval(name, index)
        target_by_chrom.setdefault(chrom, []).append((start, end, original_index))

    candidates = []
    target_candidate_counts = {}
    current_candidate_counts = {}
    for chrom, target_intervals in target_by_chrom.items():
        current_intervals = sorted(current_by_chrom.get(chrom, []))
        if not current_intervals:
            continue
        active = []
        cursor = 0
        for target_start, target_end, target_index in sorted(target_intervals):
            while cursor < len(current_intervals) and current_intervals[cursor][0] < target_end:
                active.append(current_intervals[cursor])
                cursor += 1
            active = [entry for entry in active if entry[1] > target_start]
            for current_start, current_end, current_index in active:
                overlap = min(target_end, current_end) - max(target_start, current_start)
                if overlap <= 0:
                    continue
                union = max(target_end, current_end) - min(target_start, current_start)
                iou = overlap / union
                if iou + 1e-12 < min_iou:
                    continue
                candidates.append((-iou, -overlap, target_index, current_index))
                target_candidate_counts[target_index] = target_candidate_counts.get(target_index, 0) + 1
                current_candidate_counts[current_index] = current_candidate_counts.get(current_index, 0) + 1

    target_to_current = [-1] * len(target_names)
    match_iou = [0.0] * len(target_names)
    used_current = set()
    for negative_iou, _, target_index, current_index in sorted(candidates):
        if target_to_current[target_index] >= 0 or current_index in used_current:
            continue
        target_to_current[target_index] = current_index
        match_iou[target_index] = -negative_iou
        used_current.add(current_index)

    matched_targets = [index for index, value in enumerate(target_to_current) if value >= 0]
    exact_count = sum(
        target_names[target_index] == current_names[target_to_current[target_index]]
        for target_index in matched_targets
    )
    matched_scores = [match_iou[index] for index in matched_targets]
    stats = {
        "mode": "genomic_interval_iou_one_to_one",
        "target_count": int(len(target_names)),
        "current_count": int(len(current_names)),
        "matched_count": int(len(matched_targets)),
        "missing_count": int(len(target_names) - len(matched_targets)),
        "overlap_fraction": float(len(matched_targets) / len(target_names)) if target_names else 0.0,
        "exact_count": int(exact_count),
        "interval_shifted_count": int(len(matched_targets) - exact_count),
        "minimum_iou": float(min_iou),
        "mean_matched_iou": float(np.mean(matched_scores)) if matched_scores else 0.0,
        "median_matched_iou": float(np.median(matched_scores)) if matched_scores else 0.0,
        "candidate_pair_count": int(len(candidates)),
        "ambiguous_target_count": int(sum(value > 1 for value in target_candidate_counts.values())),
        "ambiguous_current_count": int(sum(value > 1 for value in current_candidate_counts.values())),
    }
    return target_to_current, match_iou, stats

def parse_gtf_tss(gtf_path):
    """
    解析 GTF 文件 (支持 .gz)，提取基因 TSS
    """
    print(f"Parsing GTF: {gtf_path} ...")
    gene_tss = {}

    # 自动判断压缩
    if gtf_path.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'

    try:
        with open_func(gtf_path, mode) as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split('\t')
                if len(parts) < 9 or parts[2] != 'gene': continue

                try:
                    attr = parts[8]
                    # 适配 gencode 格式
                    if 'gene_name "' in attr:
                        name = attr.split('gene_name "')[1].split('"')[0]
                    elif 'gene_name=' in attr:
                        name = attr.split('gene_name=')[1].split(';')[0]
                    else:
                        continue

                    strand = parts[6]
                    start, end = int(parts[3]), int(parts[4])
                    tss = start if strand == '+' else end
                    chrom = parts[0]
                    if not chrom.startswith('chr'): chrom = 'chr' + chrom
                    gene_tss[name] = (chrom, tss)
                except:
                    continue
    except FileNotFoundError:
        print(f"Error: GTF file not found at {gtf_path}")
        return {}

    return gene_tss

def parse_peak_coords(peak_names):
    """
    智能解析 Peak 坐标，支持多种分隔符
    Formats supported:
      - chr1:100-200
      - chr1-100-200
      - chr1,100,200
      - chr1_100_200
    """
    parsed = []
    # 正则表达式提取: (chr...) (数字) (数字)
    # 忽略中间的分隔符
    pattern = re.compile(r"(chr[\w\d]+)[^\w\d]+(\d+)[^\w\d]+(\d+)")

    print(f"DEBUG: First 5 peak names in data: {peak_names[:5]}")

    for p in peak_names:
        # 尝试正则匹配
        match = pattern.search(p)
        if match:
            chrom = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            center = (start + end) // 2
            parsed.append((chrom, center))
        else:
            # 备用方案：如果名字里没有 chr，可能只是 1:100-200
            try:
                # 尝试通过分割符暴力拆解
                parts = re.split(r'[:\-_,]', p)
                if len(parts) >= 3:
                    c = parts[0] if parts[0].startswith('chr') else 'chr' + parts[0]
                    s = int(parts[1])
                    e = int(parts[2])
                    parsed.append((c, (s+e)//2))
                else:
                    parsed.append((None, None))
            except:
                parsed.append((None, None))

    return parsed

def filter_peaks_by_tss(adata, gtf_path, rna_genes, window=100000, n_final=30000):
    """
    根据 RNA 基因的 TSS 筛选物理相关的 Peaks，同时构建 Gene-Peak 掩码矩阵。

    返回:
      filtered_adata: 筛选后的 ATAC AnnData
      gene_peak_mask: scipy.sparse.coo_matrix, shape = (len(rna_genes), n_peaks_kept)
    """
    print(f"Executing Physical Association Filter (Window = +/- {window}bp)...")
    stats = {
        "gtf_path": str(gtf_path),
        "window": int(window),
        "rna_gene_count": int(len(rna_genes) if rna_genes is not None else 0),
        "matched_gene_count": 0,
        "peaks_before": int(adata.shape[1]),
        "peaks_valid": 0,
        "peaks_retained": 0,
        "fallback_used": False,
        "status": "started",
    }

    # 1. 准备 GTF 数据
    gene_tss = parse_gtf_tss(gtf_path)
    target_genes = sorted(set(rna_genes) & set(gene_tss.keys()))
    stats["matched_gene_count"] = int(len(target_genes))
    print(f"Matching {len(target_genes)} RNA genes to Peaks...")

    if len(target_genes) == 0:
        print("Warning: No matching genes found between RNA data and GTF file!")
        stats["status"] = "no_matching_genes"
        adata.uns["gtf_filter_stats"] = stats
        return adata, None

    # 2. 解析 Peak 坐标
    peak_coords = parse_peak_coords(adata.var_names.tolist())

    # 构建 DataFrame
    peaks_df = pd.DataFrame(peak_coords, columns=['chrom', 'center'], index=adata.var_names)

    # 移除解析失败的 Peaks (None)
    valid_peaks_mask = peaks_df['chrom'].notna()
    stats["peaks_valid"] = int(valid_peaks_mask.sum())
    peaks_df = peaks_df[valid_peaks_mask]

    if len(peaks_df) == 0:
        print("Error: Failed to parse any peak coordinates! Check 'DEBUG' output above.")
        stats["status"] = "no_valid_peaks"
        empty_adata = adata[:, []].copy()
        empty_adata.uns["gtf_filter_stats"] = stats
        return empty_adata, None  # Return empty to trigger error safely

    # 3. 筛选逻辑，记录 gene-peak 对应关系
    keep_names_set = set()
    row_indices = []
    col_names = []
    gene_to_row = {g: i for i, g in enumerate(rna_genes)}

    # UTF-8 console configuration keeps tqdm's block bar readable on Windows.
    for gene in tqdm(
        target_genes,
        desc="Filtering Peaks",
        ascii=False,
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        chrom, tss = gene_tss[gene]

        # 筛选同染色体
        candidates = peaks_df[peaks_df['chrom'] == chrom]
        if candidates.empty:
            continue

        # 筛选距离
        matched = candidates[
            (candidates['center'] >= tss - window) &
            (candidates['center'] <= tss + window)
        ]
        if matched.empty:
            continue

        keep_names_set.update(matched.index)
        row_idx = gene_to_row.get(gene, None)
        if row_idx is None:
            continue
        # 记录每个匹配峰的 gene-row 与 peak-name（列稍后映射）
        for peak_name in matched.index:
            row_indices.append(row_idx)
            col_names.append(peak_name)

    stats["peaks_retained"] = int(len(keep_names_set))
    print(f"Retained {len(keep_names_set)} peaks out of {adata.shape[1]}")

    if len(keep_names_set) == 0:
        print("\u26a0\ufe0f Warning: Physical filter removed ALL peaks. Falling back to retaining TOP variance peaks to prevent crash.")
        fallback_n = min(n_final, adata.shape[1])
        stats["fallback_used"] = True
        stats["peaks_retained"] = int(fallback_n)
        stats["status"] = "fallback_top_variance"
        fallback_adata = adata[:, :fallback_n].copy()
        fallback_adata.uns["gtf_filter_stats"] = stats
        return fallback_adata, None

    # 按原始顺序保留峰，保证列顺序可追踪
    kept_peaks_in_order = [p for p in adata.var_names if p in keep_names_set]
    filtered_adata = adata[:, kept_peaks_in_order].copy()

    # 将 col_names 转为最终列索引
    peak_to_col = {p: i for i, p in enumerate(kept_peaks_in_order)}
    col_indices = [peak_to_col[p] for p in col_names if p in peak_to_col]
    row_indices = [r for p, r in zip(col_names, row_indices) if p in peak_to_col]

    if len(col_indices) == 0:
        # 极端情况：有保留峰但未记录映射（不应发生），安全返回 None
        print("\u26a0\ufe0f Warning: Gene-Peak mapping is empty after filtering.")
        gene_peak_mask = None
    else:
        data = np.ones(len(col_indices), dtype=np.float32)
        shape = (len(rna_genes), len(kept_peaks_in_order))
        gene_peak_mask = sparse.coo_matrix((data, (row_indices, col_indices)), shape=shape, dtype=np.float32)

    stats["status"] = "ok"
    filtered_adata.uns["gtf_filter_stats"] = stats

    return filtered_adata, gene_peak_mask

def custom_tf_idf(adata, eps=1e-6):
    """
    TF-IDF 变换
    """
    print("Applying Custom TF-IDF transform...")
    eps = float(eps)
    if adata.shape[1] == 0:
        print("Error: Input matrix has 0 features (peaks). Skipping TF-IDF.")
        return adata

    X = adata.X
    # 确保数据为 float32
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)
    else:
        X = np.array(X, dtype=np.float32)
    adata.X = X

    idf = X.shape[0] / (X.sum(axis=0) + eps) # 加个极小值防止除零
    idf = np.array(idf).flatten()

    if sparse.issparse(X):
        sum_per_cell = np.array(X.sum(axis=1)).flatten()
        sum_per_cell[sum_per_cell == 0] = 1
        tf = X.multiply(1.0 / sum_per_cell[:, None])
        adata.X = tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        adata.X = tf * idf

    adata.X = sparse.csr_matrix(adata.X, dtype=np.float32)
    return adata

def process_atac_pipeline(
    adata,
    rna_genes,
    gtf_path,
    n_global=50000,
    n_final=30000,
    window=100000,
    min_cells=50,
    target_sum=1e4,
    tfidf_eps=1e-6
):
    print("--- Processing ATAC Data ---")
    gene_peak_mask = None

    # 1. 基础过滤
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # 2. 全局变异度筛选
    # ATAC 输入是原始计数，使用 cell_ranger 口径可避免 seurat 路径中的 expm1 溢出。
    print(f"Selecting top {n_global} peaks by variance...")
    try:
        sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=n_global)
        adata = adata[:, adata.var['highly_variable']].copy()
    except Exception as e:
        print(f"Warning in HVG selection: {e}. Skipping.")

    # 3. TSS 物理筛选
    if gtf_path and rna_genes is not None:
        adata, gene_peak_mask = filter_peaks_by_tss(adata, gtf_path, rna_genes, window=window, n_final=n_final)

        # 二次筛选
        selected_idx = None
        if adata.shape[1] > n_final:
            print(f"Downsampling to {n_final} peaks...")
            sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=n_final)
            hvg_mask = adata.var['highly_variable'].values
            # 处理 ties 或浮动导致的 > n_final 的情况，强制截断到 n_final
            if hvg_mask.sum() > n_final:
                keep_idx = np.flatnonzero(hvg_mask)[:n_final]
                mask_trim = np.zeros_like(hvg_mask, dtype=bool)
                mask_trim[keep_idx] = True
                hvg_mask = mask_trim

            adata = adata[:, hvg_mask].copy()
            selected_idx = np.nonzero(hvg_mask)[0]
        if "gtf_filter_stats" in adata.uns:
            adata.uns["gtf_filter_stats"]["peaks_after_downsample"] = int(adata.shape[1])
            if gene_peak_mask is not None and selected_idx is not None:
                # 同步裁剪掩码的列（仅当发生二次筛选）
                gene_peak_mask = gene_peak_mask.tocsr()[:, selected_idx]

    # 4. TF-IDF
    adata = custom_tf_idf(adata, eps=tfidf_eps)

    # 5. 使用 Scanpy 进行 Normalize + Log1p
    if adata.shape[1] > 0:
        if sparse.issparse(adata.X):
            adata.X.data[np.isinf(adata.X.data)] = 0.0
            adata.X.data[np.isnan(adata.X.data)] = 0.0
            adata.X = adata.X.tocsr().astype(np.float32)
        else:
            X = np.array(adata.X, dtype=np.float32)
            X[np.isinf(X)] = 0.0
            X[np.isnan(X)] = 0.0
            adata.X = X

        sc.pp.normalize_total(adata, target_sum=float(target_sum))
        sc.pp.log1p(adata)

    print(f"Final ATAC shape: {adata.shape}")
    return adata, gene_peak_mask
