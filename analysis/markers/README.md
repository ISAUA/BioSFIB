# Marker、空间表达与 RNA–ADT 热图

输入：固定结果 H5AD（含原生 cluster/BioSFIB 标签与原始 barcode）、原始 RNA/ADT 和对应数据集 YAML。结果不可重新聚类来迎合 marker 表现。脚本检查 spot 数量、barcode/空间顺序；不同数据的变换与 marker 筛选以代码和配置为准。

```powershell
python -m analysis.markers.markers --config configs/misar/e18_5_s1.yaml --output-dir results/analysis/markers/e18_5_s1
python -m analysis.markers.rna_adt_heatmaps --config configs/human_tonsil/human_tonsil_slice1.yaml --output-dir results/analysis/markers/tonsil1
python -m analysis.markers.domain_report --config configs/misar/e18_5_s1.yaml --truth-key Combined_Clusters --pred-key BioSFIB --output-dir results/analysis/domains/e18_5_s1
```

空间表达需显式给出原始 RNA：

```powershell
python -m analysis.markers.spatial_expression --rna-h5ad data/raw/misar/misar_e18_5-s1/adata_RNA.h5ad --genes Ttn Nrg1 Bcl11b --output-dir results/analysis/expression/e18_5_s1
```

主要输出为完整/筛选 marker CSV、dotplot、空间表达图和热图。默认 normalize_total+log1p 后做统计，并非在 128 维 embedding 上寻找基因。具体 p 值校正、top-n、检测比例和热图标准化见各入口的 --help 与代码。

带组织名称的论文 marker 图还有独立入口：

```powershell
python -m analysis.benchmark.run_biosfib_marker_genes --analysis misar_e18_5_s1
```

该入口读取 benchmark staging 的 H5AD，按固定规则做 Hungarian 匹配，可能排除未匹配簇；这是展示映射，不能称为独立生物学注释或改变原生聚类。完整原生标签/未匹配簇不能在统计报告中隐去。各图的手工 marker 顺序、方向和最终面板选择仍需对照最终图件确认。
