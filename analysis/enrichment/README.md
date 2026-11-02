# GO-BP / KEGG 与跨切片汇总

输入：原始 RNA、已固定簇的 marker 表，或 source_data 中的完成结果。g:Profiler 使用实际表达基因背景，而非全部基因组；默认背景检测率≥1%，marker padj<0.01、logFC>0、每簇10–50基因，term size10–500，BH FDR<0.05。原 manifest 中的 parameters、database_versions 才是各次分析的来源记录。

## 不访问 API 的既有结果汇总

```powershell
python -m analysis.enrichment.misar_summary --config analysis/enrichment/misar_summary.yaml --output-dir results/analysis/enrichment/misar
python -m analysis.enrichment.p22_comparison --atac-results source_data/enrichment/mouse_brain_ATAC/enrichment_significant_results.csv --h3k27ac-results source_data/enrichment/mouse_brain_H3K27ac/enrichment_significant_results.csv --output-dir results/analysis/enrichment/p22_paired
python -m analysis.enrichment.p22_comparison --atac-results source_data/enrichment/mouse_brain_ATAC/enrichment_significant_results.csv --h3k27ac-results source_data/enrichment/mouse_brain_H3K27ac/enrichment_significant_results.csv --h3k4me3-results source_data/enrichment/mouse_brain_H3K4me3/enrichment_significant_results.csv --h3k27me3-results source_data/enrichment/mouse_brain_H3K27me3/enrichment_significant_results.csv --output-dir results/analysis/enrichment/p22_four
```

输出功能点阵、显著 term 汇总、跨切片重叠/一致性表及来源记录。MISAR 代表性 term 是看过结果后的报告选择，配置明确标为 post_result_reporting_selection；完整 term Jaccard 不仅取展示的子集。P22 的两测量和四测量模块映射分别保留在 p22_paired_modules.yaml、p22_four_modalities.yaml，不互相替代。

## 新查询示例

```powershell
python -m analysis.enrichment.enrich --config configs/mouse_brain_p22/mouse_brain_ATAC.yaml --dataset-key mouse_brain_ATAC --organism mmusculus --marker-csv source_data/enrichment/mouse_brain_ATAC/input_markers.csv --background-gene-file source_data/enrichment/mouse_brain_ATAC/expressed_rna_background.txt --module-map analysis/enrichment/p22_four_modalities.yaml --output-dir results/analysis/enrichment/new_p22_atac
```

该命令需要原始 RNA且访问外部 API，是新查询而非保证原数据库版本复现。准确重放旧查询还需原 api_cache，并添加 --offline-from-cache --cache-dir <缓存目录>。约399 MB历史响应未打包，当前也没有已验证的公开归档链接；保留的显著结果与版本信息可以重画汇总，不能替代完整查询缓存。不要把新数据库版本结果覆盖到 source_data。
