# cis gene–peak 关联与位点图

`associations.py` 从原始 RNA、ATAC 和 GTF 构造 TSS 邻域候选对，RNA normalize_total(10000)+log1p，ATAC 按指定方式转换；计算双侧 Pearson/Spearman 相关，BH 校正并按显式阈值导出。不是因果调控验证，也不把跨模态预测值当作原始分子证据。SSR 的 cluster/residual/gate 附加表是同一旧入口的中间输出，不代表这些附加图都进入论文。

## 重绘已有位点

包内保留 P22 ATAC 的 Meg3、E15.5-S1 的 Neurod2，以及 E18.5-S1 位点图的原结果表：

```powershell
python -m analysis.gene_peak.locus_plot --regulatory-dir source_data/gene_peak/p22_atac --gene-tss-csv source_data/gene_peak/gene_tss_annotation.csv --target-genes Meg3 --max-links-per-gene 10 --fig-width 22 --output-dir results/analysis/loci/p22_Meg3
python -m analysis.gene_peak.locus_plot --regulatory-dir source_data/gene_peak/misar_e15_5_s1 --gene-tss-csv source_data/gene_peak/gene_tss_annotation.csv --target-genes Neurod2 --region chr11:98270000-98390000 --max-links-per-gene 10 --fig-width 22 --output-dir results/analysis/loci/e15_Neurod2
```

原始选中链接见各目录 target_gene_locus_links.csv，用于核对绘出的十条关联。历史 relaxed 目录不能被默认为 |r|≥0.2；按已有论文记录，主展示采用 FDR<0.05、|r|≥0.05、至多十条。旧 manifest 未完整记载相关性筛选参数，不能据以上绘图命令声称已经完整恢复每次历史计算。最终 Neurod2 排版和红/绿弧的人工调整不由本通用入口逐像素再现。

Meg3 的文献是位点背景，不是逐 peak 的外部验证；Neurod2 的五个文献重叠证据还需补齐公共补表/基因组版本/区间交集源材料，当前不宣称这项已闭环。

## 从原始输入计算：示例，不是覆盖旧结果的命令

```powershell
python -m analysis.gene_peak.export_scores --config configs/misar/e18_5_s1.yaml --output-dir results/analysis/scores/e18_5_s1
python -m analysis.gene_peak.associations --config configs/misar/e18_5_s1.yaml --ssr-score-csv results/analysis/scores/e18_5_s1/ssr_core_spot_scores.csv --tss-window 150000 --min-peak-detection 0.01 --min-gene-detection 0.01 --max-links 200000 --min-abs-pearson 0.2 --fdr-cutoff 0.05 --output-dir results/analysis/gene_peak/e18_5_s1
```

上例是显式的通用参数，不套用到所有论文位点。export_scores 只做固定 checkpoint 推理并核对已有 X_model，不训练、不重新选择输出。基因组版本和 GTF 文件应与峰坐标相同；小鼠历史 GTF 为配置指定的 gencode.vM23，发布包不包含 GTF 本身。

附件 E13.5-S2 Igfbpl1 的既有记录：2183 spots，±150 kb，RNA 10000+log1p、ATAC log1p，双方检测率≥1%，401316 对全体 BH，FDR<0.05、不额外限制 |r|，画十条。通用入口默认最多200000对，**不能直接用默认值代替该协议**；需显式提高上限、核对实际候选对数。其专用冻结表目前未在本包定位完整，列为待补，不启动新计算填补历史空缺。
