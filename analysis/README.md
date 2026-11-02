# 论文下游分析

本目录整理支撑论文的自定义分析代码，不包含废弃模型、调参入口、第三方模型源码或原始数据。它读取已经固定的 BioSFIB/外部方法结果，不向 BioSFIB 训练回传任何标签或分子分析结果。

| 模块 | 内容 | 入口说明 |
|---|---|---|
| benchmark | 统一 PCA+mclust、原生二维 WNN、指标合并和比较图 | [README](benchmark/README.md) |
| markers | marker、dotplot、空间表达、RNA–ADT 热图 | [README](markers/README.md) |
| gene_peak | cis peak–gene 相关性、位点图及所需 SSR 中间量 | [README](gene_peak/README.md) |
| enrichment | GO-BP/KEGG、P22 比较、MISAR 八切片汇总 | [README](enrichment/README.md) |
| meta | Y7_T 独立参考辅助解释、四类图和邻域 Jaccard | [README](meta/README.md) |
| factors | Y7_T 事后因子图谱 | [README](factors/README.md) |

[图—数据—脚本对应清单](figure_manifest.yaml)是按当前论文记录整理的索引。主图拼接、裁切、部分文字和颜色由作者在 Illustrator 中完成；代码生成底层图表不等于逐像素重建最终版式。最终面板级对应和部分外部来源尚待人工核对，不宣称材料已全部齐备。

## 运行约定

从仓库根目录用模块形式启动，例如：

```powershell
python -m pip install -r analysis/requirements.txt
python -m analysis.markers.markers --config configs/misar/e18_5_s1.yaml
```

相对路径以仓库根目录为基准；benchmark 输入另放在 `data/benchmark/`，具体命名见配置。META 的 DestVI 和 Seurat 转换有独立环境文件，建议与模型环境分开管理。下列说明区分重绘现有表格与重新计算；发布版本尚未完成独立运行验证。

## 已保留与尚缺的复现材料

- `source_data/` 保留既有分析的表格/背景快照，不是新计算结果。数据表字节不变，元数据仅脱敏本机路径；原始和发布校验值见其 manifest。
- 保留富集的实际表达背景、marker 输入、显著表、QC 和数据库版本；约数百 MB 历史 API 响应未纳入 Git，不能用当前在线数据库冒充原版本重放。
- gene–peak 保留已画图的关联表和基因注释；历史关联 manifest 未完整记录筛选参数的情形必须标明，不从最小相关系数反推阈值。
- 外部模型的训练代码、结果 H5AD、公共参考原矩阵、GTF 和权重需依原论文/数据来源另取，本仓库不复制第三方源码或数据。
- 补充材料 Igfbpl1 和 Neurod2 文献区间重叠的专用源数据/完整来源仍需补充核对，见清单中的 pending 项。通用位点分析入口已纳入。
- 尚未运行独立代码包的语法、导入、权重兼容或数值回归检查。静态检查不能代替这些验证。

分析脚本来源见 `source_provenance.json`。本次移植主要调整路径、模块依赖和来源记录；不改动原项目及最佳结果。
