# 肾脏事后因子图谱（当前附件 S14）

该分析读取固定 Y7_T checkpoint、128维结果、原始 RNA/代谢组及 META 四类注释。先按 META README 产生结果与 four_class_spot_annotation_source.csv，再运行：

```powershell
python -m analysis.factors.atlas --config configs/meta/Y7_T.yaml --device cpu
```

输出位置默认 results/analysis/meta/Y7_T/manuscript_factor_atlas_20260813/，包括空间因子、分子关联图、源表和 manifest。

保留原算法：z_spec 的 full-SVD PCA，以及各模态编码对 z_spec 做 ridge residualization 后的 PCA；默认20个候选分量，物理近邻Moran k=6、最低0.05。随后按四类暂定区域的标准化 one-vs-rest 对比做唯一分配及定向，每类显示2个联合因子、每模态各1个因子；分子关联比较各模态1000个高变非恒定特征的前10个正相关。

这是使用冻结模型的事后解释，不是模型原生训练出的 shared/private topics，也不是无监督发现的独立细胞类型。区域注释参与因子的展示分配，不能再用同一注释证明发现这些区域。阈值、展示配额、相关性定义完整保留，不为得到好看的面板自动调参。

此目录只整理当前附件实际采用的 Y7_T，不纳入其他 RCC 切片、旧版门控面板或外部主题模型比较。实际运行和原结果数值回归尚待使用者验证。
