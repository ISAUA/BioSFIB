# Y7_T RNA–代谢组的独立辅助解释

正式模型配置：`configs/meta/Y7_T.yaml`，RNA PCA128、谱重构权重0.01、1500末轮、固定 z_out、PCA20 / Mclust EEE K14。不是早期 RNA512/K11 版本。训练输入仅来自原始 Y7_T 及其标准预处理。

```powershell
python run_pipeline.py --dataset Y7_T --steps preprocess,train,evaluate
```

## Jaccard

评估在启用 compute_jaccard 时，计算固定128维 z_out（mclust PCA之前）和预处理 RNA/代谢组特征各自的欧氏 kNN 集合交并比，k=50、排除自身；逐 spot 与摘要保存到 H5AD，formal_metrics.json 保存摘要。不需要真实标签，不比较人工四类注释；不能把使用不同原特征空间的得分混为同定义。

## 四类注释和分子图

RNA原始文件：data/raw/renal/Y7_T/adata_joint_Y7_T_raw_ST.h5ad；
代谢组：同目录 adata_joint_Y7_T_raw_SM.h5ad。辅助单核参考为 Hu 等 ccRCC snRNA.rds，原文件 MD5与 celltype映射见 annotation.yaml / celltype_to_four_class.csv。参考原下载入口/数据使用条件仍需作者补充登记，不以本地路径代替公共来源。

转换环境与 DestVI 环境单独管理，见两份 environment_*.yml。后者保留 scvi-tools1.4.3及 NumPy VAMP prior兼容处理；不是声称这些版本在新机器已验证。

```powershell
Rscript analysis/meta/convert_reference.R --input data/raw/renal/ccRCC_scRNA_reference/snRNA.rds --output data/raw/renal/ccRCC_scRNA_reference/snRNA_seurat_converted.h5ad
python -m analysis.meta.annotate --config analysis/meta/annotation.yaml --stage prepare-reference
python -m analysis.meta.annotate --config analysis/meta/annotation.yaml --stage run-destvi
python -m analysis.meta.figures
```

run-destvi 会训练独立的参考/解卷积模型，但不会训练或修改 BioSFIB。这些运行由使用者按需要执行。输出四类 marker-guided 注释 H5AD、DestVI比例、簇证据表、来源记录、空间图和 marker热图；额外导出 four_class_spot_annotation_source.csv 供因子图谱读取，避免依赖未报告的 gate面板。

Imm/Endo/Stro/Mal 是暂定的生物学解释；图中的 marker证据包含注释使用的marker，不应称为完全独立验证。DestVI仅提供辅助证据，不能充当病理真值、ARI标签、训练目标、K选择或checkpoint筛选依据。m/z信号未独立鉴定时仍以m/z命名。
