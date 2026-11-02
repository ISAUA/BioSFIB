# BioSFIB SSR

本仓库提供 BioSFIB 的谱域—空间残差模型（SSR）主线代码。模型输出固定的 128 维融合表示 `z_out`，支持 RNA+ATAC、RNA+ADT、RNA+代谢组，以及 RNA+ADT+ATAC 三模态输入。

## 当前包含哪些功能？

当前版本覆盖完整的主线流程：**数据预处理 → SSR 训练 → 聚类与指标评估 → UMAP/空间图 → 标准 H5AD 导出**。

本版本不包含生物学下游分析脚本，包括 marker 基因筛选与 dotplot、GO/KEGG 富集、跨切片功能比较、调控网络、peak–gene 分析及论文专用图。它们仍保留在原开发项目中，尚未整理进入本仓库。翻译模型、实验支线、自动调参和范围评估入口也未纳入。

### 评估并不只计算 ARI

- 有可用真实标签且启用标签评估时，计算 ARI、NMI、AMI、同质性（HOM）、完整性（COM）和 V-measure。
- 计算融合表示或聚类所用特征的轮廓系数（Silhouette）。
- 计算聚类空间分布的 Moran’s I、Geary’s C、CHAOS 和 PAS；具体定义与邻域参数见评估代码。
- 生成按模型聚类着色的 UMAP 和空间分布 PDF；配置允许时，还可展示参考注释空间图。
- 导出融合表示、聚类标签、评估信息及用于后续自定义分析的 H5AD。

没有真实标签时跳过相应标签指标，不补造标签。空间指标计算失败时会记录日志并跳过，不能将缺失值视为有效得分。当前标准入口没有计算模态邻域 Jaccard；H5AD 中保留的相应接口不代表已经生成该指标。

## 目录结构

```text
BioSFIB/
├── configs/                 # 按数据集分类的 32 份 SSR 配置
├── sf_model/
│   ├── model/               # 双模态与三模态 SSR
│   ├── preprocess/          # 各模态数据读取与标准预处理
│   ├── trainer.py           # 双模态训练
│   ├── trainer_tri.py       # 三模态训练
│   ├── result_adata.py      # 导出时加载原始 RNA AnnData
│   └── utils.py             # 空间图、随机种子与工具函数
├── scripts/pipeline/        # 预处理、训练和评估的实际实现
├── run_pipeline.py          # 统一启动入口
├── run_preprocess.py        # 单独运行预处理
├── run_train.py             # 单独运行训练
├── run_evaluate.py          # 单独运行评估
├── requirements.txt         # Python 依赖清单
└── .gitignore
```

训练采用全图计算，每次优化步骤使用完整空间图，没有 mini-batch 大小参数。

## 环境配置

已有项目用户可先使用原来的 `sc_bridge` 环境验证。新环境需要 Python、适合当前 CPU/CUDA 环境的 PyTorch、`requirements.txt` 中的依赖，以及 R 和 R 包 `mclust`。

```powershell
conda activate sc_bridge
Set-Location path\to\BioSFIB
python -m pip install -r requirements.txt
```

在 R 中安装聚类依赖：

```r
install.packages("mclust")
```

尤其在 Windows 下，需要保证 `R_HOME` 指向所用的 R 安装目录。`requirements.txt` 是依赖安装清单，尚不是经过独立环境验证的版本锁定文件。

## 数据与配置

按照各 YAML 的 `data.raw_path` 和 `data.files` 放置原始数据。数据集目录通常位于 `data/raw/`；预处理缓存、结果和权重分别保存在配置指定的位置。所有相对路径均以本仓库根目录为基准，不会自动读取原开发项目中的数据。

| 配置目录 | 数据集数量 |
|---|---:|
| `configs/human_lymph/` | 5 |
| `configs/human_tonsil/` | 3 |
| `configs/misar/` | 8 |
| `configs/mouse_brain_p22/` | 4 |
| `configs/mouse_spleen/` | 2 |
| `configs/mouse_thymus/` | 4 |
| `configs/simulation/` | 6 |

`--dataset` 使用配置文件名（去掉 `.yaml`），例如 `e11_0_s1`、`human_lymph_A1`、`mouse_brain_ATAC`、`simulation6`。完整列表见 `configs/`。

各数据集的训练轮数、预处理、对齐方式、损失权重和聚类参数保留原有 SSR 配置，并非所有数据集使用完全相同的参数。它们属于历史结果的复现设置，不意味着经过独立验证集选择；任何基于 ARI 的 epoch 选择都应视为探索性分析。真实标签仅用于评估，不作为训练目标。

数据、预处理缓存、权重、结果和日志均由 `.gitignore` 排除，本仓库只发布源码、配置和说明。

## 启动命令

在本仓库根目录执行：

```powershell
# 全流程：双模态 MISAR 示例
python run_pipeline.py --dataset e11_0_s1 --steps preprocess,train,evaluate

# 全流程：三模态 simulation6
python run_pipeline.py --dataset simulation6 --steps preprocess,train,evaluate

# 分别执行各阶段
python run_preprocess.py --config configs/misar/e11_0_s1.yaml
python run_train.py --config configs/misar/e11_0_s1.yaml
python run_evaluate.py --config configs/misar/e11_0_s1.yaml
```

训练只在配置规定的末轮优化完成后保存一个 `ckpt_final.pth`，不保存中间 epoch 或训练损失最低的权重。全部随附配置均评估该末轮权重。评估不自动搜索 epoch、embedding 分支或 PCA 维数，也没有 ARI 通过/失败门槛。

权重与配置应配套保存；评估图和元数据中的末轮信息来自 `train.epochs`。训练中断不会生成新的最终权重，建议为不同运行使用独立输出目录，避免把旧权重误认为本次训练结果。缺少数据、缺少权重或子进程失败会报错；评估严格加载模型参数。

## 输出文件与指标位置

以各配置的结果目录为根，主要输出为：

| 文件 | 内容 |
|---|---|
| `checkpoints/ckpt_final.pth` | 训练末轮模型参数 |
| `train.log` | 预处理、训练和评估日志 |
| `figures/spatial_analysis.pdf` | UMAP、聚类空间图及配置允许的参考注释图 |
| `predictions/embedding_joint.h5ad` | 原始 RNA 信息、融合表示、聚类标签和评估信息 |
| `predictions/formal_metrics.json` | checkpoint、epoch、seed、ARI 及可用的 SSR 诊断摘要 |

完整标签指标保存在 H5AD 的 `.uns["BioSFIB_evaluation"]["ground_truth_metrics"]`，空间指标和轮廓系数保存在 `.uns["BioSFIB_evaluation"]["spatial_clustering"]`；`formal_metrics.json` 并不包含全部指标。使用 `--output-suffix` 时，PDF、H5AD 和 JSON 文件名会追加相应后缀。

### H5AD 保存结构

- `.X`、`.obs`、`.var`、原始 spot ID 和 `layers` 来自原始 RNA AnnData；`.obsm["spatial"]` 保留其空间坐标。
- `.obsm["BioSFIB"]` 与 `.obsm["X_model"]` 保存固定的 128 维融合表示。
- `.obsm["X_umap"]` 保存 UMAP 坐标。
- `.obs["BioSFIB"]`、`.obs["cluster"]` 与 `.obs["official_cluster"]` 保存相同的原生聚类标签，类型为 `category`；按位置写入并检查缺失，避免 barcode 索引错位。
- 启用真实标签评估且有有效标签时，额外保存 `.obs["reference_label"]`。
- 使用 mclust 时，额外保存 `.obsm["X_mclust_pca"]`；该字段是评估 PCA 特征，不替代原生融合表示。
- `.uns["BioSFIB_metadata"]` 保存结果格式、checkpoint、epoch、seed 和聚类设置等信息。

## 代码来源与验证状态

双模态 SSR 核心和训练器基于开发仓库的干净源码提交 `5967ef2` 提取；三模态 SSR、标准预处理修复和正确的 AnnData 导出逻辑来自本地主线版本。本仓库是精简后的 SSR 发布版本，未复制整个开发工作区。

目前已完成文件内容、路径、本地模块引用和 Git 提交的静态检查。尚未执行此独立版本的 Python 语法/导入检查、权重兼容检查和端到端数值复现，不能据此宣称已经完成运行验证。

可在 `sc_bridge` 环境中先执行：

```powershell
python -m compileall -q sf_model scripts run_pipeline.py run_preprocess.py run_train.py run_evaluate.py
python run_pipeline.py --help
python run_evaluate.py --help
```

随后至少验证一个双模态数据集和三模态 simulation6，并与对应的原 SSR 结果核对。仓库目前尚未指定开源许可证。
