# 模型与分析配置清单

本版本提供 **33 份**按输入样本组织的 SSR 模型配置，包括 Y7_T 肾脏案例。全部固定128维z_out并评估最终权重；各样本的训练和聚类参数见下表。

## 按文件列出

PCA列是评估聚类使用的维数，不是模型原生输出维数。

| 类别 | 配置文件 | 训练末轮 | PCA | K |
|---|---|---:|---:|---:|
| human_lymph | [human_lymph_A1](configs/human_lymph/human_lymph_A1.yaml) | 600 | 20 | 10 |
| human_lymph | [human_lymph_D1](configs/human_lymph/human_lymph_D1.yaml) | 600 | 20 | 11 |
| human_lymph | [human_lymph1](configs/human_lymph/human_lymph1.yaml) | 600 | 20 | 10 |
| human_lymph | [human_lymph2](configs/human_lymph/human_lymph2.yaml) | 600 | 20 | 10 |
| human_lymph | [human_lymph3](configs/human_lymph/human_lymph3.yaml) | 600 | 20 | 10 |
| human_tonsil | [human_tonsil_slice1](configs/human_tonsil/human_tonsil_slice1.yaml) | 1300 | 20 | 6 |
| human_tonsil | [human_tonsil_slice2](configs/human_tonsil/human_tonsil_slice2.yaml) | 1300 | 20 | 6 |
| human_tonsil | [human_tonsil_slice3](configs/human_tonsil/human_tonsil_slice3.yaml) | 1300 | 20 | 6 |
| meta | [Y7_T](configs/meta/Y7_T.yaml) | 1500 | 20 | 14 |
| misar | [e11_0_s1](configs/misar/e11_0_s1.yaml) | 1300 | 25 | 8 |
| misar | [e11_0_s2](configs/misar/e11_0_s2.yaml) | 1300 | 25 | 13 |
| misar | [e13_5_s1](configs/misar/e13_5_s1.yaml) | 1300 | 25 | 12 |
| misar | [e13_5_s2](configs/misar/e13_5_s2.yaml) | 1300 | 25 | 15 |
| misar | [e15_5_s1](configs/misar/e15_5_s1.yaml) | 1300 | 25 | 12 |
| misar | [e15_5_s2](configs/misar/e15_5_s2.yaml) | 1300 | 25 | 15 |
| misar | [e18_5_s1](configs/misar/e18_5_s1.yaml) | 1300 | 25 | 14 |
| misar | [e18_5_s2](configs/misar/e18_5_s2.yaml) | 1300 | 25 | 16 |
| mouse_brain_p22 | [mouse_brain_ATAC](configs/mouse_brain_p22/mouse_brain_ATAC.yaml) | 1800 | 25 | 18 |
| mouse_brain_p22 | [mouse_brain_H3K27ac](configs/mouse_brain_p22/mouse_brain_H3K27ac.yaml) | 1800 | 25 | 16 |
| mouse_brain_p22 | [mouse_brain_H3K27me3](configs/mouse_brain_p22/mouse_brain_H3K27me3.yaml) | 1800 | 25 | 18 |
| mouse_brain_p22 | [mouse_brain_H3K4me3](configs/mouse_brain_p22/mouse_brain_H3K4me3.yaml) | 1800 | 25 | 18 |
| mouse_spleen | [mouse_spleen1](configs/mouse_spleen/mouse_spleen1.yaml) | 1500 | 20 | 5 |
| mouse_spleen | [mouse_spleen2](configs/mouse_spleen/mouse_spleen2.yaml) | 1500 | 20 | 5 |
| mouse_thymus | [mouse_thymus1](configs/mouse_thymus/mouse_thymus1.yaml) | 1500 | 20 | 8 |
| mouse_thymus | [mouse_thymus2](configs/mouse_thymus/mouse_thymus2.yaml) | 1500 | 20 | 8 |
| mouse_thymus | [mouse_thymus3](configs/mouse_thymus/mouse_thymus3.yaml) | 1500 | 20 | 8 |
| mouse_thymus | [mouse_thymus4](configs/mouse_thymus/mouse_thymus4.yaml) | 1500 | 20 | 8 |
| simulation | [simulation1](configs/simulation/simulation1.yaml) | 400 | 20 | 5 |
| simulation | [simulation2](configs/simulation/simulation2.yaml) | 400 | 20 | 5 |
| simulation | [simulation3](configs/simulation/simulation3.yaml) | 400 | 20 | 5 |
| simulation | [simulation4](configs/simulation/simulation4.yaml) | 400 | 20 | 5 |
| simulation | [simulation5](configs/simulation/simulation5.yaml) | 400 | 20 | 5 |
| simulation | [simulation6](configs/simulation/simulation6.yaml) | 600 | 20 | 5 |

## 输入版本与使用范围

1. **Human lymph**：保留 A1/D1 与编号1/2/3两套输入配置。论文三切片比较使用编号版；其前两份来源对应 SpatialGlue A1/D1，但不能视为额外独立样本重复计数。A1/D1读取adata_RNA.h5ad及Ground Truth，编号1/2读取s1/s2_adata_rna.h5ad及final_annot；D1的K=11、编号2的K=10。两套配置不互换，也不默认合并统计。
2. **Simulation**：simulation1–5对应五个双模态场景，simulation6为三模态扩展。
3. **Spleen / thymus**：两份脾脏和四份胸腺配置支持补充材料中的组织比较。
4. **MISAR / P22 / tonsil**：分别保留八切片、四种配对测量和三切片配置，用于对应汇总分析。
5. **Y7_T**：正文使用RNA PCA128、谱重构权重0.01、末轮1500、PCA20和K14；不包含早期RNA512/K11版本。

## 不计入33份模型配置的下游参数文件

- benchmark（6份）：benchmark_config.yaml、seurat_wnn_native_config.yaml、supplementary_metrics_config.yaml、biosfib_marker_genes_config.yaml、biosfib_recovered_domains_config.yaml、biosfib_all_domains_config.yaml。
- enrichment（3份）：misar_summary.yaml、p22_paired_modules.yaml、p22_four_modalities.yaml。
- META（1份）：annotation.yaml。
- 图件对应索引（1份）：analysis/figure_manifest.yaml，不是可执行训练配置。
- 环境描述（2份.yml）：environment_destvi.yml、environment_reference_conversion.yml。

analysis目录中共有10份分析参数YAML、1份图件来源索引，以及2份独立环境描述；它们不计入33份模型配置。
