# 统一 benchmark 与论文比较图

这里只发布自定义的读取、评估、汇总和绘图，不包含第三方模型训练源码。输入按 `benchmark_config.yaml` 中的 file_glob 放到 `data/benchmark/<数据集>/<方法>/<切片>/`；注意切片名中的下划线和连字符与配置一致，不能从训练 YAML 文件名直接猜测。

输入必须有 `.obsm["X_model"]`、空间坐标及一致的 barcode 顺序。BioSFIB 只提供共同真值/坐标，不把其聚类作为真值。主入口逐方法 PCA（不逐维标准化）再 mclust EEE，seed=42；PCA20/25、K 和主/敏感性角色均由配置显式固定。无真值的 P22/脾脏/胸腺样本不补造外部指标。

```powershell
python -m analysis.benchmark.evaluate_benchmark --config analysis/benchmark/benchmark_config.yaml --dry-run
python -m analysis.benchmark.evaluate_benchmark --config analysis/benchmark/benchmark_config.yaml
python -m analysis.benchmark.evaluate_seurat_wnn_native --config analysis/benchmark/seurat_wnn_native_config.yaml --datasets simulation human_lymph human_tonsil misar mouse_brain_p22
python -m analysis.benchmark.merge_benchmark_metrics
```

前两项对应输入核查和重新计算，二选一按需要执行。发布版将前七种标准方法统一写到 `results/benchmark_results_expanded/`；脾脏/胸腺保持六种可用方法，不添加不存在的 MultiGATE 输入。WNN 原生二维不再 PCA，单独记录 native_sensitivity，不能冒称同维主比较。合并拒绝重复指标键，不静默替换历史结果。

```powershell
python -m analysis.benchmark.plot_benchmark_figures --list
python -m analysis.benchmark.plot_benchmark_figures --figure misar_pca25_boxplot
python -m analysis.benchmark.plot_biosfib_recovered_domains --analysis misar_e18_5_s1
python -m analysis.benchmark.plot_biosfib_all_domains --analysis p22_h3k27ac
```

指标绘图默认读取随附的 `source_data/benchmark/benchmark_metrics_combined.csv`。要画新计算结果，将配置 input_csv 改为 `results/benchmark_results_combined/benchmark_metrics_combined.csv`。空间/UMAP 图还需要评估生成的 prediction H5AD；域图配色依赖先生成地图的 color JSON，不能仅凭指标 CSV 重建。全部域图不把数字簇名解释为真实组织；优选域图保留选择阈值、匹配和审计表，不能代表全体区域。

补充内部/空间指标：

```powershell
python -m analysis.benchmark.evaluate_supplementary_metrics --config analysis/benchmark/supplementary_metrics_config.yaml
```

默认 stability_repeats=0；可选子采样稳定性仅是敏感性分析，不用于选模型/epoch。补充指标图需先有对应 metrics_comparison.csv。外部方法版本、原始获取入口和最终各面板输入仍需补充来源登记；当前配置不是第三方训练的完整复现包。
