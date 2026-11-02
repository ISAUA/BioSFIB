# BioSFIB SSR

Standalone source distribution of the BioSFIB spectral-spatial residual (SSR) model. The formal output is one fixed 128-dimensional `z_out` embedding. Supported inputs are RNA+ATAC, RNA+ADT, RNA+metabolomics, and RNA+ADT+ATAC.

## Contents

- `sf_model/model/`: two-modality and three-modality SSR models.
- `sf_model/preprocess/`: RNA, ATAC, ADT, metabolomics, and spatial input processing.
- `sf_model/trainer.py`, `trainer_tri.py`: SSR training objectives and checkpoint saving.
- Training uses the full spatial graph in each optimizer step; there is no mini-batch size parameter.
- `scripts/pipeline/`: preprocessing, training, clustering, plots, and H5AD export.
- `configs/`: 32 retained SSR dataset configurations grouped by dataset family.
- `run_pipeline.py`: unified entry point for one configured dataset and checkpoint.

## Environment

Use the existing `sc_bridge` environment for the first validation. A new environment needs Python, PyTorch (CPU or a CUDA-compatible build), the packages in `requirements.txt`, R, and the R package `mclust`. Install `mclust` with `install.packages("mclust")` in R. Ensure `R_HOME` points to that R installation, especially on Windows. The requirement list is an installation checklist, not a tested environment lockfile.

```powershell
conda activate sc_bridge
Set-Location path\to\BioSFIB
python -m pip install -r requirements.txt
```

## Data and configuration

Place the original datasets under `data/raw/` following each YAML's `data.raw_path` and `data.files`. Data, processed caches, results, and checkpoints are excluded from Git. This directory contains source and configuration only. Paths are relative to this directory; it does not use the parent project's data automatically.

Configuration stems are dataset identifiers: `simulation1` through `simulation6`, `e11_0_s1` through `e18_5_s2`, `human_lymph_A1`, `human_lymph1`, `human_tonsil_slice1`, `mouse_brain_ATAC`, `mouse_spleen1`, etc. Consult `configs/` for the complete list.

The fixed epochs and clustering settings were retained from historical SSR configurations. They are reproduction settings, not evidence of a held-out epoch-selection protocol. Any ARI-based choice of epoch remains exploratory. Ground-truth annotations are used only for evaluation.

## Commands

```powershell
python run_pipeline.py --dataset e11_0_s1 --steps preprocess,train,evaluate
python run_pipeline.py --dataset simulation6 --steps preprocess,train,evaluate
python run_preprocess.py --config configs/misar/e11_0_s1.yaml
python run_train.py --config configs/misar/e11_0_s1.yaml
python run_evaluate.py --config configs/misar/e11_0_s1.yaml
```

Run individual commands from this directory. Missing data/checkpoints and failed child processes raise errors. Checkpoints are loaded strictly; do not silently use weights from a different architecture.

Training saves one `ckpt_final.pth` after the configured last optimizer step. All supplied configurations evaluate that final checkpoint. No intermediate or minimum-training-loss checkpoint is saved. Evaluation reports metrics without ARI pass/fail thresholds and does not automatically select an epoch, embedding branch, or PCA dimension. The H5AD metadata records the final epoch from `train.epochs`; keep the configuration paired with its checkpoint. An interrupted run produces no new final checkpoint; use a fresh output directory for each run to avoid confusing an older checkpoint with a completed run.

## Result H5AD

The RNA reference supplies `.X`, `.obs`, `.var`, original spot identifiers, layers, and spatial coordinates. Embeddings are stored in `.obsm["BioSFIB"]` and `.obsm["X_model"]`; UMAP is in `.obsm["X_umap"]`. Native cluster labels are complete categorical columns `BioSFIB`, `cluster`, and `official_cluster`, assigned by position to avoid barcode-index alignment errors. When ground truth is enabled and available, it is exported as `reference_label`. Evaluation PCA features may additionally be stored in `X_mclust_pca`.

## Source and validation status

The SSR core model and two-modality trainer were extracted from the clean source commit `5967ef2` of the development repository. The three-modality SSR implementation, standard preprocessing fixes, and corrected AnnData exporter were retained from the local mainline snapshot. Experimental architecture dispatch, translation models, audits, sweeps, downstream manuscript figures, datasets, and saved outputs are excluded. This is a curated mainline distribution, not a copy of every current working-tree option.

Assembly checked file content, paths, and local imports statically. Python syntax/import checks, checkpoint compatibility, and end-to-end numerical reproduction have not been executed. Before publishing, run these checks in `sc_bridge`:

```powershell
python -m compileall -q sf_model scripts run_pipeline.py run_preprocess.py run_train.py run_evaluate.py
python run_pipeline.py --help
python run_evaluate.py --help
```

Validate at least one two-modality and the three-modality dataset against existing SSR checkpoints before claiming reproducibility. This repository currently contains source and configuration only; checkpoint compatibility and end-to-end numerical reproduction remain unverified. No open-source license has been assigned yet.
