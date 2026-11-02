"""Export frozen SSR spot scores required by the gene-peak analysis."""
import argparse
import json
from pathlib import Path
import numpy as np
import scanpy as sc
from sf_model.config import load_config
from sf_model.utils import set_seed, validate_ssr_graph
from analysis.common.ssr_state import (
    resolve_path, resolve_processed_data, resolve_checkpoint, resolve_eval_h5ad,
    torch_load, choose_device, load_ssr_state, get_coords, build_spot_table,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-h5ad", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cluster-key", default="cluster")
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    config = load_config(resolve_path(args.config))
    set_seed(int(config.get("project", {}).get("seed", 42)))
    processed = resolve_processed_data(config)
    checkpoint = resolve_checkpoint(config, args.checkpoint)
    result = resolve_eval_h5ad(config, args.eval_h5ad)
    data = torch_load(processed, map_location="cpu", weights_only=False)
    validate_ssr_graph(data, config["data"]["parameters"]["knn_k"])
    adata = sc.read_h5ad(result)
    state = load_ssr_state(config, checkpoint, data, choose_device(args.device))
    coords = get_coords(data, adata)
    if adata.n_obs != coords.shape[0] or not np.allclose(np.asarray(adata.obsm["spatial"])[:, :2], coords[:, :2]):
        raise ValueError("Saved result and processed spots/coordinates must match")
    if not np.allclose(np.asarray(adata.obsm["X_model"]), state["z_out"].detach().cpu().numpy(), rtol=1e-4, atol=1e-5):
        raise ValueError("Checkpoint output differs from the supplied result embedding")
    table = build_spot_table(state, coords, adata, args.cluster_key, args.top_fraction)
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "ssr_core_spot_scores.csv", index=False)
    manifest = dict(config=str(args.config), checkpoint=str(checkpoint), processed=str(processed), result=str(result), top_fraction=args.top_fraction, interpretation_only=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
