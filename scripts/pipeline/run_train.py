import os
import argparse
import logging
import torch
from sf_model.config import load_config as load_config_with_base
from sf_model.model.bio_sfinet_tri import BioSFINetTri
from sf_model.trainer_tri import SFTrainerTri
from sf_model.model.bio_sfinet import BioSFINet
from sf_model.trainer import SFTrainer
from sf_model.utils import configure_utf8_console, set_seed, validate_ssr_graph

def load_config(config_path="configs/misar/e11_0_s1.yaml"):
    return load_config_with_base(config_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bio-SFINet")
    parser.add_argument("--config", default="configs/misar/e11_0_s1.yaml", help="Path to YAML config file")
    return parser.parse_args()


def resolve_train_log_path(save_dir):
    save_dir = save_dir.rstrip('/\\')
    if os.path.basename(save_dir) == 'checkpoints':
        return os.path.join(os.path.dirname(save_dir), "train.log")
    return os.path.join(save_dir, "train.log")


def setup_logger(log_path):
    logger = logging.getLogger("SFTrainer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def append_log_separator(log_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 88 + "\n")


def torch_load_compat(path, map_location, weights_only):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)

def main():
    configure_utf8_console()
    args = parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    set_seed(config['project'].get('seed', 42))
    save_dir = config['project']['save_dir']
    log_path = resolve_train_log_path(save_dir)
    logger = setup_logger(log_path)
    logger.info("[Phase 2] Starting Model Training...")

    processed_dir = config['data']['processed_path']
    data_path = os.path.join(processed_dir, "processed_data.pt")

    # 2. 加载预处理好的数据
    if not os.path.exists(data_path):
        logger.error("Data file not found at %s", data_path)
        logger.error("Please run 'python run_preprocess.py' first.")
        raise FileNotFoundError(data_path)

    logger.info("Loading processed data...")
    # 使用 cpu 加载，trainer 会自动搬运到 cuda
    data_dict = torch_load_compat(data_path, map_location='cpu', weights_only=False)
    expected_knn_k = int(config['data']['parameters']['knn_k'])
    validate_ssr_graph(data_dict, expected_knn_k=expected_knn_k)

    if config.get("tri_modal", {}).get("enabled", False):
        config["model"]["rna_in_dim"] = int(data_dict["rna_feat"].shape[1])
        model = BioSFINetTri(config, adt_dim=data_dict["adt_feat"].shape[1], atac_dim=data_dict["atac_feat"].shape[1])
        trainer = SFTrainerTri(model, config)
        trainer.run(data_dict["rna_feat"], data_dict["adt_feat"], data_dict["atac_feat"], data_dict["edge_index"], data_dict["u_basis"], data_dict.get("evals"), edge_weight=data_dict.get("edge_weight"))
        return

    rna_feat = data_dict["rna_feat"]
    atac_feat = data_dict["atac_feat"]
    coords = data_dict["coords"]
    edge_index = data_dict["edge_index"]
    edge_weight = data_dict.get("edge_weight", None)
    u_basis = data_dict["u_basis"]
    evals = data_dict.get("evals", None)
    rna_dim = int(data_dict.get("rna_dim", rna_feat.shape[1]))
    atac_dim = data_dict["atac_dim"]

    logger.info("Data ready: RNA=%s, ATAC=%s, edges=%d", rna_feat.shape, atac_feat.shape, edge_index.shape[1])

    # 3. 初始化模型
    logger.info("Initializing model...")
    # 以预处理后的实际维度覆盖配置，避免 PCA 维度变更导致不匹配
    config['model']['rna_in_dim'] = rna_dim
    model = BioSFINet(config, atac_dim=atac_dim)
    os.makedirs(config['project']['save_dir'], exist_ok=True)

    # 4. 初始化训练器
    trainer = SFTrainer(model, config)

    # 5. 开始训练
    logger.info("Training started.")
    trainer.run(rna_feat, atac_feat, edge_index, u_basis, evals, edge_weight=edge_weight)
    logger.info("Training run complete.")
    if os.environ.get("SF_PIPELINE_RUN") != "1":
        append_log_separator(log_path)

if __name__ == "__main__":
    main()
