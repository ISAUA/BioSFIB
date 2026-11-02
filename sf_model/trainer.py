import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import trange


class WeightedMSELoss(nn.Module):
    """Variance-normalized MSE for dense PCA/LSI features."""

    def __init__(self, mode="feature_std", std_eps=1e-4, std_floor_factor=0.10):
        super().__init__()
        self.mode = str(mode or "feature_std").strip().lower()
        if self.mode in {"standard", "plain", "l2"}:
            self.mode = "mse"
        if self.mode in {"variance", "variance_normalized", "std"}:
            self.mode = "feature_std"
        if self.mode not in {"mse", "feature_std"}:
            raise ValueError(f"Unsupported reconstruction_loss_mode: {mode}")
        self.std_eps = float(std_eps)
        self.std_floor_factor = float(std_floor_factor)

    def _feature_scale(self, target):
        scale = target.detach().std(dim=0, unbiased=False)
        positive = scale[scale > self.std_eps]
        if positive.numel():
            floor = positive.median() * max(0.0, self.std_floor_factor)
            min_scale = torch.clamp(floor, min=self.std_eps)
        else:
            min_scale = scale.new_tensor(self.std_eps)
        return scale.clamp_min(min_scale).unsqueeze(0)

    def forward(self, pred, target):
        diff = pred - target
        if self.mode == "feature_std":
            diff = diff / self._feature_scale(target)
        return diff.pow(2).mean()


def _off_diagonal(matrix):
    n, m = matrix.shape
    if n != m:
        raise ValueError("covariance matrix must be square")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class SFTrainer:
    def __init__(self, model, config, device=None):
        self.config = config
        self.train_cfg = config["train"]
        requested = device or self.train_cfg.get("device", "cuda")
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.model = model.to(self.device)
        mode = self.train_cfg.get("reconstruction_loss_mode", "feature_std")
        eps = float(self.train_cfg.get("reconstruction_std_eps", 1e-4))
        floor = float(self.train_cfg.get("reconstruction_std_floor_factor", 0.10))
        self.criterion_rna = WeightedMSELoss(mode=mode, std_eps=eps, std_floor_factor=floor).to(self.device)
        self.criterion_atac = WeightedMSELoss(mode=mode, std_eps=eps, std_floor_factor=floor).to(self.device)

        lr_high = float(self.train_cfg.get("learning_rate", 1e-3))
        lr_low = float(self.train_cfg.get("learning_rate_low", 1e-4))
        self.lr_switch = int(self.train_cfg.get("lr_switch_epoch", 1000))
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr_high,
            weight_decay=float(self.train_cfg.get("weight_decay", 1e-4)),
        )
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: 1.0 if epoch < self.lr_switch else lr_low / lr_high,
        )
        self.epochs = int(self.train_cfg.get("epochs", 1500))
        self.gradient_clip = float(self.train_cfg.get("gradient_clip", 5.0))
        self.lambda_align = float(self.train_cfg.get("lambda_align", 0.1))
        self.lambda_residual = float(self.train_cfg.get("lambda_residual", 1e-3))
        self.lambda_spec_recon = float(self.train_cfg.get("lambda_spec_recon", 0.0))
        if self.lambda_spec_recon < 0:
            raise ValueError("lambda_spec_recon must be non-negative")
        self.alignment_mode = str(self.train_cfg.get("alignment_mode", "vicreg")).strip().lower()
        self.save_dir = config["project"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_interval = int(self.train_cfg.get("log_interval", 10))
        self.final_path = os.path.join(self.save_dir, "ckpt_final.pth")
        self.logger = logging.getLogger("SFTrainer")

    @staticmethod
    def vicreg_loss(x, y, covariance_weight=0.04, variance_target=1.0, eps=1e-4):
        invariance = F.mse_loss(x, y)
        std_x = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
        std_y = torch.sqrt(y.var(dim=0, unbiased=False) + eps)
        variance = 0.5 * (
            F.relu(variance_target - std_x).mean()
            + F.relu(variance_target - std_y).mean()
        )
        x_centered = x - x.mean(dim=0)
        y_centered = y - y.mean(dim=0)
        denom = max(x.shape[0] - 1, 1)
        cov_x = x_centered.T.matmul(x_centered) / denom
        cov_y = y_centered.T.matmul(y_centered) / denom
        covariance = (
            _off_diagonal(cov_x).pow(2).sum() / x.shape[1]
            + _off_diagonal(cov_y).pow(2).sum() / y.shape[1]
        )
        total = invariance + variance + float(covariance_weight) * covariance
        return total, invariance, variance, covariance

    def _alignment_loss(self, p_rna, p_atac):
        zero = p_rna.new_zeros(())
        if self.alignment_mode == "none" or self.lambda_align <= 0:
            return zero, zero, zero, zero
        if self.alignment_mode == "vicreg":
            return self.vicreg_loss(p_rna, p_atac)
        if self.alignment_mode == "correspondence":
            loss = self.model.correspondence_loss()
            return loss, loss, zero, zero
        raise ValueError(f"Unsupported alignment_mode: {self.alignment_mode}")

    def train_epoch(self, rna_feat, atac_feat, edge_index, u_basis, evals, epoch, edge_weight=None):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        tensors = [rna_feat, atac_feat, edge_index, u_basis]
        rna_feat, atac_feat, edge_index, u_basis = [value.to(self.device) for value in tensors]
        evals = None if evals is None else evals.to(self.device)
        edge_weight = None if edge_weight is None else edge_weight.to(self.device)

        outputs = self.model(rna_feat, atac_feat, edge_index, u_basis, evals, edge_weight=edge_weight)
        z_out, p_rna, p_atac, rec_rna, rec_atac, _, _, z_spec, z_detail = outputs
        loss_rna = self.criterion_rna(rec_rna, rna_feat)
        loss_atac = self.criterion_atac(rec_atac, atac_feat)
        loss_align, loss_inv, loss_var, loss_cov = self._alignment_loss(p_rna, p_atac)
        loss_residual = z_detail.pow(2).mean()
        if self.lambda_spec_recon > 0:
            spec_rec_rna = self.model.rna_dec(z_spec)
            spec_rec_atac = self.model.atac_dec(z_spec)
            loss_spec_rna = self.criterion_rna(spec_rec_rna, rna_feat)
            loss_spec_atac = self.criterion_atac(spec_rec_atac, atac_feat)
            loss_spec_recon = loss_spec_rna + loss_spec_atac
        else:
            loss_spec_rna = z_out.new_zeros(())
            loss_spec_atac = z_out.new_zeros(())
            loss_spec_recon = z_out.new_zeros(())
        total = (
            loss_rna
            + loss_atac
            + self.lambda_align * loss_align
            + self.lambda_residual * loss_residual
            + self.lambda_spec_recon * loss_spec_recon
        )
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        detail_ratio = (
            torch.linalg.vector_norm(z_detail, dim=-1).mean()
            / torch.linalg.vector_norm(z_spec, dim=-1).mean().clamp_min(1e-8)
        )
        return {
            "total": float(total.detach()),
            "rec_rna": float(loss_rna.detach()),
            "rec_atac": float(loss_atac.detach()),
            "align": float(loss_align.detach()),
            "align_invariance": float(loss_inv.detach()),
            "align_variance": float(loss_var.detach()),
            "align_covariance": float(loss_cov.detach()),
            "residual": float(loss_residual.detach()),
            "spec_recon": float(loss_spec_recon.detach()),
            "spec_rec_rna": float(loss_spec_rna.detach()),
            "spec_rec_atac": float(loss_spec_atac.detach()),
            "detail_ratio": float(detail_ratio.detach()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
        }

    def run(self, rna_data, atac_data, edge_index, u_basis, evals=None, edge_weight=None):
        self.logger.info(
            "SSR training started: epochs=%d alignment=%s fixed_output=z_out",
            self.epochs,
            self.alignment_mode,
        )
        progress = trange(
            1,
            self.epochs + 1,
            desc="Training SSR",
            unit="epoch",
            dynamic_ncols=True,
            ascii=False,
            file=sys.stdout,
        )
        for epoch in progress:
            metrics = self.train_epoch(
                rna_data, atac_data, edge_index, u_basis, evals, epoch,
                edge_weight=edge_weight,
            )
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            progress.set_postfix(
                total=f'{metrics["total"]:.4f}',
                rna=f'{metrics["rec_rna"]:.4f}',
                atac=f'{metrics["rec_atac"]:.4f}',
                align=f'{metrics["align"]:.4f}',
                spec=f'{metrics["spec_recon"]:.4f}',
                detail=f'{metrics["detail_ratio"]:.3f}',
                lr=f"{current_lr:.1e}",
                refresh=False,
            )
            if epoch % self.log_interval == 0:
                self.logger.info(
                    "Epoch %04d | total %.6f | rec_rna %.6f | rec_atac %.6f | "
                    "align %.6f | inv %.6f | var %.6f | cov %.6f | residual %.6f | "
                    "spec_recon %.6f | spec_rna %.6f | spec_atac %.6f | "
                    "detail_ratio %.6f | grad_norm %.6f | lr %.2e",
                    epoch, metrics["total"], metrics["rec_rna"], metrics["rec_atac"],
                    metrics["align"], metrics["align_invariance"],
                    metrics["align_variance"], metrics["align_covariance"],
                    metrics["residual"], metrics["spec_recon"],
                    metrics["spec_rec_rna"], metrics["spec_rec_atac"],
                    metrics["detail_ratio"], metrics["grad_norm"], current_lr,
                )
        torch.save(self.model.state_dict(), self.final_path)
        self.logger.info("SSR training finished: epoch=%d checkpoint=%s", self.epochs, self.final_path)
