import logging
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import trange

from .trainer import WeightedMSELoss, _off_diagonal


class SFTrainerTri:
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
        self.criterion_adt = WeightedMSELoss(mode=mode, std_eps=eps, std_floor_factor=floor).to(self.device)
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
        self.lambda_rec_rna = float(self.train_cfg.get("lambda_rec_rna", 1.0))
        self.lambda_rec_adt = float(self.train_cfg.get("lambda_rec_adt", 1.0))
        self.lambda_rec_atac = float(self.train_cfg.get("lambda_rec_atac", 1.0))
        if min(self.lambda_rec_rna, self.lambda_rec_adt, self.lambda_rec_atac) < 0:
            raise ValueError("tri-modal reconstruction weights must be non-negative")
        if self.lambda_rec_rna == 0 and self.lambda_rec_adt == 0 and self.lambda_rec_atac == 0:
            raise ValueError("at least one tri-modal reconstruction weight must be positive")
        self.lambda_residual = float(self.train_cfg.get("lambda_residual", 1e-3))
        self.lambda_spec_recon = float(self.train_cfg.get("lambda_spec_recon", 0.0))
        if self.lambda_spec_recon < 0:
            raise ValueError("lambda_spec_recon must be non-negative")
        self.alignment_mode = str(self.train_cfg.get("alignment_mode", "vicreg")).strip().lower()
        if self.alignment_mode not in {"vicreg", "none"}:
            raise ValueError("SFTrainerTri alignment_mode must be vicreg or none")
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

    def _alignment_loss(self, p_rna, p_adt, p_atac):
        zero = p_rna.new_zeros(())
        if self.alignment_mode == "none" or self.lambda_align <= 0:
            return zero, zero, zero, zero, zero, zero, zero
        losses = [
            self.vicreg_loss(p_rna, p_adt),
            self.vicreg_loss(p_rna, p_atac),
            self.vicreg_loss(p_adt, p_atac),
        ]
        total = sum(item[0] for item in losses) / 3.0
        inv = sum(item[1] for item in losses) / 3.0
        var = sum(item[2] for item in losses) / 3.0
        cov = sum(item[3] for item in losses) / 3.0
        return total, inv, var, cov, losses[0][0], losses[1][0], losses[2][0]

    def train_epoch(self, rna_feat, adt_feat, atac_feat, edge_index, u_basis, evals, epoch, edge_weight=None):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        tensors = [rna_feat, adt_feat, atac_feat, edge_index, u_basis]
        rna_feat, adt_feat, atac_feat, edge_index, u_basis = [value.to(self.device) for value in tensors]
        evals = None if evals is None else evals.to(self.device)
        edge_weight = None if edge_weight is None else edge_weight.to(self.device)

        outputs = self.model(rna_feat, adt_feat, atac_feat, edge_index, u_basis, evals, edge_weight=edge_weight)
        (
            z_out,
            p_rna,
            p_adt,
            p_atac,
            rec_rna,
            rec_adt,
            rec_atac,
            scale_gates,
            _,
            z_spec,
            z_detail,
        ) = outputs
        loss_rna = self.criterion_rna(rec_rna, rna_feat)
        loss_adt = self.criterion_adt(rec_adt, adt_feat)
        loss_atac = self.criterion_atac(rec_atac, atac_feat)
        loss_rec_weighted = (
            self.lambda_rec_rna * loss_rna
            + self.lambda_rec_adt * loss_adt
            + self.lambda_rec_atac * loss_atac
        )
        loss_align, loss_inv, loss_var, loss_cov, align_ra, align_rt, align_at = self._alignment_loss(
            p_rna, p_adt, p_atac
        )
        loss_residual = z_detail.pow(2).mean()
        if self.lambda_spec_recon > 0:
            spec_rec_rna = self.model.rna_dec(z_spec)
            spec_rec_adt = self.model.adt_dec(z_spec)
            spec_rec_atac = self.model.atac_dec(z_spec)
            loss_spec_rna = self.criterion_rna(spec_rec_rna, rna_feat)
            loss_spec_adt = self.criterion_adt(spec_rec_adt, adt_feat)
            loss_spec_atac = self.criterion_atac(spec_rec_atac, atac_feat)
            loss_spec_recon = (
                self.lambda_rec_rna * loss_spec_rna
                + self.lambda_rec_adt * loss_spec_adt
                + self.lambda_rec_atac * loss_spec_atac
            )
        else:
            loss_spec_rna = z_out.new_zeros(())
            loss_spec_adt = z_out.new_zeros(())
            loss_spec_atac = z_out.new_zeros(())
            loss_spec_recon = z_out.new_zeros(())
        total = (
            loss_rec_weighted
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
        gate_means = scale_gates.detach().mean(dim=0)
        return {
            "total": float(total.detach()),
            "rec_weighted": float(loss_rec_weighted.detach()),
            "rec_rna": float(loss_rna.detach()),
            "rec_adt": float(loss_adt.detach()),
            "rec_atac": float(loss_atac.detach()),
            "lambda_rec_rna": self.lambda_rec_rna,
            "lambda_rec_adt": self.lambda_rec_adt,
            "lambda_rec_atac": self.lambda_rec_atac,
            "align": float(loss_align.detach()),
            "align_invariance": float(loss_inv.detach()),
            "align_variance": float(loss_var.detach()),
            "align_covariance": float(loss_cov.detach()),
            "align_rna_adt": float(align_ra.detach()),
            "align_rna_atac": float(align_rt.detach()),
            "align_adt_atac": float(align_at.detach()),
            "residual": float(loss_residual.detach()),
            "spec_recon": float(loss_spec_recon.detach()),
            "spec_rec_rna": float(loss_spec_rna.detach()),
            "spec_rec_adt": float(loss_spec_adt.detach()),
            "spec_rec_atac": float(loss_spec_atac.detach()),
            "detail_ratio": float(detail_ratio.detach()),
            "gate_low_rna": float(gate_means[0, 0].cpu()),
            "gate_low_adt": float(gate_means[0, 1].cpu()),
            "gate_low_atac": float(gate_means[0, 2].cpu()),
            "gate_mid_rna": float(gate_means[1, 0].cpu()),
            "gate_mid_adt": float(gate_means[1, 1].cpu()),
            "gate_mid_atac": float(gate_means[1, 2].cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
        }

    def run(self, rna_data, adt_data, atac_data, edge_index, u_basis, evals=None, edge_weight=None):
        self.logger.info(
            "Tri-modal SSR training started: epochs=%d alignment=%s fixed_output=z_out",
            self.epochs,
            self.alignment_mode,
        )
        progress = trange(
            1,
            self.epochs + 1,
            desc="Training SSR-Tri",
            unit="epoch",
            dynamic_ncols=True,
            ascii=False,
            file=sys.stdout,
        )
        for epoch in progress:
            metrics = self.train_epoch(
                rna_data, adt_data, atac_data, edge_index, u_basis, evals, epoch,
                edge_weight=edge_weight,
            )
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            progress.set_postfix(
                total=f'{metrics["total"]:.4f}',
                rec=f'{metrics["rec_weighted"]:.4f}',
                rna=f'{metrics["rec_rna"]:.4f}',
                adt=f'{metrics["rec_adt"]:.4f}',
                atac=f'{metrics["rec_atac"]:.4f}',
                align=f'{metrics["align"]:.4f}',
                detail=f'{metrics["detail_ratio"]:.3f}',
                lr=f"{current_lr:.1e}",
                refresh=False,
            )
            if epoch % self.log_interval == 0:
                self.logger.info(
                    "Epoch %04d | total %.6f | rec_weighted %.6f | rec_rna %.6f | rec_adt %.6f | rec_atac %.6f | "
                    "align %.6f | align_rna_adt %.6f | align_rna_atac %.6f | align_adt_atac %.6f | "
                    "residual %.6f | spec_recon %.6f | spec_rna %.6f | spec_adt %.6f | spec_atac %.6f | "
                    "gate_low %.4f/%.4f/%.4f | gate_mid %.4f/%.4f/%.4f | detail_ratio %.6f | grad_norm %.6f | lr %.2e",
                    epoch, metrics["total"], metrics["rec_weighted"],
                    metrics["rec_rna"], metrics["rec_adt"], metrics["rec_atac"],
                    metrics["align"], metrics["align_rna_adt"],
                    metrics["align_rna_atac"], metrics["align_adt_atac"],
                    metrics["residual"], metrics["spec_recon"],
                    metrics["spec_rec_rna"], metrics["spec_rec_adt"], metrics["spec_rec_atac"],
                    metrics["gate_low_rna"], metrics["gate_low_adt"], metrics["gate_low_atac"],
                    metrics["gate_mid_rna"], metrics["gate_mid_adt"], metrics["gate_mid_atac"],
                    metrics["detail_ratio"], metrics["grad_norm"], current_lr,
                )
        torch.save(self.model.state_dict(), self.final_path)
        self.logger.info("Tri-modal SSR training finished: epoch=%d checkpoint=%s", self.epochs, self.final_path)
