
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssr import ModalityAdapter, SpectralSpatialResidual


class BioSFINet(nn.Module):
    """BioSFINet-SSR: physical spectral trunk with bounded spatial residual."""

    def __init__(self, config, atac_dim):
        super().__init__()
        model_cfg = config["model"]
        train_cfg = config.get("train", {})
        rna_dim = int(model_cfg["rna_in_dim"])
        atac_dim = int(atac_dim)
        sfib_dim = int(model_cfg.get("sfib_dim", 128))
        if sfib_dim != 128:
            raise ValueError("BioSFINet-SSR currently fixes sfib_dim=128")

        dropout = float(model_cfg.get("dropout", 0.1))
        self.alignment_mode = str(
            train_cfg.get("alignment_mode", model_cfg.get("alignment_mode", "vicreg"))
        ).strip().lower()
        if self.alignment_mode not in {"vicreg", "correspondence", "none"}:
            raise ValueError("alignment_mode must be vicreg, correspondence, or none")
        self.output_fusion_mode = str(
            model_cfg.get("output_fusion_mode", "ssr")
        ).strip().lower()
        if self.output_fusion_mode != "ssr":
            raise ValueError("This release supports only output_fusion_mode=ssr")

        self.rna_enc = ModalityAdapter(rna_dim, sfib_dim, dropout)
        self.atac_enc = ModalityAdapter(atac_dim, sfib_dim, dropout)

        self.sfib = SpectralSpatialResidual(
            dim=sfib_dim,
            wavelet_t_low=float(model_cfg.get("sfib_wavelet_t_low", 8.0)),
            wavelet_t_high=float(model_cfg.get("sfib_wavelet_t_high", 1.5)),
            mid_scale_init=float(model_cfg.get("sfib_wavelet_mid_scale_init", 0.10)),
            mid_scale_max=float(model_cfg.get("sfib_wavelet_mid_scale_max", 0.25)),
            boundary_slope_init=float(model_cfg.get("sfib_boundary_slope_init", 0.75)),
            modality_gate_mode=model_cfg.get("sfib_modality_gate_mode", "learned"),
            modality_gate_rho=float(model_cfg.get("sfib_modality_gate_rho", 0.5)),
            spatial_residual_enabled=bool(model_cfg.get("sfib_spatial_residual_enabled", True)),
            spatial_residual_source=model_cfg.get("sfib_spatial_residual_source", "spectral"),
        )
        self.rna_dec = nn.Linear(sfib_dim, rna_dim)
        self.atac_dec = nn.Linear(sfib_dim, atac_dim)

        if self.alignment_mode == "vicreg":
            self.rna_align_head = nn.Sequential(
                nn.Linear(sfib_dim, sfib_dim),
                nn.GELU(),
                nn.Linear(sfib_dim, 64),
            )
            self.atac_align_head = nn.Sequential(
                nn.Linear(sfib_dim, sfib_dim),
                nn.GELU(),
                nn.Linear(sfib_dim, 64),
            )
        else:
            self.rna_align_head = nn.Identity()
            self.atac_align_head = nn.Identity()
        self._last_modal_latents = None

    def correspondence_loss(self):
        if self._last_modal_latents is None:
            raise RuntimeError("correspondence_loss requires a preceding forward pass")
        h_rna, h_atac = self._last_modal_latents
        rna_to_atac = self.atac_enc(self.atac_dec(h_rna))
        atac_to_rna = self.rna_enc(self.rna_dec(h_atac))
        return 0.5 * (
            F.mse_loss(rna_to_atac, h_rna)
            + F.mse_loss(atac_to_rna, h_atac)
        )

    def forward(self, x_rna, x_atac, edge_index, u_basis, evals=None, edge_weight=None):
        h_rna = self.rna_enc(x_rna, edge_index)
        h_atac = self.atac_enc(x_atac)
        self._last_modal_latents = (h_rna, h_atac)
        z_ssr, z_spec, z_detail, scale_gates, spatial_gate = self.sfib(
            h_rna, h_atac, edge_index, u_basis, evals, edge_weight=edge_weight
        )
        z_out = z_ssr
        rna_low, atac_low, _, _ = self.sfib.last_scale_embeddings
        if self.alignment_mode == "vicreg":
            p_rna = self.rna_align_head(rna_low)
            p_atac = self.atac_align_head(atac_low)
        else:
            p_rna, p_atac = rna_low, atac_low
        # Reconstruction and evaluation both use the fixed 128D SSR latent.
        rec_rna = self.rna_dec(z_ssr)
        rec_atac = self.atac_dec(z_ssr)
        return (
            z_out,
            p_rna,
            p_atac,
            rec_rna,
            rec_atac,
            scale_gates,
            spatial_gate,
            z_spec,
            z_detail,
        )
