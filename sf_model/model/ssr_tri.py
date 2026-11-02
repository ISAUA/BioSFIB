import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssr import WeightedSpatialResidual, _inverse_softplus, _logit


class TriModalSoftmaxGate(nn.Module):
    """Three-way modality gate initialized to uniform weights."""

    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(dim) * 6, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rna, adt, atac):
        features = torch.cat(
            [
                rna,
                adt,
                atac,
                torch.abs(rna - adt),
                torch.abs(rna - atac),
                torch.abs(adt - atac),
            ],
            dim=-1,
        )
        return torch.softmax(self.net(features), dim=-1)


class SpectralSpatialResidualTri(nn.Module):
    """Three-modality SSR trunk for RNA, ADT, and ATAC."""

    def __init__(
        self,
        dim=128,
        wavelet_t_low=8.0,
        wavelet_t_high=1.5,
        mid_scale_init=0.10,
        mid_scale_max=0.25,
        boundary_slope_init=0.75,
        modality_gate_mode="learned",
        modality_gate_rho=0.5,
        spatial_residual_enabled=True,
        spatial_residual_source="spectral",
    ):
        super().__init__()
        self.dim = int(dim)
        self.wavelet_t_low = float(wavelet_t_low)
        self.wavelet_t_high = float(wavelet_t_high)
        self.mid_scale_max = float(mid_scale_max)
        self.modality_gate_mode = str(modality_gate_mode).strip().lower()
        self.modality_gate_rho = float(modality_gate_rho)
        self.spatial_residual_enabled = bool(spatial_residual_enabled)
        self.spatial_residual_source = str(spatial_residual_source).strip().lower()
        self.architecture = "ssr_tri"
        self.output_mode = "ssr"
        if self.modality_gate_mode == "uniform_0_5":
            self.modality_gate_mode = "uniform_1_3"
        if self.modality_gate_mode not in {"learned", "restricted_learned", "uniform_1_3", "uniform"}:
            raise ValueError(
                "tri-modal modality_gate_mode must be learned, restricted_learned, or uniform_1_3"
            )
        if not 0.0 <= self.modality_gate_rho <= 1.0:
            raise ValueError("modality_gate_rho must be within [0, 1]")
        if self.spatial_residual_source not in {"spectral", "encoded_local_contrast"}:
            raise ValueError(
                "spatial_residual_source must be spectral or encoded_local_contrast"
            )

        self.low_norm_rna = nn.LayerNorm(self.dim)
        self.low_norm_adt = nn.LayerNorm(self.dim)
        self.low_norm_atac = nn.LayerNorm(self.dim)
        self.mid_norm_rna = nn.LayerNorm(self.dim)
        self.mid_norm_adt = nn.LayerNorm(self.dim)
        self.mid_norm_atac = nn.LayerNorm(self.dim)
        self.low_gate = TriModalSoftmaxGate(self.dim)
        self.mid_gate = TriModalSoftmaxGate(self.dim)
        scale_fraction = float(mid_scale_init) / max(self.mid_scale_max, 1e-8)
        self.mid_scale_logit = nn.Parameter(torch.tensor(_logit(scale_fraction), dtype=torch.float32))
        self.boundary_bias = nn.Parameter(torch.tensor(0.0))
        self.boundary_log_slope = nn.Parameter(
            torch.tensor(_inverse_softplus(boundary_slope_init), dtype=torch.float32)
        )
        self.spatial = WeightedSpatialResidual(dim=self.dim, num_heads=4, head_dim=32)
        self.spectral_norm = nn.LayerNorm(self.dim)
        self.output_norm = nn.LayerNorm(self.dim)
        self.last_scale_embeddings = None
        self.last_boundary = None
        self.last_spatial_source = None

    @staticmethod
    def _normalized_evals(evals, n_freq, device, dtype):
        if evals is None:
            return torch.linspace(0.0, 1.0, n_freq, device=device, dtype=dtype)
        values = evals.to(device=device, dtype=dtype).flatten()
        if values.numel() < n_freq:
            raise ValueError(f"evals has {values.numel()} values but u_basis has {n_freq} columns")
        values = values[:n_freq] - values[:n_freq].min()
        return values / values.max().clamp_min(1e-8)

    @staticmethod
    def neighbor_mean(x, edge_index, edge_weight):
        if edge_index is None or edge_index.numel() == 0:
            return x
        src, dst = edge_index[0].long(), edge_index[1].long()
        weight = edge_weight.to(device=x.device, dtype=x.dtype).flatten()
        out = torch.zeros_like(x)
        denom = x.new_zeros((x.shape[0], 1))
        out.index_add_(0, dst, x[src] * weight.unsqueeze(-1))
        denom.index_add_(0, dst, weight.unsqueeze(-1))
        return out / denom.clamp_min(1e-8)

    @staticmethod
    def _filter(x, u_basis, kernel):
        coefficients = u_basis.transpose(0, 1).matmul(x)
        return u_basis.matmul(coefficients * kernel.unsqueeze(-1))

    def _gate(self, rna, adt, atac, gate_module):
        learned = gate_module(rna, adt, atac)
        uniform = torch.full_like(learned, 1.0 / 3.0)
        if self.modality_gate_mode in {"uniform_1_3", "uniform"}:
            gate = uniform
        elif self.modality_gate_mode == "restricted_learned":
            gate = uniform + self.modality_gate_rho * (learned - uniform)
            gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            gate = learned
        fused = (
            gate[:, 0:1] * rna
            + gate[:, 1:2] * adt
            + gate[:, 2:3] * atac
        )
        return fused, gate

    def forward(self, h_rna, h_adt, h_atac, edge_index, u_basis, evals=None, edge_weight=None):
        if u_basis is None:
            raise ValueError("Tri-modal SSR requires a physical-graph GFT basis")
        u_basis = u_basis.to(device=h_rna.device, dtype=h_rna.dtype)
        edge_weight = (
            h_rna.new_ones(edge_index.shape[1])
            if edge_weight is None
            else edge_weight.to(device=h_rna.device, dtype=h_rna.dtype)
        )
        lam = self._normalized_evals(evals, u_basis.shape[1], h_rna.device, h_rna.dtype)
        kernel_low = torch.exp(-self.wavelet_t_low * lam)
        kernel_mid = torch.exp(-self.wavelet_t_high * lam) - kernel_low

        rna_low = self.low_norm_rna(self._filter(h_rna, u_basis, kernel_low))
        adt_low = self.low_norm_adt(self._filter(h_adt, u_basis, kernel_low))
        atac_low = self.low_norm_atac(self._filter(h_atac, u_basis, kernel_low))
        rna_mid = self.mid_norm_rna(self._filter(h_rna, u_basis, kernel_mid))
        adt_mid = self.mid_norm_adt(self._filter(h_adt, u_basis, kernel_mid))
        atac_mid = self.mid_norm_atac(self._filter(h_atac, u_basis, kernel_mid))

        z_low, gate_low = self._gate(rna_low, adt_low, atac_low, self.low_gate)
        z_mid, gate_mid = self._gate(rna_mid, adt_mid, atac_mid, self.mid_gate)

        low_neighbor = self.neighbor_mean(z_low, edge_index, edge_weight)
        boundary = torch.linalg.vector_norm(z_low - low_neighbor, dim=-1, keepdim=True) / math.sqrt(self.dim)
        boundary_z = (
            (boundary - boundary.mean().detach())
            / boundary.std(unbiased=False).detach().clamp_min(1e-6)
        ).detach()
        q_mid = torch.sigmoid(self.boundary_bias + F.softplus(self.boundary_log_slope) * boundary_z)
        alpha_mid = self.mid_scale_max * torch.sigmoid(self.mid_scale_logit)
        z_spec = self.spectral_norm(z_low + alpha_mid * q_mid * z_mid)

        if self.spatial_residual_source == "encoded_local_contrast":
            encoded_joint = (
                gate_low[:, 0:1] * h_rna
                + gate_low[:, 1:2] * h_adt
                + gate_low[:, 2:3] * h_atac
            )
            encoded_neighbor = self.neighbor_mean(encoded_joint, edge_index, edge_weight)
            spatial_source = encoded_joint - encoded_neighbor
        else:
            spatial_source = z_spec
        spatial_neighbor = self.neighbor_mean(spatial_source, edge_index, edge_weight)
        z_detail, spatial_gate = self.spatial(
            spatial_source,
            edge_index,
            edge_weight,
            spatial_neighbor,
            boundary_z,
            enabled=self.spatial_residual_enabled,
        )
        z_out = self.output_norm(z_spec + z_detail)
        self.last_scale_embeddings = (rna_low, adt_low, atac_low, rna_mid, adt_mid, atac_mid)
        self.last_boundary = boundary.detach()
        self.last_spatial_source = spatial_source.detach()
        scale_gates = torch.stack([gate_low, gate_mid], dim=1)
        return z_out, z_spec, z_detail, scale_gates, spatial_gate

    @property
    def out_norm(self):
        return self.output_norm

    def detail_scale(self):
        return self.mid_scale_logit.new_tensor(1.0)
