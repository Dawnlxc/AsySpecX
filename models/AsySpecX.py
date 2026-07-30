"""AsySpecX forecaster with FITS-style phase-1 ablation controls.

Pipeline:
    1. RIN normalize input
    2. rfft(time) and low-pass cut to ``cut_freq`` bins
    3. Per-channel spectral lift:
       - ``complex_mlp``: old ComplexLinear -> ModReLU -> ComplexLinear path
       - ``fits_linear``: single channel-independent ComplexLinear
    4. Optional asymmetric low-rank cross-channel residual
    5. Zero-pad to the output spectrum, irfft, scale by length ratio
    6. De-RIN, return horizon forecast by default

The phase-1 flags are intentionally small additions to the original model:
``spectral_lift``, ``cross_mode``, ``gate_type``, ``gate_init_logit``,
``mask_self_transfer``, ``residual_clip_eta``, ``backcast_loss_weight``,
``force_cross_off`` and ``skip_dc_cross``.
"""

import math
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _config(configs, name, default=None):
    return getattr(configs, name, default)


class ComplexLinear(nn.Module):
    """Complex Linear via two real parameter blocks."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_re = nn.Parameter(torch.zeros(out_features))
            self.bias_im = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_re", None)
            self.register_parameter("bias_im", None)
        nn.init.xavier_uniform_(self.weight_re)
        nn.init.xavier_uniform_(self.weight_im)
        with torch.no_grad():
            scale = 1.0 / math.sqrt(2.0)
            self.weight_re.mul_(scale)
            self.weight_im.mul_(scale)

    def forward(self, x):
        x_r, x_i = x.real, x.imag
        y_r = F.linear(x_r, self.weight_re, self.bias_re) - F.linear(x_i, self.weight_im, None)
        y_i = F.linear(x_r, self.weight_im, self.bias_im) + F.linear(x_i, self.weight_re, None)
        return torch.complex(y_r, y_i)


class ChannelLowRankComplexLinear(nn.Module):
    """Shared complex lift plus a low-rank channel-conditioned weight delta.

    For channel ``c``, ``W_c = W_shared + sum_r coeff[c,r] * DeltaW[r]``.
    The delta weights start at exactly zero while channel coefficients start
    nonzero, preserving the shared-lift output at initialization without
    suppressing the first gradient into ``DeltaW``.
    """

    def __init__(self, in_features, out_features, channels, rank=2, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.channels = int(channels)
        requested_rank = int(rank)
        if requested_rank < 1:
            raise ValueError(f"lowrank_channel lift rank must be positive, got {requested_rank}")
        self.rank = min(requested_rank, self.channels)
        self.base = ComplexLinear(self.in_features, self.out_features, bias=bias)
        self.delta_weight_re = nn.Parameter(torch.zeros(
            self.rank, self.out_features, self.in_features
        ))
        self.delta_weight_im = nn.Parameter(torch.zeros(
            self.rank, self.out_features, self.in_features
        ))
        self.channel_coeff = nn.Parameter(torch.empty(self.channels, self.rank))
        nn.init.normal_(self.channel_coeff, std=1.0 / math.sqrt(self.rank))

    def forward(self, x):
        # x: [B, C, F_in] complex -> [B, C, F_out] complex
        if x.dim() != 3 or x.size(1) != self.channels or x.size(2) != self.in_features:
            raise ValueError(
                f"lowrank_channel lift expects [B,{self.channels},{self.in_features}], "
                f"got {tuple(x.shape)}"
            )
        # Explicitly synthesize one effective weight per channel before the
        # batch contraction.  This avoids materializing [B,C,R,F_out] and
        # keeps training cost close to one vectorized individual lift.
        delta_weight = torch.complex(self.delta_weight_re, self.delta_weight_im)
        coeff = self.channel_coeff.to(delta_weight.dtype)
        effective_weight = torch.complex(
            self.base.weight_re, self.base.weight_im
        ).unsqueeze(0) + torch.einsum("cr,roi->coi", coeff, delta_weight)
        y = torch.einsum("bci,coi->bco", x, effective_weight)
        if self.base.bias_re is not None:
            bias = torch.complex(self.base.bias_re, self.base.bias_im)
            y = y + bias.view(1, 1, self.out_features)
        return y


class ModReLUBins(nn.Module):
    """Per-frequency-bin phase-preserving complex activation."""

    def __init__(self, num_freqs, init_bias=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.full((int(num_freqs),), float(init_bias)))

    def forward(self, z):
        mag = z.abs()
        bias = self.bias.view(*([1] * (z.dim() - 1)), -1)
        scale = torch.relu(mag + bias) / (mag + 1e-6)
        return z * scale.to(z.dtype)


class ComplexMLP(nn.Module):
    """Old two-layer complex MLP spectral lift."""

    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.fc1 = ComplexLinear(in_features, hidden_features, bias=True)
        self.act = ModReLUBins(hidden_features, init_bias=0.0)
        self.fc2 = ComplexLinear(hidden_features, out_features, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _GateController(nn.Module):
    """Gate parameterization shared by low-rank and self-gain adapters."""

    def __init__(self, channels, num_bands, gate_type, gate_init_logit, gate_max, split=False):
        super().__init__()
        self.channels = int(channels)
        self.num_bands = int(num_bands)
        self.gate_type = str(gate_type)
        self.gate_max = float(gate_max)
        self.split = bool(split)

        if self.gate_type == "global":
            if self.split:
                self.gate_logit_diag = nn.Parameter(torch.tensor(float(gate_init_logit)))
                self.gate_logit_cross = nn.Parameter(torch.tensor(float(gate_init_logit)))
            else:
                self.gate_logit = nn.Parameter(torch.tensor(float(gate_init_logit)))
        elif self.gate_type == "channel_band":
            shape = (self.channels, self.num_bands)
            if self.split:
                self.gate_logits_diag = nn.Parameter(torch.full(shape, float(gate_init_logit)))
                self.gate_logits_cross = nn.Parameter(torch.full(shape, float(gate_init_logit)))
            else:
                self.gate_logits = nn.Parameter(torch.full(shape, float(gate_init_logit)))
        elif self.gate_type == "hier_channel_band":
            shape = (self.channels, self.num_bands)
            if self.split:
                self.global_gate_logit_diag = nn.Parameter(torch.tensor(float(gate_init_logit)))
                self.global_gate_logit_cross = nn.Parameter(torch.tensor(float(gate_init_logit)))
                self.local_gate_logits_diag = nn.Parameter(torch.zeros(shape))
                self.local_gate_logits_cross = nn.Parameter(torch.zeros(shape))
            else:
                self.global_gate_logit = nn.Parameter(torch.tensor(float(gate_init_logit)))
                self.local_gate_logits = nn.Parameter(torch.zeros(shape))
        else:
            raise ValueError(f"Unsupported gate_type={self.gate_type!r}")

    def _suffix(self, component):
        if not self.split:
            return ""
        if component not in {"diag", "cross"}:
            raise ValueError("split gates require component='diag' or 'cross'")
        return f"_{component}"

    def values(self, component=None):
        suffix = self._suffix(component)
        if self.gate_type == "global":
            logit = getattr(self, f"gate_logit{suffix}")
            return self.gate_max * torch.sigmoid(logit)
        if self.gate_type == "channel_band":
            logits = getattr(self, f"gate_logits{suffix}")
            return self.gate_max * torch.sigmoid(logits)
        global_logit = getattr(self, f"global_gate_logit{suffix}")
        local_logits = getattr(self, f"local_gate_logits{suffix}")
        global_gate = self.gate_max * torch.sigmoid(global_logit)
        local_scale = 2.0 * torch.sigmoid(local_logits)
        return torch.clamp(global_gate * local_scale, max=self.gate_max)

    def local_scale(self, component=None):
        if self.gate_type != "hier_channel_band":
            return None
        suffix = self._suffix(component)
        return 2.0 * torch.sigmoid(getattr(self, f"local_gate_logits{suffix}"))

    def global_gate(self, component=None):
        if self.gate_type != "hier_channel_band":
            return None
        suffix = self._suffix(component)
        return self.gate_max * torch.sigmoid(getattr(self, f"global_gate_logit{suffix}"))

    def for_band(self, band_idx, dtype, component=None):
        gate = self.values(component)
        if gate.dim() == 2:
            return gate[:, band_idx].view(1, self.channels, 1).to(dtype)
        return gate.to(dtype)

    def diagnostics(self):
        with torch.no_grad():
            out = {}
            if self.split:
                for component in ("diag", "cross"):
                    vals = self.values(component).detach().float().cpu()
                    out[f"gate_{component}_mean"] = float(vals.mean())
                    out[f"gate_{component}_min"] = float(vals.min())
                    out[f"gate_{component}_max"] = float(vals.max())
                    if self.gate_type == "hier_channel_band":
                        global_gate = self.global_gate(component).detach().float().cpu()
                        local = self.local_scale(component).detach().float().cpu()
                        out[f"global_gate_{component}"] = float(global_gate)
                        out[f"local_scale_{component}_mean"] = float(local.mean())
                        out[f"local_scale_{component}_min"] = float(local.min())
                        out[f"local_scale_{component}_max"] = float(local.max())
                return out

            vals = self.values().detach().float().cpu()
            out["gate_mean"] = float(vals.mean())
            out["gate_min"] = float(vals.min())
            out["gate_max"] = float(vals.max())
            if self.gate_type == "hier_channel_band":
                global_gate = self.global_gate().detach().float().cpu()
                local = self.local_scale().detach().float().cpu()
                out["global_gate"] = float(global_gate)
                out["local_scale_mean"] = float(local.mean())
                out["local_scale_min"] = float(local.min())
                out["local_scale_max"] = float(local.max())
            return out


class AsymCross(nn.Module):
    """Asymmetric low-rank residual with Phase-2 decomposition controls."""

    VALID_PARTS = {"all", "diag_only", "offdiag_only", "split"}

    def __init__(
        self,
        channels,
        num_freqs,
        rank,
        num_bands,
        gate_init_logit,
        gate_max,
        gate_type="global",
        mask_self_transfer=False,
        residual_clip_eta=None,
        skip_dc_cross=True,
        residual_part=None,
        energy_control="none",
        learned_clip_scope="component_channel_band",
        learned_clip_eta_init=1.0,
        learned_clip_eta_max=2.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_freqs = int(num_freqs)
        self.rank = max(1, min(int(rank), self.channels))
        self.num_bands = max(1, min(int(num_bands), self.num_freqs))
        self.gate_type = str(gate_type)
        self.gate_max = float(gate_max)
        self.mask_self_transfer = _as_bool(mask_self_transfer)
        self.skip_dc_cross = _as_bool(skip_dc_cross)
        self.residual_part = residual_part or ("offdiag_only" if self.mask_self_transfer else "all")
        if self.residual_part not in self.VALID_PARTS:
            raise ValueError(f"Unsupported residual_part={self.residual_part!r}")
        self.residual_clip_eta = residual_clip_eta
        if self.residual_clip_eta is not None and float(self.residual_clip_eta) <= 0.0:
            self.residual_clip_eta = None

        # Phase 7: learned energy-controlled clip. eta is learned but the applied
        # scale is still clamped to max 1.0 (see clip_residual*), so it can only
        # SHRINK the residual -- never amplify.
        self.energy_control = str(energy_control)
        self.learned_clip_scope = str(learned_clip_scope)
        self.learned_clip_eta_max = float(learned_clip_eta_max)
        if self.energy_control not in {"none", "learned_clip"}:
            raise ValueError(f"Unsupported energy_control={self.energy_control!r}")
        if self.energy_control == "learned_clip":
            if self.learned_clip_scope != "component_channel_band":
                raise ValueError(f"Unsupported learned_clip_scope={self.learned_clip_scope!r}")
            ratio = min(max(float(learned_clip_eta_init) / max(self.learned_clip_eta_max, 1e-6), 1e-4), 1 - 1e-4)
            init_logit = math.log(ratio / (1.0 - ratio))
            # [component(diag=0, cross=1), band, channel]
            self.clip_logit = nn.Parameter(torch.full((2, self.num_bands, self.channels), float(init_logit)))
        else:
            self.register_parameter("clip_logit", None)

        self.A = nn.Parameter(torch.empty(self.channels, self.rank))
        self.B = nn.Parameter(torch.empty(self.channels, self.rank))
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.B)

        self.g_re = nn.Parameter(torch.empty(self.num_bands, self.rank))
        self.g_im = nn.Parameter(torch.empty(self.num_bands, self.rank))
        nn.init.normal_(self.g_re, std=0.02)
        nn.init.normal_(self.g_im, std=0.02)

        band_ids = torch.div(
            torch.arange(self.num_freqs) * self.num_bands,
            self.num_freqs,
            rounding_mode="floor",
        ).clamp(0, self.num_bands - 1)
        self.register_buffer("band_ids", band_ids.long())
        self.gates = _GateController(
            self.channels,
            self.num_bands,
            self.gate_type,
            gate_init_logit,
            self.gate_max,
            split=self.residual_part == "split",
        )
        self._last_diagnostics: Dict[str, float] = {}

    @staticmethod
    def decompose_lowrank_residual(U_band, A, B_src, g_band):
        dtype = U_band.dtype
        A_c = A.to(dtype)
        B_c = B_src.to(dtype)
        g_c = g_band.to(dtype)
        S = torch.einsum("cr,bcf->brf", B_c, U_band)
        S = g_c.view(1, -1, 1) * S
        R_all = torch.einsum("cr,brf->bcf", A_c, S)
        h_diag = torch.sum(A_c * B_c * g_c.view(1, -1), dim=1)
        R_diag = h_diag.view(1, -1, 1) * U_band
        R_offdiag = R_all - R_diag
        return R_all, R_diag, R_offdiag

    @staticmethod
    def lowrank_residual(U_band, A, B_src, g_band, mask_self_transfer=False):
        R_all, _R_diag, R_offdiag = AsymCross.decompose_lowrank_residual(U_band, A, B_src, g_band)
        return R_offdiag if mask_self_transfer else R_all

    @staticmethod
    def _rms_ratio(R, U_band, eps=1e-8):
        u_rms = torch.sqrt(torch.mean(torch.abs(U_band) ** 2, dim=-1) + eps)
        r_rms = torch.sqrt(torch.mean(torch.abs(R) ** 2, dim=-1) + eps)
        return r_rms / (u_rms + eps)

    @staticmethod
    def clip_residual(R, U_band, eta: Optional[float], eps=1e-8):
        if eta is None or float(eta) <= 0.0:
            return R
        u_rms = torch.sqrt(torch.mean(torch.abs(U_band) ** 2, dim=-1, keepdim=True) + eps)
        r_rms = torch.sqrt(torch.mean(torch.abs(R) ** 2, dim=-1, keepdim=True) + eps)
        scale = torch.clamp(float(eta) * u_rms / (r_rms + eps), max=1.0)
        return R * scale.to(R.dtype)

    @staticmethod
    def clip_residual_learned(R, U_band, eta_vec, eps=1e-8):
        """Clip with a learned per-channel eta tensor (shape [1, C, 1]).

        The applied scale is clamped to max 1.0, so a large learned eta can only
        leave the residual unchanged -- it can never amplify it. Returns (R_out,
        binding_fraction) where binding_fraction is how often the clip is active.
        """
        u_rms = torch.sqrt(torch.mean(torch.abs(U_band) ** 2, dim=-1, keepdim=True) + eps)
        r_rms = torch.sqrt(torch.mean(torch.abs(R) ** 2, dim=-1, keepdim=True) + eps)
        raw = eta_vec * u_rms / (r_rms + eps)
        scale = torch.clamp(raw, max=1.0)
        with torch.no_grad():
            binding = float((raw > 1.0).float().mean().cpu())
        return R * scale.to(R.dtype), binding

    def _learned_eta(self, component_idx, band_idx, dtype=None):
        # component_idx: 0=diag, 1=cross. Returns a REAL eta broadcastable to
        # [1, C, 1] (kept real so the downstream clamp is valid; dtype ignored).
        logit = self.clip_logit[component_idx, band_idx, :]
        eta = self.learned_clip_eta_max * torch.sigmoid(logit)
        return eta.view(1, self.channels, 1)

    def gate_values(self, component=None):
        return self.gates.values(component)

    def _effective_part(self, eval_residual_part):
        if eval_residual_part in {None, "default"}:
            return self.residual_part
        if eval_residual_part == "none":
            return "none"
        if eval_residual_part not in {"all", "diag_only", "offdiag_only"}:
            raise ValueError(f"Unsupported eval_residual_part={eval_residual_part!r}")
        return eval_residual_part

    def forward(self, U, force_off=False, eval_residual_part="default"):
        part = self._effective_part(eval_residual_part)
        if force_off or part == "none":
            self._last_diagnostics = {"cross_active": 0.0}
            return U

        out = U.clone()
        g_all = torch.complex(self.g_re, self.g_im)
        learned = self.energy_control == "learned_clip"
        clip_binding = []
        stats = {
            "residual_all_ratio": [],
            "residual_diag_ratio": [],
            "residual_offdiag_ratio": [],
            "gated_residual_ratio": [],
            "residual_selected_ratio_preclip": [],
            "residual_selected_ratio_postclip": [],
            "diag_fraction": [],
            "offdiag_fraction": [],
        }

        for band_idx in range(self.num_bands):
            idx = torch.nonzero(self.band_ids == band_idx, as_tuple=False).flatten()
            if self.skip_dc_cross:
                idx = idx[idx != 0]
            if idx.numel() == 0:
                continue

            U_band = U.index_select(dim=2, index=idx)
            R_all, R_diag, R_offdiag = self.decompose_lowrank_residual(U_band, self.A, self.B, g_all[band_idx])

            if part == "split":
                if learned:
                    R_diag_used, b0 = self.clip_residual_learned(R_diag, U_band, self._learned_eta(0, band_idx, U.dtype))
                    R_offdiag_used, b1 = self.clip_residual_learned(R_offdiag, U_band, self._learned_eta(1, band_idx, U.dtype))
                    clip_binding += [b0, b1]
                else:
                    R_diag_used = self.clip_residual(R_diag, U_band, self.residual_clip_eta)
                    R_offdiag_used = self.clip_residual(R_offdiag, U_band, self.residual_clip_eta)
                gate_diag = self.gates.for_band(band_idx, U.dtype, component="diag")
                gate_cross = self.gates.for_band(band_idx, U.dtype, component="cross")
                delta = gate_diag * R_diag_used + gate_cross * R_offdiag_used
                selected_pre = R_diag + R_offdiag
                selected_post = R_diag_used + R_offdiag_used
            else:
                if part == "all":
                    R = R_all
                elif part == "diag_only":
                    R = R_diag
                else:
                    R = R_offdiag
                selected_pre = R
                if learned:
                    # non-split parts use the diag(0) eta slot as the single control
                    R, b0 = self.clip_residual_learned(R, U_band, self._learned_eta(0, band_idx, U.dtype))
                    clip_binding.append(b0)
                else:
                    R = self.clip_residual(R, U_band, self.residual_clip_eta)
                selected_post = R
                gate = self.gates.for_band(band_idx, U.dtype)
                delta = gate * R

            out[:, :, idx] = U_band + delta

            with torch.no_grad():
                all_ratio = self._rms_ratio(R_all, U_band)
                diag_ratio = self._rms_ratio(R_diag, U_band)
                off_ratio = self._rms_ratio(R_offdiag, U_band)
                gated_ratio = self._rms_ratio(delta, U_band)
                pre_ratio = self._rms_ratio(selected_pre, U_band)
                post_ratio = self._rms_ratio(selected_post, U_band)
                stats["residual_all_ratio"].append(all_ratio)
                stats["residual_diag_ratio"].append(diag_ratio)
                stats["residual_offdiag_ratio"].append(off_ratio)
                stats["gated_residual_ratio"].append(gated_ratio)
                stats["residual_selected_ratio_preclip"].append(pre_ratio)
                stats["residual_selected_ratio_postclip"].append(post_ratio)
                all_rms = torch.sqrt(torch.mean(torch.abs(R_all) ** 2, dim=-1) + 1e-8)
                diag_rms = torch.sqrt(torch.mean(torch.abs(R_diag) ** 2, dim=-1) + 1e-8)
                off_rms = torch.sqrt(torch.mean(torch.abs(R_offdiag) ** 2, dim=-1) + 1e-8)
                stats["diag_fraction"].append(diag_rms / (all_rms + 1e-8))
                stats["offdiag_fraction"].append(off_rms / (all_rms + 1e-8))

        diagnostics = {"cross_active": 1.0, "residual_part": part}
        diagnostics.update(self.gates.diagnostics())
        if learned and self.clip_logit is not None:
            with torch.no_grad():
                eta = (self.learned_clip_eta_max * torch.sigmoid(self.clip_logit)).detach().float().cpu()
                diagnostics["eta_mean"] = float(eta.mean())
                diagnostics["eta_min"] = float(eta.min())
                diagnostics["eta_max"] = float(eta.max())
                if clip_binding:
                    diagnostics["clip_active_fraction"] = float(sum(clip_binding) / len(clip_binding))
        for name, tensors in stats.items():
            if not tensors:
                continue
            flat = torch.cat([t.reshape(-1).detach().float().cpu() for t in tensors])
            diagnostics[f"{name}_mean"] = float(flat.mean())
            diagnostics[f"{name}_max"] = float(flat.max())
        self._last_diagnostics = diagnostics
        return out

    def get_diagnostics(self):
        return dict(self._last_diagnostics)


class SelfBandGain(nn.Module):
    """Per-channel per-band complex self spectral adapter baseline."""

    def __init__(
        self,
        channels,
        num_freqs,
        num_bands,
        gate_init_logit,
        gate_max,
        gate_type="global",
        residual_clip_eta=None,
        skip_dc_cross=True,
        self_gain_init_std=1e-3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_freqs = int(num_freqs)
        self.num_bands = max(1, min(int(num_bands), self.num_freqs))
        self.gate_type = str(gate_type)
        self.skip_dc_cross = _as_bool(skip_dc_cross)
        self.residual_clip_eta = residual_clip_eta
        if self.residual_clip_eta is not None and float(self.residual_clip_eta) <= 0.0:
            self.residual_clip_eta = None
        self.self_gain_re = nn.Parameter(torch.empty(self.num_bands, self.channels))
        self.self_gain_im = nn.Parameter(torch.empty(self.num_bands, self.channels))
        nn.init.normal_(self.self_gain_re, std=float(self_gain_init_std))
        nn.init.normal_(self.self_gain_im, std=float(self_gain_init_std))
        band_ids = torch.div(
            torch.arange(self.num_freqs) * self.num_bands,
            self.num_freqs,
            rounding_mode="floor",
        ).clamp(0, self.num_bands - 1)
        self.register_buffer("band_ids", band_ids.long())
        self.gates = _GateController(
            self.channels,
            self.num_bands,
            self.gate_type,
            gate_init_logit,
            gate_max,
            split=False,
        )
        self._last_diagnostics: Dict[str, float] = {}

    def gate_values(self, component=None):
        return self.gates.values(component)

    def forward(self, U, force_off=False, eval_residual_part="default"):
        if force_off or eval_residual_part == "none":
            self._last_diagnostics = {"cross_active": 0.0}
            return U
        out = U.clone()
        gain = torch.complex(self.self_gain_re, self.self_gain_im).to(U.dtype)
        ratios = []
        gated_ratios = []
        for band_idx in range(self.num_bands):
            idx = torch.nonzero(self.band_ids == band_idx, as_tuple=False).flatten()
            if self.skip_dc_cross:
                idx = idx[idx != 0]
            if idx.numel() == 0:
                continue
            U_band = U.index_select(dim=2, index=idx)
            R = gain[band_idx].view(1, self.channels, 1) * U_band
            R = AsymCross.clip_residual(R, U_band, self.residual_clip_eta)
            gate = self.gates.for_band(band_idx, U.dtype)
            delta = gate * R
            out[:, :, idx] = U_band + delta
            with torch.no_grad():
                ratios.append(AsymCross._rms_ratio(R, U_band))
                gated_ratios.append(AsymCross._rms_ratio(delta, U_band))
        diagnostics = {"cross_active": 1.0, "residual_part": "self_band_gain"}
        diagnostics.update(self.gates.diagnostics())
        if ratios:
            flat = torch.cat([t.reshape(-1).detach().float().cpu() for t in ratios])
            gated = torch.cat([t.reshape(-1).detach().float().cpu() for t in gated_ratios])
            diagnostics["residual_all_ratio_mean"] = float(flat.mean())
            diagnostics["residual_all_ratio_max"] = float(flat.max())
            diagnostics["gated_residual_ratio_mean"] = float(gated.mean())
            diagnostics["gated_residual_ratio_max"] = float(gated.max())
        self._last_diagnostics = diagnostics
        return out

    def get_diagnostics(self):
        return dict(self._last_diagnostics)


def parse_periods(raw, fallback_period=24):
    """Parse a period list. Accepts ``,`` or ``+`` separators (``+`` avoids the
    sbatch --export comma-truncation trap). Empty -> [fallback_period]."""
    if raw is None:
        raw = ""
    text = str(raw).replace("+", ",")
    items = [p.strip() for p in text.split(",") if p.strip() != ""]
    if items:
        return [max(1, int(float(x))) for x in items]
    return [max(1, int(fallback_period))]


class SparsePeriodAdapter(nn.Module):
    """Masked same-phase temporal linear adapter (multi-period capable).

    For each period ``p`` a same-phase mask ``mask_p[h, t] = 1 if (t % p) ==
    ((T + h) % p)`` selects the lookback positions that share the forecast
    phase. Each period owns a dense weight ``W_raw[p]`` that is masked so only
    same-phase taps contribute. Period predictions are fused (``sum_gated`` or
    ``softmax``) then blended with the spectral backbone via a temporal gate.

    Single-period configs (``--period`` only, no ``--periods``) reproduce the
    Phase-3 behavior exactly: the period gate is skipped (weight == 1) so the
    only blend is the temporal fusion gate.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        periods=(24,),
        periodic_init="seasonal_naive",
        periodic_sharing="shared",
        temporal_fusion="convex",
        temporal_gate_type="global",
        temporal_gate_init_logit=-4.0,
        period_fusion="sum_gated",
        period_gate_type="period",
        period_gate_init_logit=0.0,
        periodic_l1_weight=0.0,
        periodic_l2_weight=0.0,
        temporal_gate_l1_weight=0.0,
        period=None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        if period is not None:  # backward-compat single-period constructor
            periods = [int(period)]
        periods = [max(1, int(p)) for p in periods]
        if not periods:
            periods = [24]
        self.periods = periods
        self.num_periods = len(periods)
        self.period = periods[0]  # backward-compat scalar
        self.periodic_init = str(periodic_init)
        self.periodic_sharing = str(periodic_sharing)
        self.temporal_fusion = str(temporal_fusion)
        self.temporal_gate_type = str(temporal_gate_type)
        self.period_fusion = str(period_fusion)
        self.period_gate_type = str(period_gate_type)
        self.periodic_l1_weight = float(periodic_l1_weight)
        self.periodic_l2_weight = float(periodic_l2_weight)
        self.temporal_gate_l1_weight = float(temporal_gate_l1_weight)
        if self.periodic_sharing != "shared":
            raise NotImplementedError("periodic_sharing=individual_channel is not implemented")
        if self.temporal_fusion not in {"convex", "additive"}:
            raise ValueError(f"Unsupported temporal_fusion={self.temporal_fusion!r}")
        if self.temporal_gate_type not in {"global", "channel", "horizon", "horizon_channel"}:
            raise ValueError(f"Unsupported temporal_gate_type={self.temporal_gate_type!r}")
        if self.period_fusion not in {"sum_gated", "softmax"}:
            raise ValueError(f"Unsupported period_fusion={self.period_fusion!r}")
        if self.period_gate_type not in {
            "global", "period", "period_horizon", "period_channel", "period_horizon_channel",
        }:
            raise ValueError(f"Unsupported period_gate_type={self.period_gate_type!r}")

        mask = torch.stack(
            [self._build_mask(self.seq_len, self.pred_len, p) for p in self.periods], dim=0
        )  # [P, H, T]
        self.register_buffer("periodic_mask", mask)
        self.W_raw = nn.Parameter(torch.zeros(self.num_periods, self.pred_len, self.seq_len))
        self._init_weights()

        # Period fusion gate only exists for genuine multi-period adapters; a
        # single period keeps weight == 1 (Phase-3 compatible, softmax(P=1)==1).
        self.use_period_gate = self.num_periods > 1
        if self.use_period_gate:
            self.period_gate_logits = nn.Parameter(self._init_period_gate_logits(period_gate_init_logit))

        self.temporal_gate_logit = nn.Parameter(self._init_temporal_gate_logits(temporal_gate_init_logit))
        self._last_diagnostics: Dict[str, float] = {}

    @staticmethod
    def _build_mask(seq_len, pred_len, period):
        t = torch.arange(seq_len).view(1, seq_len)
        h = torch.arange(pred_len).view(pred_len, 1)
        target_phase = (seq_len + h) % period
        return (t % period == target_phase).float()

    def _init_period_gate_logits(self, init):
        P, H, C = self.num_periods, self.pred_len, self.channels
        t = self.period_gate_type
        if t == "global":
            shape = ()
        elif t == "period":
            shape = (P,)
        elif t == "period_horizon":
            shape = (P, H)
        elif t == "period_channel":
            shape = (P, C)
        else:  # period_horizon_channel
            shape = (P, H, C)
        return torch.full(shape, float(init))

    def _init_temporal_gate_logits(self, init):
        H, C = self.pred_len, self.channels
        t = self.temporal_gate_type
        if t == "global":
            shape = ()
        elif t == "channel":
            shape = (C,)
        elif t == "horizon":
            shape = (H,)
        else:  # horizon_channel
            shape = (H, C)
        return torch.full(shape, float(init))

    def _nearest_same_phase(self, period, h):
        """Largest (nearest to forecast origin) lookback index sharing phase.

        Returns None when no same-phase index exists in the lookback window.
        Fallback to t=T-1 is unsafe here: T-1 may fall outside the same-phase
        mask, which would zero the initialized weight. So the caller leaves the
        row all-zero instead of seeding an out-of-mask tap.
        """
        target_phase = (self.seq_len + h) % period
        candidates = [t for t in range(self.seq_len) if t % period == target_phase]
        if candidates:
            return candidates[-1]
        return None

    def _init_weights(self):
        with torch.no_grad():
            self.W_raw.zero_()
            if self.periodic_init == "seasonal_naive":
                for p_idx, period in enumerate(self.periods):
                    for h in range(self.pred_len):
                        t = self._nearest_same_phase(period, h)
                        if t is not None:
                            self.W_raw[p_idx, h, t] = 1.0
            elif self.periodic_init == "zeros":
                pass
            elif self.periodic_init == "small_random":
                self.W_raw.normal_(std=1e-3)
                self.W_raw.mul_(self.periodic_mask)  # keep only in-mask taps
            else:
                raise ValueError(f"Unsupported periodic_init={self.periodic_init!r}")

    def alpha(self):
        a = torch.sigmoid(self.temporal_gate_logit)
        t = self.temporal_gate_type
        if t == "global":
            return a.view(1, 1, 1)
        if t == "channel":
            return a.view(1, 1, self.channels)
        if t == "horizon":
            return a.view(1, self.pred_len, 1)
        return a.view(1, self.pred_len, self.channels)

    def period_weight(self):
        """Return per-period fusion weights broadcast to [P, H, C]."""
        P, H, C = self.num_periods, self.pred_len, self.channels
        if not self.use_period_gate:
            return torch.ones(P, H, C, device=self.W_raw.device, dtype=self.W_raw.dtype)
        logits = self.period_gate_logits
        t = self.period_gate_type
        if t == "global":
            logits = logits.view(1, 1, 1).expand(P, H, C)
        elif t == "period":
            logits = logits.view(P, 1, 1).expand(P, H, C)
        elif t == "period_horizon":
            logits = logits.view(P, H, 1).expand(P, H, C)
        elif t == "period_channel":
            logits = logits.view(P, 1, C).expand(P, H, C)
        else:  # period_horizon_channel
            logits = logits.view(P, H, C)
        if self.period_fusion == "softmax":
            return torch.softmax(logits, dim=0)
        return torch.sigmoid(logits)

    def extra_loss(self):
        """Periodic-adapter regularization on in-mask weights only.

        Returns a scalar tensor (0 when both weights are disabled). Mask-out
        positions are excluded from both the penalty and the mean denominator.
        """
        mask = self.periodic_mask.to(self.W_raw.dtype)
        W_eff = self.W_raw * mask
        loss = W_eff.sum() * 0.0  # scalar tensor on the right device/dtype
        if (self.periodic_l1_weight <= 0.0 and self.periodic_l2_weight <= 0.0
                and self.temporal_gate_l1_weight <= 0.0):
            return loss
        denom = mask.sum().clamp_min(1.0)
        if self.periodic_l1_weight > 0.0:
            loss = loss + self.periodic_l1_weight * (W_eff.abs().sum() / denom)
        if self.periodic_l2_weight > 0.0:
            loss = loss + self.periodic_l2_weight * ((W_eff ** 2).sum() / denom)
        if self.temporal_gate_l1_weight > 0.0:
            # Encourage the temporal fusion gate to stay closed (weather guard).
            loss = loss + self.temporal_gate_l1_weight * torch.sigmoid(self.temporal_gate_logit).mean()
        return loss

    def forward(self, x_norm, pred_spec_norm):
        W_eff = self.W_raw * self.periodic_mask.to(self.W_raw.dtype)  # [P, H, T]
        pred_periods = torch.einsum("pht,btc->bphc", W_eff, x_norm)  # [B, P, H, C]
        weight = self.period_weight().to(pred_periods.dtype)  # [P, H, C]
        pred_period_norm = (pred_periods * weight.unsqueeze(0)).sum(dim=1)  # [B, H, C]

        alpha = self.alpha().to(pred_spec_norm.dtype)
        if self.temporal_fusion == "convex":
            pred_fused_norm = pred_spec_norm + alpha * (pred_period_norm - pred_spec_norm)
        else:
            pred_fused_norm = pred_spec_norm + alpha * pred_period_norm

        with torch.no_grad():
            alpha_det = alpha.detach().float().cpu()
            weight_det = weight.detach().float().cpu()
            self._last_diagnostics = {
                "temporal_adapter_enabled": 1.0,
                "num_periods": float(self.num_periods),
                "period": float(self.periods[0]),
                "periods": "+".join(str(p) for p in self.periods),
                "temporal_gate_mean": float(alpha_det.mean()),
                "temporal_gate_min": float(alpha_det.min()),
                "temporal_gate_max": float(alpha_det.max()),
                "temporal_gate_l1_value": float(self.temporal_gate_l1_weight * alpha_det.mean()),
                "period_weight_mean": float(weight_det.mean()),
                "period_weight_min": float(weight_det.min()),
                "period_weight_max": float(weight_det.max()),
                "periodic_weight_abs_mean": float(W_eff.detach().abs().mean().cpu()),
                "periodic_weight_abs_max": float(W_eff.detach().abs().max().cpu()),
                "periodic_mask_density": float(self.periodic_mask.detach().mean().cpu()),
                "pred_period_norm_rms": float(torch.sqrt(torch.mean(pred_period_norm.detach() ** 2) + 1e-8).cpu()),
                "pred_spec_norm_rms": float(torch.sqrt(torch.mean(pred_spec_norm.detach() ** 2) + 1e-8).cpu()),
                "fused_delta_rms": float(torch.sqrt(torch.mean((pred_fused_norm.detach() - pred_spec_norm.detach()) ** 2) + 1e-8).cpu()),
            }
        return pred_fused_norm, pred_period_norm

    def get_diagnostics(self):
        return dict(self._last_diagnostics)


class _StridedPhaseLinear(torch.autograd.Function):
    """Same-phase contraction that saves the strided input view, not a copy.

    PyTorch's generic einsum backward may retain an internal contiguous
    ``[B, T, C]``-sized packing of a strided phase view.  The explicit backward
    below saves only the original view (shared storage) and the compact weight;
    any packing needed by a contraction is therefore short-lived.
    """

    @staticmethod
    def forward(ctx, weight, x_phase):
        ctx.save_for_backward(weight, x_phase)
        return torch.einsum("glk,bgkc->bglc", weight, x_phase)

    @staticmethod
    def backward(ctx, grad_output):
        weight, x_phase = ctx.saved_tensors
        compute_dtype = torch.promote_types(weight.dtype, x_phase.dtype)
        grad_compute = grad_output.to(compute_dtype)
        grad_weight = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_weight = torch.einsum(
                "bglc,bgkc->glk", grad_compute, x_phase.to(compute_dtype)
            ).to(weight.dtype)
        if ctx.needs_input_grad[1]:
            grad_x = torch.einsum(
                "glk,bglc->bgkc", weight.to(compute_dtype), grad_compute
            ).to(x_phase.dtype)
        return grad_weight, grad_x


class _CompactSinglePeriod(nn.Module):
    """Ragged same-phase linear maps for one period.

    A dense masked period map stores ``[H, T]`` values even though only pairs
    with matching phases can contribute.  This block stores one legal matrix
    ``[L_phase, K_phase]`` per active phase instead.  The ragged representation
    matters when ``T`` or ``H`` is not divisible by the period: padding a
    rectangular ``[period, L, K]`` tensor would re-introduce invalid parameters.
    """

    def __init__(self, seq_len, pred_len, period):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period = max(1, int(period))
        self.weights = nn.ParameterList()
        self._input_index_names = []
        self._horizon_index_names = []
        self._phase_starts = []

        input_position = torch.arange(self.seq_len)
        horizon = torch.arange(self.pred_len)
        horizon_phase = (self.seq_len + horizon) % self.period
        runs = []
        current_run = []
        current_shape = None
        for phase in range(self.period):
            input_index = input_position[input_position % self.period == phase]
            horizon_index = horizon[horizon_phase == phase]
            # With no input tap the corresponding dense-mask rows are exactly
            # zero, so there is neither a legal parameter nor work to perform.
            if input_index.numel() == 0 or horizon_index.numel() == 0:
                if current_run:
                    runs.append(current_run)
                    current_run = []
                    current_shape = None
                continue
            shape = (horizon_index.numel(), input_index.numel())
            if current_run and shape != current_shape:
                runs.append(current_run)
                current_run = []
            current_run.append((phase, input_index, horizon_index))
            current_shape = shape
        if current_run:
            runs.append(current_run)

        # Non-divisible sequence/horizon lengths make phase groups ragged, but
        # each consecutive run still has a rectangular [G, L_phase, K_phase]
        # legal weight tensor.  Keeping phases consecutive lets forward expose
        # the interleaved input as an as_strided view instead of gather-copying
        # [B, T, C] once per period.
        for run in runs:
            phase_start = run[0][0]
            horizon_count = run[0][2].numel()
            input_count = run[0][1].numel()
            group = len(self.weights)
            self.weights.append(nn.Parameter(torch.zeros(
                len(run), horizon_count, input_count
            )))
            input_name = f"input_index_{group}"
            horizon_name = f"horizon_index_{group}"
            self.register_buffer(input_name, torch.stack([item[1] for item in run]))
            self.register_buffer(horizon_name, torch.stack([item[2] for item in run]))
            self._input_index_names.append(input_name)
            self._horizon_index_names.append(horizon_name)
            self._phase_starts.append(phase_start)

    def groups(self):
        for weight, input_name, horizon_name, phase_start in zip(
            self.weights,
            self._input_index_names,
            self._horizon_index_names,
            self._phase_starts,
        ):
            yield weight, getattr(self, input_name), getattr(self, horizon_name), phase_start

    def forward(self, x_norm):
        if not x_norm.is_contiguous():
            raise ValueError("_CompactSinglePeriod expects one contiguous [B, T, C] input")
        # Each temporary is [B, G, L_phase, C].  In particular, this never
        # forms the dense masked-product intermediate [B, H, T, C].
        predictions = []
        horizon_indices = []
        for weight, _, horizon_index, phase_start in self.groups():
            group_count, _, input_count = weight.shape
            x_phase = x_norm.as_strided(
                size=(x_norm.size(0), group_count, input_count, x_norm.size(2)),
                stride=(
                    x_norm.stride(0),
                    x_norm.stride(1),
                    self.period * x_norm.stride(1),
                    x_norm.stride(2),
                ),
                storage_offset=x_norm.storage_offset() + phase_start * x_norm.stride(1),
            )  # zero-copy [B, G, K_phase, C] phase view
            if torch.is_grad_enabled() and (weight.requires_grad or x_phase.requires_grad):
                pred_group = _StridedPhaseLinear.apply(weight, x_phase)
            else:
                pred_group = torch.einsum("glk,bgkc->bglc", weight, x_phase)
            predictions.append(pred_group.flatten(1, 2))  # [B, G*L_phase, C]
            horizon_indices.append(horizon_index.flatten())

        if predictions:
            # Under AMP einsum may produce bf16/fp16 from fp32 inputs.  Anchor
            # the scatter destination to its compute dtype, matching the dense
            # SparsePeriodAdapter raw branch.
            out = predictions[0].new_zeros(
                x_norm.size(0), self.pred_len, x_norm.size(2)
            )
            grouped = torch.cat(predictions, dim=1)
            scatter_index = torch.cat(horizon_indices, dim=0)
            out = out.index_copy(1, scatter_index, grouped)
        else:
            # Preserve autocast compute dtype even for the unusual case where
            # no forecast phase has a legal historical tap.
            probe_taps = min(2, x_norm.size(1))
            zero_weight = x_norm.new_zeros(1, probe_taps)
            probe = torch.einsum(
                "lk,bkc->blc", zero_weight, x_norm[:, :probe_taps, :]
            )
            out = probe.new_zeros(x_norm.size(0), self.pred_len, x_norm.size(2))
        return out


class CompactPeriodAdapter(SparsePeriodAdapter):
    """Parameter- and memory-efficient FP32 form of ``SparsePeriodAdapter``.

    Only legal same-phase taps are parameters.  For a period ``p`` the input
    and forecast are split into phase groups, each group is evaluated as
    ``[L_phase, K_phase] x [B, K_phase, C]``, and the result is scattered back
    to horizon order.  Fusion gates, regularization, initialization and
    diagnostics intentionally retain ``SparsePeriodAdapter`` semantics.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        periods=(24,),
        periodic_init="seasonal_naive",
        periodic_sharing="shared",
        temporal_fusion="convex",
        temporal_gate_type="global",
        temporal_gate_init_logit=-4.0,
        period_fusion="sum_gated",
        period_gate_type="period",
        period_gate_init_logit=0.0,
        periodic_l1_weight=0.0,
        periodic_l2_weight=0.0,
        temporal_gate_l1_weight=0.0,
        period=None,
    ):
        # Bypass SparsePeriodAdapter.__init__: allocating its dense W_raw would
        # defeat the compact representation.  Its gate helpers are inherited.
        nn.Module.__init__(self)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        if period is not None:
            periods = [int(period)]
        periods = [max(1, int(p)) for p in periods]
        if not periods:
            periods = [24]
        self.periods = periods
        self.num_periods = len(periods)
        self.period = periods[0]
        self.periodic_init = str(periodic_init)
        self.periodic_sharing = str(periodic_sharing)
        self.temporal_fusion = str(temporal_fusion)
        self.temporal_gate_type = str(temporal_gate_type)
        self.period_fusion = str(period_fusion)
        self.period_gate_type = str(period_gate_type)
        self.periodic_l1_weight = float(periodic_l1_weight)
        self.periodic_l2_weight = float(periodic_l2_weight)
        self.temporal_gate_l1_weight = float(temporal_gate_l1_weight)
        if self.periodic_sharing != "shared":
            raise NotImplementedError("periodic_sharing=individual_channel is not implemented")
        if self.temporal_fusion not in {"convex", "additive"}:
            raise ValueError(f"Unsupported temporal_fusion={self.temporal_fusion!r}")
        if self.temporal_gate_type not in {"global", "channel", "horizon", "horizon_channel"}:
            raise ValueError(f"Unsupported temporal_gate_type={self.temporal_gate_type!r}")
        if self.period_fusion not in {"sum_gated", "softmax"}:
            raise ValueError(f"Unsupported period_fusion={self.period_fusion!r}")
        if self.period_gate_type not in {
            "global", "period", "period_horizon", "period_channel", "period_horizon_channel",
        }:
            raise ValueError(f"Unsupported period_gate_type={self.period_gate_type!r}")

        self.period_blocks = nn.ModuleList([
            _CompactSinglePeriod(self.seq_len, self.pred_len, p) for p in self.periods
        ])
        self._num_legal_weights = sum(
            weight.numel()
            for block in self.period_blocks
            for weight in block.weights
        )
        self._init_compact_weights()

        self.use_period_gate = self.num_periods > 1
        if self.use_period_gate:
            self.period_gate_logits = nn.Parameter(
                self._init_period_gate_logits(period_gate_init_logit)
            )
        self.temporal_gate_logit = nn.Parameter(
            self._init_temporal_gate_logits(temporal_gate_init_logit)
        )
        self._last_diagnostics: Dict[str, float] = {}

    def _copy_from_dense(self, dense_weights):
        """Copy the legal entries of a dense ``[P, H, T]`` representation."""
        expected = (self.num_periods, self.pred_len, self.seq_len)
        if tuple(dense_weights.shape) != expected:
            raise ValueError(
                f"dense period weights must have shape {expected}, got {tuple(dense_weights.shape)}"
            )
        with torch.no_grad():
            for period_index, block in enumerate(self.period_blocks):
                for weight, input_index, horizon_index, _ in block.groups():
                    dense_period = dense_weights[period_index].to(input_index.device)
                    selected = dense_period[
                        horizon_index.unsqueeze(-1), input_index.unsqueeze(1)
                    ]
                    weight.copy_(selected.to(device=weight.device, dtype=weight.dtype))

    def _init_compact_weights(self):
        with torch.no_grad():
            for block in self.period_blocks:
                for weight in block.weights:
                    weight.zero_()
            if self.periodic_init == "seasonal_naive":
                for block in self.period_blocks:
                    for weight in block.weights:
                        # Input indices are ascending; the final tap is the
                        # nearest historical observation with the same phase.
                        weight[:, :, -1] = 1.0
            elif self.periodic_init == "zeros":
                pass
            elif self.periodic_init == "small_random":
                # Draw the same dense-shaped random tensor as the reference
                # adapter, then retain only legal taps.  This makes seeded
                # initialization exactly reproducible across implementations
                # without retaining the dense parameter afterwards.
                anchor = next(self.parameters(), None)
                kwargs = {}
                if anchor is not None:
                    kwargs = {"device": anchor.device, "dtype": anchor.dtype}
                dense = torch.empty(
                    self.num_periods, self.pred_len, self.seq_len, **kwargs
                ).normal_(std=1e-3)
                self._copy_from_dense(dense)
            else:
                raise ValueError(f"Unsupported periodic_init={self.periodic_init!r}")

    def period_weight(self):
        """Return per-period fusion weights broadcast to [P, H, C]."""
        P, H, C = self.num_periods, self.pred_len, self.channels
        anchor = self.temporal_gate_logit
        if not self.use_period_gate:
            return torch.ones(P, H, C, device=anchor.device, dtype=anchor.dtype)
        logits = self.period_gate_logits
        t = self.period_gate_type
        if t == "global":
            logits = logits.view(1, 1, 1).expand(P, H, C)
        elif t == "period":
            logits = logits.view(P, 1, 1).expand(P, H, C)
        elif t == "period_horizon":
            logits = logits.view(P, H, 1).expand(P, H, C)
        elif t == "period_channel":
            logits = logits.view(P, 1, C).expand(P, H, C)
        else:
            logits = logits.view(P, H, C)
        if self.period_fusion == "softmax":
            return torch.softmax(logits, dim=0)
        return torch.sigmoid(logits)

    def _weight_sums(self):
        anchor = self.temporal_gate_logit
        abs_sum = anchor.sum() * 0.0
        square_sum = anchor.sum() * 0.0
        maxima = []
        for block in self.period_blocks:
            for weight in block.weights:
                abs_weight = weight.abs()
                abs_sum = abs_sum + abs_weight.sum()
                square_sum = square_sum + (weight ** 2).sum()
                if weight.numel() > 0:
                    maxima.append(abs_weight.max())
        abs_max = torch.stack(maxima).max() if maxima else abs_sum
        return abs_sum, square_sum, abs_max

    def extra_loss(self):
        abs_sum, square_sum, _ = self._weight_sums()
        loss = abs_sum * 0.0
        if self._num_legal_weights > 0:
            denom = float(self._num_legal_weights)
            if self.periodic_l1_weight > 0.0:
                loss = loss + self.periodic_l1_weight * (abs_sum / denom)
            if self.periodic_l2_weight > 0.0:
                loss = loss + self.periodic_l2_weight * (square_sum / denom)
        if self.temporal_gate_l1_weight > 0.0:
            loss = loss + self.temporal_gate_l1_weight * torch.sigmoid(
                self.temporal_gate_logit
            ).mean()
        return loss

    def forward(self, x_norm, pred_spec_norm):
        try:
            cpu_autocast_enabled = torch.is_autocast_enabled("cpu")
        except TypeError:  # older PyTorch
            cpu_autocast_enabled = getattr(
                torch, "is_autocast_cpu_enabled", lambda: False
            )()
        if torch.is_autocast_enabled() or cpu_autocast_enabled:
            raise RuntimeError(
                "temporal_adapter=compact_period currently requires autocast/AMP off"
            )
        # Advanced indexing would copy roughly [B, T, C] for every period.
        # Make at most one contiguous copy here, then let all period blocks use
        # lightweight as_strided phase views sharing this storage.
        x_period = x_norm.contiguous()
        weight = None
        pred_period_norm = None
        for period_index, block in enumerate(self.period_blocks):
            pred_period = block(x_period)  # [B, H, C], assembled from phase views
            if weight is None:
                # SparsePeriodAdapter casts period gates to einsum's output
                # dtype, which can differ from both x_norm and pred_spec under
                # autocast.  Delay allocation until that dtype is observable.
                weight = self.period_weight().to(pred_period.dtype)  # [P, H, C]
                pred_period_norm = pred_period.new_zeros(
                    pred_period.size(0), self.pred_len, pred_period.size(2)
                )
            pred_period_norm = pred_period_norm + pred_period * weight[period_index].unsqueeze(0)

        alpha = self.alpha().to(pred_spec_norm.dtype)
        if self.temporal_fusion == "convex":
            pred_fused_norm = pred_spec_norm + alpha * (pred_period_norm - pred_spec_norm)
        else:
            pred_fused_norm = pred_spec_norm + alpha * pred_period_norm

        with torch.no_grad():
            abs_sum, _, abs_max = self._weight_sums()
            alpha_det = alpha.detach().float().cpu()
            weight_det = weight.detach().float().cpu()
            dense_count = max(1, self.num_periods * self.pred_len * self.seq_len)
            self._last_diagnostics = {
                "temporal_adapter_enabled": 1.0,
                "num_periods": float(self.num_periods),
                "period": float(self.periods[0]),
                "periods": "+".join(str(p) for p in self.periods),
                "temporal_gate_mean": float(alpha_det.mean()),
                "temporal_gate_min": float(alpha_det.min()),
                "temporal_gate_max": float(alpha_det.max()),
                "temporal_gate_l1_value": float(self.temporal_gate_l1_weight * alpha_det.mean()),
                "period_weight_mean": float(weight_det.mean()),
                "period_weight_min": float(weight_det.min()),
                "period_weight_max": float(weight_det.max()),
                # Keep the dense reference denominator so dashboards comparing
                # sparse_period and compact_period retain identical semantics.
                "periodic_weight_abs_mean": float((abs_sum / dense_count).detach().cpu()),
                "periodic_weight_abs_max": float(abs_max.detach().cpu()),
                "periodic_mask_density": float(self._num_legal_weights / dense_count),
                "periodic_legal_parameters": float(self._num_legal_weights),
                "pred_period_norm_rms": float(torch.sqrt(torch.mean(pred_period_norm.detach() ** 2) + 1e-8).cpu()),
                "pred_spec_norm_rms": float(torch.sqrt(torch.mean(pred_spec_norm.detach() ** 2) + 1e-8).cpu()),
                "fused_delta_rms": float(torch.sqrt(torch.mean((pred_fused_norm.detach() - pred_spec_norm.detach()) ** 2) + 1e-8).cpu()),
            }
        return pred_fused_norm, pred_period_norm


class PatchLinearAdapter(nn.Module):
    """Tiny channel-independent patch-linear temporal residual (Phase 7).

    Splits the normalized lookback into overlapping patches, extracts a small
    per-patch feature (mean+last, or a learned basis), and maps the flattened
    per-channel features to the horizon with a single shared Linear. Fused into
    the horizon prediction via a gate that starts near-off. No attention.
    """

    def __init__(self, seq_len, pred_len, channels, patch_len=16, patch_stride=8,
                 patch_basis_dim=0, patch_fusion="convex", patch_gate_type="horizon",
                 patch_gate_init_logit=-6.0, patch_l1_weight=0.0, patch_l2_weight=0.0):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.patch_len = max(1, min(int(patch_len), self.seq_len))
        self.patch_stride = max(1, int(patch_stride))
        self.patch_basis_dim = int(patch_basis_dim)
        self.patch_fusion = str(patch_fusion)
        self.patch_gate_type = str(patch_gate_type)
        self.patch_l1_weight = float(patch_l1_weight)
        self.patch_l2_weight = float(patch_l2_weight)
        if self.patch_fusion not in {"convex", "additive"}:
            raise ValueError(f"Unsupported patch_fusion={self.patch_fusion!r}")
        if self.patch_gate_type not in {"global", "channel", "horizon", "horizon_channel"}:
            raise ValueError(f"Unsupported patch_gate_type={self.patch_gate_type!r}")

        self.num_patches = (self.seq_len - self.patch_len) // self.patch_stride + 1
        if self.num_patches < 1:
            self.num_patches = 1
        if self.patch_basis_dim > 0:
            self.basis = nn.Linear(self.patch_len, self.patch_basis_dim)
            self.feat_dim = self.patch_basis_dim
        else:
            self.basis = None
            self.feat_dim = 2  # mean, last
        self.proj = nn.Linear(self.num_patches * self.feat_dim, self.pred_len)

        H, C = self.pred_len, self.channels
        t = self.patch_gate_type
        shape = () if t == "global" else (C,) if t == "channel" else (H,) if t == "horizon" else (H, C)
        self.patch_gate_logit = nn.Parameter(torch.full(shape, float(patch_gate_init_logit)))
        self._last_diagnostics: Dict[str, float] = {}

    def alpha(self):
        a = torch.sigmoid(self.patch_gate_logit)
        t = self.patch_gate_type
        if t == "global":
            return a.view(1, 1, 1)
        if t == "channel":
            return a.view(1, 1, self.channels)
        if t == "horizon":
            return a.view(1, self.pred_len, 1)
        return a.view(1, self.pred_len, self.channels)

    def extra_loss(self):
        loss = self.proj.weight.sum() * 0.0
        if self.patch_l1_weight > 0.0:
            loss = loss + self.patch_l1_weight * self.proj.weight.abs().mean()
        if self.patch_l2_weight > 0.0:
            loss = loss + self.patch_l2_weight * (self.proj.weight ** 2).mean()
        return loss

    def raw(self, x_norm):
        """Standalone patch prediction pred_patch_norm [B, H, C] (no fusion)."""
        B, T, C = x_norm.shape
        xp = x_norm.permute(0, 2, 1)  # [B, C, T]
        patches = xp.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)  # [B, C, N, patch_len]
        if patches.size(2) != self.num_patches:  # guard against off-by-one on odd T
            patches = patches[:, :, :self.num_patches, :]
        if self.basis is not None:
            feat = self.basis(patches)  # [B, C, N, basis_dim]
        else:
            feat = torch.stack([patches.mean(dim=-1), patches[..., -1]], dim=-1)  # [B, C, N, 2]
        feat_flat = feat.reshape(B, C, self.num_patches * self.feat_dim)
        pred_patch = self.proj(feat_flat.reshape(B * C, -1)).reshape(B, C, self.pred_len)
        return pred_patch.permute(0, 2, 1)  # [B, H, C]

    def forward(self, x_norm, pred_base_norm):
        pred_patch_norm = self.raw(x_norm)
        alpha = self.alpha().to(pred_base_norm.dtype)
        if self.patch_fusion == "convex":
            pred = pred_base_norm + alpha * (pred_patch_norm - pred_base_norm)
        else:
            pred = pred_base_norm + alpha * pred_patch_norm

        with torch.no_grad():
            a = alpha.detach().float().cpu()
            self._last_diagnostics = {
                "patch_adapter_enabled": 1.0,
                "patch_gate_mean": float(a.mean()),
                "patch_gate_min": float(a.min()),
                "patch_gate_max": float(a.max()),
                "patch_pred_rms": float(torch.sqrt(torch.mean(pred_patch_norm.detach() ** 2) + 1e-8).cpu()),
                "patch_delta_rms": float(torch.sqrt(torch.mean((pred.detach() - pred_base_norm.detach()) ** 2) + 1e-8).cpu()),
                "patch_num_patches": float(self.num_patches),
                "patch_len": float(self.patch_len),
                "patch_stride": float(self.patch_stride),
            }
        return pred

    def get_diagnostics(self):
        return dict(self._last_diagnostics)


def _moving_avg(x_norm, kernel):
    """Length-preserving causal-safe moving average (replicate padding). x [B,T,C]."""
    kernel = max(1, int(kernel))
    if kernel == 1:
        return x_norm
    pad = (kernel - 1) // 2
    xp = x_norm.permute(0, 2, 1)  # [B, C, T]
    front = xp[..., :1].repeat(1, 1, pad)
    end = xp[..., -1:].repeat(1, 1, kernel - 1 - pad)
    xpad = torch.cat([front, xp, end], dim=-1)
    avg = F.avg_pool1d(xpad, kernel_size=kernel, stride=1)
    return avg.permute(0, 2, 1)  # [B, T, C]


class _CILinear(nn.Module):
    """Channel-independent Linear over the time axis. x [B,C,Tin] -> [B,C,H]."""

    def __init__(self, in_len, out_len, channels, sharing="shared", init="zeros",
                 bias=True, max_channels=64):
        super().__init__()
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.channels = int(channels)
        self.sharing = str(sharing)
        if self.sharing == "individual" and self.channels > int(max_channels):
            raise ValueError(f"linear_sharing=individual needs channels<={max_channels}, got {self.channels}")
        if self.sharing == "shared":
            self.weight = nn.Parameter(torch.zeros(self.out_len, self.in_len))
            self.bias = nn.Parameter(torch.zeros(self.out_len)) if bias else None
        else:
            self.weight = nn.Parameter(torch.zeros(self.channels, self.out_len, self.in_len))
            self.bias = nn.Parameter(torch.zeros(self.channels, self.out_len)) if bias else None
        self._init_weight(init)

    def _init_weight(self, init):
        with torch.no_grad():
            if init == "zeros":
                pass
            elif init == "last":
                # copy last lookback value to every horizon step
                if self.sharing == "shared":
                    self.weight[:, -1] = 1.0
                else:
                    self.weight[:, :, -1] = 1.0
            elif init == "small_random":
                self.weight.normal_(std=1e-3)
            else:
                raise ValueError(f"Unsupported linear_init={init!r}")

    def forward(self, x_ct):  # [B, C, Tin] -> [B, C, H]
        if self.sharing == "shared":
            return F.linear(x_ct, self.weight, self.bias)
        y = torch.einsum("cht,bct->bch", self.weight, x_ct)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)
        return y

    def weight_penalty(self, l1, l2):
        loss = self.weight.sum() * 0.0
        if l1 > 0:
            loss = loss + l1 * self.weight.abs().mean()
        if l2 > 0:
            loss = loss + l2 * (self.weight ** 2).mean()
        return loss


class LinearAdapter(nn.Module):
    """Direct-linear / DLinear-decomposition / multiscale-DLinear temporal branch.

    Operates on normalized time input x_norm [B,T,C] -> pred_linear_norm [B,H,C].
    Owns its own fusion gate; sequential fusion done in forward(), raw() used by
    the Hydra branch fusion.
    """

    def __init__(self, kind, seq_len, pred_len, channels, linear_sharing="shared",
                 linear_init="zeros", individual_linear_max_channels=64, moving_avg_kernel=25,
                 multiscale_factors=(1, 2, 4), multiscale_fusion="softmax",
                 multiscale_gate_type="scale", linear_fusion="convex",
                 linear_gate_type="horizon", linear_gate_init_logit=-6.0,
                 linear_l1_weight=0.0, linear_l2_weight=0.0):
        super().__init__()
        self.kind = str(kind)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.moving_avg_kernel = int(moving_avg_kernel)
        self.multiscale_fusion = str(multiscale_fusion)
        self.multiscale_gate_type = str(multiscale_gate_type)
        self.linear_fusion = str(linear_fusion)
        self.linear_gate_type = str(linear_gate_type)
        self.linear_l1_weight = float(linear_l1_weight)
        self.linear_l2_weight = float(linear_l2_weight)
        if self.kind not in {"direct_linear", "dlinear_decomp", "multiscale_dlinear"}:
            raise ValueError(f"Unsupported linear_adapter={self.kind!r}")
        if self.linear_fusion not in {"convex", "additive"}:
            raise ValueError(f"Unsupported linear_fusion={self.linear_fusion!r}")
        if self.linear_gate_type not in {"global", "channel", "horizon", "horizon_channel"}:
            raise ValueError(f"Unsupported linear_gate_type={self.linear_gate_type!r}")

        mk = dict(channels=self.channels, sharing=linear_sharing, init=linear_init,
                  max_channels=individual_linear_max_channels)
        if self.kind == "direct_linear":
            self.direct = _CILinear(self.seq_len, self.pred_len, **mk)
        elif self.kind == "dlinear_decomp":
            self.lin_trend = _CILinear(self.seq_len, self.pred_len, **mk)
            self.lin_seasonal = _CILinear(self.seq_len, self.pred_len, **mk)
        else:  # multiscale_dlinear
            factors = [int(f) for f in multiscale_factors if int(f) >= 1]
            self.scales = [f for f in factors if (self.seq_len // f) >= 2] or [1]
            self.scale_lins = nn.ModuleList(
                [_CILinear(self.seq_len // f, self.pred_len, **mk) for f in self.scales])
            S = len(self.scales)
            if self.multiscale_gate_type == "global":
                shp = (S,)
            elif self.multiscale_gate_type == "scale":
                shp = (S,)
            else:  # scale_horizon
                shp = (S, self.pred_len)
            self.scale_logit = nn.Parameter(torch.zeros(shp))

        H, C = self.pred_len, self.channels
        t = self.linear_gate_type
        gshape = () if t == "global" else (C,) if t == "channel" else (H,) if t == "horizon" else (H, C)
        self.linear_gate_logit = nn.Parameter(torch.full(gshape, float(linear_gate_init_logit)))
        self._last_diagnostics: Dict[str, float] = {}

    def alpha(self):
        a = torch.sigmoid(self.linear_gate_logit)
        t = self.linear_gate_type
        if t == "global":
            return a.view(1, 1, 1)
        if t == "channel":
            return a.view(1, 1, self.channels)
        if t == "horizon":
            return a.view(1, self.pred_len, 1)
        return a.view(1, self.pred_len, self.channels)

    def _scale_weights(self):
        S = len(self.scales)
        if self.multiscale_gate_type == "scale_horizon":
            logit = self.scale_logit  # [S, H]
        else:
            logit = self.scale_logit.view(S, 1)
        if self.multiscale_fusion == "softmax":
            return torch.softmax(logit, dim=0)  # [S,1] or [S,H]
        return torch.sigmoid(logit)

    def raw(self, x_norm):
        """Standalone linear prediction pred_linear_norm [B, H, C]."""
        xp = x_norm.permute(0, 2, 1)  # [B, C, T]
        if self.kind == "direct_linear":
            y = self.direct(xp)  # [B, C, H]
        elif self.kind == "dlinear_decomp":
            trend = _moving_avg(x_norm, self.moving_avg_kernel)
            seasonal = x_norm - trend
            y = self.lin_trend(trend.permute(0, 2, 1)) + self.lin_seasonal(seasonal.permute(0, 2, 1))
        else:  # multiscale_dlinear
            w = self._scale_weights()  # [S,1] or [S,H]
            acc = None
            for i, f in enumerate(self.scales):
                xs = F.avg_pool1d(xp, kernel_size=f, stride=f) if f > 1 else xp  # [B, C, T_s]
                ys = self.scale_lins[i](xs)  # [B, C, H]
                wi = w[i]  # [1] or [H]
                acc = ys * wi.view(1, 1, -1) if acc is None else acc + ys * wi.view(1, 1, -1)
            y = acc
        return y.permute(0, 2, 1)  # [B, H, C]

    def forward(self, x_norm, pred_base_norm):
        pred_linear_norm = self.raw(x_norm)
        alpha = self.alpha().to(pred_base_norm.dtype)
        if self.linear_fusion == "convex":
            pred = pred_base_norm + alpha * (pred_linear_norm - pred_base_norm)
        else:
            pred = pred_base_norm + alpha * pred_linear_norm
        with torch.no_grad():
            a = alpha.detach().float().cpu()
            self._last_diagnostics = {
                "linear_adapter_enabled": 1.0,
                "linear_kind": self.kind,
                "linear_gate_mean": float(a.mean()),
                "linear_gate_min": float(a.min()),
                "linear_gate_max": float(a.max()),
                "pred_linear_rms": float(torch.sqrt(torch.mean(pred_linear_norm.detach() ** 2) + 1e-8).cpu()),
                "linear_delta_rms": float(torch.sqrt(torch.mean((pred.detach() - pred_base_norm.detach()) ** 2) + 1e-8).cpu()),
            }
            if self.kind == "multiscale_dlinear":
                self._last_diagnostics["multiscale_scale_weight_mean"] = float(self._scale_weights().detach().float().mean().cpu())
        return pred

    def extra_loss(self):
        l1, l2 = self.linear_l1_weight, self.linear_l2_weight
        mods = []
        if self.kind == "direct_linear":
            mods = [self.direct]
        elif self.kind == "dlinear_decomp":
            mods = [self.lin_trend, self.lin_seasonal]
        else:
            mods = list(self.scale_lins)
        loss = mods[0].weight.sum() * 0.0
        if l1 <= 0 and l2 <= 0:
            return loss
        for m in mods:
            loss = loss + m.weight_penalty(l1, l2)
        return loss

    def get_diagnostics(self):
        return dict(self._last_diagnostics)


class ForecastabilityAdapter(nn.Module):
    """Low-rank, channel-separable past-to-future forecast operator.

    The effective temporal kernel for channel ``c`` is

    ``W_c = future_basis @ diag(channel_scale[c]) @ past_basis``.

    There is deliberately no contraction across the channel axis: every output
    channel only consumes the matching input channel.  The factors are shared
    across channels by default; the optional channel scale changes mode
    amplitudes without introducing cross-variable lead--lag.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        rank=8,
        init="small_random",
        init_path="",
        expected_data="",
        expected_data_path="",
        expected_norm_mode="",
        channel_scale=False,
        spectral_mixtures=0,
        sm_sharing="mode",
        sm_max_log_gain=1.0,
        sm_min_width=0.02,
        sm_init_width=0.20,
        sm_center_max_shift=0.15,
        sm_base_trainable=True,
        extension_shrink="none",
        phase_basis_dim=0,
        phase_max=math.pi / 4.0,
        fusion="convex",
        gate_type="horizon",
        gate_init_logit=-6.0,
        l1_weight=0.0,
        l2_weight=0.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        requested_rank = int(rank)
        if requested_rank < 1:
            raise ValueError(f"forecast_kernel_rank must be positive, got {requested_rank}")
        self.rank = min(requested_rank, self.seq_len, self.pred_len)
        self.init = str(init)
        self.init_path = str(init_path or "")
        self.expected_data = str(expected_data or "")
        self.expected_data_path = str(expected_data_path or "")
        self.expected_norm_mode = str(expected_norm_mode or "")
        self.channel_scale_enabled = _as_bool(channel_scale)
        self.spectral_mixtures = int(spectral_mixtures)
        self.sm_sharing = str(sm_sharing)
        self.sm_max_log_gain = float(sm_max_log_gain)
        self.sm_min_width = float(sm_min_width)
        self.sm_init_width = float(sm_init_width)
        self.sm_center_max_shift = float(sm_center_max_shift)
        self.sm_base_trainable = _as_bool(sm_base_trainable)
        self.extension_shrink = str(extension_shrink)
        self.phase_basis_dim = int(phase_basis_dim)
        self.phase_max = float(phase_max)
        self.fusion = str(fusion)
        self.gate_type = str(gate_type)
        self.l1_weight = float(l1_weight)
        self.l2_weight = float(l2_weight)
        if self.init not in {"zeros", "small_random", "ridge_svd"}:
            raise ValueError(f"Unsupported forecast_kernel_init={self.init!r}")
        if self.fusion not in {"convex", "additive"}:
            raise ValueError(f"Unsupported forecast_kernel_fusion={self.fusion!r}")
        if self.gate_type not in {"global", "channel", "horizon", "horizon_channel"}:
            raise ValueError(f"Unsupported forecast_kernel_gate_type={self.gate_type!r}")
        if self.spectral_mixtures < 0:
            raise ValueError(
                "forecast_kernel_spectral_mixtures must be non-negative, "
                f"got {self.spectral_mixtures}"
            )
        if self.sm_sharing not in {"shared", "mode"}:
            raise ValueError(f"Unsupported forecast_kernel_sm_sharing={self.sm_sharing!r}")
        if self.sm_max_log_gain <= 0.0:
            raise ValueError("forecast_kernel_sm_max_log_gain must be positive")
        if not 0.0 < self.sm_min_width < self.sm_init_width:
            raise ValueError(
                "forecast_kernel_sm_widths require 0 < min_width < init_width"
            )
        if self.sm_center_max_shift < 0.0:
            raise ValueError("forecast_kernel_sm_center_max_shift must be non-negative")
        if self.extension_shrink not in {"none", "tail2_linear"}:
            raise ValueError(
                "Unsupported forecast_kernel_extension_shrink="
                f"{self.extension_shrink!r}"
            )
        if self.extension_shrink != "none" and self.spectral_mixtures <= 0:
            raise ValueError(
                "forecast_kernel_extension_shrink requires a spectral-mixture envelope"
            )
        if self.phase_basis_dim < 0:
            raise ValueError("forecast_kernel_phase_basis_dim must be non-negative")
        if self.phase_basis_dim > 0 and self.spectral_mixtures <= 0:
            raise ValueError("complex phase requires a real spectral-mixture envelope")
        if not 0.0 < self.phase_max <= math.pi:
            raise ValueError("forecast_kernel_phase_max must be in (0, pi]")

        # Stage D: the only promoted schedule is fixed before training.  It is
        # exact identity until the forecast extends beyond twice the observed
        # context, then opens linearly with H/L.  This is a Python scalar, not
        # a parameter or dataset-dependent statistic.
        if self.extension_shrink == "tail2_linear":
            self.extension_scale = max(
                0.0, 1.0 - 2.0 * float(self.seq_len) / float(self.pred_len)
            )
        else:
            self.extension_scale = 1.0

        self.past_basis = nn.Parameter(torch.empty(self.rank, self.seq_len))
        self.future_basis = nn.Parameter(torch.empty(self.pred_len, self.rank))
        self.horizon_bias = nn.Parameter(torch.zeros(self.pred_len))
        if self.channel_scale_enabled:
            self.channel_scale = nn.Parameter(torch.ones(self.channels, self.rank))
        else:
            self.register_parameter("channel_scale", None)

        # Stage B: a positive real spectral-mixture envelope modulates each
        # past-analysis mode.  It is shared across channels and therefore
        # cannot encode inter-variable lead--lag.  A zero tanh gate makes the
        # initial effective basis exactly equal to the Stage-A basis.
        if self.spectral_mixtures > 0:
            sm_rows = 1 if self.sm_sharing == "shared" else self.rank
            components = self.spectral_mixtures
            if components == 1:
                anchors = torch.full((1,), 0.5)
            else:
                anchors = torch.linspace(0.0, 1.0, components)
            self.register_buffer(
                "sm_center_anchor",
                anchors.view(1, components).expand(sm_rows, components).clone(),
            )
            self.register_buffer(
                "sm_frequency",
                torch.linspace(0.0, 1.0, self.seq_len // 2 + 1),
                persistent=False,
            )
            self.sm_weight_logits = nn.Parameter(torch.zeros(sm_rows, components))
            self.sm_center_offset = nn.Parameter(torch.zeros(sm_rows, components))
            width_delta = max(self.sm_init_width - self.sm_min_width, 1e-6)
            width_raw = math.log(math.expm1(width_delta))
            self.sm_width_raw = nn.Parameter(
                torch.full((sm_rows, components), width_raw)
            )
            self.forecast_kernel_sm_gate_logit = nn.Parameter(torch.zeros(sm_rows))
            if not self.sm_base_trainable:
                self.past_basis.requires_grad_(False)

        # Stage B.2: a compact smooth phase curve rotates an analytic
        # (in-phase, Hilbert-quadrature) latent independently within each
        # channel.  Zero coefficients recover the real-SM model exactly.
        if self.phase_basis_dim > 0:
            horizon = torch.arange(self.pred_len, dtype=torch.float32).view(-1, 1)
            modes = torch.arange(self.phase_basis_dim, dtype=torch.float32).view(1, -1)
            phase_basis = torch.cos(
                math.pi * (horizon + 0.5) * modes / float(self.pred_len)
            )
            self.register_buffer(
                "phase_horizon_basis", phase_basis, persistent=False
            )
            multiplier = torch.full(
                (self.seq_len // 2 + 1,), -1j, dtype=torch.complex64
            )
            multiplier[0] = 0.0
            if self.seq_len % 2 == 0:
                multiplier[-1] = 0.0
            self.register_buffer(
                "phase_quadrature_multiplier", multiplier, persistent=False
            )
            self.phase_coeff = nn.Parameter(
                torch.zeros(self.rank, self.phase_basis_dim)
            )

        H, C = self.pred_len, self.channels
        shape = (
            () if self.gate_type == "global"
            else (C,) if self.gate_type == "channel"
            else (H,) if self.gate_type == "horizon"
            else (H, C)
        )
        self.forecast_kernel_gate_logit = nn.Parameter(
            torch.full(shape, float(gate_init_logit))
        )
        self.init_metadata = {}
        self._init_factors()
        self._last_diagnostics: Dict[str, float] = {}
        self._sm_eval_factor_cache = None
        self._sm_eval_factor_signature = None
        self._sm_eval_basis_cache = None
        self._sm_eval_basis_signature = None
        self._phase_eval_cache = None
        self._phase_eval_signature = None
        self._quadrature_eval_cache = None
        self._quadrature_eval_signature = None

    @staticmethod
    def _dct_rows(rank, length, dtype=torch.float32):
        """Return the first ``rank`` orthonormal DCT-II rows."""
        positions = torch.arange(length, dtype=dtype).view(1, length)
        modes = torch.arange(rank, dtype=dtype).view(rank, 1)
        basis = torch.cos(math.pi * (positions + 0.5) * modes / float(length))
        basis[0].mul_(1.0 / math.sqrt(float(length)))
        if rank > 1:
            basis[1:].mul_(math.sqrt(2.0 / float(length)))
        return basis

    def _init_factors(self):
        with torch.no_grad():
            self.past_basis.copy_(self._dct_rows(self.rank, self.seq_len))
            self.future_basis.zero_()
            self.horizon_bias.zero_()

        if self.init == "small_random":
            # A well-conditioned past projection plus a tiny future map keeps
            # the standalone forecast finite while the near-off fusion gate
            # protects the existing spectral prediction.
            with torch.no_grad():
                self.future_basis.normal_(std=1e-3)
            self.init_metadata = {"source": "small_random"}
            return
        if self.init == "zeros":
            self.init_metadata = {"source": "zeros"}
            return
        if not self.init_path:
            raise ValueError("forecast_kernel_init=ridge_svd requires --forecast_kernel_init_path")
        self._load_ridge_svd(self.init_path)

    def _load_ridge_svd(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("forecastability initializer must contain a dictionary payload")
        meta = dict(payload.get("meta", {}))
        for name, expected in (
            ("seq_len", self.seq_len),
            ("pred_len", self.pred_len),
        ):
            actual = int(meta.get(name, -1))
            if actual != expected:
                raise ValueError(
                    f"forecastability initializer {name} mismatch: expected {expected}, got {actual}"
                )
        artifact_rank = int(meta.get("rank", -1))
        if artifact_rank < self.rank:
            raise ValueError(
                f"forecastability initializer rank mismatch: need >= {self.rank}, got {artifact_rank}"
            )
        if self.expected_data and str(meta.get("data", "")) != self.expected_data:
            raise ValueError(
                "forecastability initializer data mismatch: "
                f"expected {self.expected_data!r}, got {meta.get('data', '')!r}"
            )
        if self.expected_data_path:
            expected_name = os.path.basename(self.expected_data_path)
            actual_name = os.path.basename(str(meta.get("data_path", "")))
            if actual_name != expected_name:
                raise ValueError(
                    "forecastability initializer data_path mismatch: "
                    f"expected {expected_name!r}, got {actual_name!r}"
                )
        if self.expected_norm_mode and str(meta.get("norm_mode", "")) != self.expected_norm_mode:
            raise ValueError(
                "forecastability initializer norm_mode mismatch: "
                f"expected {self.expected_norm_mode!r}, got {meta.get('norm_mode', '')!r}"
            )
        if meta.get("split") != "train" or meta.get("train_only") is not True:
            raise ValueError("forecastability initializer must be fitted on the train split only")
        past = torch.as_tensor(payload.get("past_basis"), dtype=self.past_basis.dtype)
        future = torch.as_tensor(payload.get("future_basis"), dtype=self.future_basis.dtype)
        bias = torch.as_tensor(payload.get("horizon_bias"), dtype=self.horizon_bias.dtype)
        if past.dim() != 2 or past.size(1) != self.seq_len or past.size(0) < self.rank:
            raise ValueError(
                f"initializer past_basis must be [rank>={self.rank},{self.seq_len}], got {tuple(past.shape)}"
            )
        if future.dim() != 2 or future.size(0) != self.pred_len or future.size(1) < self.rank:
            raise ValueError(
                f"initializer future_basis must be [{self.pred_len},rank>={self.rank}], got {tuple(future.shape)}"
            )
        if tuple(bias.shape) != (self.pred_len,):
            raise ValueError(
                f"initializer horizon_bias must be [{self.pred_len}], got {tuple(bias.shape)}"
            )
        with torch.no_grad():
            self.past_basis.copy_(past[: self.rank])
            self.future_basis.copy_(future[:, : self.rank])
            self.horizon_bias.copy_(bias)
        self.init_metadata = meta
        self.init_metadata["source"] = "ridge_svd"
        self.init_metadata["path"] = str(path)

    def alpha(self):
        alpha = torch.sigmoid(self.forecast_kernel_gate_logit)
        if self.gate_type == "global":
            return alpha.view(1, 1, 1)
        if self.gate_type == "channel":
            return alpha.view(1, 1, self.channels)
        if self.gate_type == "horizon":
            return alpha.view(1, self.pred_len, 1)
        return alpha.view(1, self.pred_len, self.channels)

    def spectral_mixture_factor(self):
        """Return a positive real, channel-shared frequency response [R,F]."""
        if self.spectral_mixtures <= 0:
            return None
        cacheable = not self.training and not torch.is_grad_enabled()
        signature = tuple(
            parameter._version
            for parameter in (
                self.sm_weight_logits,
                self.sm_center_offset,
                self.sm_width_raw,
                self.forecast_kernel_sm_gate_logit,
            )
        )
        cached = self._sm_eval_factor_cache
        if (
            cacheable
            and cached is not None
            and signature == self._sm_eval_factor_signature
            and cached.device == self.past_basis.device
            and cached.dtype == self.past_basis.dtype
        ):
            return cached
        dtype = self.past_basis.dtype
        device = self.past_basis.device
        frequency = self.sm_frequency.to(dtype=dtype, device=device).view(1, 1, -1)
        centers = self.sm_center_anchor.to(dtype=dtype, device=device)
        centers = centers + self.sm_center_max_shift * torch.tanh(self.sm_center_offset)
        centers = centers.clamp(0.0, 1.0).unsqueeze(-1)
        widths = (
            torch.nn.functional.softplus(self.sm_width_raw) + self.sm_min_width
        ).unsqueeze(-1)
        weights = torch.softmax(self.sm_weight_logits, dim=-1).unsqueeze(-1)
        mixture = torch.sum(
            weights * torch.exp(-0.5 * ((frequency - centers) / widths) ** 2),
            dim=1,
        )
        # Geometric-mean normalization separates overall mode scale (already
        # represented by U and s[c,r]) from frequency selectivity.
        log_shape = torch.log(mixture.clamp_min(1e-8))
        log_shape = (log_shape - log_shape.mean(dim=-1, keepdim=True)).clamp(-3.0, 3.0)
        gain = self.extension_scale * self.sm_max_log_gain * torch.tanh(
            self.forecast_kernel_sm_gate_logit
        ).view(-1, 1)
        factor = torch.exp(gain * log_shape)
        if self.sm_sharing == "shared":
            factor = factor.expand(self.rank, -1)
        if cacheable:
            self._sm_eval_factor_cache = factor.detach()
            self._sm_eval_factor_signature = signature
        return factor

    def effective_past_basis(self):
        """Apply the zero-phase envelope while retaining exact identity at gate=0."""
        cacheable = not self.training and not torch.is_grad_enabled()
        signature = None
        if self.spectral_mixtures > 0:
            if self.extension_scale == 0.0:
                return self.past_basis
            signature = (
                self.past_basis._version,
                self.sm_weight_logits._version,
                self.sm_center_offset._version,
                self.sm_width_raw._version,
                self.forecast_kernel_sm_gate_logit._version,
            )
            cached = self._sm_eval_basis_cache
            if (
                cacheable
                and cached is not None
                and signature == self._sm_eval_basis_signature
                and cached.device == self.past_basis.device
                and cached.dtype == self.past_basis.dtype
            ):
                return cached
        factor = self.spectral_mixture_factor()
        if factor is None:
            return self.past_basis
        spectrum = torch.fft.rfft(self.past_basis, dim=-1)
        # Adding only the spectral delta avoids an FFT round-trip on the base
        # path.  At the zero gate this is base + irfft(0), exactly Stage A.
        delta = torch.fft.irfft(
            spectrum * (factor.to(spectrum.dtype) - 1.0),
            n=self.seq_len,
            dim=-1,
        )
        effective = self.past_basis + delta
        if cacheable:
            self._sm_eval_basis_cache = effective.detach()
            self._sm_eval_basis_signature = signature
        return effective

    def horizon_phase(self):
        """Return the bounded smooth within-variable phase [H,R]."""
        if self.phase_basis_dim <= 0:
            return None
        cacheable = not self.training and not torch.is_grad_enabled()
        signature = self.phase_coeff._version
        cached = self._phase_eval_cache
        if (
            cacheable
            and cached is not None
            and signature == self._phase_eval_signature
            and cached.device == self.past_basis.device
            and cached.dtype == self.past_basis.dtype
        ):
            return cached
        basis = self.phase_horizon_basis.to(
            dtype=self.past_basis.dtype, device=self.past_basis.device
        )
        raw_phase = basis @ self.phase_coeff.transpose(0, 1)
        phase = self.extension_scale * self.phase_max * torch.tanh(raw_phase)
        if cacheable:
            self._phase_eval_cache = phase.detach()
            self._phase_eval_signature = signature
        return phase

    def quadrature_past_basis(self, past_basis=None):
        """Return the cyclic Hilbert quadrature of the real SM basis."""
        if self.phase_basis_dim <= 0:
            return None
        cacheable = not self.training and not torch.is_grad_enabled()
        signature = (
            self.past_basis._version,
            self.sm_weight_logits._version,
            self.sm_center_offset._version,
            self.sm_width_raw._version,
            self.forecast_kernel_sm_gate_logit._version,
        )
        cached = self._quadrature_eval_cache
        if (
            cacheable
            and cached is not None
            and signature == self._quadrature_eval_signature
            and cached.device == self.past_basis.device
            and cached.dtype == self.past_basis.dtype
        ):
            return cached
        if past_basis is None:
            past_basis = self.effective_past_basis()
        spectrum = torch.fft.rfft(past_basis, dim=-1)
        multiplier = self.phase_quadrature_multiplier.to(
            dtype=spectrum.dtype, device=spectrum.device
        )
        quadrature = torch.fft.irfft(
            spectrum * multiplier,
            n=self.seq_len,
            dim=-1,
        )
        if cacheable:
            self._quadrature_eval_cache = quadrature.detach()
            self._quadrature_eval_signature = signature
        return quadrature

    def train(self, mode=True):
        # A training/evaluation transition is an explicit cache boundary.  The
        # parameter-version signature also invalidates direct in-place edits.
        self._sm_eval_factor_cache = None
        self._sm_eval_factor_signature = None
        self._sm_eval_basis_cache = None
        self._sm_eval_basis_signature = None
        self._phase_eval_cache = None
        self._phase_eval_signature = None
        self._quadrature_eval_cache = None
        self._quadrature_eval_signature = None
        return super().train(mode)

    def raw(self, x_norm):
        """Return a standalone forecast without any cross-channel mixing."""
        if x_norm.dim() != 3 or x_norm.size(1) != self.seq_len or x_norm.size(2) != self.channels:
            raise ValueError(
                f"forecastability kernel expects [B,{self.seq_len},{self.channels}], "
                f"got {tuple(x_norm.shape)}"
            )
        x_ct = x_norm.permute(0, 2, 1)  # [B,C,T]
        # Two small GEMMs are substantially faster than generic einsum on H100
        # while preserving exactly the same channel-separable contraction.
        past_basis = self.effective_past_basis()
        latent = torch.matmul(x_ct, past_basis.transpose(0, 1))
        if self.channel_scale is not None:
            latent = latent * self.channel_scale.unsqueeze(0)
        phase = self.horizon_phase()
        if phase is None or self.extension_scale == 0.0:
            pred = torch.matmul(latent, self.future_basis.transpose(0, 1))
        else:
            quadrature_basis = self.quadrature_past_basis(past_basis)
            quadrature_latent = torch.matmul(
                x_ct, quadrature_basis.transpose(0, 1)
            )
            if self.channel_scale is not None:
                quadrature_latent = quadrature_latent * self.channel_scale.unsqueeze(0)
            future_cos = self.future_basis * torch.cos(phase)
            future_sin = self.future_basis * torch.sin(phase)
            pred = torch.matmul(latent, future_cos.transpose(0, 1))
            pred = pred + torch.matmul(
                quadrature_latent, future_sin.transpose(0, 1)
            )
        pred = pred + self.horizon_bias.view(1, 1, self.pred_len)
        return pred.permute(0, 2, 1)  # [B,H,C]

    def weight_matrix(self, channel=None):
        """Materialize the small H x T effective kernel for tests/audits."""
        future = self.future_basis
        if self.channel_scale is not None:
            if channel is None:
                raise ValueError("channel must be provided when channel scaling is enabled")
            future = future * self.channel_scale[int(channel)].view(1, self.rank)
        past = self.effective_past_basis()
        phase = self.horizon_phase()
        if phase is None or self.extension_scale == 0.0:
            return future @ past
        quadrature = self.quadrature_past_basis(past)
        return (future * torch.cos(phase)) @ past + (
            future * torch.sin(phase)
        ) @ quadrature

    def forward(self, x_norm, pred_base_norm):
        pred_kernel_norm = self.raw(x_norm)
        alpha = self.alpha().to(pred_base_norm.dtype)
        if self.fusion == "convex":
            pred = pred_base_norm + alpha * (pred_kernel_norm - pred_base_norm)
        else:
            pred = pred_base_norm + alpha * pred_kernel_norm

        with torch.no_grad():
            a = alpha.detach().float().cpu()
            # Avoid an HxR/RxT SVD on every batch.  A mode is numerically active
            # when the product of its two factor norms is material relative to
            # the strongest mode; this detects collapsed/zero modes in O(R).
            mode_activity = (
                torch.linalg.vector_norm(self.future_basis.detach().float(), dim=0)
                * torch.linalg.vector_norm(self.past_basis.detach().float(), dim=1)
            )
            activity_max = mode_activity.max().clamp_min(1e-12)
            effective_rank = int((mode_activity > activity_max * 1e-6).sum().cpu())
            self._last_diagnostics = {
                "forecast_kernel_enabled": 1.0,
                "forecast_kernel_rank": float(self.rank),
                "forecast_kernel_effective_rank": float(effective_rank),
                "forecast_kernel_channel_scale": float(self.channel_scale is not None),
                "forecast_kernel_gate_mean": float(a.mean()),
                "forecast_kernel_gate_min": float(a.min()),
                "forecast_kernel_gate_max": float(a.max()),
                "forecast_kernel_raw_rms": float(
                    torch.sqrt(torch.mean(pred_kernel_norm.detach() ** 2) + 1e-8).cpu()
                ),
                "forecast_kernel_delta_rms": float(
                    torch.sqrt(torch.mean((pred.detach() - pred_base_norm.detach()) ** 2) + 1e-8).cpu()
                ),
                "forecast_kernel_extension_scale": float(self.extension_scale),
                "forecast_kernel_extension_identity": float(
                    self.extension_scale == 0.0
                ),
            }
            if self.spectral_mixtures > 0:
                sm_gate = torch.tanh(
                    self.forecast_kernel_sm_gate_logit.detach().float()
                )
                if self.extension_scale == 0.0:
                    factor_min = 1.0
                    factor_max = 1.0
                else:
                    factor_cpu = self.spectral_mixture_factor().detach().float().cpu()
                    factor_min = float(factor_cpu.min())
                    factor_max = float(factor_cpu.max())
                self._last_diagnostics.update(
                    {
                        "forecast_kernel_sm_enabled": 1.0,
                        "forecast_kernel_sm_components": float(self.spectral_mixtures),
                        "forecast_kernel_sm_gate_mean": float(sm_gate.mean().cpu()),
                        "forecast_kernel_sm_gate_abs_max": float(sm_gate.abs().max().cpu()),
                        "forecast_kernel_sm_effective_gate_abs_max": float(
                            self.extension_scale * sm_gate.abs().max().cpu()
                        ),
                        "forecast_kernel_sm_factor_min": factor_min,
                        "forecast_kernel_sm_factor_max": factor_max,
                    }
                )
            else:
                self._last_diagnostics["forecast_kernel_sm_enabled"] = 0.0
            phase = self.horizon_phase()
            if phase is not None:
                phase_cpu = phase.detach().float().cpu()
                self._last_diagnostics.update(
                    {
                        "forecast_kernel_phase_enabled": 1.0,
                        "forecast_kernel_phase_basis_dim": float(self.phase_basis_dim),
                        "forecast_kernel_phase_abs_mean": float(phase_cpu.abs().mean()),
                        "forecast_kernel_phase_abs_max": float(phase_cpu.abs().max()),
                        "forecast_kernel_phase_rms": float(
                            torch.sqrt(torch.mean(phase_cpu ** 2) + 1e-12)
                        ),
                    }
                )
            else:
                self._last_diagnostics["forecast_kernel_phase_enabled"] = 0.0
        return pred

    def extra_loss(self):
        loss = (self.past_basis.sum() + self.future_basis.sum() + self.horizon_bias.sum()) * 0.0
        params = [self.past_basis, self.future_basis]
        if self.channel_scale is not None:
            params.append(self.channel_scale)
        if self.phase_basis_dim > 0:
            params.append(self.phase_coeff)
        if self.l1_weight > 0.0:
            loss = loss + self.l1_weight * sum(p.abs().mean() for p in params)
        if self.l2_weight > 0.0:
            loss = loss + self.l2_weight * sum((p ** 2).mean() for p in params)
        return loss

    def get_diagnostics(self):
        return dict(self._last_diagnostics)


class BranchFusion(nn.Module):
    """Static softmax mixture over parallel branch horizon predictions (Hydra).

    Softmax is over the branch dimension only (weights sum to 1 -> bounded, no
    additive explosion). The spec branch starts strongly favored so the model
    begins close to the spectral/cross prediction.
    """

    def __init__(self, branch_names, pred_len, channels, scope="horizon",
                 main_logit=4.0, aux_logit=-4.0):
        super().__init__()
        self.branch_names = list(branch_names)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.scope = str(scope)
        n = len(self.branch_names)
        if self.scope not in {"global", "horizon", "channel", "horizon_channel"}:
            raise ValueError(f"Unsupported branch_fusion_scope={self.scope!r}")
        H, C = self.pred_len, self.channels
        if self.scope == "global":
            shp = (n,)
        elif self.scope == "horizon":
            shp = (n, H)
        elif self.scope == "channel":
            shp = (n, C)
        else:
            shp = (n, H, C)
        init = torch.full(shp, float(aux_logit))
        init[0] = float(main_logit)  # branch 0 == spec
        self.branch_logit = nn.Parameter(init)
        self._last_diagnostics: Dict[str, float] = {}

    def weights(self):
        """Return [n, H, C] softmax weights over the branch dim."""
        n, H, C = len(self.branch_names), self.pred_len, self.channels
        logit = self.branch_logit
        if self.scope == "global":
            logit = logit.view(n, 1, 1).expand(n, H, C)
        elif self.scope == "horizon":
            logit = logit.view(n, H, 1).expand(n, H, C)
        elif self.scope == "channel":
            logit = logit.view(n, 1, C).expand(n, H, C)
        else:
            logit = logit.view(n, H, C)
        return torch.softmax(logit, dim=0)

    def forward(self, branch_preds):
        # branch_preds: dict name -> [B, H, C]; ordered by self.branch_names
        names = [b for b in self.branch_names if b in branch_preds]
        preds = torch.stack([branch_preds[b] for b in names], dim=0)  # [n, B, H, C]
        w = self.weights().to(preds.dtype)  # [n, H, C]
        if len(names) != len(self.branch_names):
            # some branch missing at runtime: renormalize over the present subset
            idx = [self.branch_names.index(b) for b in names]
            w = w[idx]
            w = w / w.sum(dim=0, keepdim=True).clamp_min(1e-8)
        out = (preds * w.unsqueeze(1)).sum(dim=0)  # [B, H, C]
        with torch.no_grad():
            wd = w.detach().float()
            self._last_diagnostics = {"branch_fusion": 1.0, "branches": "+".join(names)}
            for i, b in enumerate(names):
                self._last_diagnostics[f"branch_weight_{b}_mean"] = float(wd[i].mean().cpu())
            ent = -(wd.clamp_min(1e-8) * wd.clamp_min(1e-8).log()).sum(dim=0).mean()
            self._last_diagnostics["branch_entropy"] = float(ent.cpu())
        return out

    def get_diagnostics(self):
        return dict(self._last_diagnostics)


class CycleResidual(nn.Module):
    """Tiny learned phase template used as a residual preconditioner.

    ``cycle_index`` is the forecast-origin phase emitted by the data loader.
    Input phases therefore start at ``cycle_index - seq_len`` and future phases
    start at ``cycle_index``.  The table is initialized to exactly zero, so
    enabling this module preserves the legacy AsySpecX prediction at step zero.

    A full table costs ``cycle_len * channels`` parameters.  The factorized
    form costs ``rank * (cycle_len + channels)`` and also starts at zero by
    initializing only the channel factor nonzero.
    """

    def __init__(self, cycle_len, channels, rank=0, init_std=0.02):
        super().__init__()
        self.cycle_len = max(1, int(cycle_len))
        self.channels = int(channels)
        self.rank = max(0, min(int(rank), self.cycle_len, self.channels))
        if self.rank > 0:
            if float(init_std) <= 0.0:
                raise ValueError("factorized cycle_residual_init_std must be > 0")
            self.phase_factor = nn.Parameter(torch.zeros(self.cycle_len, self.rank))
            self.channel_factor = nn.Parameter(torch.empty(self.channels, self.rank))
            nn.init.normal_(self.channel_factor, std=float(init_std))
            self.register_parameter("table", None)
        else:
            self.table = nn.Parameter(torch.zeros(self.cycle_len, self.channels))
            self.register_parameter("phase_factor", None)
            self.register_parameter("channel_factor", None)

    def values(self):
        if self.rank > 0:
            return self.phase_factor @ self.channel_factor.transpose(0, 1)
        return self.table

    def gather(self, cycle_index, start_offset, length):
        if cycle_index is None:
            raise ValueError("cycle_residual=1 requires cycle_index from the data loader")
        index = cycle_index.to(dtype=torch.long).view(-1, 1)
        steps = torch.arange(int(length), device=index.device, dtype=torch.long).view(1, -1)
        phase = (index + int(start_offset) + steps) % self.cycle_len
        return self.values()[phase]

    def split(self, cycle_index, seq_len, pred_len):
        cycle_in = self.gather(cycle_index, -int(seq_len), int(seq_len))
        cycle_future = self.gather(cycle_index, 0, int(pred_len))
        return cycle_in, cycle_future

    def full(self, cycle_index, seq_len, pred_len):
        return self.gather(cycle_index, -int(seq_len), int(seq_len) + int(pred_len))


class Model(nn.Module):
    """AsySpecX forecaster."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.spectral_lift = str(_config(configs, "spectral_lift", "complex_mlp"))
        self.lift_sharing = str(_config(configs, "lift_sharing", "shared"))
        if self.lift_sharing not in {"shared", "individual", "lowrank_channel"}:
            raise ValueError(f"Unsupported lift_sharing={self.lift_sharing!r}")
        self.individual = self.lift_sharing == "individual" or _as_bool(_config(configs, "individual", 0))
        if self.lift_sharing == "lowrank_channel" and self.individual:
            raise ValueError("lift_sharing=lowrank_channel is incompatible with individual=1")
        self.lift_rank = int(_config(configs, "lift_rank", 2))
        self.norm_mode = str(_config(configs, "norm_mode", "rin_noaffine"))
        if self.norm_mode not in {"rin_noaffine", "revin_affine", "subtract_last", "none"}:
            raise ValueError(f"Unsupported norm_mode={self.norm_mode!r}")
        self.force_cross_off = _as_bool(_config(configs, "force_cross_off", 0))
        self.log_diagnostics = _as_bool(_config(configs, "log_asyspecx_diagnostics", 0))
        self._last_diagnostics: Dict[str, float] = {}
        if self.norm_mode == "revin_affine":
            self.revin_gamma = nn.Parameter(torch.ones(1, 1, self.channels))
            self.revin_beta = nn.Parameter(torch.zeros(1, 1, self.channels))
        else:
            self.register_parameter("revin_gamma", None)
            self.register_parameter("revin_beta", None)

        cut_freq = int(_config(configs, "cut_freq", 0))
        cut_freq = cut_freq if cut_freq > 0 else self.seq_len // 2 + 1
        self.dominance_freq = max(1, min(cut_freq, self.seq_len // 2 + 1))
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        self.total_bins = (self.seq_len + self.pred_len) // 2 + 1
        self.out_bins = max(
            1,
            min(int(math.floor(self.dominance_freq * self.length_ratio)), self.total_bins),
        )

        def _make_lift():
            if self.spectral_lift == "fits_linear":
                return ComplexLinear(self.dominance_freq, self.out_bins, bias=True)
            if self.spectral_lift == "complex_mlp":
                hidden = max(1, min(self.dominance_freq, self.out_bins))
                return ComplexMLP(self.dominance_freq, self.out_bins, hidden_features=hidden)
            raise ValueError(f"Unsupported spectral_lift={self.spectral_lift!r}")

        if self.lift_sharing == "lowrank_channel":
            if self.spectral_lift != "fits_linear":
                raise ValueError("lift_sharing=lowrank_channel currently requires spectral_lift=fits_linear")
            self.freq_upsampler = ChannelLowRankComplexLinear(
                self.dominance_freq,
                self.out_bins,
                channels=self.channels,
                rank=self.lift_rank,
                bias=True,
            )
            self.lift_rank = self.freq_upsampler.rank
        elif self.individual:
            self.freq_upsampler = nn.ModuleList([_make_lift() for _ in range(self.channels)])
        else:
            self.freq_upsampler = _make_lift()

        self.temporal_adapter_name = str(_config(configs, "temporal_adapter", "none"))
        if self.temporal_adapter_name == "none":
            self.temporal_adapter = None
        elif self.temporal_adapter_name in {"sparse_period", "compact_period"}:
            periods = parse_periods(
                _config(configs, "periods", ""),
                fallback_period=int(_config(configs, "period", 24)),
            )
            adapter_class = (
                SparsePeriodAdapter
                if self.temporal_adapter_name == "sparse_period"
                else CompactPeriodAdapter
            )
            self.temporal_adapter = adapter_class(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                channels=self.channels,
                periods=periods,
                periodic_init=str(_config(configs, "periodic_init", "seasonal_naive")),
                periodic_sharing=str(_config(configs, "periodic_sharing", "shared")),
                temporal_fusion=str(_config(configs, "temporal_fusion", "convex")),
                temporal_gate_type=str(_config(configs, "temporal_gate_type", "global")),
                temporal_gate_init_logit=float(_config(configs, "temporal_gate_init_logit", -4.0)),
                period_fusion=str(_config(configs, "period_fusion", "sum_gated")),
                period_gate_type=str(_config(configs, "period_gate_type", "period")),
                period_gate_init_logit=float(_config(configs, "period_gate_init_logit", 0.0)),
                periodic_l1_weight=float(_config(configs, "periodic_l1_weight", 0.0)),
                periodic_l2_weight=float(_config(configs, "periodic_l2_weight", 0.0)),
                temporal_gate_l1_weight=float(_config(configs, "temporal_gate_l1_weight", 0.0)),
            )
        else:
            raise ValueError(f"Unsupported temporal_adapter={self.temporal_adapter_name!r}")

        self.patch_adapter_name = str(_config(configs, "patch_adapter", "none"))
        if self.patch_adapter_name == "none":
            self.patch_adapter = None
        elif self.patch_adapter_name == "patch_linear":
            self.patch_adapter = PatchLinearAdapter(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                channels=self.channels,
                patch_len=int(_config(configs, "patch_len", 16)),
                patch_stride=int(_config(configs, "patch_stride", 8)),
                patch_basis_dim=int(_config(configs, "patch_basis_dim", 0)),
                patch_fusion=str(_config(configs, "patch_fusion", "convex")),
                patch_gate_type=str(_config(configs, "patch_gate_type", "horizon")),
                patch_gate_init_logit=float(_config(configs, "patch_gate_init_logit", -6.0)),
                patch_l1_weight=float(_config(configs, "patch_l1_weight", 0.0)),
                patch_l2_weight=float(_config(configs, "patch_l2_weight", 0.0)),
            )
        else:
            raise ValueError(f"Unsupported patch_adapter={self.patch_adapter_name!r}")

        self.linear_adapter_name = str(_config(configs, "linear_adapter", "none"))
        if self.linear_adapter_name == "none":
            self.linear_adapter = None
        else:
            self.linear_adapter = LinearAdapter(
                kind=self.linear_adapter_name,
                seq_len=self.seq_len, pred_len=self.pred_len, channels=self.channels,
                linear_sharing=str(_config(configs, "linear_sharing", "shared")),
                linear_init=str(_config(configs, "linear_init", "zeros")),
                individual_linear_max_channels=int(_config(configs, "individual_linear_max_channels", 64)),
                moving_avg_kernel=int(_config(configs, "moving_avg_kernel", 25)),
                multiscale_factors=parse_periods(_config(configs, "multiscale_factors", "1,2,4"), 1),
                multiscale_fusion=str(_config(configs, "multiscale_fusion", "softmax")),
                multiscale_gate_type=str(_config(configs, "multiscale_gate_type", "scale")),
                linear_fusion=str(_config(configs, "linear_fusion", "convex")),
                linear_gate_type=str(_config(configs, "linear_gate_type", "horizon")),
                linear_gate_init_logit=float(_config(configs, "linear_gate_init_logit", -6.0)),
                linear_l1_weight=float(_config(configs, "linear_l1_weight", 0.0)),
                linear_l2_weight=float(_config(configs, "linear_l2_weight", 0.0)),
            )

        # Phase 11: a strictly channel-separable past -> future kernel.  This
        # branch is intentionally independent of the spectral cross block: its
        # channel index is never contracted, so it cannot encode inter-variable
        # lead--lag.
        self.forecast_kernel_name = str(_config(configs, "forecast_kernel", "none"))
        if self.forecast_kernel_name == "none":
            self.forecast_kernel = None
        elif self.forecast_kernel_name == "lowrank_time":
            self.forecast_kernel = ForecastabilityAdapter(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                channels=self.channels,
                rank=int(_config(configs, "forecast_kernel_rank", 8)),
                init=str(_config(configs, "forecast_kernel_init", "small_random")),
                init_path=str(_config(configs, "forecast_kernel_init_path", "")),
                expected_data=str(_config(configs, "data", "")),
                expected_data_path=str(_config(configs, "data_path", "")),
                expected_norm_mode=self.norm_mode,
                channel_scale=_as_bool(_config(configs, "forecast_kernel_channel_scale", 0)),
                spectral_mixtures=int(
                    _config(configs, "forecast_kernel_spectral_mixtures", 0)
                ),
                sm_sharing=str(_config(configs, "forecast_kernel_sm_sharing", "mode")),
                sm_max_log_gain=float(
                    _config(configs, "forecast_kernel_sm_max_log_gain", 1.0)
                ),
                sm_min_width=float(
                    _config(configs, "forecast_kernel_sm_min_width", 0.02)
                ),
                sm_init_width=float(
                    _config(configs, "forecast_kernel_sm_init_width", 0.20)
                ),
                sm_center_max_shift=float(
                    _config(configs, "forecast_kernel_sm_center_max_shift", 0.15)
                ),
                sm_base_trainable=_as_bool(
                    _config(configs, "forecast_kernel_sm_base_trainable", 1)
                ),
                extension_shrink=str(
                    _config(configs, "forecast_kernel_extension_shrink", "none")
                ),
                phase_basis_dim=int(
                    _config(configs, "forecast_kernel_phase_basis_dim", 0)
                ),
                phase_max=float(
                    _config(configs, "forecast_kernel_phase_max", math.pi / 4.0)
                ),
                fusion=str(_config(configs, "forecast_kernel_fusion", "convex")),
                gate_type=str(_config(configs, "forecast_kernel_gate_type", "horizon")),
                gate_init_logit=float(_config(configs, "forecast_kernel_gate_init_logit", -6.0)),
                l1_weight=float(_config(configs, "forecast_kernel_l1_weight", 0.0)),
                l2_weight=float(_config(configs, "forecast_kernel_l2_weight", 0.0)),
            )
        else:
            raise ValueError(f"Unsupported forecast_kernel={self.forecast_kernel_name!r}")

        # Hydra parallel branch fusion (default sequential = Phase 7 behavior).
        self.branch_fusion_mode = str(_config(configs, "branch_fusion", "sequential"))
        if self.branch_fusion_mode not in {"sequential", "softmax_static"}:
            raise ValueError(f"Unsupported branch_fusion={self.branch_fusion_mode!r}")
        if self.branch_fusion_mode == "softmax_static":
            branch_names = ["spec"]
            if self.temporal_adapter is not None:
                branch_names.append("period")
            if self.patch_adapter is not None:
                branch_names.append("patch")
            if self.linear_adapter is not None:
                branch_names.append("linear")
            if self.forecast_kernel is not None:
                branch_names.append("forecast")
            self.branch_fusion = BranchFusion(
                branch_names, self.pred_len, self.channels,
                scope=str(_config(configs, "branch_fusion_scope", "horizon")),
                main_logit=float(_config(configs, "branch_init_main_logit", 4.0)),
                aux_logit=float(_config(configs, "branch_init_aux_logit", -4.0)),
            )
        else:
            self.branch_fusion = None

        self.cycle_residual_enabled = _as_bool(_config(configs, "cycle_residual", 0))
        if self.cycle_residual_enabled:
            self.cycle_residual = CycleResidual(
                cycle_len=int(_config(configs, "cycle", 24)),
                channels=self.channels,
                rank=int(_config(configs, "cycle_residual_rank", 0)),
                init_std=float(_config(configs, "cycle_residual_init_std", 0.02)),
            )
        else:
            self.cycle_residual = None

        cross_mode = str(_config(configs, "cross_mode", "asym_lowrank"))
        if cross_mode in {"asym", "hybrid"}:
            cross_mode = "asym_lowrank"
        self.cross_mode = cross_mode
        if self.cross_mode not in {"none", "asym_lowrank", "self_band_gain"}:
            raise ValueError(
                f"AsySpecX supports cross_mode none/asym_lowrank/self_band_gain, got {self.cross_mode!r}"
            )

        gate_init_logit = _config(configs, "gate_init_logit", None)
        if gate_init_logit is None:
            gate_init_logit = _config(configs, "gate_init", 0.0)

        residual_clip_eta = _config(configs, "residual_clip_eta", None)
        if residual_clip_eta is not None and float(residual_clip_eta) <= 0.0:
            residual_clip_eta = None

        residual_part = _config(configs, "residual_part", None)
        if residual_part == "":
            residual_part = None

        if self.cross_mode == "none":
            self.cross_block = None
        elif self.cross_mode == "asym_lowrank":
            self.cross_block = AsymCross(
                channels=self.channels,
                num_freqs=self.out_bins,
                rank=int(_config(configs, "rank", 8)),
                num_bands=int(_config(configs, "num_bands", 8)),
                gate_init_logit=float(gate_init_logit),
                gate_max=float(_config(configs, "gate_max", 1.0)),
                gate_type=str(_config(configs, "gate_type", "global")),
                mask_self_transfer=_config(configs, "mask_self_transfer", 0),
                residual_clip_eta=residual_clip_eta,
                skip_dc_cross=_config(configs, "skip_dc_cross", 1),
                residual_part=residual_part,
                energy_control=str(_config(configs, "energy_control", "none")),
                learned_clip_scope=str(_config(configs, "learned_clip_scope", "component_channel_band")),
                learned_clip_eta_init=float(_config(configs, "learned_clip_eta_init", 1.0)),
                learned_clip_eta_max=float(_config(configs, "learned_clip_eta_max", 2.0)),
            )
        else:
            self.cross_block = SelfBandGain(
                channels=self.channels,
                num_freqs=self.out_bins,
                num_bands=int(_config(configs, "num_bands", 8)),
                gate_init_logit=float(gate_init_logit),
                gate_max=float(_config(configs, "gate_max", 1.0)),
                gate_type=str(_config(configs, "gate_type", "global")),
                residual_clip_eta=residual_clip_eta,
                skip_dc_cross=_config(configs, "skip_dc_cross", 1),
                self_gain_init_std=float(_config(configs, "self_gain_init_std", 1e-3)),
            )

    def _spectral_lift(self, spec):
        if self.lift_sharing == "lowrank_channel":
            return self.freq_upsampler(spec.permute(0, 2, 1)).permute(0, 2, 1)
        if self.individual:
            lifted = torch.zeros(
                [spec.size(0), self.out_bins, spec.size(2)],
                dtype=spec.dtype,
                device=spec.device,
            )
            for i in range(self.channels):
                lifted[:, :, i] = self.freq_upsampler[i](spec[:, :, i])
            return lifted
        return self.freq_upsampler(spec.permute(0, 2, 1)).permute(0, 2, 1)

    def _normalize(self, x):
        eps = 1e-5
        if self.norm_mode == "rin_noaffine":
            loc = torch.mean(x, dim=1, keepdim=True)
            centered = x - loc
            scale = torch.sqrt(torch.var(centered, dim=1, keepdim=True) + eps)
            return centered / scale, {"loc": loc, "scale": scale, "mode": self.norm_mode}
        if self.norm_mode == "revin_affine":
            loc = torch.mean(x, dim=1, keepdim=True)
            centered = x - loc
            scale = torch.sqrt(torch.var(centered, dim=1, keepdim=True) + eps)
            x_norm = centered / scale
            return self.revin_gamma * x_norm + self.revin_beta, {"loc": loc, "scale": scale, "mode": self.norm_mode}
        if self.norm_mode == "subtract_last":
            loc = x[:, -1:, :]
            return x - loc, {"loc": loc, "scale": None, "mode": self.norm_mode}
        return x, {"loc": None, "scale": None, "mode": self.norm_mode}

    def _denormalize(self, y_norm, state):
        mode = state["mode"]
        if mode == "rin_noaffine":
            return y_norm * state["scale"] + state["loc"]
        if mode == "revin_affine":
            y_plain = (y_norm - self.revin_beta) / (self.revin_gamma + 1e-5)
            return y_plain * state["scale"] + state["loc"]
        if mode == "subtract_last":
            return y_norm + state["loc"]
        return y_norm

    @staticmethod
    def _low_freq_energy_ratio(spec_full, keep_bins):
        with torch.no_grad():
            energy = torch.abs(spec_full.detach()) ** 2
            kept = energy[:, :keep_bins, :].sum(dim=1)
            total = energy.sum(dim=1).clamp_min(1e-12)
            return float((kept / total).mean().cpu())

    def _force_real_edge_bins(self, spec):
        first = torch.complex(spec[:, :1, :].real, torch.zeros_like(spec[:, :1, :].real))
        spec = torch.cat([first, spec[:, 1:, :]], dim=1)
        if (self.seq_len + self.pred_len) % 2 == 0 and spec.size(1) > 1:
            last = torch.complex(spec[:, -1:, :].real, torch.zeros_like(spec[:, -1:, :].real))
            spec = torch.cat([spec[:, :-1, :], last], dim=1)
        return spec

    def forward(
        self,
        x,
        return_full=False,
        force_cross_off=None,
        eval_residual_part="default",
        cycle_index=None,
    ):
        # x: [B, T, C]
        x_norm, norm_state = self._normalize(x)

        cycle_future = cycle_full = None
        if self.cycle_residual is not None:
            cycle_in, cycle_future = self.cycle_residual.split(
                cycle_index, self.seq_len, self.pred_len
            )
            x_model = x_norm - cycle_in.to(x_norm.dtype)
            if return_full:
                cycle_full = self.cycle_residual.full(
                    cycle_index, self.seq_len, self.pred_len
                ).to(x_norm.dtype)
        else:
            x_model = x_norm

        spec_full = torch.fft.rfft(x_model, dim=1)
        low_freq_energy_ratio = self._low_freq_energy_ratio(spec_full, self.dominance_freq)
        spec = spec_full[:, : self.dominance_freq, :]
        lifted = self._spectral_lift(spec)

        if self.cross_block is not None:
            U = lifted.permute(0, 2, 1).contiguous()
            cross_off = self.force_cross_off if force_cross_off is None else _as_bool(force_cross_off)
            U = self.cross_block(U, force_off=cross_off, eval_residual_part=eval_residual_part)
            lifted = U.permute(0, 2, 1).contiguous()
            self._last_diagnostics = self.cross_block.get_diagnostics()
        else:
            self._last_diagnostics = {"cross_active": 0.0}
        self._last_diagnostics.update(
            {
                "cut_freq": float(self.dominance_freq),
                "F_in": float(self.dominance_freq),
                "F_out": float(self.out_bins),
                "K_out": float(self.total_bins),
                "low_freq_energy_ratio": low_freq_energy_ratio,
                "norm_mode": self.norm_mode,
                "lift_sharing": self.lift_sharing,
                "lift_rank": float(self.lift_rank if self.lift_sharing == "lowrank_channel" else 0),
                "temporal_adapter_enabled": 0.0,
                "forecast_kernel_enabled": float(self.forecast_kernel is not None),
                "forecast_kernel_rank": float(
                    self.forecast_kernel.rank if self.forecast_kernel is not None else 0
                ),
                "cycle_residual_enabled": float(self.cycle_residual is not None),
                "cycle_residual_rank": float(
                    self.cycle_residual.rank if self.cycle_residual is not None else 0
                ),
            }
        )

        if lifted.size(1) < self.total_bins:
            pad = lifted.new_zeros(lifted.size(0), self.total_bins - lifted.size(1), lifted.size(2))
            full_spec = torch.cat([lifted, pad], dim=1)
        else:
            full_spec = lifted[:, : self.total_bins, :]
        full_spec = self._force_real_edge_bins(full_spec)

        y_full_norm = torch.fft.irfft(full_spec, n=self.seq_len + self.pred_len, dim=1) * self.length_ratio
        pred_spec_norm = y_full_norm[:, -self.pred_len :, :]

        # Raw per-branch horizon predictions (normalized scale).
        period_raw = patch_raw = linear_raw = forecast_raw = None
        period_fused = None
        if self.temporal_adapter is not None:
            period_fused, period_raw = self.temporal_adapter(x_model, pred_spec_norm)
            self._last_diagnostics.update(self.temporal_adapter.get_diagnostics())
        if self.patch_adapter is not None:
            patch_raw = self.patch_adapter.raw(x_model)
        if self.linear_adapter is not None:
            linear_raw = self.linear_adapter.raw(x_model)
        if self.forecast_kernel is not None and self.branch_fusion is not None:
            forecast_raw = self.forecast_kernel.raw(x_model)

        if self.branch_fusion is not None:
            # Hydra: softmax mixture over available branches (all normalized scale).
            branch_preds = {"spec": pred_spec_norm}
            if period_raw is not None:
                branch_preds["period"] = period_raw
            if patch_raw is not None:
                branch_preds["patch"] = patch_raw
            if linear_raw is not None:
                branch_preds["linear"] = linear_raw
            if forecast_raw is not None:
                branch_preds["forecast"] = forecast_raw
            pred_norm = self.branch_fusion(branch_preds)
            self._last_diagnostics.update(self.branch_fusion.get_diagnostics())
        else:
            # Sequential fusion == exact Phase 7 behavior.
            pred_norm = pred_spec_norm
            if self.temporal_adapter is not None:
                pred_norm = period_fused
            if self.patch_adapter is not None:
                pred_norm = self.patch_adapter(x_model, pred_norm)
                self._last_diagnostics.update(self.patch_adapter.get_diagnostics())
            if self.linear_adapter is not None:
                pred_norm = self.linear_adapter(x_model, pred_norm)
                self._last_diagnostics.update(self.linear_adapter.get_diagnostics())
            if self.forecast_kernel is not None:
                pred_norm = self.forecast_kernel(x_model, pred_norm)
                self._last_diagnostics.update(self.forecast_kernel.get_diagnostics())

        if cycle_future is not None:
            pred_norm = pred_norm + cycle_future.to(pred_norm.dtype)
            if cycle_full is not None:
                y_full_norm = y_full_norm + cycle_full.to(y_full_norm.dtype)
        y_full = self._denormalize(y_full_norm, norm_state)
        pred = self._denormalize(pred_norm, norm_state)

        if self.log_diagnostics and self.training and self._last_diagnostics:
            diag = " ".join(
                f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in sorted(self._last_diagnostics.items())
            )
            print(f"[AsySpecX diagnostics] {diag}")

        if return_full:
            return {
                "pred": pred,
                "full": y_full,
                "backcast": y_full[:, : self.seq_len, :],
            }
        return pred

    def extra_loss(self):
        """Optional training-only regularization (periodic adapter L1/L2).

        Returns None when no adapter is active so the training loop can skip it;
        val/test metrics never call this, keeping them leakage-free.
        """
        total = None
        if self.temporal_adapter is not None and hasattr(self.temporal_adapter, "extra_loss"):
            total = self.temporal_adapter.extra_loss()
        if self.patch_adapter is not None and hasattr(self.patch_adapter, "extra_loss"):
            pl = self.patch_adapter.extra_loss()
            total = pl if total is None else total + pl
        if self.linear_adapter is not None and hasattr(self.linear_adapter, "extra_loss"):
            ll = self.linear_adapter.extra_loss()
            total = ll if total is None else total + ll
        if self.forecast_kernel is not None and hasattr(self.forecast_kernel, "extra_loss"):
            fl = self.forecast_kernel.extra_loss()
            total = fl if total is None else total + fl
        return total

    def get_diagnostics(self):
        return dict(self._last_diagnostics)
