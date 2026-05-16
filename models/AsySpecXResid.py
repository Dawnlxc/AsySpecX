"""AsySpecXResid — residual 3-block frequency-domain forecaster.

Architecture motivated by probing findings:
    Block 1 (Self):  FITS-style per-channel frequency interpolation. Carries the
                     bulk of the signal on standard MTSF where self-prediction
                     dominates.
    Block 2 (Sym):   Hermitian low-rank cross block applied to the residual
                     r = x - Ŷ_self_seq. Captures bidirectional cross-channel
                     coupling (the dominant cross structure per the sym ceiling).
    Block 3 (Asym):  Optional sparse asymmetric top-K residual head. Requires an
                     offline lag profile (top-K driver per target). Deferred —
                     gated off by default; activate via configs.use_asym_topk
                     and a precomputed adjacency.

Output: ŷ = Ŷ_self + g_sym · Ŷ_sym  (+ g_asym · Ŷ_asym  if Block 3 enabled).

configs fields used: seq_len, pred_len, enc_in, cut_freq, rank, num_bands,
                     gate_init, gate_max.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    """Complex-weighted linear layer over the last dim."""
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


class HermitianLowRankCross(nn.Module):
    """H_b = U · diag(g_b) · U^T  applied as a residual contribution.

    Returns the cross *contribution* (without the input identity), so the parent
    module can place it on top of the self-prediction explicitly.
    """
    def __init__(self, channels, num_freqs, rank, num_bands, gate_init=0.0, gate_max=1.0):
        super().__init__()
        self.channels = int(channels)
        self.num_freqs = int(num_freqs)
        self.rank = max(1, min(int(rank), self.channels))
        self.num_bands = max(1, min(int(num_bands), self.num_freqs))

        self.U = nn.Parameter(torch.empty(self.channels, self.rank))
        nn.init.orthogonal_(self.U)
        self.g_re = nn.Parameter(torch.empty(self.num_bands, self.rank))
        self.g_im = nn.Parameter(torch.empty(self.num_bands, self.rank))
        nn.init.normal_(self.g_re, std=0.02)
        nn.init.normal_(self.g_im, std=0.02)

        edges = torch.linspace(0, self.num_bands, self.num_freqs + 1)[:-1]
        band_ids = torch.clamp(edges.long(), 0, self.num_bands - 1)
        self.register_buffer("band_ids", band_ids)

        self.gate_max = float(gate_max)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, spec):
        # spec: complex [B, F, C] → returns cross *contribution* [B, F, C]
        U = self.U.to(spec.dtype)
        latent = spec @ U                                             # [B, F, r]
        g = torch.complex(self.g_re, self.g_im)[self.band_ids]        # [F, r]
        latent = latent * g.unsqueeze(0).to(spec.dtype)
        contrib = latent @ U.transpose(-2, -1)                        # [B, F, C]
        gate = self.gate_max * torch.sigmoid(self.gate_logit)
        return gate.to(spec.dtype) * contrib


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # FITS-style cut: dominance_freq bins on input, scaled out_dim on output
        cut = getattr(configs, 'cut_freq', 0)
        self.dominance_freq = cut if cut and cut > 0 else self.seq_len // 4 + 1
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        self.out_dim = int(self.dominance_freq * self.length_ratio)
        self.full_out_F = (self.seq_len + self.pred_len) // 2 + 1

        # Block 1: per-channel freq upsampler (FITS body)
        self.self_upsampler = ComplexLinear(self.dominance_freq, self.out_dim)
        # Block 2: Hermitian cross on the in-band residual spectrum
        self.cross = HermitianLowRankCross(
            channels=self.channels,
            num_freqs=self.dominance_freq,
            rank=getattr(configs, 'rank', 8),
            num_bands=getattr(configs, 'num_bands', 8),
            gate_init=getattr(configs, 'gate_init', -2.0),  # start small — let model open up
            gate_max=getattr(configs, 'gate_max', 1.0),
        )
        # Block 2 output is at length K (dominance_freq); upsample to K' for the same horizon
        self.cross_upsampler = ComplexLinear(self.dominance_freq, self.out_dim)

    def _to_full_freq(self, spec_low):
        """Zero-pad [B, out_dim, C] complex to [B, full_out_F, C]."""
        out = torch.zeros(
            spec_low.size(0), self.full_out_F, spec_low.size(2),
            dtype=spec_low.dtype, device=spec_low.device,
        )
        n = min(spec_low.size(1), self.full_out_F)
        out[:, :n, :] = spec_low[:, :n, :]
        return out

    def forward(self, x):
        # x: [B, T, C]
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        # rFFT and low-pass
        spec_full = torch.fft.rfft(x, dim=1)                          # [B, S//2+1, C]
        spec_low = spec_full[:, :self.dominance_freq, :]              # [B, K, C]

        # ----- Block 1: per-channel self predictor (FITS) -----
        s = self.self_upsampler(spec_low.permute(0, 2, 1)).permute(0, 2, 1)  # [B, K', C]
        self_full = self._to_full_freq(s)
        y_self_time = torch.fft.irfft(self_full, dim=1) * self.length_ratio   # [B, S+P, C]
        # Self-prediction at seq portion (used to form the residual for Block 2)
        y_self_seq = y_self_time[:, :self.seq_len, :]
        y_self_pred = y_self_time[:, -self.pred_len:, :]

        # ----- Block 2: Hermitian cross on residual spectrum -----
        resid_seq = x - y_self_seq                                    # [B, S, C]
        resid_spec = torch.fft.rfft(resid_seq, dim=1)[:, :self.dominance_freq, :]  # [B, K, C]
        cross_low = self.cross(resid_spec)                            # [B, K, C] (already gated)
        # Upsample cross contribution to forecast horizon
        c = self.cross_upsampler(cross_low.permute(0, 2, 1)).permute(0, 2, 1)  # [B, K', C]
        cross_full = self._to_full_freq(c)
        y_cross_time = torch.fft.irfft(cross_full, dim=1) * self.length_ratio
        y_cross_pred = y_cross_time[:, -self.pred_len:, :]

        # ----- Sum + de-RIN -----
        y = y_self_pred + y_cross_pred
        y = y * torch.sqrt(x_var) + x_mean
        return y
