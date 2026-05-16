"""FreqHerm — FreqMLP backbone + Hermitian low-rank cross-channel block.

Combines:
    - FreqMLP architecture (no spectral interpolation; time-domain MLP head)
    - Hermitian low-rank cross block: H_b = U diag(g_b) U^T  (A = B in SVD form)

The Hermitian sym constraint matches the empirical finding (Granger probe + sym/asym
ablation) that 75-80% of cross-channel "helpful" pairs are bidirectional. Per-band
g_b lets each frequency band have its own coupling profile while sharing channel
embeddings U across bands.

Pipeline:
    1. RIN normalize
    2. rfft + optional cut_freq truncation
    3. ComplexLinear in-channel: F -> F (per-channel, shared across C)
    4. HermitianLowRankCross: H_b = U diag(g_b) U^T  with rank r << C, gated
    5. Zero-pad if cut_freq, irfft -> seq_len
    6. Real Linear head: seq_len -> pred_len, per-channel-shared
    7. De-RIN

Selling points:
    - Param count for cross: O(C·r + r·B) instead of O(C²) per band  → 100-400× fewer
    - U is interpretable as channel embedding in r-dim latent space
    - g_b reveals which freq bands carry strong cross-coupling
    - Gate auto-disables cross on datasets where it doesn't help

`configs` fields used: seq_len, pred_len, enc_in, cut_freq, rank, num_bands, gate_init, gate_max.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
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
    """Hermitian low-rank cross-channel residual block.

        H_b U_in = U · diag(g_b) · U^T · U_in,   gated residual sum.

    A = B = U (single channel embedding). Per-band complex gain g_b.
    Frequency bins partitioned into num_bands contiguous bands.
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
        # spec: complex [B, F, C]
        U = self.U.to(spec.dtype)
        latent = spec @ U                                         # [B, F, r]
        g = torch.complex(self.g_re, self.g_im)[self.band_ids]    # [F, r]
        latent = latent * g.unsqueeze(0).to(spec.dtype)           # [B, F, r]
        out = latent @ U.transpose(-2, -1)                        # [B, F, C]
        gate = self.gate_max * torch.sigmoid(self.gate_logit)
        return spec + gate.to(spec.dtype) * out


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        full_F = self.seq_len // 2 + 1
        cut_freq = configs.cut_freq if configs.cut_freq > 0 else full_F
        self.dominance_freq = max(1, min(cut_freq, full_F))
        self.full_F = full_F

        # Frequency-domain stages: bin-preserving, no interpolation.
        self.in_bin = ComplexLinear(self.dominance_freq, self.dominance_freq)
        self.cross = HermitianLowRankCross(
            channels=self.channels,
            num_freqs=self.dominance_freq,
            rank=configs.rank,
            num_bands=configs.num_bands,
            gate_init=configs.gate_init,
            gate_max=configs.gate_max,
        )

        # Time-domain head.
        self.head = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [B, T, C]
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        spec = torch.fft.rfft(x, dim=1)[:, : self.dominance_freq, :]    # [B, K, C]

        # In-channel complex linear over bins (per-channel, shared across C).
        spec_t = spec.permute(0, 2, 1)                                  # [B, C, K]
        spec_t = self.in_bin(spec_t)                                    # [B, C, K]
        spec = spec_t.permute(0, 2, 1)                                  # [B, K, C]

        # Hermitian low-rank cross-channel.
        spec = self.cross(spec)                                         # [B, K, C]

        # Zero-pad to full rfft length, ifft to seq_len.
        full = torch.zeros(
            [spec.size(0), self.full_F, spec.size(2)],
            dtype=spec.dtype, device=spec.device,
        )
        full[:, : spec.size(1), :] = spec
        h = torch.fft.irfft(full, n=self.seq_len, dim=1)                # [B, seq_len, C]

        # Time-domain head: project seq_len -> pred_len, per-channel-shared.
        y = self.head(h.permute(0, 2, 1)).permute(0, 2, 1)              # [B, pred_len, C]

        y = y * torch.sqrt(x_var) + x_mean
        return y
