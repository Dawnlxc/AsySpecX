"""AsySpecX — asymmetric low-rank spectral transfer for time-series forecasting.

Pipeline:
    1. RIN normalize input
    2. rfft(time) and low-pass cut to `cut_freq` bins
    3. Per-channel ComplexMLP lift: in_bins -> out_bins  (out_bins covers seq+pred)
    4. AsymCross residual: U <- U + gate * (A diag(g_m(f)) B^T) U
    5. Zero-pad to (seq+pred)/2+1, irfft, scale by length_ratio
    6. De-RIN, take last pred_len steps

The asymmetric cross block carries the contribution: with A != B and complex
per-band gain g_m, the learned channel-routing matrix H_m = A diag(g_m) B^T is
generically non-Hermitian, so it can model directional dependencies that any
symmetric attention (iTransformer, Crossformer, FreTS) cannot represent.

`configs` fields used:
    seq_len, pred_len, enc_in, cut_freq, individual,
    rank, num_bands, gate_init, gate_max
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    """Complex Linear via two real Parameter blocks. Xavier init scaled by
    1/sqrt(2) so that |W| has unit-variance gain like a real Linear."""

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


class ModReLUBins(nn.Module):
    """Per-frequency-bin phase-preserving complex activation:
        z -> z * relu(|z| + bias) / (|z| + eps)
    Identity at init when bias=0."""

    def __init__(self, num_freqs, init_bias=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.full((int(num_freqs),), float(init_bias)))

    def forward(self, z):
        mag = z.abs()
        bias = self.bias.view(*([1] * (z.dim() - 1)), -1)
        scale = torch.relu(mag + bias) / (mag + 1e-6)
        return z * scale.to(z.dtype)


class ComplexMLP(nn.Module):
    """Two-layer phase-preserving complex MLP for the per-channel spectral lift:
        y = W2 . ModReLU( W1 z + b1 ) + b2
    """

    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.fc1 = ComplexLinear(in_features, hidden_features, bias=True)
        self.act = ModReLUBins(hidden_features, init_bias=0.0)
        self.fc2 = ComplexLinear(hidden_features, out_features, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AsymCross(nn.Module):
    """Asymmetric low-rank cross-channel residual.

        Z = U + gate * (A diag(g_m(f)) B^T) U

    Shapes:
        A, B : R^{C x r}                 independent (orthogonal init) -> H asymmetric
        g    : C^{num_bands x r}          per-band complex gain
        gate : scalar in [0, gate_max]    gate = gate_max * sigmoid(gate_logit)

    Frequency bins are partitioned into `num_bands` contiguous bands; band b is
    routed by g_m at index b.
    """

    def __init__(self, channels, num_freqs, rank, num_bands, gate_init, gate_max):
        super().__init__()
        self.channels = int(channels)
        self.num_freqs = int(num_freqs)
        self.rank = max(1, min(int(rank), self.channels))
        self.num_bands = max(1, min(int(num_bands), self.num_freqs))

        self.A = nn.Parameter(torch.empty(self.channels, self.rank))
        self.B = nn.Parameter(torch.empty(self.channels, self.rank))
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.B)

        self.g_re = nn.Parameter(torch.empty(self.num_bands, self.rank))
        self.g_im = nn.Parameter(torch.empty(self.num_bands, self.rank))
        nn.init.normal_(self.g_re, std=0.02)
        nn.init.normal_(self.g_im, std=0.02)

        edges = torch.linspace(0, self.num_bands, self.num_freqs + 1)[:-1]
        band_ids = torch.clamp(edges.long(), 0, self.num_bands - 1)
        self.register_buffer("band_ids", band_ids)

        self.gate_max = float(gate_max)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, U):
        # U: complex [B, C, F]
        S = torch.einsum("cr,bcf->brf", self.B.to(U.dtype), U)         # [B, r, F]
        g = torch.complex(self.g_re, self.g_im)[self.band_ids]          # [F, r]
        Sg = S * g.transpose(0, 1).unsqueeze(0).to(U.dtype)             # [B, r, F]
        R = torch.einsum("cr,brf->bcf", self.A.to(U.dtype), Sg)         # [B, C, F]
        gate = self.gate_max * torch.sigmoid(self.gate_logit)
        return U + gate.to(U.dtype) * R


class Model(nn.Module):
    """AsySpecX forecaster."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in

        # Low-pass cutoff (FITS convention: configs.cut_freq=0 -> use seq_len // 2 + 1)
        cut_freq = configs.cut_freq if configs.cut_freq > 0 else self.seq_len // 2 + 1
        self.dominance_freq = max(1, min(cut_freq, self.seq_len // 2 + 1))
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        out_bins = max(1, int(self.dominance_freq * self.length_ratio))
        self.out_bins = out_bins

        def _make_lift():
            hidden = min(self.dominance_freq, out_bins)
            return ComplexMLP(self.dominance_freq, out_bins, hidden_features=hidden)

        if self.individual:
            self.freq_upsampler = nn.ModuleList([_make_lift() for _ in range(self.channels)])
        else:
            self.freq_upsampler = _make_lift()

        self.cross_block = AsymCross(
            channels=self.channels,
            num_freqs=out_bins,
            rank=configs.rank,
            num_bands=configs.num_bands,
            gate_init=configs.gate_init,
            gate_max=configs.gate_max,
        )

    def forward(self, x):
        # x: [B, T, C]
        # RIN normalize
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        # rfft + low-pass cut
        spec = torch.fft.rfft(x, dim=1)[:, : self.dominance_freq, :]

        # Per-channel ComplexMLP lift in the frequency domain
        if self.individual:
            lifted = torch.zeros(
                [spec.size(0), self.out_bins, spec.size(2)],
                dtype=spec.dtype, device=spec.device,
            )
            for i in range(self.channels):
                lifted[:, :, i] = self.freq_upsampler[i](spec[:, :, i])
        else:
            lifted = self.freq_upsampler(spec.permute(0, 2, 1)).permute(0, 2, 1)

        # AsymCross residual
        U = lifted.permute(0, 2, 1).contiguous()        # [B, C, out_bins]
        U = self.cross_block(U)
        lifted = U.permute(0, 2, 1).contiguous()        # [B, out_bins, C]

        # Zero-pad to full output spectrum, irfft back to time, scale
        total_bins = int((self.seq_len + self.pred_len) / 2 + 1)
        full_spec = torch.zeros(
            [lifted.size(0), total_bins, lifted.size(2)],
            dtype=lifted.dtype, device=lifted.device,
        )
        full_spec[:, : lifted.size(1), :] = lifted

        y = torch.fft.irfft(full_spec, dim=1) * self.length_ratio

        # De-RIN, return only the prediction segment
        y = y * torch.sqrt(x_var) + x_mean
        return y[:, -self.pred_len:, :]
