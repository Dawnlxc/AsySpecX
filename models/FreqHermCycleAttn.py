"""FreqHermCycleAttn — FreqHermCycle with TQNet-style cycle-conditioned cross-channel attention.

Pipeline:
    1. Subtract learnable cycle Q[t mod period, c] from input → residual r [B,T,C]
    2. RIN normalize r
    3. Frequency-domain Hermitian cross block on r → cross-corrected residual
    4. **NEW**: cycle-conditioned channel attention: query = future cycle prior,
       key/value = current channel embeddings; the output is added as a residual to the
       cross block output, capturing per-sample channel coupling adapted to where in the
       cycle this batch is positioned.
    5. Time-domain head, de-RIN
    6. Add cycle pattern to forecast horizon

This combines the strengths:
    - FreqHerm: explicit frequency-domain cross structure (interpretable)
    - CycleNet cycle decomp: removes daily/weekly periodic structure
    - TQNet cycle-conditioned attention: per-sample adaptation

configs: seq_len, pred_len, enc_in, cycle, cut_freq, rank, num_bands, gate_init, gate_max,
         use_cycle_attn (default 1), n_attn_heads (default 4), attn_dropout (default 0.1)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.wr = nn.Parameter(torch.empty(out_features, in_features))
        self.wi = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.br = nn.Parameter(torch.zeros(out_features))
            self.bi = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("br", None)
            self.register_parameter("bi", None)
        nn.init.xavier_uniform_(self.wr); nn.init.xavier_uniform_(self.wi)
        with torch.no_grad():
            scale = 1.0 / math.sqrt(2.0); self.wr.mul_(scale); self.wi.mul_(scale)

    def forward(self, x):
        xr, xi = x.real, x.imag
        yr = F.linear(xr, self.wr, self.br) - F.linear(xi, self.wi, None)
        yi = F.linear(xr, self.wi, self.bi) + F.linear(xi, self.wr, None)
        return torch.complex(yr, yi)


class HermitianLowRankCross(nn.Module):
    def __init__(self, channels, num_freqs, rank, num_bands, gate_init=0.0, gate_max=1.0):
        super().__init__()
        self.C = int(channels); self.K = int(num_freqs)
        self.r = max(1, min(int(rank), self.C))
        self.B = max(1, min(int(num_bands), self.K))
        self.U = nn.Parameter(torch.empty(self.C, self.r))
        nn.init.orthogonal_(self.U)
        self.g_re = nn.Parameter(torch.empty(self.B, self.r))
        self.g_im = nn.Parameter(torch.empty(self.B, self.r))
        nn.init.normal_(self.g_re, std=0.02); nn.init.normal_(self.g_im, std=0.02)
        edges = torch.linspace(0, self.B, self.K + 1)[:-1]
        self.register_buffer("band_ids", torch.clamp(edges.long(), 0, self.B - 1))
        self.gate_max = float(gate_max)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, spec):
        U = self.U.to(spec.dtype)
        latent = spec @ U
        g = torch.complex(self.g_re, self.g_im)[self.band_ids]
        latent = latent * g.unsqueeze(0).to(spec.dtype)
        out = latent @ U.transpose(-2, -1)
        gate = self.gate_max * torch.sigmoid(self.gate_logit)
        return spec + gate.to(spec.dtype) * out


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.cycle_len = int(configs.cycle)
        self.use_revin = bool(getattr(configs, 'use_revin', 1))

        # Learnable cycle table
        self.cycle_table = nn.Parameter(torch.zeros(self.cycle_len, self.channels))
        nn.init.normal_(self.cycle_table, std=0.01)

        full_F = self.seq_len // 2 + 1
        cut_freq = getattr(configs, 'cut_freq', 0)
        self.dominance_freq = max(1, min(cut_freq if cut_freq > 0 else full_F, full_F))
        self.full_F = full_F

        # Frequency-domain Hermitian cross block (on residual after cycle removal)
        self.in_bin = ComplexLinear(self.dominance_freq, self.dominance_freq)
        self.cross = HermitianLowRankCross(
            channels=self.channels,
            num_freqs=self.dominance_freq,
            rank=int(getattr(configs, 'rank', 8)),
            num_bands=int(getattr(configs, 'num_bands', 8)),
            gate_init=float(getattr(configs, 'gate_init', 0.0)),
            gate_max=float(getattr(configs, 'gate_max', 1.0)),
        )

        # Cycle-conditioned channel attention
        self.use_cycle_attn = bool(getattr(configs, 'use_cycle_attn', 1))
        if self.use_cycle_attn:
            n_heads = int(getattr(configs, 'n_attn_heads', 4))
            attn_dropout = float(getattr(configs, 'attn_dropout', 0.1))
            self.attn = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=n_heads,
                                              batch_first=True, dropout=attn_dropout)
            self.attn_gate = nn.Parameter(torch.tensor(-2.0))   # sigmoid → ~0.12 init

        # Time-domain head (residual → pred_len)
        self.head = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, cycle_index):
        # x: [B, T, C]; cycle_index: [B]
        B, T, C = x.shape

        # Cycle removal
        cycle_in = self.cycle_table[
            (cycle_index.view(-1, 1) + torch.arange(T, device=x.device).view(1, -1)) % self.cycle_len
        ]                                                                # [B, T, C]
        x_residual = x - cycle_in

        # RIN on residual
        x_mean = x_residual.mean(dim=1, keepdim=True)
        x_var = x_residual.var(dim=1, keepdim=True) + 1e-5
        x_norm = (x_residual - x_mean) / torch.sqrt(x_var)

        # Freq-domain Hermitian cross
        spec = torch.fft.rfft(x_norm, dim=1)[:, :self.dominance_freq, :]
        spec_t = self.in_bin(spec.permute(0, 2, 1)).permute(0, 2, 1)
        spec_t = self.cross(spec_t)                                       # [B, K, C]
        full = torch.zeros(spec_t.size(0), self.full_F, spec_t.size(2),
                           dtype=spec_t.dtype, device=spec_t.device)
        full[:, :spec_t.size(1), :] = spec_t
        h_freq = torch.fft.irfft(full, n=self.seq_len, dim=1)             # [B, T, C]

        # Cycle-conditioned channel attention (TQNet-style)
        if self.use_cycle_attn:
            x_input = x_norm.permute(0, 2, 1)                             # [B, C, T]
            cycle_query = cycle_in.permute(0, 2, 1)                       # [B, C, T]
            attn_out, _ = self.attn(query=cycle_query, key=x_input, value=x_input)
            attn_out = attn_out.permute(0, 2, 1)                          # [B, T, C]
            gate = torch.sigmoid(self.attn_gate)
            h_combined = h_freq + gate * attn_out
        else:
            h_combined = h_freq

        # Project to pred_len
        y_residual = self.head(h_combined.permute(0, 2, 1)).permute(0, 2, 1)

        # De-RIN
        y_residual = y_residual * torch.sqrt(x_var) + x_mean

        # Add cycle for forecast horizon
        cycle_out = self.cycle_table[
            (cycle_index.view(-1, 1) + (T + torch.arange(self.pred_len, device=x.device)).view(1, -1)) % self.cycle_len
        ]
        return y_residual + cycle_out
