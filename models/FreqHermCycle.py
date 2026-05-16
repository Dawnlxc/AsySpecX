"""FreqHermCycle — FreqHerm + CycleNet-style learnable cycle decomposition.

Pipeline:
    1. Subtract learned cycle pattern Q[t mod period, c] from input → residual r
    2. RIN normalize r
    3. rfft + in_bin lift + Hermitian low-rank cross + ifft (FreqHerm core on r)
    4. De-RIN → ŷ_residual
    5. Add cycle pattern Q[(t+H) mod period, c] for prediction window
    → ŷ = ŷ_residual + Q[future]

Cycle table is shared with CycleNet's interface (batch_cycle from data loader
indicates the cycle phase of the first input timestep).

Per-channel cycle: Q ∈ R^{period × C}, learned end-to-end.
Cross block operates on cycle-removed residual where:
    - DFT assumption (stationarity) is much closer to satisfied
    - Per-band g_b doesn't waste capacity modeling fixed periodicity

Differentiation from FreCycle:
    - Hermitian low-rank cross (vs dense)
    - Per-band complex g_b (vs uniform)
    - Probe-driven design rationale
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
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.cycle_len = int(configs.cycle)

        # Learnable cycle table — like CycleNet
        self.cycle_table = nn.Parameter(torch.zeros(self.cycle_len, self.channels))
        nn.init.normal_(self.cycle_table, std=0.01)

        full_F = self.seq_len // 2 + 1
        cut_freq = configs.cut_freq if configs.cut_freq > 0 else full_F
        self.dominance_freq = max(1, min(cut_freq, full_F))
        self.full_F = full_F

        self.in_bin = ComplexLinear(self.dominance_freq, self.dominance_freq)
        self.cross = HermitianLowRankCross(
            channels=self.channels,
            num_freqs=self.dominance_freq,
            rank=configs.rank,
            num_bands=configs.num_bands,
            gate_init=configs.gate_init,
            gate_max=configs.gate_max,
        )
        self.head = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, cycle_index):
        # x: [B, T, C]; cycle_index: [B] (start phase from data loader)
        B, T, C = x.shape

        # Cycle subtract: input
        cycle_in = self.cycle_table[
            (cycle_index.view(-1, 1) + torch.arange(T, device=x.device).view(1, -1)) % self.cycle_len
        ]                                                                    # [B, T, C]
        x_residual = x - cycle_in

        # RIN on residual
        x_mean = torch.mean(x_residual, dim=1, keepdim=True)
        x_centered = x_residual - x_mean
        x_var = torch.var(x_residual, dim=1, keepdim=True) + 1e-5
        x_norm = x_centered / torch.sqrt(x_var)

        # Frequency-domain processing on residual
        spec = torch.fft.rfft(x_norm, dim=1)[:, : self.dominance_freq, :]
        spec_t = self.in_bin(spec.permute(0, 2, 1)).permute(0, 2, 1)
        spec = spec_t                                                        # [B, K, C]
        spec = self.cross(spec)                                              # [B, K, C]

        # ifft, head
        full = torch.zeros([spec.size(0), self.full_F, spec.size(2)],
                           dtype=spec.dtype, device=spec.device)
        full[:, : spec.size(1), :] = spec
        h = torch.fft.irfft(full, n=self.seq_len, dim=1)                     # [B, T, C]
        y_residual = self.head(h.permute(0, 2, 1)).permute(0, 2, 1)          # [B, pred_len, C]

        # De-RIN
        y_residual = y_residual * torch.sqrt(x_var) + x_mean

        # Cycle add: output
        cycle_out = self.cycle_table[
            (cycle_index.view(-1, 1) + (T + torch.arange(self.pred_len, device=x.device)).view(1, -1)) % self.cycle_len
        ]                                                                    # [B, pred_len, C]
        y = y_residual + cycle_out
        return y
