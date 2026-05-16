"""FITS — Frequency Interpolation Time Series.

Ported from VEWOXIC/FITS official repo (ICLR 2024 Spotlight). Returns a single
tensor [B, pred_len, C]; the original implementation returned a tuple, which is
incompatible with the shared exp loop in this repo.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in

        # cut_freq=0 → fall back to seq_len // 4 + 1 (FITS default heuristic)
        self.dominance_freq = configs.cut_freq if configs.cut_freq > 0 else self.seq_len // 4 + 1
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        out_dim = int(self.dominance_freq * self.length_ratio)
        if self.individual:
            self.freq_upsampler = nn.ModuleList([
                nn.Linear(self.dominance_freq, out_dim).to(torch.cfloat)
                for _ in range(self.channels)
            ])
        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, out_dim).to(torch.cfloat)

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx = low_specx[:, 0:self.dominance_freq, :]

        out_dim = int(self.dominance_freq * self.length_ratio)
        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), out_dim, low_specx.size(2)],
                dtype=low_specx.dtype, device=low_specx.device,
            )
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i])
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)

        low_specxy = torch.zeros(
            [low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)],
            dtype=low_specxy_.dtype, device=low_specxy_.device,
        )
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_

        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio

        xy = low_xy * torch.sqrt(x_var) + x_mean
        return xy[:, -self.pred_len:, :]
