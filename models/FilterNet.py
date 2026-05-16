"""FilterNet — Harnessing Frequency Filters for Time Series Forecasting.

Ported from aikunyi/FilterNet official repo (NeurIPS 2024). Wraps PaiFilter
(low-channel datasets) and TexFilter (high-channel datasets); switch via
configs.model_variant. forward() takes a single tensor x [B, L, C] for
compatibility with the shared exp loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


class _PaiFilter(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size

        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        return torch.fft.irfft(y, n=self.embed_size, dim=2, norm='ortho')

    def forward(self, x):
        x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)
        x = self.circular_convolution(x, self.w.to(x.device))
        x = self.fc(x).permute(0, 2, 1)
        x = self.revin_layer(x, 'denorm')
        return x


class _TexFilter(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.scale = 0.02
        self.sparsity_threshold = 0.01

        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.embedding = nn.Linear(self.seq_len, self.embed_size)

        self.w = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )
        self.output = nn.Linear(self.embed_size, self.pred_len)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.layernorm1 = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(configs.dropout)

    def texfilter(self, x):
        o1_real = F.relu(
            torch.einsum('bid,d->bid', x.real, self.w[0])
            - torch.einsum('bid,d->bid', x.imag, self.w[1])
            + self.rb1
        )
        o1_imag = F.relu(
            torch.einsum('bid,d->bid', x.imag, self.w[0])
            + torch.einsum('bid,d->bid', x.real, self.w[1])
            + self.ib1
        )
        o2_real = (
            torch.einsum('bid,d->bid', o1_real, self.w1[0])
            - torch.einsum('bid,d->bid', o1_imag, self.w1[1])
            + self.rb2
        )
        o2_imag = (
            torch.einsum('bid,d->bid', o1_imag, self.w1[0])
            + torch.einsum('bid,d->bid', o1_real, self.w1[1])
            + self.ib2
        )
        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        return torch.view_as_complex(y)

    def forward(self, x):
        B, L, N = x.shape
        x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.layernorm(x)
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = self.texfilter(x)
        x = x * weight
        x = torch.fft.irfft(x, n=N, dim=1, norm='ortho')
        x = self.layernorm1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x).permute(0, 2, 1)
        x = self.revin_layer(x, 'denorm')
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        variant = getattr(configs, 'model_variant', 'PaiFilter')
        if variant == 'PaiFilter':
            self.impl = _PaiFilter(configs)
        elif variant == 'TexFilter':
            self.impl = _TexFilter(configs)
        else:
            raise ValueError(f"FilterNet model_variant must be PaiFilter or TexFilter, got {variant!r}")

    def forward(self, x):
        return self.impl(x)
