"""FreTS — Frequency-domain MLPs for Time Series Forecasting.

Ported from aikunyi/FreTS official repo (NeurIPS 2023). Verbatim model body;
configs.channel_independence is read as the string '0'/'1' as in the original.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in
        self.seq_length = configs.seq_len
        self.channel_independence = str(configs.channel_independence)
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length),
        )

    def tokenEmb(self, x):
        x = x.permute(0, 2, 1).unsqueeze(3)
        return x * self.embeddings

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r)
            - torch.einsum('bijd,dd->bijd', x.imag, i)
            + rb
        )
        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r)
            + torch.einsum('bijd,dd->bijd', x.real, i)
            + ib
        )
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        return torch.view_as_complex(y)

    def MLP_temporal(self, x, B, N, L):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        return torch.fft.irfft(y, n=self.seq_length, dim=2, norm='ortho')

    def MLP_channel(self, x, B, N, L):
        x = x.permute(0, 2, 1, 3)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm='ortho')
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        B, T, N = x.shape
        x = self.tokenEmb(x)
        bias = x
        if self.channel_independence == '1':
            x = self.MLP_channel(x, B, N, T)
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x
