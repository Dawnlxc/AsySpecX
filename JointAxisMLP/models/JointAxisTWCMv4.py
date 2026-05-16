"""JointAxisTWCMv4 — completes Axis 2 by making g_{k, t'} per-frame.

v3 had per-(sample, freq_bin) gain g_k(x) that was *invariant across STFT
frames* — captures Axis 2 (frequency) but NOT Axis 2 (time-within-window).
v4 adds the missing dimension: g now varies per STFT frame t', enabling
non-stationary cross-channel mixing within the window — the same dimension
TQNet captures via temporalQuery[cycle_index + t].

Init: g_base[f, t', r] replicated from v3's [f, r] across t' → initially
equivalent to v3, learns t'-variation only where data drives it.

Original v3 docstring follows.
---
JointAxisTWCMv3 — per-bin g + per-bin learnable gate (no num_bands hyperparam).

Step-by-step rationale:
  v1 (JointAxisTWCM):     g shared across freq bins inside each of M=8 bands.
                          → Band-0 (k=1..6 depending on W) lumps slow trend +
                            daily cycle, ECL hurt +5.7 %.
  v2 (JointAxisTWCMv2):   per-band gate (low band suppressed) — partial fix,
                          but a band's physical meaning still depends on W.
  v3 (this file):         drop bands entirely.  g_base shape becomes [F, R];
                          freq_gate ∈ R^F.  Each g_k has a stable physical
                          meaning — frequency = k / W — regardless of seq_len.
                          Capacity self-adapts to W.  No num_bands hyper.

Empirical motivation (band_analysis.py):
  - For sl=96, energy at k=1 (period = W): ECL 91 %, Traffic 57 %, PEMS 17 %.
  - Per-bin gate lets the model surgically zero out k=1 on ECL/traffic while
    leaving it open on PEMS.
---
JointAxisTWCM — single-forward-pass joint relaxation of three stationarity axes.

The base cross-channel operator is

    H_{b, t', m, c, j} = Σ_r A_{c, r} · g^{(m)}_{r}(x_b) · B_{j, r}

with three axes simultaneously activated:

    axis 1 (Sample):  g depends on per-sample spec features through delta_mlp
                      g(x_b) = g_base + δ_mlp(spec(x_b))
    axis 2 (Time):    STFT decomposition causes each frame t' to see different
                      local content, even though g is t'-independent.
    axis 3 (Channel): low-rank A·B factorisation of cross structure.

Unlike sequential composition (TWCM-on-TQNet etc.), all three axes share the
**same low-rank latent space (r dim)**. The three axes are not independent
operators stacked; they are factors of a single H tensor parameterised through
one rank-r bottleneck. This avoids inter-operator gradient conflict that we
observed empirically (TWCM-on-TQNet, +9-10% on traffic/electricity).

Reduces to STFTLocalCross when delta_mlp output is identically zero
(controlled by zero-init of delta_mlp's final layer).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _moving_average_(x, kernel):
    T = x.size(-1)
    k = min(int(kernel), T)
    if k % 2 == 0:
        k -= 1
    if k < 3:
        return x
    pad = k // 2
    return F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"),
                        kernel_size=k, stride=1)


def _band_ids_(num_freqs: int, num_bands: int, device=None) -> torch.Tensor:
    idx = torch.arange(num_freqs, device=device)
    bid = torch.div(idx * num_bands, num_freqs, rounding_mode="floor")
    return bid.clamp(0, num_bands - 1).long()


def compute_sample_spec_features(
    x_tok: torch.Tensor,             # [B, C, T]
    num_bands: int = 8,
    smooth_kernel: int = 25,
    exclude_dc: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-sample spectral features for the axis-1 modulator.

    Returns [B, D] real tensor with:
        - mean band-energy fraction (over channels)        [M]
        - std  band-energy fraction (over channels)         [M]
        - mean per-channel entropy                          [1]
        - std  per-channel entropy                          [1]
        - drift (early vs late half)                        [M]
        - early energy ratio (low/mid/high)                 [3]
      Total D = 3M + 5

    Detached: no gradient back into the input.
    """
    with torch.no_grad():
        B, C, T = x_tok.shape
        # innovation
        smooth = _moving_average_(x_tok, smooth_kernel)
        e = x_tok - smooth
        # global per-channel spectrum
        Z = torch.fft.rfft(e, dim=-1)
        start = 1 if exclude_dc else 0
        Z = Z[..., start:]
        valid_F = Z.size(-1)
        if valid_F <= 2:
            D = 3 * num_bands + 5
            return torch.zeros(B, D, device=x_tok.device, dtype=x_tok.dtype)
        band_ids = _band_ids_(valid_F, num_bands, device=x_tok.device)
        power = (Z.real * Z.real + Z.imag * Z.imag).clamp_min(eps)
        # band energies
        E = torch.zeros((B, C, num_bands), device=x_tok.device, dtype=x_tok.dtype)
        for m in range(num_bands):
            mask = (band_ids == m)
            if mask.any():
                E[..., m] = power[..., mask].mean(dim=-1)
        log_E = torch.log(E + eps)
        p = torch.softmax(log_E, dim=-1)                  # [B, C, M]
        entropy_per_ch = -(p * (p + eps).log()).sum(dim=-1) / math.log(max(num_bands, 2))   # [B, C]

        # drift: early vs late half (in time domain, on innovation)
        T2 = max(2, T // 2)
        e_early = e[..., :T2]
        e_late = e[..., -T2:]
        Z_e = torch.fft.rfft(e_early, dim=-1)[..., start:]
        Z_l = torch.fft.rfft(e_late, dim=-1)[..., start:]
        if Z_e.size(-1) >= 2 and Z_l.size(-1) >= 2:
            bid_e = _band_ids_(Z_e.size(-1), num_bands, device=x_tok.device)
            bid_l = _band_ids_(Z_l.size(-1), num_bands, device=x_tok.device)
            E_e = torch.zeros((B, C, num_bands), device=x_tok.device, dtype=x_tok.dtype)
            E_l = torch.zeros((B, C, num_bands), device=x_tok.device, dtype=x_tok.dtype)
            for m in range(num_bands):
                me = (bid_e == m); ml = (bid_l == m)
                if me.any(): E_e[..., m] = ((Z_e.real**2 + Z_e.imag**2).clamp_min(eps))[..., me].mean(-1)
                if ml.any(): E_l[..., m] = ((Z_l.real**2 + Z_l.imag**2).clamp_min(eps))[..., ml].mean(-1)
            p_e = torch.softmax(torch.log(E_e + eps), dim=-1)
            p_l = torch.softmax(torch.log(E_l + eps), dim=-1)
            drift_band = (p_l - p_e).abs()                # [B, C, M]
        else:
            drift_band = torch.zeros((B, C, num_bands), device=x_tok.device, dtype=x_tok.dtype)

        # Sample-level aggregation
        feat_pieces = [
            p.mean(dim=1),                                # [B, M] mean over C
            p.std(dim=1),                                 # [B, M]
            drift_band.mean(dim=1),                       # [B, M]
            entropy_per_ch.mean(dim=1, keepdim=True),     # [B, 1]
            entropy_per_ch.std(dim=1, keepdim=True),      # [B, 1]
            # high/mid/low ratio (mean over C)
            p[..., :max(1, num_bands // 3)].sum(-1).mean(1, keepdim=True),
            p[..., max(1, num_bands // 3):2 * max(1, num_bands // 3)].sum(-1).mean(1, keepdim=True),
            p[..., 2 * max(1, num_bands // 3):].sum(-1).mean(1, keepdim=True),
        ]
        return torch.cat(feat_pieces, dim=-1)             # [B, 3M+5]


class JointAxisTWCMv4(nn.Module):
    """JointAxisTWCMv4 — completes Axis 2 with per-frame g_{k, t'}.

    Difference vs v3:
      g_base is [F, T_hist, R] instead of [F, R].
      Cross-channel mixing now varies across STFT frames within a window —
      captures the *time-within-window non-stationarity* TQNet gets via
      temporalQuery[cycle_index + t].

    Forward signature (matches STFTLocalCross):
        x_tok: real [B, C, T]
        out:   real [B, C, T]
    """

    def __init__(
        self,
        channels: int,
        seq_len: int,
        window: int = 24,
        stride: int = 12,
        rank: int = 8,
        num_bands: int = 8,                # IGNORED for residual path; only used for spec features
        gate_init: float = -1.0,
        gate_max: float = 1.0,
        spec_smooth_kernel: int = 25,
        spec_exclude_dc: bool = True,
        spec_eps: float = 1e-6,
        delta_hidden: int = 64,
        delta_zero_init: bool = True,
        use_scale_norm: bool = True,
        scale_eps: float = 1e-5,
        use_entropy_gate: bool = True,
        dc_gate_alpha_init: float = 10.0,
        dc_gate_beta_init: float = -1.0,
        dc_gate_center: float = 0.6,
        freq_gate_init: float = 0.0,
        freq_gate_low_init: float = -3.0,
        freq_gate_low_count: int = 2,        # suppress k=0 (DC) and k=1 (lowest non-DC) by default
    ):
        super().__init__()
        # ---- STFT settings ----
        W = min(int(window), int(seq_len))
        if W % 2 != 0:
            W = max(2, W - 1)
        S = max(1, min(int(stride), W // 2 if W > 1 else 1))
        self.window = W
        self.stride = S
        self.num_freqs = W // 2 + 1
        self.channels = int(channels)
        self.rank = max(1, min(int(rank), self.channels))
        # spec_num_bands is the resolution for δ_MLP's input features (coarse summary)
        self.spec_num_bands = max(1, min(int(num_bands), self.num_freqs))

        # ---- Compute T_hist from dummy STFT (needed for per-frame g) ----
        with torch.no_grad():
            _win = torch.hann_window(W)
            _Z = torch.stft(torch.zeros(1, int(seq_len)),
                             n_fft=W, hop_length=S, win_length=W,
                             window=_win, center=True, return_complex=True,
                             normalized=False)
        self.T_hist = _Z.shape[-1]

        # ---- Axis 3: low-rank channel basis ----
        self.A = nn.Parameter(torch.empty(self.channels, self.rank))
        self.B = nn.Parameter(torch.empty(self.channels, self.rank))
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.B)

        # ---- Axis 2: per-(BIN, FRAME) base gain (the v4 upgrade) ----
        # Shape [F, T_hist, R].  Initialised as the [F, R] base replicated
        # across frames + small per-frame noise → starts identical to v3.
        base_re = torch.empty(self.num_freqs, 1, self.rank)
        base_im = torch.empty(self.num_freqs, 1, self.rank)
        nn.init.normal_(base_re, std=0.02)
        nn.init.normal_(base_im, std=0.02)
        self.g_base_re = nn.Parameter(base_re.expand(self.num_freqs, self.T_hist, self.rank).contiguous().clone())
        self.g_base_im = nn.Parameter(base_im.expand(self.num_freqs, self.T_hist, self.rank).contiguous().clone())

        # ---- Axis 1: sample-modulated δg via spec features ----
        self.spec_smooth_kernel = int(spec_smooth_kernel)
        self.spec_exclude_dc = bool(spec_exclude_dc)
        self.spec_eps = float(spec_eps)
        D_spec = 3 * self.spec_num_bands + 5
        # δ_MLP now outputs 2 * F * R (per-bin sample modulation)
        self.delta_mlp = nn.Sequential(
            nn.Linear(D_spec, delta_hidden),
            nn.GELU(),
            nn.Linear(delta_hidden, 2 * self.num_freqs * self.rank),
        )
        if delta_zero_init:
            with torch.no_grad():
                self.delta_mlp[-1].weight.zero_()
                self.delta_mlp[-1].bias.zero_()

        # ---- Spec-feature band partition (only for δ_MLP input) ----
        edges = torch.linspace(0, self.spec_num_bands, self.num_freqs + 1)[:-1]
        spec_band_ids = torch.clamp(edges.long(), 0, self.spec_num_bands - 1)
        self.register_buffer("spec_band_ids", spec_band_ids)

        # ---- Gates ----
        self.gate_max = float(gate_max)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

        # ---- Per-bin gate (replaces per-band gate) ----
        # Init: lowest `freq_gate_low_count` bins suppressed (k=0 DC, k=1 longest
        # period = W = the slow trend / weekly drift bin). Higher k start neutral
        # (sigmoid=0.5). Model learns to open/close each bin from data.
        fg = torch.full((self.num_freqs,), float(freq_gate_init))
        for k in range(min(int(freq_gate_low_count), self.num_freqs)):
            fg[k] = float(freq_gate_low_init)
        self.freq_gate_logit = nn.Parameter(fg)

        # ---- Optional: entropy-aware DC gate (per-channel) ----
        self.use_entropy_gate = bool(use_entropy_gate)
        if self.use_entropy_gate:
            self.dc_alpha = nn.Parameter(torch.tensor(float(dc_gate_alpha_init)))
            self.dc_beta = nn.Parameter(torch.tensor(float(dc_gate_beta_init)))
            self.dc_center = float(dc_gate_center)

        # ---- Optional: scale-norm ----
        self.use_scale_norm = bool(use_scale_norm)
        self.scale_eps = float(scale_eps)

        # ---- Window function (Hann, COLA-compliant) ----
        self.register_buffer("win", torch.hann_window(W))

        # stats
        self.stats: dict = {}

    def _per_channel_entropy(self, x_tok):
        """Per-(sample, channel) normalized entropy for the entropy gate.

        Detached: data statistic only.
        """
        with torch.no_grad():
            smooth = _moving_average_(x_tok, self.spec_smooth_kernel)
            e = x_tok - smooth
            Z = torch.fft.rfft(e, dim=-1)
            start = 1 if self.spec_exclude_dc else 0
            Z = Z[..., start:]
            valid_F = Z.size(-1)
            if valid_F <= 2:
                return torch.zeros(x_tok.size(0), x_tok.size(1),
                                   device=x_tok.device, dtype=x_tok.dtype)
            bid = _band_ids_(valid_F, self.spec_num_bands, device=x_tok.device)
            power = (Z.real * Z.real + Z.imag * Z.imag).clamp_min(self.spec_eps)
            E = torch.zeros((x_tok.size(0), x_tok.size(1), self.spec_num_bands),
                            device=x_tok.device, dtype=x_tok.dtype)
            for m in range(self.spec_num_bands):
                mask = (bid == m)
                if mask.any():
                    E[..., m] = power[..., mask].mean(-1)
            p = torch.softmax(torch.log(E + self.spec_eps), dim=-1)
            H = -(p * (p + self.spec_eps).log()).sum(-1) / math.log(max(self.spec_num_bands, 2))
            return H

    def forward(self, x_tok):
        # x_tok: real [B, C, T]
        B, C, T = x_tok.shape

        # ---- Scale norm ----
        if self.use_scale_norm:
            mu = x_tok.mean(dim=-1, keepdim=True)
            sigma = x_tok.std(dim=-1, keepdim=True).clamp_min(self.scale_eps)
            x_in = (x_tok - mu) / sigma
        else:
            mu = None; sigma = None
            x_in = x_tok

        # ---- Axis 1: per-sample δg via spec features ----
        spec_feat = compute_sample_spec_features(
            x_tok,
            num_bands=self.spec_num_bands,
            smooth_kernel=self.spec_smooth_kernel,
            exclude_dc=self.spec_exclude_dc,
            eps=self.spec_eps,
        )                                                  # [B, D_spec], detached
        delta = self.delta_mlp(spec_feat)                  # [B, 2*F*R]
        d_re, d_im = delta.chunk(2, dim=-1)
        # g_base_re/im: [F, T_hist, R] → unsqueeze batch → [1, F, T_hist, R]
        # δ broadcasts over T_hist (sample-conditional contribution is t'-invariant)
        g_re = self.g_base_re.unsqueeze(0) + d_re.view(B, self.num_freqs, 1, self.rank)  # [B, F, T_hist, R]
        g_im = self.g_base_im.unsqueeze(0) + d_im.view(B, self.num_freqs, 1, self.rank)

        # ---- Axis 2: STFT decomposition ----
        x_flat = x_in.reshape(B * C, T)
        Z = torch.fft.rfft(self._stft_frames(x_flat), dim=-1)
        # Direct torch.stft equivalent:
        Z = torch.stft(
            x_flat, n_fft=self.window, hop_length=self.stride,
            win_length=self.window, window=self.win, center=True,
            return_complex=True, normalized=False,
        )                                                  # [B*C, F, T']
        F_bins, T_frames = Z.shape[-2], Z.shape[-1]
        Z = Z.reshape(B, C, F_bins, T_frames)

        # ---- Joint cross application (axis 3 + sample-modulated g) ----
        cdtype = Z.dtype
        A = self.A.to(cdtype)
        B_param = self.B.to(cdtype)
        # source projection: S[b, t', f, r] = Σ_j B[j, r] Z[b, j, f, t']
        S = torch.einsum("jr,bjft->btfr", B_param, Z)
        # per-sample, per-bin, PER-FRAME complex gain (v4 upgrade)
        g_complex = torch.complex(g_re, g_im)              # [B, F, T_hist, R]
        # per-bin gate: each bin self-selects whether to participate in cross-mixing.
        freq_gate = torch.sigmoid(self.freq_gate_logit)    # [F]
        g_per_freq = g_complex * freq_gate.view(1, -1, 1, 1)  # [B, F, T_hist, R]
        # Permute g [B, F, T_hist, R] → [B, T_hist, F, R] to align with S [B, T', F, R]
        # (T_hist == T' by construction since sl is fixed)
        g_per_freq_c = g_per_freq.permute(0, 2, 1, 3).to(cdtype)
        # apply gain at each (sample, frame, freq)
        Sg = S * g_per_freq_c                              # [B, T_hist, F, R]
        # target projection: R[b, c, f, t'] = Σ_r A[c, r] Sg[b, t', f, r]
        R = torch.einsum("cr,btfr->bcft", A, Sg)

        # global gate
        gate = self.gate_max * torch.sigmoid(self.gate_logit)
        Z_new = Z + gate.to(cdtype) * R

        # ---- Axis 2: iSTFT ----
        Z_flat = Z_new.reshape(B * C, F_bins, T_frames)
        y_norm = torch.istft(
            Z_flat, n_fft=self.window, hop_length=self.stride,
            win_length=self.window, window=self.win, center=True,
            length=T, normalized=False,
        ).reshape(B, C, T)

        # ---- De-normalize ----
        if self.use_scale_norm:
            y = y_norm * sigma + mu
        else:
            y = y_norm

        # ---- Entropy gate (per-channel data-conditional safety) ----
        if self.use_entropy_gate:
            H = self._per_channel_entropy(x_tok)           # [B, C]
            dc_gate = torch.sigmoid(self.dc_alpha * (H - self.dc_center) + self.dc_beta)
            # Re-scale correction by dc_gate to suppress on peaked channels
            correction = y - x_tok
            y = x_tok + dc_gate.unsqueeze(-1) * correction
        else:
            dc_gate = None

        # ---- Stats ----
        with torch.no_grad():
            # Measure time-within-window variation: how much do per-frame g's
            # differ from each other?  std over t' divided by mean magnitude.
            g_re_per_frame = self.g_base_re  # [F, T_hist, R]
            t_std = g_re_per_frame.std(dim=1).mean()  # avg std across t' per (f, r)
            t_mean_abs = g_re_per_frame.abs().mean()
            self.stats = {
                "twcm_gate": float(gate.item()),
                "delta_re_abs_mean": float(d_re.abs().mean().item()),
                "delta_im_abs_mean": float(d_im.abs().mean().item()),
                "g_re_rms": float(g_re.pow(2).mean().sqrt().item()),
                "g_t_std": float(t_std.item()),
                "g_t_mean_abs": float(t_mean_abs.item()),
                "g_t_variation_ratio": float((t_std / (t_mean_abs + 1e-9)).item()),
                "correction_rms": float((y - x_tok).pow(2).mean().sqrt().item()),
                "freq_gate_min": float(freq_gate.min().item()),
                "freq_gate_max": float(freq_gate.max().item()),
                "freq_gate_k0":  float(freq_gate[0].item()),
                "freq_gate_k1":  float(freq_gate[1].item()) if self.num_freqs > 1 else 0.0,
                "T_hist": int(self.T_hist),
            }
            if self.use_entropy_gate:
                self.stats.update({
                    "dc_gate_mean": float(dc_gate.mean().item()),
                    "H_mean": float(H.mean().item()),
                    "H_min": float(H.min().item()),
                    "H_max": float(H.max().item()),
                })
        return y

    def _stft_frames(self, x_flat):
        # placeholder kept for editor consistency; actual STFT is via torch.stft below
        return x_flat


# ----------------------------------------------------------------- smoke test
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, T = 4, 7, 96
    x = torch.randn(B, C, T)
    mod = JointAxisTWCMv4(channels=C, seq_len=T,
                          window=24, stride=12,
                          rank=8, num_bands=8,
                          gate_init=-1.0,
                          use_entropy_gate=True,
                          delta_zero_init=True)
    y = mod(x)
    assert y.shape == (B, C, T), f"shape {tuple(y.shape)}"
    loss = y.mean(); loss.backward()
    print(f"params = {sum(p.numel() for p in mod.parameters())}")
    print(f"stats  = {mod.stats}")

    # Peaked input (low entropy → dc_gate near 0, but δ still varies sample-wise)
    t = torch.arange(T, dtype=torch.float32)
    x_peaked = torch.sin(2 * math.pi * t / 24).view(1, T, 1).repeat(B, 1, C).permute(0, 2, 1)
    y2 = mod(x_peaked)
    print(f"peaked: dc_gate_mean={mod.stats['dc_gate_mean']:.4f} "
          f"correction_rms={mod.stats['correction_rms']:.4f}")

    # Broadband (high entropy → dc_gate high)
    y3 = mod(torch.randn(B, C, T))
    print(f"broad:  dc_gate_mean={mod.stats['dc_gate_mean']:.4f} "
          f"correction_rms={mod.stats['correction_rms']:.4f}")

    print("smoke test ok")
