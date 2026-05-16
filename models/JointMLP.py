"""JointMLP — TQNet's MLP forecaster with JA cross-channel replacing
TQNet's `temporalQuery[cycle_index]` AND `MultiheadAttention(channel)`.

Hypothesis under test
---------------------
JA's frequency-conditioned cross-channel mixing supplies the same kind of
contextual cross-channel information that TQNet obtained from
(a) `temporalQuery[cycle_index]`  (cycle-bound, hardcoded period)
(b) `MultiheadAttention(channel)`  (flat, single matrix pooled over time)
so we can remove both and let JA carry that role, keeping only TQNet's
strong MLP backbone.

Architecture
------------
    x [B, T, C]
      ├─ RIN
      ├─ permute → x_input [B, C, T]
      ├─ JointAxisTWCMv4(x_input)        ← supplies freq-conditioned cross-channel info
      │     output = x_input + freq-resolved correction
      ├─ input_proj  : Linear(seq_len → d_model)
      ├─ MLP backbone: 2-layer GELU residual         (kept from TQNet)
      ├─ output_proj : Dropout + Linear(d_model → pred_len)
      ├─ permute → [B, pred_len, C]
      └─ RIN denorm

Removed vs TQNet
----------------
  temporalQuery [cycle_len, enc_in]   — cycle hyperparameter eliminated
  MultiheadAttention(embed=seq_len)   — flat channel attention removed

Kept from TQNet
---------------
  RIN, input_proj, MLP backbone, output_proj.  No cycle_index in forward().
"""
import os, sys
import torch
import torch.nn as nn

if __name__ == "__main__" or "models" not in sys.modules:
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    if _root not in sys.path:
        sys.path.insert(0, _root)

from models.JointAxisTWCMv4 import JointAxisTWCMv4


def _auto_window(seq_len: int, w_max: int = 64) -> int:
    W = max(8, min(int(seq_len) // 4, w_max))
    if W % 2: W -= 1
    return max(2, W)


class Model(nn.Module):
    """JointMLP — JA cross-channel + TQNet MLP backbone, no cycle."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.d_model = int(getattr(configs, "d_model", 512))
        self.dropout = float(getattr(configs, "dropout", 0.5))
        self.use_revin = bool(getattr(configs, "use_revin", 1))

        # ---- JA cross-channel (frequency-conditioned) ----
        W = int(getattr(configs, "jmlp_window", -1))
        if W <= 0:
            W = _auto_window(self.seq_len)
        S = int(getattr(configs, "jmlp_stride", -1))
        if S <= 0:
            S = max(1, W // 2)
        R = int(getattr(configs, "jmlp_rank", 8))
        # Adaptive rank: R=-1 → R = min(C, max(8, 2*sqrt(C)))
        # PEMS07 (C=883) → 59, ECL (C=321) → 36, ETT (C=7) → 7, weather (C=21) → 9
        if R <= 0:
            import math as _math
            R = min(self.enc_in, max(8, int(2 * _math.sqrt(self.enc_in))))

        # JA v4 backend: per-bin per-frame gain g_{k, t'} — see JointAxisTWCMv4 docstring.
        self.cross = JointAxisTWCMv4(
            channels=self.enc_in, seq_len=self.seq_len,
            window=W, stride=S, rank=R,
            num_bands=8,                                # only used for δ_MLP spec features
            gate_init=float(getattr(configs, "jmlp_gate_init", -1.0)),
            gate_max=1.0,
            delta_hidden=int(getattr(configs, "jmlp_delta_hidden", 64)),
            delta_zero_init=True,
            use_scale_norm=False,                       # RIN handles it
            use_entropy_gate=bool(getattr(configs, "jmlp_use_entropy_gate", 1)),
            dc_gate_alpha_init=10.0,
            dc_gate_beta_init=-1.0,
            dc_gate_center=0.6,
            freq_gate_init=0.0,
            freq_gate_low_init=-3.0,
            freq_gate_low_count=2,
        )

        # ---- MLP backbone (borrowed from TQNet, unchanged) ----
        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(),
            nn.Linear(self.d_model, self.d_model), nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )

        # Innovation-only mode: apply JA on (x - moving_avg(x)) only
        self.innovation_only = bool(getattr(configs, "jmlp_innovation_only", 0))
        self.innovation_kernel = int(getattr(configs, "jmlp_innovation_kernel", 25))

        self.cross_stats: dict = {}

    def _moving_avg(self, x):
        """x: [B, C, T] → smoothed [B, C, T]."""
        import torch.nn.functional as F
        T = x.size(-1)
        k = min(int(self.innovation_kernel), T)
        if k % 2 == 0: k -= 1
        if k < 3: return x
        pad = k // 2
        return F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), kernel_size=k, stride=1)

    def forward(self, x):
        # x: [B, T, C]
        if self.use_revin:
            mu = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True) + 1e-5
            x = (x - mu) / torch.sqrt(var)

        x_input = x.permute(0, 2, 1)                            # [B, C, T]
        if self.innovation_only:
            # JA only mixes the deviation; slow trend bypasses JA untouched
            trend = self._moving_avg(x_input)
            inno = x_input - trend
            inno_aug = self.cross(inno)
            x_aug = trend + inno_aug
        else:
            x_aug = self.cross(x_input)                         # JA-enriched [B, C, T]

        feat = self.input_proj(x_aug)                           # [B, C, d_model]
        hidden = self.mlp(feat)                                 # [B, C, d_model]
        output = self.output_proj(hidden + feat).permute(0, 2, 1)  # [B, pl, C]

        if self.use_revin:
            output = output * torch.sqrt(var) + mu

        if not self.training:
            with torch.no_grad():
                self.cross_stats = dict(self.cross.stats)
        return output


# ----------------------------------------------------------------- smoke test
if __name__ == "__main__":
    from types import SimpleNamespace
    torch.manual_seed(0)
    for sl, pl, C in [(96, 96, 7), (96, 96, 321), (96, 96, 170), (336, 192, 21), (720, 720, 170)]:
        cfg = SimpleNamespace(
            seq_len=sl, pred_len=pl, enc_in=C,
            d_model=512, dropout=0.5, use_revin=1,
        )
        m = Model(cfg)
        B = 4
        x = torch.randn(B, sl, C)
        y = m(x)
        assert y.shape == (B, pl, C), y.shape
        (y.mean()).backward()
        n = sum(p.numel() for p in m.parameters())
        print(f"sl={sl:>4} pl={pl:>4} C={C:>4}  →  y={tuple(y.shape)}  params={n:>10}")
    print("\nsmoke test ok")
