# Shared dataset metadata for AsySpecX baseline scripts.
# Usage: source scripts/_common.sh; load_dataset <key>
# Exposes: data_name, data_path, subdir, enc_in, cycle, period_len, bs, lr, revin, pred_lens, data_key.
# data files are read from ./dataset/${subdir}/${data_path}

load_dataset() {
    case "$1" in
        ETTh1)       data_name=ETTh1;  data_path=ETTh1.csv;       subdir=ETT-small;   enc_in=7;   cycle=24;  period_len=24;  bs=256; lr=0.001; revin=1; pred_lens="96 192 336 720" ;;
        ETTh2)       data_name=ETTh2;  data_path=ETTh2.csv;       subdir=ETT-small;   enc_in=7;   cycle=24;  period_len=24;  bs=256; lr=0.001; revin=1; pred_lens="96 192 336 720" ;;
        ETTm1)       data_name=ETTm1;  data_path=ETTm1.csv;       subdir=ETT-small;   enc_in=7;   cycle=96;  period_len=96;  bs=256; lr=0.001; revin=1; pred_lens="96 192 336 720" ;;
        ETTm2)       data_name=ETTm2;  data_path=ETTm2.csv;       subdir=ETT-small;   enc_in=7;   cycle=96;  period_len=96;  bs=256; lr=0.001; revin=1; pred_lens="96 192 336 720" ;;
        weather)     data_name=custom; data_path=weather.csv;     subdir=weather;     enc_in=21;  cycle=144; period_len=144; bs=64;  lr=0.001; revin=1; pred_lens="96 192 336 720" ;;
        electricity) data_name=custom; data_path=electricity.csv; subdir=electricity; enc_in=321; cycle=168; period_len=24;  bs=32;  lr=0.003; revin=1; pred_lens="96 192 336 720" ;;
        traffic)     data_name=custom; data_path=traffic.csv;     subdir=traffic;     enc_in=862; cycle=168; period_len=24;  bs=16;  lr=0.003; revin=1; pred_lens="96 192 336 720" ;;
        exchange_rate) data_name=custom; data_path=exchange_rate.csv;             subdir=exchange_rate; enc_in=8;  cycle=1;   period_len=1;  bs=32;  lr=0.0005; revin=1; pred_lens="96 192 336 720" ;;
        illness)     data_name=custom; data_path=national_illness.csv;        subdir=illness; enc_in=7;   cycle=1;   period_len=1;  bs=16;  lr=0.001; revin=1; pred_lens="24 36 48 60" ;;
        Beijing-PM25)  data_name=custom; data_path=air_pm25_beijing_12.csv;            subdir=PM25; enc_in=12;  cycle=24;  period_len=24; bs=256; lr=0.001; revin=1; pred_lens="96 192 336 720" ;;
        Beijing-Air72) data_name=custom; data_path=air_quality_beijing_pollutants_72.csv; subdir=PM25; enc_in=72;  cycle=24;  period_len=24; bs=64;  lr=0.001; revin=1; pred_lens="96 192 336 720" ;;
        Beijing-Air132) data_name=custom; data_path=air_quality_beijing_132.csv;          subdir=PM25; enc_in=132; cycle=24;  period_len=24; bs=32;  lr=0.001; revin=1; pred_lens="96 192 336 720" ;;
        Beijing_AQ)  data_name=custom; data_path=Beijing_AQ.csv;  subdir=Beijing_AQ;  enc_in=132; cycle=24;  period_len=24; bs=32;  lr=0.001; revin=1; pred_lens="96 192 336 720" ;;
        Beijing_PM25) data_name=custom; data_path=Beijing_PM25.csv; subdir=Beijing_AQ; enc_in=12;  cycle=24;  period_len=24; bs=64;  lr=0.001; revin=1; pred_lens="96 192 336 720" ;;
        METR_LA)  data_name=custom; data_path=METR_LA.csv;  subdir=Spatial; enc_in=207; cycle=288; period_len=288; bs=32; lr=0.001; revin=1; pred_lens="12 24 48 96" ;;
        PEMS_BAY) data_name=custom; data_path=PEMS_BAY.csv; subdir=Spatial; enc_in=325; cycle=288; period_len=288; bs=32; lr=0.001; revin=1; pred_lens="12 24 48 96" ;;
        D1NAMO_001) data_name=custom; data_path=D1NAMO_001.csv;  subdir=Beijing_AQ; enc_in=15;  cycle=3600; period_len=3600; bs=32;  lr=0.001; revin=1; pred_lens="60 300 600 3600" ;;
        PEMS03)      data_name=PEMS;   data_path=PEMS03.npz;      subdir=PEMS;        enc_in=358; cycle=288; period_len=288; bs=32;  lr=0.003; revin=0; pred_lens="12 24 48 96" ;;
        PEMS04)      data_name=PEMS;   data_path=PEMS04.npz;      subdir=PEMS;        enc_in=307; cycle=288; period_len=288; bs=32;  lr=0.003; revin=0; pred_lens="12 24 48 96" ;;
        PEMS07)      data_name=PEMS;   data_path=PEMS07.npz;      subdir=PEMS;        enc_in=883; cycle=288; period_len=288; bs=32;  lr=0.003; revin=0; pred_lens="12 24 48 96" ;;
        PEMS08)      data_name=PEMS;   data_path=PEMS08.npz;      subdir=PEMS;        enc_in=170; cycle=288; period_len=288; bs=32;  lr=0.003; revin=1; pred_lens="12 24 48 96" ;;  # NB: PEMS08 uses revin=1 (matches official TQNet); others use 0
        *) echo "Unknown dataset key: $1" >&2; return 1 ;;
    esac
    data_key="$1"
}

# Selects FilterNet variant per official scripts.
# PaiFilter: ETT-h, ETT-m (low-channel, simple frequency filter)
# TexFilter: weather, electricity, traffic, PEMS (uses learnable spectral kernel)
filternet_variant() {
    case "$1" in
        ETTh1|ETTh2|ETTm1|ETTm2|Beijing-PM25) echo "PaiFilter" ;;
        weather|electricity|traffic|PEMS03|PEMS04|PEMS07|PEMS08|Beijing-Air72|Beijing-Air132) echo "TexFilter" ;;
        *) echo "PaiFilter" ;;
    esac
}

# SparseTSF/MixLinear require seq_len % period_len == 0 and pred_len % period_len == 0.
period_compatible() {
    local sl=$1 pl=$2 p=$3
    if [ $((sl % p)) -ne 0 ] || [ $((pl % p)) -ne 0 ]; then
        return 1
    fi
    return 0
}

# CycleNet/Linear paper-aligned overrides — mutates `lr` and `bs`.
# Source: github.com/ACAT-SCUT/CycleNet/scripts/CycleNet/Linear-Input-{96,336,720}/<dataset>.sh
# Pattern: lr=0.01 for sl<=96, lr=0.005 for sl>=336; bs=64 for electricity/traffic, bs=256 elsewhere.
apply_cyclenet_overrides() {
    local ds=$1 sl=$2
    case "$ds" in
        electricity|traffic) bs=64 ;;
        *) bs=256 ;;
    esac
    if [ "$sl" -le 96 ]; then
        lr=0.01
    else
        lr=0.005
    fi
}

# PatchTST paper-aligned overrides — mutates lr, bs, d_model, d_ff, n_heads, dropout, fc_dropout.
# Source: github.com/yuqinie98/PatchTST/PatchTST_supervised/scripts/PatchTST/<dataset>.sh
# Two regimes:
#   ETTh1/ETTh2 → small (d_model=16, n_heads=4, dropout=0.3) per the original ETT-h scripts
#   others       → standard (d_model=128, d_ff=256, n_heads=16, dropout=0.2)
apply_patchtst_overrides() {
    local ds=$1
    lr=0.0001
    fc_dropout=0.2
    head_dropout=0
    case "$ds" in
        ETTh1|ETTh2)
            d_model=16; d_ff=128; n_heads=4; dropout=0.3; fc_dropout=0.3; bs=128
            ;;
        ETTm1|ETTm2|weather|Beijing-PM25)
            d_model=128; d_ff=256; n_heads=16; dropout=0.2; bs=128
            ;;
        electricity|PEMS03|PEMS04|PEMS07|PEMS08|Beijing-Air72|Beijing-Air132)
            d_model=128; d_ff=256; n_heads=16; dropout=0.2; bs=32
            ;;
        traffic)
            d_model=128; d_ff=256; n_heads=16; dropout=0.2; bs=24
            ;;
    esac
}

# iTransformer paper-aligned overrides — mutates lr, bs, d_model, d_ff, e_layers.
# Source: github.com/thuml/iTransformer/scripts/multivariate_forecasting/<group>/iTransformer*.sh
# n_heads/dropout left at defaults (8, 0.1).
apply_itransformer_overrides() {
    local ds=$1 pl=$2
    case "$ds" in
        ETTh1)
            lr=0.0001; bs=32; e_layers=2
            if [ "$pl" -ge 336 ]; then d_model=512; d_ff=512; else d_model=256; d_ff=256; fi
            ;;
        ETTh2|ETTm1|ETTm2)
            lr=0.0001; bs=32; d_model=128; d_ff=128; e_layers=2
            ;;
        weather|Beijing-PM25)
            lr=0.0001; bs=32; d_model=512; d_ff=512; e_layers=3
            ;;
        electricity|PEMS03|PEMS04|PEMS07|PEMS08|Beijing-Air72|Beijing-Air132)
            lr=0.0005; bs=16; d_model=512; d_ff=512; e_layers=3
            ;;
        traffic)
            lr=0.001; bs=16; d_model=512; d_ff=512; e_layers=4
            ;;
    esac
}

# DLinear paper-aligned overrides.
# Source: github.com/cure-lab/LTSF-Linear/scripts/EXP-LongForecasting/Linear/<dataset>.sh
apply_dlinear_overrides() {
    local ds=$1
    case "$ds" in
        ETTh1|ETTh2)             lr=0.005; bs=32 ;;
        ETTm1|ETTm2|weather)     lr=0.005; bs=32 ;;
        electricity)             lr=0.001; bs=16 ;;
        traffic)                 lr=0.05;  bs=16 ;;
        PEMS*)                   lr=0.005; bs=32 ;;
        Beijing-PM25)            lr=0.005; bs=32 ;;
        Beijing-Air72|Beijing-Air132) lr=0.001; bs=16 ;;
    esac
}

# FITS paper-aligned overrides.
# Source: github.com/VEWOXIC/FITS/scripts/FITS/*.sh — universal lr=0.0005, bs=64, patience=20.
# cut_freq computed from base_T (per dataset) and H_order=6 for ETT/weather/electricity/traffic.
# PEMS not in original FITS paper; follow Scope's pems_smoke setup: cut_freq=sl//4, bs=16, patience=10.
apply_fits_overrides() {
    local ds=$1 sl=$2
    lr=0.0005; bs=64; patience=20; epochs=30
    case "$ds" in
        PEMS*)
            cut_freq=$(( sl / 4 ))
            bs=16; patience=10
            return
            ;;
    esac
    local base_T=24
    case "$ds" in
        ETTh1|ETTh2)         base_T=24 ;;
        ETTm1|ETTm2)         base_T=96 ;;
        weather)             base_T=144 ;;
        electricity|traffic) base_T=24 ;;
    esac
    cut_freq=$(( sl / base_T * 6 + 1 ))
    if [ "$cut_freq" -le 1 ]; then cut_freq=2; fi
}

# FreTS overrides (official has no per-dataset scripts; uses Informer-style defaults).
apply_frets_overrides() {
    lr=0.0001
    bs=32
    embed_size=128
    hidden_size=256
    dropout=0.05
}

# FilterNet paper-aligned overrides — selects variant, mutates lr/bs/embed_size/hidden_size/dropout.
# Source: github.com/aikunyi/FilterNet/scripts/{PaiFilter,TexFilter}/<dataset>.sh
# PaiFilter uses embed_size=seq_len internally; we only set hidden_size for it.
# Note: weather uses TexFilter (not PaiFilter despite low channel count) per official scripts.
apply_filternet_overrides() {
    local ds=$1
    epochs=15; patience=5; dropout=0; embed_size=128
    case "$ds" in
        ETTh1|ETTh2)
            variant=PaiFilter; lr=0.005; bs=16; hidden_size=256
            ;;
        ETTm1|ETTm2)
            variant=PaiFilter; lr=0.01;  bs=32; hidden_size=256
            ;;
        Beijing-PM25)
            variant=PaiFilter; lr=0.005; bs=32; hidden_size=256
            ;;
        weather)
            variant=TexFilter; lr=0.01;  bs=128; embed_size=128; hidden_size=128; epochs=20; patience=6
            ;;
        electricity|PEMS*)
            variant=TexFilter; lr=0.001; bs=4;   embed_size=512; hidden_size=512; epochs=20; patience=6
            ;;
        traffic)
            variant=TexFilter; lr=0.005; bs=16;  embed_size=256; hidden_size=512; epochs=20; patience=6
            ;;
        Beijing-Air72|Beijing-Air132)
            variant=TexFilter; lr=0.001; bs=16;  embed_size=256; hidden_size=256; epochs=20; patience=6
            ;;
    esac
}

# SparseTSF paper-aligned overrides.
# Source: github.com/lss-1138/SparseTSF/scripts/SparseTSF/linear/<dataset>.sh
# Note: paper uses period_len=96 for ETT-m and 24 for ETT-h/electricity/traffic.
# We force period_len=24 universally so all sl ∈ {96,336,512,720} × pl combinations
# are divisible (otherwise SparseTSF's reshape fails). Slight deviation from paper
# for ETT-m/weather/PEMS (paper would use 96/144/288) — period_len=24 is a smaller
# divisor and remains semantically valid (sub-daily cross-period aggregation).
apply_sparsetsf_overrides() {
    local ds=$1
    lr=0.02
    case "$ds" in
        ETTh1|ETTh2|ETTm1|ETTm2)  bs=256 ;;
        weather)                  bs=256 ;;
        electricity|traffic)      bs=128 ;;
        PEMS*)                    bs=128 ;;
    esac
    period_len=24
}

# AsySpecX overrides — probe-driven settings (PROBE_VERDICT.md, 2026-05-03).
#   - gate_init=0.0, gate_max=1.0  → balanced gate (was -6/0.2 which suppressed cross block)
#   - rank=2 for high-C datasets   → P8b best rank for 321/862/PEMS-scale; rank=full for small-C
#   - num_bands=8                   → matches probe per-band analysis
#   - cut_freq from FITS H_order/base_T (same convention as apply_fits_overrides)
#   - lr/bs/patience/epochs follow FITS-family defaults (lr=5e-4, bs=64, patience=10, epochs=30)
apply_asyspecx_overrides() {
    local ds=$1 sl=$2
    lr=0.0005; bs=64; patience=10; epochs=30
    num_bands=8; gate_init=0.0; gate_max=1.0
    case "$ds" in
        electricity|traffic|PEMS*|Beijing-Air72|Beijing-Air132)
            rank=2 ;;
        ETTh1|ETTh2|ETTm1|ETTm2|weather|Beijing-PM25|exchange_rate|illness)
            rank=8 ;;
        *)  rank=8 ;;
    esac
    # cut_freq: FITS-style based on dataset period
    case "$ds" in
        PEMS*)
            # PEMS-friendly: full spectrum, small batch (matches FilterNet baseline setup)
            cut_freq=$(( sl / 2 + 1 ))   # full freq spectrum (was sl/4, too aggressive)
            bs=4                          # match FilterNet (was 16)
            lr=0.001                      # slightly more aggressive (was 0.0005)
            num_bands=16                  # more bands for richer per-band coupling
            return
            ;;
    esac
    local base_T=24
    case "$ds" in
        ETTh1|ETTh2)         base_T=24 ;;
        ETTm1|ETTm2)         base_T=96 ;;
        weather)             base_T=144 ;;
        electricity|traffic) base_T=24 ;;
        Beijing-PM25|Beijing-Air72|Beijing-Air132) base_T=24 ;;
        exchange_rate)       base_T=1 ;;     # daily, no clear period
        illness)             base_T=1 ;;     # weekly, no clear period
    esac
    cut_freq=$(( sl / base_T * 6 + 1 ))
    if [ "$cut_freq" -le 1 ]; then cut_freq=2; fi
}

# JointMLP (JA v4) overrides — rank adapts to channel count.
# Rationale: cross-channel coupling on high-C datasets has effective rank ≈ 2
# (road-graph clusters / building archetypes / PEMS topology), so rank=8 is
# both wasteful and harder to optimize (gradient signal diluted across unused
# directions). Low-C datasets (ETT/weather) have full-rank semantic channels,
# so rank=8 ≈ dense and is the right default.
# Exposes: jmlp_rank. Other JMLP_* knobs stay at their defaults in _template.sh.
apply_jmlp_overrides() {
    local ds=$1
    case "$ds" in
        electricity|traffic|PEMS*|Beijing-Air72|Beijing-Air132)
            jmlp_rank=2 ;;
        ETTh1|ETTh2|ETTm1|ETTm2|weather|Beijing-PM25|exchange_rate|illness)
            jmlp_rank=8 ;;
        *)  jmlp_rank=8 ;;
    esac
}

# MixLinear paper-aligned overrides — per-dataset period_len/alpha/lpf.
# Source: github.com/aitianma/MixLinear/scripts/MixLinear/*.sh
apply_mixlinear_overrides() {
    local ds=$1
    epochs=30; patience=10
    case "$ds" in
        ETTh1)
            lr=0.03;  bs=256; period_len=24; alpha=0.95; lpf=5
            ;;
        ETTh2)
            lr=0.02;  bs=256; period_len=24; alpha=0.99; lpf=15
            ;;
        ETTm1|ETTm2)
            lr=0.005; bs=64;  period_len=2;  alpha=0.99; lpf=144; epochs=15
            ;;
        weather)
            lr=0.02;  bs=64;  period_len=4;  alpha=0.5;  lpf=5
            ;;
        electricity)
            lr=0.03;  bs=64;  period_len=24; alpha=0.5;  lpf=5
            ;;
        traffic)
            lr=0.03;  bs=64;  period_len=24; alpha=0.5;  lpf=5
            ;;
        PEMS*)
            # period_len=12 (instead of paper's 24) so pl=12 row is not skipped
            lr=0.03;  bs=64;  period_len=12; alpha=0.5;  lpf=5
            ;;
        Beijing-PM25|Beijing-Air72|Beijing-Air132)
            lr=0.02;  bs=64;  period_len=24; alpha=0.5;  lpf=5
            ;;
    esac
}

