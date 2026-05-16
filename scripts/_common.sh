# Shared dataset metadata for AsySpecX scripts.
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

# AsySpecX per-dataset hyperparameter overrides (probe-driven).
#   - gate_init=0.0, gate_max=1.0  → balanced gate
#   - rank=2 for high-C datasets   → P8b best rank for 321/862/PEMS-scale; rank=8 for small-C
#   - num_bands=8                   → matches probe per-band analysis
#   - cut_freq from FITS H_order/base_T (FITS convention)
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
    case "$ds" in
        PEMS*)
            cut_freq=$(( sl / 2 + 1 ))   # full freq spectrum
            bs=4
            lr=0.001
            num_bands=16
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
        exchange_rate)       base_T=1 ;;
        illness)             base_T=1 ;;
    esac
    cut_freq=$(( sl / base_T * 6 + 1 ))
    if [ "$cut_freq" -le 1 ]; then cut_freq=2; fi
}
