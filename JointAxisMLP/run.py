import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2026, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='TQNet',
                    help='model name, options: [TQNet, Informer, Autoformer, ...]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# TQNet & CycleNet
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# FITS
parser.add_argument('--cut_freq', type=int, default=0, help='FITS frequency cutoff bin (set per dataset)')

# AsySpecX
parser.add_argument('--rank', type=int, default=8, help='AsySpecX low-rank dimension r in H = A diag(g_m) B^T')
parser.add_argument('--num_bands', type=int, default=8, help='AsySpecX number of contiguous frequency bands sharing g_m')
parser.add_argument('--gate_init', type=float, default=0.0, help='AsySpecX gate logit init (0 = balanced, -6 = suppress)')
parser.add_argument('--gate_max', type=float, default=1.0, help='AsySpecX gate cap (1.0 = full strength, 0.2 = restrict)')

# FreTS / FilterNet
parser.add_argument('--embed_size', type=int, default=128, help='FreTS/FilterNet embed size')
parser.add_argument('--hidden_size', type=int, default=256, help='FreTS/FilterNet hidden size')
parser.add_argument('--channel_independence', type=str, default='0', help='FreTS channel-independence (string 0/1)')

# FilterNet
parser.add_argument('--model_variant', type=str, default='PaiFilter', help='FilterNet variant: PaiFilter or TexFilter')

# SparseTSF / MixLinear
parser.add_argument('--period_len', type=int, default=24, help='SparseTSF/MixLinear period length')
parser.add_argument('--alpha', type=float, default=0.5, help='MixLinear time/freq blend coefficient')
parser.add_argument('--lpf', type=int, default=4, help='MixLinear low-pass filter bins')
parser.add_argument('--ah_lambda', type=float, default=0.0, help='Anti-Hermitian regularization weight (>0 to enable)')
parser.add_argument('--l1_lambda', type=float, default=0.05, help='L1 reg on cross block amp')
parser.add_argument('--l1_imag_lambda', type=float, default=-1.0, help='L1 gate on imag head amp (FreqDualHead). <0 means 2*l1_lambda')
parser.add_argument('--alpha_mode', type=str, default='learnable', choices=['learnable', 'fixed'], help='FreqSymOnly: alpha blend mode')
parser.add_argument('--alpha_fixed', type=float, default=0.5, help='FreqSymOnly: fixed alpha when alpha_mode=fixed')
parser.add_argument('--lag_K', type=int, default=10, help='FreqSymLag: top-K source channels per target')
parser.add_argument('--lag_max', type=int, default=96, help='FreqSymLag: max lag for pair selection')
parser.add_argument('--lag_min', type=int, default=4, help='FreqSymLag: min lag for pair selection (excludes trivial mutual)')
parser.add_argument('--n_phase_bands', type=int, default=3, help='FreqPhaseGraphB / FreqHCoupling: number of frequency bands')
parser.add_argument('--rank_skew', type=int, default=8, help='FreqHCoupling: rank of per-band skew low-rank decomposition')
parser.add_argument('--oracle_mode', type=str, default='full', help='FreqOracleMix(V2): oracle ceiling mode (none/real/full or none/symLR/full)')
parser.add_argument('--conditional', type=int, default=0, help='Conditional MTSF: zero a random channel of input per batch sample, score loss only on that channel (leave-one-channel-out forecasting)')
parser.add_argument('--mask_ratio', type=float, default=0.0, help='Missing sensor: fraction of channels to mask per batch sample (0=use --conditional k=1 if set)')
parser.add_argument('--temporal_mask_ratio', type=float, default=0.0, help='Temporal masking: fraction of random input timesteps to zero-fill per sample (applied to all channels at those timesteps).')
parser.add_argument('--mask_persistent_train', type=int, default=1, help='If 1, apply mask during training too (mode A3 robust); else only at vali/test (mode A2 cold)')
parser.add_argument('--mask_seed', type=int, default=2026, help='Reproducibility seed for vali/test masks')
parser.add_argument('--mask_held_out', type=str, default='', help='Mode B virtual sensor: comma-separated channel indices held out (always masked, never in target except scored)')
parser.add_argument('--mask_recent_r', type=int, default=0, help='If >0, mask only the LAST r timesteps of failed channels (recent-outage protocol). 0 = mask the full L window (legacy).')
parser.add_argument('--mask_protocol', type=str, default='', choices=['', 'P1', 'P2', 'P3', 'P4'], help='Outage protocol: P1 random, P2 long-outage (uses mask_recent_r), P3 modality-target, P4 node/station')
parser.add_argument('--channel_meta', type=str, default='', help='Path to channel_meta.csv (cols: channel_idx,node_id,modality). Required for P3/P4.')
parser.add_argument('--mask_modality', type=str, default='', help='P3 modality name to mask (e.g. PM2.5)')
parser.add_argument('--mask_node', type=str, default='', help='P4 node id to mask (all channels at this node)')
parser.add_argument('--mask_as_feature', type=int, default=0, help='If 1, concat input mask to x along channel dim (doubles enc_in seen by model). Set --enc_in to 2*C manually.')
parser.add_argument('--teacher_driver_path', type=str, default='', help='CoSTDriver: path to teacher_driver_<dataset>.npz with [C,C] teacher distribution')
parser.add_argument('--cost_lambda_driver', type=float, default=0.1, help='CoSTDriver: KL aux-loss weight')
parser.add_argument('--cost_alpha_init', type=float, default=-3.0, help='CoSTDriver: initial bias of routing gate (negative = self-pred default)')
parser.add_argument('--n_cross_layers', type=int, default=3, help='FreqHermV2: number of stacked cross blocks')
parser.add_argument('--modrelu_bias_init', type=float, default=-0.1, help='FreqHermV2: modReLU bias init')
parser.add_argument('--use_complex_ln', type=int, default=1, help='FreqHermV2: enable per-real/imag LayerNorm')
parser.add_argument('--use_tq', type=int, default=1, help='TQNet: enable temporal query (cycle prior)')
parser.add_argument('--channel_aggre', type=int, default=1, help='TQNet: enable channel aggregation via attention')
parser.add_argument('--use_cycle_attn', type=int, default=1, help='FreqHermCycleAttn: enable cycle-conditioned channel attention')
parser.add_argument('--n_attn_heads', type=int, default=4, help='FreqHermCycleAttn: number of attention heads')
parser.add_argument('--attn_dropout', type=float, default=0.1, help='FreqHermCycleAttn: attention dropout')
parser.add_argument('--d_emb', type=int, default=64, help='FreqLagAttn: complex embedding dim')
parser.add_argument('--bias_init_scale', type=float, default=1.0, help='FreqLagAttn: lag prior bias init scale')
parser.add_argument('--d_band', type=int, default=32, help='FreqLagAlign: per-band embed dim')
parser.add_argument('--w_rc_init', type=float, default=0.5, help='FreqLagAlign: RC bias init weight')
parser.add_argument('--w_ic_init', type=float, default=1.0, help='FreqLagAlign: IC bias init weight')
parser.add_argument('--n_bands', type=int, default=8, help='FreqLagAlign: number of frequency bands')
parser.add_argument('--n_bands_lo', type=int, default=2, help='FreqPAC: # low-freq bands')
parser.add_argument('--n_bands_hi', type=int, default=2, help='FreqPAC: # high-freq bands')
parser.add_argument('--d_attn', type=int, default=64, help='FreqPAC: attention dim')
parser.add_argument('--warp_max', type=float, default=12.0, help='FreqWarpFlow: max delay in steps')
parser.add_argument('--warp_hidden', type=int, default=32, help='FreqWarpFlow: warp estimator hidden dim')
parser.add_argument('--tau_reg', type=float, default=0.001, help='FreqWarpFlowV2: tau L2 reg')
parser.add_argument('--rrf_beta_init', type=float, default=0.5, help='FreqWarpFlowV5: RRF beta init (threshold for warp engagement)')
parser.add_argument('--m_channel_tokens', type=int, default=4, help='FreqCTF: # channel tokens per channel')
parser.add_argument('--patch_dropout', type=float, default=0.2, help='FreqCTF: patch dropout for missingness simulation')
parser.add_argument('--n_layers', type=int, default=2, help='FreqCTF: # attention layers')
parser.add_argument('--phase_bias_scale', type=float, default=0.5, help='FreqCTF: phase bias init scale')
parser.add_argument('--warp_dropout', type=float, default=0.1, help='FreqWarpFlowV2: dropout in warp estimator')
parser.add_argument('--dmd_rank', type=int, default=32, help='FreqDMD: rank truncation')
parser.add_argument('--use_residual_nn', type=int, default=1, help='FreqDMD: enable NN residual head')
parser.add_argument('--n_sources', type=int, default=32, help='FreqICA: number of latent sources')
parser.add_argument('--div_lambda', type=float, default=0.01, help='FreqICA: source decorrelation weight')
parser.add_argument('--max_delay', type=float, default=24.0, help='FreqPhaseGraph(B): max |a| in steps')
parser.add_argument('--canon_type', type=str, default='margin', choices=['margin', 'entropy', 'sort'], help='CanonV2 sort regularizer type')
parser.add_argument('--sort_margin', type=float, default=0.01, help='Margin for margin-based sort reg in CanonV2')

# FreqSelectiveCross
parser.add_argument('--cross_mode', type=str, default='hybrid', choices=['none', 'herm', 'asym', 'hybrid'], help='FreqSelectiveCross: which cross branches to use')
parser.add_argument('--cross_diag_mode', type=str, default='offdiag', choices=['offdiag', 'full', 'diag'], help='FreqSelectiveCross: how to handle diagonal H_cc(f)')
parser.add_argument('--target_h_gate_init', type=float, default=-2.0, help='FreqSelectiveCross: Herm target-gate logit init')
parser.add_argument('--target_a_gate_init', type=float, default=-3.0, help='FreqSelectiveCross: Asym target-gate logit init')
parser.add_argument('--source_gate_init', type=float, default=0.0, help='FreqSelectiveCross: source-filter logit init')
parser.add_argument('--l1_target', type=float, default=1e-4, help='FreqSelectiveCross: L1 weight on target gates')
parser.add_argument('--l1_source', type=float, default=0.0, help='FreqSelectiveCross: L1 weight on source filter')
parser.add_argument('--use_source_filter', type=int, default=1, help='FreqSelectiveCross: 1 enables per-(band,src) source filter')
parser.add_argument('--use_target_gate', type=int, default=1, help='FreqSelectiveCross: 1 enables per-(band,tgt) target gate')

# FreqTemporalSelectiveCross (shares rank/gate_max/target_*_gate_init/source_gate_init/l1_*/use_source_filter/use_target_gate from FreqSelectiveCross)
parser.add_argument('--tcross_mode', type=str, default='hybrid', choices=['none', 'herm', 'asym', 'hybrid'], help='FreqTemporalSelectiveCross: which time-domain cross branches to use')
parser.add_argument('--tcross_diag_mode', type=str, default='offdiag', choices=['offdiag', 'full', 'diag'], help='FreqTemporalSelectiveCross: diag-handling for time-domain low-rank H')
parser.add_argument('--use_two_head_self', type=int, default=0, help='FreqTemporalSelectiveCross: 1 replaces cross with extra bias-free self head (capacity ablation)')

# TQNet_RKV
parser.add_argument('--kv_mode', type=str, default='raw_res_lag', choices=['raw', 'residual', 'raw_res', 'raw_res_lag'], help='TQNet_RKV: K/V feature stack')
parser.add_argument('--innovation_mode', type=str, default='smooth_residual', choices=['none', 'smooth_residual', 'diff'], help='TQNet_RKV: innovation e construction')
parser.add_argument('--smooth_kernel', type=int, default=25, help='TQNet_RKV: moving-average kernel for smooth_residual')
parser.add_argument('--use_lag_bank', type=int, default=1, help='TQNet_RKV: enable depthwise lag bank on innovation')
parser.add_argument('--lag_dilations', type=str, default='1,2,4,8,16,24', help='TQNet_RKV: comma-separated dilations for lag bank')
parser.add_argument('--lag_kernel_size', type=int, default=3, help='TQNet_RKV: depthwise lag conv kernel size')
parser.add_argument('--kv_init_identity', type=int, default=1, help='TQNet_RKV: init kv_proj so K,V approx raw at start')
parser.add_argument('--cross_gate_init', type=float, default=-2.0, help='TQNet_RKV: logit init for residual cross gate (sigmoid)')
parser.add_argument('--query_mode', type=str, default='hard', choices=['hard', 'sm', 'hybrid'], help='TQNet_RKV: temporal-query mode')
parser.add_argument('--sm_query_modes', type=int, default=8, help='TQNet_RKV: number of SM sinusoidal modes')
parser.add_argument('--sm_p_min', type=float, default=2.0, help='TQNet_RKV: SM minimum period')
parser.add_argument('--sm_p_max_mult', type=float, default=8.0, help='TQNet_RKV: SM max period = mult * seq_len')
parser.add_argument('--sm_query_gate_init', type=float, default=-2.0, help='TQNet_RKV: hybrid SM-query gate logit init')
parser.add_argument('--use_shared_query', type=int, default=0, help='TQNet_RKV: 1 shares query across channels (diagnostic)')
parser.add_argument('--log_attn_stats', type=int, default=0, help='TQNet_RKV: 1 enables attention entropy/max stats')
parser.add_argument('--tq_attn_heads', type=int, default=4, help='TQNet_RKV: attention head count for channelAggregator')

# SpecCoherenceRKV
parser.add_argument('--use_spec_coherence', type=int, default=1, help='SpecCoherenceRKV: 1 enables spectral coherence bias on attention logits')
parser.add_argument('--spec_num_bands', type=int, default=8, help='SpecCoherenceRKV: number of frequency bands')
parser.add_argument('--spec_smooth_kernel', type=int, default=25, help='SpecCoherenceRKV: moving-average kernel for innovation')
parser.add_argument('--spec_exclude_dc', type=int, default=1, help='SpecCoherenceRKV: drop DC bin from spectrum')
parser.add_argument('--spec_eps', type=float, default=1e-6, help='SpecCoherenceRKV: numerical eps')
parser.add_argument('--spec_use_innovation', type=int, default=1, help='SpecCoherenceRKV: 1 use innovation spectrum; 0 use raw x')
parser.add_argument('--spec_use_drift', type=int, default=0, help='SpecCoherenceRKV: 1 compute spectral drift for target gate')
parser.add_argument('--spec_drift_windows', type=int, default=2, help='SpecCoherenceRKV: temporal windows for drift')
parser.add_argument('--spec_bias_gate_init', type=float, default=-3.0, help='SpecCoherenceRKV: lambda_spec sigmoid logit init')
parser.add_argument('--spec_bias_gate_max', type=float, default=1.0, help='SpecCoherenceRKV: lambda_spec maximum')
parser.add_argument('--spec_band_weight_mode', type=str, default='softmax', choices=['softmax', 'softplus', 'free'], help='SpecCoherenceRKV: band weight parameterization')
parser.add_argument('--spec_learn_phase_lag', type=int, default=1, help='SpecCoherenceRKV: 1 learn per-band delta_m')
parser.add_argument('--spec_normalize_bias', type=int, default=1, help='SpecCoherenceRKV: normalize bias along source dim per target')
parser.add_argument('--spec_zero_diag', type=int, default=1, help='SpecCoherenceRKV: zero out diagonal of bias')
parser.add_argument('--spec_bias_type', type=str, default='energy_phase', choices=['energy', 'energy_phase'], help='SpecCoherenceRKV: bias formula')
parser.add_argument('--spec_disable_reliability', type=int, default=0, help='SpecCoherenceRKV: 1 set rho to 1 everywhere (ablation)')
parser.add_argument('--use_spec_source_gate', type=int, default=0, help='SpecCoherenceRKV: 1 enable source-reliability gate')
parser.add_argument('--use_spec_target_gate', type=int, default=0, help='SpecCoherenceRKV: 1 enable target-drift gate')
parser.add_argument('--spec_source_gate_init', type=float, default=0.0, help='SpecCoherenceRKV: source gate bias init')
parser.add_argument('--spec_target_gate_init', type=float, default=-2.0, help='SpecCoherenceRKV: target gate bias init')
parser.add_argument('--log_spec_stats', type=int, default=1, help='SpecCoherenceRKV: log spectral stats')

# SAC_RKV
parser.add_argument('--use_sac', type=int, default=1, help='SAC_RKV: 1 enables spectral activity controller')
parser.add_argument('--sac_adapter_mode', type=str, default='adapter', choices=['adapter', 'replace', 'off'], help='SAC_RKV: adapter / replace / off')
parser.add_argument('--sac_granularity', type=str, default='channel', choices=['channel', 'sample'], help='SAC_RKV: per-channel or per-sample expert mixing')
parser.add_argument('--sac_controller_hidden', type=int, default=64, help='SAC_RKV: controller MLP hidden width')
parser.add_argument('--sac_controller_dropout', type=float, default=0.0, help='SAC_RKV: controller dropout')
parser.add_argument('--sac_gate_init', type=float, default=-2.0, help='SAC_RKV: global cross-gate logit init')
parser.add_argument('--sac_raw_bias_init', type=float, default=2.0, help='SAC_RKV: raw-expert logit bias init')
parser.add_argument('--sac_res_bias_init', type=float, default=-1.0, help='SAC_RKV: residual-expert logit bias init')
parser.add_argument('--sac_lag_bias_init', type=float, default=-1.0, help='SAC_RKV: lag-expert logit bias init')
parser.add_argument('--sac_use_static_gate', type=int, default=0, help='SAC_RKV: 1 disables spec input (static gate ablation)')
parser.add_argument('--sac_random_stats', type=int, default=0, help='SAC_RKV: 1 shuffles spec features (random ablation)')
parser.add_argument('--sac_normalize_experts', type=int, default=1, help='SAC_RKV: normalize res/lag experts to raw RMS')
parser.add_argument('--sac_separate_kv_gates', type=int, default=0, help='SAC_RKV: 1 emits separate alpha_K and alpha_V')
parser.add_argument('--lag_combine_mode', type=str, default='softmax', choices=['mean', 'softmax', 'linear'], help='SAC_RKV: how to combine multi-dilation lag features')

# Spectral activity (shared with SAC_RKV)
parser.add_argument('--spec_use_entropy', type=int, default=1, help='SpecActivity: 1 include entropy in feat_channel')
parser.add_argument('--spec_use_phase', type=int, default=0, help='SpecActivity: 1 include phase concentration')
parser.add_argument('--spec_use_high_low_ratio', type=int, default=1, help='SpecActivity: 1 include low/mid/high band ratios')

# SAC freq loss (optional, not wired into training loop)
parser.add_argument('--use_sac_freq_loss', type=int, default=0, help='SAC_RKV: 1 enable band-wise frequency loss helper')
parser.add_argument('--sac_freq_loss_weight', type=float, default=0.0, help='SAC_RKV: weight for frequency_loss helper')
parser.add_argument('--sac_freq_loss_bands', type=int, default=8, help='SAC_RKV: number of bands for frequency loss')
parser.add_argument('--disable_res_expert', type=int, default=0, help='SAC_RKV/TC-SAC: 1 sets residual-expert logit to -1e4')

# TC_SAC_RKV (transfer/coherence calibration)
parser.add_argument('--use_transfer_calibration', type=int, default=1, help='TC-SAC: enable transfer/coherence calibration')
parser.add_argument('--transfer_num_bands', type=int, default=8, help='TC-SAC: number of bands for transfer stats')
parser.add_argument('--transfer_num_windows', type=int, default=4, help='TC-SAC: number of time windows for cross-spectrum')
parser.add_argument('--transfer_exclude_dc', type=int, default=1, help='TC-SAC: drop DC bin')
parser.add_argument('--transfer_eps', type=float, default=1e-6, help='TC-SAC: numerical eps')
parser.add_argument('--transfer_use_innovation', type=int, default=1, help='TC-SAC: 1 use innovation spectrum')
parser.add_argument('--transfer_detach_stats', type=int, default=1, help='TC-SAC: 1 detach transfer stats from autograd')
parser.add_argument('--transfer_topk', type=int, default=16, help='TC-SAC: top-k sources per target for aggregation')
parser.add_argument('--transfer_chunk_size', type=int, default=128, help='TC-SAC: target-channel chunk size for [B,C,C] safety')
parser.add_argument('--transfer_zero_diag', type=int, default=1, help='TC-SAC: zero c==j in coherence matrix')
parser.add_argument('--use_imag_coh', type=int, default=1, help='TC-SAC: 1 use imaginary coherence')
parser.add_argument('--use_real_coh', type=int, default=1, help='TC-SAC: 1 use real coherence')
parser.add_argument('--use_transfer_gain', type=int, default=1, help='TC-SAC: 1 use transfer gain |S|/P_src')
parser.add_argument('--use_lag_reliability_calibration', type=int, default=1, help='TC-SAC: 1 calibrate lag logit with imag/abs(coh)')
parser.add_argument('--lag_rel_logit_scale', type=float, default=2.0, help='TC-SAC: scale for lag-reliability logit add')
parser.add_argument('--lag_rel_center', type=float, default=0.25, help='TC-SAC: center for lag-reliability calibration')
parser.add_argument('--lag_rel_min', type=float, default=0.1, help='TC-SAC: post-hoc floor on alpha_lag')
parser.add_argument('--use_common_sync_suppression', type=int, default=1, help='TC-SAC: 1 subtract common_sync from lag logit')
parser.add_argument('--common_sync_scale', type=float, default=1.0, help='TC-SAC: scale for common-sync suppression')
parser.add_argument('--use_raw_lock_gate', type=int, default=1, help='TC-SAC: 1 multiply lambda by (1-alpha_raw)^gamma')
parser.add_argument('--raw_lock_gamma', type=float, default=1.0, help='TC-SAC: gamma exponent for raw-lock gate')
parser.add_argument('--use_effective_ratio_cap', type=int, default=1, help='TC-SAC: 1 cap ||lambda*z_sac||/||x|| per sample')
parser.add_argument('--sac_eff_ratio_cap', type=float, default=0.5, help='TC-SAC: effective-ratio upper bound')
parser.add_argument('--spec_context_len', type=int, default=0, help='TC-SAC: if >0, restrict spec/transfer to last spec_context_len timesteps')
parser.add_argument('--transfer_add_to_controller', type=int, default=1, help='TC-SAC: 1 concat transfer features into controller input')

# BSUA_RKV
parser.add_argument('--use_bsua', type=int, default=1, help='BSUA: enable budgeted spectral utility adapter')
parser.add_argument('--bsua_adapter_mode', type=str, default='adapter', choices=['adapter', 'replace', 'off'], help='BSUA: adapter / replace / off')
parser.add_argument('--bsua_legacy_raw_mix', type=int, default=0, help='BSUA: 1 = legacy SAC [raw,res,lag] mixture (ablation)')
parser.add_argument('--bsua_granularity', type=str, default='channel', choices=['channel', 'sample'], help='BSUA: per-channel or per-sample controller')
parser.add_argument('--bsua_controller_hidden', type=int, default=64, help='BSUA: controller MLP hidden')
parser.add_argument('--bsua_controller_dropout', type=float, default=0.0, help='BSUA: controller dropout')
parser.add_argument('--bsua_skip_bias_init', type=float, default=2.0, help='BSUA: skip-expert logit bias init')
parser.add_argument('--bsua_res_bias_init', type=float, default=-1.0, help='BSUA: residual-expert logit bias init')
parser.add_argument('--bsua_lag_bias_init', type=float, default=-1.0, help='BSUA: lag-expert logit bias init')
parser.add_argument('--bsua_gate_init', type=float, default=-2.0, help='BSUA: global adapter-gate logit init')
parser.add_argument('--bsua_gate_head_zero_init', type=int, default=1, help='BSUA: zero-init the gate head')
parser.add_argument('--bsua_skip_gamma', type=float, default=1.0, help='BSUA: gamma exponent for (1-alpha_skip)^gamma')
parser.add_argument('--bsua_use_static_gate', type=int, default=0, help='BSUA: 1 = static gate without spec input (ablation)')
parser.add_argument('--bsua_random_stats', type=int, default=0, help='BSUA: 1 = shuffle spec/transfer features (ablation)')
parser.add_argument('--bsua_normalize_experts', type=int, default=1, help='BSUA: 1 = renormalize res/lag to raw RMS')
parser.add_argument('--bsua_hard_skip_eval', type=int, default=0, help='BSUA: 1 = skip adapter at eval when alpha_skip > threshold (sample-level only)')
parser.add_argument('--bsua_hard_skip_threshold', type=float, default=0.95, help='BSUA: threshold for hard-skip in eval')
parser.add_argument('--bsua_eff_ratio_cap', type=float, default=0.5, help='BSUA: effective-ratio cap for adapter perturbation')
parser.add_argument('--use_dynamic_ratio_cap', type=int, default=0, help='BSUA: 1 = enable dynamic cap based on lag_rel - common_sync')
parser.add_argument('--bsua_eff_ratio_cap_min', type=float, default=0.25, help='BSUA: dynamic cap minimum')
parser.add_argument('--bsua_eff_ratio_cap_max', type=float, default=0.75, help='BSUA: dynamic cap maximum')
parser.add_argument('--bsua_dynamic_cap_scale', type=float, default=2.0, help='BSUA: dynamic cap sigmoid scale')
parser.add_argument('--bsua_dynamic_cap_center', type=float, default=0.0, help='BSUA: dynamic cap sigmoid center')
parser.add_argument('--disable_lag_expert', type=int, default=0, help='BSUA: 1 disables lag expert (sets logit to -1e4)')

# S2TV_BSUA_RKV
parser.add_argument('--use_s2tv', type=int, default=1, help='S2TV: enable spectral-Q / spectral-K / temporal-V adapter')
parser.add_argument('--s2tv_adapter_mode', type=str, default='adapter', choices=['adapter', 'replace', 'off'], help='S2TV: adapter / replace / off')
parser.add_argument('--s2tv_attn_mode', type=str, default='s2tv', choices=['legacy_temporal', 's2tv', 'hybrid_q', 'spec_only'], help='S2TV: adapter attention mode')
parser.add_argument('--s2tv_dim', type=int, default=64, help='S2TV: total embed dim for Q/K')
parser.add_argument('--s2tv_heads', type=int, default=4, help='S2TV: number of attention heads')
parser.add_argument('--s2tv_dropout', type=float, default=0.0, help='S2TV: attention dropout')
parser.add_argument('--s2tv_use_learned_channel_query', type=int, default=1, help='S2TV: 1 = add learnable per-channel Q embedding')
parser.add_argument('--s2tv_use_learned_channel_key', type=int, default=1, help='S2TV: 1 = add learnable per-channel K embedding')
parser.add_argument('--s2tv_use_tq_query_feature', type=int, default=1, help='S2TV: 1 = project TQ temporal query into Q')
parser.add_argument('--s2tv_use_transfer_in_key', type=int, default=1, help='S2TV: 1 = concat transfer stats into K input')
parser.add_argument('--s2tv_use_transfer_in_query', type=int, default=0, help='S2TV: 1 = concat transfer stats into Q input')
parser.add_argument('--s2tv_q_layernorm', type=int, default=1, help='S2TV: LayerNorm on Q')
parser.add_argument('--s2tv_k_layernorm', type=int, default=1, help='S2TV: LayerNorm on K')
parser.add_argument('--s2tv_value_out_proj', type=int, default=1, help='S2TV: identity-init Linear(T,T) on attention output')
parser.add_argument('--s2tv_temperature', type=float, default=1.0, help='S2TV: softmax temperature divisor')
parser.add_argument('--s2tv_source_topk', type=int, default=0, help='S2TV: keep top-k sources per target (0 = full attention)')
parser.add_argument('--use_transfer_stats', type=int, default=1, help='S2TV: enable transfer stats encoder')
parser.add_argument('--normalize_adapter_experts', type=int, default=1, help='S2TV: renormalize res/lag to raw RMS')

# Spectral auxiliary loss (model-agnostic — works on any model)
parser.add_argument('--use_spec_loss', type=int, default=0, help='Add a frequency-domain auxiliary loss to the training criterion')
parser.add_argument('--spec_loss_weight', type=float, default=0.5, help='Weight λ for spec_loss: loss = mse + λ·spec_loss')
parser.add_argument('--spec_loss_mode', type=str, default='amp', choices=['amp', 'log_amp', 'complex'], help='amp: ||Yp|-|Yt||^2; log_amp: ||log|Yp|-log|Yt|||^2; complex: |Yp-Yt|^2 (≈ time MSE)')
parser.add_argument('--spec_loss_band_weighting', type=str, default='uniform', choices=['uniform', 'lowpass', 'highpass'], help='band weighting: uniform / lowpass (emphasize low freq) / highpass')

# AsySpecXLocal
parser.add_argument('--use_local_cross', type=int, default=1, help='AsySpecXLocal: enable STFT-based local cross on input')
parser.add_argument('--keep_global_cross', type=int, default=1, help='AsySpecXLocal: keep original AsymCross on lifted output spectrum')
parser.add_argument('--stft_window', type=int, default=24, help='AsySpecXLocal: STFT window size')
parser.add_argument('--stft_stride', type=int, default=12, help='AsySpecXLocal: STFT hop length')
parser.add_argument('--local_gate_init', type=float, default=-2.0, help='AsySpecXLocal: local-cross gate logit init')
parser.add_argument('--local_rank', type=int, default=-1, help='AsySpecXLocal: local AsymCross rank (-1 = use --rank)')
parser.add_argument('--local_num_bands', type=int, default=-1, help='AsySpecXLocal: local AsymCross num_bands (-1 = use --num_bands)')

# TQNet_DC (Data-Conditional TWCM gating)
parser.add_argument('--use_dc_gate', type=int, default=1, help='TQNet_DC: enable data-conditional gate')
parser.add_argument('--dc_gate_alpha_init', type=float, default=10.0, help='TQNet_DC: gate sigmoid scale init')
parser.add_argument('--dc_gate_beta_init', type=float, default=-1.0, help='TQNet_DC: gate sigmoid offset init')
parser.add_argument('--dc_gate_center', type=float, default=0.6, help='TQNet_DC: entropy center')
parser.add_argument('--dc_num_bands', type=int, default=8, help='TQNet_DC: entropy num bands')
parser.add_argument('--dc_smooth_kernel', type=int, default=25, help='TQNet_DC: innovation smooth kernel')
parser.add_argument('--dc_exclude_dc', type=int, default=1, help='TQNet_DC: exclude DC bin in entropy')

# TQNet_JA (Joint-Axis TWCM)
parser.add_argument('--ja_delta_hidden', type=int, default=64, help='TQNet_JA: hidden dim of axis-1 delta MLP')
parser.add_argument('--ja_delta_zero_init', type=int, default=1, help='TQNet_JA: zero-init delta MLP final layer for graceful fallback')
parser.add_argument('--ja_use_scale_norm', type=int, default=1, help='TQNet_JA: internal per-channel scale normalisation')
parser.add_argument('--ja_band_gate_init', type=float, default=0.0, help='TQNet_JAv2: per-band gate logit init (default 0 → sigmoid 0.5)')
parser.add_argument('--ja_band_gate_low_init', type=float, default=-3.0, help='TQNet_JAv2: low-band gate logit init (default -3 → sigmoid 0.047)')
parser.add_argument('--ja_band_gate_low_count', type=int, default=1, help='TQNet_JAv2: number of lowest bands initialised suppressed')
parser.add_argument('--ja_freq_gate_init', type=float, default=0.0, help='TQNet_JAv3: per-bin gate logit init (default 0 → sigmoid 0.5)')
parser.add_argument('--ja_freq_gate_low_init', type=float, default=-3.0, help='TQNet_JAv3: low-bin gate logit init (default -3 → sigmoid 0.047)')
parser.add_argument('--ja_freq_gate_low_count', type=int, default=2, help='TQNet_JAv3: number of lowest bins initialised suppressed (k=0 DC + k=1)')

# JAX (Joint-Axis eXtrapolator standalone forecaster)
parser.add_argument('--jax_window', type=int, default=-1, help='JAX: STFT window (default auto = min(sl//4, 64))')
parser.add_argument('--jax_stride', type=int, default=-1, help='JAX: STFT stride (default W//2)')
parser.add_argument('--jax_rank', type=int, default=8, help='JAX: rank-R bottleneck')
parser.add_argument('--jax_delta_hidden', type=int, default=64, help='JAX: delta MLP hidden')
parser.add_argument('--jax_smooth_kernel', type=int, default=25, help='JAX: spec feature smoothing kernel')

# SpecQ (Spectral-Query forecaster)
parser.add_argument('--specq_window', type=int, default=-1, help='SpecQ: STFT window')
parser.add_argument('--specq_stride', type=int, default=-1, help='SpecQ: STFT stride')
parser.add_argument('--specq_rank', type=int, default=8, help='SpecQ: rank-R')

# JointMLP (TQNet MLP backbone + JA cross-channel, no cycle, no channel attn)
parser.add_argument('--jmlp_window', type=int, default=-1, help='JointMLP: STFT window')
parser.add_argument('--jmlp_stride', type=int, default=-1, help='JointMLP: STFT stride')
parser.add_argument('--jmlp_rank', type=int, default=8, help='JointMLP: rank-R')
parser.add_argument('--jmlp_delta_hidden', type=int, default=64, help='JointMLP: delta MLP hidden')
parser.add_argument('--jmlp_gate_init', type=float, default=-1.0, help='JointMLP: JA gate init logit')
parser.add_argument('--jmlp_use_entropy_gate', type=int, default=1, help='JointMLP: per-channel entropy gate')
parser.add_argument('--jmlp_innovation_only', type=int, default=0, help='JointMLP: apply JA only on x - moving_avg(x) (innovation)')
parser.add_argument('--jmlp_innovation_kernel', type=int, default=25, help='JointMLP: moving average kernel size for innovation extraction')
parser.add_argument('--jmlp_ja_version', type=int, default=3, help='JointMLP: 3 = per-bin g (t-invariant), 4 = per-bin per-frame g (t-varying)')

# PredFlow (Predictive Transfer Kernel)
parser.add_argument('--pf_d_model', type=int, default=64, help='PredFlow: token dim')
parser.add_argument('--pf_use_cross', type=int, default=1, help='PredFlow: enable cross-channel transfer path')
parser.add_argument('--pf_use_self_mask', type=int, default=1, help='PredFlow: exclude self in cross path')
parser.add_argument('--pf_chunk_i', type=int, default=-1, help='PredFlow: target-channel chunk size for memory; -1 = auto')

# ComplexFlow (Complex-plane Transition forecaster)
parser.add_argument('--cf_d_model', type=int, default=64, help='ComplexFlow: latent dim D')
parser.add_argument('--cf_d_complex', type=int, default=32, help='ComplexFlow: complex space dim Dc')
parser.add_argument('--cf_max_gamma', type=float, default=0.1, help='ComplexFlow: max residual scale γ')
parser.add_argument('--cf_max_log_scale', type=float, default=0.2, help='ComplexFlow: max complex log-magnitude shift')
parser.add_argument('--cf_max_phase_shift', type=float, default=0.628, help='ComplexFlow: max complex phase shift (rad)')
parser.add_argument('--cf_top_k', type=int, default=-1, help='ComplexFlow: top-k sources (-1 = all)')
parser.add_argument('--cf_window', type=int, default=-1, help='ComplexFlow: STFT window (-1 = auto)')
parser.add_argument('--cf_chunk_i', type=int, default=-1, help='ComplexFlow: target-channel chunk size (-1 = auto)')

# SegRNN
parser.add_argument('--rnn_type', default='gru', help='rnn_type')
parser.add_argument('--dec_way', default='pmf', help='decode way')
parser.add_argument('--seg_len', type=int, default=48, help='segment length')
parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main


if args.is_training:
    for ii in range(args.itr):

        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.cycle,
            fix_seed)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.cycle,
        fix_seed)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
