from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import AsySpecX
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'AsySpecX': AsySpecX,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _resolve_protocol_channels(self, C):
        """Return list of channel indices selected by P3 (modality) / P4 (node) using channel_meta.csv."""
        meta_path = getattr(self.args, 'channel_meta', '') or ''
        mod = getattr(self.args, 'mask_modality', '') or ''
        node = getattr(self.args, 'mask_node', '') or ''
        if not meta_path or (not mod and not node):
            return []
        if not hasattr(self, '_channel_meta_cache'):
            import pandas as _pd
            self._channel_meta_cache = _pd.read_csv(meta_path)
        df = self._channel_meta_cache
        sel = df.copy()
        if mod:
            sel = sel[sel['modality'].astype(str) == mod]
        if node:
            sel = sel[sel['node_id'].astype(str) == node]
        idx = sel['channel_idx'].astype(int).tolist()
        return [c for c in idx if 0 <= c < C]

    def _maybe_temporal_mask(self, batch_x):
        """Random temporal masking — zero-fill r fraction of input timesteps per sample.

        Applied to all channels at the picked timesteps. Deterministic in vali/test
        via --mask_seed. Persistent at train if --mask_persistent_train=1 (default).
        """
        ratio = float(getattr(self.args, 'temporal_mask_ratio', 0.0))
        if ratio <= 0.0:
            return batch_x
        if self.model.training and not int(getattr(self.args, 'mask_persistent_train', 1)):
            return batch_x
        B, L, C = batch_x.shape
        n_mask = max(0, int(round(ratio * L)))
        if n_mask <= 0:
            return batch_x
        device = batch_x.device
        batch_x = batch_x.clone()
        if not self.model.training:
            g = torch.Generator(device='cpu').manual_seed(int(getattr(self.args, 'mask_seed', 2026)))
        else:
            g = None
        for b in range(B):
            perm = torch.randperm(L, generator=g) if g is not None else torch.randperm(L)
            pick = perm[:n_mask].to(device)
            batch_x[b, pick, :] = 0.0
        return batch_x

    def _maybe_conditional_mask(self, batch_x, batch_y):
        """Structured sensor outage masking.

        Flags:
          --conditional 1           legacy k=1 random channel
          --mask_ratio r            mask round(r*C) random channels per sample
          --mask_held_out "i,j,..." always-masked channels (mode B / P4-static)
          --mask_protocol P3|P4     plus --channel_meta + (--mask_modality | --mask_node)
          --mask_recent_r N         mask only last N timesteps of failed channels (else full L)
          --mask_persistent_train 0 only mask at vali/test (cold mode); 1 also at train
          --mask_seed N             deterministic mask in vali/test

        Loss is scored only on masked-future positions.
        """
        cond = getattr(self.args, 'conditional', 0)
        ratio = float(getattr(self.args, 'mask_ratio', 0.0))
        held = (getattr(self.args, 'mask_held_out', '') or '').strip()
        proto = (getattr(self.args, 'mask_protocol', '') or '').strip()
        recent_r = int(getattr(self.args, 'mask_recent_r', 0))
        # Resolve P3/P4 deterministic channel set
        proto_idx = []
        if proto in ('P3', 'P4'):
            proto_idx = self._resolve_protocol_channels(batch_x.shape[-1])
        any_mask = cond or ratio > 0.0 or held or proto_idx
        if not any_mask:
            return batch_x, None
        # Mode A2 cold: skip masking at train (only static held-out / proto applies always)
        if self.model.training and not int(getattr(self.args, 'mask_persistent_train', 1)) and not held and not proto_idx:
            return batch_x, None
        B, L, C = batch_x.shape
        device = batch_x.device
        # Time-window for masking: last r steps if recent_r>0, else full L
        s_lo = max(0, L - recent_r) if recent_r > 0 else 0
        target_mask = torch.zeros((B, batch_y.shape[1], C), device=device)
        batch_x = batch_x.clone()

        static_idx = []
        if held:
            static_idx += [int(x) for x in held.split(',') if x.strip() != '']
        if proto_idx:
            static_idx += proto_idx
        static_idx = sorted(set(static_idx))
        for c in static_idx:
            batch_x[:, s_lo:, c] = 0.0
            target_mask[:, :, c] = 1.0

        # Random k masking
        k = 0
        if ratio > 0.0:
            k = max(1, int(round(ratio * C)))
        elif cond:
            k = 1
        if k > 0:
            avail = [c for c in range(C) if c not in static_idx]
            if len(avail) >= k:
                if not self.model.training:
                    g = torch.Generator(device='cpu').manual_seed(int(getattr(self.args, 'mask_seed', 2026)))
                else:
                    g = None
                avail_t = torch.tensor(avail, dtype=torch.long)
                for b in range(B):
                    perm = torch.randperm(len(avail_t), generator=g) if g is not None else torch.randperm(len(avail_t))
                    pick = avail_t[perm[:k]].to(device)
                    batch_x[b, s_lo:, pick] = 0.0
                    target_mask[b, :, pick] = 1.0
        return batch_x, target_mask

    def _maybe_masked_loss(self, outputs, target, target_mask, criterion):
        if target_mask is None:
            return criterion(outputs, target)
        f_dim = -1 if self.args.features == 'MS' else 0
        m = target_mask[:, -self.args.pred_len:, f_dim:]
        se = (outputs - target) ** 2
        return (se * m).sum() / m.sum().clamp_min(1.0)

    def _spectral_aux_loss(self, pred, true, eps=1e-6):
        """Frequency-domain auxiliary loss on (pred, true).

        Both are [B, pred_len, C]; we rFFT along time.
        mode:
            "amp"     : (|Yp| - |Yt|)^2          (phase-agnostic, the usual spec loss)
            "log_amp" : (log|Yp| - log|Yt|)^2    (emphasize low-energy bands)
            "complex" : |Yp - Yt|^2              (≈ time-MSE by Parseval, mostly redundant)
        """
        if pred.size(1) < 4:
            return pred.new_zeros(())
        mode = str(getattr(self.args, "spec_loss_mode", "amp"))
        weight_mode = str(getattr(self.args, "spec_loss_band_weighting", "uniform"))
        lam = float(getattr(self.args, "spec_loss_weight", 0.5))
        if lam <= 0.0:
            return pred.new_zeros(())
        Yp = torch.fft.rfft(pred, dim=1)
        Yt = torch.fft.rfft(true, dim=1)
        if mode == "amp":
            diff = (Yp.abs() - Yt.abs()).pow(2)
        elif mode == "log_amp":
            diff = (torch.log(Yp.abs() + eps) - torch.log(Yt.abs() + eps)).pow(2)
        else:  # complex
            diff = (Yp - Yt).abs().pow(2)
        # diff: [B, F, C]
        if weight_mode == "uniform":
            spec_loss = diff.mean()
        else:
            F_ = diff.size(1)
            idx = torch.arange(F_, device=diff.device, dtype=diff.dtype)
            if weight_mode == "lowpass":
                w = torch.exp(-idx / max(F_ // 4, 1))   # decay with freq
            else:  # highpass
                w = torch.exp(-(F_ - 1 - idx) / max(F_ // 4, 1))
            w = w / w.sum().clamp_min(eps) * F_
            spec_loss = (diff * w.view(1, -1, 1)).mean()
        return lam * spec_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # Conditional MTSF mask (use deterministic seed for reproducible vali)
                _by_dev = batch_y.to(self.device)
                batch_x, _cond_mask = self._maybe_conditional_mask(batch_x, _by_dev)
                batch_x = self._maybe_temporal_mask(batch_x)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ', 'FreqHermCycle', 'FreqLagCycle', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTFCycle', 'PeriodCrossDecomp', 'SpecCoherenceRKV', 'SAC_RKV', 'TC_SAC_RKV', 'BSUA_RKV', 'S2TV_BSUA_RKV'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'FITS', 'FreTS', 'Filter', 'SparseTSF', 'MixLinear', 'AsySpecX', 'AsySpecXLocal', 'AsySpecXResid', 'JAX', 'SpecQ', 'SpecFlow', 'PredFlow', 'PredFlowS', 'ComplexFlow', 'CoSTDriver', 'VanillaTC', 'VanillaFreqTC', 'FreqHermV2', 'FreqLagAttn', 'FreqLagAlign', 'FreqLagAlignV2', 'FreqPAC', 'FreqWarpFlow', 'FreqPACv2', 'FreqDMD', 'FreqCepstrum', 'FreqICA', 'FreqWarpFlowV2', 'FreqWarpFlowV3', 'FreqWarpFlowV5', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTF', 'FreqCTFSimple', 'FreqCTFPhase', 'FreqCTFCycle', 'PeriodCrossDecomp', 'FITSCross', 'FreqMLP', 'FreqHerm', 'FreqAsym', 'FreqHermCC', 'FreqHermPUG', 'FreqHermPUGv2', 'FreqHermAPG', 'FreqHermGDG', 'FreqHermSpLR', 'FreqPhaseLag', 'FreqContPhaseLag', 'FreqDirAware', 'FreqContDirAware', 'FreqContDirAwareAH', 'FreqAHv2', 'FreqContDirAwareCanon', 'FreqContDirAwareCanonV2', 'FreqContDirAwareCanonL1', 'FreqDualHead', 'FreqDynDualHead', 'FreqSymOnly', 'FreqAsymOnly', 'FreqSymPhi', 'FreqSymLag', 'FreqPhaseGraph', 'FreqPhaseAttn', 'FreqPhaseGraphB', 'FreqHCoupling', 'FreqOracleMix', 'FreqOracleMixV2', 'FreqHermFITS', 'FreqHermFITSPre', 'PatchPhase', 'FreqPatchSpLR', 'FreqSTFTSSM', 'FreqSelectiveCross', 'FreqTemporalSelectiveCross'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ', 'FreqHermCycle', 'FreqLagCycle', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTFCycle', 'PeriodCrossDecomp', 'SpecCoherenceRKV', 'SAC_RKV', 'TC_SAC_RKV', 'BSUA_RKV', 'S2TV_BSUA_RKV'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in
                             {'Linear', 'MLP', 'SegRNN', 'TST', 'FITS', 'FreTS', 'Filter', 'SparseTSF', 'MixLinear', 'AsySpecX', 'AsySpecXLocal', 'AsySpecXResid', 'JAX', 'SpecQ', 'SpecFlow', 'PredFlow', 'PredFlowS', 'ComplexFlow', 'CoSTDriver', 'VanillaTC', 'VanillaFreqTC', 'FreqHermV2', 'FreqLagAttn', 'FreqLagAlign', 'FreqLagAlignV2', 'FreqPAC', 'FreqWarpFlow', 'FreqPACv2', 'FreqDMD', 'FreqCepstrum', 'FreqICA', 'FreqWarpFlowV2', 'FreqWarpFlowV3', 'FreqWarpFlowV5', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTF', 'FreqCTFSimple', 'FreqCTFPhase', 'FreqCTFCycle', 'PeriodCrossDecomp', 'FITSCross', 'FreqMLP', 'FreqHerm', 'FreqAsym', 'FreqHermCC', 'FreqHermPUG', 'FreqHermPUGv2', 'FreqHermAPG', 'FreqHermGDG', 'FreqHermSpLR', 'FreqPhaseLag', 'FreqContPhaseLag', 'FreqDirAware', 'FreqContDirAware', 'FreqContDirAwareAH', 'FreqAHv2', 'FreqContDirAwareCanon', 'FreqContDirAwareCanonV2', 'FreqContDirAwareCanonL1', 'FreqDualHead', 'FreqDynDualHead', 'FreqSymOnly', 'FreqAsymOnly', 'FreqSymPhi', 'FreqSymLag', 'FreqPhaseGraph', 'FreqPhaseAttn', 'FreqPhaseGraphB', 'FreqHCoupling', 'FreqOracleMix', 'FreqOracleMixV2', 'FreqHermFITS', 'FreqHermFITSPre', 'PatchPhase', 'FreqPatchSpLR', 'FreqSTFTSSM', 'FreqSelectiveCross', 'FreqTemporalSelectiveCross'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if _cond_mask is not None:
                    loss_t = self._maybe_masked_loss(outputs, batch_y, _cond_mask, criterion)
                    loss = loss_t.detach().cpu()
                else:
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        n_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('n_param: {}'.format(n_param))
        train_start = time.time()
        time_now = train_start

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # Conditional MTSF: zero a random channel in input; loss only on that channel
                batch_x, _cond_mask = self._maybe_conditional_mask(batch_x, batch_y)
                batch_x = self._maybe_temporal_mask(batch_x)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ', 'FreqHermCycle', 'FreqLagCycle', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTFCycle', 'PeriodCrossDecomp', 'SpecCoherenceRKV', 'SAC_RKV', 'TC_SAC_RKV', 'BSUA_RKV', 'S2TV_BSUA_RKV'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif 'FreqHermAPG' in self.args.model:
                            outputs = self.model(batch_x, batch_y.to(self.device))
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'FITS', 'FreTS', 'Filter', 'SparseTSF', 'MixLinear', 'AsySpecX', 'AsySpecXLocal', 'AsySpecXResid', 'JAX', 'SpecQ', 'SpecFlow', 'PredFlow', 'PredFlowS', 'ComplexFlow', 'CoSTDriver', 'VanillaTC', 'VanillaFreqTC', 'FreqHermV2', 'FreqLagAttn', 'FreqLagAlign', 'FreqLagAlignV2', 'FreqPAC', 'FreqWarpFlow', 'FreqPACv2', 'FreqDMD', 'FreqCepstrum', 'FreqICA', 'FreqWarpFlowV2', 'FreqWarpFlowV3', 'FreqWarpFlowV5', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTF', 'FreqCTFSimple', 'FreqCTFPhase', 'FreqCTFCycle', 'PeriodCrossDecomp', 'FITSCross', 'FreqMLP', 'FreqHerm', 'FreqAsym', 'FreqHermCC', 'FreqHermPUG', 'FreqHermPUGv2', 'FreqHermGDG', 'FreqHermSpLR', 'FreqPhaseLag', 'FreqContPhaseLag', 'FreqDirAware', 'FreqContDirAware', 'FreqContDirAwareAH', 'FreqAHv2', 'FreqContDirAwareCanon', 'FreqContDirAwareCanonV2', 'FreqContDirAwareCanonL1', 'FreqDualHead', 'FreqDynDualHead', 'FreqSymOnly', 'FreqAsymOnly', 'FreqSymPhi', 'FreqSymLag', 'FreqPhaseGraph', 'FreqPhaseAttn', 'FreqPhaseGraphB', 'FreqHCoupling', 'FreqOracleMix', 'FreqOracleMixV2', 'FreqHermFITS', 'FreqHermFITSPre', 'PatchPhase', 'FreqPatchSpLR', 'FreqSTFTSSM', 'FreqSelectiveCross', 'FreqTemporalSelectiveCross'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = self._maybe_masked_loss(outputs, batch_y, _cond_mask, criterion)
                        if hasattr(self.model, "aux_loss") and self.model.aux_loss is not None:
                            loss = loss + self.model.aux_loss
                        if bool(getattr(self.args, "use_spec_loss", 0)):
                            loss = loss + self._spectral_aux_loss(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ', 'FreqHermCycle', 'FreqLagCycle', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTFCycle', 'PeriodCrossDecomp', 'SpecCoherenceRKV', 'SAC_RKV', 'TC_SAC_RKV', 'BSUA_RKV', 'S2TV_BSUA_RKV'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif 'FreqHermAPG' in self.args.model:
                        outputs = self.model(batch_x, batch_y.to(self.device))
                    elif any(substr in self.args.model for substr in
                             {'Linear', 'MLP', 'SegRNN', 'TST', 'FITS', 'FreTS', 'Filter', 'SparseTSF', 'MixLinear', 'AsySpecX', 'AsySpecXLocal', 'AsySpecXResid', 'JAX', 'SpecQ', 'SpecFlow', 'PredFlow', 'PredFlowS', 'ComplexFlow', 'CoSTDriver', 'VanillaTC', 'VanillaFreqTC', 'FreqHermV2', 'FreqLagAttn', 'FreqLagAlign', 'FreqLagAlignV2', 'FreqPAC', 'FreqWarpFlow', 'FreqPACv2', 'FreqDMD', 'FreqCepstrum', 'FreqICA', 'FreqWarpFlowV2', 'FreqWarpFlowV3', 'FreqWarpFlowV5', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTF', 'FreqCTFSimple', 'FreqCTFPhase', 'FreqCTFCycle', 'PeriodCrossDecomp', 'FITSCross', 'FreqMLP', 'FreqHerm', 'FreqAsym', 'FreqHermCC', 'FreqHermPUG', 'FreqHermPUGv2', 'FreqHermGDG', 'FreqHermSpLR', 'FreqPhaseLag', 'FreqContPhaseLag', 'FreqDirAware', 'FreqContDirAware', 'FreqContDirAwareAH', 'FreqAHv2', 'FreqContDirAwareCanon', 'FreqContDirAwareCanonV2', 'FreqContDirAwareCanonL1', 'FreqDualHead', 'FreqDynDualHead', 'FreqSymOnly', 'FreqAsymOnly', 'FreqSymPhi', 'FreqSymLag', 'FreqPhaseGraph', 'FreqPhaseAttn', 'FreqPhaseGraphB', 'FreqHCoupling', 'FreqOracleMix', 'FreqOracleMixV2', 'FreqHermFITS', 'FreqHermFITSPre', 'PatchPhase', 'FreqPatchSpLR', 'FreqSTFTSSM', 'FreqSelectiveCross', 'FreqTemporalSelectiveCross'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self._maybe_masked_loss(outputs, batch_y, _cond_mask, criterion)
                    if hasattr(self.model, "aux_loss") and self.model.aux_loss is not None:
                        loss = loss + self.model.aux_loss
                    if bool(getattr(self.args, "use_spec_loss", 0)):
                        loss = loss + self._spectral_aux_loss(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        t_train = time.time() - train_start
        print('t_train: {:.4f}'.format(t_train))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        test_start = time.time()
        n_test_samples = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                n_test_samples += batch_x.size(0)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # Conditional MTSF mask
                batch_x, _cond_mask = self._maybe_conditional_mask(batch_x, batch_y)
                batch_x = self._maybe_temporal_mask(batch_x)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ', 'FreqHermCycle', 'FreqLagCycle', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTFCycle', 'PeriodCrossDecomp', 'SpecCoherenceRKV', 'SAC_RKV', 'TC_SAC_RKV', 'BSUA_RKV', 'S2TV_BSUA_RKV'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'FITS', 'FreTS', 'Filter', 'SparseTSF', 'MixLinear', 'AsySpecX', 'AsySpecXLocal', 'AsySpecXResid', 'JAX', 'SpecQ', 'SpecFlow', 'PredFlow', 'PredFlowS', 'ComplexFlow', 'CoSTDriver', 'VanillaTC', 'VanillaFreqTC', 'FreqHermV2', 'FreqLagAttn', 'FreqLagAlign', 'FreqLagAlignV2', 'FreqPAC', 'FreqWarpFlow', 'FreqPACv2', 'FreqDMD', 'FreqCepstrum', 'FreqICA', 'FreqWarpFlowV2', 'FreqWarpFlowV3', 'FreqWarpFlowV5', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTF', 'FreqCTFSimple', 'FreqCTFPhase', 'FreqCTFCycle', 'PeriodCrossDecomp', 'FITSCross', 'FreqMLP', 'FreqHerm', 'FreqAsym', 'FreqHermCC', 'FreqHermPUG', 'FreqHermPUGv2', 'FreqHermAPG', 'FreqHermGDG', 'FreqHermSpLR', 'FreqPhaseLag', 'FreqContPhaseLag', 'FreqDirAware', 'FreqContDirAware', 'FreqContDirAwareAH', 'FreqAHv2', 'FreqContDirAwareCanon', 'FreqContDirAwareCanonV2', 'FreqContDirAwareCanonL1', 'FreqDualHead', 'FreqDynDualHead', 'FreqSymOnly', 'FreqAsymOnly', 'FreqSymPhi', 'FreqSymLag', 'FreqPhaseGraph', 'FreqPhaseAttn', 'FreqPhaseGraphB', 'FreqHCoupling', 'FreqOracleMix', 'FreqOracleMixV2', 'FreqHermFITS', 'FreqHermFITSPre', 'PatchPhase', 'FreqPatchSpLR', 'FreqSTFTSSM', 'FreqSelectiveCross', 'FreqTemporalSelectiveCross'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ', 'FreqHermCycle', 'FreqLagCycle', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTFCycle', 'PeriodCrossDecomp', 'SpecCoherenceRKV', 'SAC_RKV', 'TC_SAC_RKV', 'BSUA_RKV', 'S2TV_BSUA_RKV'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in
                             {'Linear', 'MLP', 'SegRNN', 'TST', 'FITS', 'FreTS', 'Filter', 'SparseTSF', 'MixLinear', 'AsySpecX', 'AsySpecXLocal', 'AsySpecXResid', 'JAX', 'SpecQ', 'SpecFlow', 'PredFlow', 'PredFlowS', 'ComplexFlow', 'CoSTDriver', 'VanillaTC', 'VanillaFreqTC', 'FreqHermV2', 'FreqLagAttn', 'FreqLagAlign', 'FreqLagAlignV2', 'FreqPAC', 'FreqWarpFlow', 'FreqPACv2', 'FreqDMD', 'FreqCepstrum', 'FreqICA', 'FreqWarpFlowV2', 'FreqWarpFlowV3', 'FreqWarpFlowV5', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTF', 'FreqCTFSimple', 'FreqCTFPhase', 'FreqCTFCycle', 'PeriodCrossDecomp', 'FITSCross', 'FreqMLP', 'FreqHerm', 'FreqAsym', 'FreqHermCC', 'FreqHermPUG', 'FreqHermPUGv2', 'FreqHermAPG', 'FreqHermGDG', 'FreqHermSpLR', 'FreqPhaseLag', 'FreqContPhaseLag', 'FreqDirAware', 'FreqContDirAware', 'FreqContDirAwareAH', 'FreqAHv2', 'FreqContDirAwareCanon', 'FreqContDirAwareCanonV2', 'FreqContDirAwareCanonL1', 'FreqDualHead', 'FreqDynDualHead', 'FreqSymOnly', 'FreqAsymOnly', 'FreqSymPhi', 'FreqSymLag', 'FreqPhaseGraph', 'FreqPhaseAttn', 'FreqPhaseGraphB', 'FreqHCoupling', 'FreqOracleMix', 'FreqOracleMixV2', 'FreqHermFITS', 'FreqHermFITSPre', 'PatchPhase', 'FreqPatchSpLR', 'FreqSTFTSSM', 'FreqSelectiveCross', 'FreqTemporalSelectiveCross'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if _cond_mask is not None:
                    # In conditional mode, score only on masked channels;
                    # multiply outputs and batch_y by mask so downstream metrics get masked entries
                    m = _cond_mask[:, -self.args.pred_len:, f_dim:]
                    # Mark unmasked channels as 0 (will be excluded via mask reduction below)
                    outputs = outputs * m
                    batch_y = batch_y * m
                    # Save mask for metric scaling
                    if not hasattr(self, '_cond_mask_accum'):
                        self._cond_mask_accum = []
                    self._cond_mask_accum.append(m.detach().cpu().numpy())

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    # np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        ### denorm ###
        # denorm_preds = np.stack([test_data.inverse_transform(pred) for pred in preds])
        # denorm_trues = np.stack([test_data.inverse_transform(true) for true in trues])

        ### denorm ###

        t_inf = time.time() - test_start
        print('t_inf: {:.4f} (n={})'.format(t_inf, n_test_samples))

        # result save (folder already created above; metrics + visuals share one dir)
        if hasattr(self, '_cond_mask_accum') and self._cond_mask_accum:
            masks = np.concatenate(self._cond_mask_accum, axis=0)
            masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
            denom = max(masks.sum(), 1.0)
            mse = float((((preds - trues) ** 2) * masks).sum() / denom)
            mae = float((np.abs(preds - trues) * masks).sum() / denom)
            rmse = float(np.sqrt(mse))
            mape = mspe = rse = corr = 0.0
            self._cond_mask_accum = []
            print('[conditional] masked metric:')
        else:
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ', 'FreqHermCycle', 'FreqLagCycle', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTFCycle', 'PeriodCrossDecomp', 'SpecCoherenceRKV', 'SAC_RKV', 'TC_SAC_RKV', 'BSUA_RKV', 'S2TV_BSUA_RKV'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ', 'FreqHermCycle', 'FreqLagCycle', 'PeriodCross', 'PeriodCrossRRF', 'FreqCTFCycle', 'PeriodCrossDecomp', 'SpecCoherenceRKV', 'SAC_RKV', 'TC_SAC_RKV', 'BSUA_RKV', 'S2TV_BSUA_RKV'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
