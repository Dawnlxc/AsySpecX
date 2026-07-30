# ICLR 2026 lightweight baseline reproduction

Selection rule: for duplicate candidates in one cell, choose the minimum validation loss; never select on test MSE.

| Model | Dataset | L->H | Actual MSE | Paper MSE | Asy1 MSE | Delta vs Asy1 | Params (numel) | Asy1 params |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mixlinear | electricity | 720->96 | 0.138606 | 0.138 | 0.128007 | 0.010599 | 95 | 80592 |
| mixlinear | electricity | 720->192 | 0.154282 | 0.154 | 0.145293 | 0.008989 | 107 | 86736 |
| mixlinear | electricity | 720->336 | 0.170718 | 0.170 | 0.160480 | 0.010238 | 131 | 95952 |
| mixlinear | electricity | 720->720 | 0.209489 | 0.209 | 0.197044 | 0.012445 | 187 | 120528 |
| mixlinear | weather | 720->96 | 0.179305 | 0.170 | 0.139562 | 0.039743 | 195 | 13722 |
| mixlinear | weather | 720->192 | 0.221808 | 0.212 | 0.181457 | 0.040351 | 299 | 15258 |
| mixlinear | weather | 720->336 | 0.267432 | 0.257 | 0.231628 | 0.035804 | 455 | 17562 |
| mixlinear | weather | 720->720 | 0.329212 | 0.321 | 0.304462 | 0.024750 | 759 | 23706 |
| phaseformer | electricity | 720->96 | 0.128465 | 0.129 | 0.128007 | 0.000458 | 3666 | 80592 |
| phaseformer | electricity | 720->192 | 0.146529 | 0.148 | 0.145293 | 0.001236 | 273160 | 86736 |
| phaseformer | electricity | 720->336 | 0.166116 | 0.165 | 0.160480 | 0.005636 | 3756 | 95952 |
| phaseformer | electricity | 720->720 | 0.198819 | 0.201 | 0.197044 | 0.001775 | 275998 | 120528 |
| phaseformer | weather | 720->96 | 0.148620 | 0.148 | 0.139562 | 0.009058 | 5616 | 13722 |
| phaseformer | weather | 720->192 | 0.192744 | 0.193 | 0.181457 | 0.011287 | 3702 | 15258 |
| phaseformer | weather | 720->336 | 0.245097 | 0.242 | 0.231628 | 0.013469 | 3756 | 17562 |
| phaseformer | weather | 720->720 | 0.320802 | 0.309 | 0.304462 | 0.016340 | 3900 | 23706 |

Paper parameter references: MixLinear reports 0.176K at horizon 720; PhaseFormer Table 4 reports 308 on Weather and 1.156K on Electricity for the 720->96 efficiency setting. Actual code-instantiated counts above are the audit authority for this reproduction.

Completed selected cells: 16/16
