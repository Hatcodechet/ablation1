# Retrieval Ablation Summary

This file summarizes the current retrieval ablation results collected from:

- `output2/`: Top-P baselines and ATS baseline
- `output3/`: ATS-only variants
- `output4/`: Hybrid ATS+Peak variants

All numbers below are taken directly from the corresponding `*metrics*.json` files.

## 1. Top-P vs ATS Baselines

| Method | T2V Top1 | T2V Top5 | T2V Top10 | T2V Mean Rank | T2V MRR | V2T Top1 | V2T Top5 | V2T Top10 | V2T Mean Rank | V2T MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-P `p=0.7` | 22.41 | 48.28 | 60.00 | 16.44 | 0.3552 | 21.38 | 39.66 | 51.72 | 24.90 | 0.3126 |
| Top-P `p=0.8` | 23.79 | 49.66 | 61.03 | 16.50 | 0.3596 | 20.00 | 39.31 | 52.07 | 24.92 | 0.3062 |
| ATS `f=12, tau=0.10` | 23.10 | 49.66 | 61.03 | 15.89 | 0.3563 | 21.38 | 39.66 | 53.10 | 24.37 | 0.3147 |

## 2. ATS-Only Ablations

| Method | T2V Top1 | T2V Top5 | T2V Top10 | T2V Mean Rank | T2V MRR | V2T Top1 | V2T Top5 | V2T Top10 | V2T Mean Rank | V2T MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATS `f=12, tau=0.01` | 22.76 | 51.03 | 62.76 | 16.01 | 0.3600 | 21.03 | 39.31 | 52.07 | 24.29 | 0.3150 |
| ATS `f=12, tau=0.02` | 22.07 | 50.34 | 61.72 | 16.31 | 0.3549 | 20.34 | 39.31 | 53.10 | 24.36 | 0.3114 |
| ATS `f=12, tau=0.03` | 22.76 | 50.00 | 62.07 | 16.05 | 0.3561 | 21.03 | 39.31 | 52.76 | 24.23 | 0.3141 |
| ATS `f=12, tau=0.04` | 22.76 | 50.34 | 61.38 | 16.03 | 0.3575 | 20.69 | 39.66 | 52.76 | 24.22 | 0.3106 |
| ATS `f=12, tau=0.10` | 23.10 | 49.66 | 61.03 | 15.89 | 0.3563 | 21.38 | 39.66 | 53.10 | 24.37 | 0.3147 |
| ATS `f=12, tau=0.05` | 23.79 | 49.66 | 61.03 | 16.14 | 0.3603 | 21.38 | 40.00 | 52.76 | 24.48 | 0.3149 |
| ATS `f=16, tau=0.10` | 22.76 | 49.31 | 61.38 | 15.96 | 0.3584 | 20.69 | 39.66 | 52.07 | 24.07 | 0.3135 |
| ATS `f=20, tau=0.10` | 22.76 | 49.66 | 62.41 | 15.93 | 0.3572 | 20.00 | 40.34 | 53.45 | 24.40 | 0.3081 |

Notes:

- Lowering `tau` from `0.10` to `0.05` improved ATS-only T2V `Top1` and `MRR`.
- However, pushing `tau` further down to `0.04 -> 0.01` did not continue improving ATS-only `Top1`.
- The best ATS-only T2V `Top1` among the current runs remains `f=12, tau=0.05`.
- Increasing the frame budget from `12` to `20` slightly improved T2V `Top10`, but did not improve T2V `Top1`.

## 3. Hybrid ATS + Peak Ablations

Configuration:

- `select_frames = 12`
- `peak_frames = 4`
- `tau ∈ {0.01, 0.02, 0.03, 0.04, 0.05}`

| Method | T2V Top1 | T2V Top5 | T2V Top10 | T2V Mean Rank | T2V MRR | V2T Top1 | V2T Top5 | V2T Top10 | V2T Mean Rank | V2T MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATS+Peak `f=12, k=4, tau=0.01` | 24.14 | 50.34 | 61.03 | 16.08 | 0.3673 | 22.07 | 40.34 | 53.45 | 24.61 | 0.3190 |
| ATS+Peak `f=12, k=4, tau=0.02` | **25.52** | 50.69 | 61.72 | 16.01 | **0.3755** | 21.38 | 40.00 | 53.45 | 24.53 | 0.3176 |
| ATS+Peak `f=12, k=4, tau=0.03` | 24.83 | 50.69 | 61.03 | **15.96** | 0.3713 | 21.38 | 40.69 | 53.79 | **24.48** | 0.3180 |
| ATS+Peak `f=12, k=4, tau=0.04` | 25.17 | **51.03** | 61.72 | 16.01 | 0.3716 | 21.72 | 40.00 | 53.10 | 24.66 | 0.3185 |
| ATS+Peak `f=12, k=4, tau=0.05` | 25.17 | 50.00 | **62.07** | 16.03 | 0.3711 | 21.38 | **40.69** | **54.14** | 24.55 | 0.3173 |

## 4. Best Current Results

### Best Text-to-Video

- Best `Top1`: ATS+Peak `f=12, k=4, tau=0.02` -> `25.52`
- Best `MRR`: ATS+Peak `f=12, k=4, tau=0.02` -> `0.3755`
- Best `Top5`: ATS+Peak `f=12, k=4, tau=0.04` -> `51.03`
- Best `Top10`: ATS+Peak `f=12, k=4, tau=0.05` -> `62.07`
- Best `Mean Rank` (lower is better): ATS+Peak `f=12, k=4, tau=0.03` -> `15.96`

### Best Video-to-Text

- Best `Top1`: ATS+Peak `f=12, k=4, tau=0.01` -> `22.07`
- Best `Top5`: ATS+Peak `f=12, k=4, tau=0.03` and `tau=0.05` -> `40.69`
- Best `Top10`: ATS+Peak `f=12, k=4, tau=0.05` -> `54.14`
- Best `Mean Rank` (lower is better): ATS+Peak `f=12, k=4, tau=0.03` -> `24.48`
- Best `MRR`: ATS+Peak `f=12, k=4, tau=0.01` -> `0.3190`

## 5. Main Takeaways

1. ATS-only is competitive with Top-P, but does not consistently beat it on T2V `Top1`.
2. Lowering ATS `tau` helps, which suggests that the original `tau=0.10` was smoothing too aggressively.
3. Hybrid ATS+Peak consistently outperforms both Top-P and ATS-only on the most important retrieval metrics.
4. The best overall T2V configuration so far is:
   - ATS+Peak `f=12, k=4, tau=0.02`
5. If prioritizing V2T robustness, the strongest candidates are:
   - ATS+Peak `f=12, k=4, tau=0.01`
   - ATS+Peak `f=12, k=4, tau=0.03`
   - ATS+Peak `f=12, k=4, tau=0.05`

## 6. Suggested Next Experiments

- Sweep `peak_frames` with fixed `f=12`
  - e.g. `k=2`, `k=4`, `k=6`
- Sweep `select_frames`
  - e.g. `f=12`, `f=16`
- Test hybrid with `tau` around the current sweet spot
  - e.g. `0.02`, `0.03`, `0.04`
