# Retrieval Ablation Summary

This file summarizes the current retrieval ablation results collected from:

- `output2/`: Top-P baselines and ATS baseline
- `output3/`: ATS-only variants
- `output4/`: Hybrid ATS+Peak variants

All numbers below are taken directly from the corresponding `*metrics*.json` files.

## 1. Top-P vs ATS Baselines

| Method | T->V R@1↑ | R@5↑ | R@10↑ | MdR↓ | MRR↑ | V->T R@1↑ | R@5↑ | R@10↑ | MdR↓ | MRR↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-P `p=0.7` | 22.41 | 48.28 | 60.00 | 6.0 | 0.3552 | 21.38 | 39.66 | 51.72 | 10.0 | 0.3126 |
| Top-P `p=0.8` | 23.79 | 49.66 | 61.03 | 6.0 | 0.3596 | 20.00 | 39.31 | 52.07 | 10.0 | 0.3062 |
| ATS `f=12, tau=0.10` | 23.10 | 49.66 | 61.03 | 6.0 | 0.3563 | 21.38 | 39.66 | 53.10 | 9.5 | 0.3147 |

## 2. ATS-Only Ablations

| Method | T->V R@1↑ | R@5↑ | R@10↑ | MdR↓ | MRR↑ | V->T R@1↑ | R@5↑ | R@10↑ | MdR↓ | MRR↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATS `f=12, tau=0.01` | 22.76 | 51.03 | 62.76 | 5.0 | 0.3600 | 21.03 | 39.31 | 52.07 | 10.0 | 0.3150 |
| ATS `f=12, tau=0.02` | 22.07 | 50.34 | 61.72 | 5.0 | 0.3549 | 20.34 | 39.31 | 53.10 | 10.0 | 0.3114 |
| ATS `f=12, tau=0.03` | 22.76 | 50.00 | 62.07 | 5.5 | 0.3561 | 21.03 | 39.31 | 52.76 | 10.0 | 0.3141 |
| ATS `f=12, tau=0.04` | 22.76 | 50.34 | 61.38 | 5.0 | 0.3575 | 20.69 | 39.66 | 52.76 | 9.5 | 0.3106 |
| ATS `f=12, tau=0.05` | **23.79** | 49.66 | 61.03 | 6.0 | 0.3603 | 21.38 | 40.00 | 52.76 | 10.0 | 0.3149 |
| ATS `f=12, tau=0.10` | 23.10 | 49.66 | 61.03 | 6.0 | 0.3563 | 21.38 | 39.66 | 53.10 | 9.5 | 0.3147 |
| ATS `f=16, tau=0.10` | 22.76 | 49.31 | 61.38 | 6.0 | 0.3584 | 20.69 | 39.66 | 52.07 | 9.0 | 0.3135 |
| ATS `f=20, tau=0.10` | 22.76 | 49.66 | 62.41 | 6.0 | 0.3572 | 20.00 | 40.34 | 53.45 | 9.5 | 0.3081 |

Notes:

- Lowering `tau` from `0.10` to `0.05` improved ATS-only T->V `R@1` and `MRR`.
- However, pushing `tau` further down to `0.04 -> 0.01` did not continue improving ATS-only T->V `R@1`.
- The best ATS-only T->V `R@1` among the current runs remains `f=12, tau=0.05`.
- Increasing the frame budget from `12` to `20` slightly improved T->V `R@10`, but did not improve T->V `R@1`.

## 3. Hybrid ATS + Peak Ablations

Configuration:

- `select_frames = 12`
- `peak_frames = 4`
- `tau ∈ {0.01, 0.02, 0.03, 0.04, 0.05}`

| Method | T->V R@1↑ | R@5↑ | R@10↑ | MdR↓ | MRR↑ | V->T R@1↑ | R@5↑ | R@10↑ | MdR↓ | MRR↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ATS+Peak `f=12, k=4, tau=0.01` | 24.14 | 50.34 | 61.03 | 5.0 | 0.3673 | 22.07 | 40.34 | 53.45 | 9.0 | 0.3190 |
| ATS+Peak `f=12, k=4, tau=0.02` | **25.52** | 50.69 | 61.72 | 5.0 | **0.3755** | 21.38 | 40.00 | 53.45 | 8.5 | 0.3176 |
| ATS+Peak `f=12, k=4, tau=0.03` | 24.83 | 50.69 | 61.03 | 5.0 | 0.3713 | 21.38 | 40.69 | 53.79 | 8.5 | 0.3180 |
| ATS+Peak `f=12, k=4, tau=0.04` | 25.17 | **51.03** | 61.72 | 5.0 | 0.3716 | 21.72 | 40.00 | 53.10 | 8.5 | 0.3185 |
| ATS+Peak `f=12, k=4, tau=0.05` | 25.17 | 50.00 | **62.07** | 5.5 | 0.3711 | 21.38 | **40.69** | **54.14** | 8.5 | 0.3173 |

## 4. Best Current Results

### Best Text-to-Video

- Best `R@1`: ATS+Peak `f=12, k=4, tau=0.02` -> `25.52`
- Best `MRR`: ATS+Peak `f=12, k=4, tau=0.02` -> `0.3755`
- Best `R@5`: ATS+Peak `f=12, k=4, tau=0.04` -> `51.03`
- Best `R@10`: ATS+Peak `f=12, k=4, tau=0.05` -> `62.07`
- Best `MdR` (lower is better): ATS+Peak `f=12, k=4, tau=0.02/0.03/0.04` -> `5.0`

### Best Video-to-Text

- Best `R@1`: ATS+Peak `f=12, k=4, tau=0.01` -> `22.07`
- Best `R@5`: ATS+Peak `f=12, k=4, tau=0.03` and `tau=0.05` -> `40.69`
- Best `R@10`: ATS+Peak `f=12, k=4, tau=0.05` -> `54.14`
- Best `MdR` (lower is better): ATS+Peak `f=12, k=4, tau=0.02/0.03/0.04/0.05` -> `8.5`
- Best `MRR`: ATS+Peak `f=12, k=4, tau=0.01` -> `0.3190`

## 5. Main Takeaways

1. ATS-only is competitive with Top-P, but does not consistently beat it on T->V `R@1`.
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
