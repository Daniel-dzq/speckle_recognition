# Fig3_authentication report

## Data sources
- `outputs/final_15fiber_training/summary_15fibers.csv` — per-fiber test accuracy (panel a)
- `outputs/final_15fiber_training/auth_matrix_15x15.csv` — 15×15 authentication matrix (panels b–d)
- `outputs/final_15fiber_training/auth_matrix_report.md` — reference text (values verified against CSV)

## Key values (computed from CSV)
| Metric | Value |
|--------|-------|
| Diagonal mean | 98.02% |
| Off-diagonal mean | 12.69% |
| Random baseline | 12.5% |

## Figure role
Main-text authentication performance figure.

## Outputs
- `figures/paper/Fig3_authentication/Fig3_authentication.png`
- `figures/paper/Fig3_authentication/Fig3_authentication.pdf`
- `figures/paper/Fig3_authentication/Fig3_authentication.svg`
- `figures/paper/Fig3_authentication/Fig3_authentication_data_summary.csv`

## Caption draft
Fifteen fiber-specific models trained on eight challenge classes achieve high per-fiber test accuracy (a) and strong diagonal entries in the cross-fiber authentication matrix (b). Same-fiber accuracies cluster near 98% while cross-fiber scores match the 12.5% random baseline (c, d), demonstrating fiber-specific acceptance and cross-fiber rejection.
