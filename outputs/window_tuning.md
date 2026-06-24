# Window-Type Tuning — Expanding vs. Rolling (REVISION_PLAN.md A2)

Generated: 2026-06-24 19:06 UTC

Validation period: 2024-01-01 → 2024-12-31 (2024, Ridge only, walk-forward)

| Window type | Ridge MAE (EUR/MWh) |
|---|---|
| Expanding (all history) | 14.32 |
| Rolling (728d trailing) | 15.05 |

**Winner: expanding.** Set as `config.WINDOW_TYPE`. Used for the Test-period (2025) backtest below and for predictions.csv.

## Model-Selection Comparison (Validation 2024, winning window type)

Ridge vs. LightGBM MAE on the 2024 Validation backtest, under the winning window type (expanding). **This comparison — not the 2025 Test numbers — is what justifies the Ridge-vs-LightGBM model selection decision** (see full_report.pdf §4/§5). The Test-period comparison is reported separately as out-of-sample confirmation only.

| Model | MAE (EUR/MWh) |
|---|---|
| Ridge | 14.32 |
| LightGBM | 11.22 |