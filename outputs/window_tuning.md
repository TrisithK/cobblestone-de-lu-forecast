# Window-Type Tuning — Expanding vs. Rolling (REVISION_PLAN.md A2)

Generated: 2026-06-21 10:00 UTC

Validation period: 2024-01-01 → 2024-12-31 (2024, Ridge only, walk-forward)

| Window type | Ridge MAE (EUR/MWh) |
|---|---|
| Expanding (all history) | 16.00 |
| Rolling (728d trailing) | 16.43 |

**Winner: expanding.** Set as `config.WINDOW_TYPE`. Used for the Test-period (2025) backtest below and for predictions.csv.
