# Walk-Forward Validation Metrics — DE-LU Day-Ahead Price

Generated: 2026-06-24 01:17 UTC

Backtest period: 2025-01-01 → 2025-12-31  (full calendar year — spans all four seasons)  (8,759 hourly predictions)

Training window: **expanding** (all history before the prediction day) — chosen on the 2024 validation period, see outputs/window_tuning.md.

> **No MAPE reported.** DE-LU day-ahead prices are frequently zero or negative (high-renewables hours). MAPE is undefined when y_true = 0 and explosive / uninterpretable near zero. MAE and RMSE are used throughout.

## 1. Overall Metrics

| model     |   MAE |   RMSE | Skill_vs_D7   |
|:----------|------:|-------:|:--------------|
| Naïve D-1 | 25.97 |  40.45 | +20.9%        |
| Naïve D-7 | 32.83 |  49.33 | +0.0%         |
| Ridge     | 14.5  |  23.58 | +55.8%        |
| LightGBM  | 11.23 |  18.93 | +65.8%        |


> Skill score = 1 − MAE_model / MAE(Naïve D-7). Positive = model beats the D-7 naïve benchmark.

## 2. MAE by Hour of Day

|   hour |   Naïve D-1 |   Naïve D-7 |   Ridge |   LightGBM |
|-------:|------------:|------------:|--------:|-----------:|
|      0 |       18    |       25.29 |   11.14 |       7.8  |
|      1 |       17.73 |       25.36 |   10.81 |       7.62 |
|      2 |       17.8  |       25.47 |   10.88 |       7.77 |
|      3 |       18.22 |       25.28 |   10.98 |       7.9  |
|      4 |       17.77 |       24.8  |   10.71 |       7.75 |
|      5 |       17.49 |       24.3  |   10.09 |       7.39 |
|      6 |       24.19 |       26.74 |   11.2  |       9.03 |
|      7 |       34.12 |       34.11 |   15.71 |      11.58 |
|      8 |       37.4  |       36.35 |   17.13 |      12.49 |
|      9 |       33.3  |       34.22 |   14.7  |      11.16 |
|     10 |       29.9  |       37.02 |   15.1  |      11.91 |
|     11 |       28.26 |       38.14 |   15.14 |      12.51 |
|     12 |       29.08 |       39.14 |   15.5  |      13.48 |
|     13 |       30.06 |       39.46 |   16.48 |      14.25 |
|     14 |       29.67 |       38.37 |   16.45 |      13.46 |
|     15 |       28.05 |       36.47 |   16.21 |      12.63 |
|     16 |       29.09 |       38.84 |   16.68 |      13.91 |
|     17 |       32.82 |       40.15 |   18.43 |      14.77 |
|     18 |       31.17 |       39.49 |   19.66 |      16.26 |
|     19 |       34.03 |       44.38 |   21.99 |      16.25 |
|     20 |       31.23 |       38.75 |   19.08 |      14.39 |
|     21 |       23.02 |       30.11 |   14.1  |      10.45 |
|     22 |       15.84 |       23.23 |   10.46 |       7.67 |
|     23 |       14.97 |       22.34 |    9.36 |       7.17 |


## 3. MAE by Market Regime

| regime               |   n_hours |   Naïve D-1 |   Naïve D-7 |   Ridge |   LightGBM |
|:---------------------|----------:|------------:|------------:|--------:|-----------:|
| Peak (08-20 weekday) |      3393 |       32.85 |       41.79 |   18.18 |      14.82 |
| Off-peak             |      5366 |       21.62 |       27.16 |   12.17 |       8.97 |
| High residual load   |      4380 |       24.63 |       31.06 |   14.77 |      11.67 |
| Low residual load    |      4379 |       27.31 |       34.6  |   14.23 |      10.79 |


## 4. Model Selection Decision

**Decided on the 2024 Validation period — the 2025 Test set is never consulted in this choice.**

Ridge MAE: **14.32 EUR/MWh** | LightGBM MAE: **11.22 EUR/MWh** | Absolute gap: **3.10 EUR/MWh (21.7% of Ridge MAE)** (Validation 2024, expanding window — see outputs/window_tuning.md)


**Selected model: Ridge.**

LightGBM posts a lower MAE by 3.10 EUR/MWh (21.7%) on Validation, which looks large in isolation. The selection still goes to Ridge for three reasons:

1. **A fair-value signal must be interrogable.** A trader needs to know *why* the model says 95 EUR/MWh, not just that it does. Ridge coefficients are inspectable; a 500-tree ensemble is not. Trust, not raw accuracy, is the production constraint.

2. **Power markets break regime; flexible models break with them.** The Validation period alone (2024) already includes a meaningful regime mix, and a regularised linear form still degrades more gracefully than a 500-tree ensemble when the next regime shift (a gas shock, a step-change in renewables build-out) looks nothing like 2019-2024.

3. **LightGBM confirms Ridge's design, not that Ridge is mis-specified.** Feature importances (residual load and price lags dominate) are exactly the drivers Ridge is built around. The extra accuracy comes from nonlinear interactions Ridge cannot represent — real but not the dominant source of signal. Including the depth-3 tree figure captures that story without committing to the full black box.

**Out-of-sample confirmation, not part of the decision:** the same gap shows up on the 2025 Test backtest below (LightGBM ahead by 3.27 EUR/MWh), spanning summer solar saturation, spring negative-price spells, and winter scarcity hours. That the pattern holds out-of-sample is reassuring — it means this isn't a single-season artefact of the Validation year — but it confirms a decision already made on Validation, rather than informing it.

LightGBM is retained as a **parallel challenger signal**: run alongside Ridge each day; divergence flags that a nonlinear regime shift may be in play.
