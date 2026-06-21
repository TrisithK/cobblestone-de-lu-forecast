# QA Report — DE-LU Day-Ahead Price Forecasting Data

Generated: 2026-06-21 10:00 UTC

---


## 1. Data Overview

| Series | Rows | Start | End | NaN (raw) |
|--------|-----:|-------|-----|----------:|
| prices | 61,368 | 2019-01-01 01:00:00+01:00 | 2026-01-01 00:00:00+01:00 | 0 |
| load | 61,368 | 2019-01-01 01:00:00+01:00 | 2026-01-01 00:00:00+01:00 | 50 |
| wind | 61,368 | 2019-01-01 01:00:00+01:00 | 2026-01-01 00:00:00+01:00 | 0 |
| solar | 61,368 | 2019-01-01 01:00:00+01:00 | 2026-01-01 00:00:00+01:00 | 3 |
| ttf (daily) | 1,762 | 2019-01-02 | 2025-12-31 | 0 |

## 2. Duplicate Timestamps

- **prices**: 0 duplicates → **PASS**
- **load**: 0 duplicates → **PASS**
- **wind**: 0 duplicates → **PASS**
- **solar**: 0 duplicates → **PASS**

## 3. Gap Detection & Imputation

Short-gap threshold: ≤ 3 hours → linear interpolation.  Long gaps → flagged, left as NaN.

### prices
No gaps detected. **PASS**

### load
| Start | End | Length (h) | Kind |
|-------|-----|:----------:|------|
| 2022-02-22 00:00:00+01:00 | 2022-02-22 23:00:00+01:00 | 24 | long |
| 2022-03-24 00:00:00+01:00 | 2022-03-24 23:00:00+01:00 | 24 | long |
| 2023-10-29 00:00:00+02:00 | 2023-10-29 00:00:00+02:00 | 1 | short |
| 2024-10-27 00:00:00+02:00 | 2024-10-27 00:00:00+02:00 | 1 | short |

Short gaps imputed: 2  |  Long gaps flagged: 2

> **WARNING**: 2 long gap(s) in `load` left as NaN.  Downstream features must handle these rows.

### wind
No gaps detected. **PASS**

### solar
| Start | End | Length (h) | Kind |
|-------|-----|:----------:|------|
| 2023-10-29 00:00:00+02:00 | 2023-10-29 00:00:00+02:00 | 1 | short |
| 2024-10-27 00:00:00+02:00 | 2024-10-27 00:00:00+02:00 | 1 | short |
| 2025-10-26 00:00:00+02:00 | 2025-10-26 00:00:00+02:00 | 1 | short |

Short gaps imputed: 3  |  Long gaps flagged: 0


## 4. DST Validation (23h spring / 25h fall)

| Date | Transition | Expected h | Actual h | Status |
|------|-----------|:----------:|:--------:|--------|
| 2019-03-31 | spring (→23h) | 23 | 23 | **PASS** |
| 2020-03-29 | spring (→23h) | 23 | 23 | **PASS** |
| 2021-03-28 | spring (→23h) | 23 | 23 | **PASS** |
| 2022-03-27 | spring (→23h) | 23 | 23 | **PASS** |
| 2023-03-26 | spring (→23h) | 23 | 23 | **PASS** |
| 2024-03-31 | spring (→23h) | 23 | 23 | **PASS** |
| 2025-03-30 | spring (→23h) | 23 | 23 | **PASS** |
| 2019-10-27 | fall (→25h) | 25 | 25 | **PASS** |
| 2020-10-25 | fall (→25h) | 25 | 25 | **PASS** |
| 2021-10-31 | fall (→25h) | 25 | 25 | **PASS** |
| 2022-10-30 | fall (→25h) | 25 | 25 | **PASS** |
| 2023-10-29 | fall (→25h) | 25 | 25 | **PASS** |
| 2024-10-27 | fall (→25h) | 25 | 25 | **PASS** |
| 2025-10-26 | fall (→25h) | 25 | 25 | **PASS** |

## 5. Range & Sanity Checks

> Negative prices are **valid** in DE-LU (high-renewables hours) and are preserved.  The price lower bound is the ENTSO-E floor (−500 EUR/MWh) with margin.

| Series | Min | Max | Bound lo | Bound hi | Out-of-bound | Status |
|--------|----:|----:|---------:|---------:|:------------:|--------|
| price_eur_mwh | -500.0 | 936.3 | -600 | 5000 | 0 | **PASS** |
| load_forecast_mw | 30893.1 | 78154.4 | 10000 | 120000 | 0 | **PASS** |
| wind_forecast_mw | 244.6 | 50447.0 | 0 | 130000 | 0 | **PASS** |
| solar_forecast_mw | 0.0 | 50917.7 | 0 | 80000 | 0 | **PASS** |

Negative price hours in full history: **2,051** (3.3% of non-NaN rows) — preserved ✓

## 6. Point-in-Time Firewall Verification

Confirms that loaded `load_forecast_mw` matches the  `Day-ahead Total Load Forecast (MW)` column in the raw CSV,  **not** the `Actual Total Load (MW)` column.

- Max |loaded − forecast| across first 100h: **0.0000 MW**
- Max |loaded − actual|  across first 100h: **9033.8 MW**
- Loaded values match day-ahead forecast column (not actuals). PIT firewall OK. ✓

**PASS**

## 7. Post-Imputation Missingness Summary

| Series | NaN before | NaN after | Δ imputed |
|--------|:----------:|:---------:|:---------:|
| prices | 0 | 0 | 0 |
| load | 50 | 48 | 2 |
| wind | 0 | 0 | 0 |
| solar | 3 | 0 | 3 |

## 8. Overall Verdict

**ONE OR MORE WARNINGS — see sections above**

Long gaps in `load` (2022-02-22, 2022-03-24 — full days) are a known ENTSO-E data absence. These 48 rows remain NaN and will be excluded during feature engineering via a validity mask.
