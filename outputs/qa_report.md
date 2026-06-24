# QA Report — DE-LU Day-Ahead Price Forecasting Data

Generated: 2026-06-24 01:06 UTC

---


## 1. Data Overview

| Series | Rows | Start | End | NaN (raw) |
|--------|-----:|-------|-----|----------:|
| prices | 61,368 | 2019-01-01 01:00:00+01:00 | 2026-01-01 00:00:00+01:00 | 0 |
| exaa_price | 61,368 | 2019-01-01 01:00:00+01:00 | 2026-01-01 00:00:00+01:00 | 0 |
| load | 61,368 | 2019-01-01 01:00:00+01:00 | 2026-01-01 00:00:00+01:00 | 50 |
| wind | 61,368 | 2019-01-01 01:00:00+01:00 | 2026-01-01 00:00:00+01:00 | 0 |
| solar | 61,368 | 2019-01-01 01:00:00+01:00 | 2026-01-01 00:00:00+01:00 | 3 |
| load_fr | 61,368 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 0 |
| wind_fr | 61,368 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 219 |
| ttf (daily) | 1,879 | 2019-01-02 | 2026-06-22 | 0 |
| co2_proxy (daily) | 1,767 | 2019-01-02 | 2025-12-31 | 0 |

> `load_fr` / `wind_fr` are ENTSO-E day-ahead forecast documents for BZN|FR (`query_load_forecast`, `query_wind_and_solar_forecast` via entsoe-py), fetched live via the ENTSO-E API rather than the manually-exported GUI CSVs used for DE-LU. `co2_proxy` is CARB.L (WisdomTree Carbon ETC, LSE) via yfinance — a tradable proxy for the EU ETS EUA price, not an official settlement print (see `data/fetch_fr_co2.py` docstring for why this proxy was chosen). **[v2 round 4]** `exaa_price` is BZN|DE-LU's **Sequence 2** column in the same raw CSVs as `prices` (Sequence 1) — EXAA's own day-ahead auction for the same zone, a real settlement, not a forecast or proxy. It is point-in-time safe by its own earlier gate closure (~10:15 CET D-1, vs. EPEX Sequence 1's ~12:00 CET D-1), not by any imputation here. Full-history correlation with the target series: 0.986.


## 2. Duplicate Timestamps

- **prices**: 0 duplicates → **PASS**
- **exaa_price**: 0 duplicates → **PASS**
- **load**: 0 duplicates → **PASS**
- **wind**: 0 duplicates → **PASS**
- **solar**: 0 duplicates → **PASS**
- **load_fr**: 0 duplicates → **PASS**
- **wind_fr**: 0 duplicates → **PASS**

## 3. Gap Detection & Imputation

Short-gap threshold: ≤ 3 hours → linear interpolation.  Long gaps → flagged, left as NaN.

### prices
No gaps detected. **PASS**

### exaa_price
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

### load_fr
No gaps detected. **PASS**

### wind_fr
| Start | End | Length (h) | Kind |
|-------|-----|:----------:|------|
| 2019-10-27 02:00:00+01:00 | 2019-10-27 23:00:00+01:00 | 22 | long |
| 2020-07-12 00:00:00+02:00 | 2020-07-13 23:00:00+02:00 | 48 | long |
| 2020-10-25 02:00:00+01:00 | 2020-10-25 23:00:00+01:00 | 22 | long |
| 2020-11-11 00:00:00+01:00 | 2020-11-12 23:00:00+01:00 | 48 | long |
| 2021-03-28 23:00:00+02:00 | 2021-03-28 23:00:00+02:00 | 1 | short |
| 2021-06-23 00:00:00+02:00 | 2021-06-23 23:00:00+02:00 | 24 | long |
| 2021-10-31 03:00:00+01:00 | 2021-10-31 23:00:00+01:00 | 21 | long |
| 2022-03-27 23:00:00+02:00 | 2022-03-27 23:00:00+02:00 | 1 | short |
| 2022-10-30 23:00:00+01:00 | 2022-10-30 23:00:00+01:00 | 1 | short |
| 2022-11-12 17:00:00+01:00 | 2022-11-12 23:00:00+01:00 | 7 | long |
| 2023-04-18 00:00:00+02:00 | 2023-04-18 23:00:00+02:00 | 24 | long |

Short gaps imputed: 3  |  Long gaps flagged: 8

> **WARNING**: 8 long gap(s) in `wind_fr` left as NaN.  Downstream features must handle these rows.


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
| exaa_price_eur_mwh | -167.4 | 1029.3 | -600 | 5000 | 0 | **PASS** |
| load_forecast_mw | 30893.1 | 78154.4 | 10000 | 120000 | 0 | **PASS** |
| wind_forecast_mw | 244.6 | 50447.0 | 0 | 130000 | 0 | **PASS** |
| solar_forecast_mw | 0.0 | 50917.7 | 0 | 80000 | 0 | **PASS** |
| load_forecast_fr_mw | 27650.0 | 87850.0 | 15000 | 105000 | 0 | **PASS** |
| wind_forecast_fr_mw | 0.0 | 22902.8 | 0 | 35000 | 0 | **PASS** |
| price_fr_eur_mwh | -134.9 | 2987.8 | -600 | 3500 | 0 | **PASS** |
| price_nl_eur_mwh | -500.0 | 873.0 | -600 | 1000 | 0 | **PASS** |
| price_be_eur_mwh | -500.0 | 871.0 | -600 | 1000 | 0 | **PASS** |
| price_pl_eur_mwh | -132.9 | 771.0 | -600 | 1000 | 0 | **PASS** |
| price_cz_eur_mwh | -224.5 | 871.0 | -600 | 1000 | 0 | **PASS** |
| load_forecast_nl_mw | 478.8 | 21256.9 | 3000 | 25000 | 67 | **FAIL** |
| load_forecast_be_mw | 6091.4 | 13244.9 | 5000 | 16000 | 0 | **PASS** |
| load_forecast_pl_mw | 10400.0 | 27502.9 | 8000 | 30000 | 0 | **PASS** |
| load_forecast_cz_mw | 3818.0 | 11009.0 | 3000 | 13000 | 0 | **PASS** |
| wind_forecast_nl_mw | 11.8 | 8245.2 | 0 | 10000 | 0 | **PASS** |
| solar_forecast_nl_mw | 0.0 | 7821.5 | 0 | 10000 | 0 | **PASS** |
| wind_forecast_be_mw | 1.2 | 5094.5 | 0 | 7000 | 0 | **PASS** |
| solar_forecast_be_mw | 0.0 | 7874.7 | 0 | 10000 | 0 | **PASS** |
| wind_forecast_pl_mw | 19.5 | 9526.3 | 0 | 11000 | 0 | **PASS** |
| solar_forecast_pl_mw | 0.0 | 13900.1 | 0 | 16000 | 0 | **PASS** |
| solar_forecast_cz_mw | 0.0 | 3145.5 | 0 | 4000 | 0 | **PASS** |
| solar_forecast_fr_mw | 0.0 | 19745.7 | 0 | 22000 | 0 | **PASS** |

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
| exaa_price | 0 | 0 | 0 |
| load | 50 | 48 | 2 |
| wind | 0 | 0 | 0 |
| solar | 3 | 0 | 3 |
| load_fr | 0 | 0 | 0 |
| wind_fr | 219 | 216 | 3 |

## 8. Forecast Transfer Capacity (NTC) Coverage

Day-Ahead NTC (`query_net_transfer_capacity_dayahead`, ENTSO-E document A61) is published for only some of DE-LU's physical borders, and the set has shrunk over 2019-2025 as parts of Europe moved to flow-based capacity calculation (no single bilateral NTC number once a border joins a flow-based region). The columns below are exactly what's available — **not** padded or estimated for the missing borders. `ntc_import_capacity_mw` / `ntc_export_capacity_mw` (built in `features.py`) sum whichever of these are live at each hour, so their composition changes over time; see `data/fetch_ntc.py` docstring for the full per-border timeline.

| Column | Rows | Start | End | NaN |
|--------|-----:|-------|-----|----:|
| ntc_DE_LU_to_NL_mw | 43,824 | 2021-01-01 | 2025-12-31 | 17544 |
| ntc_NL_to_DE_LU_mw | 43,824 | 2021-01-01 | 2025-12-31 | 17544 |
| ntc_DE_LU_to_CH_mw | 61,368 | 2019-01-01 | 2025-12-31 | 0 |
| ntc_CH_to_DE_LU_mw | 61,368 | 2019-01-01 | 2025-12-31 | 0 |
| ntc_DE_LU_to_AT_mw | 8,760 | 2019-01-01 | 2019-12-31 | 52608 |
| ntc_AT_to_DE_LU_mw | 8,760 | 2019-01-01 | 2019-12-31 | 52608 |
| ntc_DE_LU_to_CZ_mw | 30,119 | 2019-01-01 | 2022-06-08 | 31249 |
| ntc_CZ_to_DE_LU_mw | 30,119 | 2019-01-01 | 2022-06-08 | 31249 |
| ntc_DE_LU_to_DK_1_mw | 61,368 | 2019-01-01 | 2025-12-31 | 0 |
| ntc_DK_1_to_DE_LU_mw | 61,368 | 2019-01-01 | 2025-12-31 | 0 |
| ntc_DE_LU_to_DK_2_mw | 43,824 | 2019-01-01 | 2023-12-31 | 17544 |
| ntc_DK_2_to_DE_LU_mw | 43,824 | 2019-01-01 | 2023-12-31 | 17544 |

Borders with **no** published Day-Ahead NTC anywhere in 2019-2025 (checked all seven years): FR, BE, PL, NO_2, SE_4 — see `data/fetch_ntc_metadata.json` for the per-border rationale. These do not appear as columns at all.

## 9. Neighbor Bidding Zone (FR/NL/BE/PL/CZ) Coverage

Used to build cross-border price-lag and residual-load features for DE-LU's five largest neighbors. Three findings, all flagged rather than silently patched:

**a) Neighbor day-ahead prices truncate in late September 2025 in this build environment.** All five zones' live `query_day_ahead_prices` calls return data through 2025-09-29/30 and stop — including a same-day check against `DE_LU` itself via the live API, so this is a live-data-window boundary, not a per-zone fault. DE-LU's own price series is unaffected here because it's read from the committed GUI-export snapshot (`data/fetch_data.py`), not this live call. `features.py` forward-fills the last known neighbor price across the missing ~3 months (Oct-Dec 2025) so price-lag features stay defined without truncating the full-year Test backtest — flagged here and in the report, not hidden.

**b) PL has no published day-ahead solar forecast before ~2020.** Treated as 0 in `features.py` (the same precedent as FR's pre-2022 offshore wind in `fetch_fr_co2.py`) — Poland's utility-scale solar buildout was genuinely minimal before then, not a publication gap.

**c) CZ has no published day-ahead wind forecast at all**, in any year (consistent with CZ's near-zero installed wind capacity — same finding as the NTC scan for CZ's border data). `residual_load_cz_mw` is therefore `load - solar` only, never `load - wind - solar`.

| Column | Rows | Start | End | NaN |
|--------|-----:|-------|-----|----:|
| price_fr_eur_mwh | 59,135 | 2019-01-01 00:00:00+01:00 | 2025-09-29 23:00:00+02:00 | 24 |
| price_nl_eur_mwh | 59,159 | 2019-01-01 00:00:00+01:00 | 2025-09-30 23:00:00+02:00 | 0 |
| price_be_eur_mwh | 59,135 | 2019-01-01 00:00:00+01:00 | 2025-09-29 23:00:00+02:00 | 24 |
| price_pl_eur_mwh | 59,159 | 2019-01-01 00:00:00+01:00 | 2025-09-30 23:00:00+02:00 | 0 |
| price_cz_eur_mwh | 59,159 | 2019-01-01 00:00:00+01:00 | 2025-09-30 23:00:00+02:00 | 0 |
| load_forecast_nl_mw | 61,368 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 0 |
| load_forecast_be_mw | 61,368 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 0 |
| load_forecast_pl_mw | 61,368 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 0 |
| load_forecast_cz_mw | 61,368 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 0 |
| wind_forecast_nl_mw | 61,344 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 24 |
| solar_forecast_nl_mw | 61,344 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 24 |
| wind_forecast_be_mw | 61,368 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 0 |
| solar_forecast_be_mw | 61,368 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 0 |
| wind_forecast_pl_mw | 61,344 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 24 |
| solar_forecast_pl_mw | 50,209 | 2020-04-10 00:00:00+02:00 | 2025-12-31 23:00:00+01:00 | 11159 |
| solar_forecast_cz_mw | 61,368 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 0 |
| solar_forecast_fr_mw | 61,016 | 2019-01-01 00:00:00+01:00 | 2025-12-31 23:00:00+01:00 | 352 |

## 10. Overall Verdict

**ONE OR MORE WARNINGS — see sections above**

Long gaps in `load` (2022-02-22, 2022-03-24 — full days) are a known ENTSO-E data absence. These 48 rows remain NaN and will be excluded during feature engineering via a validity mask.
