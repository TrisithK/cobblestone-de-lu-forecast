# Hourly/Block Tradable DA View — DE-LU (REVISION_PLAN.md C1)

**OOS window:** 2025-12-08 → 2025-12-31 (576 hourly predictions)  

This view reads the model's own OOS hourly forecast as the fair-value curve — it does **not** depend on the EEX forward reference (see `outputs/prompt_curve_view.md` for that, separate, comparison). Each hour is flagged against **that day's own baseload average**, so the signal is purely about shape: which individual hours or blocks screen rich/cheap *within* the day, tradable via hourly DA bids, peak/off-peak block products, or intraday.

## 1. Shape Summary

| Metric | Value |
|---|---:|
| Peak average | 83.93 EUR/MWh |
| Off-peak average | 63.30 EUR/MWh |
| Peak − off-peak spread | **+20.63 EUR/MWh** |
| Rich hours (> +15 EUR/MWh vs. day avg) | 102 |
| Cheap hours (< −15 EUR/MWh vs. day avg) | 114 |
| ...of which negative-price | 2 |
| Scarcity hours (top decile residual load) | 58 |

## 2. Top Rich Hours (sell candidates)

| Datetime | Price (EUR/MWh) | Deviation vs. day avg |
|---|---:|---:|
| 2025-12-10T16:00:00+01:00 | 112.68 | +38.33 |
| 2025-12-17T17:00:00+01:00 | 126.26 | +37.38 |
| 2025-12-19T16:00:00+01:00 | 86.93 | +36.70 |
| 2025-12-19T17:00:00+01:00 | 86.02 | +35.79 |
| 2025-12-17T16:00:00+01:00 | 123.62 | +34.75 |
| 2025-12-10T08:00:00+01:00 | 109.05 | +34.69 |
| 2025-12-19T15:00:00+01:00 | 84.38 | +34.14 |
| 2025-12-16T17:00:00+01:00 | 123.81 | +33.11 |
| 2025-12-16T16:00:00+01:00 | 123.51 | +32.81 |
| 2025-12-12T17:00:00+01:00 | 140.06 | +30.47 |

## 3. Top Cheap Hours (buy / load-shift candidates)

| Datetime | Price (EUR/MWh) | Deviation vs. day avg |
|---|---:|---:|
| 2025-12-19T03:00:00+01:00 | -4.30 | -54.54 |
| 2025-12-19T02:00:00+01:00 | -2.89 | -53.13 |
| 2025-12-19T04:00:00+01:00 | 0.25 | -49.98 |
| 2025-12-19T01:00:00+01:00 | 0.74 | -49.50 |
| 2025-12-08T03:00:00+01:00 | 0.70 | -48.42 |
| 2025-12-08T02:00:00+01:00 | 1.42 | -47.70 |
| 2025-12-11T02:00:00+01:00 | 25.80 | -44.95 |
| 2025-12-08T04:00:00+01:00 | 4.88 | -44.24 |
| 2025-12-11T01:00:00+01:00 | 27.21 | -43.54 |
| 2025-12-09T01:00:00+01:00 | 20.60 | -43.49 |

## 4. Negative-Price Hours

| Datetime | Price (EUR/MWh) |
|---|---:|
| 2025-12-19T03:00:00+01:00 | -4.30 |
| 2025-12-19T02:00:00+01:00 | -2.89 |

## 5. Trading Guidance

Peak screens +20.6 EUR/MWh above off-peak over the OOS window — that spread is itself a tradable block product (peak/off-peak swap or DA block bid). 102 individual hours screen >15 EUR/MWh rich vs. their own day's baseload (mostly evening peak / scarcity hours) — candidates for selling that specific hour or block in the DA auction / intraday rather than the whole day. 114 hours (including 2 negative-price hours) screen deeply cheap — candidates for buying / shifting flexible load (storage charging, demand response) into those hours rather than paying the day's average. 58 hours sit in the top decile of residual load (scarcity pricing) — these are where the model's forecast error is largest (see by-regime MAE) and where a wind-forecast miss or outage would do the most damage to the call.

## 6. Invalidation Triggers

Same triggers as the forward-basis view: a wind-forecast revision >5 GW, a TTF/gas gap >5% overnight, an unplanned outage on REMIT, or a demand surprise all reprice the hourly shape, not just the daily level — re-check before acting on any single-hour call above.

> *Figure: `figures/hourly_block_view.png`.*