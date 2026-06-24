# Hourly/Block Tradable DA View — DE-LU (REVISION_PLAN.md C1)

**OOS window:** 2025-01-01 → 2025-12-31 (8759 hourly predictions)  

This view reads the model's own OOS hourly forecast as the fair-value curve — it does not depend on any external forward reference. Each hour is flagged against **that day's own baseload average**, so the signal is purely about shape: which individual hours or blocks screen rich/cheap *within* the day, tradable via hourly DA bids, peak/off-peak block products, or intraday. (`figures/hourly_block_view.png` plots a Q4 2025 slice for readability — the stats below are full OOS year.)

## 1. Shape Summary

| Metric | Value |
|---|---:|
| Peak average | 88.05 EUR/MWh |
| Off-peak average | 80.24 EUR/MWh |
| Peak − off-peak spread | **+7.80 EUR/MWh** |
| Rich hours (> +15 EUR/MWh vs. day avg) | 2902 |
| Cheap hours (< −15 EUR/MWh vs. day avg) | 2472 |
| ...of which negative-price | 442 |
| Scarcity hours (top decile residual load) | 876 |

## 2. Top Rich Hours (sell candidates)

| Datetime | Price (EUR/MWh) | Deviation vs. day avg |
|---|---:|---:|
| 2025-09-20T19:00:00+02:00 | 180.29 | +116.33 |
| 2025-01-21T17:00:00+01:00 | 286.14 | +111.09 |
| 2025-06-09T20:00:00+02:00 | 156.13 | +106.58 |
| 2025-07-02T20:00:00+02:00 | 210.52 | +105.44 |
| 2025-09-09T19:00:00+02:00 | 221.90 | +103.02 |
| 2025-09-30T19:00:00+02:00 | 224.83 | +101.47 |
| 2025-09-16T19:00:00+02:00 | 96.93 | +95.59 |
| 2025-09-15T19:00:00+02:00 | 109.07 | +94.18 |
| 2025-10-01T19:00:00+02:00 | 202.21 | +93.62 |
| 2025-06-09T21:00:00+02:00 | 141.03 | +91.49 |

## 3. Top Cheap Hours (buy / load-shift candidates)

| Datetime | Price (EUR/MWh) | Deviation vs. day avg |
|---|---:|---:|
| 2025-06-21T13:00:00+02:00 | -42.25 | -111.52 |
| 2025-06-19T14:00:00+02:00 | -52.86 | -111.30 |
| 2025-06-19T13:00:00+02:00 | -51.89 | -110.33 |
| 2025-06-21T14:00:00+02:00 | -40.49 | -109.76 |
| 2025-06-23T14:00:00+02:00 | -74.93 | -108.69 |
| 2025-06-23T15:00:00+02:00 | -70.47 | -104.22 |
| 2025-05-11T13:00:00+02:00 | -65.95 | -103.11 |
| 2025-06-21T12:00:00+02:00 | -33.75 | -103.02 |
| 2025-01-02T00:00:00+01:00 | -31.44 | -102.91 |
| 2025-06-19T12:00:00+02:00 | -43.32 | -101.77 |

## 4. Negative-Price Hours

| Datetime | Price (EUR/MWh) |
|---|---:|
| 2025-06-23T14:00:00+02:00 | -74.93 |
| 2025-06-23T15:00:00+02:00 | -70.47 |
| 2025-06-08T14:00:00+02:00 | -69.09 |
| 2025-09-15T14:00:00+02:00 | -68.73 |
| 2025-09-15T13:00:00+02:00 | -68.65 |
| 2025-09-16T13:00:00+02:00 | -68.38 |
| 2025-06-08T15:00:00+02:00 | -67.40 |
| 2025-09-16T12:00:00+02:00 | -66.94 |
| 2025-05-11T13:00:00+02:00 | -65.95 |
| 2025-09-16T14:00:00+02:00 | -65.37 |

## 5. Trading Guidance

Peak screens +7.8 EUR/MWh above off-peak over the OOS window — that spread is itself a tradable block product (peak/off-peak swap or DA block bid). 2902 individual hours screen >15 EUR/MWh rich vs. their own day's baseload (mostly evening peak / scarcity hours) — candidates for selling that specific hour or block in the DA auction / intraday rather than the whole day. 2472 hours (including 442 negative-price hours) screen deeply cheap — candidates for buying / shifting flexible load (storage charging, demand response) into those hours rather than paying the day's average. 876 hours sit in the top decile of residual load (scarcity pricing) — these are where the model's forecast error is largest (see by-regime MAE) and where a wind-forecast miss or outage would do the most damage to the call.

## 6. Invalidation Triggers

These hourly/block calls are invalidated by the same fundamental shocks as the model's daily fair-value view (see `outputs/prompt_curve_view.md`): a wind-forecast revision >5 GW, a TTF/gas gap >5% overnight, an unplanned outage on REMIT, or a demand surprise all reprice the hourly shape, not just the daily level — re-check before acting on any single-hour call above.

> *Figure: `figures/hourly_block_view.png`.*