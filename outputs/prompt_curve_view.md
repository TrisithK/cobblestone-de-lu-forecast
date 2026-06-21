# Prompt-Curve Translation — DE-LU Day-Ahead Power

**Forecast model:** Ridge (selected model, CLAUDE.md §6)  
**OOS window:** 2025-12-08 → 2025-12-31 (24 days, 576 hourly predictions)  
**EEX reference:** Front-month: ICE ENDEX German Power Financial Base Futures (GABF2026), January 2026 delivery, final settlement 2026-01-30 (range 102.45–104.09, mid 103.99 EUR/MWh) — real, publicly sourced (TradingView / ICE ENDEX). Front-week: NOT a real settlement — estimated from early-December EPEX spot context (~110 EUR/MWh); the actual EEX week-future print requires an EEX DataSource subscription and could not be sourced publicly. Treated as illustrative throughout (see EEX_FRONTWEEK_IS_ILLUSTRATIVE).

> **NOTE:** `EEX_FRONTWEEK_BASELOAD_EUR` is an **illustrative placeholder**, not a sourced settlement print — no public EEX Phelix DE Week Future settlement could be found (EEX market-data pages only show a rolling 45-day window; full history needs an EEX Group DataSource subscription). **Conviction is capped at MODERATE and size at HALF** as a result (see `src/config.py` `EEX_FRONTWEEK_IS_ILLUSTRATIVE`). The front-month reference (103.99 EUR/MWh, ICE ENDEX GABF2026) **is** a real, dated, sourced settlement.


## 1. Model Fair-Value Aggregates (Ridge Forecast)

### Full OOS period (Dec 8-31 2025)

| Aggregate | EUR/MWh |
|-----------|---------|
| Baseload average (all hours) | **70.75** |
| Peak average (08-20 weekday) | **83.93** |
| Off-peak average             | **63.30** |

### Front-week window (2025-12-08 → 2025-12-14)

| Aggregate | EUR/MWh |
|-----------|---------|
| Baseload average | **75.46** |
| Peak average     | **91.43** |

## 2. EEX Reference Settlement

| Contract | Settlement Date | EUR/MWh | Status |
|----------|----------------|---------|--------|
| DE Front-Month Baseload (GABF2026, Jan-2026) | 2026-01-30 | 103.99 | Real, sourced print |
| DE Front-Week Baseload (w/c Dec 8)     | 2025-12-05  | 107.50 | **ILLUSTRATIVE** — no public print found |

## 3. Basis & Directional View

| Comparison | Basis (EUR/MWh) | Note |
|------------|----------------|------|
| Front-week: model Dec 8-14 baseload vs EEX front-week | **-32.04** | **Primary** comparison (matched delivery window) — drives the call below |
| Front-month: model full-OOS avg vs EEX front-month | -33.24 | Indicative curve-shape context only — Dec-2025 model avg vs. a Jan-2026 contract, **not a matched delivery window**; not used to size or direct the trade |

### Direction: **SHORT / SELL** — Conviction: **MODERATE** — Size: **HALF SIZE (1/2 normal prompt risk)**

> Conviction/size shown above are **already capped** for the illustrative front-week reference (MODERATE / HALF ceiling). No HIGH-conviction / FULL-SIZE call rests on an unsourced number.


Model fair value for the front-week delivery window (Dec 8-14) is 75.5 EUR/MWh baseload, 32.0 EUR/MWh BELOW the pinned EEX front-week settlement (107.5 EUR/MWh, 2025-12-05). The prompt forward looks rich vs. near-term model fair value. Directional lean: short prompt (sell the front-week or reduce long). Suggested size: HALF SIZE (1/2 normal prompt risk).

## 4. Invalidation Triggers

Position should be re-evaluated if any of the following occur:

- TTF front-month moves >5 % overnight (gas repricing shifts the absolute price level across all hours; the model lags because it uses yesterday's TTF close).
- Wind-power forecast revision >5 GW vs. the D-1 model run (residual load — the primary merit-order driver — would reprice the full day materially).
- Unplanned nuclear / large thermal outage notified on REMIT (supply removal lifts scarcity hours disproportionately; model can't anticipate).
- Demand surprise: cold snap or anomalous holiday-week consumption not captured in the load forecast (especially Christmas week).
- EEX settlement revises intraday — confirm vs. live screen before trading.

## 5. Risk-Premium Caveat

The EEX forward price embeds a risk premium over E[spot]; it is not an unbiased expectation of the realised daily average. This basis view is a directional lean on relative value, not a claim that the forward will converge to the model forecast. Size positions accordingly.

> *Structured output of this module is fed directly into the LLM commentary engine (Step 9) as the grounding fact-object — the LLM originates no numbers.*