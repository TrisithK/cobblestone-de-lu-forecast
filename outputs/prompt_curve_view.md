# Model Fair-Value View — DE-LU Day-Ahead Power

**Forecast model:** Ridge (selected model, CLAUDE.md §6)  
**OOS window:** 2025-01-01 → 2025-12-31 (8759 hourly predictions)  

No EEX forward print is used in this view (none could be sourced for the OOS delivery dates). The directional trading call (direction / conviction / size) shown in the morning note is computed in `llm_commentary.py` primarily against the **EXAA (Sequence 2) day-ahead auction price** for the same delivery day — a real, observable pre-auction print that settles earlier the same day (~10:15 CET D-1) than the EPEX auction this model forecasts (~12:00 CET D-1). The earlier self-referential basis (forecast vs. trailing D-1/D-7 realised baseload) is kept as secondary context. See full_report.pdf §7 / CLAUDE.md §7 for the rationale.

## 1. Model Fair-Value Aggregates (Ridge Forecast, full OOS year)

| Aggregate | EUR/MWh |
|-----------|---------|
| Baseload average (all hours) | **83.16** |
| Peak average (08-20 weekday) | **88.05** |
| Off-peak average             | **80.24** |

> *`figures/prompt_curve.png` plots a Q4 2025 (Oct–Dec) slice of the hourly forecast for readability — the table above reflects the full OOS year.*

## 2. Invalidation Triggers

The fair-value level above should be re-evaluated if any of the following occur:

- TTF front-month moves >5 % overnight (gas repricing shifts the absolute price level across all hours; the model lags because it uses yesterday's TTF close).
- Wind-power forecast revision >5 GW vs. the D-1 model run (residual load — the primary merit-order driver — would reprice the full day materially).
- Unplanned nuclear / large thermal outage notified on REMIT (supply removal lifts scarcity hours disproportionately; model can't anticipate).
- Demand surprise: cold snap or anomalous holiday-week consumption not captured in the load forecast (especially Christmas week).
- [v2 round 4] EXAA print revises materially between its own auction (~10:15 CET D-1) and EPEX gate closure (~12:00 CET D-1) — the basis driving the trading call was set against the earlier EXAA read, so a fresh fundamentals move in that window would stale it.

> *Aggregates here feed the morning note's Fair-Value Numbers table (`src/morning_note.py`). The trading view itself — direction, conviction, size — is computed directly from the model's own backtest history in `src/llm_commentary.py` and fed into the LLM commentary engine (Step 9) as part of the grounding fact-object; the LLM originates no numbers.*