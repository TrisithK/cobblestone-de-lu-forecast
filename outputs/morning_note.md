# DE-LU Morning Desk Note — 2025-12-31

_Generated 2026-06-24 01:17 UTC from cached model + LLM outputs — static, no live data._

---

## Action

**SHORT / SELL** — Conviction: **MODERATE** — Size: **QUARTER SIZE (1/4 normal prompt risk)**

> EXAA-referenced call [v2 round 4]: tomorrow's forecast vs. the EXAA (Sequence 2) day-ahead auction price for the same delivery day — a real pre-auction print that settles earlier the same day (~10:15 CET D-1) than the EPEX auction this model forecasts (~12:00 CET D-1), sized as a multiple of the model's own backtest MAE (see Basis section below).

## Fair-Value Numbers

| Metric | Value |
|---|---:|
| Delivery date | 2025-12-31 |
| Model | Ridge regression (selected model) |
| Baseload forecast | 76.49 EUR/MWh |
| Peak forecast | 80.85 EUR/MWh |
| Model confidence band | +/-14.5 EUR/MWh (backtest MAE) |
| OOS baseload avg (full window) | 83.16 EUR/MWh |
| OOS peak avg (full window) | 88.05 EUR/MWh |

## Key Drivers

- [bearish] Wind surge +6,750 MW vs prior day collapses residual load by 8,656 MW to 24,998 MW — 23rd percentile — crushing marginal cost support well below EXAA print of 84.00 EUR/MWh
- [bearish] Model baseload fair value 76.49 EUR/MWh implies basis vs. EXAA of -7.51 EUR/MWh, with 22 of 24 hours screening SELL vs. zero hours screening BUY
- [bearish] Load demand -3,703 MW vs prior day (New Year's Eve holiday suppression) amplifies the bearish residual load dynamic
- [bearish] NTC net transfer capacity at -3,200 MW (net export constraint) limits upside absorption of surplus generation, reinforcing downward price pressure on DE-LU
- [bullish] Trailing realised baseload at 78.57 EUR/MWh sits only 2.08 EUR below EXAA, suggesting EXAA is already pricing a meaningful premium to recent settlement history
- [bullish] Solar delta -1,798 MW vs prior day modestly tightens residual load at the margin relative to wind offset

Residual load: **24,998 MW** (23.3th percentile vs. pre-OOS history) | Wind Δ vs. D-1: **+6,750 MW** | Solar Δ: **-1,798 MW** | Load Δ: **-3,703 MW** | TTF front-month: **27.77 EUR/MWh**

## Basis vs. EXAA Reference

| Metric | Value (EUR/MWh) |
|---|---:|
| EXAA (Sequence 2) day-ahead auction price, same delivery day | 84.00 |
| Tomorrow's forecast vs. EXAA (basis) | -7.51 |
| Model backtest MAE (sizing denominator) | 14.50 |

EXAA settles its own day-ahead auction for BZN|DE-LU earlier the same day (~10:15 CET D-1) than the EPEX auction this model forecasts (~12:00 CET D-1) — a real, observable, point-in-time-safe pre-auction print, not a forecast or a different product. No EEX forward print is used (none could be sourced for the OOS delivery dates).

*Secondary context (not used for the call above): basis vs. the trailing realised baseload (avg of D-1 + D-7 actual) 78.57 was -2.08.*

**Hourly/block shape (model's own forecast, no forward dependency):** peak-offpeak spread +7.8 EUR/MWh, 2902 rich hours, 2472 cheap hours (442 negative), 876 scarcity hours. See `outputs/hourly_block_view.md` for the full hour-by-hour table.

## Invalidation Triggers

- Wind actual materially underperforms forecast — a shortfall of 5,000+ MW intraday would close the basis gap and invalidate the SELL
- Unexpected demand recovery (cold snap, industrial return) pushing load significantly above the 55,427 MW forecast
- Gas/CO2 complex spike — TTF front-month well above 27.77 EUR/MWh or CO2 proxy above 32.22 USD reprices the marginal cost stack toward EXAA levels
- NTC constraint reversal or significant cross-border flow change importing high-priced power into DE-LU and compressing the surplus

## Commentary

> SELL conviction MODERATE: 22 of 24 hours screen SELL, zero screen BUY, with model baseload fair value at 76.49 EUR/MWh versus EXAA at 84.00 EUR/MWh — a -7.51 EUR/MWh basis driven primarily by a wind surge of +6,750 MW and a demand-holiday load drop of -3,703 MW collapsing residual load to the 23rd percentile at 24,998 MW. Position sizing is QUARTER SIZE given a backtest MAE of +/-14.50 EUR/MWh that spans the full basis, meaning the signal clears the threshold on tally count but not on magnitude decisiveness. Key uncertainty is wind delivery risk on New Year's Eve — any material underperformance of the 28,807 MW wind forecast rapidly erodes the short thesis.

_Grounding check: PASSED — all numbers traced to the fact object._

## Figures

- `figures/prompt_curve.png` — model fair-value curve (baseload/peak, Q4 2025 slice)
- `figures/hourly_block_view.png` — hourly/block tradable view (model's own shape)
- `figures/validation_mae_by_hour.png`, `figures/validation_mae_by_regime.png` — model accuracy context

---

_This note is generated entirely from committed/cached pipeline outputs (`outputs/prompt_curve_view.md`, `outputs/hourly_block_view.md`, `ai_logs/commentary_cache.json`) — `python main.py` reproduces it deterministically, no API key or network access required._