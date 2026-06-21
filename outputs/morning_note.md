# DE-LU Morning Desk Note — 2025-12-08

_Generated 2026-06-21 10:00 UTC from cached model + LLM outputs — static, no live data._

---

## Action

**SHORT / SELL** — Conviction: **MODERATE** — Size: **HALF SIZE (1/2 normal prompt risk)**

> Conviction/size already capped vs. what the basis magnitude alone implies — the forward reference behind this call is an illustrative placeholder, not a sourced EEX print (see Basis section below).

## Fair-Value Numbers

| Metric | Value |
|---|---:|
| Delivery date | 2025-12-08 |
| Model | Ridge regression (selected model) |
| Baseload forecast | 49.12 EUR/MWh |
| Peak forecast | 68.75 EUR/MWh |
| Model confidence band | +/-16.49 EUR/MWh (backtest MAE) |
| OOS baseload avg (full window) | 70.75 EUR/MWh |
| OOS peak avg (full window) | 83.93 EUR/MWh |

## Key Drivers

- Wind surge +11,107 MW vs prior day collapses residual load to 18th percentile (21,588 MW), heavily pressuring baseload
- Spot baseload forecast 49.12 EUR/MWh sits 32.04 EUR/MWh below illustrative front-week at 107.50 EUR/MWh, confirming sharp near-term bearish basis
- Residual load delta of -6,695 MW on the day underscores oversupplied merit order despite load lift of +5,012 MW

Residual load: **21,588 MW** (18.4th percentile vs. pre-OOS history) | Wind Δ vs. D-1: **+11,107 MW** | Solar Δ: **+600 MW** | Load Δ: **+5,012 MW** | TTF front-month: **27.27 EUR/MWh**

## Basis vs. Curve

| Comparison | Basis (EUR/MWh) | Reference status |
|---|---:|---|
| Front-week (primary, matched delivery window) | -32.04 | ILLUSTRATIVE — no public print found |
| Front-month (indicative context only — month mismatch) | -33.24 | Real, sourced print |

**Hourly/block shape (model's own forecast, no forward dependency):** peak-offpeak spread +20.6 EUR/MWh, 102 rich hours, 114 cheap hours (2 negative), 58 scarcity hours. See `outputs/hourly_block_view.md` for the full hour-by-hour table.

## Invalidation Triggers

- Wind realisation materially undershoots forecast, pushing residual load above 28,000 MW and erasing the generation surplus
- Intraday gas spike on TTF front-month well above 27.27 EUR/MWh reprices thermal floor and lifts spot clearing prices
- Unexpected demand surge beyond 58,972 MW load forecast (cold snap, industrial draw) tightening the residual load percentile above the 50th
- Front-week EEX print confirmed materially below the illustrative 107.50 EUR/MWh level, narrowing the basis and reducing short edge

## Commentary

> Sell DE-LU day-ahead baseload for 2025-12-08 with moderate conviction: the wind surge of +11,107 MW vs the prior day drives residual load to just 21,588 MW — the 18th percentile — putting the ridge model fair value at 49.12 EUR/MWh baseload and 68.75 EUR/MWh peak. The -6,695 MW residual load swing confirms a deeply oversupplied merit order even as demand lifts +5,012 MW, cementing the directional sell thesis. Key uncertainty is the model MAE of +/-16.49 EUR/MWh, which is wide relative to the forecast itself and leaves the range straddling materially different dispatch outcomes should wind verification disappoint.

_Grounding check: PASSED — all numbers traced to the fact object._

## Figures

- `figures/prompt_curve.png` — forward-basis view (OOS forecast vs. EEX reference)
- `figures/hourly_block_view.png` — hourly/block tradable view (model's own shape)
- `figures/validation_mae_by_hour.png`, `figures/validation_mae_by_regime.png` — model accuracy context

---

_This note is generated entirely from committed/cached pipeline outputs (`outputs/prompt_curve_view.md`, `outputs/hourly_block_view.md`, `ai_logs/commentary_cache.json`) — `python main.py` reproduces it deterministically, no API key or network access required._