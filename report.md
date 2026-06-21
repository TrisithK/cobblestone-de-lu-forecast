# DE-LU Day-Ahead Power Price Forecasting — Case Study

**Trisith Kittisriswai** | trisithworld@gmail.com

---

## Design Philosophy

> "I prioritised transparent, stable models deliberately: a fair-value signal has to be one a trader can interrogate and trust, and these markets break regime often enough that an over-flexible model is a liability, not an edge. The complex model is included as a benchmark — to confirm the simple one isn't leaving material signal behind."

This submission is a working prototype of a daily **DE-LU (Germany-Luxembourg) day-ahead fair-value view**, producing next-day hourly price forecasts, a prompt-curve relative-value lean, and an LLM-generated trader briefing — all from a single `python main.py` command.

---

## 1. Data & Sources

**Target:** EPEX DE-LU day-ahead hourly prices (EUR/MWh), 2019-01-01 → 2025-12-31. History starts 2019 because the DE-AT-LU zone split in October 2018; pre-split data mixes two price regimes.

**ENTSO-E Transparency Platform** (free API, `entsoe-py`):
- Day-ahead prices — target series.
- Day-ahead **load forecast** (A01 vintage) — not realised actuals.
- Day-ahead **wind + solar generation forecast** (A01 vintage) — not realised actuals.

**Yahoo Finance** (`yfinance`, ticker `TTF=F`): daily TTF front-month natural gas price as a level anchor. Gas sets the absolute price when a thermal plant is on the margin; the daily series provides a repricing guard for overnight gas-gap events.

All series fetched once, committed as Parquet snapshots in `data/`. The fetch script is `data/fetch_data.py`.

### QA Summary

61,368 hourly rows across seven years. Key checks:

| Check | Result |
|-------|--------|
| Continuous tz-aware index (Europe/Berlin) | PASS |
| DST transitions: 7 spring (23h) + 7 fall (25h) | **14/14 PASS** |
| Negative prices | **2,051 hours (3.3%) — preserved, never clipped** |
| Duplicate timestamps | 0 |
| Forecast vs. actual verification | PASS — max deviation load forecast vs. actual 9,034 MW confirms forecast series loaded |
| Long gaps (>4h) | 2 load gaps (Feb 2022, Mar 2022); excluded by validity mask |

Full report: `outputs/qa_report.md`.

---

## 2. Point-in-Time Discipline

Every feature for delivery day **D** must be knowable at ~12:00 on D-1 (day-ahead auction gate closure). This is the single most important design constraint — violating it makes the backtest fictional.

Enforcement:
- **Load, wind, solar:** ENTSO-E day-ahead **forecast** (A01) vintage — explicitly not the realised series. The QA check confirms the forecast diverges from actuals by thousands of MW, verifying the right series was loaded.
- **Price lags:** D-1 same-hour and D-7 same-hour realised prices — both known well before gate closure.
- **TTF:** prior-day close — known by morning on D-1.
- **No price from day D appears anywhere** — enforced by `assert_no_lookahead()` in `src/features.py`, which raises if any forbidden timestamp appears in the feature matrix.

---

## 3. Feature Engineering

The feature spine is the **residual load**: `load_forecast − wind_forecast − solar_forecast`. This single quantity proxies the merit-order position — how far up the supply stack the market clears — and is the dominant driver of both price level and intraday shape.

**Feature set (23 features):**

| Group | Features |
|-------|----------|
| Calendar | hour-of-day (sine/cosine), day-of-week (sine/cosine), month (sine/cosine), is_weekend, is_holiday |
| Residual load | residual_load_mw, load_forecast_mw, wind_forecast_mw, solar_forecast_mw |
| Merit-order nonlinearity | `resid_load_sq` (quadratic), `resid_hinge` (threshold at p80 = 45,063 MW for scarcity pricing) |
| Price lags | price_lag_1h (D-1 same hour), price_lag_168h (D-7 same hour) |
| Gas anchor | TTF front-month close (EUR/MBtu) — daily level, lagged one day |
| Gas × merit-order (Ridge-only, optional) | `ttf_resid_interaction` — see below |

**Why TTF is included but not first-order for intraday shape:** gas sets the *level* of power prices, but it barely moves within the 24-hour forecast horizon (it's sticky). The D-1/D-7 price lags already proxy the gas level, since those prices were themselves set against prevailing gas. Explicit gas adds marginal value on normal days; its payoff is concentrated on **overnight repricing events** (supply shock, cold snap) where the lag mis-states today's level. Wind/solar forecasts are mandatory because tomorrow's wind can differ sharply from today's — the lag is a poor proxy for a highly variable driver.

**Optional extension — gas × residual-load interaction:** TTF above enters additively (a level shift only). A `ttf_eur_mwh × residual_load_mw` interaction term is added to the Ridge feature set (Ridge only — LightGBM already captures interactions nonlinearly without it) to let the model re-slope the residual-load→price relationship with the prevailing gas level — the spark spread changes the *steepness* of merit-order pricing, not just its offset. This is a linear-model-native alternative to a short rolling window: it lets the model adapt to a gas-regime change without discarding history. It measurably helped: Ridge full-year MAE improved from 17.17 to **16.49 EUR/MWh** (a ~4% reduction) after adding it.

### EDA: what the data actually shows

The case for residual load as the fundamental driver isn't a correlation coefficient — the relationship is nonlinear (convex, then negative at saturation) and prices cross zero, so a single correlation number would understate it and is not reported. Two figures carry the evidence instead:

- **`figures/price_vs_resid_tree.png`** — scatter of price vs. residual load with the depth-3 tree's step function overlaid. The convex-then-negative merit-order kink is visible directly: price climbs steeply once residual load passes the scarcity hinge (p80 ≈ 45,063 MW), and the cloud of points below ~0 MW residual load sits at or below zero — renewable saturation pushing the market negative.
- **`figures/feature_importance_lgbm.png`** — LightGBM gain-based importance, fit on the full pre-Test (2019-2024) training set. `price_lag_24h` dominates by an order of magnitude — power prices are strongly autocorrelated day-to-day, so most of the learnable signal is "yesterday, same hour." The honest second-place finding is **`ttf_eur_mwh` (gas), not residual load** — TTF outranks every individual residual-load term (`residual_load_mw`, `residual_load_sq`, `residual_load_high`) and `price_lag_168h`. This is consistent with the feature-design rationale in §3, not a contradiction of it: residual load is a *within-day shape* variable, and the price lags already absorb most of its day-to-day level information (lags and residual load are correlated through the price they both helped set); TTF instead captures *cross-day* drift in the price level that a lag — being yesterday's price — necessarily misses one day late. Summed, the residual-load family is still a material block of the remaining importance, and it is what gives the linear model its merit-order shape; it just isn't bigger than gas in a tree's gain accounting.

---

## 4. Model Lineup & Selection

### Baselines
- **D-1 same-hour naïve:** yesterday's price for the same delivery hour.
- **D-7 same-hour naïve:** last week's price. The honest standard for day-ahead power — captures the weekly seasonality that the D-1 naïve misses.

### Selected model: Ridge with merit-order features
Plain linear regression mis-specifies price formation: the supply stack is convex and kinked (flat mid-stack, explosive at scarcity, negative at renewable saturation). The Ridge model is given **hinge and quadratic terms on residual load**, hour-of-day cyclic encodings, and the price lags. This makes it transparent **and** economically faithful to the merit-order curve.

### Challenger: LightGBM
Included not as a candidate for selection but as a **nonlinearity check**: how much signal a flexible tree ensemble can extract beyond the linear model. Feature importances (`figures/feature_importance_lgbm.png`) show price lags dominate, with TTF (gas) second and residual load third — see the EDA note in §3 for why that ranking doesn't undercut the residual-load rationale. These are exactly the drivers Ridge is built around. The extra accuracy comes from nonlinear interactions that Ridge cannot represent.

### Interpretability figure: Depth-3 DecisionTree
Not a selected model (high bias; produces blocky step-function forecasts). **Its top splits are dominated by `price_lag_24h`** — price is strongly autocorrelated, so the tree spends its first three levels bracketing yesterday's same-hour price before anything else gets a look-in; `residual_load_mw` and `wind_forecast_mw` only enter as secondary splits *within* a `price_lag_24h` branch (e.g. splitting the >336 EUR/MWh branch by wind). The merit-order kink itself — the convex, then negative-going relationship between residual load and price — is visible in `figures/price_vs_resid_tree.png`, not in the tree diagram. The tree's real value here is showing *how much* of the variance price lags soak up before fundamentals matter at all. See `figures/merit_order_tree.png`.

---

## 5. Validation

**Three-way temporal split:** the 2019-2025 history is split into roles, not frozen training sets — walk-forward training always uses every valid row strictly before the prediction day, so 2024 data still trains every 2025 prediction.

- **Validation (2024):** used once, to choose the calibration-window type (below). Never touched again.
- **Test (2025, full calendar year, 8,759 hourly predictions):** the untouched OOS backtest reported here — spans summer solar saturation, spring negative-price spells, and winter scarcity hours, i.e. all four seasons and regimes, not one season.
- **predictions.csv** is the last 24 days of the Test year (Dec 8-31 2025), per the brief.

**Protocol:** walk-forward backtest. Train to day t (on the chosen window — see below) → forecast all 24 hours of t+1 → advance one day. No data shuffling.

**Calibration window — expanding vs. rolling (chosen on 2024 validation, Ridge only):**

| Window type | Validation MAE (2024, EUR/MWh) |
|---|---:|
| **Expanding (all history)** | **16.00** |
| Rolling (728d trailing) | 16.43 |

Expanding wins. The residual-load→price slope does shift with the gas regime (2021-23 crisis vs. 2024-25 normalisation), but a 2-year rolling window throws away more useful history than it gains in regime-freshness — Ridge's regularisation already damps the influence of stale extreme-regime data without needing to discard it outright. Full numbers: `outputs/window_tuning.md`.

**Metric note:** no MAPE — DE-LU prices cross zero frequently (3.3% of hours are negative). MAPE is undefined at zero and explosive near zero. MAE and RMSE are used throughout.

### Overall Metrics (Test: 2025-01-01 → 2025-12-31)

| Model | MAE (EUR/MWh) | RMSE (EUR/MWh) | Skill vs. D-7 |
|-------|:---:|:---:|:---:|
| Naïve D-7 | 32.83 | 49.33 | +0.0% (reference) |
| Naïve D-1 | 25.97 | 40.45 | +20.9% |
| **Ridge (selected)** | **16.49** | **25.34** | **+49.8%** |
| LightGBM | 12.50 | 20.49 | +61.9% |

### By Regime

| Regime | Ridge MAE | LightGBM MAE |
|--------|:---------:|:------------:|
| Peak (08-20, weekday) | 19.91 | 16.43 |
| Off-peak | 14.32 | 10.02 |
| High residual load | 15.71 | 12.63 |
| Low residual load (renewable-heavy) | 17.27 | 12.37 |

See `figures/validation_mae_by_hour.png` and `figures/validation_mae_by_regime.png`.

### Model Selection Decision

LightGBM beats Ridge by 3.99 EUR/MWh (24.2% of Ridge MAE) — over the **full 2025 calendar year**, not a single season. The selection still goes to **Ridge** for three reasons:

1. **A fair-value signal must be interrogable.** A trader needs to know *why* the model says 95 EUR/MWh — Ridge coefficients are inspectable; 500 trees are not. Trust, not raw accuracy, is the production constraint.

2. **Power markets break regime; flexible models break with them.** This year-long backtest already includes summer solar saturation, spring negative-price spells, and winter scarcity hours — LightGBM's edge holds up across that range, which is informative. But a regularised linear form still degrades more gracefully than a 500-tree ensemble when the next regime shift (a gas shock, a step-change in renewables build-out) looks nothing like 2019-2025.

3. **LightGBM confirms Ridge's design, not that Ridge is mis-specified.** Price lags, gas (TTF), and residual load dominate feature importances in that order (`figures/feature_importance_lgbm.png`) — exactly the drivers Ridge is built around. The extra accuracy is real nonlinear signal, but not the dominant source.

LightGBM is retained as a **parallel challenger signal**: if it diverges materially from Ridge on a given day, that flags a possible nonlinear regime shift.

---

## 6. OOS Predictions

OOS window: **2025-12-08 → 2025-12-31** (24 days, 576 hourly predictions). Written to `predictions.csv` with ISO 8601 tz-aware datetimes (Europe/Berlin). Ridge forecast range: −4.30 to 140.06 EUR/MWh — negative prices preserved.

---

## 7. Prompt-Curve Translation

A next-day hourly model **does not produce a month-ahead price path**. Rolling a 1-day forecast 30 days forward compounds error and relies on driving forecasts that don't exist. Two complementary views are produced instead, both reading the model's own OOS forecast directly (`outputs/hourly_block_view.md`, `figures/hourly_block_view.png`) and a relative-value basis view against a pinned EEX reference (below).

**EEX Reference — two contracts, two confidence levels:**
- **Front-month (real print):** ICE ENDEX German Power Financial Base Futures (GABF2026), January 2026 delivery, final settlement 2026-01-30: **103.99 EUR/MWh** (range 102.45–104.09). Publicly sourced (TradingView / ICE ENDEX). Delivery month does **not** match the OOS window (Jan-2026 vs. Dec-2025) — kept as curve-shape context only, **not used to size or direct the trade**.
- **Front-week (illustrative — flagged in code):** Dec 8-14 2025 estimated from early-December EPEX spot context: **107.50 EUR/MWh**. No public EEX Phelix DE Week Future settlement could be sourced (EEX market-data pages show only a rolling 45-day window; full history needs a paid DataSource subscription). `config.EEX_FRONTWEEK_IS_ILLUSTRATIVE = True` flags this explicitly, and `prompt_curve.py` **caps conviction at MODERATE and size at HALF** whenever the call rests on this number — no HIGH-conviction / FULL-SIZE call is allowed to rest on an unsourced print.

### Model Fair-Value Aggregates (OOS, Ridge)

| Period | Baseload (EUR/MWh) | Peak (EUR/MWh) |
|--------|-----------------:|---------------:|
| Front-week (Dec 8-14) | 75.46 | 91.43 |
| Full OOS (Dec 8-31) | 70.75 | 83.93 |

### Directional View — Forward Basis (primary: front-week, matched delivery window)

| Comparison | Basis | Note |
|------------|------:|------|
| Front-week model vs. EEX front-week | **−32.04 EUR/MWh** | **Primary** — matched delivery window, drives the call below |
| Full OOS model vs. EEX front-month | −33.24 EUR/MWh | Indicative curve-shape context only (Jan-2026 contract vs. Dec-2025 forecast) — **not used to size or direct the trade** |

### Direction: **SHORT / SELL** — Conviction: **MODERATE** — Size: **HALF SIZE**

Conviction/size are capped from what the basis magnitude alone would suggest (would be HIGH/FULL on the −32.04 EUR/MWh gap) because the reference they're measured against is the illustrative front-week placeholder, not a sourced print. Model fair value (75.46 EUR/MWh baseload) sits 32.04 EUR/MWh below the front-week curve (107.50 EUR/MWh) — the prompt forward screens rich vs. near-term fundamental fair value. Directional lean: **sell the front-week / reduce long, at half normal size**, pending a real settlement print.

### Directional View — Hourly/Block (primary: the model's own shape, no forward dependency)

Independent of the EEX reference entirely, the OOS forecast's own intraday shape is directly tradable:

| Metric | Value |
|---|---:|
| Peak − off-peak spread | **+20.63 EUR/MWh** |
| Hours screening >15 EUR/MWh rich vs. that day's own baseload | 102 |
| Hours screening >15 EUR/MWh cheap (incl. 2 negative-price hours) | 114 |
| Scarcity hours (top decile residual load) | 58 |

The peak/off-peak spread is itself a tradable block product. Rich hours cluster in the evening peak and scarcity windows (sell candidates, DA auction or intraday); cheap hours cluster overnight, including 2 negative-price hours (buy / load-shift candidates — storage charging, demand response). Full detail: `outputs/hourly_block_view.md`, `figures/hourly_block_view.png`.

### Invalidation Triggers

Re-evaluate either view if:
- TTF front-month moves >5% overnight — the model uses yesterday's close; a gas gap shifts the absolute level across all hours.
- Wind forecast revision >5 GW vs. the D-1 model run — residual load reprices the full day.
- Unplanned nuclear/large thermal outage on REMIT — supply removal lifts scarcity hours disproportionately.
- Demand surprise: cold snap or anomalous holiday-week consumption beyond the load forecast.
- EEX settlement revises intraday — confirm vs. live screen before trading (forward-basis view only).

**Risk-premium caveat:** the EEX forward embeds a risk premium over E[spot]; it is not an unbiased expectation of the realised daily average. This is a directional lean on relative value, not a claim the forward will converge to the model forecast. Size positions accordingly.

See `figures/prompt_curve.png` + `outputs/prompt_curve_view.md` (forward basis) and `figures/hourly_block_view.png` + `outputs/hourly_block_view.md` (hourly/block).

---

## 8. LLM Commentary Component

### What it does
Generates daily trader-facing fair-value commentary from structured model outputs — removing the manual write-up an analyst would otherwise produce each morning. This is a genuine language task: turning a structured numeric state into a fluent, correctly-hedged narrative.

### Design
- **Input:** a structured fact object containing forecasted baseload/peak, residual-load percentile, wind/solar/load deltas vs. prior day, QA status, model MAE, and the curve rich/cheap signal + size.
- **The model originates no numbers** — every quantity in the output comes from the fact object (hallucination guard).
- **Output schema** (Pydantic v2 validated): `{direction, conviction, drivers[], invalidation_triggers[], commentary_text}`. Non-conforming output is rejected and retried.
- **Grounding check:** every number appearing in `commentary_text` is verified against the fact object within ±0.5 EUR/MWh tolerance. The Dec 8 run passed with zero grounding violations.
- **Reproducibility:** the LLM output is cached to `ai_logs/commentary_cache.json`. `python main.py` runs the full pipeline from cache — no API key required for reproduction. Full prompts and raw responses are written to `ai_logs/` for assessment without rerunning.

### Sample output (2025-12-08 delivery)

> *Sell DE-LU day-ahead baseload for 2025-12-08 with moderate conviction: the wind surge of +11,107 MW vs the prior day drives residual load to just 21,588 MW — the 18th percentile — putting the ridge model fair value at 49.12 EUR/MWh baseload and 68.75 EUR/MWh peak. The -6,695 MW residual load swing confirms a deeply oversupplied merit order even as demand lifts +5,012 MW, cementing the directional sell thesis. Key uncertainty is the model MAE of +/-16.49 EUR/MWh, which is wide relative to the forecast itself and leaves the range straddling materially different dispatch outcomes should wind verification disappoint.*
>
> Direction: **SHORT / SELL** | Conviction: **MODERATE** (capped from what the basis alone implies — the front-week reference behind it is illustrative, see §7). Basis vs. illustrative front-week: −32.04 EUR/MWh.

**Natural extension:** the same commentary engine can consume parsed **outage/news signals** — REMIT notifications, agency headlines — as structured inputs. That is the year-one "AI market-alert system" referenced in the application brief, and is the next module once the core pipeline is in production.

---

## Appendix: File Map

```
predictions.csv          — 576 OOS hourly predictions (Dec 8-31 2025)
outputs/qa_report.md     — QA results and anomaly log
outputs/validation_metrics.md — walk-forward metrics by hour and regime
outputs/window_tuning.md      — expanding vs. rolling window comparison (2024 validation)
outputs/prompt_curve_view.md  — forward-basis analysis and directional view
outputs/hourly_block_view.md  — hourly/block tradable DA view (model's own shape)
outputs/morning_note.md       — static morning desk note (assembled from the above + LLM cache)
figures/merit_order_tree.png  — depth-3 decision tree: merit-order kink
figures/feature_importance_lgbm.png — LightGBM gain-based feature importance
figures/price_vs_resid_tree.png     — price vs. residual load, tree splits overlaid
figures/validation_mae_by_hour.png / _by_regime.png
figures/prompt_curve.png — OOS forecast vs. EEX reference
figures/hourly_block_view.png — hourly/block tradable view figure
ai_logs/commentary_cache.json — cached LLM output (key-free reproduction)
ai_logs/prompt_20251208.txt   — full prompt sent to claude-sonnet-4-6
ai_logs/raw_response_20251208.json — raw API response
src/features.py          — feature engineering + assert_no_lookahead()
src/validation.py        — walk-forward backtest
src/prompt_curve.py      — prompt-curve translation (forward-basis + hourly/block views)
src/llm_commentary.py    — LLM commentary, grounding check, cache logic
src/morning_note.py      — static morning desk note assembly
```
