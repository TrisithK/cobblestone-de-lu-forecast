# DE-LU Day-Ahead Power Price Forecasting

**Candidate:** Trisith Kittisriswai | trisithworld@gmail.com  
**Market:** Germany-Luxembourg (DE-LU) bidding zone, EPEX day-ahead, hourly, EUR/MWh  
**Forecast option:** A — next-day hourly day-ahead prices  
**OOS window:** 2025-12-08 → 2025-12-31 (24 days, 576 hourly predictions)

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.9+)
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

`python main.py` is the **single command** that reproduces everything end-to-end:
- QA on committed data snapshot → `outputs/qa_report.md`
- Feature engineering with point-in-time firewall
- Window-type tuning (expanding vs. rolling, 2024 validation) → `outputs/window_tuning.md`
- Walk-forward backtest (2025 test year) → `outputs/validation_metrics.md`
- OOS predictions → `predictions.csv`
- Prompt-curve translation (forward-basis view) → `outputs/prompt_curve_view.md`, `figures/prompt_curve.png`
- Hourly/block tradable DA view (model's own shape, no forward dependency) → `outputs/hourly_block_view.md`, `figures/hourly_block_view.png`
- LLM trader commentary → loaded from cache (`ai_logs/commentary_cache.json`)
- Static morning desk note (assembled from the above) → `outputs/morning_note.md`

**No API keys required to reproduce.** The LLM step runs from committed cache. Full run takes **~7 minutes** on a laptop, well within the 10-minute budget.

---

## API Keys (optional — only needed for re-fetching or regenerating LLM output)

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
# Edit .env and add:
# ENTSOE_API_KEY=<your token from transparency.entsoe.eu>
# ANTHROPIC_API_KEY=<your Anthropic key>
```

Keys are read via `python-dotenv`. **Never committed to version control.**

To re-fetch data from ENTSO-E:
```bash
python data/fetch_data.py
```

---

## Repo Structure

```
trisith_kittisriswai/
├── README.md              # this file
├── requirements.txt       # pinned dependencies
├── main.py                # single entry point
├── report.md              # 1-3 page write-up
├── predictions.csv        # OOS predictions: datetime (ISO 8601), y_pred (EUR/MWh)
├── .env.example           # template — copy to .env and fill in keys
├── src/
│   ├── config.py          # shared constants (OOS window, EEX reference, seeds)
│   ├── qa.py              # QA checks + report
│   ├── features.py        # feature engineering + assert_no_lookahead()
│   ├── models.py          # Ridge / LightGBM / depth-3 tree
│   ├── validation.py      # walk-forward backtest + metrics
│   ├── prompt_curve.py    # prompt-curve translation (forward-basis + hourly/block views)
│   ├── llm_commentary.py  # LLM commentary (pydantic schema, grounding check, cache)
│   └── morning_note.py    # static morning desk note assembly
├── data/
│   ├── fetch_data.py      # ENTSO-E + yfinance fetch script
│   ├── da_prices.parquet          # day-ahead prices snapshot (2019-2025)
│   ├── load_forecast.parquet      # day-ahead load forecast snapshot
│   ├── wind_solar_forecast.parquet # day-ahead wind + solar forecast snapshot
│   └── ttf_daily.parquet          # TTF front-month daily close snapshot
├── figures/               # EDA, validation, merit-order tree, prompt-curve, hourly-block plots
├── outputs/               # QA report, validation metrics, window tuning, prompt-curve + hourly-block views, morning note
└── ai_logs/               # LLM prompts, raw responses, committed cache
```

---

## Key Design Choices

- **Selected model: Ridge with merit-order features** — residual load (load − wind − solar) with hinge and quadratic nonlinearity terms, price lags D-1/D-7, TTF daily anchor, cyclic calendar encodings, and an optional `ttf × residual_load` interaction (Ridge-only) that lets gas re-slope the merit-order relationship rather than only shift its level. LightGBM is included as a nonlinearity-check challenger but not selected; see `outputs/validation_metrics.md` §4 for the principled selection rationale.
- **Point-in-time firewall enforced** — all features for delivery day D use only data knowable at ~12:00 D-1. Assertion in `src/features.py` raises if any forbidden timestamp appears.
- **Negative prices never clipped** — 3.3% of DE-LU hours go negative; treating them as zero is a domain error.
- **DST handled** — tz-aware `Europe/Berlin` index; 23-hour spring days and 25-hour fall days verified in QA.
- **No MAPE** — prices cross zero; MAPE is undefined and explosive near zero. MAE + RMSE used throughout.

---

## Reproducibility Checklist

| Check | Status |
|-------|--------|
| Public data only (ENTSO-E, Yahoo Finance) | ✓ |
| `python main.py` — single command, end-to-end | ✓ |
| Deterministic (seeds: numpy, random, LightGBM) | ✓ |
| <10 minutes on a laptop (data pre-committed; ~7 min measured) | ✓ |
| Committed data snapshot (`data/*.parquet`) | ✓ |
| Fetch script in `data/` (`data/fetch_data.py`) | ✓ |
| No secrets committed (`.env` in `.gitignore`) | ✓ |
| LLM step runs from cache (`ai_logs/commentary_cache.json`) | ✓ |
| Dependencies pinned (`requirements.txt`) | ✓ |
| No large/unused data | ⚠ partial — see note below |

> **Known deviation:** `data/` also retains the raw ENTSO-E manual-export CSVs (`GUI_*.csv`, ~120MB) used to build the committed parquet snapshots, rather than only the few-MB snapshot the brief describes. Of these, exactly one (`GUI_TOTAL_LOAD_DAYAHEAD_201901010000-202001010000.csv`) is still read at runtime, by `src/qa.py`'s forecast-vs-actual spot-check (it degrades gracefully to "skipped" if missing). The rest are unused leftovers from the fetch step and could be pruned; left in place for this submission so the spot-check input is traceable to its raw source.
