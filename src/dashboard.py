"""
dashboard.py — Trader-facing fair-value-vs-EXAA HTML dashboard (v2 round 5).

Two-layer architecture, deliberately not one ad-hoc script:
  1. build_dashboard_data() — single source of truth. Loads every input from
     already-committed pipeline outputs (predictions.csv, the data/ parquet
     snapshot, the cached LLM fact object, outputs/validation_metrics.md),
     computes everything in plain arithmetic, and writes
     outputs/dashboard_data.json. No number is invented here.
  2. render_dashboard_html() — a pure function from that dict to HTML. Inlines
     the JSON so the page is self-contained, and references the vendored
     Chart.js asset (static/chart.umd.min.js) by relative path — no CDN, the
     page renders fully offline.

Per-hour decision layer (the core of this module — there is NO single
whole-day call): for each delivery hour h,
    basis_h      = fair_value_h - EXAA_h
    conviction_h = |basis_h| / MAE_for_hour_h   (Ridge's own by-hour backtest MAE)
    direction_h  = FLAT if conviction_h < config.FLAT_CONVICTION,
                   else SELL if basis_h < 0, else BUY
This is pure arithmetic, never an LLM judgement — see compute_hourly_decisions()
below. The identical formula is reused at block level (baseload / peak /
off-peak), using each block's average basis divided by that block's average
MAE — see compute_block_decision(). llm_commentary.py imports these same
functions so its fact object and this dashboard never disagree.

Delivery day: defaults to the last day in predictions.csv (2025-12-31 in this
build) but is a parameter. The header labels this explicitly as a STATIC
BACKTEST REPLAY — the committed snapshot ends 2025-12-31; this is not a live
feed and implies no refresh schedule.

Honesty constraints carried over from the rest of this build (do not undo):
  - No MAPE anywhere — DE-LU prices cross zero / go negative.
  - CO2 stays the CARB.L ETC proxy in USD / the unitless gas_co2_pressure_index
    composite — never presented as an official EUR/tonne EUA print or gauge.
  - No outage/REMIT capacity panel — that feature is excluded by design
    (CLAUDE.md §2); the footer notes this rather than silently omitting why.
  - No fabricated trading-cost "edge-after-buffer" gate and no OPEN/SELECTIVE
    gate — the only edge logic is basis vs. the model's own MAE, above.
  - The selected model is named "Ridge", never "Generalized Linear Model".
  - NO realised/settled EPEX price (Sequence 1, da_prices.parquet) anywhere on
    this page. That series is this model's own training TARGET — by
    construction it isn't known until the EPEX auction it concerns has
    cleared (~12:00 CET D-1 gate closure), strictly after EXAA's own auction
    (~10:15 CET D-1) and long after a trader would be using this page to
    decide a bid. Showing it alongside Fair Value/EXAA would be the same
    point-in-time violation CLAUDE.md §3 Rule 1 forbids for model features,
    just applied to a dashboard instead of a feature matrix. (Removed after
    a correctness review — see CLAUDE.md §14 item 13.)
"""

import html as html_lib
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

import holidays
import numpy as np
import pandas as pd

from config import FLAT_CONVICTION

_ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(_ROOT, "data")
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
AI_LOGS_DIR = os.path.join(_ROOT, "ai_logs")
STATIC_DIR  = os.path.join(_ROOT, "static")

DASHBOARD_DATA_PATH = os.path.join(OUTPUTS_DIR, "dashboard_data.json")
DASHBOARD_HTML_PATH = os.path.join(OUTPUTS_DIR, "fair_value_dashboard.html")

# Same threshold + definition used by prompt_curve.py's hourly/block view —
# imported, not redefined, so "rich/cheap vs. the day's own baseload" means
# the same thing everywhere in this build.
from prompt_curve import RICH_CHEAP_THRESHOLD_EUR  # noqa: E402

_DE_HOLIDAYS = holidays.Germany()


def _is_peak(ts: pd.Timestamp) -> bool:
    """EEX peak convention: hours 08-20 on weekdays, excl. German public holidays."""
    return (8 <= ts.hour <= 20) and (ts.dayofweek < 5) and (ts.date() not in _DE_HOLIDAYS)


# ---------------------------------------------------------------------------
# Pure decision arithmetic — imported by llm_commentary.py too, so the LLM's
# fact object and this dashboard are always computed from the identical
# formula and never disagree.
# ---------------------------------------------------------------------------

def compute_hourly_decisions(
    fv: pd.Series,
    exaa: pd.Series,
    mae_by_hour: dict,
    flat_conviction: float = FLAT_CONVICTION,
) -> pd.DataFrame:
    """
    fv, exaa: same tz-aware hourly index, one delivery day (or any span — the
        formula is per-hour and doesn't care).
    mae_by_hour: {hour_of_day (0-23): Ridge backtest MAE, EUR/MWh}.

    Returns a DataFrame indexed like fv with columns:
        fv, exaa, basis, mae, conviction, direction
    """
    df = pd.DataFrame({"fv": fv, "exaa": exaa.reindex(fv.index)})
    df["basis"] = df["fv"] - df["exaa"]
    df["mae"] = [mae_by_hour[ts.hour] for ts in df.index]
    df["conviction"] = df["basis"].abs() / df["mae"]
    df["direction"] = np.where(
        df["conviction"] < flat_conviction, "FLAT",
        np.where(df["basis"] < 0, "SELL", "BUY"),
    )
    return df


def compute_block_decision(
    hourly: pd.DataFrame,
    mask: pd.Series,
    label: str,
    flat_conviction: float = FLAT_CONVICTION,
) -> dict:
    """
    hourly: DataFrame from compute_hourly_decisions().
    mask: boolean Series aligned to hourly.index selecting the block's hours.

    Block conviction = |block average basis| / block average MAE — both
    averaged over the same hours, not a mean of per-hour convictions.
    """
    sub = hourly.loc[mask]
    fv_avg = float(sub["fv"].mean())
    exaa_avg = float(sub["exaa"].mean())
    basis_avg = fv_avg - exaa_avg
    mae_avg = float(sub["mae"].mean())
    conviction = abs(basis_avg) / mae_avg if mae_avg else 0.0
    if conviction < flat_conviction:
        direction = "FLAT"
    elif basis_avg < 0:
        direction = "SELL"
    else:
        direction = "BUY"
    return {
        "label": label,
        "n_hours": int(mask.sum()),
        "fv_avg_eur": round(fv_avg, 2),
        "exaa_avg_eur": round(exaa_avg, 2),
        "basis_eur": round(basis_avg, 2),
        "mae_avg_eur": round(mae_avg, 2),
        "conviction": round(conviction, 3),
        "direction": direction,
    }


def compute_tally(hourly: pd.DataFrame) -> dict:
    counts = hourly["direction"].value_counts()
    return {
        "buy": int(counts.get("BUY", 0)),
        "sell": int(counts.get("SELL", 0)),
        "flat": int(counts.get("FLAT", 0)),
    }


# ---------------------------------------------------------------------------
# validation_metrics.md parser — by-hour Ridge MAE and overall Test metrics.
# Parses this build's own generated artifact (validation.py's _write_report());
# not arbitrary external markdown, so simple string splitting on the section
# headers it always writes is acceptable here.
# ---------------------------------------------------------------------------

def _parse_markdown_table(block: str) -> pd.DataFrame:
    lines = [l for l in block.strip().splitlines() if l.strip().startswith("|")]
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    header, sep, *data = rows
    return pd.DataFrame(data, columns=[h.strip() for h in header])


def _load_validation_metrics() -> dict:
    path = os.path.join(OUTPUTS_DIR, "validation_metrics.md")
    with open(path) as f:
        text = f.read()

    sec1 = text.split("## 1. Overall Metrics")[1].split("## 2.")[0]
    overall_df = _parse_markdown_table(sec1)
    overall = {}
    for _, row in overall_df.iterrows():
        overall[row["model"].strip()] = {
            "mae": float(row["MAE"]),
            "rmse": float(row["RMSE"]),
            "skill_vs_d7": row["Skill_vs_D7"].strip(),
        }

    sec2 = text.split("## 2. MAE by Hour of Day")[1].split("## 3.")[0]
    hour_df = _parse_markdown_table(sec2)
    mae_by_hour = {int(row["hour"]): float(row["Ridge"]) for _, row in hour_df.iterrows()}
    lgbm_mae_by_hour = {int(row["hour"]): float(row["LightGBM"]) for _, row in hour_df.iterrows()}

    window_match = re.search(r"Training window: \*\*(\w+)\*\*", text)
    training_window = window_match.group(1) if window_match else None

    backtest_period_match = re.search(
        r"Backtest period: ([\d-]+) .* ([\d-]+) ", text
    )

    return {
        "overall": overall,
        "mae_by_hour": mae_by_hour,
        "lgbm_mae_by_hour": lgbm_mae_by_hour,
        "training_window": training_window,
    }


def _safe_round(value, ndigits=2):
    """None/NaN-safe round() — JSON has no NaN, so missing data becomes null."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return round(f, ndigits)


def _series_to_records(series: pd.Series, ndigits=2) -> list:
    return [_safe_round(v, ndigits) for v in series.values]


# ---------------------------------------------------------------------------
# Layer 1: build_dashboard_data() — single source of truth
# ---------------------------------------------------------------------------

def build_dashboard_data(delivery_date: Optional[object] = None) -> dict:
    """
    Loads every input from already-committed pipeline outputs, computes the
    per-hour and block decision layers, and returns the full dashboard dict
    (also written to outputs/dashboard_data.json by build_dashboard() below).

    delivery_date: defaults to the last day present in predictions.csv.
    """
    preds = pd.read_csv(os.path.join(_ROOT, "predictions.csv"))
    preds["datetime"] = pd.to_datetime(preds["datetime"], utc=True).dt.tz_convert("Europe/Berlin")
    preds = preds.set_index("datetime").sort_index()
    fv_full = preds["y_pred"]

    if delivery_date is None:
        delivery_date = fv_full.index.max().date()
    dt = pd.Timestamp(str(delivery_date), tz="Europe/Berlin")

    fv_day = fv_full.loc[fv_full.index.normalize() == dt.normalize()]
    if len(fv_day) == 0:
        raise ValueError(
            f"No predictions found for delivery date {delivery_date} in predictions.csv "
            f"(available range: {fv_full.index.min().date()} -> {fv_full.index.max().date()})"
        )
    idx = fv_day.index  # this delivery day's own hourly index — 23/24/25 rows, DST-safe by construction

    # NOTE: data/da_prices.parquet (the EPEX Sequence 1 settlement) is
    # deliberately NOT loaded here. That series is this model's own training
    # target — by construction it isn't known until the auction it concerns
    # clears (~12:00 CET D-1 gate closure), after EXAA's own auction
    # (~10:15 CET D-1) and after the point a trader would use this page to
    # decide a bid. Showing it would be a point-in-time violation. See
    # module docstring and CLAUDE.md §14 item 13.
    exaa = pd.read_parquet(os.path.join(DATA_DIR, "exaa_prices.parquet"))["exaa_price_eur_mwh"]
    load = pd.read_parquet(os.path.join(DATA_DIR, "load_forecast.parquet"))["load_forecast_mw"]
    ws   = pd.read_parquet(os.path.join(DATA_DIR, "wind_solar_forecast.parquet"))
    wind, solar = ws["wind_forecast_mw"], ws["solar_forecast_mw"]
    fr_load = pd.read_parquet(os.path.join(DATA_DIR, "fr_load_forecast.parquet"))["load_forecast_fr_mw"]
    fr_wind = pd.read_parquet(os.path.join(DATA_DIR, "fr_wind_forecast.parquet"))["wind_forecast_fr_mw"]

    exaa_day = exaa.reindex(idx)
    load_day = load.reindex(idx)
    wind_day = wind.reindex(idx)
    solar_day = solar.reindex(idx)
    residual_day = load_day - wind_day - solar_day
    fr_load_day = fr_load.reindex(idx)
    fr_wind_day = fr_wind.reindex(idx)

    val = _load_validation_metrics()
    mae_by_hour = val["mae_by_hour"]

    # --- Per-hour decision layer (pure arithmetic, see module docstring) ---
    hourly = compute_hourly_decisions(fv_day, exaa_day, mae_by_hour)

    day_fv_baseload = float(fv_day.mean())
    hourly["deviation_vs_own_baseload_eur"] = hourly["fv"] - day_fv_baseload
    hourly["shape_flag"] = np.where(
        hourly["deviation_vs_own_baseload_eur"] >= RICH_CHEAP_THRESHOLD_EUR, "RICH",
        np.where(hourly["deviation_vs_own_baseload_eur"] <= -RICH_CHEAP_THRESHOLD_EUR, "CHEAP", "NEUTRAL"),
    )

    # --- Block decision layer ---
    is_peak_mask = pd.Series([_is_peak(ts) for ts in idx], index=idx)
    baseload_mask = pd.Series(True, index=idx)
    offpeak_mask = ~is_peak_mask

    blocks = [
        compute_block_decision(hourly, baseload_mask, "Baseload (00-24)"),
        compute_block_decision(hourly, is_peak_mask, "Peak (08-20 weekday)"),
        compute_block_decision(hourly, offpeak_mask, "Off-peak"),
    ]
    tally = compute_tally(hourly)

    # --- LLM fact object + validated output (already on disk, committed) ---
    with open(os.path.join(AI_LOGS_DIR, "commentary_cache.json")) as f:
        commentary_cache = json.load(f)
    fact = commentary_cache["fact_object"]
    llm_out = commentary_cache["validated_output"]

    overall_ridge = val["overall"]["Ridge"]
    overall_lgbm = val["overall"]["LightGBM"]

    data = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "delivery_date": str(delivery_date),
            "mode": "STATIC BACKTEST REPLAY",
            "mode_note": (
                "This page replays one delivery day from the committed 2019-2025 data "
                "snapshot. There is no live feed and no refresh schedule — the snapshot "
                "ends 2025-12-31. Regenerate with build_dashboard(delivery_date=...) "
                "(src/dashboard.py) to replay a different day in predictions.csv."
            ),
            "flat_conviction": FLAT_CONVICTION,
            "rich_cheap_threshold_eur": RICH_CHEAP_THRESHOLD_EUR,
            "selected_model": "Ridge",
            "training_window": val["training_window"],
        },
        "kpi": {
            "fv_baseload_eur": blocks[0]["fv_avg_eur"],
            "exaa_baseload_eur": blocks[0]["exaa_avg_eur"],
            "basis_baseload_eur": blocks[0]["basis_eur"],
            "basis_baseload_conviction": blocks[0]["conviction"],
            "baseload_direction": blocks[0]["direction"],
            "selected_model": "Ridge",
            "model_mae_eur": overall_ridge["mae"],
            "model_rmse_eur": overall_ridge["rmse"],
            "challenger_model": "LightGBM",
            "challenger_mae_eur": overall_lgbm["mae"],
            "challenger_rmse_eur": overall_lgbm["rmse"],
        },
        "hourly": {
            "timestamps": [ts.isoformat() for ts in idx],
            "hour": [int(ts.hour) for ts in idx],
            "fv_eur": _series_to_records(hourly["fv"]),
            "exaa_eur": _series_to_records(hourly["exaa"]),
            "basis_eur": _series_to_records(hourly["basis"]),
            "mae_eur": _series_to_records(hourly["mae"]),
            "conviction": _series_to_records(hourly["conviction"], 3),
            "direction": hourly["direction"].tolist(),
            "deviation_vs_own_baseload_eur": _series_to_records(hourly["deviation_vs_own_baseload_eur"]),
            "shape_flag": hourly["shape_flag"].tolist(),
        },
        "tally": tally,
        "blocks": blocks,
        "drivers": {
            "timestamps": [ts.isoformat() for ts in idx],
            "de_load_mw": _series_to_records(load_day, 0),
            "de_wind_mw": _series_to_records(wind_day, 0),
            "de_solar_mw": _series_to_records(solar_day, 0),
            "de_residual_load_mw": _series_to_records(residual_day, 0),
            "fr_load_mw": _series_to_records(fr_load_day, 0),
            "fr_wind_mw": _series_to_records(fr_wind_day, 0),
        },
        "level_drivers": {
            "ttf_front_month_eur_mwh": fact.get("ttf_front_month_eur_mwh"),
            "co2_proxy_usd": fact.get("co2_proxy_usd"),
            "gas_co2_pressure_index": fact.get("gas_co2_pressure_index"),
            "ntc_net_transfer_capacity_mw": fact.get("ntc_net_transfer_capacity_mw"),
            "residual_load_fr_forecast_mw": fact.get("residual_load_fr_forecast_mw"),
        },
        "llm": {
            "drivers_bullish": llm_out.get("drivers_bullish", []),
            "drivers_bearish": llm_out.get("drivers_bearish", []),
            "invalidation_triggers": llm_out.get("invalidation_triggers", []),
            "commentary_text": llm_out.get("commentary_text", ""),
            "grounding_issues": commentary_cache.get("grounding_issues", []),
        },
        "footer": {
            "data_coverage": "ENTSO-E + EXAA + yfinance committed snapshot, 2019-01-01 to 2025-12-31 (DE-LU).",
            "model_performance_note": (
                f"Selected model Ridge: Test-2025 MAE {overall_ridge['mae']:.2f} EUR/MWh, "
                f"RMSE {overall_ridge['rmse']:.2f} EUR/MWh, skill vs. D-7 naive "
                f"{overall_ridge['skill_vs_d7']}. No MAPE is reported: DE-LU day-ahead "
                "prices cross zero and go negative on high-renewables hours, so MAPE is "
                "undefined at zero and explosive near it."
            ),
            "lightgbm_note": (
                f"LightGBM is a nonlinearity-check challenger, not the selected model "
                f"(Test-2025 MAE {overall_lgbm['mae']:.2f}, RMSE {overall_lgbm['rmse']:.2f} "
                "— better on raw accuracy, but a regularised linear model is kept for "
                "interpretability; see outputs/validation_metrics.md §4 for the full "
                "selection rationale)."
            ),
            "sources": (
                "Fair value: this model's own Ridge forecast (predictions.csv). EXAA: "
                "ENTSO-E DE-LU day-ahead price document, Sequence 2 (settles ~10:15 CET "
                "D-1). DE/FR load, wind, solar: ENTSO-E day-ahead forecasts. TTF: Yahoo "
                "Finance TTF=F. CO2 proxy: CARB.L (WisdomTree Carbon ETC), USD — not an "
                "official EUR/tonne EUA settlement print. The realised EPEX settlement "
                "(Sequence 1) is deliberately NOT shown — it is this model's own training "
                "target and isn't known until that auction clears (~12:00 CET D-1), after "
                "the point this page would be used to decide a bid."
            ),
            "engineered_features_note": (
                "47 engineered features feed the Ridge/LightGBM models (residual load, "
                "merit-order nonlinearity, price lags incl. 5 neighbor zones, rolling "
                "stats, gas+carbon composite, Forecast Transfer Capacity); see "
                "src/features.py and CLAUDE.md §5 for the full list and rationale."
            ),
            "exclusions_note": (
                "Generation-unit and transmission-grid outage/REMIT data is excluded by "
                "design (sparse, unstructured pre-2021 publication) — no outage panel is "
                "shown here. See CLAUDE.md §2."
            ),
        },
    }
    return data


def build_dashboard(delivery_date: Optional[object] = None) -> dict:
    """
    Public entry point: builds the data dict, writes outputs/dashboard_data.json,
    renders outputs/fair_value_dashboard.html from that same dict, and runs a
    grounding check on the rendered HTML before returning.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    data = build_dashboard_data(delivery_date)

    with open(DASHBOARD_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Dashboard data written -> {DASHBOARD_DATA_PATH}")

    html = render_dashboard_html(data)
    with open(DASHBOARD_HTML_PATH, "w") as f:
        f.write(html)
    print(f"Dashboard HTML written -> {DASHBOARD_HTML_PATH}")

    issues = _grounding_check_html(html, data)
    if issues:
        raise AssertionError(
            f"Dashboard grounding check FAILED — {len(issues)} numeric token(s) in the "
            f"rendered HTML could not be traced to dashboard_data.json or the LLM cache: "
            f"{issues[:20]}"
        )
    print(f"Dashboard grounding check: PASSED ({len(html):,} bytes rendered)")

    return data


# ---------------------------------------------------------------------------
# Grounding check — every numeric value rendered in the human-visible parts
# of the HTML must trace back to dashboard_data.json or the LLM cache.
# Scope/limits documented honestly: structural numbers that are not "facts"
# (hour-of-day axis labels 0-23, calendar years, decimal-precision digits,
# CSS pixel sizes, percentages, the inlined JSON blob itself) are allow-listed
# below rather than silently special-cased one by one in the renderer.
# ---------------------------------------------------------------------------

def _collect_numbers(obj, acc: set) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if obj is not None and not (isinstance(obj, float) and np.isnan(obj)):
            acc.add(round(float(obj), 2))
    elif isinstance(obj, str):
        # Strip thousand-separator commas first ("6,750" -> "6750"), same
        # reason as the rendered-HTML side in _grounding_check_html().
        cleaned = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", obj)
        for m in re.findall(r"-?\d+(?:\.\d+)?", cleaned):
            acc.add(round(float(m), 2))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, acc)


def _grounding_check_html(html: str, data: dict) -> list:
    known = set()
    _collect_numbers(data, known)
    # Structural / non-data numbers that legitimately appear in markup and
    # are not claims about the world: hour-of-day labels, calendar years,
    # the FLAT_CONVICTION threshold (already in `data["meta"]` but listed
    # here too for clarity), small structural counts.
    allow_list = set(range(0, 32)) | {2019, 2024, 2025, 2026, 100, 1, 2, 3, 4, 5}
    known |= {float(v) for v in allow_list}

    # Only scan the human-visible body: strip the <style> block (CSS pixel
    # sizes/opacities aren't facts) and every <script> block — both the
    # inlined dashboard_data.json (trivially self-grounded; excluding it
    # changes nothing) and the chart-init code (hex colours, line tension,
    # dash patterns are presentation config, not claims about the world).
    # The vendored Chart.js bundle itself is never inlined (loaded via a
    # separate <script src>), so it's outside this string entirely.
    visible = re.sub(r"<style.*?</style>", "", html, flags=re.DOTALL)
    visible = re.sub(r"<script.*?</script>", "", visible, flags=re.DOTALL)
    # Thousand-separator commas (e.g. "62,427") would otherwise tokenize as
    # two separate numbers ("62" and "427") under the regex below.
    visible = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", visible)

    # Tolerance, not exact match: the per-hour conviction strip deliberately
    # displays 1 decimal place (spec: "conviction_h (1 decimal)") while
    # dashboard_data.json stores it at 3 decimals, so a displayed "0.3" can
    # legitimately come from a stored 0.273 — that's rounding, not an
    # invented number. 0.05 covers any single-decimal rounding gap.
    # 0.005 covers the known-set's own 2-decimal rounding, 0.05 covers the
    # 1-decimal hour-strip display rounding from a 3-decimal stored value —
    # 0.06 gives a small safety margin over the 0.055 worst case so this
    # doesn't trip on float representation right at the boundary.
    TOLERANCE = 0.06
    known_arr = np.array(sorted(known))

    tokens = re.findall(r"-?\d+(?:\.\d+)?", visible)
    ungrounded = []
    for tok in tokens:
        num = round(float(tok), 2)
        if len(known_arr) and np.min(np.abs(known_arr - num)) <= TOLERANCE:
            continue
        if len(known_arr) and np.min(np.abs(known_arr - abs(num))) <= TOLERANCE:
            continue
        ungrounded.append(tok)
    return sorted(set(ungrounded), key=float)


# ---------------------------------------------------------------------------
# Layer 2: render_dashboard_html() — pure function, dict -> HTML string.
# Every value rendered below is read from `data`; nothing here is invented.
# ---------------------------------------------------------------------------

def _eur(v, signed=False):
    if v is None:
        return "—"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def _mw(v):
    if v is None:
        return "—"
    return f"{v:,.0f}"


def _pct1(v):
    if v is None:
        return "—"
    return f"{v:.1f}"


_DIR_CLASS = {"BUY": "buy", "SELL": "sell", "FLAT": "flat"}
_DIR_LETTER = {"BUY": "B", "SELL": "S", "FLAT": "F"}
_SHAPE_CLASS = {"RICH": "rich", "CHEAP": "cheap", "NEUTRAL": "neutral"}


def _esc(s: Optional[str]) -> str:
    return html_lib.escape(s) if s else ""


def _kpi_card(label: str, value: str, sub: str = "", cls: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card {cls}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_html}</div>'
    )


def _hour_strip_html(hourly: dict) -> str:
    cells = []
    for h, direction, conv in zip(hourly["hour"], hourly["direction"], hourly["conviction"]):
        cls = _DIR_CLASS.get(direction, "flat")
        letter = _DIR_LETTER.get(direction, "F")
        conv_s = f"{conv:.1f}" if conv is not None else "—"
        cells.append(
            f'<div class="hour-cell {cls}"><div class="hour-num">{h:02d}</div>'
            f'<div class="hour-dir">{letter}</div><div class="hour-conv">{conv_s}</div></div>'
        )
    return "".join(cells)


def _blocks_table_html(blocks: list) -> str:
    rows = []
    for b in blocks:
        cls = _DIR_CLASS.get(b["direction"], "flat")
        rows.append(
            "<tr>"
            f'<td>{_esc(b["label"])}</td>'
            f'<td>{b["n_hours"]}</td>'
            f'<td>{_eur(b["fv_avg_eur"])}</td>'
            f'<td>{_eur(b["exaa_avg_eur"])}</td>'
            f'<td>{_eur(b["basis_eur"], signed=True)}</td>'
            f'<td>{_eur(b["mae_avg_eur"])}</td>'
            f'<td>{b["conviction"]:.3f}</td>'
            f'<td><span class="pill {cls}">{b["direction"]}</span></td>'
            "</tr>"
        )
    return "".join(rows)


def _hourly_detail_table_html(hourly: dict) -> str:
    rows = []
    n = len(hourly["hour"])
    for i in range(n):
        d = hourly["direction"][i]
        shape = hourly["shape_flag"][i]
        rows.append(
            "<tr>"
            f'<td>{hourly["hour"][i]:02d}:00</td>'
            f'<td>{_eur(hourly["fv_eur"][i])}</td>'
            f'<td>{_eur(hourly["exaa_eur"][i])}</td>'
            f'<td>{_eur(hourly["basis_eur"][i], signed=True)}</td>'
            f'<td><span class="pill {_DIR_CLASS.get(d, "flat")}">{d}</span></td>'
            f'<td>{hourly["conviction"][i]:.2f}</td>'
            f'<td><span class="shape {_SHAPE_CLASS.get(shape, "neutral")}">{shape}</span></td>'
            "</tr>"
        )
    return "".join(rows)


def _list_html(items: list) -> str:
    if not items:
        return "<li><em>(none)</em></li>"
    return "".join(f"<li>{_esc(item)}</li>" for item in items)


_CSS = """
  body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 1100px;
         margin: 2rem auto; padding: 0 1.5rem; color: #1a1a1a; line-height: 1.5; background: #fcfcfc; }
  h1 { font-size: 1.7rem; margin-bottom: 0.2rem; }
  h2 { font-size: 1.2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; margin-top: 2.2rem; }
  .meta { color: #666; font-size: 0.85rem; margin-top: 0; }
  .badge { display: inline-block; padding: 0.25rem 0.7rem; border-radius: 4px; font-size: 0.78rem;
           font-weight: 700; letter-spacing: 0.03em; background: #fff3cd; color: #7a5b00;
           border: 1px solid #f0d68a; }
  .mode-note { background: #fff8e1; border-left: 3px solid #f2b705; padding: 0.5rem 1rem;
               font-size: 0.85rem; margin: 0.6rem 0 1.2rem; color: #444; }
  .kpi-row { display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 1rem 0; }
  .kpi-card { flex: 1 1 150px; background: #fff; border: 1px solid #e2e2e2; border-radius: 8px;
              padding: 0.8rem 1rem; }
  .kpi-label { font-size: 0.72rem; color: #777; text-transform: uppercase; letter-spacing: 0.04em; }
  .kpi-value { font-size: 1.45rem; font-weight: 700; margin-top: 0.15rem; }
  .kpi-sub { font-size: 0.78rem; color: #888; margin-top: 0.15rem; }
  .kpi-unit { font-size: 0.6em; }
  .kpi-card.sell .kpi-value { color: #b3261e; }
  .kpi-card.buy .kpi-value { color: #1a7a3c; }
  .kpi-card.flat .kpi-value { color: #555; }
  table { border-collapse: collapse; width: 100%; margin: 0.8rem 0; font-size: 0.88rem; }
  th, td { border: 1px solid #e2e2e2; padding: 0.4rem 0.55rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: #fafafa; font-size: 0.78rem; text-transform: uppercase; color: #666; }
  .pill { display: inline-block; padding: 0.12rem 0.55rem; border-radius: 10px; font-size: 0.78rem;
          font-weight: 700; }
  .pill.sell { background: #fdeaea; color: #b3261e; }
  .pill.buy { background: #e3f5e9; color: #1a7a3c; }
  .pill.flat { background: #eee; color: #555; }
  .shape { font-size: 0.78rem; padding: 0.1rem 0.4rem; border-radius: 4px; }
  .shape.rich { background: #fdeaea; color: #b3261e; }
  .shape.cheap { background: #e3f5e9; color: #1a7a3c; }
  .shape.neutral { color: #999; }
  .hour-strip { display: flex; flex-wrap: wrap; gap: 3px; margin: 0.8rem 0; }
  .hour-cell { width: 44px; text-align: center; border-radius: 5px; padding: 0.3rem 0.1rem;
               font-size: 0.72rem; }
  .hour-cell.sell { background: #fdeaea; color: #8c1d17; }
  .hour-cell.buy { background: #e3f5e9; color: #14602f; }
  .hour-cell.flat { background: #ececec; color: #555; }
  .hour-num { font-weight: 700; font-size: 0.74rem; }
  .hour-dir { font-weight: 700; font-size: 0.95rem; }
  .hour-conv { font-size: 0.68rem; opacity: 0.8; }
  .tally-line { font-size: 0.95rem; margin: 0.3rem 0 1rem; }
  .chart-wrap { background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; padding: 0.8rem;
                margin: 0.8rem 0; }
  canvas { max-height: 320px; }
  .driver-cols { display: flex; gap: 1.5rem; flex-wrap: wrap; }
  .driver-col { flex: 1 1 260px; }
  .driver-col h3 { font-size: 0.95rem; margin: 0.4rem 0; }
  ul { padding-left: 1.3rem; }
  blockquote { border-left: 3px solid #aaa; margin: 0.5rem 0; padding: 0.4rem 1rem; color: #333;
               background: #fafafa; }
  .footer { color: #888; font-size: 0.8rem; margin-top: 2.5rem; border-top: 1px solid #ddd;
            padding-top: 0.8rem; }
  .footer p { margin: 0.35rem 0; }
"""


def render_dashboard_html(data: dict) -> str:
    """
    Pure function: dashboard data dict -> self-contained HTML string. Every
    rendered number/string is read from `data` (built by build_dashboard_data()
    from already-committed pipeline outputs) — nothing is computed or
    hand-edited here. The full dict is also inlined as JSON so the page is
    self-contained, and Chart.js is referenced via a relative path to the
    vendored static/chart.umd.min.js asset (no CDN — works fully offline).
    """
    meta = data["meta"]
    kpi = data["kpi"]
    hourly = data["hourly"]
    tally = data["tally"]
    blocks = data["blocks"]
    drivers = data["drivers"]
    level = data["level_drivers"]
    llm = data["llm"]
    footer = data["footer"]

    baseload_cls = _DIR_CLASS.get(kpi["baseload_direction"], "flat")

    kpi_html = "".join([
        _kpi_card("Fair Value (baseload)", f'{_eur(kpi["fv_baseload_eur"])} <span class="kpi-unit">EUR/MWh</span>'),
        _kpi_card("EXAA (baseload)", f'{_eur(kpi["exaa_baseload_eur"])} <span class="kpi-unit">EUR/MWh</span>'),
        _kpi_card(
            "Baseload Basis (FV − EXAA)",
            f'{_eur(kpi["basis_baseload_eur"], signed=True)}',
            sub=f'conviction {kpi["basis_baseload_conviction"]:.3f} → <strong>{kpi["baseload_direction"]}</strong>',
            cls=baseload_cls,
        ),
        _kpi_card("Selected Model", kpi["selected_model"]),
        _kpi_card(
            "Model Accuracy (Test 2025)",
            f'MAE {_eur(kpi["model_mae_eur"])}',
            sub=f'RMSE {_eur(kpi["model_rmse_eur"])} EUR/MWh — no MAPE (prices cross zero)',
        ),
    ])

    period_avg_rows = "".join(
        "<tr>"
        f'<td>{_esc(b["label"])}</td>'
        f'<td>{_eur(b["fv_avg_eur"])}</td>'
        f'<td>{_eur(b["exaa_avg_eur"])}</td>'
        "</tr>"
        for b in blocks
    )

    tally_line = (
        f'net: <strong>{tally["sell"]} SELL</strong> / {tally["flat"]} FLAT / {tally["buy"]} BUY '
        f'(of {tally["sell"] + tally["flat"] + tally["buy"]} hours, FLAT_CONVICTION = {meta["flat_conviction"]})'
    )

    driver_table_rows = (
        "<tr><td>TTF front-month (gas)</td>"
        f'<td>{_eur(level["ttf_front_month_eur_mwh"])} EUR/MWh</td></tr>'
        "<tr><td>EU ETS CO2 proxy (CARB.L)</td>"
        f'<td>{_eur(level["co2_proxy_usd"])} USD — tradable ETC, <strong>not</strong> an official EUR/tonne EUA print</td></tr>'
        "<tr><td>Gas+CO2 pressure index</td>"
        f'<td>{level["gas_co2_pressure_index"]:.2f} (standardised, unitless — see CLAUDE.md §5)</td></tr>'
        "<tr><td>NTC net transfer capacity</td>"
        f'<td>{_mw(level["ntc_net_transfer_capacity_mw"])} MW (export − import, whichever borders publish — see CLAUDE.md §2)</td></tr>'
        "<tr><td>FR residual load forecast</td>"
        f'<td>{_mw(level["residual_load_fr_forecast_mw"])} MW</td></tr>'
    )

    json_blob = json.dumps(data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DE-LU Fair Value vs. EXAA — {meta["delivery_date"]}</title>
<style>{_CSS}</style>
</head>
<body>

<h1>DE-LU Day-Ahead Fair Value vs. EXAA</h1>
<p class="meta">Delivery day {meta["delivery_date"]} &middot; generated {meta["generated_at_utc"]} &middot;
<span class="badge">{meta["mode"]}</span></p>
<div class="mode-note">{_esc(meta["mode_note"])}</div>

<h2>Key Metrics</h2>
<div class="kpi-row">{kpi_html}</div>

<h2>Hourly Fair Value vs. EXAA</h2>
<p class="meta">Both series are point-in-time safe for a D-1 bidding decision: Fair Value is this model's own
forecast, EXAA is a real pre-auction print settling ~10:15 CET D-1. The realised EPEX settlement (this model's
training target) is deliberately not shown here — see the footer.</p>
<div class="chart-wrap"><canvas id="priceChart"></canvas></div>
<table>
<tr><th>Period</th><th>Fair Value avg</th><th>EXAA avg</th></tr>
{period_avg_rows}
</table>

<h2>Per-Hour Decision Strip</h2>
<p class="tally-line">{tally_line}</p>
<div class="hour-strip">{_hour_strip_html(hourly)}</div>
<p class="meta">Each cell: hour, direction (B=BUY / S=SELL / F=FLAT), conviction (|basis| / that hour's own Ridge backtest MAE).
Pure arithmetic — see src/dashboard.py — never an LLM judgement.</p>

<h2>Block-Level Decision</h2>
<table>
<tr><th>Block</th><th>Hours</th><th>FV avg</th><th>EXAA avg</th><th>Basis</th><th>MAE avg</th><th>Conviction</th><th>Direction</th></tr>
{_blocks_table_html(blocks)}
</table>
<p class="meta">No single whole-day call is made — the baseload row above is the closest thing to a one-glance verdict,
but the per-hour strip is the actual headline.</p>

<h2>Hourly Detail</h2>
<table>
<tr><th>Hour</th><th>Fair Value</th><th>EXAA</th><th>Basis</th><th>Direction</th><th>Conviction</th><th>Shape vs. own baseload</th></tr>
{_hourly_detail_table_html(hourly)}
</table>
<p class="meta">"Shape" flags RICH/CHEAP at &ge;{meta["rich_cheap_threshold_eur"]:.0f} EUR/MWh deviation from this day's
own fair-value baseload average — a within-day shape signal, distinct from the basis-vs-EXAA direction above.</p>

<h2>Fundamental Drivers</h2>
<div class="driver-cols">
  <div class="driver-col">
    <h3>DE: Load / Wind / Solar / Residual Load</h3>
    <div class="chart-wrap"><canvas id="deDriverChart"></canvas></div>
  </div>
  <div class="driver-col">
    <h3>FR: Load / Wind (cross-border pressure proxy)</h3>
    <div class="chart-wrap"><canvas id="frDriverChart"></canvas></div>
  </div>
</div>
<table>
<tr><th>Level driver</th><th>Value</th></tr>
{driver_table_rows}
</table>

<h2>LLM Driver Commentary</h2>
<p class="meta">Rendered verbatim from the cached LLM output — the model narrates the code-computed tally above,
it does not originate direction or conviction.</p>
<div class="driver-cols">
  <div class="driver-col">
    <h3>Bullish drivers</h3>
    <ul>{_list_html(llm["drivers_bullish"])}</ul>
  </div>
  <div class="driver-col">
    <h3>Bearish drivers</h3>
    <ul>{_list_html(llm["drivers_bearish"])}</ul>
  </div>
</div>
<h3>Invalidation triggers</h3>
<ul>{_list_html(llm["invalidation_triggers"])}</ul>
<h3>Commentary</h3>
<blockquote>{_esc(llm["commentary_text"])}</blockquote>

<div class="footer">
<p>{_esc(footer["data_coverage"])}</p>
<p>{_esc(footer["model_performance_note"])}</p>
<p>{_esc(footer["lightgbm_note"])}</p>
<p>{_esc(footer["sources"])}</p>
<p>{_esc(footer["engineered_features_note"])}</p>
<p>{_esc(footer["exclusions_note"])}</p>
<p>Single source of truth for every number on this page: <code>outputs/dashboard_data.json</code>.</p>
</div>

<script id="dashboard-data" type="application/json">{json_blob}</script>
<script src="../static/chart.umd.min.js"></script>
<script>
const DASHBOARD_DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
const H = DASHBOARD_DATA.hourly;
const D = DASHBOARD_DATA.drivers;
const hourLabels = H.hour.map(h => String(h).padStart(2, '0') + ':00');

new Chart(document.getElementById('priceChart'), {{
  type: 'line',
  data: {{
    labels: hourLabels,
    datasets: [
      {{ label: 'Fair Value (Ridge)', data: H.fv_eur, borderColor: '#2166ac', backgroundColor: 'transparent', tension: 0.15 }},
      {{ label: 'EXAA (Sequence 2)', data: H.exaa_eur, borderColor: '#b3261e', backgroundColor: 'transparent', tension: 0.15 }},
    ],
  }},
  options: {{
    responsive: true,
    scales: {{ y: {{ title: {{ display: true, text: 'EUR/MWh' }} }} }},
    plugins: {{ legend: {{ position: 'top' }} }},
  }},
}});

new Chart(document.getElementById('deDriverChart'), {{
  type: 'line',
  data: {{
    labels: hourLabels,
    datasets: [
      {{ label: 'Load', data: D.de_load_mw, borderColor: '#444444', backgroundColor: 'transparent', tension: 0.15 }},
      {{ label: 'Wind', data: D.de_wind_mw, borderColor: '#2a9d8f', backgroundColor: 'transparent', tension: 0.15 }},
      {{ label: 'Solar', data: D.de_solar_mw, borderColor: '#e9c46a', backgroundColor: 'transparent', tension: 0.15 }},
      {{ label: 'Residual load', data: D.de_residual_load_mw, borderColor: '#264653', backgroundColor: 'transparent', tension: 0.15 }},
    ],
  }},
  options: {{
    responsive: true,
    scales: {{ y: {{ title: {{ display: true, text: 'MW' }} }} }},
    plugins: {{ legend: {{ position: 'top' }} }},
  }},
}});

new Chart(document.getElementById('frDriverChart'), {{
  type: 'line',
  data: {{
    labels: hourLabels,
    datasets: [
      {{ label: 'FR Load', data: D.fr_load_mw, borderColor: '#6a4c93', backgroundColor: 'transparent', tension: 0.15 }},
      {{ label: 'FR Wind', data: D.fr_wind_mw, borderColor: '#1982c4', backgroundColor: 'transparent', tension: 0.15 }},
    ],
  }},
  options: {{
    responsive: true,
    scales: {{ y: {{ title: {{ display: true, text: 'MW' }} }} }},
    plugins: {{ legend: {{ position: 'top' }} }},
  }},
}});
</script>

</body>
</html>
"""
    return html


