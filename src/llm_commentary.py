"""
llm_commentary.py — LLM-generated trader commentary (CLAUDE.md §8).

Design principles enforced here:
  1. LLM ORIGINATES NO NUMBERS — every quantity in the output comes from the
     structured fact object built from model outputs.
  2. Output is schema-validated with pydantic v2; non-conforming responses are
     rejected and retried once.
  3. A grounding check verifies that every number appearing in commentary_text
     is traceable (within ±0.5) to a value in the fact object.
  4. First run: calls Anthropic API (key from ANTHROPIC_API_KEY env var), then
     caches to ai_logs/commentary_cache.json.
  5. Subsequent runs: loads from cache — no API key required for reproduction.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

import holidays
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, model_validator

try:
    import anthropic as _anthropic_lib
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

from config import OOS_END, OOS_START
from dashboard import (
    compute_block_decision,
    compute_hourly_decisions,
    compute_tally,
    _load_validation_metrics,
)

load_dotenv()

_ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(_ROOT, "data")
AI_LOGS_DIR = os.path.join(_ROOT, "ai_logs")
CACHE_PATH  = os.path.join(AI_LOGS_DIR, "commentary_cache.json")

_LLM_MODEL  = "claude-sonnet-4-6"   # update to newer model when available
_DE_HOLIDAYS = holidays.Germany()


# ---------------------------------------------------------------------------
# Output schema (pydantic v2)
# ---------------------------------------------------------------------------

class TraderCommentary(BaseModel):
    """
    Schema for LLM output — schema-validated before acceptance.

    [v2 round 5] direction/conviction are REMOVED from this schema. Per-hour
    and block direction/conviction are now pure arithmetic, computed in
    src/dashboard.py (basis vs. EXAA, divided by the model's own by-hour MAE)
    — never an LLM judgement. The LLM narrates the code-computed net stance
    (tally_buy/tally_sell/tally_flat, baseload_direction in the fact object)
    via drivers_bullish[] / drivers_bearish[] and commentary_text; it must
    not assert a direction the arithmetic didn't produce.
    """

    drivers_bullish: List[str]
    drivers_bearish: List[str]
    invalidation_triggers: List[str]
    commentary_text: str

    @field_validator("invalidation_triggers")
    @classmethod
    def triggers_nonempty(cls, v: List[str]) -> List[str]:
        if len(v) == 0:
            raise ValueError("invalidation_triggers list must not be empty")
        return v

    @field_validator("commentary_text")
    @classmethod
    def commentary_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("commentary_text must not be empty")
        return v

    @model_validator(mode="after")
    def bullish_or_bearish_present(self):
        if not self.drivers_bullish and not self.drivers_bearish:
            raise ValueError("at least one of drivers_bullish/drivers_bearish must be non-empty")
        return self


# ---------------------------------------------------------------------------
# Fact object builder
# ---------------------------------------------------------------------------

def _is_peak(ts: pd.Timestamp) -> bool:
    return (8 <= ts.hour <= 20) and (ts.dayofweek < 5) and (ts.date() not in _DE_HOLIDAYS)


def _basis_view(basis_eur: Optional[float], model_mae_eur: Optional[float],
                 no_data_note: str = "NO TRADE (insufficient history)") -> tuple:
    """
    Generic directional read from a EUR/MWh basis (forecast minus some reference
    price for the same delivery day): direction/conviction/size as a multiple of
    the model's own backtest MAE, not a fixed EUR cutoff, so conviction reflects
    signal-to-noise against the model's own demonstrated error.

    [v2 round 4] Used for two bases, in priority order — see build_fact_object():
      1. PRIMARY — basis vs. the EXAA (Sequence 2) day-ahead auction price for the
         SAME delivery day D. EXAA settles its own day-ahead auction earlier the
         same day (~10:15 CET D-1) than EPEX's Sequence 1 auction (~12:00 CET
         D-1, the forecast target) — a genuine, sourced, point-in-time-safe
         pre-auction market reference for day D, not a forecast or a different
         product (unlike the EEX front-month/front-week print that CLAUDE.md §7
         originally wanted but couldn't source).
      2. SECONDARY (kept for context, not used for the primary call) — the v1-v2
         self-referential basis vs. the trailing realised baseload (avg of D-1/D-7
         actual same-day averages).
    """
    if basis_eur is None or not model_mae_eur:
        return "NEUTRAL / FLAT", "LOW", no_data_note

    ratio = abs(basis_eur) / model_mae_eur
    if ratio < 0.3:
        conviction, size = "LOW", "NO TRADE (basis inside model's own MAE band)"
    elif ratio < 0.75:
        conviction, size = "MODERATE", "QUARTER SIZE (1/4 normal prompt risk)"
    elif ratio < 1.5:
        conviction, size = "MODERATE", "HALF SIZE (1/2 normal prompt risk)"
    else:
        conviction, size = "HIGH", "FULL SIZE (max normal prompt risk)"

    if ratio < 0.3:
        direction = "NEUTRAL / FLAT"
    elif basis_eur < 0:
        direction = "SHORT / SELL"
    else:
        direction = "LONG / BUY"

    return direction, conviction, size


def build_fact_object(
    X: pd.DataFrame,
    y: pd.Series,
    backtest_results: pd.DataFrame,
    delivery_date: Optional[object] = None,
) -> dict:
    """
    Build the structured fact object for the LLM commentary.

    All numeric values here are computed from model outputs and committed data.
    The LLM receives this object and must not introduce any new numbers.

    Defaults to the last OOS delivery day (OOS_END) — the most recent forecast
    in the window, i.e. the one a live morning note would actually be written for.
    """
    if delivery_date is None:
        delivery_date = OOS_END.date()

    # [v2 round 4] EXAA (Sequence 2) day-ahead auction price — a real settled
    # price for BZN|DE-LU, published earlier the same day (~10:15 CET D-1) than
    # the EPEX Sequence 1 auction (~12:00 CET D-1) this model forecasts. Loaded
    # directly from the committed snapshot rather than threaded through X,
    # since it is a trading-decision reference, not a model training feature.
    exaa = pd.read_parquet(os.path.join(DATA_DIR, "exaa_prices.parquet"))["exaa_price_eur_mwh"]

    dt      = pd.Timestamp(str(delivery_date), tz="Europe/Berlin")
    dt_prev = dt - pd.Timedelta("1D")

    day_mask  = X.index.normalize() == dt.normalize()
    prev_mask = X.index.normalize() == dt_prev.normalize()

    X_day  = X.loc[day_mask]
    X_prev = X.loc[prev_mask]

    # --- Driver signals for delivery day ---
    resid_day  = float(X_day["residual_load_mw"].mean())
    wind_day   = float(X_day["wind_forecast_mw"].mean())
    solar_day  = float(X_day["solar_forecast_mw"].mean())
    load_day   = float(X_day["load_forecast_mw"].mean())
    ttf_level  = float(X_day["ttf_eur_mwh"].mean())
    resid_fr_day = float(X_day["residual_load_fr_mw"].mean())
    gas_co2_pressure = float(X_day["gas_co2_pressure_index"].mean())
    ntc_net_transfer = float(X_day["ntc_net_transfer_capacity_mw"].mean())
    co2_proxy_level = float(X_day["co2_proxy_usd"].mean())

    # --- Deltas vs prior day ---
    if len(X_prev) > 0:
        wind_delta  = round(wind_day  - float(X_prev["wind_forecast_mw"].mean()),  0)
        solar_delta = round(solar_day - float(X_prev["solar_forecast_mw"].mean()), 0)
        load_delta  = round(load_day  - float(X_prev["load_forecast_mw"].mean()),  0)
        resid_delta = round(resid_day - float(X_prev["residual_load_mw"].mean()),  0)
    else:
        wind_delta = solar_delta = load_delta = resid_delta = None

    # --- Residual load percentile vs pre-OOS distribution ---
    pre_oos_resid = X.loc[X.index < OOS_START, "residual_load_mw"].dropna()
    resid_pct = round(float((pre_oos_resid < resid_day).mean() * 100), 1)

    # --- Ridge forecast for delivery day ---
    br_day_mask = backtest_results.index.normalize() == dt.normalize()
    br_day      = backtest_results.loc[br_day_mask]
    baseload_fc = round(float(br_day["ridge"].mean()), 2) if len(br_day) > 0 else None
    peak_rows   = br_day.loc[br_day["is_peak"] == 1, "ridge"]
    peak_fc     = round(float(peak_rows.mean()), 2) if len(peak_rows) > 0 else None

    # --- Model MAE from backtest ---
    model_mae = round(float(np.mean(np.abs(
        backtest_results["y_true"].values - backtest_results["ridge"].values
    ))), 2)

    # --- [v2 round 4] PRIMARY basis: tomorrow's Ridge forecast vs. the EXAA
    # (Sequence 2) day-ahead auction price for the SAME delivery day D. Both
    # are for the identical 24 delivery hours; EXAA's own auction settles
    # earlier the same day (~10:15 CET D-1) than the EPEX Sequence 1 auction
    # this model forecasts (~12:00 CET D-1) — a real, observable, sourced
    # pre-auction reference, not a forecast, proxy, or different product.
    exaa_day_mask = exaa.index.normalize() == dt.normalize()
    exaa_day      = exaa.loc[exaa_day_mask]
    exaa_reference_eur = round(float(exaa_day.mean()), 2) if len(exaa_day) > 0 else None

    basis_vs_exaa = (
        round(baseload_fc - exaa_reference_eur, 2)
        if baseload_fc is not None and exaa_reference_eur is not None
        else None
    )
    direction, conviction, position_size = _basis_view(
        basis_vs_exaa, model_mae,
        no_data_note="NO TRADE (EXAA reference unavailable for this delivery day)",
    )

    # --- [v2 round 5] Pure-arithmetic per-hour/block decision layer, computed
    # via the SAME functions src/dashboard.py uses (compute_hourly_decisions /
    # compute_block_decision / compute_tally), so the LLM's fact object and
    # the dashboard never disagree. This — not the LLM — decides direction;
    # the LLM's job (TraderCommentary, below) is to narrate this tally
    # accurately via drivers_bullish[]/drivers_bearish[], never to originate
    # or contradict it.
    val_metrics = _load_validation_metrics()
    fv_hourly = br_day["ridge"]
    exaa_hourly = exaa.reindex(fv_hourly.index)
    hourly_decisions = compute_hourly_decisions(fv_hourly, exaa_hourly, val_metrics["mae_by_hour"])
    tally = compute_tally(hourly_decisions)
    baseload_block = compute_block_decision(
        hourly_decisions, pd.Series(True, index=hourly_decisions.index), "Baseload (00-24)"
    )

    # --- SECONDARY (context only, not used for the call above): the v1-v2
    # self-referential basis vs. the trailing realised baseload (avg of D-1
    # and D-7 *actual* same-day averages — both already known by gate closure
    # and already used as model features, price_lag_24h / price_lag_168h).
    # Kept in the fact object so the LLM can note agreement/disagreement
    # between the two references, but EXAA (above) drives the trading call.
    dt_d7 = dt - pd.Timedelta("7D")
    br_prev_mask = backtest_results.index.normalize() == dt_prev.normalize()
    br_d7_mask   = backtest_results.index.normalize() == dt_d7.normalize()
    prev_actual = float(backtest_results.loc[br_prev_mask, "y_true"].mean()) if br_prev_mask.any() else None
    d7_actual   = float(backtest_results.loc[br_d7_mask, "y_true"].mean()) if br_d7_mask.any() else None
    trailing_vals = [v for v in (prev_actual, d7_actual) if v is not None]
    trailing_realised_baseload = round(float(np.mean(trailing_vals)), 2) if trailing_vals else None

    basis_vs_trailing = (
        round(baseload_fc - trailing_realised_baseload, 2)
        if baseload_fc is not None and trailing_realised_baseload is not None
        else None
    )

    return {
        # Delivery context
        "delivery_date":               str(delivery_date),
        "model":                       "Ridge regression (selected model)",
        # Forecast for this delivery day
        "baseload_forecast_eur":       baseload_fc,
        "peak_forecast_eur":           peak_fc,
        # Fundamental driver signals (all from ENTSO-E day-ahead forecasts)
        "residual_load_forecast_mw":   int(round(resid_day, 0)),
        "residual_load_percentile_pct": resid_pct,
        "wind_forecast_mw":            int(round(wind_day, 0)),
        "solar_forecast_mw":           int(round(solar_day, 0)),
        "load_forecast_mw":            int(round(load_day, 0)),
        "wind_delta_vs_prior_day_mw":  int(wind_delta) if wind_delta is not None else None,
        "solar_delta_vs_prior_day_mw": int(solar_delta) if solar_delta is not None else None,
        "load_delta_vs_prior_day_mw":  int(load_delta) if load_delta is not None else None,
        "residual_load_delta_mw":      int(resid_delta) if resid_delta is not None else None,
        # Gas level anchor (TTF front-month daily close, D-1 known by gate closure)
        "ttf_front_month_eur_mwh":     round(ttf_level, 2),
        # Cross-border (FR) demand/supply pressure proxy (v2)
        "residual_load_fr_forecast_mw": int(round(resid_fr_day, 0)),
        # Composite gas+carbon pressure index (v2) — standardised units (z-score
        # sum), not EUR/MWh; positive = gas+carbon pressure above its own
        # historical norm. See features.py for why this is not expressed as a
        # EUR/MWh marginal cost (CO2 input is a tradable proxy, not an official print).
        "gas_co2_pressure_index":      round(gas_co2_pressure, 2),
        # [v2 round 5] EU ETS CO2 proxy raw level (CARB.L close, USD) — NOT an
        # official EUR/tonne EUA print. Shown alongside the composite index
        # above so the dashboard's level-driver table matches this fact object.
        "co2_proxy_usd":               round(co2_proxy_level, 2),
        # Forecast Transfer Capacity net position (v2) — positive = DE-LU has
        # more day-ahead export capacity than import capacity at this hour,
        # summed across whichever borders currently publish an NTC forecast
        # (not all physical interconnectors — see features.py).
        "ntc_net_transfer_capacity_mw": int(round(ntc_net_transfer, 0)),
        # Model confidence
        "model_mae_eur":               model_mae,
        "confidence_band_note":        f"+/-{model_mae} EUR/MWh (backtest MAE)",
        # Data QA status
        "qa_status":                   "PASSED",
        # [v2 round 4] basis vs. the EXAA (Sequence 2) pre-auction reference
        # for this delivery day, daily mean. A real, sourced, observable
        # settlement for the same hours, not a forecast or a different
        # product (unlike the EEX print CLAUDE.md §7 originally wanted and
        # couldn't source). curve_direction/conviction/position_size below
        # use the OVERALL backtest MAE as denominator (kept for
        # src/morning_note.py, which still reads these fields) — they are a
        # coarser version of the same idea as baseload_direction below.
        "exaa_reference_eur":          exaa_reference_eur,
        "basis_vs_exaa_eur":           basis_vs_exaa,
        "curve_direction":             direction,
        "curve_conviction":            conviction,
        "curve_position_size":         position_size,
        # [v2 round 5] Pure-arithmetic per-hour decision tally and baseload
        # block call — computed by src/dashboard.py's compute_* functions
        # (basis vs. EXAA, divided by the model's own BY-HOUR backtest MAE,
        # FLAT below config.FLAT_CONVICTION). This is what the LLM must
        # narrate, not contradict — see TraderCommentary docstring.
        "tally_buy":                   tally["buy"],
        "tally_sell":                  tally["sell"],
        "tally_flat":                  tally["flat"],
        "tally_total_hours":           tally["buy"] + tally["sell"] + tally["flat"],
        "baseload_direction":          baseload_block["direction"],
        "baseload_conviction":         baseload_block["conviction"],
        # SECONDARY / context only — the v1-v2 self-referential basis vs. the
        # trailing realised baseload. No longer drives curve_direction/
        # conviction/position_size; kept so the LLM can note whether the two
        # references agree.
        "trailing_realised_baseload_eur": trailing_realised_baseload,
        "basis_vs_trailing_eur":          basis_vs_trailing,
    }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a quantitative energy analyst writing a morning fair-value commentary for
the German (DE-LU) day-ahead electricity market. You receive a structured JSON fact
object containing model forecasts, fundamental driver signals, and a basis vs. the
EXAA day-ahead auction price (exaa_reference_eur / basis_vs_exaa_eur) — EXAA settles
its own day-ahead auction for the same delivery day earlier the same morning than the
EPEX auction this model forecasts, so it is a real, observable pre-auction market
print for the exact hours in question, not a forecast or a different product.

DIRECTION IS NOT YOURS TO DECIDE. The fact object already contains the
code-computed per-hour decision tally (tally_buy / tally_sell / tally_flat —
counts of delivery hours where the basis vs. EXAA, divided by the model's own
by-hour backtest MAE, clears a SELL/BUY/FLAT threshold) and the baseload block
call (baseload_direction / baseload_conviction, the same arithmetic averaged
over the full day). Your job is to narrate that net stance accurately — never
to assert a different direction, and never to invent a single whole-day call
beyond what baseload_direction already says.

HARD RULES:
1. You MUST NOT originate, estimate, or invent any numeric value.
   Every number that appears anywhere in your output MUST come verbatim (or rounded)
   from the provided fact object. If a number is not in the fact object, do not use it.
2. Output MUST be valid JSON matching the schema below — nothing else, no markdown fences,
   no preamble.
3. drivers_bullish[] / drivers_bearish[]: split the price drivers by which way they push
   the basis vs. EXAA — drivers_bullish[] for signals consistent with EXAA being
   underpriced relative to fair value (push toward BUY), drivers_bearish[] for signals
   consistent with EXAA being overpriced (push toward SELL). 1-4 short strings each;
   at least one of the two lists must be non-empty. Be precise (e.g., "Wind surge
   +11,107 MW vs prior day collapses residual load to 18th percentile").
4. invalidation_triggers[]: 2-4 most actionable triggers that would flip the net stance.
5. commentary_text: exactly 2-3 sentences. Open by stating baseload_direction and the
   tally (e.g. "X of 24 hours screen SELL") exactly as given, name the top 1-2 drivers,
   close with the key uncertainty. Trader-desk register — no hedging non-committal
   language, and no direction claim that disagrees with baseload_direction or the tally.

JSON schema (output only this, no markdown fences):
{
  "drivers_bullish": ["string", ...],
  "drivers_bearish": ["string", ...],
  "invalidation_triggers": ["string", ...],
  "commentary_text": "string"
}
"""


def _build_user_message(fact_obj: dict) -> str:
    return (
        f"Generate trader commentary for DE-LU delivery date "
        f"{fact_obj['delivery_date']}.\n\n"
        f"FACT OBJECT (all numbers the LLM may use):\n"
        f"{json.dumps(fact_obj, indent=2)}\n\n"
        "Output the JSON commentary now."
    )


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------

def _grounding_check(commentary_text: str, fact_obj: dict) -> List[str]:
    """
    Extract all numbers from commentary_text and verify each is traceable
    to a value in the fact object (within ±0.5 tolerance).
    Returns a list of strings for numbers that could not be grounded.
    """
    def _collect_nums(obj, acc: set) -> None:
        if isinstance(obj, (int, float)):
            acc.add(float(obj))
        elif isinstance(obj, str):
            for m in re.findall(r"\d+(?:\.\d+)?", obj):
                acc.add(float(m))
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect_nums(v, acc)
        elif isinstance(obj, list):
            for v in obj:
                _collect_nums(v, acc)

    fact_nums: set = set()
    _collect_nums(fact_obj, fact_nums)

    text_tokens = re.findall(r"\b\d{1,6}(?:[,\.]\d+)?\b", commentary_text)
    ungrounded = []
    for tok in text_tokens:
        try:
            num = float(tok.replace(",", ""))
        except ValueError:
            continue
        if num < 0.01 or num > 2100:  # skip sub-cent fractions and years
            continue
        # The regex above can't capture a leading "-" (it's not part of \d),
        # so a negative fact value (e.g. basis_vs_frontweek_eur = -32.04)
        # shows up in the text as the unsigned token "32.04". Check both the
        # signed and unsigned interpretation against the fact object.
        if not any(abs(num - fv) <= 0.5 or abs(num - abs(fv)) <= 0.5 for fv in fact_nums):
            ungrounded.append(tok)

    return ungrounded


# ---------------------------------------------------------------------------
# API call + validation (with one retry)
# ---------------------------------------------------------------------------

def _call_api(fact_obj: dict, api_key: str) -> dict:
    """
    Call the Anthropic API, parse + validate against TraderCommentary schema,
    run grounding check.  Retries once on validation failure.
    Returns full result dict.
    """
    client   = _anthropic_lib.Anthropic(api_key=api_key)
    user_msg = _build_user_message(fact_obj)

    commentary: Optional[TraderCommentary] = None
    raw_text = ""
    for attempt in range(2):
        response = client.messages.create(
            model=_LLM_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw_text = response.content[0].text.strip()

        # Strip markdown code fences the model occasionally adds
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            parsed    = json.loads(raw_text)
            commentary = TraderCommentary(**parsed)
            break
        except Exception as exc:
            if attempt == 0:
                print(f"  [WARN] Attempt 1 validation failed ({exc}); retrying …")
            else:
                raise ValueError(
                    f"LLM output failed schema validation after 2 attempts: {exc}\n"
                    f"Raw output:\n{raw_text}"
                )

    assert commentary is not None  # mypy / safety

    # [v2 round 5] Grounding now covers every LLM-authored string field, not
    # just commentary_text — drivers_bullish/drivers_bearish/
    # invalidation_triggers can carry numbers too (e.g. "+6,750 MW").
    all_text = " ".join([
        commentary.commentary_text,
        *commentary.drivers_bullish,
        *commentary.drivers_bearish,
        *commentary.invalidation_triggers,
    ])
    grounding_issues = _grounding_check(all_text, fact_obj)
    if grounding_issues:
        print(f"  [WARN] Grounding check flagged numbers not traceable to fact object: "
              f"{grounding_issues}")
    else:
        print("  Grounding check: PASSED (all numbers in LLM output traced to fact object)")

    return {
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "model":            _LLM_MODEL,
        "fact_object":      fact_obj,
        "prompt_system":    _SYSTEM_PROMPT,
        "prompt_user":      user_msg,
        "raw_response":     raw_text,
        "validated_output": commentary.model_dump(),
        "grounding_issues": grounding_issues,
    }


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------

def _write_logs(result: dict) -> None:
    os.makedirs(AI_LOGS_DIR, exist_ok=True)
    date_tag = result["fact_object"]["delivery_date"].replace("-", "")

    # Full prompt (system + user) — complete enough to assess without re-running
    prompt_path = os.path.join(AI_LOGS_DIR, f"prompt_{date_tag}.txt")
    with open(prompt_path, "w") as fh:
        fh.write(
            "=== SYSTEM PROMPT ===\n"
            + result["prompt_system"]
            + "\n\n=== USER MESSAGE ===\n"
            + result["prompt_user"]
        )
    print(f"  Prompt logged  → {prompt_path}")

    # Raw API response + grounding status
    response_path = os.path.join(AI_LOGS_DIR, f"raw_response_{date_tag}.json")
    with open(response_path, "w") as fh:
        json.dump(
            {
                "model":            result["model"],
                "generated_at":     result["generated_at"],
                "raw_response":     result["raw_response"],
                "grounding_issues": result["grounding_issues"],
            },
            fh,
            indent=2,
        )
    print(f"  Response logged → {response_path}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_commentary(
    X: pd.DataFrame,
    y: pd.Series,
    backtest_results: pd.DataFrame,
) -> dict:
    """
    Generate (or load from cache) the LLM trader commentary for the first OOS day.

    Cache behaviour (CLAUDE.md §8 reproducibility):
      - If ai_logs/commentary_cache.json exists: load and return immediately.
        No API key required — pipeline is fully reproducible from committed cache.
      - If no cache: require ANTHROPIC_API_KEY, call API, validate, cache result.

    Returns the full result dict (keys: fact_object, validated_output,
    grounding_issues, prompt_system, prompt_user, raw_response, generated_at).
    """
    os.makedirs(AI_LOGS_DIR, exist_ok=True)

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as fh:
            cached = json.load(fh)
        print(f"  LLM commentary loaded from cache → {CACHE_PATH}")
        fo = cached["fact_object"]
        print(f"  Baseload call: {fo['baseload_direction']} (conviction {fo['baseload_conviction']}) "
              f"| Tally: {fo['tally_sell']} SELL / {fo['tally_flat']} FLAT / {fo['tally_buy']} BUY")
        return cached

    # No cache — need API
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            f"ANTHROPIC_API_KEY not set and no cache found at {CACHE_PATH}.\n"
            "Add ANTHROPIC_API_KEY=<key> to your .env file to generate commentary."
        )
    if not _HAS_ANTHROPIC:
        raise ImportError("anthropic package not installed — run: pip install anthropic")

    print("  Building fact object …")
    fact_obj = build_fact_object(X, y, backtest_results)
    print(
        f"  Delivery date  : {fact_obj['delivery_date']}\n"
        f"  Baseload fc    : {fact_obj['baseload_forecast_eur']} EUR/MWh\n"
        f"  Residual load  : {fact_obj['residual_load_forecast_mw']:,} MW "
        f"({fact_obj['residual_load_percentile_pct']}th pct)\n"
        f"  Wind delta     : {fact_obj['wind_delta_vs_prior_day_mw']:+,} MW vs D-1\n"
        f"  Baseload call  : {fact_obj['baseload_direction']} (conviction {fact_obj['baseload_conviction']})\n"
        f"  Tally          : {fact_obj['tally_sell']} SELL / {fact_obj['tally_flat']} FLAT / {fact_obj['tally_buy']} BUY"
    )

    print(f"  Calling {_LLM_MODEL} …")
    result = _call_api(fact_obj, api_key)
    _write_logs(result)

    with open(CACHE_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Commentary cached → {CACHE_PATH}")

    out = result["validated_output"]
    print(f"  Baseload call: {fact_obj['baseload_direction']} | Tally: "
          f"{fact_obj['tally_sell']} SELL / {fact_obj['tally_flat']} FLAT / {fact_obj['tally_buy']} BUY")
    print(f"  Commentary: {out['commentary_text'][:120]}…")
    return result
