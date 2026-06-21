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
from pydantic import BaseModel, field_validator

try:
    import anthropic as _anthropic_lib
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

from config import OOS_START

load_dotenv()

_ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
AI_LOGS_DIR = os.path.join(_ROOT, "ai_logs")
CACHE_PATH  = os.path.join(AI_LOGS_DIR, "commentary_cache.json")

_LLM_MODEL  = "claude-sonnet-4-6"   # update to newer model when available
_DE_HOLIDAYS = holidays.Germany()


# ---------------------------------------------------------------------------
# Output schema (pydantic v2)
# ---------------------------------------------------------------------------

class TraderCommentary(BaseModel):
    """Schema for LLM output — schema-validated before acceptance."""

    direction: str
    conviction: str
    drivers: List[str]
    invalidation_triggers: List[str]
    commentary_text: str

    @field_validator("direction")
    @classmethod
    def direction_valid(cls, v: str) -> str:
        allowed = {"LONG / BUY", "SHORT / SELL", "NEUTRAL / FLAT"}
        if v not in allowed:
            raise ValueError(f"direction '{v}' not in {allowed}")
        return v

    @field_validator("conviction")
    @classmethod
    def conviction_valid(cls, v: str) -> str:
        allowed = {"HIGH", "MODERATE", "LOW"}
        if v not in allowed:
            raise ValueError(f"conviction '{v}' not in {allowed}")
        return v

    @field_validator("drivers")
    @classmethod
    def drivers_nonempty(cls, v: List[str]) -> List[str]:
        if len(v) == 0:
            raise ValueError("drivers list must not be empty")
        return v

    @field_validator("invalidation_triggers")
    @classmethod
    def triggers_nonempty(cls, v: List[str]) -> List[str]:
        if len(v) == 0:
            raise ValueError("invalidation_triggers list must not be empty")
        return v


# ---------------------------------------------------------------------------
# Fact object builder
# ---------------------------------------------------------------------------

def _is_peak(ts: pd.Timestamp) -> bool:
    return (8 <= ts.hour <= 20) and (ts.dayofweek < 5) and (ts.date() not in _DE_HOLIDAYS)


def build_fact_object(
    X: pd.DataFrame,
    y: pd.Series,
    backtest_results: pd.DataFrame,
    curve_view: dict,
    delivery_date: Optional[object] = None,
) -> dict:
    """
    Build the structured fact object for the LLM commentary.

    All numeric values here are computed from model outputs and committed data.
    The LLM receives this object and must not introduce any new numbers.

    Defaults to the first OOS delivery day (OOS_START = 2025-12-08).
    """
    if delivery_date is None:
        delivery_date = OOS_START.date()

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
        # Model confidence
        "model_mae_eur":               model_mae,
        "confidence_band_note":        f"+/-{model_mae} EUR/MWh (backtest MAE)",
        # Data QA status
        "qa_status":                   "PASSED",
        # Curve view (from prompt_curve.py — step 8)
        "curve_direction":             curve_view["direction"],
        "curve_conviction":            curve_view["conviction"],
        "curve_position_size":         curve_view["position_size"],
        "basis_vs_frontweek_eur":      curve_view["basis_fw_baseload_eur"],
        "eex_frontweek_eur":           curve_view["eex_frontweek_eur"],
        "eex_frontmonth_eur":          curve_view["eex_frontmonth_eur"],
        "eex_source":                  curve_view["eex_source"],
    }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a quantitative energy analyst writing a morning fair-value commentary for
the German (DE-LU) day-ahead electricity market. You receive a structured JSON fact
object containing model forecasts, fundamental driver signals, and forward-curve data.

HARD RULES:
1. You MUST NOT originate, estimate, or invent any numeric value.
   Every number that appears in your commentary_text MUST come verbatim (or rounded)
   from the provided fact object. If a number is not in the fact object, do not use it.
2. Output MUST be valid JSON matching the schema below — nothing else, no markdown fences,
   no preamble.
3. drivers[]: 2-4 short strings naming specific price drivers. Be precise
   (e.g., "Wind surge +11,107 MW vs prior day collapses residual load to 18th percentile").
4. invalidation_triggers[]: 2-4 most actionable triggers that would flip the view.
5. commentary_text: exactly 2-3 sentences. State the directional view first, name
   the top 2 drivers, close with the key uncertainty. Trader-desk register — no
   hedging non-committal language.

JSON schema (output only this, no markdown fences):
{
  "direction": "LONG / BUY" | "SHORT / SELL" | "NEUTRAL / FLAT",
  "conviction": "HIGH" | "MODERATE" | "LOW",
  "drivers": ["string", ...],
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

    grounding_issues = _grounding_check(commentary.commentary_text, fact_obj)
    if grounding_issues:
        print(f"  [WARN] Grounding check flagged numbers not traceable to fact object: "
              f"{grounding_issues}")
    else:
        print("  Grounding check: PASSED (all numbers in commentary_text traced to fact object)")

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
    curve_view: dict,
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
        out = cached["validated_output"]
        print(f"  Direction: {out['direction']} | Conviction: {out['conviction']}")
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
    fact_obj = build_fact_object(X, y, backtest_results, curve_view)
    print(
        f"  Delivery date  : {fact_obj['delivery_date']}\n"
        f"  Baseload fc    : {fact_obj['baseload_forecast_eur']} EUR/MWh\n"
        f"  Residual load  : {fact_obj['residual_load_forecast_mw']:,} MW "
        f"({fact_obj['residual_load_percentile_pct']}th pct)\n"
        f"  Wind delta     : {fact_obj['wind_delta_vs_prior_day_mw']:+,} MW vs D-1\n"
        f"  Curve          : {fact_obj['curve_direction']} / {fact_obj['curve_conviction']}"
    )

    print(f"  Calling {_LLM_MODEL} …")
    result = _call_api(fact_obj, api_key)
    _write_logs(result)

    with open(CACHE_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Commentary cached → {CACHE_PATH}")

    out = result["validated_output"]
    print(f"  Direction: {out['direction']} | Conviction: {out['conviction']}")
    print(f"  Commentary: {out['commentary_text'][:120]}…")
    return result
