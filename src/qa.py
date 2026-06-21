"""
qa.py — Data quality checks for the DE-LU pipeline.

Checks:
  1. Duplicate timestamps
  2. Continuous hourly index — gap detection and classification
  3. DST validation — 23h spring / 25h fall days
  4. Range / sanity checks (negatives explicitly preserved for prices)
  5. Point-in-time firewall spot-check vs raw CSV
  6. Short-gap imputation (linear interpolation, ≤ SHORT_GAP_MAX_H hours)
  7. Missingness summary per series

Writes: outputs/qa_report.md

Returns: dict of clean (imputed) series for use by downstream modules.
"""

import glob
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(_ROOT, "data")
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")

SHORT_GAP_MAX_H = 3  # gaps ≤ this are imputed; longer ones are flagged only

# Hard bounds — note prices can be NEGATIVE (CLAUDE.md §3.2)
BOUNDS = {
    "price_eur_mwh":     (-600.0,  5_000.0),
    "load_forecast_mw":  (10_000.0, 120_000.0),
    "wind_forecast_mw":  (0.0,      130_000.0),
    "solar_forecast_mw": (0.0,       80_000.0),
}

# DST transition dates in Europe/Berlin (last Sunday of March / October)
SPRING_FORWARD = [
    "2019-03-31", "2020-03-29", "2021-03-28", "2022-03-27",
    "2023-03-26", "2024-03-31", "2025-03-30",
]
FALL_BACK = [
    "2019-10-27", "2020-10-25", "2021-10-31", "2022-10-30",
    "2023-10-29", "2024-10-27", "2025-10-26",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expected_index(series: pd.Series) -> pd.DatetimeIndex:
    """
    Build the complete expected tz-aware hourly index between the series'
    first and last timestamps. pd.date_range with tz='Europe/Berlin' handles
    DST naturally: spring-forward days have 23 entries, fall-back days 25.
    """
    return pd.date_range(
        start=series.index[0],
        end=series.index[-1],
        freq="h",
        tz="Europe/Berlin",
    )


def _detect_gaps(series: pd.Series) -> pd.DataFrame:
    """
    Reindex to expected hourly index, find NaN runs, return a DataFrame with
    columns: start, end, length_h, kind ('short' | 'long').
    """
    expected = _expected_index(series)
    reindexed = series.reindex(expected)
    is_nan = reindexed.isna()

    gaps = []
    in_gap = False
    gap_start = None
    for ts, nan in zip(reindexed.index, is_nan):
        if nan and not in_gap:
            in_gap = True
            gap_start = ts
        elif not nan and in_gap:
            length = int((ts - gap_start) / pd.Timedelta("1h"))
            gaps.append({
                "start": gap_start,
                "end": ts - pd.Timedelta("1h"),
                "length_h": length,
                "kind": "short" if length <= SHORT_GAP_MAX_H else "long",
            })
            in_gap = False
    if in_gap:
        length = int((reindexed.index[-1] - gap_start) / pd.Timedelta("1h")) + 1
        gaps.append({
            "start": gap_start,
            "end": reindexed.index[-1],
            "length_h": length,
            "kind": "short" if length <= SHORT_GAP_MAX_H else "long",
        })

    return pd.DataFrame(gaps, columns=["start", "end", "length_h", "kind"])


def _impute_short_gaps(series: pd.Series) -> pd.Series:
    """
    Reindex to expected hourly index, then linearly interpolate short gaps
    (≤ SHORT_GAP_MAX_H hours).  Long gaps are left as NaN so downstream code
    can see them explicitly.
    """
    expected = _expected_index(series)
    out = series.reindex(expected)

    # Mark which positions were originally NaN
    originally_nan = out.isna()

    # Interpolate everything, then restore NaN for long-gap positions
    interpolated = out.interpolate(method="time")

    # Identify long-gap positions to NOT impute
    gap_df = _detect_gaps(series)
    long_gap_mask = pd.Series(False, index=out.index)
    for _, row in gap_df[gap_df["kind"] == "long"].iterrows():
        long_gap_mask.loc[row["start"]:row["end"]] = True

    out = interpolated.where(~(originally_nan & long_gap_mask), other=np.nan)
    return out


def _check_dst(series: pd.Series) -> list[dict]:
    """
    For each known DST transition date, verify the correct number of hours:
    23 for spring-forward, 25 for fall-back.
    """
    results = []
    for date_str, expected_h in [(d, 23) for d in SPRING_FORWARD] + \
                                [(d, 25) for d in FALL_BACK]:
        date = pd.Timestamp(date_str, tz="Europe/Berlin")
        if date < series.index[0].normalize() or date > series.index[-1].normalize():
            continue  # outside our data range
        day_data = series[date_str]
        actual_h = len(day_data)
        results.append({
            "date": date_str,
            "transition": "spring (→23h)" if expected_h == 23 else "fall (→25h)",
            "expected_h": expected_h,
            "actual_h": actual_h,
            "pass": actual_h == expected_h,
        })
    return results


def _check_range(series: pd.Series, name: str) -> dict:
    """Check that all values fall within the expected bounds."""
    lo, hi = BOUNDS[name]
    below = (series < lo).sum()
    above = (series > hi).sum()
    return {
        "name": name,
        "min": float(series.min()),
        "max": float(series.max()),
        "below_bound": int(below),
        "above_bound": int(above),
        "bound_lo": lo,
        "bound_hi": hi,
        "pass": below == 0 and above == 0,
    }


def _firewall_spot_check() -> dict:
    """
    Confirm that our loaded load_forecast values match the 'Day-ahead Total
    Load Forecast (MW)' column in a raw CSV, NOT the 'Actual Total Load (MW)'
    column.  Uses the first year's file as a representative sample.
    """
    pattern = os.path.join(DATA_DIR, "GUI_TOTAL_LOAD_DAYAHEAD_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return {"pass": None, "note": "No raw load CSV found — skipped"}

    raw = pd.read_csv(files[0])
    raw = raw[raw["Area"] == "BZN|DE-LU"].copy()
    mtu_start = raw["MTU (UTC)"].str.split(" - ").str[0].str.strip()
    raw.index = pd.to_datetime(mtu_start, dayfirst=True, utc=True).dt.tz_convert("Europe/Berlin")

    # Resample to 1h mean — same as fetch pipeline
    forecast_raw = pd.to_numeric(raw["Day-ahead Total Load Forecast (MW)"], errors="coerce")
    actual_raw   = pd.to_numeric(raw["Actual Total Load (MW)"], errors="coerce")
    forecast_h = forecast_raw.resample("1h").mean()
    actual_h   = actual_raw.resample("1h").mean()

    loaded = pd.read_parquet(os.path.join(DATA_DIR, "load_forecast.parquet"))["load_forecast_mw"]

    # Align on common index for comparison
    common = forecast_h.index.intersection(loaded.index).intersection(actual_h.index)
    common = common[:100]  # first 100 hours is sufficient

    diff_vs_forecast = (loaded.loc[common] - forecast_h.loc[common]).abs().max()
    diff_vs_actual   = (loaded.loc[common] - actual_h.loc[common]).abs().max()

    matches_forecast = diff_vs_forecast < 1e-3
    matches_actual   = diff_vs_actual < 1e-3

    return {
        "pass": bool(matches_forecast and not matches_actual),
        "max_abs_diff_vs_forecast_mw": float(diff_vs_forecast),
        "max_abs_diff_vs_actual_mw":   float(diff_vs_actual),
        "note": (
            "Loaded values match day-ahead forecast column (not actuals). PIT firewall OK. ✓"
            if matches_forecast and not matches_actual
            else "WARNING: loaded values may not match forecast column — check fetch_data.py"
        ),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_qa() -> dict:
    """
    Run all QA checks. Writes outputs/qa_report.md.

    Returns
    -------
    dict with keys:
        'prices', 'load', 'wind_forecast_mw', 'solar_forecast_mw' — imputed pd.Series
        'ttf'   — raw daily TTF Series (no imputation; gaps are exchange holidays)
        'passed' — True if no FAIL-level issues
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    lines: list[str] = []
    all_pass = True

    def h(text: str):
        lines.append(f"\n## {text}\n")

    def row(text: str):
        lines.append(text)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    prices = pd.read_parquet(os.path.join(DATA_DIR, "da_prices.parquet"))["price_eur_mwh"]
    load   = pd.read_parquet(os.path.join(DATA_DIR, "load_forecast.parquet"))["load_forecast_mw"]
    ws     = pd.read_parquet(os.path.join(DATA_DIR, "wind_solar_forecast.parquet"))
    wind   = ws["wind_forecast_mw"]
    solar  = ws["solar_forecast_mw"]
    ttf    = pd.read_parquet(os.path.join(DATA_DIR, "ttf_daily.parquet"))["ttf_eur_mwh"]

    lines.append("# QA Report — DE-LU Day-Ahead Price Forecasting Data")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("\n---\n")

    # ------------------------------------------------------------------
    # 1. Data overview
    # ------------------------------------------------------------------
    h("1. Data Overview")
    row("| Series | Rows | Start | End | NaN (raw) |")
    row("|--------|-----:|-------|-----|----------:|")
    for name, s in [("prices", prices), ("load", load), ("wind", wind), ("solar", solar)]:
        row(f"| {name} | {len(s):,} | {s.index[0]} | {s.index[-1]} | {s.isna().sum()} |")
    row(f"| ttf (daily) | {len(ttf):,} | {ttf.index[0].date()} | {ttf.index[-1].date()} | {ttf.isna().sum()} |")

    # ------------------------------------------------------------------
    # 2. Duplicate timestamps
    # ------------------------------------------------------------------
    h("2. Duplicate Timestamps")
    dup_results = {}
    for name, s in [("prices", prices), ("load", load), ("wind", wind), ("solar", solar)]:
        n_dups = s.index.duplicated().sum()
        dup_results[name] = n_dups
        status = "PASS" if n_dups == 0 else "FAIL"
        if n_dups > 0:
            all_pass = False
        row(f"- **{name}**: {n_dups} duplicates → **{status}**")

    # ------------------------------------------------------------------
    # 3. Gap detection & imputation
    # ------------------------------------------------------------------
    h("3. Gap Detection & Imputation")
    row(f"Short-gap threshold: ≤ {SHORT_GAP_MAX_H} hours → linear interpolation.  "
        "Long gaps → flagged, left as NaN.\n")

    imputed = {}
    for name, s in [("prices", prices), ("load", load), ("wind", wind), ("solar", solar)]:
        gap_df = _detect_gaps(s)
        n_short = (gap_df["kind"] == "short").sum() if len(gap_df) else 0
        n_long  = (gap_df["kind"] == "long").sum()  if len(gap_df) else 0
        imputed[name] = _impute_short_gaps(s)

        row(f"### {name}")
        if len(gap_df) == 0:
            row("No gaps detected. **PASS**\n")
        else:
            row(f"| Start | End | Length (h) | Kind |")
            row(f"|-------|-----|:----------:|------|")
            for _, g in gap_df.iterrows():
                row(f"| {g['start']} | {g['end']} | {g['length_h']} | {g['kind']} |")
            row(f"\nShort gaps imputed: {n_short}  |  Long gaps flagged: {n_long}")
            if n_long > 0:
                all_pass = False
                row(f"\n> **WARNING**: {n_long} long gap(s) in `{name}` left as NaN.  "
                    "Downstream features must handle these rows.")
            row("")

    # ------------------------------------------------------------------
    # 4. DST validation
    # ------------------------------------------------------------------
    h("4. DST Validation (23h spring / 25h fall)")
    row("| Date | Transition | Expected h | Actual h | Status |")
    row("|------|-----------|:----------:|:--------:|--------|")
    dst_results = _check_dst(prices)  # check on prices; same index as all hourly series
    for r in dst_results:
        status = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            all_pass = False
        row(f"| {r['date']} | {r['transition']} | {r['expected_h']} | {r['actual_h']} | **{status}** |")

    # ------------------------------------------------------------------
    # 5. Range / sanity checks
    # ------------------------------------------------------------------
    h("5. Range & Sanity Checks")
    row("> Negative prices are **valid** in DE-LU (high-renewables hours) and are preserved.  "
        "The price lower bound is the ENTSO-E floor (−500 EUR/MWh) with margin.\n")
    row("| Series | Min | Max | Bound lo | Bound hi | Out-of-bound | Status |")
    row("|--------|----:|----:|---------:|---------:|:------------:|--------|")
    for name, s in [
        ("price_eur_mwh", prices),
        ("load_forecast_mw", load),
        ("wind_forecast_mw", wind),
        ("solar_forecast_mw", solar),
    ]:
        r = _check_range(s.dropna(), name)
        status = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            all_pass = False
        oob = r["below_bound"] + r["above_bound"]
        row(
            f"| {name} | {r['min']:.1f} | {r['max']:.1f} | "
            f"{r['bound_lo']:.0f} | {r['bound_hi']:.0f} | {oob} | **{status}** |"
        )

    neg_hours = int((prices < 0).sum())
    row(f"\nNegative price hours in full history: **{neg_hours:,}** "
        f"({100 * neg_hours / len(prices.dropna()):.1f}% of non-NaN rows) — preserved ✓")

    # ------------------------------------------------------------------
    # 6. Point-in-time firewall verification
    # ------------------------------------------------------------------
    h("6. Point-in-Time Firewall Verification")
    row("Confirms that loaded `load_forecast_mw` matches the  "
        "`Day-ahead Total Load Forecast (MW)` column in the raw CSV,  "
        "**not** the `Actual Total Load (MW)` column.\n")
    fw = _firewall_spot_check()
    if fw["pass"] is None:
        row(f"- {fw['note']}")
    else:
        status = "PASS" if fw["pass"] else "FAIL"
        if not fw["pass"]:
            all_pass = False
        row(f"- Max |loaded − forecast| across first 100h: **{fw['max_abs_diff_vs_forecast_mw']:.4f} MW**")
        row(f"- Max |loaded − actual|  across first 100h: **{fw['max_abs_diff_vs_actual_mw']:.1f} MW**")
        row(f"- {fw['note']}")
        row(f"\n**{status}**")

    # ------------------------------------------------------------------
    # 7. Post-imputation missingness summary
    # ------------------------------------------------------------------
    h("7. Post-Imputation Missingness Summary")
    row("| Series | NaN before | NaN after | Δ imputed |")
    row("|--------|:----------:|:---------:|:---------:|")
    for name, s_raw, s_imp in [
        ("prices",  prices, imputed["prices"]),
        ("load",    load,   imputed["load"]),
        ("wind",    wind,   imputed["wind"]),
        ("solar",   solar,  imputed["solar"]),
    ]:
        before = s_raw.isna().sum()
        after  = s_imp.isna().sum()
        row(f"| {name} | {before} | {after} | {before - after} |")

    # ------------------------------------------------------------------
    # 8. Overall verdict
    # ------------------------------------------------------------------
    h("8. Overall Verdict")
    verdict = "ALL CHECKS PASSED" if all_pass else "ONE OR MORE WARNINGS — see sections above"
    row(f"**{verdict}**\n")
    row(
        "Long gaps in `load` (2022-02-22, 2022-03-24 — full days) are a known ENTSO-E "
        "data absence. These 48 rows remain NaN and will be excluded during feature "
        "engineering via a validity mask."
    )

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    report_path = os.path.join(OUTPUTS_DIR, "qa_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"QA report written → {report_path}")

    return {
        "prices":          imputed["prices"],
        "load":            imputed["load"],
        "wind_forecast_mw": imputed["wind"],
        "solar_forecast_mw": imputed["solar"],
        "ttf":             ttf,
        "passed":          all_pass,
    }


if __name__ == "__main__":
    result = run_qa()
    print(f"\nOverall pass: {result['passed']}")
    print(f"Remaining NaN after imputation:")
    for k in ("prices", "load", "wind_forecast_mw", "solar_forecast_mw"):
        print(f"  {k}: {result[k].isna().sum()}")
