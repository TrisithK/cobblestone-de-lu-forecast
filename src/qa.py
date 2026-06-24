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
  8. [v2] Forecast Transfer Capacity (NTC) coverage per border
  9. [v2 round 4] EXAA (Sequence 2) pre-auction reference — range/dup/gap checks,
     same treatment as the Sequence 1 target series

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
    "price_eur_mwh":      (-600.0,  5_000.0),
    "exaa_price_eur_mwh": (-600.0,  5_000.0),  # [v2 round 4] EXAA auction (Sequence 2)
    "load_forecast_mw":   (10_000.0, 120_000.0),
    "wind_forecast_mw":   (0.0,      130_000.0),
    "solar_forecast_mw":  (0.0,       80_000.0),
    "load_forecast_fr_mw": (15_000.0, 105_000.0),
    "wind_forecast_fr_mw": (0.0,       35_000.0),
    # [v2 round 3] Neighbor bidding zones (FR/NL/BE/PL/CZ)
    "price_fr_eur_mwh":     (-600.0,  3_500.0),
    "price_nl_eur_mwh":     (-600.0,  1_000.0),
    "price_be_eur_mwh":     (-600.0,  1_000.0),
    "price_pl_eur_mwh":     (-600.0,  1_000.0),
    "price_cz_eur_mwh":     (-600.0,  1_000.0),
    "load_forecast_nl_mw":  (3_000.0,  25_000.0),
    "load_forecast_be_mw":  (5_000.0,  16_000.0),
    "load_forecast_pl_mw":  (8_000.0,  30_000.0),
    "load_forecast_cz_mw":  (3_000.0,  13_000.0),
    "wind_forecast_nl_mw":  (0.0,      10_000.0),
    "wind_forecast_be_mw":  (0.0,       7_000.0),
    "wind_forecast_pl_mw":  (0.0,      11_000.0),
    "solar_forecast_nl_mw": (0.0,      10_000.0),
    "solar_forecast_be_mw": (0.0,      10_000.0),
    "solar_forecast_pl_mw": (0.0,      16_000.0),
    "solar_forecast_cz_mw": (0.0,       4_000.0),
    "solar_forecast_fr_mw": (0.0,      22_000.0),
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
    exaa   = pd.read_parquet(os.path.join(DATA_DIR, "exaa_prices.parquet"))["exaa_price_eur_mwh"]
    load   = pd.read_parquet(os.path.join(DATA_DIR, "load_forecast.parquet"))["load_forecast_mw"]
    ws     = pd.read_parquet(os.path.join(DATA_DIR, "wind_solar_forecast.parquet"))
    wind   = ws["wind_forecast_mw"]
    solar  = ws["solar_forecast_mw"]
    ttf    = pd.read_parquet(os.path.join(DATA_DIR, "ttf_daily.parquet"))["ttf_eur_mwh"]
    fr_load = pd.read_parquet(os.path.join(DATA_DIR, "fr_load_forecast.parquet"))["load_forecast_fr_mw"]
    fr_wind = pd.read_parquet(os.path.join(DATA_DIR, "fr_wind_forecast.parquet"))["wind_forecast_fr_mw"]
    co2     = pd.read_parquet(os.path.join(DATA_DIR, "co2_proxy_daily.parquet"))["co2_proxy_usd"]
    ntc     = pd.read_parquet(os.path.join(DATA_DIR, "ntc_forecast.parquet"))
    neighbor_prices = pd.read_parquet(os.path.join(DATA_DIR, "neighbor_prices.parquet"))
    neighbor_load   = pd.read_parquet(os.path.join(DATA_DIR, "neighbor_load_forecast.parquet"))
    neighbor_ws     = pd.read_parquet(os.path.join(DATA_DIR, "neighbor_wind_solar_forecast.parquet"))
    fr_solar = pd.read_parquet(os.path.join(DATA_DIR, "fr_solar_forecast.parquet"))["solar_forecast_fr_mw"]

    lines.append("# QA Report — DE-LU Day-Ahead Price Forecasting Data")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("\n---\n")

    hourly_series = [
        ("prices", prices), ("exaa_price", exaa), ("load", load), ("wind", wind), ("solar", solar),
        ("load_fr", fr_load), ("wind_fr", fr_wind),
    ]

    # ------------------------------------------------------------------
    # 1. Data overview
    # ------------------------------------------------------------------
    h("1. Data Overview")
    row("| Series | Rows | Start | End | NaN (raw) |")
    row("|--------|-----:|-------|-----|----------:|")
    for name, s in hourly_series:
        row(f"| {name} | {len(s):,} | {s.index[0]} | {s.index[-1]} | {s.isna().sum()} |")
    row(f"| ttf (daily) | {len(ttf):,} | {ttf.index[0].date()} | {ttf.index[-1].date()} | {ttf.isna().sum()} |")
    row(f"| co2_proxy (daily) | {len(co2):,} | {co2.index[0].date()} | {co2.index[-1].date()} | {co2.isna().sum()} |")
    row(
        "\n> `load_fr` / `wind_fr` are ENTSO-E day-ahead forecast documents for BZN|FR "
        "(`query_load_forecast`, `query_wind_and_solar_forecast` via entsoe-py), fetched "
        "live via the ENTSO-E API rather than the manually-exported GUI CSVs used for "
        "DE-LU. `co2_proxy` is CARB.L (WisdomTree Carbon ETC, LSE) via yfinance — a "
        "tradable proxy for the EU ETS EUA price, not an official settlement print "
        "(see `data/fetch_fr_co2.py` docstring for why this proxy was chosen). "
        "**[v2 round 4]** `exaa_price` is BZN|DE-LU's **Sequence 2** column in the same raw "
        "CSVs as `prices` (Sequence 1) — EXAA's own day-ahead auction for the same zone, "
        "a real settlement, not a forecast or proxy. It is point-in-time safe by its own "
        "earlier gate closure (~10:15 CET D-1, vs. EPEX Sequence 1's ~12:00 CET D-1), not "
        "by any imputation here. Full-history correlation with the target series: 0.986.\n"
    )

    # ------------------------------------------------------------------
    # 2. Duplicate timestamps
    # ------------------------------------------------------------------
    h("2. Duplicate Timestamps")
    dup_results = {}
    for name, s in hourly_series:
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
    for name, s in hourly_series:
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
        ("exaa_price_eur_mwh", exaa),
        ("load_forecast_mw", load),
        ("wind_forecast_mw", wind),
        ("solar_forecast_mw", solar),
        ("load_forecast_fr_mw", fr_load),
        ("wind_forecast_fr_mw", fr_wind),
    ] + [(c, neighbor_prices[c]) for c in neighbor_prices.columns] \
      + [(c, neighbor_load[c]) for c in neighbor_load.columns] \
      + [(c, neighbor_ws[c]) for c in neighbor_ws.columns] \
      + [("solar_forecast_fr_mw", fr_solar)]:
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
        ("exaa_price", exaa, imputed["exaa_price"]),
        ("load",    load,   imputed["load"]),
        ("wind",    wind,   imputed["wind"]),
        ("solar",   solar,  imputed["solar"]),
        ("load_fr", fr_load, imputed["load_fr"]),
        ("wind_fr", fr_wind, imputed["wind_fr"]),
    ]:
        before = s_raw.isna().sum()
        after  = s_imp.isna().sum()
        row(f"| {name} | {before} | {after} | {before - after} |")

    # ------------------------------------------------------------------
    # 8. Forecast Transfer Capacity (NTC) coverage
    #    Not standard gap detection: borders going in/out of coverage as
    #    Europe shifts to flow-based capacity calculation is the EXPECTED
    #    shape of this data, not a data-quality fault. See
    #    data/fetch_ntc.py docstring for the full explanation.
    # ------------------------------------------------------------------
    h("8. Forecast Transfer Capacity (NTC) Coverage")
    row(
        "Day-Ahead NTC (`query_net_transfer_capacity_dayahead`, ENTSO-E document A61) is "
        "published for only some of DE-LU's physical borders, and the set has shrunk over "
        "2019-2025 as parts of Europe moved to flow-based capacity calculation (no single "
        "bilateral NTC number once a border joins a flow-based region). The columns below "
        "are exactly what's available — **not** padded or estimated for the missing borders. "
        "`ntc_import_capacity_mw` / `ntc_export_capacity_mw` (built in `features.py`) sum "
        "whichever of these are live at each hour, so their composition changes over time; "
        "see `data/fetch_ntc.py` docstring for the full per-border timeline.\n"
    )
    row("| Column | Rows | Start | End | NaN |")
    row("|--------|-----:|-------|-----|----:|")
    for col in ntc.columns:
        s = ntc[col].dropna()
        if len(s) == 0:
            row(f"| {col} | 0 | — | — | — |")
            continue
        row(f"| {col} | {len(s):,} | {s.index[0].date()} | {s.index[-1].date()} | {ntc[col].isna().sum()} |")
    row(
        "\nBorders with **no** published Day-Ahead NTC anywhere in 2019-2025 (checked all "
        "seven years): FR, BE, PL, NO_2, SE_4 — see `data/fetch_ntc_metadata.json` for the "
        "per-border rationale. These do not appear as columns at all."
    )

    # ------------------------------------------------------------------
    # 9. [v2 round 3] Neighbor bidding zone (FR/NL/BE/PL/CZ) coverage
    #    Like NTC above: gaps here are a real, expected shape (a live-API
    #    data-window boundary, an early-history capacity buildout), not a
    #    pipeline fault. See data/fetch_neighbors.py docstring.
    # ------------------------------------------------------------------
    h("9. Neighbor Bidding Zone (FR/NL/BE/PL/CZ) Coverage")
    row(
        "Used to build cross-border price-lag and residual-load features for DE-LU's five "
        "largest neighbors. Three findings, all flagged rather than silently patched:\n"
    )
    row("**a) Neighbor day-ahead prices truncate in late September 2025 in this build "
        "environment.** All five zones' live `query_day_ahead_prices` calls return data "
        "through 2025-09-29/30 and stop — including a same-day check against `DE_LU` itself "
        "via the live API, so this is a live-data-window boundary, not a per-zone fault. "
        "DE-LU's own price series is unaffected here because it's read from the committed "
        "GUI-export snapshot (`data/fetch_data.py`), not this live call. `features.py` "
        "forward-fills the last known neighbor price across the missing ~3 months (Oct-Dec "
        "2025) so price-lag features stay defined without truncating the full-year Test "
        "backtest — flagged here and in the report, not hidden.\n")
    row("**b) PL has no published day-ahead solar forecast before ~2020.** Treated as 0 in "
        "`features.py` (the same precedent as FR's pre-2022 offshore wind in "
        "`fetch_fr_co2.py`) — Poland's utility-scale solar buildout was genuinely minimal "
        "before then, not a publication gap.\n")
    row("**c) CZ has no published day-ahead wind forecast at all**, in any year (consistent "
        "with CZ's near-zero installed wind capacity — same finding as the NTC scan for CZ's "
        "border data). `residual_load_cz_mw` is therefore `load - solar` only, never "
        "`load - wind - solar`.\n")
    row("| Column | Rows | Start | End | NaN |")
    row("|--------|-----:|-------|-----|----:|")
    for df in (neighbor_prices, neighbor_load, neighbor_ws):
        for col in df.columns:
            s = df[col].dropna()
            if len(s) == 0:
                row(f"| {col} | 0 | — | — | — |")
                continue
            row(f"| {col} | {len(s):,} | {s.index[0]} | {s.index[-1]} | {df[col].isna().sum()} |")
    s = fr_solar.dropna()
    row(f"| solar_forecast_fr_mw | {len(s):,} | {s.index[0]} | {s.index[-1]} | {fr_solar.isna().sum()} |")

    # ------------------------------------------------------------------
    # 10. Overall verdict
    # ------------------------------------------------------------------
    h("10. Overall Verdict")
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
        "exaa_price":      imputed["exaa_price"],
        "load":            imputed["load"],
        "wind_forecast_mw": imputed["wind"],
        "solar_forecast_mw": imputed["solar"],
        "load_forecast_fr_mw": imputed["load_fr"],
        "wind_forecast_fr_mw": imputed["wind_fr"],
        "ttf":             ttf,
        "co2_proxy":       co2,
        "ntc":             ntc,
        "neighbor_prices": neighbor_prices,
        "neighbor_load":   neighbor_load,
        "neighbor_wind_solar": neighbor_ws,
        "fr_solar":        fr_solar,
        "passed":          all_pass,
    }


if __name__ == "__main__":
    result = run_qa()
    print(f"\nOverall pass: {result['passed']}")
    print(f"Remaining NaN after imputation:")
    for k in ("prices", "load", "wind_forecast_mw", "solar_forecast_mw",
              "load_forecast_fr_mw", "wind_forecast_fr_mw"):
        print(f"  {k}: {result[k].isna().sum()}")
