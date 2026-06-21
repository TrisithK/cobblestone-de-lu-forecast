"""
fetch_data.py — Build the DE-LU data snapshot from manually exported ENTSO-E CSVs.

Run once (no API key required):
    python data/fetch_data.py

Expected CSV files in data/ (7 annual files each, 2019-2026):
    GUI_ENERGY_PRICES_*.csv                        — DA prices (Sequence 1, EUR/MWh)
    GUI_TOTAL_LOAD_DAYAHEAD_*.csv                  — DA load forecast (MW)
    GUI_WIND_SOLAR_GENERATION_FORECAST_OFFSHORE_*  — DA wind offshore forecast (MW)
    GUI_WIND_SOLAR_GENERATION_FORECAST_ONSHORE_*   — DA wind onshore forecast (MW)
    GUI_WIND_SOLAR_GENERATION_FORECAST_SOLAR_*     — DA solar forecast (MW)

All series: Area = BZN|DE-LU, 15-min resolution → resampled to 1h mean.

Point-in-time firewall (CLAUDE.md §3.1):
    Load  : 'Day-ahead Total Load Forecast (MW)' — NOT 'Actual Total Load (MW)'
    Wind  : 'Day-ahead (MW)' — NOT 'Actual (MW)'
    Solar : 'Day-ahead (MW)' — NOT 'Actual (MW)'
    Prices: Sequence 1 day-ahead auction settlement

TTF front-month: fetched via yfinance (no key needed).

Outputs:
    data/da_prices.parquet          — hourly DA prices (EUR/MWh)
    data/load_forecast.parquet      — hourly DA load forecast (MW)
    data/wind_solar_forecast.parquet — hourly wind + solar DA forecast (MW)
    data/ttf_daily.parquet          — daily TTF front-month close
    data/fetch_metadata.json        — provenance record
"""

import glob
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_START = pd.Timestamp("2019-01-01", tz="Europe/Berlin")
HISTORY_END = pd.Timestamp.now(tz="Europe/Berlin").floor("D")
AREA = "BZN|DE-LU"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _glob(pattern: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not files:
        sys.exit(f"ERROR: no files matching data/{pattern}\nCheck the data/ directory.")
    return files


def _parse_mtu_start(col: pd.Series) -> pd.DatetimeIndex:
    """
    Parse the start of an MTU interval to a UTC-aware DatetimeIndex.
    Handles both formats present in ENTSO-E GUI exports:
      - 'DD/MM/YYYY HH:MM'     (load files)
      - 'DD/MM/YYYY HH:MM:SS'  (wind/solar files)
    """
    start_str = col.str.split(" - ").str[0].str.strip()
    ts = pd.to_datetime(start_str, dayfirst=True, utc=True)
    return ts.dt.tz_convert("Europe/Berlin")


def _load_csv_series(
    pattern: str,
    value_col: str,
    output_name: str,
    area_filter: bool = True,
) -> pd.Series:
    """
    Load all CSVs matching pattern, optionally filter Area == BZN|DE-LU,
    parse MTU start, resample 15-min → 1h mean, trim to HISTORY_START.
    """
    files = _glob(pattern)
    print(f"  {len(files)} file(s) matched: {pattern}")
    chunks = []
    for f in files:
        df = pd.read_csv(f)
        if area_filter and "Area" in df.columns:
            df = df[df["Area"] == AREA].copy()
        if df.empty:
            print(f"  WARNING: {os.path.basename(f)} — no rows for Area={AREA}, skipping")
            continue
        if value_col not in df.columns:
            sys.exit(
                f"ERROR: column '{value_col}' not found in {os.path.basename(f)}.\n"
                f"Available: {df.columns.tolist()}"
            )
        df.index = _parse_mtu_start(df["MTU (UTC)"])
        s = pd.to_numeric(df[value_col], errors="coerce")
        s.name = output_name
        s.index.name = "timestamp"
        chunks.append(s)

    raw = pd.concat(chunks).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    raw = raw.loc[HISTORY_START:]
    hourly = raw.resample("1h").mean()
    hourly.name = output_name
    return hourly


# ---------------------------------------------------------------------------
# 1. Prices
# ---------------------------------------------------------------------------

def load_prices() -> pd.Series:
    print("Loading day-ahead prices …")
    files = _glob("GUI_ENERGY_PRICES_*.csv")
    print(f"  {len(files)} file(s) matched: GUI_ENERGY_PRICES_*.csv")
    chunks = []
    for f in files:
        df = pd.read_csv(f)
        mask = (df["Area"] == AREA) & (df["Sequence"] == "Sequence 1")
        df = df[mask].copy()
        if df.empty:
            print(f"  WARNING: {os.path.basename(f)} — no DE-LU Sequence 1 rows, skipping")
            continue
        df.index = _parse_mtu_start(df["MTU (UTC)"])
        s = pd.to_numeric(df["Day-ahead Price (EUR/MWh)"], errors="coerce")
        s.name = "price_eur_mwh"
        s.index.name = "timestamp"
        chunks.append(s)

    raw = pd.concat(chunks).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    raw = raw.loc[HISTORY_START:]
    hourly = raw.resample("1h").mean()
    hourly.name = "price_eur_mwh"
    print(f"  → {len(hourly):,} hourly rows | {hourly.index[0]} → {hourly.index[-1]}")
    return hourly


# ---------------------------------------------------------------------------
# 2. Load forecast
#    Column: 'Day-ahead Total Load Forecast (MW)'  ← NOT 'Actual Total Load (MW)'
# ---------------------------------------------------------------------------

def load_load_forecast() -> pd.Series:
    print("Loading day-ahead load forecast …")
    hourly = _load_csv_series(
        pattern="GUI_TOTAL_LOAD_DAYAHEAD_*.csv",
        value_col="Day-ahead Total Load Forecast (MW)",
        output_name="load_forecast_mw",
    )
    print(f"  → {len(hourly):,} hourly rows | {hourly.index[0]} → {hourly.index[-1]}")
    return hourly


# ---------------------------------------------------------------------------
# 3. Wind + solar forecast
#    Column: 'Day-ahead (MW)'  ← NOT 'Actual (MW)'
#    Wind total = onshore + offshore (separate files)
# ---------------------------------------------------------------------------

def load_wind_solar_forecast() -> pd.DataFrame:
    print("Loading day-ahead wind (onshore + offshore) forecast …")
    onshore = _load_csv_series(
        pattern="GUI_WIND_SOLAR_GENERATION_FORECAST_ONSHORE_*.csv",
        value_col="Day-ahead (MW)",
        output_name="wind_onshore_mw",
    )
    offshore = _load_csv_series(
        pattern="GUI_WIND_SOLAR_GENERATION_FORECAST_OFFSHORE_*.csv",
        value_col="Day-ahead (MW)",
        output_name="wind_offshore_mw",
    )
    print("Loading day-ahead solar forecast …")
    solar = _load_csv_series(
        pattern="GUI_WIND_SOLAR_GENERATION_FORECAST_SOLAR_*.csv",
        value_col="Day-ahead (MW)",
        output_name="solar_forecast_mw",
    )

    # Align on shared index, sum onshore + offshore into total wind
    df = pd.DataFrame({
        "wind_forecast_mw": onshore.add(offshore, fill_value=0),
        "solar_forecast_mw": solar,
    })
    df = df.sort_index()
    print(f"  → {len(df):,} hourly rows | {df.index[0]} → {df.index[-1]}")
    return df


# ---------------------------------------------------------------------------
# 4. TTF front-month (yfinance — no key needed)
# ---------------------------------------------------------------------------

def load_ttf() -> pd.Series:
    print("Fetching TTF front-month (yfinance TTF=F) …")
    df = yf.download(
        "TTF=F",
        start=HISTORY_START.strftime("%Y-%m-%d"),
        end=HISTORY_END.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        print("  WARNING: TTF=F returned no data. Pipeline will fall back to residual-load lags.")
        return pd.Series(name="ttf_eur_mwh", dtype=float)
    close = df["Close"].squeeze()
    close.name = "ttf_eur_mwh"
    print(f"  → {len(close):,} daily rows | {close.index[0]} → {close.index[-1]}")
    return close


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("DE-LU data fetch — CSV-based, no API key required")
    print("=" * 55)

    prices = load_prices()
    load = load_load_forecast()
    wind_solar = load_wind_solar_forecast()
    ttf = load_ttf()

    # -----------------------------------------------------------------------
    # Point-in-time firewall assertion (CLAUDE.md §3.1)
    # Load / wind / solar are day-ahead FORECAST columns, confirmed by column
    # name selection above. Actuals columns exist in the CSVs but are ignored.
    # -----------------------------------------------------------------------
    print("\nPoint-in-time firewall:")
    print("  prices     → Day-ahead auction settlement (Sequence 1) ✓")
    print("  load       → 'Day-ahead Total Load Forecast (MW)' ✓")
    print("  wind/solar → 'Day-ahead (MW)' ✓")
    print("  (Actual columns present in CSVs but NOT used) ✓")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    prices.to_frame().to_parquet(os.path.join(DATA_DIR, "da_prices.parquet"))
    load.to_frame().to_parquet(os.path.join(DATA_DIR, "load_forecast.parquet"))
    wind_solar.to_parquet(os.path.join(DATA_DIR, "wind_solar_forecast.parquet"))
    if not ttf.empty:
        ttf.to_frame().to_parquet(os.path.join(DATA_DIR, "ttf_daily.parquet"))

    metadata = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_prices": "ENTSO-E GUI CSV — DA prices, BZN|DE-LU, Sequence 1, 15-min→1h mean",
        "source_load": "ENTSO-E GUI CSV — 'Day-ahead Total Load Forecast (MW)', BZN|DE-LU",
        "source_wind": "ENTSO-E GUI CSV — 'Day-ahead (MW)' onshore + offshore, BZN|DE-LU",
        "source_solar": "ENTSO-E GUI CSV — 'Day-ahead (MW)' solar, BZN|DE-LU",
        "source_ttf": "Yahoo Finance — TTF=F front-month daily close (yfinance)",
        "history_start": str(HISTORY_START.date()),
        "history_end": str(HISTORY_END.date()),
        "price_rows": len(prices),
        "load_rows": len(load),
        "wind_solar_rows": len(wind_solar),
        "ttf_rows": len(ttf),
        "pit_note": (
            "Load/wind/solar use day-ahead forecast columns only. "
            "Actual columns are present in the raw CSVs but are NOT loaded. "
            "This satisfies the point-in-time firewall: all features knowable by 12:00 D-1."
        ),
        "zone_note": (
            "DE-LU bidding zone only. History starts 2019-01-01 to exclude "
            "the pre-split DE-AT-LU regime (zone split October 2018)."
        ),
    }
    with open(os.path.join(DATA_DIR, "fetch_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nOutputs written to data/")
    print(f"  da_prices.parquet          {len(prices):>7,} rows")
    print(f"  load_forecast.parquet      {len(load):>7,} rows")
    print(f"  wind_solar_forecast.parquet{len(wind_solar):>7,} rows")
    print(f"  ttf_daily.parquet          {len(ttf):>7,} rows")
    print(f"\nLatest price : {prices.index[-1]}")
    print("Next: tell Claude the latest date so the OOS window can be locked.")


if __name__ == "__main__":
    main()
