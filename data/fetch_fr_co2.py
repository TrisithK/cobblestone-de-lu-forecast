"""
fetch_fr_co2.py — Build the v2 cross-border (FR) + carbon snapshot.

Adds three series to the data/ snapshot, additive to the existing DE-LU build
(fetch_data.py is untouched):

  FR day-ahead load forecast   — ENTSO-E API (entsoe-py), BZN|FR
  FR day-ahead wind forecast   — ENTSO-E API (entsoe-py), BZN|FR (onshore + offshore)
  EU ETS CO2 price proxy       — yfinance, WisdomTree Carbon ETC (CARB.L, LSE, USD)

Run once (requires ENTSOE_API_KEY in .env — register a free token at
https://transparency.entsoe.eu -> My Account -> Security token):

    python data/fetch_fr_co2.py

Point-in-time firewall (CLAUDE.md §3.1, same standard as the DE-LU build):
    FR load : query_load_forecast() -> ENTSO-E "Day-ahead Total Load Forecast"
              document (A65/A01 process), never realised load.
    FR wind : query_wind_and_solar_forecast() -> ENTSO-E day-ahead generation
              forecast document (A69), never actual generation.

CO2 proxy honesty note (mirrors the TTF=F precedent already in fetch_data.py):
    CARB.L is a tradable ETC that tracks the ICE EUA carbon futures total
    return index — it is NOT an official EEX/ICE daily settlement print, the
    same way TTF=F is a tradable future and not a PEGAS settlement print.
    It was selected after free, full-2019-2025-history alternatives were
    checked and ruled out: KRBN (KraneShares Global Carbon ETF) only starts
    2020-07-31; Nasdaq Data Link's ICE_C1 EUA dataset and Stooq's futures
    endpoints both blocked unauthenticated/automated access in this build
    environment. CARB.L has continuous daily history back to 2019-01-02.

Outputs:
    data/fr_load_forecast.parquet  — hourly FR DA load forecast (MW)
    data/fr_wind_forecast.parquet  — hourly FR DA wind forecast, onshore+offshore (MW)
    data/co2_proxy_daily.parquet   — daily CARB.L close (USD), EU ETS proxy
    data/fetch_fr_co2_metadata.json — provenance record
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_START = pd.Timestamp("2019-01-01", tz="Europe/Paris")
HISTORY_END = pd.Timestamp("2026-01-01", tz="Europe/Paris")  # exclusive upper bound
FR_AREA = "FR"


# ---------------------------------------------------------------------------
# 1. FR load forecast (ENTSO-E API, year-by-year chunks)
# ---------------------------------------------------------------------------

def fetch_fr_load(client) -> pd.Series:
    print("Fetching FR day-ahead load forecast (ENTSO-E API) …")
    chunks = []
    for year in range(2019, 2026):
        start = pd.Timestamp(f"{year}-01-01", tz="Europe/Paris")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Paris")
        try:
            df = client.query_load_forecast(FR_AREA, start=start, end=end)
        except Exception as exc:
            print(f"  WARNING: {year} FR load fetch failed ({exc}); skipping")
            continue
        s = df["Forecasted Load"] if "Forecasted Load" in df.columns else df.iloc[:, 0]
        s.name = "load_forecast_fr_mw"
        chunks.append(s)
        print(f"  {year}: {len(s)} rows")

    raw = pd.concat(chunks).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    raw.index = raw.index.tz_convert("Europe/Berlin")
    hourly = raw.resample("1h").mean()
    hourly.name = "load_forecast_fr_mw"
    print(f"  -> {len(hourly):,} hourly rows | {hourly.index[0]} -> {hourly.index[-1]}")
    return hourly


# ---------------------------------------------------------------------------
# 2. FR wind forecast (onshore + offshore, year-by-year chunks)
#    Offshore column only appears once FR offshore capacity exists (~2022+);
#    treated as 0 for earlier years.
# ---------------------------------------------------------------------------

def fetch_fr_wind(client) -> pd.Series:
    print("Fetching FR day-ahead wind forecast (onshore + offshore, ENTSO-E API) …")
    chunks = []
    for year in range(2019, 2026):
        start = pd.Timestamp(f"{year}-01-01", tz="Europe/Paris")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Paris")
        try:
            df = client.query_wind_and_solar_forecast(FR_AREA, start=start, end=end, psr_type=None)
        except Exception as exc:
            print(f"  WARNING: {year} FR wind fetch failed ({exc}); skipping")
            continue
        onshore = df["Wind Onshore"] if "Wind Onshore" in df.columns else pd.Series(0.0, index=df.index)
        offshore = df["Wind Offshore"] if "Wind Offshore" in df.columns else pd.Series(0.0, index=df.index)
        total = onshore.add(offshore, fill_value=0.0)
        total.name = "wind_forecast_fr_mw"
        chunks.append(total)
        print(f"  {year}: {len(total)} rows (offshore present: {'Wind Offshore' in df.columns})")

    raw = pd.concat(chunks).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    raw.index = raw.index.tz_convert("Europe/Berlin")
    hourly = raw.resample("1h").mean()
    hourly.name = "wind_forecast_fr_mw"
    print(f"  -> {len(hourly):,} hourly rows | {hourly.index[0]} -> {hourly.index[-1]}")
    return hourly


# ---------------------------------------------------------------------------
# 3. EU ETS CO2 price proxy (yfinance — no key needed)
# ---------------------------------------------------------------------------

def fetch_co2_proxy() -> pd.Series:
    print("Fetching EU ETS CO2 proxy (yfinance CARB.L — WisdomTree Carbon ETC) …")
    df = yf.download(
        "CARB.L",
        start="2019-01-01",
        end="2026-01-01",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        print("  WARNING: CARB.L returned no data. gas_co2_pressure_index will fall back to TTF-only.")
        return pd.Series(name="co2_proxy_usd", dtype=float)
    close = df["Close"].squeeze()
    close.name = "co2_proxy_usd"
    print(f"  -> {len(close):,} daily rows | {close.index[0].date()} -> {close.index[-1].date()}")
    return close


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("v2 cross-border (FR) + carbon data fetch")
    print("=" * 55)

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ENTSOE_API_KEY not set. Add it to .env (register a free token at "
            "https://transparency.entsoe.eu -> My Account -> Security token)."
        )

    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=api_key)

    fr_load = fetch_fr_load(client)
    fr_wind = fetch_fr_wind(client)
    co2 = fetch_co2_proxy()

    print("\nPoint-in-time firewall:")
    print("  FR load -> query_load_forecast() day-ahead forecast document ✓")
    print("  FR wind -> query_wind_and_solar_forecast() day-ahead forecast document ✓")
    print("  (Neither call touches an 'actual generation/load' endpoint) ✓")

    fr_load.to_frame().to_parquet(os.path.join(DATA_DIR, "fr_load_forecast.parquet"))
    fr_wind.to_frame().to_parquet(os.path.join(DATA_DIR, "fr_wind_forecast.parquet"))
    if not co2.empty:
        co2.to_frame().to_parquet(os.path.join(DATA_DIR, "co2_proxy_daily.parquet"))

    metadata = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_fr_load": "ENTSO-E API (entsoe-py) — query_load_forecast('FR'), day-ahead forecast",
        "source_fr_wind": "ENTSO-E API (entsoe-py) — query_wind_and_solar_forecast('FR'), onshore+offshore day-ahead forecast",
        "source_co2": "Yahoo Finance — CARB.L (WisdomTree Carbon ETC, LSE, USD) daily close (yfinance)",
        "co2_proxy_note": (
            "CARB.L tracks the ICE EUA carbon futures total-return index. It is a tradable "
            "proxy, not an official EEX/ICE daily settlement print — same status as TTF=F "
            "for gas. Selected because it has continuous daily history for the full "
            "2019-2025 build window; free alternatives with full coverage (Nasdaq Data "
            "Link ICE_C1, Stooq futures) were blocked from automated access; KRBN only "
            "starts 2020-07-31."
        ),
        "pit_note": (
            "FR load/wind use day-ahead forecast ENTSO-E document types only "
            "(query_load_forecast, query_wind_and_solar_forecast) — never actual/realised "
            "generation or load endpoints. Satisfies the same point-in-time firewall as the "
            "DE-LU series in fetch_data.py."
        ),
        "fr_load_rows": len(fr_load),
        "fr_wind_rows": len(fr_wind),
        "co2_rows": len(co2),
    }
    with open(os.path.join(DATA_DIR, "fetch_fr_co2_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nOutputs written to data/")
    print(f"  fr_load_forecast.parquet   {len(fr_load):>7,} rows")
    print(f"  fr_wind_forecast.parquet   {len(fr_wind):>7,} rows")
    print(f"  co2_proxy_daily.parquet    {len(co2):>7,} rows")


if __name__ == "__main__":
    main()
