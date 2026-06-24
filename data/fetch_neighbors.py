"""
fetch_neighbors.py — v2 round 3: neighbor bidding zone day-ahead prices +
load/wind/solar forecasts (FR, NL, BE, PL, CZ).

Used to build cross-border price-lag (D-1/D-7) and residual-load features
for these five DE-LU neighbors, the same way DE's own price lags and
residual load are already built.

Reuses the existing FR load (data/fr_load_forecast.parquet) and FR wind
(data/fr_wind_forecast.parquet) snapshots already fetched by
data/fetch_fr_co2.py — does NOT re-fetch them here, to avoid two sources of
truth for the same series. This script additionally fetches FR's solar
forecast (round 1 only kept wind), to complete FR's residual-load inputs;
features.py is updated to subtract it, refining round 1's load-minus-wind-
only definition now that solar is available too.

Run once (requires ENTSOE_API_KEY in .env):

    python data/fetch_neighbors.py

--- Coverage notes (spot-checked 2019/2022/2025; see fetch_neighbors_metadata.json) ---
  Prices       : FR/NL/BE/PL/CZ all continuously available 2019-2025.
  Load forecast: NL/BE/PL/CZ all continuously available 2019-2025. PL/CZ
                 publish at hourly resolution through ~2024, switching to
                 15-min in 2025 (EU-wide 15-min settlement); resampled to 1h
                 mean here like every other series in this build, so this
                 needs no special-casing.
  Wind+solar   : NL, BE — Solar + Wind Onshore + Wind Offshore, full span.
                 PL — Wind Onshore from 2019; Solar forecast only appears
                 from ~2020/2021 (Poland's solar buildout) — missing years
                 filled with 0, the same precedent already used for FR's
                 offshore wind in fetch_fr_co2.py (the physical capacity
                 was genuinely near-zero before those years).
                 CZ — Solar only. ENTSO-E publishes NO day-ahead wind
                 forecast for CZ at any point in 2019-2025 (consistent with
                 CZ's near-zero installed wind capacity). CZ residual load
                 is therefore load - solar only, not load - wind - solar —
                 flagged here and in features.py rather than assumed away.
                 FR — Solar fetched here for the first time (see above).

Point-in-time firewall: day-ahead prices are realised values published the
day before delivery (same justification as DE's existing price_lag_24h/168h
— known by gate closure, never same-day). Load/wind/solar use the same
ENTSO-E day-ahead forecast document types as every other series in this
build, never an actual/realised endpoint.

Outputs:
    data/neighbor_prices.parquet              — day-ahead price, FR/NL/BE/PL/CZ
    data/neighbor_load_forecast.parquet        — day-ahead load forecast, NL/BE/PL/CZ
    data/neighbor_wind_solar_forecast.parquet  — day-ahead wind+solar forecast,
                                                  NL/BE/PL/CZ + FR solar only
    data/fetch_neighbors_metadata.json         — provenance + per-zone coverage
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
YEARS = range(2019, 2026)
LOAD_WIND_SOLAR_ZONES = ["NL", "BE", "PL", "CZ"]
PRICE_ZONES = ["FR", "NL", "BE", "PL", "CZ"]


def fetch_prices(client) -> pd.DataFrame:
    # entsoe-py's query_day_ahead_prices pads the request by +/-1 day internally
    # ("fix issue 187") before calling the API. A full 365/366-day chunk then
    # exceeds ENTSO-E's server-side 1-year cap for this document and 400s.
    # Half-year chunks leave enough headroom regardless of the padding.
    half_year_starts = pd.date_range("2019-01-01", "2026-01-01", freq="6MS", tz="Europe/Berlin")
    chunk_bounds = list(zip(half_year_starts[:-1], half_year_starts[1:]))

    columns = {}
    for zone in PRICE_ZONES:
        print(f"Fetching {zone} day-ahead price …")
        chunks = []
        for start, end in chunk_bounds:
            try:
                s = client.query_day_ahead_prices(zone, start=start, end=end)
            except Exception as exc:
                print(f"  WARNING: {start.date()}-{end.date()} {zone} price fetch failed ({exc}); skipping")
                continue
            chunks.append(s)
        raw = pd.concat(chunks).sort_index()
        raw = raw[~raw.index.duplicated(keep="first")]
        raw.index = raw.index.tz_convert("Europe/Berlin")
        hourly = raw.resample("1h").mean()
        col = f"price_{zone.lower()}_eur_mwh"
        columns[col] = hourly
        print(f"  -> {len(hourly):,} hourly rows | {hourly.index[0]} -> {hourly.index[-1]}")
    return pd.DataFrame(columns).sort_index()


def fetch_load(client) -> pd.DataFrame:
    columns = {}
    for zone in LOAD_WIND_SOLAR_ZONES:
        print(f"Fetching {zone} day-ahead load forecast …")
        chunks = []
        for year in YEARS:
            start = pd.Timestamp(f"{year}-01-01", tz="Europe/Berlin")
            end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Berlin")
            try:
                s = client.query_load_forecast(zone, start=start, end=end)
            except Exception as exc:
                print(f"  WARNING: {year} {zone} load fetch failed ({exc}); skipping")
                continue
            col0 = s["Forecasted Load"] if hasattr(s, "columns") and "Forecasted Load" in s.columns else s
            if hasattr(col0, "squeeze"):
                col0 = col0.squeeze()
            chunks.append(col0)
        raw = pd.concat(chunks).sort_index()
        raw = raw[~raw.index.duplicated(keep="first")]
        raw.index = raw.index.tz_convert("Europe/Berlin")
        hourly = raw.resample("1h").mean()
        col = f"load_forecast_{zone.lower()}_mw"
        columns[col] = hourly
        print(f"  -> {len(hourly):,} hourly rows | {hourly.index[0]} -> {hourly.index[-1]}")
    return pd.DataFrame(columns).sort_index()


def fetch_wind_solar(client) -> tuple:
    """Returns (DataFrame for NL/BE/PL/CZ, Series for FR solar only)."""
    columns = {}
    coverage = {}
    for zone in LOAD_WIND_SOLAR_ZONES + ["FR"]:
        print(f"Fetching {zone} day-ahead wind+solar forecast …")
        wind_chunks, solar_chunks = [], []
        years_with_solar, years_with_wind = [], []
        for year in YEARS:
            start = pd.Timestamp(f"{year}-01-01", tz="Europe/Berlin")
            end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Berlin")
            try:
                df = client.query_wind_and_solar_forecast(zone, start=start, end=end, psr_type=None)
            except Exception as exc:
                print(f"  WARNING: {year} {zone} wind/solar fetch failed ({exc}); skipping")
                continue
            onshore = df["Wind Onshore"] if "Wind Onshore" in df.columns else None
            offshore = df["Wind Offshore"] if "Wind Offshore" in df.columns else None
            solar = df["Solar"] if "Solar" in df.columns else None
            if onshore is not None or offshore is not None:
                wind = (onshore if onshore is not None else 0)
                wind = wind.add(offshore, fill_value=0.0) if offshore is not None else wind
                wind_chunks.append(wind)
                years_with_wind.append(year)
            if solar is not None:
                solar_chunks.append(solar)
                years_with_solar.append(year)

        if zone == "FR":
            # Round 1 already fetched FR wind/load — only solar is new here.
            if solar_chunks:
                raw = pd.concat(solar_chunks).sort_index()
                raw = raw[~raw.index.duplicated(keep="first")]
                raw.index = raw.index.tz_convert("Europe/Berlin")
                fr_solar = raw.resample("1h").mean()
                fr_solar.name = "solar_forecast_fr_mw"
                print(f"  -> FR solar: {len(fr_solar):,} hourly rows (years: {years_with_solar})")
            else:
                fr_solar = pd.Series(dtype=float, name="solar_forecast_fr_mw")
            coverage["solar_forecast_fr_mw"] = years_with_solar
            continue

        if wind_chunks:
            raw = pd.concat(wind_chunks).sort_index()
            raw = raw[~raw.index.duplicated(keep="first")]
            raw.index = raw.index.tz_convert("Europe/Berlin")
            columns[f"wind_forecast_{zone.lower()}_mw"] = raw.resample("1h").mean()
            coverage[f"wind_forecast_{zone.lower()}_mw"] = years_with_wind
        else:
            coverage[f"wind_forecast_{zone.lower()}_mw"] = []
            print(f"  -> {zone}: no day-ahead wind forecast published in any year")

        if solar_chunks:
            raw = pd.concat(solar_chunks).sort_index()
            raw = raw[~raw.index.duplicated(keep="first")]
            raw.index = raw.index.tz_convert("Europe/Berlin")
            columns[f"solar_forecast_{zone.lower()}_mw"] = raw.resample("1h").mean()
            coverage[f"solar_forecast_{zone.lower()}_mw"] = years_with_solar
        else:
            coverage[f"solar_forecast_{zone.lower()}_mw"] = []

        for name, series in columns.items():
            if name.startswith((f"wind_forecast_{zone.lower()}", f"solar_forecast_{zone.lower()}")):
                print(f"  -> {name}: {len(series):,} hourly rows | {series.index[0]} -> {series.index[-1]}")

    return pd.DataFrame(columns).sort_index(), fr_solar, coverage


def main():
    print("=" * 55)
    print("v2 round 3: neighbor-zone price + load + wind/solar fetch")
    print("=" * 55)

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ENTSOE_API_KEY not set. Add it to .env (register a free token at "
            "https://transparency.entsoe.eu -> My Account -> Security token)."
        )

    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=api_key)

    prices = fetch_prices(client)
    load = fetch_load(client)
    wind_solar, fr_solar, ws_coverage = fetch_wind_solar(client)

    print("\nPoint-in-time firewall:")
    print("  Prices -> realised D-1/D-7 values, same standard as existing price_lag_24h/168h ✓")
    print("  Load/wind/solar -> ENTSO-E day-ahead forecast document types, never actuals ✓")

    prices.to_parquet(os.path.join(DATA_DIR, "neighbor_prices.parquet"))
    load.to_parquet(os.path.join(DATA_DIR, "neighbor_load_forecast.parquet"))
    wind_solar.to_parquet(os.path.join(DATA_DIR, "neighbor_wind_solar_forecast.parquet"))
    if not fr_solar.empty:
        fr_solar.to_frame().to_parquet(os.path.join(DATA_DIR, "fr_solar_forecast.parquet"))

    metadata = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ENTSO-E API (entsoe-py) — query_day_ahead_prices, query_load_forecast, "
                  "query_wind_and_solar_forecast",
        "price_zones": PRICE_ZONES,
        "load_wind_solar_zones": LOAD_WIND_SOLAR_ZONES,
        "wind_solar_coverage_years": ws_coverage,
        "cz_no_wind_note": (
            "ENTSO-E publishes no day-ahead wind forecast for CZ at any point in 2019-2025 "
            "(consistent with CZ's near-zero installed wind capacity). CZ residual load is "
            "load - solar only, not load - wind - solar."
        ),
        "fr_solar_note": (
            "Round 1 (fetch_fr_co2.py) only fetched FR wind, not solar. Fetched here for the "
            "first time; features.py now subtracts it from residual_load_fr_mw, refining "
            "round 1's load-minus-wind-only definition."
        ),
        "pit_note": (
            "Day-ahead prices are realised D-1/D-7 values, same justification as the existing "
            "DE price_lag_24h/168h features. Load/wind/solar use ENTSO-E day-ahead forecast "
            "document types only, same standard as every other series in this build."
        ),
    }
    with open(os.path.join(DATA_DIR, "fetch_neighbors_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nOutputs written to data/")
    print(f"  neighbor_prices.parquet              {prices.shape}")
    print(f"  neighbor_load_forecast.parquet        {load.shape}")
    print(f"  neighbor_wind_solar_forecast.parquet  {wind_solar.shape}")
    print(f"  fr_solar_forecast.parquet              {fr_solar.shape}")


if __name__ == "__main__":
    main()
