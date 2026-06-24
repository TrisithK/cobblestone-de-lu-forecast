"""
fetch_ntc.py — Build the v2 Forecast Transfer Capacity (NTC, day-ahead) snapshot.

Raw layer for the "Forecast Transfer Capacities" feature requested for v2:
one column per (border, direction) pair that ENTSO-E actually publishes a
bilateral Day-Ahead Net Transfer Capacity for. The engineered aggregates
(Total Import/Export/Net Transfer Capacity) are built from these raw columns
in src/features.py, not here — same fetch-raw / engineer-in-features.py split
used throughout this project.

Run once (requires ENTSOE_API_KEY in .env):

    python data/fetch_ntc.py

--- Why not "all ~10 DE-LU interconnectors"? ---
A coverage scan (2019-2025, sampled per year) found ENTSO-E's Day-Ahead NTC
document (A61) is published for only SOME of DE-LU's physical borders, and
the set has shrunk over time as Europe has moved to flow-based capacity
calculation (no single bilateral NTC number once a border joins a flow-based
region):

    Border   Years with a published Day-Ahead NTC (both directions)
    ------   ---------------------------------------------------------
    NL       2021-2025
    CH       2019-2025  (full window)
    AT       2019 only
    CZ       2019-2022
    DK_1     2019-2025  (full window)
    DK_2     2019-2023
    FR       never (CWE/Core flow-based market coupling predates this window)
    BE       never (same)
    PL       never (Core FBMC went live 2022; no NTC found even pre-2022)
    NO_2     never (NordLink HVDC — different/no day-ahead capacity publication)
    SE_4     never (Baltic Cable HVDC — same)

This means the "Total Import/Export Capacity" engineered features (built in
features.py) are a sum over WHICHEVER of these six borders are publishing at
each point in time, not a constant set of ~10 borders. When CZ stops in 2022
or DK_2 stops in 2023, the aggregate level-shifts for a reporting/methodology
reason, not a real change in Germany's physical interconnection. This is
documented again in qa.py / features.py / the report — flagged honestly
rather than backfilled with an estimate, consistent with how this project
already handled the EEX-print and CO2-proxy decisions.

--- Parser note ---
entsoe-py's built-in query_net_transfer_capacity_dayahead()/
parse_crossborder_flows() assumes one <Point> per resolution step. NTC is
published as a step function (curveType A03): a <Point> only appears when
the value CHANGES, and holds until the next one. That mismatch makes the
library's parser raise ("Length mismatch") or return nothing, so this script
calls EntsoeRawClient directly and parses the step function itself
(_parse_ntc_stepfunction below).

Point-in-time firewall: this is ENTSO-E's day-ahead (A61, process A01)
capacity forecast — the same publication timing class as the load/wind/solar
day-ahead forecasts already used elsewhere in this build. No shift is needed
in features.py, same as those.

Outputs:
    data/ntc_forecast.parquet   — hourly NTC per (border, direction), Europe/Berlin
    data/fetch_ntc_metadata.json — provenance + coverage record
"""

import json
import os
from datetime import datetime, timezone

import bs4
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# (border, [years with confirmed Day-Ahead NTC coverage, both directions])
# from the 2019-2025 coverage scan described in the module docstring.
BORDER_COVERAGE = {
    "NL":   range(2021, 2026),
    "CH":   range(2019, 2026),
    "AT":   range(2019, 2020),
    "CZ":   range(2019, 2023),
    "DK_1": range(2019, 2026),
    "DK_2": range(2019, 2024),
}


def _parse_ntc_stepfunction(xml_text: str, tz: str) -> pd.Series:
    """
    Parse an ENTSO-E A61 (NTC Day-Ahead) Publication_MarketDocument into a
    continuous hourly series, expanding the step-function <Point> encoding
    (curveType A03: a point marks where the value CHANGES, not one point per
    hour). Returns an empty Series if no <TimeSeries> is present.
    """
    soup = bs4.BeautifulSoup(xml_text, "html.parser")
    chunks = []
    period_ends = []
    for ts in soup.find_all("timeseries"):
        for period in ts.find_all("period"):
            period_start = pd.Timestamp(period.find("start").text)
            period_end = pd.Timestamp(period.find("end").text)
            res_str = period.find("resolution").text
            freq = {
                "PT60M": pd.Timedelta(hours=1),
                "PT30M": pd.Timedelta(minutes=30),
                "PT15M": pd.Timedelta(minutes=15),
            }.get(res_str)
            if freq is None:
                raise ValueError(f"Unhandled NTC resolution: {res_str}")
            points = period.find_all("point")
            positions = [int(p.find("position").text) for p in points]
            quantities = [float(p.find("quantity").text) for p in points]
            idx = [period_start + (pos - 1) * freq for pos in positions]
            chunks.append(pd.Series(quantities, index=idx))
            period_ends.append(period_end)

    if not chunks:
        return pd.Series(dtype=float)

    s = pd.concat(chunks).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    overall_end = max(period_ends) - pd.Timedelta(hours=1)
    full_idx = pd.date_range(s.index.min(), overall_end, freq="1h", tz="UTC")
    s = s.reindex(full_idx).ffill()
    s.index = s.index.tz_convert(tz)
    return s


def fetch_border_year(client, frm: str, to: str, year: int) -> pd.Series:
    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Berlin")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Berlin")
    text = client.query_net_transfer_capacity_dayahead(frm, to, start=start, end=end)
    return _parse_ntc_stepfunction(text, tz="Europe/Berlin")


def main():
    print("=" * 55)
    print("v2 Forecast Transfer Capacity (NTC, day-ahead) fetch")
    print("=" * 55)

    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ENTSOE_API_KEY not set. Add it to .env (register a free token at "
            "https://transparency.entsoe.eu -> My Account -> Security token)."
        )

    from entsoe import EntsoeRawClient
    client = EntsoeRawClient(api_key=api_key)

    columns = {}
    coverage_record = {}
    for border, years in BORDER_COVERAGE.items():
        for frm, to, col_name in [
            ("DE_LU", border, f"ntc_DE_LU_to_{border}_mw"),
            (border, "DE_LU", f"ntc_{border}_to_DE_LU_mw"),
        ]:
            print(f"Fetching {frm} -> {to} ({min(years)}-{max(years)}) …")
            chunks = []
            for year in years:
                try:
                    s = fetch_border_year(client, frm, to, year)
                except Exception as exc:
                    print(f"  WARNING: {year} {frm}->{to} fetch failed ({exc}); skipping")
                    continue
                if not s.empty:
                    chunks.append(s)
            if not chunks:
                print(f"  -> no data returned for {frm}->{to}, skipping column")
                continue
            full = pd.concat(chunks).sort_index()
            full = full[~full.index.duplicated(keep="first")]
            columns[col_name] = full
            coverage_record[col_name] = {
                "start": str(full.index[0]),
                "end": str(full.index[-1]),
                "rows": len(full),
            }
            print(f"  -> {len(full):,} hourly rows | {full.index[0]} -> {full.index[-1]}")

    df = pd.DataFrame(columns)
    df = df.sort_index()

    print("\nPoint-in-time firewall:")
    print("  NTC -> ENTSO-E A61 Day-Ahead capacity document (process A01) ✓")
    print("  Same publication timing class as load/wind/solar day-ahead forecasts ✓")

    out_path = os.path.join(DATA_DIR, "ntc_forecast.parquet")
    df.to_parquet(out_path)

    metadata = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ENTSO-E API (entsoe-py EntsoeRawClient) — query_net_transfer_capacity_dayahead, "
                   "custom step-function parser (see module docstring)",
        "borders_covered": list(BORDER_COVERAGE.keys()),
        "borders_never_available": ["FR", "BE", "PL", "NO_2", "SE_4"],
        "borders_never_available_note": (
            "No bilateral Day-Ahead NTC TimeSeries was found for these borders at any point "
            "in 2019-2025 (spot-checked across all 7 years). Most likely cause: flow-based "
            "market coupling (CWE/Core region covers FR/BE; Core FBMC went live for PL in "
            "2022, but no NTC was found even pre-2022) or, for the NO_2/SE_4 HVDC links, a "
            "different capacity-allocation mechanism not published as a simple Day-Ahead NTC."
        ),
        "coverage_per_column": coverage_record,
        "pit_note": (
            "Day-Ahead NTC (A61, process A01) is the same publication timing class as the "
            "load/wind/solar day-ahead forecasts already used in this pipeline — known by "
            "gate closure, no additional shift needed in features.py."
        ),
    }
    with open(os.path.join(DATA_DIR, "fetch_ntc_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nOutput written -> {out_path}  ({df.shape[0]:,} rows x {df.shape[1]} columns)")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
