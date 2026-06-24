"""
features.py — Point-in-time feature engineering for the DE-LU pipeline.

FIREWALL RULE (CLAUDE.md §3.1):
    Every feature for delivery hour h on day D must be knowable at ~12:00 D-1.
    - DA forecasts (load/wind/solar): published by ENTSO-E before gate closure ✓
    - Price lags: shift(24) = D-1 same-hour, shift(168) = D-7 same-hour ✓
    - Rolling stats: computed on resid.shift(24) → only uses data up to D-1 ✓
    - TTF: D-1 daily close (forward-filled over weekends/holidays) ✓
    - FORBIDDEN: any realised value from day D; any actual generation series

Feature matrix columns (v2 — DE + FR cross-border + NTC + neighbor-zone
lags/residual-load, 47 features + target):
    Calendar    : hour, dow, month, is_weekend, is_holiday,
                  hour_sin/cos, dow_sin/cos, month_sin/cos
    DA forecasts: load_forecast_mw, wind_forecast_mw, solar_forecast_mw,
                  residual_load_mw
    Merit-order : residual_load_sq (scaled), residual_load_high (scarcity hinge)
    Price lags  : price_lag_24h, price_lag_168h
    Rolling     : rolling_resid_mean_7d, rolling_resid_std_7d,
                  rolling_price_mean_7d, rolling_price_std_7d
    Gas anchor  : ttf_eur_mwh
    Gas x merit-order (optional): ttf_resid_interaction
        — lets the Ridge model re-slope residual_load->price with gas, rather
        than only shifting the level. Ridge-only (see models.RIDGE_FEATURES);
        LightGBM already captures interactions nonlinearly without it.
    Cross-border (v2): residual_load_fr_mw = load_forecast_fr_mw - wind_forecast_fr_mw
        — FR day-ahead forecasts (ENTSO-E API), same point-in-time standard as DE.
        FR has no day-ahead interconnector-flow forecast published (confirmed by the
        NTC coverage scan below — FR is one of the borders with no published NTC at
        all), so FR residual load is used as a demand/supply pressure proxy instead
        (DE-FR interconnectors run near-saturated, so FR scarcity/surplus tends to
        move through to DE-LU price pressure via implicit flows).
    Gas+carbon (v2): gas_co2_pressure_index
        — composite of standardised (expanding z-score, point-in-time safe) TTF
        and CO2-proxy levels. Deliberately NOT expressed in EUR/MWh: the CO2
        input (CARB.L, a tradable ETC — see data/fetch_fr_co2.py) is not an
        official EUR/tonne EUA print, so forcing it into a literal spark-spread
        formula would manufacture false precision. A standardised composite lets
        gas and carbon contribute a combined "thermal marginal-cost pressure"
        signal without pretending the units compose into a real cost figure.
    Forecast Transfer Capacity / NTC (v2): ntc_import_capacity_mw,
        ntc_export_capacity_mw, ntc_net_transfer_capacity_mw (export - import)
        — ENTSO-E day-ahead NTC (A61), summed across whichever DE-LU borders are
        actually publishing one at each hour (CH and DK_1 cover the full
        2019-2025 span; NL from 2021; AT only 2019; CZ through mid-2022; DK_2
        through 2023). FR/BE/PL/NO_2/SE_4 never publish a bilateral Day-Ahead
        NTC in this window (flow-based coupling / different HVDC mechanism) and
        are not included. The aggregate's border composition therefore changes
        over time — flagged here, in qa.py, and in the report rather than
        backfilled with an estimate. See data/fetch_ntc.py docstring for the
        full per-border timeline.
    Neighbor zone price lags (v2 round 3): price_lag_24h_{fr,nl,be,pl,cz},
        price_lag_168h_{fr,nl,be,pl,cz} — D-1/D-7 same-hour realised prices
        for each neighbor, identical point-in-time justification as DE's own
        price_lag_24h/168h. Forward-filled past the 2025-09-30 live-API data
        boundary (data/fetch_neighbors.py) so the lags stay defined through
        the rest of the Test year rather than truncating it.
    Neighbor zone residual load (v2 round 3): residual_load_{nl,be,pl,cz}_mw
        = load - wind - solar wherever both are published; CZ has no
        day-ahead wind forecast at all (load - solar only); PL's missing
        pre-2020-04 solar is filled with 0 (genuine pre-buildout capacity,
        same precedent as FR's pre-2022 offshore wind). FR's own
        residual_load_fr_mw (above) is also refined here to subtract FR
        solar, now that data/fetch_neighbors.py fetches it.
"""

import os
from typing import Optional

import holidays as hols
import numpy as np
import pandas as pd

from config import RESID_HIGH_MW

_ROOT    = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(_ROOT, "data")

_DE_HOLIDAYS = hols.Germany(years=range(2019, 2027))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cyclic(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    sin_ = np.sin(2 * np.pi * series / period)
    cos_ = np.cos(2 * np.pi * series / period)
    return sin_, cos_


def _align_daily_to_d1(daily: pd.Series, hourly_index: pd.DatetimeIndex, output_name: str) -> pd.Series:
    """
    Map a daily series (e.g. TTF close, CO2 proxy close) to each delivery hour
    using the D-1 value. Forward-filled over weekends and exchange holidays.
    Daily index is tz-naive; compared against tz-naive D-1 dates.
    """
    daily_copy = daily.copy()
    daily_copy.index = pd.to_datetime(daily_copy.index).normalize()

    # Forward-fill to cover all calendar days
    full_range = pd.date_range(
        start=daily_copy.index.min(),
        end=daily_copy.index.max() + pd.Timedelta(days=2),
        freq="D",
    )
    daily_ff = daily_copy.reindex(full_range).ffill()

    # For each delivery hour h on day D, look up the daily value from D-1
    # (tz-naive midnight)
    d1_naive = pd.DatetimeIndex(
        hourly_index.normalize().tz_localize(None) - pd.Timedelta(days=1)
    )
    values = daily_ff.reindex(d1_naive).values
    return pd.Series(values, index=hourly_index, name=output_name)


def _align_ttf(ttf: pd.Series, hourly_index: pd.DatetimeIndex) -> pd.Series:
    return _align_daily_to_d1(ttf, hourly_index, "ttf_eur_mwh")


def _expanding_zscore(series: pd.Series, min_periods: int = 30) -> pd.Series:
    """
    Point-in-time-safe standardisation: at each row, mean/std are computed over
    all rows up to and including the current one (no future information). Used
    to build gas_co2_pressure_index without needing a fixed train-only scaler.
    """
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std()
    return (series - mean) / std


# ---------------------------------------------------------------------------
# Firewall assertion (CLAUDE.md §3.1)
# ---------------------------------------------------------------------------

def assert_no_lookahead(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Runtime check that lagged features are correct shifted copies of the target.
    Raises AssertionError on any violation — run this after build_features().
    """
    # price_lag_24h must equal y.shift(24)
    common24 = X["price_lag_24h"].dropna().index
    expected24 = y.shift(24).reindex(common24)
    diff24 = (X.loc[common24, "price_lag_24h"] - expected24).abs().max()
    assert diff24 < 1e-6, (
        f"FIREWALL VIOLATION: price_lag_24h ≠ target.shift(24).  Max diff: {diff24:.6f}"
    )

    # price_lag_168h must equal y.shift(168)
    common168 = X["price_lag_168h"].dropna().index
    expected168 = y.shift(168).reindex(common168)
    diff168 = (X.loc[common168, "price_lag_168h"] - expected168).abs().max()
    assert diff168 < 1e-6, (
        f"FIREWALL VIOLATION: price_lag_168h ≠ target.shift(168).  Max diff: {diff168:.6f}"
    )

    # Rolling stats must be based on shifted residual (no same-day values)
    # Verify by checking that rolling_resid_mean_7d is NaN for the first 24+168 rows
    first_valid_resid = X["rolling_resid_mean_7d"].first_valid_index()
    first_idx = X.index[0]
    hours_until_valid = int((first_valid_resid - first_idx) / pd.Timedelta("1h"))
    assert hours_until_valid >= 24, (
        f"FIREWALL VIOLATION: rolling_resid_mean_7d has values within first 24h "
        f"(first valid at hour {hours_until_valid})"
    )

    print("Point-in-time firewall assertion: PASSED ✓")
    print(f"  price_lag_24h  matches target.shift(24)  — max diff {diff24:.2e} EUR/MWh")
    print(f"  price_lag_168h matches target.shift(168) — max diff {diff168:.2e} EUR/MWh")
    print(f"  rolling_resid first valid at hour +{hours_until_valid} (≥ 24 required) ✓")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_features(clean: Optional[dict] = None) -> tuple:
    """
    Build the full point-in-time feature matrix over 2019-2025.

    Parameters
    ----------
    clean : dict returned by qa.run_qa(), containing imputed Series.
            If None, reloads from parquet files in data/.

    Returns
    -------
    X : pd.DataFrame — feature matrix, tz-aware Europe/Berlin index
    y : pd.Series    — target (price_eur_mwh), same index
    """
    # ------------------------------------------------------------------
    # 0. Load / receive data
    # ------------------------------------------------------------------
    if clean is not None:
        prices  = clean["prices"]
        load    = clean["load"]
        wind    = clean["wind_forecast_mw"]
        solar   = clean["solar_forecast_mw"]
        load_fr = clean["load_forecast_fr_mw"]
        wind_fr = clean["wind_forecast_fr_mw"]
        ttf     = clean["ttf"]
        co2     = clean["co2_proxy"]
        ntc     = clean["ntc"]
        neighbor_prices = clean["neighbor_prices"]
        neighbor_load   = clean["neighbor_load"]
        neighbor_ws     = clean["neighbor_wind_solar"]
        fr_solar        = clean["fr_solar"]
    else:
        prices  = pd.read_parquet(os.path.join(DATA_DIR, "da_prices.parquet"))["price_eur_mwh"]
        load    = pd.read_parquet(os.path.join(DATA_DIR, "load_forecast.parquet"))["load_forecast_mw"]
        ws      = pd.read_parquet(os.path.join(DATA_DIR, "wind_solar_forecast.parquet"))
        wind    = ws["wind_forecast_mw"]
        solar   = ws["solar_forecast_mw"]
        load_fr = pd.read_parquet(os.path.join(DATA_DIR, "fr_load_forecast.parquet"))["load_forecast_fr_mw"]
        wind_fr = pd.read_parquet(os.path.join(DATA_DIR, "fr_wind_forecast.parquet"))["wind_forecast_fr_mw"]
        ttf     = pd.read_parquet(os.path.join(DATA_DIR, "ttf_daily.parquet"))["ttf_eur_mwh"]
        co2     = pd.read_parquet(os.path.join(DATA_DIR, "co2_proxy_daily.parquet"))["co2_proxy_usd"]
        ntc     = pd.read_parquet(os.path.join(DATA_DIR, "ntc_forecast.parquet"))
        neighbor_prices = pd.read_parquet(os.path.join(DATA_DIR, "neighbor_prices.parquet"))
        neighbor_load   = pd.read_parquet(os.path.join(DATA_DIR, "neighbor_load_forecast.parquet"))
        neighbor_ws     = pd.read_parquet(os.path.join(DATA_DIR, "neighbor_wind_solar_forecast.parquet"))
        fr_solar        = pd.read_parquet(os.path.join(DATA_DIR, "fr_solar_forecast.parquet"))["solar_forecast_fr_mw"]

    idx = prices.index  # tz-aware hourly spine

    # ------------------------------------------------------------------
    # 1. Calendar features
    # ------------------------------------------------------------------
    hour  = pd.Series(idx.hour,  index=idx, name="hour")
    dow   = pd.Series(idx.dayofweek, index=idx, name="dow")
    month = pd.Series(idx.month, index=idx, name="month")

    hour_sin,  hour_cos  = _cyclic(hour,  24)
    dow_sin,   dow_cos   = _cyclic(dow,   7)
    month_sin, month_cos = _cyclic(month, 12)
    hour_sin.name, hour_cos.name   = "hour_sin", "hour_cos"
    dow_sin.name,  dow_cos.name    = "dow_sin",  "dow_cos"
    month_sin.name, month_cos.name = "month_sin", "month_cos"

    is_weekend = pd.Series(
        (idx.dayofweek >= 5).astype(int), index=idx, name="is_weekend"
    )
    is_holiday = pd.Series(
        [int(ts.date() in _DE_HOLIDAYS) for ts in idx], index=idx, name="is_holiday"
    )

    # ------------------------------------------------------------------
    # 2. DA forecast features  [all known by 12:00 D-1 ✓]
    # ------------------------------------------------------------------
    residual_load = (load - wind - solar).rename("residual_load_mw")

    # ------------------------------------------------------------------
    # 3. Merit-order nonlinearity terms on residual load
    #    Captures the kinked supply stack:
    #      - quadratic: upward convexity at high demand
    #      - high hinge: scarcity / gas-peaker zone above p80
    # ------------------------------------------------------------------
    resid_sq   = (residual_load ** 2 / 1e6).rename("residual_load_sq")  # scaled
    resid_high = (residual_load - RESID_HIGH_MW).clip(lower=0).rename("residual_load_high")

    # ------------------------------------------------------------------
    # 4. Price lags  [D-1 same-hour: shift(24); D-7 same-hour: shift(168)]
    #    Both are realised prices published well before gate closure ✓
    # ------------------------------------------------------------------
    lag_24h  = prices.shift(24).rename("price_lag_24h")
    lag_168h = prices.shift(168).rename("price_lag_168h")

    # ------------------------------------------------------------------
    # 5. Rolling residual-load statistics
    #    .shift(24) ensures we only use information from D-1 and earlier ✓
    #    Window = 168h (7 days) captures recent demand/renewable regime.
    # ------------------------------------------------------------------
    resid_shifted = residual_load.shift(24)
    rolling_mean  = resid_shifted.rolling(168, min_periods=72).mean().rename("rolling_resid_mean_7d")
    rolling_std   = resid_shifted.rolling(168, min_periods=72).std().rename("rolling_resid_std_7d")

    # ------------------------------------------------------------------
    # 6. TTF daily close mapped to D-1  [known by gate closure ✓]
    # ------------------------------------------------------------------
    ttf_hourly = _align_ttf(ttf, idx)

    # ------------------------------------------------------------------
    # 6b. Gas x merit-order interaction (REVISION_PLAN.md A3, optional)
    #    TTF currently enters additively (a level shift only). The real
    #    effect of gas is on the *slope* of residual_load->price (the spark
    #    spread): when gas is expensive, each extra MW of residual load
    #    should move price more. This interaction lets Ridge re-slope with
    #    gas without needing a short rolling window to "forget" old data.
    #    Both inputs are already point-in-time (D-1 TTF close, D-1 forecast
    #    residual load), so the product is point-in-time too.
    # ------------------------------------------------------------------
    ttf_resid_interaction = (ttf_hourly * residual_load).rename("ttf_resid_interaction")

    # ------------------------------------------------------------------
    # 7. Rolling PRICE statistics (v2)
    #    .shift(24) ensures we only use information from D-1 and earlier ✓
    #    Distinct from the rolling residual-load stats above: this captures
    #    the recent realised-price level/volatility regime directly, which
    #    the lags alone (single points) don't summarise.
    # ------------------------------------------------------------------
    price_shifted        = prices.shift(24)
    rolling_price_mean   = price_shifted.rolling(168, min_periods=72).mean().rename("rolling_price_mean_7d")
    rolling_price_std    = price_shifted.rolling(168, min_periods=72).std().rename("rolling_price_std_7d")

    # ------------------------------------------------------------------
    # 8. FR cross-border residual load (v2)  [day-ahead forecasts, known by
    #    gate closure ✓ — same point-in-time standard as DE, see
    #    data/fetch_fr_co2.py]
    #
    #    [v2 round 3] Now subtracts FR solar too (fetched in
    #    data/fetch_neighbors.py — round 1 only fetched FR wind), refining
    #    round 1's load-minus-wind-only definition.
    # ------------------------------------------------------------------
    fr_solar_aligned = fr_solar.reindex(idx)
    residual_load_fr = (load_fr - wind_fr - fr_solar_aligned).rename("residual_load_fr_mw")

    # ------------------------------------------------------------------
    # 9. Gas + carbon composite pressure index (v2)
    #    Both TTF and the CO2 proxy are already D-1-aligned (point-in-time
    #    safe). Standardising with an expanding (not fixed-window) z-score
    #    keeps every row using only data through that row's own D-1 cutoff.
    # ------------------------------------------------------------------
    co2_hourly = _align_daily_to_d1(co2, idx, "co2_proxy_usd")
    ttf_z = _expanding_zscore(ttf_hourly)
    co2_z = _expanding_zscore(co2_hourly)
    gas_co2_pressure_index = (ttf_z + co2_z).rename("gas_co2_pressure_index")

    # ------------------------------------------------------------------
    # 10. Forecast Transfer Capacity (NTC) aggregates (v2)  [day-ahead
    #    document A61, same publication timing as load/wind/solar DA
    #    forecasts — known by gate closure, no shift needed]
    #
    #    ntc holds one column per (border, direction) that ENTSO-E actually
    #    publishes a bilateral Day-Ahead NTC for — NOT a fixed set of ~10
    #    borders (FR/BE/PL/NO_2/SE_4 never publish one; AT/CZ/DK_2 stop
    #    partway through 2019-2025 as flow-based capacity calculation
    #    takes over). See data/fetch_ntc.py docstring for the full
    #    per-border timeline.
    #
    #    ntc_import_capacity_mw / ntc_export_capacity_mw sum WHICHEVER
    #    borders are live at each hour (.sum(axis=1) skips NaN; an
    #    all-missing row sums to 0 — never happens here since CH/DK_1
    #    cover the full 2019-2025 span). This means the aggregate's
    #    border composition — and therefore its level — can shift for a
    #    reporting/methodology reason (e.g. CZ dropping out in mid-2022),
    #    not a real change in Germany's physical interconnection. Flagged
    #    here and in qa.py / the report rather than backfilled with an
    #    estimate for the missing borders.
    # ------------------------------------------------------------------
    ntc_aligned = ntc.reindex(idx)
    import_cols = [c for c in ntc_aligned.columns if c.endswith("_to_DE_LU_mw")]
    export_cols = [c for c in ntc_aligned.columns if c.startswith("ntc_DE_LU_to_")]
    ntc_import_capacity = ntc_aligned[import_cols].sum(axis=1).rename("ntc_import_capacity_mw")
    ntc_export_capacity = ntc_aligned[export_cols].sum(axis=1).rename("ntc_export_capacity_mw")
    # Sign convention: positive = DE-LU has more capacity to EXPORT than
    # import at that hour (net export-capable); negative = net import-capable.
    ntc_net_transfer = (ntc_export_capacity - ntc_import_capacity).rename("ntc_net_transfer_capacity_mw")

    # ------------------------------------------------------------------
    # 11. Neighbor bidding zone (FR/NL/BE/PL/CZ) price lags + residual load
    #    (v2 round 3, data/fetch_neighbors.py)
    #
    #    Price lags: D-1/D-7 same-hour REALISED prices for each neighbor —
    #    identical point-in-time justification as DE's own price_lag_24h/
    #    168h (known well before D's gate closure).
    #
    #    Neighbor day-ahead prices stop at 2025-09-29/30 in this build
    #    environment (a live-API data-window boundary — see
    #    data/fetch_neighbors.py / qa.py §9). Forward-filling BEFORE
    #    shifting keeps price lags defined through the rest of the Test
    #    year instead of truncating it; the cost is that the last ~3
    #    months of each neighbor's price lag carries a stale (sticky)
    #    value rather than a fresh one — flagged, not hidden.
    #
    #    Residual load: load - wind - solar wherever both are published.
    #    CZ has no day-ahead wind forecast at all (any year), so
    #    residual_load_cz_mw = load - solar only. PL's solar forecast
    #    doesn't start until 2020-04; missing PL solar is filled with 0
    #    (genuine near-zero capacity pre-buildout, same precedent as FR's
    #    pre-2022 offshore wind), not left as NaN. NL/PL wind have a
    #    handful of short gaps (24h each) left as NaN — caught by the
    #    validity mask below, same as DE's own small gaps.
    # ------------------------------------------------------------------
    # ffill() only ever propagates a PAST value forward — it cannot pull
    # information from a later timestamp — so combined with .shift(), these
    # lags are point-in-time safe by construction, the same guarantee
    # assert_no_lookahead() verifies at runtime for the DE price lags below.
    neighbor_prices_aligned = neighbor_prices.reindex(idx).ffill()
    neighbor_lags = {}
    for zone in ("fr", "nl", "be", "pl", "cz"):
        zone_price = neighbor_prices_aligned[f"price_{zone}_eur_mwh"]
        neighbor_lags[f"price_lag_24h_{zone}"] = zone_price.shift(24)
        neighbor_lags[f"price_lag_168h_{zone}"] = zone_price.shift(168)
    neighbor_lag_df = pd.DataFrame(neighbor_lags, index=idx)

    neighbor_load_aligned = neighbor_load.reindex(idx)
    neighbor_ws_aligned = neighbor_ws.reindex(idx)
    pl_solar_filled = neighbor_ws_aligned["solar_forecast_pl_mw"].fillna(0.0)

    residual_load_nl = (
        neighbor_load_aligned["load_forecast_nl_mw"]
        - neighbor_ws_aligned["wind_forecast_nl_mw"]
        - neighbor_ws_aligned["solar_forecast_nl_mw"]
    ).rename("residual_load_nl_mw")
    residual_load_be = (
        neighbor_load_aligned["load_forecast_be_mw"]
        - neighbor_ws_aligned["wind_forecast_be_mw"]
        - neighbor_ws_aligned["solar_forecast_be_mw"]
    ).rename("residual_load_be_mw")
    residual_load_pl = (
        neighbor_load_aligned["load_forecast_pl_mw"]
        - neighbor_ws_aligned["wind_forecast_pl_mw"]
        - pl_solar_filled
    ).rename("residual_load_pl_mw")
    residual_load_cz = (
        neighbor_load_aligned["load_forecast_cz_mw"]
        - neighbor_ws_aligned["solar_forecast_cz_mw"]
    ).rename("residual_load_cz_mw")

    # ------------------------------------------------------------------
    # 12. Assemble feature matrix
    # ------------------------------------------------------------------
    X = pd.concat(
        [
            # Calendar
            hour, dow, month,
            hour_sin, hour_cos,
            dow_sin, dow_cos,
            month_sin, month_cos,
            is_weekend, is_holiday,
            # DA forecasts
            load.rename("load_forecast_mw"),
            wind.rename("wind_forecast_mw"),
            solar.rename("solar_forecast_mw"),
            residual_load,
            # Merit-order terms
            resid_sq,
            resid_high,
            # Lags
            lag_24h,
            lag_168h,
            # Rolling
            rolling_mean,
            rolling_std,
            # Gas anchor
            ttf_hourly,
            # Gas x merit-order interaction (A3, optional)
            ttf_resid_interaction,
            # Rolling price stats (v2)
            rolling_price_mean,
            rolling_price_std,
            # FR cross-border (v2)
            load_fr.rename("load_forecast_fr_mw"),
            wind_fr.rename("wind_forecast_fr_mw"),
            residual_load_fr,
            # Gas + carbon composite (v2)
            co2_hourly,
            gas_co2_pressure_index,
            # Forecast Transfer Capacity aggregates (v2)
            ntc_import_capacity,
            ntc_export_capacity,
            ntc_net_transfer,
            # Neighbor zone price lags (v2 round 3)
            neighbor_lag_df,
            # Neighbor zone residual load (v2 round 3)
            residual_load_nl,
            residual_load_be,
            residual_load_pl,
            residual_load_cz,
        ],
        axis=1,
    )
    y = prices.copy()

    # ------------------------------------------------------------------
    # 12. Point-in-time firewall assertion — run on FULL matrix before
    #    filtering so y is the complete unmasked series and y.shift(24)
    #    correctly reconstructs the expected lag by calendar position.
    # ------------------------------------------------------------------
    assert_no_lookahead(X, y)

    # ------------------------------------------------------------------
    # 13. Validity mask — exclude rows with NaN in any critical feature
    #    (long-gap load rows + warm-up period for lags/rolling)
    # ------------------------------------------------------------------
    critical = [
        "load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw",
        "price_lag_24h", "price_lag_168h",
        "rolling_resid_mean_7d", "rolling_resid_std_7d",
        "rolling_price_mean_7d", "rolling_price_std_7d",
        "ttf_eur_mwh",
        "load_forecast_fr_mw", "wind_forecast_fr_mw", "residual_load_fr_mw",
        "co2_proxy_usd", "gas_co2_pressure_index",
        # v2 round 3 — neighbor price lags (forward-filled past the
        # 2025-09-30 data-window boundary, see §11 above, so these are
        # only NaN during the genuine warm-up period) + residual loads.
        "price_lag_24h_fr", "price_lag_168h_fr",
        "price_lag_24h_nl", "price_lag_168h_nl",
        "price_lag_24h_be", "price_lag_168h_be",
        "price_lag_24h_pl", "price_lag_168h_pl",
        "price_lag_24h_cz", "price_lag_168h_cz",
        "residual_load_nl_mw", "residual_load_be_mw",
        "residual_load_pl_mw", "residual_load_cz_mw",
    ]
    valid = X[critical].notna().all(axis=1) & y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    n_dropped = (~valid).sum()
    print(f"build_features: {len(X):,} valid rows | {n_dropped} dropped (warm-up + long gaps)")
    print(f"  Feature columns ({len(X.columns)}): {X.columns.tolist()}")
    print(f"  Date range: {X.index[0]} → {X.index[-1]}")

    return X, y


if __name__ == "__main__":
    X, y = build_features()
    print(f"\nFeature matrix: {X.shape}")
    print(f"Target stats: mean={y.mean():.2f}, std={y.std():.2f}, "
          f"min={y.min():.2f}, max={y.max():.2f}")
    print(f"\nNaN per feature:\n{X.isna().sum()[X.isna().sum() > 0]}")
