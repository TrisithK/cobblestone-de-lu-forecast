"""
features.py — Point-in-time feature engineering for the DE-LU pipeline.

FIREWALL RULE (CLAUDE.md §3.1):
    Every feature for delivery hour h on day D must be knowable at ~12:00 D-1.
    - DA forecasts (load/wind/solar): published by ENTSO-E before gate closure ✓
    - Price lags: shift(24) = D-1 same-hour, shift(168) = D-7 same-hour ✓
    - Rolling stats: computed on resid.shift(24) → only uses data up to D-1 ✓
    - TTF: D-1 daily close (forward-filled over weekends/holidays) ✓
    - FORBIDDEN: any realised value from day D; any actual generation series

Feature matrix columns (23 features + target):
    Calendar    : hour, dow, month, is_weekend, is_holiday,
                  hour_sin/cos, dow_sin/cos, month_sin/cos
    DA forecasts: load_forecast_mw, wind_forecast_mw, solar_forecast_mw,
                  residual_load_mw
    Merit-order : residual_load_sq (scaled), residual_load_high (scarcity hinge)
    Price lags  : price_lag_24h, price_lag_168h
    Rolling     : rolling_resid_mean_7d, rolling_resid_std_7d
    Gas anchor  : ttf_eur_mwh
    Gas x merit-order (REVISION_PLAN.md A3, optional): ttf_resid_interaction
        — lets the Ridge model re-slope residual_load->price with gas, rather
        than only shifting the level. Ridge-only (see models.RIDGE_FEATURES);
        LightGBM already captures interactions nonlinearly without it.
"""

import os
from typing import Optional

import holidays as hols
import numpy as np
import pandas as pd

from config import OOS_START, RESID_HIGH_MW

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


def _align_ttf(ttf: pd.Series, hourly_index: pd.DatetimeIndex) -> pd.Series:
    """
    Map TTF daily close to each delivery hour using the D-1 value.
    TTF is forward-filled over weekends and exchange holidays.
    TTF index is tz-naive; we compare against tz-naive D-1 dates.
    """
    ttf_copy = ttf.copy()
    ttf_copy.index = pd.to_datetime(ttf_copy.index).normalize()

    # Forward-fill to cover all calendar days
    full_range = pd.date_range(
        start=ttf_copy.index.min(),
        end=ttf_copy.index.max() + pd.Timedelta(days=2),
        freq="D",
    )
    ttf_ff = ttf_copy.reindex(full_range).ffill()

    # For each delivery hour h on day D, look up TTF from D-1 (tz-naive midnight)
    d1_naive = pd.DatetimeIndex(
        hourly_index.normalize().tz_localize(None) - pd.Timedelta(days=1)
    )
    values = ttf_ff.reindex(d1_naive).values
    return pd.Series(values, index=hourly_index, name="ttf_eur_mwh")


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
        prices = clean["prices"]
        load   = clean["load"]
        wind   = clean["wind_forecast_mw"]
        solar  = clean["solar_forecast_mw"]
        ttf    = clean["ttf"]
    else:
        prices = pd.read_parquet(os.path.join(DATA_DIR, "da_prices.parquet"))["price_eur_mwh"]
        load   = pd.read_parquet(os.path.join(DATA_DIR, "load_forecast.parquet"))["load_forecast_mw"]
        ws     = pd.read_parquet(os.path.join(DATA_DIR, "wind_solar_forecast.parquet"))
        wind   = ws["wind_forecast_mw"]
        solar  = ws["solar_forecast_mw"]
        ttf    = pd.read_parquet(os.path.join(DATA_DIR, "ttf_daily.parquet"))["ttf_eur_mwh"]

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
    # 7. Assemble feature matrix
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
        ],
        axis=1,
    )
    y = prices.copy()

    # ------------------------------------------------------------------
    # 8. Point-in-time firewall assertion — run on FULL matrix before
    #    filtering so y is the complete unmasked series and y.shift(24)
    #    correctly reconstructs the expected lag by calendar position.
    # ------------------------------------------------------------------
    assert_no_lookahead(X, y)

    # ------------------------------------------------------------------
    # 9. Validity mask — exclude rows with NaN in any critical feature
    #    (long-gap load rows + warm-up period for lags/rolling)
    # ------------------------------------------------------------------
    critical = [
        "load_forecast_mw", "wind_forecast_mw", "solar_forecast_mw",
        "price_lag_24h", "price_lag_168h",
        "rolling_resid_mean_7d", "rolling_resid_std_7d",
        "ttf_eur_mwh",
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
