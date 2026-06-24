"""
main.py — Single entry point for the DE-LU day-ahead price forecasting pipeline.

Usage:
    python main.py

Runs end-to-end:
  1. QA on committed data snapshot (data/*.parquet)
  2. Feature engineering with point-in-time firewall
  3. Walk-forward backtest (baselines + Ridge + LightGBM)
  4. OOS predictions → predictions.csv
  5. Prompt-curve translation
  6. LLM commentary (from cache; no API key required for reproduction)

Set ENTSOE_API_KEY + ANTHROPIC_API_KEY in .env only if re-fetching data or
regenerating the LLM commentary from scratch.
"""

import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.qa import run_qa
from src.features import build_features
from src.models import plot_feature_importance_lgbm, plot_merit_order_tree
from src.validation import run_validation, write_oos_predictions
from src.prompt_curve import build_hourly_block_view, translate_curve
from src.llm_commentary import generate_commentary
from src.morning_note import build_morning_note
from src.dashboard import build_dashboard


def main():
    print("=" * 60)
    print("DE-LU Day-Ahead Price Forecasting Pipeline")
    print("Trisith Kittisriswai  |  trisithworld@gmail.com")
    print("=" * 60)

    print("\n[Step 3] Running QA …")
    clean = run_qa()
    print(f"  QA complete. Remaining NaN: "
          f"prices={clean['prices'].isna().sum()}, "
          f"load={clean['load'].isna().sum()}, "
          f"wind={clean['wind_forecast_mw'].isna().sum()}, "
          f"solar={clean['solar_forecast_mw'].isna().sum()}")

    print("\n[Step 4] Building features …")
    X, y = build_features(clean)
    print(f"  Feature matrix: {X.shape}  |  target rows: {len(y)}")

    print("\n[Step 5] Fitting depth-3 merit-order tree and saving figures …")
    plot_merit_order_tree(X, y)
    print("  Figures saved to figures/")

    print("\n[Step 5b] LightGBM feature importance …")
    plot_feature_importance_lgbm(X, y)

    print("\n[Step 6] Walk-forward validation …")
    backtest_results = run_validation(X, y)
    print(f"  Backtest complete: {len(backtest_results):,} predictions")

    print("\n[Step 7] Writing OOS predictions …")
    write_oos_predictions(backtest_results)

    print("\n[Step 8] Prompt-curve translation …")
    curve_view = translate_curve()
    print(f"  Baseload avg: {curve_view['baseload_avg_eur']} EUR/MWh  |  "
          f"Peak avg: {curve_view['peak_avg_eur']} EUR/MWh")

    print("\n[Step 8b] Hourly/block tradable DA view …")
    hourly_view = build_hourly_block_view(backtest_results)

    print("\n[Step 9] LLM trader commentary …")
    commentary_result = generate_commentary(X, y, backtest_results)
    c = commentary_result["validated_output"]
    fo = commentary_result["fact_object"]
    print(f"  Baseload call: {fo['baseload_direction']} (conviction {fo['baseload_conviction']})")
    print(f"  Tally: {fo['tally_sell']} SELL / {fo['tally_flat']} FLAT / {fo['tally_buy']} BUY")
    print(f"  Bullish drivers: {c['drivers_bullish']}")
    print(f"  Bearish drivers: {c['drivers_bearish']}")

    print("\n[Step 10] Static morning desk note …")
    build_morning_note(curve_view, hourly_view, commentary_result)

    print("\n[Step 11] Fair-value vs. EXAA dashboard …")
    dashboard_data = build_dashboard()
    print(f"  Delivery day: {dashboard_data['meta']['delivery_date']}  |  "
          f"Tally: {dashboard_data['tally']['sell']} SELL / "
          f"{dashboard_data['tally']['flat']} FLAT / {dashboard_data['tally']['buy']} BUY")


if __name__ == "__main__":
    main()
