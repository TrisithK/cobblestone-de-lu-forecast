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
    print(f"  Direction: {curve_view['direction']}  |  Conviction: {curve_view['conviction']}")

    print("\n[Step 8b] Hourly/block tradable DA view …")
    hourly_view = build_hourly_block_view(backtest_results)

    print("\n[Step 9] LLM trader commentary …")
    commentary_result = generate_commentary(X, y, backtest_results, curve_view)
    c = commentary_result["validated_output"]
    print(f"  Direction: {c['direction']} | Conviction: {c['conviction']}")
    print(f"  Drivers: {c['drivers']}")

    print("\n[Step 10] Static morning desk note …")
    build_morning_note(curve_view, hourly_view, commentary_result)


if __name__ == "__main__":
    main()
