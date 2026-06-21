"""
config.py — Shared constants for the DE-LU pipeline.
"""
import pandas as pd

# ---------------------------------------------------------------------------
# OOS window (CLAUDE.md §6: most recent ~2-4 weeks of fully-available data)
# Data ends 2025-12-31; use last 24 days (3.4 weeks) as OOS.
# ---------------------------------------------------------------------------
OOS_START = pd.Timestamp("2025-12-08 00:00:00", tz="Europe/Berlin")
OOS_END   = pd.Timestamp("2025-12-31 23:00:00", tz="Europe/Berlin")

# ---------------------------------------------------------------------------
# Three-way temporal split (REVISION_PLAN.md A1).
# These dates define *roles*, not frozen training sets: walk-forward training
# always uses all valid rows strictly before the day being predicted, so 2024
# data still trains every 2025 prediction (no training freeze at 2023).
#   Validation : tune + lock all choices here (window type, hyperparameters).
#   Test       : full-year OOS backtest, untouched until the very end.
# ---------------------------------------------------------------------------
VALIDATION_START = pd.Timestamp("2024-01-01 00:00:00", tz="Europe/Berlin")
VALIDATION_END   = pd.Timestamp("2024-12-31 23:00:00", tz="Europe/Berlin")
TEST_START       = pd.Timestamp("2025-01-01 00:00:00", tz="Europe/Berlin")
TEST_END         = pd.Timestamp("2025-12-31 23:00:00", tz="Europe/Berlin")

# Calibration window type (REVISION_PLAN.md A2) — chosen empirically by
# comparing expanding vs. rolling training windows on the VALIDATION period
# (2024 walk-forward MAE, Ridge). See outputs/window_tuning.md for the numbers.
# "rolling" trains on only the trailing WINDOW_DAYS; "expanding" uses all
# history before the prediction day. Flip this to re-run the comparison.
WINDOW_TYPE = "expanding"   # "expanding" | "rolling"
WINDOW_DAYS = 728           # trailing window length in days, used only if WINDOW_TYPE == "rolling"

# ---------------------------------------------------------------------------
# EEX DE prompt-curve reference (CLAUDE.md §7)
# Front-month: ICE ENDEX German Power Financial Base Futures, January 2026
#   delivery (GABF2026). Contract settled 2026-01-30; settlement price range
#   confirmed as 102.45–104.09 EUR/MWh (mid: 103.99) from TradingView /
#   ICE ENDEX public data. Used here as the nearest publicly verifiable
#   reference for the January 2026 front-month around 2025-12-05.
#   Note: the exact EEX daily settlement on 2025-12-05 is not freely available
#   (requires EEX DataSource subscription); 103.99 is the final settlement
#   and represents the realised January 2026 EPEX average — the best public proxy.
# Front-week: EEX Power Week Future (Dec 8-14 2025) daily settlement is not
#   publicly available. Retained as illustrative; early-Dec spot context
#   (~110 EUR/MWh on EPEX in the week before Dec 8) suggests ~105–110 EUR/MWh
#   was plausible for the week future.
# These are NEVER fed into the model — used only for the curve translation step.
# ---------------------------------------------------------------------------
EEX_FRONTMONTH_DATE          = "2026-01-30"          # contract expiry / settlement date
EEX_FRONTMONTH_BASELOAD_EUR  = 103.99                # EUR/MWh  ICE ENDEX GABF2026 final settlement
EEX_FRONTMONTH_IS_ILLUSTRATIVE = False                # real, dated, sourced settlement print

EEX_FRONTWEEK_DATE           = "2025-12-05"          # reference date (illustrative)
EEX_FRONTWEEK_BASELOAD_EUR   = 107.50                # EUR/MWh  estimated from early-Dec spot context
# Checked (2026-06-21): EEX Phelix DE Power Week Future settlements are not
# published anywhere free/public (EEX market-data pages only show a rolling
# 45-day window; full history requires an EEX Group DataSource subscription).
# No real print could be sourced — this stays a fundamentals proxy.
EEX_FRONTWEEK_IS_ILLUSTRATIVE = True                  # explicit flag — see REVISION_PLAN.md C3

EEX_SOURCE = (
    "Front-month: ICE ENDEX German Power Financial Base Futures (GABF2026), January 2026 "
    "delivery, final settlement 2026-01-30 (range 102.45–104.09, mid 103.99 EUR/MWh) — real, "
    "publicly sourced (TradingView / ICE ENDEX). "
    "Front-week: NOT a real settlement — estimated from early-December EPEX spot context "
    "(~110 EUR/MWh); the actual EEX week-future print requires an EEX DataSource subscription "
    "and could not be sourced publicly. Treated as illustrative throughout "
    "(see EEX_FRONTWEEK_IS_ILLUSTRATIVE)."
)

# ---------------------------------------------------------------------------
# Residual load hinge threshold (80th percentile, computed on pre-OOS history)
# Used in features.py to capture the scarcity / top-of-stack pricing zone.
# ---------------------------------------------------------------------------
RESID_HIGH_MW = 45_063.0   # MW  (p80 of full pre-OOS residual load distribution)

# ---------------------------------------------------------------------------
# Random seeds — all models use these for reproducibility (CLAUDE.md §11)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
