"""
config.py — Shared constants for the DE-LU pipeline.
"""
import pandas as pd

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

# ---------------------------------------------------------------------------
# OOS window — the entire Test period (2025-01-01 -> 2025-12-31, 8,760 hours).
# CLAUDE.md §6 suggests "~2-4 weeks" as the predictions.csv deliverable; this
# build widens OOS to the full Test year on request, so predictions.csv and
# the prompt-curve / hourly-block figures are backed by the same full-year
# walk-forward backtest reported in validation_metrics.md, not a 24-day slice
# of it. Documented as a deliberate deviation from the brief in full_report.pdf §6.
# ---------------------------------------------------------------------------
OOS_START = TEST_START
OOS_END   = TEST_END

# Calibration window type (REVISION_PLAN.md A2) — chosen empirically by
# comparing expanding vs. rolling training windows on the VALIDATION period
# (2024 walk-forward MAE, Ridge). See outputs/window_tuning.md for the numbers.
# "rolling" trains on only the trailing WINDOW_DAYS; "expanding" uses all
# history before the prediction day. Flip this to re-run the comparison.
WINDOW_TYPE = "expanding"  # "expanding" | "rolling" — [v2 round 2] flips back to
                            # "expanding" after adding the NTC features (15.04 < rolling's
                            # 15.25, Validation 2024, Ridge). v2 round 1 (FR+CO2 only) had
                            # flipped it to "rolling" (15.14 < 15.40); each feature-set change
                            # re-triggers this comparison — see outputs/window_tuning.md for
                            # the numbers actually used to pick the current setting.
WINDOW_DAYS = 728           # trailing window length in days, used only if WINDOW_TYPE == "rolling"

# ---------------------------------------------------------------------------
# Prompt-curve figure readability window (CLAUDE.md §7 / full_report.pdf §7).
# figures/prompt_curve.png and figures/hourly_block_view.png plot a Q4 2025
# slice of the full-year OOS window for readability — the underlying
# aggregate stats in outputs/*.md and full_report.pdf remain full-OOS-year.
# ---------------------------------------------------------------------------
Q4_START = pd.Timestamp("2025-10-01 00:00:00", tz="Europe/Berlin")
Q4_END   = pd.Timestamp("2025-12-31 23:00:00", tz="Europe/Berlin")

# ---------------------------------------------------------------------------
# Residual load hinge threshold (80th percentile, computed on pre-OOS history)
# Used in features.py to capture the scarcity / top-of-stack pricing zone.
# ---------------------------------------------------------------------------
RESID_HIGH_MW = 45_063.0   # MW  (p80 of full pre-OOS residual load distribution)

# ---------------------------------------------------------------------------
# Random seeds — all models use these for reproducibility (CLAUDE.md §11)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# [v2 round 5] Dashboard per-hour / per-block decision threshold.
# direction_h = FLAT if conviction_h < FLAT_CONVICTION, else SELL/BUY by the
# sign of basis_h, where conviction_h = |basis_h| / MAE_for_hour_h. This is
# arithmetic, computed in src/dashboard.py — not an LLM judgement. Tunable:
# raising it makes the dashboard call FLAT more often (more conservative,
# requires a bigger edge relative to the model's own demonstrated error
# before committing to a side); lowering it calls a side more readily.
# ---------------------------------------------------------------------------
FLAT_CONVICTION = 0.25
