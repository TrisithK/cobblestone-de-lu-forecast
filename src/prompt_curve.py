"""
prompt_curve.py — Prompt-curve translation (CLAUDE.md §7).

Translates the OOS hourly Ridge forecasts into a model fair-value view:
  - Full-OOS-year baseload / peak / off-peak aggregates (figures + report).
  - Hourly/block tradable shape — which hours/blocks screen rich or cheap
    against that day's own baseload average.

No EEX forward print is used here (none could be sourced for the OOS delivery
dates — see report.pdf §7). [v2 round 4]: the directional trading call
(direction / conviction / size) lives in llm_commentary.py and is now
computed primarily against a real, sourced pre-auction reference instead —
the EXAA (Sequence 2) day-ahead auction price for the same delivery day,
which settles earlier the same day (~10:15 CET D-1) than the EPEX auction
this model forecasts (~12:00 CET D-1). The v1-v2 self-referential basis
(tomorrow's forecast vs. the trailing realised baseload, D-1/D-7 actuals
already used as model features) is retained as secondary context, not the
primary driver. See report.pdf §7 for the rationale and CLAUDE.md §7.

figures/prompt_curve.png and figures/hourly_block_view.png restrict their
*plotted* window to Q4 2025 (config.Q4_START/Q4_END) purely for readability
— the full-OOS-year aggregate stats they annotate are unchanged.
"""

import os

import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OOS_END, OOS_START, Q4_END, Q4_START

_ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(_ROOT, "figures")
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")

_DE_HOLIDAYS = holidays.Germany()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_peak(ts: pd.Timestamp) -> bool:
    """EEX peak convention: hours 08-20 on weekdays, excl. German public holidays."""
    return (8 <= ts.hour <= 20) and (ts.dayofweek < 5) and (ts.date() not in _DE_HOLIDAYS)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def translate_curve(predictions_path: str = None) -> dict:
    """
    Load OOS predictions and compute baseload / peak / off-peak aggregates
    over the full OOS year.

    Also writes:
      figures/prompt_curve.png       (hourly + daily view, Q4 2025 only — readability)
      outputs/prompt_curve_view.md   (full-OOS-year aggregates)
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if predictions_path is None:
        predictions_path = os.path.join(_ROOT, "predictions.csv")

    # --- Load and parse ---
    preds = pd.read_csv(predictions_path)
    # utc=True first: the full Test year spans both CET (+01:00) and CEST (+02:00)
    # offsets, so parsing without utc=True yields mixed-offset objects, not a
    # DatetimeIndex.
    preds["datetime"] = pd.to_datetime(preds["datetime"], utc=True).dt.tz_convert("Europe/Berlin")
    preds = preds.set_index("datetime").sort_index()

    # --- Peak flag ---
    preds["is_peak"] = [_is_peak(ts) for ts in preds.index]
    preds["date"]    = preds.index.date

    # --- OOS-wide aggregates (full year) ---
    baseload_avg = float(preds["y_pred"].mean())
    peak_avg     = float(preds.loc[preds["is_peak"], "y_pred"].mean())
    offpeak_avg  = float(preds.loc[~preds["is_peak"], "y_pred"].mean())

    # --- Daily aggregates (full year; figure subsets to Q4 for display) ---
    daily_baseload = preds.groupby("date")["y_pred"].mean()
    daily_peak     = preds[preds["is_peak"]].groupby("date")["y_pred"].mean()

    invalidation_triggers = [
        (
            "TTF front-month moves >5 % overnight (gas repricing shifts the absolute price "
            "level across all hours; the model lags because it uses yesterday's TTF close)."
        ),
        (
            "Wind-power forecast revision >5 GW vs. the D-1 model run (residual load "
            "— the primary merit-order driver — would reprice the full day materially)."
        ),
        (
            "Unplanned nuclear / large thermal outage notified on REMIT "
            "(supply removal lifts scarcity hours disproportionately; model can't anticipate)."
        ),
        (
            "Demand surprise: cold snap or anomalous holiday-week consumption "
            "not captured in the load forecast (especially Christmas week)."
        ),
        (
            "[v2 round 4] EXAA print revises materially between its own auction "
            "(~10:15 CET D-1) and EPEX gate closure (~12:00 CET D-1) — the basis "
            "driving the trading call was set against the earlier EXAA read, so a "
            "fresh fundamentals move in that window would stale it."
        ),
    ]

    result = {
        "oos_start":        str(OOS_START.date()),
        "oos_end":          str(OOS_END.date()),
        "n_oos_hours":      int(len(preds)),
        "baseload_avg_eur": round(baseload_avg, 2),
        "peak_avg_eur":     round(peak_avg, 2),
        "offpeak_avg_eur":  round(offpeak_avg, 2),
        "invalidation_triggers": invalidation_triggers,
    }

    _plot_curve(preds, daily_baseload, daily_peak, result)
    _write_view_report(result)
    _print_summary(result)

    return result


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _plot_curve(
    preds: pd.DataFrame,
    daily_baseload: pd.Series,
    daily_peak: pd.Series,
    r: dict,
) -> None:
    q4 = preds.loc[(preds.index >= Q4_START) & (preds.index <= Q4_END)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ---- Panel 1: Hourly time series (Q4 2025 only — readability) ----------
    ax = axes[0]

    y_lo = q4["y_pred"].min() - 10
    y_hi = q4["y_pred"].max() + 10
    ax.fill_between(
        q4.index, y_lo, y_hi,
        where=q4["is_peak"].values,
        alpha=0.09, color="orange", step="post",
        label="Peak hours (08-20 weekday)",
    )

    ax.plot(q4.index, q4["y_pred"],
            color="#2166ac", lw=0.9, alpha=0.85, label="Ridge forecast (hourly)")
    ax.axhline(r["baseload_avg_eur"], color="#2166ac", lw=1.8, ls="--",
               label=f"Full-year OOS baseload avg: {r['baseload_avg_eur']:.1f} EUR/MWh")
    ax.axhline(r["peak_avg_eur"], color="darkorange", lw=1.8, ls="--",
               label=f"Full-year OOS peak avg: {r['peak_avg_eur']:.1f} EUR/MWh")
    ax.axhline(0, color="black", lw=0.6, ls=":")

    ax.set_xlim(Q4_START, Q4_END)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Price (EUR/MWh)", fontsize=11)
    ax.set_title(
        f"DE-LU Model Fair-Value View  |  Full OOS year: {r['oos_start']} → {r['oos_end']}  "
        f"|  Showing Q4 2025 (Oct–Dec) for readability\n"
        f"Full-year baseload avg: {r['baseload_avg_eur']:.1f}  |  "
        f"peak avg: {r['peak_avg_eur']:.1f}  |  off-peak avg: {r['offpeak_avg_eur']:.1f} EUR/MWh",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    # ---- Panel 2: Daily baseload bar chart (Q4 2025 only) ------------------
    ax2 = axes[1]
    daily_idx = pd.to_datetime(daily_baseload.index.astype(str)).tz_localize("Europe/Berlin")
    q4_daily_mask = (daily_idx >= Q4_START) & (daily_idx <= Q4_END)
    x_dates = daily_idx[q4_daily_mask]
    y_daily = daily_baseload.values[q4_daily_mask]

    ax2.bar(x_dates, y_daily, color="#4393c3", alpha=0.8, width=pd.Timedelta("20h"),
            label="Daily baseload forecast")

    peak_idx = pd.to_datetime(daily_peak.index.astype(str)).tz_localize("Europe/Berlin")
    q4_peak_mask = (peak_idx >= Q4_START) & (peak_idx <= Q4_END)
    if q4_peak_mask.any():
        ax2.plot(peak_idx[q4_peak_mask], daily_peak.values[q4_peak_mask], "o-",
                 color="darkorange", ms=5, lw=1.4, label="Daily peak forecast")

    ax2.axhline(0, color="black", lw=0.6, ls=":")
    ax2.set_xlim(Q4_START, Q4_END)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Price (EUR/MWh)", fontsize=11)
    ax2.set_title("Daily Baseload & Peak Forecasts — Q4 2025", fontsize=11)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(alpha=0.3)

    plt.tight_layout(pad=2.0)
    path = os.path.join(FIGURES_DIR, "prompt_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_view_report(r: dict) -> None:
    lines = [
        "# Model Fair-Value View — DE-LU Day-Ahead Power\n",
        f"**Forecast model:** Ridge (selected model, CLAUDE.md §6)  \n"
        f"**OOS window:** {r['oos_start']} → {r['oos_end']} ({r['n_oos_hours']} hourly predictions)  \n",
        "No EEX forward print is used in this view (none could be sourced for the OOS "
        "delivery dates). The directional trading call (direction / conviction / size) "
        "shown in the morning note is computed in `llm_commentary.py` primarily against "
        "the **EXAA (Sequence 2) day-ahead auction price** for the same delivery day — a "
        "real, observable pre-auction print that settles earlier the same day (~10:15 CET "
        "D-1) than the EPEX auction this model forecasts (~12:00 CET D-1). The earlier "
        "self-referential basis (forecast vs. trailing D-1/D-7 realised baseload) is kept "
        "as secondary context. See report.pdf §7 / CLAUDE.md §7 for the rationale.\n",
        "## 1. Model Fair-Value Aggregates (Ridge Forecast, full OOS year)\n",
        "| Aggregate | EUR/MWh |",
        "|-----------|---------|",
        f"| Baseload average (all hours) | **{r['baseload_avg_eur']:.2f}** |",
        f"| Peak average (08-20 weekday) | **{r['peak_avg_eur']:.2f}** |",
        f"| Off-peak average             | **{r['offpeak_avg_eur']:.2f}** |",
        "",
        "> *`figures/prompt_curve.png` plots a Q4 2025 (Oct–Dec) slice of the hourly "
        "forecast for readability — the table above reflects the full OOS year.*\n",
        "## 2. Invalidation Triggers\n",
        "The fair-value level above should be re-evaluated if any of the following occur:\n",
    ]
    for t in r["invalidation_triggers"]:
        lines.append(f"- {t}")
    lines += [
        "",
        "> *Aggregates here feed the morning note's Fair-Value Numbers table "
        "(`src/morning_note.py`). The trading view itself — direction, conviction, "
        "size — is computed directly from the model's own backtest history in "
        "`src/llm_commentary.py` and fed into the LLM commentary engine (Step 9) as "
        "part of the grounding fact-object; the LLM originates no numbers.*",
    ]

    path = os.path.join(OUTPUTS_DIR, "prompt_curve_view.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written → {path}")


def _print_summary(r: dict) -> None:
    print(
        f"\n  Prompt-curve summary (full OOS year):\n"
        f"    Baseload avg : {r['baseload_avg_eur']:7.2f} EUR/MWh\n"
        f"    Peak avg     : {r['peak_avg_eur']:7.2f} EUR/MWh\n"
        f"    Off-peak avg : {r['offpeak_avg_eur']:7.2f} EUR/MWh"
    )


# ---------------------------------------------------------------------------
# C1 — Hourly / block tradable DA view (REVISION_PLAN.md C1)
#
# Primary prompt-curve deliverable, independent of any external forward
# reference. Reads the model's own OOS hourly forecast as the fair-value
# curve and flags which individual hours/blocks screen rich or cheap
# *against that day's own baseload average* — directly tradable via hourly
# DA bids, peak/off-peak block products, or intraday.
# ---------------------------------------------------------------------------

RICH_CHEAP_THRESHOLD_EUR = 15.0  # EUR/MWh deviation from the day's own baseload avg


def build_hourly_block_view(results: pd.DataFrame,
                             rich_threshold_eur: float = RICH_CHEAP_THRESHOLD_EUR) -> dict:
    """
    Parameters
    ----------
    results : full backtest results DataFrame (has residual_load_mw, ridge, hour, ...).
              Filtered internally to the OOS window.
    """
    oos = results.loc[(results.index >= OOS_START) & (results.index <= OOS_END)].copy()
    oos["is_peak"] = [_is_peak(ts) for ts in oos.index]
    oos["date"] = oos.index.date

    daily_baseload = oos.groupby("date")["ridge"].transform("mean")
    oos["day_baseload_eur"] = daily_baseload
    oos["deviation_eur"] = oos["ridge"] - oos["day_baseload_eur"]
    oos["is_negative"] = oos["ridge"] < 0

    resid_p90 = oos["residual_load_mw"].quantile(0.90)
    oos["is_scarcity"] = oos["residual_load_mw"] >= resid_p90

    rich_hours = oos[oos["deviation_eur"] >= rich_threshold_eur].sort_values(
        "deviation_eur", ascending=False)
    cheap_hours = oos[oos["deviation_eur"] <= -rich_threshold_eur].sort_values("deviation_eur")
    negative_hours = oos[oos["is_negative"]].sort_values("ridge")
    scarcity_hours = oos[oos["is_scarcity"]].sort_values("ridge", ascending=False)

    peak_avg    = float(oos.loc[oos["is_peak"], "ridge"].mean())
    offpeak_avg = float(oos.loc[~oos["is_peak"], "ridge"].mean())
    peak_offpeak_spread = peak_avg - offpeak_avg

    result = {
        "oos_start": str(OOS_START.date()),
        "oos_end":   str(OOS_END.date()),
        "n_hours":   len(oos),
        "rich_threshold_eur": rich_threshold_eur,
        "n_rich":     int(len(rich_hours)),
        "n_cheap":    int(len(cheap_hours)),
        "n_negative": int(len(negative_hours)),
        "n_scarcity": int(len(scarcity_hours)),
        "peak_avg_eur":    round(peak_avg, 2),
        "offpeak_avg_eur": round(offpeak_avg, 2),
        "peak_offpeak_spread_eur": round(peak_offpeak_spread, 2),
        "top_rich":   [(ts.isoformat(), round(float(row["ridge"]), 2), round(float(row["deviation_eur"]), 2))
                        for ts, row in rich_hours.head(10).iterrows()],
        "top_cheap":  [(ts.isoformat(), round(float(row["ridge"]), 2), round(float(row["deviation_eur"]), 2))
                        for ts, row in cheap_hours.head(10).iterrows()],
        "top_negative": [(ts.isoformat(), round(float(row["ridge"]), 2))
                          for ts, row in negative_hours.head(10).iterrows()],
        "guidance": (
            f"Peak screens {peak_offpeak_spread:+.1f} EUR/MWh above off-peak over the OOS window — "
            f"that spread is itself a tradable block product (peak/off-peak swap or DA block bid). "
            f"{len(rich_hours)} individual hours screen >{rich_threshold_eur:.0f} EUR/MWh rich vs. "
            f"their own day's baseload (mostly evening peak / scarcity hours) — candidates for "
            f"selling that specific hour or block in the DA auction / intraday rather than the whole day. "
            f"{len(cheap_hours)} hours (including {len(negative_hours)} negative-price hours) screen "
            f"deeply cheap — candidates for buying / shifting flexible load (storage charging, "
            f"demand response) into those hours rather than paying the day's average. "
            f"{len(scarcity_hours)} hours sit in the top decile of residual load (scarcity pricing) — "
            "these are where the model's forecast error is largest (see by-regime MAE) and where a "
            "wind-forecast miss or outage would do the most damage to the call."
        ),
    }

    _plot_hourly_block_view(oos, result)
    _write_hourly_block_report(oos, result)
    print(
        f"\n  Hourly/block view: {result['n_rich']} rich hours, {result['n_cheap']} cheap hours "
        f"({result['n_negative']} negative), {result['n_scarcity']} scarcity hours, "
        f"peak-offpeak spread {result['peak_offpeak_spread_eur']:+.1f} EUR/MWh"
    )
    return result


def _plot_hourly_block_view(oos: pd.DataFrame, r: dict) -> None:
    q4 = oos.loc[(oos.index >= Q4_START) & (oos.index <= Q4_END)]

    fig, ax = plt.subplots(figsize=(14, 6))

    y_lo = q4["ridge"].min() - 10
    y_hi = q4["ridge"].max() + 10
    ax.fill_between(
        q4.index, y_lo, y_hi,
        where=q4["is_peak"].values,
        alpha=0.09, color="orange", step="post",
        label="Peak hours (08-20 weekday)",
    )
    ax.plot(q4.index, q4["ridge"], color="#2166ac", lw=1.0, alpha=0.85,
            label="Ridge OOS forecast", zorder=3)
    ax.plot(q4.index, q4["day_baseload_eur"], color="black", lw=1.0, ls="--",
            alpha=0.6, label="Day's own baseload avg (fair value)", zorder=2)

    rich_mask = q4["deviation_eur"] >= r["rich_threshold_eur"]
    cheap_mask = (q4["deviation_eur"] <= -r["rich_threshold_eur"]) & ~q4["is_negative"]
    neg_mask = q4["is_negative"]
    scarcity_mask = q4["is_scarcity"]

    ax.scatter(q4.index[rich_mask], q4.loc[rich_mask, "ridge"],
               color="firebrick", s=28, zorder=5, label=f"Rich hour (>{r['rich_threshold_eur']:.0f} EUR/MWh)")
    ax.scatter(q4.index[cheap_mask], q4.loc[cheap_mask, "ridge"],
               color="seagreen", s=28, zorder=5, label=f"Cheap hour (<-{r['rich_threshold_eur']:.0f} EUR/MWh)")
    ax.scatter(q4.index[neg_mask], q4.loc[neg_mask, "ridge"],
               color="purple", s=28, marker="v", zorder=6, label="Negative-price hour")
    ax.scatter(q4.index[scarcity_mask], q4.loc[scarcity_mask, "ridge"],
               facecolors="none", edgecolors="black", s=70, marker="o", zorder=4,
               label="Scarcity hour (top decile residual load)")

    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlim(Q4_START, Q4_END)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Price (EUR/MWh)", fontsize=11)
    ax.set_title(
        f"Hourly/Block Tradable DA View  |  Full OOS year: {r['oos_start']} → {r['oos_end']}  "
        f"|  Showing Q4 2025 for readability\n"
        f"Full-year stats — peak-offpeak spread: {r['peak_offpeak_spread_eur']:+.1f} EUR/MWh  |  "
        f"{r['n_rich']} rich / {r['n_cheap']} cheap / {r['n_negative']} negative / "
        f"{r['n_scarcity']} scarcity hours",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    path = os.path.join(FIGURES_DIR, "hourly_block_view.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def _write_hourly_block_report(oos: pd.DataFrame, r: dict) -> None:
    lines = [
        "# Hourly/Block Tradable DA View — DE-LU (REVISION_PLAN.md C1)\n",
        f"**OOS window:** {r['oos_start']} → {r['oos_end']} ({r['n_hours']} hourly predictions)  \n",
        "This view reads the model's own OOS hourly forecast as the fair-value curve — "
        "it does not depend on any external forward reference. Each hour is flagged "
        "against **that day's own baseload average**, so the signal is purely about "
        "shape: which individual hours or blocks screen rich/cheap *within* the day, "
        "tradable via hourly DA bids, peak/off-peak block products, or intraday. "
        "(`figures/hourly_block_view.png` plots a Q4 2025 slice for readability — "
        "the stats below are full OOS year.)\n",
        "## 1. Shape Summary\n",
        "| Metric | Value |",
        "|---|---:|",
        f"| Peak average | {r['peak_avg_eur']:.2f} EUR/MWh |",
        f"| Off-peak average | {r['offpeak_avg_eur']:.2f} EUR/MWh |",
        f"| Peak − off-peak spread | **{r['peak_offpeak_spread_eur']:+.2f} EUR/MWh** |",
        f"| Rich hours (> +{r['rich_threshold_eur']:.0f} EUR/MWh vs. day avg) | {r['n_rich']} |",
        f"| Cheap hours (< −{r['rich_threshold_eur']:.0f} EUR/MWh vs. day avg) | {r['n_cheap']} |",
        f"| ...of which negative-price | {r['n_negative']} |",
        f"| Scarcity hours (top decile residual load) | {r['n_scarcity']} |",
        "",
        "## 2. Top Rich Hours (sell candidates)\n",
        "| Datetime | Price (EUR/MWh) | Deviation vs. day avg |",
        "|---|---:|---:|",
    ]
    for ts, price, dev in r["top_rich"]:
        lines.append(f"| {ts} | {price:.2f} | {dev:+.2f} |")
    lines += [
        "",
        "## 3. Top Cheap Hours (buy / load-shift candidates)\n",
        "| Datetime | Price (EUR/MWh) | Deviation vs. day avg |",
        "|---|---:|---:|",
    ]
    for ts, price, dev in r["top_cheap"]:
        lines.append(f"| {ts} | {price:.2f} | {dev:+.2f} |")
    lines += [
        "",
        "## 4. Negative-Price Hours\n",
        "| Datetime | Price (EUR/MWh) |",
        "|---|---:|",
    ]
    for ts, price in r["top_negative"]:
        lines.append(f"| {ts} | {price:.2f} |")
    lines += [
        "",
        "## 5. Trading Guidance\n",
        r["guidance"],
        "",
        "## 6. Invalidation Triggers\n",
        "These hourly/block calls are invalidated by the same fundamental shocks as the "
        "model's daily fair-value view (see `outputs/prompt_curve_view.md`): a "
        "wind-forecast revision >5 GW, a TTF/gas gap >5% overnight, an unplanned outage "
        "on REMIT, or a demand surprise all reprice the hourly shape, not just the daily "
        "level — re-check before acting on any single-hour call above.\n",
        "> *Figure: `figures/hourly_block_view.png`.*",
    ]
    path = os.path.join(OUTPUTS_DIR, "hourly_block_view.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written → {path}")
