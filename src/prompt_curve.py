"""
prompt_curve.py — Prompt-curve translation (CLAUDE.md §7).

Translates the OOS hourly Ridge forecasts into a relative-value basis view
vs. the pinned EEX DE front-month / front-week settlement.

CLAUDE.md §7 rules enforced here:
  - Do NOT roll a 1-day forecast 30 days forward.
  - Aggregate forecasted hours into baseload / peak daily averages.
  - Compare vs. the pinned EEX settlement (date + source in config.py).
  - Express a directional lean with explicit invalidation triggers.
  - Add the risk-premium caveat (forward ≠ E[spot]).
"""

import os

import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    EEX_FRONTMONTH_BASELOAD_EUR,
    EEX_FRONTMONTH_DATE,
    EEX_FRONTMONTH_IS_ILLUSTRATIVE,
    EEX_FRONTWEEK_BASELOAD_EUR,
    EEX_FRONTWEEK_DATE,
    EEX_FRONTWEEK_IS_ILLUSTRATIVE,
    EEX_SOURCE,
    OOS_END,
    OOS_START,
)

_ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(_ROOT, "figures")
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")

_DE_HOLIDAYS = holidays.Germany()

# Front-week delivery window — week starting the Monday after EEX settlement
# (Dec 8 is the first OOS day; this is the front-week reference period)
FRONTWEEK_START = pd.Timestamp("2025-12-08", tz="Europe/Berlin")
FRONTWEEK_END   = pd.Timestamp("2025-12-14 23:00:00", tz="Europe/Berlin")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_peak(ts: pd.Timestamp) -> bool:
    """EEX peak convention: hours 08-20 on weekdays, excl. German public holidays."""
    return (8 <= ts.hour <= 20) and (ts.dayofweek < 5) and (ts.date() not in _DE_HOLIDAYS)


_CONVICTION_RANK = ["LOW", "MODERATE", "HIGH"]
_SIZE_RANK = [
    "NO TRADE (basis inside estimation noise)",
    "QUARTER SIZE (1/4 normal prompt risk)",
    "HALF SIZE (1/2 normal prompt risk)",
    "FULL SIZE (max normal prompt risk)",
]


def _conviction(basis_eur: float) -> str:
    if abs(basis_eur) < 5:
        return "LOW"
    if abs(basis_eur) < 15:
        return "MODERATE"
    return "HIGH"


def _position_size(basis_eur: float) -> str:
    """
    Rough position size keyed to basis magnitude.
    Basis inside noise (<5 EUR/MWh) → no trade.
    5-10 → quarter size; 10-20 → half; >20 → full.
    """
    ab = abs(basis_eur)
    if ab < 5:
        return "NO TRADE (basis inside estimation noise)"
    if ab < 10:
        return "QUARTER SIZE (1/4 normal prompt risk)"
    if ab < 20:
        return "HALF SIZE (1/2 normal prompt risk)"
    return "FULL SIZE (max normal prompt risk)"


def _cap_for_illustrative_reference(conviction: str, position_size: str) -> tuple[str, str]:
    """
    C3 fix: when the EEX reference behind a call is an illustrative placeholder
    (no real sourced print), conviction is capped at MODERATE and size is capped
    at HALF — no HIGH-conviction / FULL-SIZE call may rest on an unsourced number.
    """
    capped_conviction = _CONVICTION_RANK[min(_CONVICTION_RANK.index(conviction), 1)]
    capped_size = _SIZE_RANK[min(_SIZE_RANK.index(position_size), 2)]
    return capped_conviction, capped_size


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def translate_curve(predictions_path: str = None) -> dict:
    """
    Load OOS predictions, compute baseload / peak aggregates, compare to the
    pinned EEX reference, and return a structured dict for the LLM commentary step.

    Also writes:
      figures/prompt_curve.png
      outputs/prompt_curve_view.md
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if predictions_path is None:
        predictions_path = os.path.join(_ROOT, "predictions.csv")

    # --- Load and parse ---
    preds = pd.read_csv(predictions_path)
    preds["datetime"] = pd.to_datetime(preds["datetime"]).dt.tz_convert("Europe/Berlin")
    preds = preds.set_index("datetime").sort_index()

    # Front-week drives direction/conviction/size below, so it's the flag that
    # matters here. C3 fix: this is now an explicit config flag, not a brittle
    # substring check on EEX_SOURCE (which never matched, so HIGH/FULL conviction
    # calls were silently resting on an unsourced placeholder).
    is_illustrative = EEX_FRONTWEEK_IS_ILLUSTRATIVE
    if is_illustrative:
        print(
            "  [WARN] EEX_FRONTWEEK_BASELOAD_EUR is an illustrative placeholder "
            "(no public EEX week-future print could be sourced — see config.py). "
            "Conviction/size are capped accordingly."
        )

    # --- Peak flag ---
    preds["is_peak"] = [_is_peak(ts) for ts in preds.index]
    preds["date"]    = preds.index.date

    # --- OOS-wide aggregates ---
    baseload_avg = float(preds["y_pred"].mean())
    peak_avg     = float(preds.loc[preds["is_peak"], "y_pred"].mean())
    offpeak_avg  = float(preds.loc[~preds["is_peak"], "y_pred"].mean())

    # --- Front-week aggregates (Dec 8-14 2025) ---
    fw_mask       = (preds.index >= FRONTWEEK_START) & (preds.index <= FRONTWEEK_END)
    fw_baseload   = float(preds.loc[fw_mask, "y_pred"].mean())
    fw_peak_mask  = fw_mask & preds["is_peak"]
    fw_peak       = float(preds.loc[fw_peak_mask, "y_pred"].mean()) if fw_peak_mask.any() else float("nan")

    # --- Daily aggregates ---
    daily_baseload = preds.groupby("date")["y_pred"].mean()
    daily_peak     = preds[preds["is_peak"]].groupby("date")["y_pred"].mean()

    # --- Basis calculations ---
    # Front-week: model week-1 baseload vs EEX front-week settlement (most comparable)
    basis_fw_baseload = fw_baseload - EEX_FRONTWEEK_BASELOAD_EUR
    # Front-month: full OOS baseload vs EEX front-month (cross-month curve shape)
    basis_fm_baseload = baseload_avg - EEX_FRONTMONTH_BASELOAD_EUR

    # --- Directional view — anchored on front-week basis (same delivery window) ---
    conviction    = _conviction(basis_fw_baseload)
    position_size = _position_size(basis_fw_baseload)
    if is_illustrative:
        conviction, position_size = _cap_for_illustrative_reference(conviction, position_size)

    if basis_fw_baseload > 5:
        direction = "LONG / BUY"
        view_text = (
            f"Model fair value for the front-week delivery window (Dec 8-14) is "
            f"{fw_baseload:.1f} EUR/MWh baseload, "
            f"{abs(basis_fw_baseload):.1f} EUR/MWh ABOVE the pinned EEX front-week "
            f"settlement ({EEX_FRONTWEEK_BASELOAD_EUR:.1f} EUR/MWh, {EEX_FRONTWEEK_DATE}). "
            "The prompt forward looks cheap vs. near-term model fair value. "
            f"Directional lean: long prompt (buy the front-week or spot equivalent). "
            f"Suggested size: {position_size}."
        )
    elif basis_fw_baseload < -5:
        direction = "SHORT / SELL"
        view_text = (
            f"Model fair value for the front-week delivery window (Dec 8-14) is "
            f"{fw_baseload:.1f} EUR/MWh baseload, "
            f"{abs(basis_fw_baseload):.1f} EUR/MWh BELOW the pinned EEX front-week "
            f"settlement ({EEX_FRONTWEEK_BASELOAD_EUR:.1f} EUR/MWh, {EEX_FRONTWEEK_DATE}). "
            "The prompt forward looks rich vs. near-term model fair value. "
            f"Directional lean: short prompt (sell the front-week or reduce long). "
            f"Suggested size: {position_size}."
        )
    else:
        direction = "NEUTRAL / FLAT"
        position_size = "NO TRADE (basis inside estimation noise)"
        view_text = (
            f"Model fair value for Dec 8-14 ({fw_baseload:.1f} EUR/MWh) is within "
            f"{abs(basis_fw_baseload):.1f} EUR/MWh of the EEX front-week settlement "
            f"({EEX_FRONTWEEK_BASELOAD_EUR:.1f} EUR/MWh) — inside estimation noise. "
            "No clear directional edge; hold flat."
        )

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
            "EEX settlement revises intraday — confirm vs. live screen before trading."
        ),
    ]

    risk_premium_note = (
        "The EEX forward price embeds a risk premium over E[spot]; it is not an "
        "unbiased expectation of the realised daily average. This basis view is a "
        "directional lean on relative value, not a claim that the forward will converge "
        "to the model forecast. Size positions accordingly."
    )

    result = {
        # Aggregates
        "oos_start":        str(OOS_START.date()),
        "oos_end":          str(OOS_END.date()),
        "baseload_avg_eur": round(baseload_avg, 2),
        "peak_avg_eur":     round(peak_avg, 2),
        "offpeak_avg_eur":  round(offpeak_avg, 2),
        # Front-week window
        "fw_start":         str(FRONTWEEK_START.date()),
        "fw_end":           str(FRONTWEEK_END.date()),
        "fw_baseload_eur":  round(fw_baseload, 2),
        "fw_peak_eur":      round(fw_peak, 2) if not np.isnan(fw_peak) else None,
        # EEX reference
        "eex_frontmonth_eur":  EEX_FRONTMONTH_BASELOAD_EUR,
        "eex_frontmonth_date": EEX_FRONTMONTH_DATE,
        "eex_frontweek_eur":   EEX_FRONTWEEK_BASELOAD_EUR,
        "eex_frontweek_date":  EEX_FRONTWEEK_DATE,
        "eex_source":          EEX_SOURCE,
        "eex_is_illustrative": is_illustrative,  # = front-week flag; drives conviction/size cap
        "eex_frontmonth_is_illustrative": EEX_FRONTMONTH_IS_ILLUSTRATIVE,
        "eex_frontweek_is_illustrative":  EEX_FRONTWEEK_IS_ILLUSTRATIVE,
        # Basis
        "basis_fw_baseload_eur": round(basis_fw_baseload, 2),
        "basis_fm_baseload_eur": round(basis_fm_baseload, 2),
        # View
        "direction":             direction,
        "conviction":            conviction,
        "position_size":         position_size,
        "view":                  view_text,
        "invalidation_triggers": invalidation_triggers,
        "risk_premium_note":     risk_premium_note,
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
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ---- Panel 1: Hourly time series ----------------------------------------
    ax = axes[0]

    # Peak-hour shading (fill_between — efficient, one call)
    y_lo = preds["y_pred"].min() - 10
    y_hi = preds["y_pred"].max() + 10
    ax.fill_between(
        preds.index, y_lo, y_hi,
        where=preds["is_peak"].values,
        alpha=0.09, color="orange", step="post",
        label="Peak hours (08-20 weekday)",
    )

    ax.plot(preds.index, preds["y_pred"],
            color="#2166ac", lw=0.9, alpha=0.85, label="Ridge forecast (hourly)")
    ax.axhline(r["baseload_avg_eur"], color="#2166ac", lw=1.8, ls="--",
               label=f"OOS baseload avg: {r['baseload_avg_eur']:.1f} EUR/MWh")
    ax.axhline(r["peak_avg_eur"], color="darkorange", lw=1.8, ls="--",
               label=f"OOS peak avg: {r['peak_avg_eur']:.1f} EUR/MWh")
    ax.axhline(r["eex_frontmonth_eur"], color="firebrick", lw=2.0, ls="-",
               label=(
                   f"EEX front-month ({r['eex_frontmonth_date']}): "
                   f"{r['eex_frontmonth_eur']:.1f} EUR/MWh  [indicative context only — month mismatch]"
                   + (" [ILLUSTRATIVE]" if r["eex_frontmonth_is_illustrative"] else " [real print]")
               ))
    ax.axhline(r["eex_frontweek_eur"], color="darkred", lw=1.6, ls=":",
               label=(
                   f"EEX front-week ({r['eex_frontweek_date']}): "
                   f"{r['eex_frontweek_eur']:.1f} EUR/MWh"
                   + (" [ILLUSTRATIVE]" if r["eex_frontweek_is_illustrative"] else " [real print]")
               ))
    ax.axhline(0, color="black", lw=0.6, ls=":")

    # Shade the front-week comparison window
    ax.axvspan(FRONTWEEK_START, FRONTWEEK_END,
               alpha=0.07, color="green", label="Front-week ref window (Dec 8-14)")

    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Price (EUR/MWh)", fontsize=11)
    illustrative_tag = "  [front-week ref is ILLUSTRATIVE — conviction/size capped]" if r["eex_is_illustrative"] else ""
    ax.set_title(
        f"DE-LU Prompt-Curve View  |  OOS: {r['oos_start']} → {r['oos_end']}"
        f"{illustrative_tag}\n"
        f"Direction: {r['direction']}  |  Conviction: {r['conviction']}  |  "
        f"Basis vs front-week: {r['basis_fw_baseload_eur']:+.1f} EUR/MWh  |  "
        f"Basis vs front-month: {r['basis_fm_baseload_eur']:+.1f} EUR/MWh",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    # ---- Panel 2: Daily baseload bar chart ----------------------------------
    ax2 = axes[1]
    x_dates = [pd.Timestamp(str(d), tz="Europe/Berlin") for d in daily_baseload.index]
    colors  = [
        "#4393c3" if pd.Timestamp(str(d), tz="Europe/Berlin") <= FRONTWEEK_END
        else "#2166ac"
        for d in daily_baseload.index
    ]
    ax2.bar(x_dates, daily_baseload.values,
            color=colors, alpha=0.75, width=pd.Timedelta("20h"),
            label="Daily baseload forecast")
    if len(daily_peak) > 0:
        x_peak = [pd.Timestamp(str(d), tz="Europe/Berlin") for d in daily_peak.index]
        ax2.plot(x_peak, daily_peak.values, "o-",
                 color="darkorange", ms=5, lw=1.4, label="Daily peak forecast")
    ax2.axhline(r["eex_frontmonth_eur"], color="firebrick", lw=2.0, ls="-",
                label=f"EEX front-month: {r['eex_frontmonth_eur']:.1f} EUR/MWh")
    ax2.axhline(r["eex_frontweek_eur"], color="darkred", lw=1.6, ls=":",
                label=f"EEX front-week: {r['eex_frontweek_eur']:.1f} EUR/MWh")
    ax2.axhline(0, color="black", lw=0.6, ls=":")

    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Price (EUR/MWh)", fontsize=11)
    ax2.set_title(
        "Daily Baseload & Peak Forecasts vs EEX Reference  "
        "(lighter bars = front-week window)",
        fontsize=11,
    )
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
    illustrative_banner = (
        "\n> **NOTE:** `EEX_FRONTWEEK_BASELOAD_EUR` is an **illustrative placeholder**, "
        "not a sourced settlement print — no public EEX Phelix DE Week Future settlement "
        "could be found (EEX market-data pages only show a rolling 45-day window; full "
        "history needs an EEX Group DataSource subscription). **Conviction is capped at "
        "MODERATE and size at HALF** as a result (see `src/config.py` "
        "`EEX_FRONTWEEK_IS_ILLUSTRATIVE`). The front-month reference (103.99 EUR/MWh, "
        "ICE ENDEX GABF2026) **is** a real, dated, sourced settlement.\n"
        if r["eex_is_illustrative"]
        else ""
    )

    lines = [
        "# Prompt-Curve Translation — DE-LU Day-Ahead Power\n",
        f"**Forecast model:** Ridge (selected model, CLAUDE.md §6)  \n"
        f"**OOS window:** {r['oos_start']} → {r['oos_end']} (24 days, 576 hourly predictions)  \n"
        f"**EEX reference:** {r['eex_source']}",
        illustrative_banner,
        "",
        "## 1. Model Fair-Value Aggregates (Ridge Forecast)\n",
        "### Full OOS period (Dec 8-31 2025)",
        "",
        "| Aggregate | EUR/MWh |",
        "|-----------|---------|",
        f"| Baseload average (all hours) | **{r['baseload_avg_eur']:.2f}** |",
        f"| Peak average (08-20 weekday) | **{r['peak_avg_eur']:.2f}** |",
        f"| Off-peak average             | **{r['offpeak_avg_eur']:.2f}** |",
        "",
        f"### Front-week window ({r['fw_start']} → {r['fw_end']})\n",
        "| Aggregate | EUR/MWh |",
        "|-----------|---------|",
        f"| Baseload average | **{r['fw_baseload_eur']:.2f}** |",
        f"| Peak average     | **{r['fw_peak_eur']:.2f}** |" if r["fw_peak_eur"] else "",
        "",
        "## 2. EEX Reference Settlement\n",
        "| Contract | Settlement Date | EUR/MWh | Status |",
        "|----------|----------------|---------|--------|",
        f"| DE Front-Month Baseload (GABF2026, Jan-2026) | {r['eex_frontmonth_date']} "
        f"| {r['eex_frontmonth_eur']:.2f} "
        f"| {'**ILLUSTRATIVE**' if r['eex_frontmonth_is_illustrative'] else 'Real, sourced print'} |",
        f"| DE Front-Week Baseload (w/c Dec 8)     | {r['eex_frontweek_date']}  "
        f"| {r['eex_frontweek_eur']:.2f} "
        f"| {'**ILLUSTRATIVE** — no public print found' if r['eex_frontweek_is_illustrative'] else 'Real, sourced print'} |",
        "",
        "## 3. Basis & Directional View\n",
        "| Comparison | Basis (EUR/MWh) | Note |",
        "|------------|----------------|------|",
        f"| Front-week: model Dec 8-14 baseload vs EEX front-week | "
        f"**{r['basis_fw_baseload_eur']:+.2f}** | **Primary** comparison (matched delivery window) — drives the call below |",
        f"| Front-month: model full-OOS avg vs EEX front-month | "
        f"{r['basis_fm_baseload_eur']:+.2f} | Indicative curve-shape context only — Dec-2025 model avg vs. a Jan-2026 contract, "
        f"**not a matched delivery window**; not used to size or direct the trade |",
        "",
        f"### Direction: **{r['direction']}** — Conviction: **{r['conviction']}** — Size: **{r['position_size']}**\n",
        (
            "> Conviction/size shown above are **already capped** for the illustrative front-week "
            "reference (MODERATE / HALF ceiling). No HIGH-conviction / FULL-SIZE call rests on an "
            "unsourced number.\n" if r["eex_is_illustrative"] else ""
        ),
        "",
        r["view"],
        "",
        "## 4. Invalidation Triggers\n",
        "Position should be re-evaluated if any of the following occur:\n",
    ]
    for t in r["invalidation_triggers"]:
        lines.append(f"- {t}")
    lines += [
        "",
        "## 5. Risk-Premium Caveat\n",
        r["risk_premium_note"],
        "",
        "> *Structured output of this module is fed directly into the LLM commentary "
        "engine (Step 9) as the grounding fact-object — the LLM originates no numbers.*",
    ]

    path = os.path.join(OUTPUTS_DIR, "prompt_curve_view.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written → {path}")


def _print_summary(r: dict) -> None:
    print(
        f"\n  Prompt-curve summary:\n"
        f"    OOS baseload avg     : {r['baseload_avg_eur']:7.2f} EUR/MWh\n"
        f"    OOS peak avg         : {r['peak_avg_eur']:7.2f} EUR/MWh\n"
        f"    Front-week baseload  : {r['fw_baseload_eur']:7.2f} EUR/MWh  "
        f"vs EEX {r['eex_frontweek_eur']:.2f} → basis {r['basis_fw_baseload_eur']:+.2f}\n"
        f"    Full OOS vs EEX fm   : {r['baseload_avg_eur']:7.2f}         "
        f"vs EEX {r['eex_frontmonth_eur']:.2f} → basis {r['basis_fm_baseload_eur']:+.2f}\n"
        f"    Direction            : {r['direction']}\n"
        f"    Conviction           : {r['conviction']}\n"
        f"    Position size        : {r['position_size']}"
    )


# ---------------------------------------------------------------------------
# C1 — Hourly / block tradable DA view (REVISION_PLAN.md C1)
#
# Primary prompt-curve deliverable, independent of any external forward
# reference (sidesteps the forward-pinning risk in C2/C3 entirely). Reads the
# model's own OOS hourly forecast as the fair-value curve and flags which
# individual hours/blocks screen rich or cheap *against that day's own
# baseload average* — directly tradable via hourly DA bids, peak/off-peak
# block products, or intraday.
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
    fig, ax = plt.subplots(figsize=(14, 6))

    y_lo = oos["ridge"].min() - 10
    y_hi = oos["ridge"].max() + 10
    ax.fill_between(
        oos.index, y_lo, y_hi,
        where=oos["is_peak"].values,
        alpha=0.09, color="orange", step="post",
        label="Peak hours (08-20 weekday)",
    )
    ax.plot(oos.index, oos["ridge"], color="#2166ac", lw=1.0, alpha=0.85,
            label="Ridge OOS forecast", zorder=3)
    ax.plot(oos.index, oos["day_baseload_eur"], color="black", lw=1.0, ls="--",
            alpha=0.6, label="Day's own baseload avg (fair value)", zorder=2)

    rich_mask = oos["deviation_eur"] >= r["rich_threshold_eur"]
    cheap_mask = (oos["deviation_eur"] <= -r["rich_threshold_eur"]) & ~oos["is_negative"]
    neg_mask = oos["is_negative"]
    scarcity_mask = oos["is_scarcity"]

    ax.scatter(oos.index[rich_mask], oos.loc[rich_mask, "ridge"],
               color="firebrick", s=28, zorder=5, label=f"Rich hour (>{r['rich_threshold_eur']:.0f} EUR/MWh)")
    ax.scatter(oos.index[cheap_mask], oos.loc[cheap_mask, "ridge"],
               color="seagreen", s=28, zorder=5, label=f"Cheap hour (<-{r['rich_threshold_eur']:.0f} EUR/MWh)")
    ax.scatter(oos.index[neg_mask], oos.loc[neg_mask, "ridge"],
               color="purple", s=28, marker="v", zorder=6, label="Negative-price hour")
    ax.scatter(oos.index[scarcity_mask], oos.loc[scarcity_mask, "ridge"],
               facecolors="none", edgecolors="black", s=70, marker="o", zorder=4,
               label="Scarcity hour (top decile residual load)")

    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("Price (EUR/MWh)", fontsize=11)
    ax.set_title(
        f"Hourly/Block Tradable DA View  |  OOS: {r['oos_start']} → {r['oos_end']}\n"
        f"Peak-offpeak spread: {r['peak_offpeak_spread_eur']:+.1f} EUR/MWh  |  "
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
        "This view reads the model's own OOS hourly forecast as the fair-value curve — it does "
        "**not** depend on the EEX forward reference (see `outputs/prompt_curve_view.md` for that, "
        "separate, comparison). Each hour is flagged against **that day's own baseload average**, "
        "so the signal is purely about shape: which individual hours or blocks screen rich/cheap "
        "*within* the day, tradable via hourly DA bids, peak/off-peak block products, or intraday.\n",
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
        "Same triggers as the forward-basis view: a wind-forecast revision >5 GW, a TTF/gas gap "
        ">5% overnight, an unplanned outage on REMIT, or a demand surprise all reprice the hourly "
        "shape, not just the daily level — re-check before acting on any single-hour call above.\n",
        "> *Figure: `figures/hourly_block_view.png`.*",
    ]
    path = os.path.join(OUTPUTS_DIR, "hourly_block_view.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written → {path}")
