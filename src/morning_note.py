"""
morning_note.py — Static morning desk note (REVISION_PLAN.md D1).

Assembles a single trader-facing markdown note from outputs already produced
by earlier pipeline steps: the forward-basis curve view (prompt_curve.py),
the hourly/block view (prompt_curve.py), and the cached LLM commentary
(llm_commentary.py). Pure formatting — no network calls, no secrets, no new
numbers originate here; everything is read from the structured dicts already
computed upstream.

Layout: action block -> fair-value numbers -> key drivers -> basis vs curve
-> invalidation triggers -> LLM commentary prose -> embedded figures.
"""

import os
from datetime import datetime, timezone

_ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")


def build_morning_note(curve_view: dict, hourly_view: dict, commentary_result: dict) -> str:
    """
    Write outputs/morning_note.md. Returns the path written.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    c = commentary_result["validated_output"]
    fact = commentary_result["fact_object"]

    lines = [
        f"# DE-LU Morning Desk Note — {fact['delivery_date']}",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        "from cached model + LLM outputs — static, no live data._",
        "",
        "---",
        "",
        "## Action",
        "",
        f"**{c['direction']}** — Conviction: **{c['conviction']}** — "
        f"Size: **{curve_view['position_size']}**",
        "",
        (
            "> Conviction/size already capped vs. what the basis magnitude alone implies — "
            "the forward reference behind this call is an illustrative placeholder, "
            "not a sourced EEX print (see Basis section below)."
            if curve_view["eex_is_illustrative"] else ""
        ),
        "",
        "## Fair-Value Numbers",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Delivery date | {fact['delivery_date']} |",
        f"| Model | {fact['model']} |",
        f"| Baseload forecast | {fact['baseload_forecast_eur']:.2f} EUR/MWh |",
        f"| Peak forecast | {fact['peak_forecast_eur']:.2f} EUR/MWh |",
        f"| Model confidence band | {fact['confidence_band_note']} |",
        f"| OOS baseload avg (full window) | {curve_view['baseload_avg_eur']:.2f} EUR/MWh |",
        f"| OOS peak avg (full window) | {curve_view['peak_avg_eur']:.2f} EUR/MWh |",
        "",
        "## Key Drivers",
        "",
    ]
    for d in c["drivers"]:
        lines.append(f"- {d}")

    lines += [
        "",
        f"Residual load: **{fact['residual_load_forecast_mw']:,} MW** "
        f"({fact['residual_load_percentile_pct']}th percentile vs. pre-OOS history) | "
        f"Wind Δ vs. D-1: **{fact['wind_delta_vs_prior_day_mw']:+,} MW** | "
        f"Solar Δ: **{fact['solar_delta_vs_prior_day_mw']:+,} MW** | "
        f"Load Δ: **{fact['load_delta_vs_prior_day_mw']:+,} MW** | "
        f"TTF front-month: **{fact['ttf_front_month_eur_mwh']:.2f} EUR/MWh**",
        "",
        "## Basis vs. Curve",
        "",
        "| Comparison | Basis (EUR/MWh) | Reference status |",
        "|---|---:|---|",
        f"| Front-week (primary, matched delivery window) | "
        f"{curve_view['basis_fw_baseload_eur']:+.2f} | "
        f"{'ILLUSTRATIVE — no public print found' if curve_view['eex_frontweek_is_illustrative'] else 'Real, sourced print'} |",
        f"| Front-month (indicative context only — month mismatch) | "
        f"{curve_view['basis_fm_baseload_eur']:+.2f} | "
        f"{'ILLUSTRATIVE' if curve_view['eex_frontmonth_is_illustrative'] else 'Real, sourced print'} |",
        "",
        f"**Hourly/block shape (model's own forecast, no forward dependency):** "
        f"peak-offpeak spread {hourly_view['peak_offpeak_spread_eur']:+.1f} EUR/MWh, "
        f"{hourly_view['n_rich']} rich hours, {hourly_view['n_cheap']} cheap hours "
        f"({hourly_view['n_negative']} negative), {hourly_view['n_scarcity']} scarcity hours. "
        "See `outputs/hourly_block_view.md` for the full hour-by-hour table.",
        "",
        "## Invalidation Triggers",
        "",
    ]
    for t in c["invalidation_triggers"]:
        lines.append(f"- {t}")

    lines += [
        "",
        "## Commentary",
        "",
        f"> {c['commentary_text']}",
        "",
        f"_Grounding check: "
        f"{'PASSED — all numbers traced to the fact object' if not commentary_result['grounding_issues'] else 'FLAGGED — ' + str(commentary_result['grounding_issues'])}._",
        "",
        "## Figures",
        "",
        "- `figures/prompt_curve.png` — forward-basis view (OOS forecast vs. EEX reference)",
        "- `figures/hourly_block_view.png` — hourly/block tradable view (model's own shape)",
        "- `figures/validation_mae_by_hour.png`, `figures/validation_mae_by_regime.png` — model accuracy context",
        "",
        "---",
        "",
        "_This note is generated entirely from committed/cached pipeline outputs "
        "(`outputs/prompt_curve_view.md`, `outputs/hourly_block_view.md`, "
        "`ai_logs/commentary_cache.json`) — `python main.py` reproduces it deterministically, "
        "no API key or network access required._",
    ]

    path = os.path.join(OUTPUTS_DIR, "morning_note.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Morning note written → {path}")
    return path
