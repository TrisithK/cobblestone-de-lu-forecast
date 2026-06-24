"""
morning_note.py — Static morning desk note (REVISION_PLAN.md D1).

Assembles a single trader-facing note from outputs already produced
by earlier pipeline steps: the fair-value curve view (prompt_curve.py),
the hourly/block view (prompt_curve.py), and the cached LLM commentary
(llm_commentary.py) — including its basis vs. the EXAA (Sequence 2)
pre-auction reference [v2 round 4], with the earlier self-referential
trailing-realised-price basis kept as secondary context (no EEX forward
print is used — none could be sourced). Pure formatting — no network calls,
no secrets, no new numbers originate here; everything is read from the
structured dicts already computed upstream.

Layout: action block -> fair-value numbers -> key drivers -> basis vs EXAA
reference -> invalidation triggers -> LLM commentary prose -> embedded figures.

Two output files are written from the same data: outputs/morning_note.md
(diff-friendly, plain text) and outputs/morning_note.html (single self-contained
file, figures embedded as base64 so it renders with no other files present —
REVISION_PLAN.md D1 allows "markdown or single-file HTML"; this ships both).
"""

import base64
import os
from datetime import datetime, timezone
from typing import Optional

_ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
FIGURES_DIR = os.path.join(_ROOT, "figures")

_NOTE_FIGURES = [
    ("prompt_curve.png", "Model fair-value curve (baseload/peak, Q4 2025 slice of OOS forecast)"),
    ("hourly_block_view.png", "Hourly/block tradable view (model's own shape)"),
    ("validation_mae_by_hour.png", "Validation MAE by hour of day"),
    ("validation_mae_by_regime.png", "Validation MAE by market regime"),
]


def _figure_data_uri(filename: str) -> Optional[str]:
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_morning_note(curve_view: dict, hourly_view: dict, commentary_result: dict) -> str:
    """
    Write outputs/morning_note.md and outputs/morning_note.html. Returns the
    path of the markdown file (the HTML sibling is written alongside it).
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
        f"**{fact['curve_direction']}** — Conviction: **{fact['curve_conviction']}** — "
        f"Size: **{fact['curve_position_size']}**",
        "",
        (
            "> EXAA-referenced call [v2 round 4]: tomorrow's forecast vs. the EXAA "
            "(Sequence 2) day-ahead auction price for the same delivery day — a real "
            "pre-auction print that settles earlier the same day (~10:15 CET D-1) "
            "than the EPEX auction this model forecasts (~12:00 CET D-1), sized as a "
            "multiple of the model's own backtest MAE (see Basis section below)."
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
    # [v2 round 5] direction/conviction/drivers (flat) were replaced by
    # drivers_bullish[]/drivers_bearish[] — the LLM no longer originates a
    # direction (see src/llm_commentary.py); the per-hour/block tally
    # (fact['tally_*'], fact['baseload_direction']) is the pure-arithmetic
    # source of truth, also surfaced in outputs/fair_value_dashboard.html.
    for d in c.get("drivers_bearish", []):
        lines.append(f"- [bearish] {d}")
    for d in c.get("drivers_bullish", []):
        lines.append(f"- [bullish] {d}")

    lines += [
        "",
        f"Residual load: **{fact['residual_load_forecast_mw']:,} MW** "
        f"({fact['residual_load_percentile_pct']}th percentile vs. pre-OOS history) | "
        f"Wind Δ vs. D-1: **{fact['wind_delta_vs_prior_day_mw']:+,} MW** | "
        f"Solar Δ: **{fact['solar_delta_vs_prior_day_mw']:+,} MW** | "
        f"Load Δ: **{fact['load_delta_vs_prior_day_mw']:+,} MW** | "
        f"TTF front-month: **{fact['ttf_front_month_eur_mwh']:.2f} EUR/MWh**",
        "",
        "## Basis vs. EXAA Reference",
        "",
        "| Metric | Value (EUR/MWh) |",
        "|---|---:|",
        f"| EXAA (Sequence 2) day-ahead auction price, same delivery day | "
        f"{fact['exaa_reference_eur']:.2f} |",
        f"| Tomorrow's forecast vs. EXAA (basis) | "
        f"{fact['basis_vs_exaa_eur']:+.2f} |",
        f"| Model backtest MAE (sizing denominator) | {fact['model_mae_eur']:.2f} |",
        "",
        "EXAA settles its own day-ahead auction for BZN|DE-LU earlier the same day "
        "(~10:15 CET D-1) than the EPEX auction this model forecasts (~12:00 CET "
        "D-1) — a real, observable, point-in-time-safe pre-auction print, not a "
        "forecast or a different product. No EEX forward print is used (none could "
        "be sourced for the OOS delivery dates).",
        "",
        (
            f"*Secondary context (not used for the call above): basis vs. the "
            f"trailing realised baseload (avg of D-1 + D-7 actual) "
            f"{fact['trailing_realised_baseload_eur']:.2f} was "
            f"{fact['basis_vs_trailing_eur']:+.2f}.*"
        ),
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
        "- `figures/prompt_curve.png` — model fair-value curve (baseload/peak, Q4 2025 slice)",
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

    html_path = _build_html_note(c, fact, curve_view, hourly_view, commentary_result)
    print(f"Morning note (HTML) written → {html_path}")

    return path


def _md_inline_to_html(text: str) -> str:
    """Minimal **bold** / `code` -> HTML inline conversion for note prose."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def _build_html_note(c, fact, curve_view, hourly_view, commentary_result) -> str:
    """
    Render a single self-contained outputs/morning_note.html from the same
    structured inputs as the markdown note, with figures embedded as base64
    data URIs (no external file references) — static, no JS, no network.
    """
    direction = fact["curve_direction"]
    direction_class = {
        "LONG": "long", "BUY": "long",
        "SHORT": "short", "SELL": "short",
    }.get(direction.upper().split("/")[0].strip(), "flat")

    figures_html = []
    for filename, caption in _NOTE_FIGURES:
        uri = _figure_data_uri(filename)
        if uri:
            figures_html.append(
                f'<figure><img src="{uri}" alt="{caption}"><figcaption>{caption} '
                f'(<code>figures/{filename}</code>)</figcaption></figure>'
            )
        else:
            figures_html.append(f"<p><em>Figure not found: figures/{filename} — {caption}</em></p>")

    drivers_html = "".join(
        f"<li><strong>[bearish]</strong> {_md_inline_to_html(d)}</li>" for d in c.get("drivers_bearish", [])
    ) + "".join(
        f"<li><strong>[bullish]</strong> {_md_inline_to_html(d)}</li>" for d in c.get("drivers_bullish", [])
    )
    triggers_html = "".join(f"<li>{_md_inline_to_html(t)}</li>" for t in c["invalidation_triggers"])

    eex_note = (
        "<p class='callout'>EXAA-referenced call [v2 round 4]: tomorrow's forecast vs. the "
        "EXAA (Sequence 2) day-ahead auction price for the same delivery day — a real "
        "pre-auction print settling earlier the same day (~10:15 CET D-1) than the EPEX "
        "auction this model forecasts (~12:00 CET D-1), sized as a multiple of the "
        "model's own backtest MAE (see Basis section below).</p>"
    )

    grounding = (
        "PASSED — all numbers traced to the fact object"
        if not commentary_result["grounding_issues"]
        else "FLAGGED — " + str(commentary_result["grounding_issues"])
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DE-LU Morning Desk Note — {fact['delivery_date']}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; max-width: 880px;
         margin: 2rem auto; padding: 0 1.5rem; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.2rem; }}
  h2 {{ font-size: 1.15rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; margin-top: 2rem; }}
  .meta {{ color: #666; font-size: 0.85rem; margin-top: 0; }}
  .action {{ display: inline-block; padding: 0.6rem 1.2rem; border-radius: 6px; font-size: 1.1rem;
             font-weight: 600; margin: 0.5rem 0; }}
  .action.long {{ background: #e3f5e9; color: #1a7a3c; }}
  .action.short {{ background: #fdeaea; color: #b3261e; }}
  .action.flat {{ background: #f0f0f0; color: #444; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0; font-size: 0.92rem; }}
  th, td {{ border: 1px solid #ddd; padding: 0.45rem 0.6rem; text-align: left; }}
  th {{ background: #fafafa; }}
  td:last-child, th:last-child {{ text-align: right; }}
  blockquote {{ border-left: 3px solid #aaa; margin: 0.5rem 0; padding: 0.4rem 1rem; color: #333;
                background: #fafafa; }}
  .callout {{ background: #fff8e1; border-left: 3px solid #f2b705; padding: 0.5rem 1rem; font-size: 0.9rem; }}
  code {{ background: #f2f2f2; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.88em; }}
  figure {{ margin: 1.2rem 0; text-align: center; }}
  figure img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  figcaption {{ font-size: 0.82rem; color: #666; margin-top: 0.4rem; }}
  .footer {{ color: #888; font-size: 0.8rem; margin-top: 2.5rem; border-top: 1px solid #ddd; padding-top: 0.8rem; }}
  ul {{ padding-left: 1.3rem; }}
</style>
</head>
<body>

<h1>DE-LU Morning Desk Note — {fact['delivery_date']}</h1>
<p class="meta">Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from cached model + LLM outputs — static, no live data.</p>

<h2>Action</h2>
<div class="action {direction_class}">{direction} — Conviction: {fact['curve_conviction']} — Size: {fact['curve_position_size']}</div>
{eex_note}

<h2>Fair-Value Numbers</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Delivery date</td><td>{fact['delivery_date']}</td></tr>
<tr><td>Model</td><td>{fact['model']}</td></tr>
<tr><td>Baseload forecast</td><td>{fact['baseload_forecast_eur']:.2f} EUR/MWh</td></tr>
<tr><td>Peak forecast</td><td>{fact['peak_forecast_eur']:.2f} EUR/MWh</td></tr>
<tr><td>Model confidence band</td><td>{fact['confidence_band_note']}</td></tr>
<tr><td>OOS baseload avg (full window)</td><td>{curve_view['baseload_avg_eur']:.2f} EUR/MWh</td></tr>
<tr><td>OOS peak avg (full window)</td><td>{curve_view['peak_avg_eur']:.2f} EUR/MWh</td></tr>
</table>

<h2>Key Drivers</h2>
<ul>{drivers_html}</ul>
<p>
Residual load: <strong>{fact['residual_load_forecast_mw']:,} MW</strong>
({fact['residual_load_percentile_pct']}th percentile vs. pre-OOS history) |
Wind &Delta; vs. D-1: <strong>{fact['wind_delta_vs_prior_day_mw']:+,} MW</strong> |
Solar &Delta;: <strong>{fact['solar_delta_vs_prior_day_mw']:+,} MW</strong> |
Load &Delta;: <strong>{fact['load_delta_vs_prior_day_mw']:+,} MW</strong> |
TTF front-month: <strong>{fact['ttf_front_month_eur_mwh']:.2f} EUR/MWh</strong>
</p>

<h2>Basis vs. EXAA Reference</h2>
<table>
<tr><th>Metric</th><th>Value (EUR/MWh)</th></tr>
<tr>
  <td>EXAA (Sequence 2) day-ahead auction price, same delivery day</td>
  <td>{fact['exaa_reference_eur']:.2f}</td>
</tr>
<tr>
  <td>Tomorrow's forecast vs. EXAA (basis)</td>
  <td>{fact['basis_vs_exaa_eur']:+.2f}</td>
</tr>
<tr>
  <td>Model backtest MAE (sizing denominator)</td>
  <td>{fact['model_mae_eur']:.2f}</td>
</tr>
</table>
<p>EXAA settles its own day-ahead auction for BZN|DE-LU earlier the same day (~10:15 CET
D-1) than the EPEX auction this model forecasts (~12:00 CET D-1) — a real, observable,
point-in-time-safe pre-auction print, not a forecast or a different product. No EEX
forward print is used (none could be sourced for the OOS delivery dates).</p>
<p><em>Secondary context (not used for the call above): basis vs. the trailing realised
baseload (avg of D-1 + D-7 actual) {fact['trailing_realised_baseload_eur']:.2f} was
{fact['basis_vs_trailing_eur']:+.2f}.</em></p>
<p><strong>Hourly/block shape</strong> (model's own forecast, no forward dependency):
peak-offpeak spread {hourly_view['peak_offpeak_spread_eur']:+.1f} EUR/MWh,
{hourly_view['n_rich']} rich hours, {hourly_view['n_cheap']} cheap hours
({hourly_view['n_negative']} negative), {hourly_view['n_scarcity']} scarcity hours.
See <code>outputs/hourly_block_view.md</code> for the full hour-by-hour table.</p>

<h2>Invalidation Triggers</h2>
<ul>{triggers_html}</ul>

<h2>Commentary</h2>
<blockquote>{c['commentary_text']}</blockquote>
<p><em>Grounding check: {grounding}.</em></p>

<h2>Figures</h2>
{''.join(figures_html)}

<p class="footer">This note is generated entirely from committed/cached pipeline outputs
(<code>outputs/prompt_curve_view.md</code>, <code>outputs/hourly_block_view.md</code>,
<code>ai_logs/commentary_cache.json</code>) — <code>python main.py</code> reproduces it deterministically,
no API key or network access required. Figures above are embedded as base64 data URIs;
this file has no external dependencies and can be opened directly in a browser.</p>

</body>
</html>
"""

    path = os.path.join(OUTPUTS_DIR, "morning_note.html")
    with open(path, "w") as f:
        f.write(html)
    return path
