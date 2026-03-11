"""LLM utilities for COEQWAL scenario analysis.

Provides functions to send plot images to an LLM (Claude) for
automated interpretation and to generate executive summaries
from collected observations.
"""

import base64
import os
import numpy as np
import pandas as pd


def _get_client(api_key=None):
    from anthropic import Anthropic

    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key is None:
        from getpass import getpass
        api_key = getpass("Enter Anthropic API key: ")

    return Anthropic(api_key=api_key)


_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp"}


def _get_scenario_series(subset, sid):
    """Find the column matching a scenario ID in a subset DataFrame."""
    tag = f"_s{sid:04d}"
    cols = [c for c in subset.columns if tag in c[1]]
    if not cols:
        cols = [c for c in subset.columns if f"s{sid:04d}" in c[1]]
    return subset[cols[0]] if cols else None


def compute_var_stats(df, varname, units, scenarios, baseline, plot_type, scenario_labels=None):
    """Compute plot-type-specific stats per scenario. Returns formatted string.

    Dispatches on plot_type to compute the same data each plot visualizes.
    New plot types added as elif branches over time.
    """
    from coeqwalpackage.metrics import create_subset_unit
    subset = create_subset_unit(df, varname, units)
    if subset.empty:
        return ""

    if plot_type == "mon_ts":
        return _stats_mon_ts(subset, scenarios, baseline)
    else:
        return _stats_mon_ts(subset, scenarios, baseline)


def _stats_mon_ts(subset, scenarios, baseline):
    """Annual means, drought/wet period means, min/max. Matches mon_ts plot."""
    # Build annual means per scenario
    annual_dict = {}
    for sid in scenarios:
        series = _get_scenario_series(subset, sid)
        if series is None:
            continue
        annual_dict[sid] = series.resample("YS-OCT").mean()

    if not annual_dict or baseline not in annual_dict:
        return ""

    bl = annual_dict[baseline]
    bl_mean = bl.mean()

    # Identify drought (bottom 10%) and wet (top 10%) years from baseline
    dry_thresh = bl.quantile(0.10)
    wet_thresh = bl.quantile(0.90)
    dry_years = bl[bl <= dry_thresh].index
    wet_years = bl[bl >= wet_thresh].index

    lines = ["Underlying data (water-year statistics):"]

    for sid in scenarios:
        if sid not in annual_dict:
            continue
        a = annual_dict[sid]
        sid_str = f"s{sid:04d}"
        mean_val = a.mean()
        dry_mean = a.loc[a.index.isin(dry_years)].mean() if len(dry_years) > 0 else float('nan')
        wet_mean = a.loc[a.index.isin(wet_years)].mean() if len(wet_years) > 0 else float('nan')
        min_val = a.min()
        max_val = a.max()

        if sid == baseline:
            lines.append(f"  {sid_str} (baseline): mean={mean_val:.1f}, drought-yr mean={dry_mean:.1f}, wet-yr mean={wet_mean:.1f}, min={min_val:.1f}, max={max_val:.1f}")
        else:
            pct = ((mean_val - bl_mean) / bl_mean * 100) if bl_mean else 0
            dry_pct = ((dry_mean - annual_dict[baseline].loc[annual_dict[baseline].index.isin(dry_years)].mean()) / annual_dict[baseline].loc[annual_dict[baseline].index.isin(dry_years)].mean() * 100) if len(dry_years) > 0 else 0
            lines.append(f"  {sid_str}: mean={mean_val:.1f} ({pct:+.1f}%), drought-yr mean={dry_mean:.1f} ({dry_pct:+.1f}%), wet-yr mean={wet_mean:.1f}, min={min_val:.1f}, max={max_val:.1f}")

    return "\n".join(lines)


def analyze_plot(image_path, prompt, model="claude-sonnet-4-20250514",
                 max_tokens=500, api_key=None):
    """Send a plot image to Claude and return the observation text."""
    client = _get_client(api_key)

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    media_type = _MEDIA_TYPES.get(ext, "image/png")

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    return response.content[0].text


def generate_summary(observations_text, scenario_info, model="claude-sonnet-4-20250514", max_tokens=10000, api_key=None):
    """Generate an executive summary from collected plot observations."""
    client = _get_client(api_key)

    prompt = (
        "You are analyzing CalSim3 water system model outputs comparing "
        "multiple California water management scenarios.\n\n"
        f"Scenarios:\n{scenario_info}\n\n"
        "Below are observations from individual plot analyses. Write a "
        "concise executive summary (3-5 paragraphs) covering:\n"
        "1. Overall patterns — which scenarios consistently perform "
        "better or worse?\n"
        "2. Key tradeoffs — where does improving one outcome worsen "
        "another?\n"
        "3. Notable findings — any unexpected results or surprises?\n"
        "4. Recommendations — which scenarios merit further "
        "investigation and why?\n\n"
        f"Observations:\n{observations_text}")

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])

    return response.content[0].text
