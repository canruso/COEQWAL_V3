"""LLM utilities for COEQWAL scenario analysis.

Provides functions to send plot images to an LLM (Claude) for
automated interpretation and to generate executive summaries
from collected observations.
"""

from __future__ import annotations

import base64
import json
import os
import re
import numpy as np
import pandas as pd

from coeqwalpackage import cqwlutils as cu


# ---------------------------------------------------------------------------
# Plot-type configuration
# ---------------------------------------------------------------------------
# Maps each PLOT_CONTEXT key to the base transform and per_scenario_series
# filter params. "transform" selects which _stats_* helper to run.
# Optional keys: use_tucp (bool), use_wyt ("wet"|"dry"), months (list[int]).
_PLOT_TYPE_CONFIG: dict[str, dict] = {
    "mon_ts":               {"transform": "mon_ts"},
    "exceed_all":           {"transform": "exceedance"},
    "exceed_tucp":          {"transform": "exceedance", "use_tucp": True},
    "exceed_wet":           {"transform": "exceedance", "use_wyt": "wet"},
    "exceed_dry":           {"transform": "exceedance", "use_wyt": "dry"},
    "exceed_oct":           {"transform": "exceedance", "months": [10]},
    "moy_all":              {"transform": "moy"},
    "moy_tucp":             {"transform": "moy", "use_tucp": True},
    "moy_wet":              {"transform": "moy", "use_wyt": "wet"},
    "moy_dry":              {"transform": "moy", "use_wyt": "dry"},
    "ann_tot":              {"transform": "annual_totals"},
    "ann_exceed_all":       {"transform": "annual_exceedance"},
    "ann_exceed_tucp":      {"transform": "annual_exceedance", "use_tucp": True},
    "ann_exceed_sept":      {"transform": "annual_exceedance", "months": [9]},
    "ann_exceed_apr":       {"transform": "annual_exceedance", "months": [4]},
    "ann_exceed_sept_tucp": {"transform": "annual_exceedance", "use_tucp": True, "months": [9]},
    "ann_exceed_apr_tucp":  {"transform": "annual_exceedance", "use_tucp": True, "months": [4]},
    "ann_exceed_mar":       {"transform": "annual_exceedance", "months": [3]},
    "ann_exceed_mar_wet":   {"transform": "annual_exceedance", "use_wyt": "wet", "months": [3]},
    "ann_exceed_mar_dry":   {"transform": "annual_exceedance", "use_wyt": "dry", "months": [3]},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    ".webp": "image/webp",
}


def _get_scenario_series(subset, sid):
    """Find the column matching a scenario ID in a subset DataFrame."""
    tag = f"_s{sid:04d}"
    cols = [c for c in subset.columns if tag in c[1]]
    if not cols:
        cols = [c for c in subset.columns if f"s{sid:04d}" in c[1]]
    return subset[cols[0]] if cols else None


def _get_series_map(
    df: pd.DataFrame,
    varname: str,
    units: str,
    scenarios: list[int],
    config: dict,
    *,
    tucp_years: dict[int, list[int]] | None = None,
    wyt_wet: list[int] | None = None,
    wyt_dry: list[int] | None = None,
    wyt_month: int = 5,
) -> dict[int, pd.Series]:
    """Call per_scenario_series with the right filter params from config."""
    use_tucp = config.get("use_tucp", False)
    use_wyt_key = config.get("use_wyt", None)  # "wet", "dry", or None
    months = config.get("months", None)

    use_wyt = use_wyt_key is not None
    wyt = None
    if use_wyt_key == "wet":
        wyt = wyt_wet
    elif use_wyt_key == "dry":
        wyt = wyt_dry

    # MOY passes months=None to per_scenario_series (groupby happens later)
    pss_months = None if config["transform"] == "moy" else months

    return cu.per_scenario_series(
        df, varname=varname, units=units, scenarios=scenarios,
        use_tucp=use_tucp, tucp_years=tucp_years,
        use_wyt=use_wyt, wyt=wyt, wyt_month=wyt_month,
        months=pss_months,
    )


def _pct_diff(val: float, ref: float) -> str:
    """Format percentage difference vs reference, returning '' if ref is zero/nan."""
    if ref == 0 or np.isnan(ref):
        return ""
    pct = (val - ref) / ref * 100
    return f" ({pct:+.1f}%)"


def _sid_label(sid: int, baseline: int) -> str:
    tag = f"s{sid:04d}"
    return f"{tag} (baseline)" if sid == baseline else tag


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

_STRUCTURED_DEFAULTS: dict = {
    "narrative": "",
    "ranking": [],
    "best_scenario": "",
    "worst_scenario": "",
    "cited_values": {},
}


def _parse_llm_response(raw_text: str) -> dict:
    """Parse LLM response text into structured dict.

    Attempts JSON parsing with multiple fallback strategies.
    Never raises - always returns a valid dict.

    Returns
    -------
    dict with keys: narrative, ranking, best_scenario, worst_scenario, cited_values
    """
    if not raw_text or not raw_text.strip():
        return {**_STRUCTURED_DEFAULTS, "narrative": raw_text or ""}

    # Strategy 1: full text is valid JSON
    parsed = _try_json_loads(raw_text.strip())
    if parsed is not None:
        return _ensure_keys(parsed)

    # Strategy 2: extract from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw_text, re.DOTALL)
    if fence_match:
        parsed = _try_json_loads(fence_match.group(1).strip())
        if parsed is not None:
            return _ensure_keys(parsed)

    # Strategy 3: find outermost JSON object boundaries
    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        parsed = _try_json_loads(raw_text[first_brace:last_brace + 1])
        if parsed is not None:
            return _ensure_keys(parsed)

    # Fallback: treat entire text as narrative
    return {**_STRUCTURED_DEFAULTS, "narrative": raw_text}


def _try_json_loads(text: str) -> dict | None:
    """Attempt json.loads, return dict or None."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _ensure_keys(parsed: dict) -> dict:
    """Ensure all expected keys exist in a parsed dict, filling defaults."""
    result = {**_STRUCTURED_DEFAULTS}
    for key in _STRUCTURED_DEFAULTS:
        if key in parsed:
            result[key] = parsed[key]
    return result


# ---------------------------------------------------------------------------
# Stats text parsing
# ---------------------------------------------------------------------------

def _extract_means_from_stats(stats_text: str, baseline: int) -> dict[str, float]:
    """Extract per-scenario mean values from formatted stats text.

    Parses lines like '  s0020 (baseline): mean=123.4, ...'
    or '  s0020 (baseline): mean=123.4 (+5.2%), ...'

    Returns dict mapping scenario tag (e.g. "s0020") to mean value.
    Returns empty dict if stats_text is empty or unparseable.
    """
    if not stats_text:
        return {}
    result: dict[str, float] = {}
    # Match lines like "  s0020 (baseline): ... mean=123.4 ..."
    # or "  s0020: ... mean=123.4 ..."
    pattern = re.compile(
        r"^\s*(s\d{4})(?:\s*\(baseline\))?\s*:.*?mean=([\d.]+)",
        re.MULTILINE,
    )
    for match in pattern.finditer(stats_text):
        tag = match.group(1)
        try:
            result[tag] = float(match.group(2))
        except ValueError:
            continue
    return result


# ---------------------------------------------------------------------------
# Stats helper: monthly time series
# ---------------------------------------------------------------------------

def _stats_mon_ts(series_map: dict[int, pd.Series], baseline: int) -> str:
    """Annual means, drought/wet period means, min/max. Mirrors plot_ts_multi."""
    annual_dict: dict[int, pd.Series] = {}
    for sid, s in series_map.items():
        annual_dict[sid] = s.resample("YS-OCT").mean()

    if not annual_dict or baseline not in annual_dict:
        return ""

    bl = annual_dict[baseline]
    bl_mean = bl.mean()

    # Identify drought (bottom 10%) and wet (top 10%) years from baseline
    dry_thresh = bl.quantile(0.10)
    wet_thresh = bl.quantile(0.90)
    dry_years = bl[bl <= dry_thresh].index
    wet_years = bl[bl >= wet_thresh].index

    bl_dry_mean = bl.loc[bl.index.isin(dry_years)].mean() if len(dry_years) > 0 else float("nan")

    lines = ["Underlying data (water-year statistics):"]

    for sid in series_map:
        if sid not in annual_dict:
            continue
        a = annual_dict[sid]
        label = _sid_label(sid, baseline)
        mean_val = a.mean()
        dry_mean = a.loc[a.index.isin(dry_years)].mean() if len(dry_years) > 0 else float("nan")
        wet_mean = a.loc[a.index.isin(wet_years)].mean() if len(wet_years) > 0 else float("nan")
        min_val = a.min()
        max_val = a.max()

        if sid == baseline:
            lines.append(
                f"  {label}: mean={mean_val:.1f}, drought-yr mean={dry_mean:.1f}, "
                f"wet-yr mean={wet_mean:.1f}, min={min_val:.1f}, max={max_val:.1f}"
            )
        else:
            pct = _pct_diff(mean_val, bl_mean)
            dry_pct = _pct_diff(dry_mean, bl_dry_mean)
            lines.append(
                f"  {label}: mean={mean_val:.1f}{pct}, drought-yr mean={dry_mean:.1f}{dry_pct}, "
                f"wet-yr mean={wet_mean:.1f}, min={min_val:.1f}, max={max_val:.1f}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stats helper: exceedance
# ---------------------------------------------------------------------------

def _stats_exceedance(series_map: dict[int, pd.Series], baseline: int) -> str:
    """Percentile stats from monthly values. Mirrors plot_exceedance_multi.

    Transform: dropna -> sort descending -> rank/(n+1) exceedance.
    Stats reported: P10, P25, P50, P75, P90, mean per scenario.
    """
    if not series_map or baseline not in series_map:
        return ""

    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    pct_labels = ["P10", "P25", "P50", "P75", "P90"]

    bl_vals: dict[str, float] = {}
    if baseline in series_map:
        bl_s = series_map[baseline].dropna()
        for pl, pv in zip(pct_labels, percentiles):
            bl_vals[pl] = bl_s.quantile(1 - pv)  # exceedance P10 = 90th quantile
        bl_vals["mean"] = bl_s.mean()

    lines = ["Exceedance statistics:"]

    for sid in series_map:
        s = series_map[sid].dropna()
        if s.empty:
            continue
        label = _sid_label(sid, baseline)
        parts = []
        for pl, pv in zip(pct_labels, percentiles):
            val = s.quantile(1 - pv)
            ref = bl_vals.get(pl, float("nan"))
            diff = _pct_diff(val, ref) if sid != baseline else ""
            parts.append(f"{pl}={val:.1f}{diff}")
        mean_val = s.mean()
        mean_diff = _pct_diff(mean_val, bl_vals.get("mean", float("nan"))) if sid != baseline else ""
        parts.append(f"mean={mean_val:.1f}{mean_diff}")
        lines.append(f"  {label}: {', '.join(parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stats helper: month-of-year averages
# ---------------------------------------------------------------------------

_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _stats_moy(series_map: dict[int, pd.Series], baseline: int) -> str:
    """Month-of-year average stats. Mirrors plot_moy_averages_multi.

    Transform: s.groupby(s.index.month).mean() for months 1-12.
    """
    moy_dict: dict[int, pd.Series] = {}
    for sid, s in series_map.items():
        clean = s.dropna()
        if clean.empty:
            continue
        moy = clean.groupby(clean.index.month).mean()
        moy = moy.reindex(np.arange(1, 13))
        moy_dict[sid] = moy

    if not moy_dict or baseline not in moy_dict:
        return ""

    bl_moy = moy_dict[baseline]

    lines = ["Month-of-year averages:"]

    for sid in series_map:
        if sid not in moy_dict:
            continue
        moy = moy_dict[sid]
        label = _sid_label(sid, baseline)
        peak_m = int(moy.idxmax()) if not moy.isna().all() else 0
        low_m = int(moy.idxmin()) if not moy.isna().all() else 0
        peak_val = moy[peak_m] if peak_m else float("nan")
        low_val = moy[low_m] if low_m else float("nan")

        if sid == baseline:
            lines.append(
                f"  {label}: peak={_MONTH_NAMES.get(peak_m, '?')} ({peak_val:.1f}), "
                f"low={_MONTH_NAMES.get(low_m, '?')} ({low_val:.1f})"
            )
        else:
            bl_peak = bl_moy[peak_m] if peak_m in bl_moy.index else float("nan")
            bl_low = bl_moy[low_m] if low_m in bl_moy.index else float("nan")
            lines.append(
                f"  {label}: peak={_MONTH_NAMES.get(peak_m, '?')} ({peak_val:.1f}{_pct_diff(peak_val, bl_peak)}), "
                f"low={_MONTH_NAMES.get(low_m, '?')} ({low_val:.1f}{_pct_diff(low_val, bl_low)})"
            )

    # Identify month with largest inter-scenario spread
    if len(moy_dict) > 1:
        spreads = {}
        for m in range(1, 13):
            vals = [moy_dict[sid][m] for sid in moy_dict if not np.isnan(moy_dict[sid].get(m, float("nan")))]
            if len(vals) >= 2:
                spreads[m] = max(vals) - min(vals)
        if spreads:
            max_spread_m = max(spreads, key=spreads.get)
            lines.append(f"  Largest inter-scenario spread: {_MONTH_NAMES.get(max_spread_m, '?')} ({spreads[max_spread_m]:.1f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stats helper: annual totals
# ---------------------------------------------------------------------------

def _stats_annual_totals(series_map: dict[int, pd.Series], baseline: int, freq: str = "YS-OCT") -> str:
    """Annual sum stats. Mirrors plot_annual_totals_ts_multi.

    Transform: s.resample(freq).sum(min_count=1).
    """
    ann_dict: dict[int, pd.Series] = {}
    for sid, s in series_map.items():
        clean = s.dropna()
        if clean.empty:
            continue
        ann_dict[sid] = clean.resample(freq).sum(min_count=1)

    if not ann_dict or baseline not in ann_dict:
        return ""

    bl_ann = ann_dict[baseline]
    bl_mean = bl_ann.mean()

    lines = ["Annual totals statistics:"]

    for sid in series_map:
        if sid not in ann_dict:
            continue
        a = ann_dict[sid]
        label = _sid_label(sid, baseline)
        mean_val = a.mean()
        std_val = a.std()
        min_val = a.min()
        max_val = a.max()

        if sid == baseline:
            lines.append(f"  {label}: mean={mean_val:.1f}, std={std_val:.1f}, min={min_val:.1f}, max={max_val:.1f}")
        else:
            pct = _pct_diff(mean_val, bl_mean)
            lines.append(f"  {label}: mean={mean_val:.1f}{pct}, std={std_val:.1f}, min={min_val:.1f}, max={max_val:.1f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stats helper: annual exceedance
# ---------------------------------------------------------------------------

def _stats_annual_exceedance(series_map: dict[int, pd.Series], baseline: int, freq: str = "YS-OCT") -> str:
    """Annual sum exceedance stats. Mirrors annualize_exceedance_multi.

    Transform: s.resample(freq).sum(min_count=1).dropna() then exceedance.
    Stats: P10, P25, P50, P75, P90 of annual sums.
    """
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    pct_labels = ["P10", "P25", "P50", "P75", "P90"]

    ann_dict: dict[int, pd.Series] = {}
    for sid, s in series_map.items():
        clean = s.dropna()
        if clean.empty:
            continue
        ann = clean.resample(freq).sum(min_count=1).dropna()
        if not ann.empty:
            ann_dict[sid] = ann

    if not ann_dict or baseline not in ann_dict:
        return ""

    bl_ann = ann_dict[baseline]
    bl_vals: dict[str, float] = {}
    for pl, pv in zip(pct_labels, percentiles):
        bl_vals[pl] = bl_ann.quantile(1 - pv)
    bl_vals["mean"] = bl_ann.mean()

    lines = ["Annual exceedance statistics:"]

    for sid in series_map:
        if sid not in ann_dict:
            continue
        a = ann_dict[sid]
        label = _sid_label(sid, baseline)
        parts = []
        for pl, pv in zip(pct_labels, percentiles):
            val = a.quantile(1 - pv)
            ref = bl_vals.get(pl, float("nan"))
            diff = _pct_diff(val, ref) if sid != baseline else ""
            parts.append(f"{pl}={val:.1f}{diff}")
        mean_val = a.mean()
        mean_diff = _pct_diff(mean_val, bl_vals.get("mean", float("nan"))) if sid != baseline else ""
        parts.append(f"mean={mean_val:.1f}{mean_diff}")
        lines.append(f"  {label}: {', '.join(parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def compute_var_stats(
    df: pd.DataFrame,
    varname: str,
    units: str,
    scenarios: list[int],
    baseline: int,
    plot_type: str,
    scenario_labels: dict[int, str] | None = None,
    *,
    tucp_years: dict[int, list[int]] | None = None,
    wyt_wet: list[int] | None = None,
    wyt_dry: list[int] | None = None,
    wyt_month: int = 5,
    freq: str = "YS-OCT",
) -> str:
    """Compute plot-type-specific stats per scenario. Returns formatted string.

    Dispatches on plot_type to compute the same data each plot visualizes.
    Never raises - returns "" on any error or unknown plot type.
    """
    config = _PLOT_TYPE_CONFIG.get(plot_type)
    if config is None:
        return ""

    # Validate required params for TUCP / WYT plot types
    if config.get("use_tucp", False) and tucp_years is None:
        return ""
    use_wyt_key = config.get("use_wyt", None)
    if use_wyt_key == "wet" and wyt_wet is None:
        return ""
    if use_wyt_key == "dry" and wyt_dry is None:
        return ""

    try:
        series_map = _get_series_map(
            df, varname, units, scenarios, config,
            tucp_years=tucp_years, wyt_wet=wyt_wet,
            wyt_dry=wyt_dry, wyt_month=wyt_month,
        )
    except (KeyError, ValueError, TypeError):
        return ""

    if not series_map:
        return ""

    transform = config["transform"]
    try:
        if transform == "mon_ts":
            return _stats_mon_ts(series_map, baseline)
        elif transform == "exceedance":
            return _stats_exceedance(series_map, baseline)
        elif transform == "moy":
            return _stats_moy(series_map, baseline)
        elif transform == "annual_totals":
            return _stats_annual_totals(series_map, baseline, freq=freq)
        elif transform == "annual_exceedance":
            return _stats_annual_exceedance(series_map, baseline, freq=freq)
        else:
            return ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_observation(
    structured: dict,
    stats_text: str,
    scenarios: list[int],
    baseline: int,
) -> list[str]:
    """Check LLM structured claims against computed statistics.

    Parameters
    ----------
    structured : dict
        Parsed LLM output with keys: ranking, best_scenario, worst_scenario,
        cited_values.
    stats_text : str
        Formatted stats string from compute_var_stats().
    scenarios : list[int]
        All scenario IDs including baseline.
    baseline : int
        Baseline scenario ID.

    Returns
    -------
    list[str] - warning strings. Empty list means all checks passed.
    Never raises exceptions.
    """
    try:
        return _validate_observation_inner(structured, stats_text, scenarios, baseline)
    except Exception:
        return ["Validation failed due to unexpected error"]


def _validate_observation_inner(
    structured: dict,
    stats_text: str,
    scenarios: list[int],
    baseline: int,
) -> list[str]:
    """Inner validation logic (may raise). Wrapped by validate_observation."""
    warnings: list[str] = []

    if not isinstance(structured, dict):
        return ["Structured output is not a dict"]

    ranking = structured.get("ranking", [])
    best = structured.get("best_scenario", "")
    worst = structured.get("worst_scenario", "")
    cited = structured.get("cited_values", {})

    # Normalize inputs - handle None, wrong types
    if not isinstance(ranking, list):
        ranking = []
        warnings.append("ranking is not a list")
    if not isinstance(best, str):
        best = str(best) if best is not None else ""
    if not isinstance(worst, str):
        worst = str(worst) if worst is not None else ""
    if not isinstance(cited, dict):
        cited = {}
        warnings.append("cited_values is not a dict")

    # Non-baseline scenario tags
    bl_tag = f"s{baseline:04d}"
    non_bl_tags = sorted([f"s{s:04d}" for s in scenarios if s != baseline])

    # Check 1: baseline exclusion from best/worst
    if best == bl_tag:
        warnings.append(f"best_scenario is the baseline ({bl_tag})")
    if worst == bl_tag:
        warnings.append(f"worst_scenario is the baseline ({bl_tag})")

    # Extract means from stats for ground-truth comparisons
    means = _extract_means_from_stats(stats_text, baseline)
    non_bl_means = {k: v for k, v in means.items() if k != bl_tag}

    # Check 2: ranking completeness
    if ranking and non_bl_tags:
        ranking_set = set(ranking)
        expected_set = set(non_bl_tags)
        missing = expected_set - ranking_set
        extra_bl = {bl_tag} & ranking_set
        if missing:
            warnings.append(f"Ranking missing scenarios: {sorted(missing)}")
        if extra_bl:
            warnings.append("Ranking includes baseline scenario")

    # Check 3: ranking order vs stats sort
    if ranking and len(non_bl_means) >= 2:
        # Skip if all non-baseline means are tied (ranking is ambiguous)
        unique_mean_vals = set(non_bl_means.values())
        if len(unique_mean_vals) > 1:
            # Sort by descending mean to get ground-truth order
            sorted_desc = sorted(non_bl_means, key=non_bl_means.get, reverse=True)
            sorted_asc = sorted(non_bl_means, key=non_bl_means.get)
            # Filter ranking to only scenarios present in means
            ranking_filtered = [r for r in ranking if r in non_bl_means]
            if len(ranking_filtered) >= 2:
                # Check if ranking matches either ascending or descending order
                if ranking_filtered != sorted_desc and ranking_filtered != sorted_asc:
                    warnings.append(
                        f"Ranking order {ranking_filtered} does not match "
                        f"stats order (desc: {sorted_desc}, asc: {sorted_asc})"
                    )

    # Check 4: best scenario
    if best and non_bl_means:
        # Check both directions - best could be highest or lowest mean
        stats_highest = max(non_bl_means, key=non_bl_means.get)
        stats_lowest = min(non_bl_means, key=non_bl_means.get)
        # All means equal -> any choice is fine
        unique_means = set(non_bl_means.values())
        if len(unique_means) > 1 and best not in (stats_highest, stats_lowest):
            warnings.append(
                f"best_scenario '{best}' is neither the highest-mean "
                f"({stats_highest}) nor lowest-mean ({stats_lowest}) scenario"
            )

    # Check 5: worst scenario
    if worst and non_bl_means:
        stats_highest = max(non_bl_means, key=non_bl_means.get)
        stats_lowest = min(non_bl_means, key=non_bl_means.get)
        unique_means = set(non_bl_means.values())
        if len(unique_means) > 1 and worst not in (stats_highest, stats_lowest):
            warnings.append(
                f"worst_scenario '{worst}' is neither the highest-mean "
                f"({stats_highest}) nor lowest-mean ({stats_lowest}) scenario"
            )

    # Check 6: cited values appear in stats_text
    if cited and stats_text:
        for key, val in cited.items():
            if val is None:
                continue
            try:
                val_float = float(val)
            except (TypeError, ValueError):
                continue
            # Format the value the same way stats text does (1 decimal)
            formatted = f"{val_float:.1f}"
            if formatted not in stats_text:
                warnings.append(f"Cited value '{key}={formatted}' not found in stats text")

    return warnings


# ---------------------------------------------------------------------------
# JSON format instruction for LLM
# ---------------------------------------------------------------------------

_JSON_FORMAT_INSTRUCTION = """

IMPORTANT: Respond with a single JSON object (no markdown fences, no text outside the JSON). Use this exact schema:
{
  "narrative": "<your full analysis text here>",
  "ranking": ["s####", "s####", ...],
  "best_scenario": "s####",
  "worst_scenario": "s####",
  "cited_values": {"<description>": <number>, ...}
}

Rules for the JSON fields:
- "narrative": your complete analysis as a single string (use \\n for line breaks)
- "ranking": ordered list of non-baseline scenario IDs from best to worst performing
- "best_scenario": the single best non-baseline scenario ID
- "worst_scenario": the single worst non-baseline scenario ID
- "cited_values": dict mapping descriptive keys to numeric values you reference (e.g. {"s0030_mean_storage": 2145.3})
"""


# ---------------------------------------------------------------------------
# LLM interaction functions
# ---------------------------------------------------------------------------

def analyze_plot(
    image_path, prompt, model="claude-sonnet-4-20250514",
    max_tokens=1000, api_key=None,
    stats_text="", scenarios=None, baseline=None,
) -> dict:
    """Send a plot image to Claude and return structured observation.

    Parameters
    ----------
    image_path : str
        Path to the plot image file.
    prompt : str
        Full prompt text (from build_prompt).
    model : str
        Claude model name.
    max_tokens : int
        Max response tokens.
    api_key : str or None
        Anthropic API key (falls back to env var).
    stats_text : str
        Formatted stats string for validation. Optional.
    scenarios : list[int] or None
        All scenario IDs including baseline. Optional.
    baseline : int or None
        Baseline scenario ID. Optional.

    Returns
    -------
    dict with keys: narrative (str), structured (dict), validation (list[str]),
    raw (str).

    When stats_text, scenarios, and baseline are all provided, runs validation.
    Otherwise validation list is empty.
    Backward compatible - callers can still call analyze_plot(path, prompt).
    """
    client = _get_client(api_key)

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    media_type = _MEDIA_TYPES.get(ext, "image/png")

    # Append JSON format instruction to the prompt
    full_prompt = prompt + _JSON_FORMAT_INSTRUCTION

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
                {"type": "text", "text": full_prompt},
            ],
        }],
    )

    raw_text = response.content[0].text
    parsed = _parse_llm_response(raw_text)

    # Build structured sub-dict (everything except narrative)
    structured = {
        "ranking": parsed.get("ranking", []),
        "best_scenario": parsed.get("best_scenario", ""),
        "worst_scenario": parsed.get("worst_scenario", ""),
        "cited_values": parsed.get("cited_values", {}),
    }

    # Run validation if all required args provided
    validation: list[str] = []
    if stats_text and scenarios is not None and baseline is not None:
        validation = validate_observation(structured, stats_text, scenarios, baseline)

    return {
        "narrative": parsed.get("narrative", raw_text),
        "structured": structured,
        "validation": validation,
        "raw": raw_text,
    }


def generate_summary(observations_text, scenario_info, model="claude-sonnet-4-20250514", max_tokens=10000, api_key=None):
    """Generate an executive summary from collected plot observations."""
    client = _get_client(api_key)

    prompt = (
        "You are analyzing CalSim3 water system model outputs comparing "
        "multiple California water management scenarios.\n\n"
        f"Scenarios:\n{scenario_info}\n\n"
        "Below are observations from individual plot analyses. Write a "
        "concise executive summary (3-5 paragraphs) covering:\n"
        "1. Overall patterns - which scenarios consistently perform "
        "better or worse?\n"
        "2. Key tradeoffs - where does improving one outcome worsen "
        "another?\n"
        "3. Notable findings - any unexpected results or surprises?\n"
        "4. Recommendations - which scenarios merit further "
        "investigation and why?\n\n"
        f"Observations:\n{observations_text}")

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])

    return response.content[0].text
