"""Generate a self-contained HTML report from LLM scenario review outputs.

Iterates DOC_SPEC in order, loads JSON results and PNG images, renders a
Jinja2 template into a single HTML file with base64-embedded images.
"""

from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from coeqwalpackage.review_config import (
    DOC_SPEC, get_units, get_subdir, build_filename,
)

# ---- Human-readable labels for plot types --------------------------------

PLOT_TYPE_LABELS = {
    "mon_ts": "Monthly Time Series",
    "exceed_all": "Exceedance (All Years)",
    "exceed_tucp": "Exceedance (TUCP Years)",
    "exceed_wet": "Exceedance (Wet Years)",
    "exceed_dry": "Exceedance (Dry Years)",
    "exceed_oct": "Exceedance (October)",
    "moy_all": "Mean of Year (All Years)",
    "moy_tucp": "Mean of Year (TUCP Years)",
    "moy_wet": "Mean of Year (Wet Years)",
    "moy_dry": "Mean of Year (Dry Years)",
    "ann_tot": "Annual Totals",
    "ann_exceed_all": "Annual Exceedance (All Years)",
    "ann_exceed_tucp": "Annual Exceedance (TUCP Years)",
    "ann_exceed_sept": "Annual Exceedance (September)",
    "ann_exceed_apr": "Annual Exceedance (April)",
    "ann_exceed_sept_tucp": "Annual Exceedance (September, TUCP)",
    "ann_exceed_apr_tucp": "Annual Exceedance (April, TUCP)",
    "ann_exceed_mar": "Annual Exceedance (March)",
    "ann_exceed_mar_wet": "Annual Exceedance (March, Wet)",
    "ann_exceed_mar_dry": "Annual Exceedance (March, Dry)",
}


def _load_image_base64(path: str) -> str | None:
    """Read a PNG and return a data URI string, or None if missing."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _load_json(path: str) -> dict | None:
    """Load a JSON file, returning None if missing or invalid."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON, with a repair pass for common LLM errors."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Repair: fix missing opening quotes in JSON arrays.
    # e.g., ["s0040", s0042", "s0041"] -> ["s0040", "s0042", "s0041"]
    # Only targets unquoted values that already end with a closing quote.
    repaired = re.sub(r'(?<=,)\s*([a-zA-Z0-9_]+)"', r' "\1"', text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # Last resort: extract narrative with regex
    m = re.search(r'"narrative"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        return {"narrative": m.group(1)}
    return None


def _extract_observation(record: dict) -> str:
    """Extract the narrative text from the observation field.

    Some older outputs have the observation field containing the full JSON
    response (with narrative, ranking, etc.) as a string instead of just the
    narrative. This detects and extracts the narrative from those cases.
    """
    obs = record.get("observation", "")
    if not obs or not isinstance(obs, str):
        return obs or ""
    stripped = obs.strip()
    if stripped.startswith("{"):
        parsed = _try_parse_json(stripped)
        if isinstance(parsed, dict) and "narrative" in parsed:
            # Backfill structured data if the record is missing it
            if not record.get("structured") and "ranking" in parsed:
                record["structured"] = {
                    k: v for k, v in parsed.items()
                    if k in ("ranking", "best_scenario", "worst_scenario", "cited_values")
                }
            return parsed["narrative"]
    return obs


def _format_label_with_units(label: str, units: str) -> str:
    """Combine label and units, avoiding duplication.

    If the label already contains the units (e.g., 'X2 Position (KM)'),
    return the label as-is. Otherwise append ' (units)'.
    """
    if re.search(r'\(' + re.escape(units) + r'\)', label, re.IGNORECASE):
        return label
    # Also check for lowercase match like (umhos/cm) vs UMHOS/CM
    if re.search(r'\([^)]*\)', label) and units.lower() in label.lower():
        return label
    return f"{label} ({units})"


def _normalize_record(
    record: dict,
    section_title: str,
    varname: str,
    label: str,
    units: str,
) -> dict:
    """Fill missing keys for old-schema JSONs using DOC_SPEC context."""
    record.setdefault("section", section_title)
    record.setdefault("varname", varname)
    record.setdefault("var_label", label)
    record.setdefault("units", units)
    record.setdefault("structured", None)
    record.setdefault("validation", [])
    record["observation"] = _extract_observation(record)
    return record


def _build_report_data(
    json_root: str,
    set_name: str,
    scenario_labels: dict[int, str],
    baseline: int,
) -> dict:
    """Build the nested data structure consumed by the Jinja2 template.

    Returns a dict with keys: title, generation_date, scenarios, sections,
    total_plots, total_variables, total_sections, total_warnings, total_missing.
    """
    all_ids = [int(x.lstrip("s")) for x in set_name.split("_")]
    base_dir = os.path.join(json_root, set_name)

    # Scenario list for cover table
    scenarios = []
    for sid in all_ids:
        scenarios.append({
            "id": f"s{sid:04d}",
            "label": scenario_labels.get(sid, f"s{sid:04d}"),
            "is_baseline": sid == baseline,
        })

    total_plots = 0
    total_warnings = 0
    total_missing = 0
    total_variables = 0

    sections = []
    for sec_idx, section in enumerate(DOC_SPEC, start=1):
        section_title = section["title"]
        var_list = []

        for var_entry in section["variables"]:
            varname = var_entry["varname"]
            label = var_entry["label"]
            units = get_units(varname)
            entries = []

            for plot_type in var_entry["plots"]:
                subdir = get_subdir(plot_type)
                json_filename = build_filename(varname, plot_type).replace(".png", ".json")
                json_path = os.path.join(base_dir, subdir, json_filename)
                png_path = os.path.join(base_dir, subdir, build_filename(varname, plot_type))

                record = _load_json(json_path)
                plot_type_label = PLOT_TYPE_LABELS.get(plot_type, plot_type)

                if record is None:
                    entries.append({"plot_type_label": plot_type_label, "missing": True})
                    total_missing += 1
                    total_plots += 1
                    continue

                record = _normalize_record(record, section_title, varname, label, units)
                structured = record.get("structured") or {}
                validation = record.get("validation") or []

                entry = {
                    "plot_type_label": plot_type_label,
                    "missing": False,
                    "image_data": _load_image_base64(png_path),
                    "observation": record.get("observation", ""),
                    "ranking": structured.get("ranking"),
                    "best_scenario": structured.get("best_scenario"),
                    "worst_scenario": structured.get("worst_scenario"),
                    "validation": validation if validation else None,
                }
                if validation:
                    total_warnings += len(validation)
                total_plots += 1
                entries.append(entry)

            if entries:
                total_variables += 1
                display_label = _format_label_with_units(label, units)
                var_list.append({"label": display_label, "entries": entries})

        if var_list:
            sections.append({"title": section_title, "idx": sec_idx, "variables": var_list})

    # Load executive summary if cached
    summary_path = os.path.join(base_dir, "executive_summary.json")
    executive_summary = None
    if os.path.isfile(summary_path):
        try:
            with open(summary_path) as f:
                executive_summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "title": f"Scenario Review: {set_name}",
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scenarios": scenarios,
        "sections": sections,
        "total_plots": total_plots,
        "total_variables": total_variables,
        "total_sections": len(sections),
        "total_warnings": total_warnings,
        "total_missing": total_missing,
        "executive_summary": executive_summary,
    }


def generate_report(
    json_root: str,
    set_name: str,
    scenario_labels: dict[int, str],
    baseline: int,
    output_path: str | None = None,
) -> str:
    """Generate an HTML report from LLM review outputs.

    Parameters
    ----------
    json_root : str
        Root directory containing set_name subdirectory with JSON/PNG outputs.
    set_name : str
        Scenario set directory name (e.g., "s20_s39_s40_s41_s42").
    scenario_labels : dict
        Mapping of scenario ID (int) -> label string.
    baseline : int
        Baseline scenario ID.
    output_path : str, optional
        Output HTML file path. Defaults to json_root/set_name/report.html.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    if output_path is None:
        output_path = os.path.join(json_root, set_name, "report.html")

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)
    template = env.get_template("report_template.html")

    data = _build_report_data(json_root, set_name, scenario_labels, baseline)
    html = template.render(**data)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Report written: {output_path} ({size_mb:.1f} MB)")
    if size_mb > 50:
        print(f"  Note: large file ({size_mb:.0f} MB) due to embedded images. "
              "Some browsers may be slow to render.")

    return output_path
