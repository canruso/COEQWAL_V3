"""Batch LLM scenario review loop.

Iterates over the document spec (review_config.DOC_SPEC), finds each plot
image, computes stats, calls the LLM, validates the response, and saves
results as JSON files. Designed to be called from a single notebook cell.
"""

from __future__ import annotations

import json
import os
import shutil
import time

from coeqwalpackage.review_config import DOC_SPEC, get_units, get_subdir, build_filename
from coeqwalpackage.prompts import build_prompt, extract_plot_type
from coeqwalpackage.llm_utils import analyze_plot, compute_var_stats


def run_batch_review(
    df,
    plots_root: str,
    set_name: str,
    all_scenarios: list[int],
    baseline: int,
    scenario_labels: dict[int, str],
    output_base: str,
    *,
    scenario_context: dict | None = None,
    tucp_years: dict[int, list[int]] | None = None,
    wyt_wet: list[int] | None = None,
    wyt_dry: list[int] | None = None,
    wyt_month: int = 5,
    sections: list[str] | None = None,
    variables: list[str] | None = None,
    plot_types: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the full LLM review loop over all variables and plot types.

    Parameters
    ----------
    df : DataFrame
        Converted data with MultiIndex columns.
    plots_root : str
        Root directory containing plot subdirs (e.g., .../plots_output/s20_s39).
    set_name : str
        Scenario set directory name (e.g., "s20_s39_s40_s41_s42").
    all_scenarios : list[int]
        All scenario IDs including baseline.
    baseline : int
        Baseline scenario ID.
    scenario_labels : dict
        Mapping of scenario ID -> label string.
    output_base : str
        Root output directory for JSON results.
    scenario_context : dict, optional
        Scenario descriptions for prompt context.
    tucp_years : dict, optional
        TUCP years per scenario for TUCP-filtered plots.
    wyt_wet, wyt_dry : list[int], optional
        Water year type classifications.
    wyt_month : int
        Month to classify water year type (default 5 = May).
    sections : list[str], optional
        Filter to specific sections (e.g., ["Reservoir Storage"]).
    variables : list[str], optional
        Filter to specific variable names (e.g., ["S_SHSTA_"]).
    plot_types : list[str], optional
        Filter to specific plot types (e.g., ["mon_ts", "moy_all"]).
    dry_run : bool
        If True, skip LLM calls and just report what would be processed.

    Returns
    -------
    dict with keys: processed, skipped, failed, results
    """
    plots_base = os.path.join(plots_root, set_name)
    scenario_names_str = ", ".join([f"s{s:04d}" for s in all_scenarios])
    baseline_str = f"s{baseline:04d}"

    processed = []
    skipped = []
    failed = []
    results = {}

    for section in DOC_SPEC:
        section_title = section["title"]
        if sections and section_title not in sections:
            continue

        for var_entry in section["variables"]:
            varname = var_entry["varname"]
            label = var_entry["label"]
            units = get_units(varname)

            if variables and varname not in variables:
                continue

            for plot_type in var_entry["plots"]:
                if plot_types and plot_type not in plot_types:
                    continue

                filename = build_filename(varname, plot_type)
                subdir = get_subdir(plot_type)
                image_path = os.path.join(plots_base, subdir, filename)
                tag = f"{varname}/{plot_type}"

                # Skip missing images (expected for TUCP when data unavailable)
                if not os.path.isfile(image_path):
                    skipped.append({"tag": tag, "reason": "image not found", "path": image_path})
                    continue

                # Check if already processed (skip re-runs)
                output_dir = os.path.join(output_base, set_name, subdir)
                json_path = os.path.join(output_dir, filename.replace(".png", ".json"))
                if os.path.isfile(json_path):
                    skipped.append({"tag": tag, "reason": "already processed"})
                    continue

                if dry_run:
                    processed.append({"tag": tag, "image": image_path, "dry_run": True})
                    continue

                # Compute stats
                stats_text = compute_var_stats(
                    df, varname, units, all_scenarios, baseline, plot_type, scenario_labels,
                    tucp_years=tucp_years, wyt_wet=wyt_wet, wyt_dry=wyt_dry, wyt_month=wyt_month,
                )

                # Build prompt
                prompt = build_prompt(
                    plot_type, label, scenario_names_str, baseline_str,
                    scenario_context=scenario_context, stats_text=stats_text,
                )

                # Call LLM
                try:
                    observation = analyze_plot(
                        image_path, prompt,
                        stats_text=stats_text, scenarios=all_scenarios, baseline=baseline,
                    )
                except Exception as e:
                    failed.append({"tag": tag, "error": str(e)})
                    continue

                # Save outputs
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy2(image_path, os.path.join(output_dir, filename))

                record = {
                    "section": section_title,
                    "varname": varname,
                    "var_label": label,
                    "units": units,
                    "filename": filename,
                    "plot_type": plot_type,
                    "stats": stats_text,
                    "prompt": prompt,
                    "observation": observation["narrative"],
                    "structured": observation["structured"],
                    "validation": observation["validation"],
                }
                with open(json_path, "w") as f:
                    json.dump(record, f, indent=2)

                n_warnings = len(observation["validation"])
                status = "PASS" if n_warnings == 0 else f"WARN ({n_warnings})"
                processed.append({"tag": tag, "status": status})
                results[tag] = record

                print(f"  [{status}] {tag}")

    summary = {"processed": len(processed), "skipped": len(skipped), "failed": len(failed)}
    print(f"\nDone: {summary['processed']} processed, {summary['skipped']} skipped, {summary['failed']} failed")
    if failed:
        print("Failures:")
        for f_item in failed:
            print(f"  {f_item['tag']}: {f_item['error']}")

    return {"processed": processed, "skipped": skipped, "failed": failed, "results": results}
