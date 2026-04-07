from __future__ import annotations

import os
from collections import defaultdict

from docx import Document
from docx.shared import Inches

from coeqwalpackage.review_config import (
    build_doc_spec_with_labels,
    get_subdir,
    build_filename,
    parse_scenario_set_configs,
)

_SECTION_PATTERNS = [
    ("Reservoir Storage",  lambda v: v.startswith("S_")),
    ("Deliveries",         lambda v: v.startswith("DEL_") or v.startswith("C_DMC") or v.startswith("C_CAA")),
    ("Flows & Salinity",   lambda v: v.startswith("C_SAC") or v.startswith("C_SJR") or v.startswith("SP_SAC")
                                     or v.startswith("X2_") or v.endswith("_EC_MONTH")),
    ("Stream Gain",        lambda v: v.startswith("SG_")),
    ("Applied Water",      lambda v: v.startswith("AWO")),
]

_SECTION_ORDER = [
    "Reservoir Storage",
    "Deliveries",
    "Flows & Salinity",
    "Stream Gain",
    "Applied Water",
    "Other",
]


def _get_section(varname: str) -> str:
    for section_name, test in _SECTION_PATTERNS:
        if test(varname):
            return section_name
    return "Other"


def _add_picture_safe(doc: Document, path: str, width_inches: float, required: bool = True) -> bool:
    if os.path.isfile(path):
        doc.add_picture(path, width=Inches(width_inches))
        return True
    if required:
        print(f"  [MISSING] {path}")
    return False


def list_available_scenario_sets(plots_root: str, scenario_groupings_csv: str):
    return parse_scenario_set_configs(plots_root, scenario_groupings_csv)


def generate_scenario_review_doc(
    plots_base: str,
    set_name: str,
    baseline: int,
    compare: list[int],
    output_path: str,
    *,
    width_inches: float = 7.0,
    placeholder: str = "{{Add analysis here}}",
    summary_placeholder: str = (
        "{{After adding and reviewing plots, summarize the outcomes in this "
        "scenario compared to the baseline. Note major differences, patterns, "
        "and anything unexpected.}}"
    ),
):
    if not os.path.isdir(plots_base):
        raise FileNotFoundError(f"Plots folder not found: {plots_base}")

    doc_spec = build_doc_spec_with_labels(plots_base)
    if not doc_spec or not doc_spec[0]["variables"]:
        print(f"[SKIP] No plot files found under {plots_base}")
        return None

    scenario_id_str = ", ".join(f"s{s:04d}" for s in compare)
    baseline_str = f"s{baseline:04d}"

    doc = Document()

    doc.add_heading(
        f"CalSim3 Scenario Output Review: {scenario_id_str} vs {baseline_str}",
        level=0,
    )
    doc.add_paragraph(f"Scenario ID(s): {scenario_id_str}")
    doc.add_paragraph(f"Baseline: {baseline_str}")
    doc.add_paragraph("Scenario Description(s):")
    doc.add_paragraph("Reference Description:")
    doc.add_paragraph("Review Date (updates appended):")
    doc.add_paragraph("Reviewer(s):")
    doc.add_paragraph(f"Summary of Findings:\n{summary_placeholder}")

    variables = doc_spec[0]["variables"]

    sections_dict = defaultdict(list)
    for var_entry in variables:
        sections_dict[_get_section(var_entry["varname"])].append(var_entry)

    for sec_idx, section_name in enumerate(_SECTION_ORDER, start=1):
        if section_name not in sections_dict:
            continue

        doc.add_heading(f"{sec_idx}. {section_name}", level=1)

        for var_idx, var_entry in enumerate(sections_dict[section_name], start=1):
            varname = var_entry["varname"]
            label = var_entry["label"]
            plot_types = var_entry["plots"]

            doc.add_heading(f"{var_idx}. {label}", level=2)

            any_added = False
            for plot_type in plot_types:
                subdir = get_subdir(plot_type)
                filename = build_filename(varname, plot_type)
                img_path = os.path.join(plots_base, subdir, filename)
                is_tucp = "tucp" in plot_type.lower()

                if is_tucp:
                    if os.path.isfile(img_path):
                        doc.add_picture(img_path, width=Inches(width_inches))
                        doc.add_paragraph(placeholder)
                        any_added = True
                else:
                    if _add_picture_safe(doc, img_path, width_inches, required=True):
                        doc.add_paragraph(placeholder)
                        any_added = True

            if not any_added:
                doc.add_paragraph(f"[No plot files found for {varname}]")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    print(f"Saved: {output_path}")
    return output_path


def run_scenario_review(
    plots_root: str,
    review_output_dir: str,
    scenario_groupings_csv: str,
    *,
    width_inches: float = 7.0,
    placeholder: str = "{{Add analysis here}}",
    summary_placeholder: str = (
        "{{After adding and reviewing plots, summarize the outcomes in this "
        "scenario compared to the baseline. Note major differences, patterns, "
        "and anything unexpected.}}"
    ),
    selected_set_name: str | None = None,
    file_prefix: str = "Scenario_Review",
) -> list[str]:
    scenario_set_configs = parse_scenario_set_configs(plots_root, scenario_groupings_csv)

    if selected_set_name is not None:
        scenario_set_configs = [
            cfg for cfg in scenario_set_configs
            if cfg["set_name"] == selected_set_name
        ]

    if not scenario_set_configs:
        raise FileNotFoundError(
            "No valid scenario sets found. Check scenario_groupings.csv and plots_root."
        )

    os.makedirs(review_output_dir, exist_ok=True)
    saved_paths = []

    for cfg in scenario_set_configs:
        set_name = cfg["set_name"]
        plots_base = cfg["plots_base"]

        print(f"\nGenerating review template for: {set_name}")

        out_filename = f"{file_prefix}_{set_name}_Not_Annotated.docx"
        out_path = os.path.join(review_output_dir, out_filename)

        saved = generate_scenario_review_doc(
            plots_base=plots_base,
            set_name=set_name,
            baseline=cfg["baseline"],
            compare=cfg["compare"],
            output_path=out_path,
            width_inches=width_inches,
            placeholder=placeholder,
            summary_placeholder=summary_placeholder,
        )

        if saved is not None:
            saved_paths.append(saved)

    return saved_paths