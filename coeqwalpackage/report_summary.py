"""Executive summary generation for LLM scenario review reports.

Pass 2 of the two-pass report workflow. Reads all per-plot JSON outputs,
assembles a structured context, calls the LLM to synthesize an executive
summary, and caches the result as JSON.
"""

from __future__ import annotations

import json
import os
import re

from coeqwalpackage.review_config import (
    DOC_SPEC, get_units, get_subdir, build_filename,
)
from coeqwalpackage.report_generator import PLOT_TYPE_LABELS, _try_parse_json


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _collect_findings(json_root: str, set_name: str) -> list[dict]:
    """Walk DOC_SPEC and collect all observations + rankings from cached JSONs.

    Returns a list of dicts, one per successfully loaded plot review:
        {section, varname, label, units, plot_type, plot_type_label,
         observation, ranking, best, worst, warnings}
    """
    base_dir = os.path.join(json_root, set_name)
    findings = []

    for section in DOC_SPEC:
        section_title = section["title"]
        for var_entry in section["variables"]:
            varname = var_entry["varname"]
            label = var_entry["label"]
            units = get_units(varname)

            for plot_type in var_entry["plots"]:
                subdir = get_subdir(plot_type)
                json_filename = build_filename(varname, plot_type).replace(".png", ".json")
                json_path = os.path.join(base_dir, subdir, json_filename)

                if not os.path.isfile(json_path):
                    continue

                try:
                    with open(json_path) as f:
                        record = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                obs = record.get("observation", "")
                # Handle raw-JSON-in-observation (same issue as report_generator)
                if isinstance(obs, str) and obs.strip().startswith("{"):
                    parsed = _try_parse_json(obs.strip())
                    if isinstance(parsed, dict) and "narrative" in parsed:
                        obs = parsed["narrative"]

                structured = record.get("structured") or {}
                validation = record.get("validation") or []

                findings.append({
                    "section": section_title,
                    "varname": varname,
                    "label": label,
                    "units": units,
                    "plot_type": plot_type,
                    "plot_type_label": PLOT_TYPE_LABELS.get(plot_type, plot_type),
                    "observation": obs,
                    "ranking": structured.get("ranking", []),
                    "best": structured.get("best_scenario", ""),
                    "worst": structured.get("worst_scenario", ""),
                    "warnings": validation,
                })

    return findings


def _format_findings_for_prompt(findings: list[dict]) -> str:
    """Format findings into structured text for the LLM prompt.

    Groups by section > variable, includes observation + ranking per plot type.
    Omits stats/prompts to keep context focused.
    """
    lines = []
    current_section = None
    current_var = None

    for f in findings:
        if f["section"] != current_section:
            current_section = f["section"]
            lines.append(f"\n## {current_section}")

        var_key = f"{f['varname']}|{f['label']}"
        if var_key != current_var:
            current_var = var_key
            lines.append(f"\n### {f['label']} ({f['units']})")

        lines.append(f"\n**{f['plot_type_label']}**")
        if f["observation"]:
            lines.append(f["observation"])
        if f["ranking"]:
            lines.append(f"Ranking (best to worst): {', '.join(f['ranking'])}")
        if f["warnings"]:
            lines.append(f"[VALIDATION WARNINGS: {'; '.join(f['warnings'])}]")

    return "\n".join(lines)


def _build_summary_prompt(
    findings_text: str,
    scenario_labels: dict[int, str],
    baseline: int,
    scenario_description: str | None = None,
    set_context: dict | None = None,
) -> str:
    """Build the full executive summary prompt."""

    scenario_table = "\n".join(
        f"  s{sid:04d}: {label}" + (" (BASELINE)" if sid == baseline else "")
        for sid, label in sorted(scenario_labels.items())
    )

    desc_block = ""
    if scenario_description:
        desc_block = f"\nScenario set context (provided by the analyst):\n{scenario_description}\n"

    # Inject rich set context (expectations + questions) if provided
    context_block = ""
    schema_extra = ""
    rules_extra = ""
    if set_context:
        from coeqwalpackage.prompts import format_set_context
        ctx_text = format_set_context(set_context)
        if ctx_text:
            context_block = f"\n--- BEGIN COMPARISON CONTEXT ---\n{ctx_text}\n--- END COMPARISON CONTEXT ---\n"

        expectations = set_context.get("expectations", [])
        questions = set_context.get("questions", [])

        if expectations:
            schema_extra += """,
  "expectations_assessment": [
    {
      "expectation": "<verbatim from the expectations list above>",
      "status": "<confirmed | contradicted | mixed | unclear | not_addressed>",
      "evidence": "<1-2 sentences citing specific findings and scenarios>"
    }
  ]"""
            rules_extra += (
                '\n- "expectations_assessment" must contain one entry for EACH expectation listed in the comparison context, in order, using the exact expectation text verbatim.'
            )

        if questions:
            schema_extra += """,
  "questions_answered": [
    {
      "question": "<verbatim from the questions list above>",
      "answer": "<direct answer grounded in the findings; cite specific scenarios and numbers>"
    }
  ]"""
            rules_extra += (
                '\n- "questions_answered" must contain one entry for EACH question listed in the comparison context, in order, using the exact question text verbatim.'
            )

    return f"""You are a senior water resources analyst synthesizing a comprehensive scenario comparison report for the COEQWAL project (California water system modeling with CalSim3).

Below are the complete findings from an automated review of {findings_text.count('**')} plot analyses across multiple variables and sections. Each finding includes an LLM-generated narrative and scenario ranking for a specific variable and plot type.

Scenarios being compared:
{scenario_table}
{desc_block}{context_block}
--- BEGIN FINDINGS ---
{findings_text}
--- END FINDINGS ---

Write a structured executive summary as a JSON object with the following schema:

{{
  "overall_findings": "<2-3 paragraphs summarizing the most important patterns across ALL sections. Lead with the single most important takeaway. Identify which scenarios consistently perform best/worst and under what conditions.>",
  "key_tradeoffs": [
    "<Each entry is a concise description of a tradeoff observed across sections, e.g., 'Scenario X improves storage but reduces deliveries by Y%'. Include specific numbers where available. 3-5 tradeoffs.>"
  ],
  "unexpected_results": [
    "<Results that would surprise a water resources engineer. Include any validation warnings that suggest LLM/data inconsistencies. 2-4 entries.>"
  ],
  "section_highlights": {{
    "<Section Name>": "<1-3 sentences summarizing the most important finding for this section. Reference specific scenarios and magnitudes.>"
  }},
  "scenario_rankings": {{
    "<scenario_id>": "<1-2 sentence overall assessment of this scenario's performance across all metrics. Include where it excels and where it falls short.>"
  }}{schema_extra}
}}

Rules:
- Use only scenario IDs (e.g., s0040), not full labels
- Cite specific numbers and percentages from the findings
- Flag any contradictions or inconsistencies you notice
- "unexpected_results" should include genuinely surprising findings, not just normal variation
- "scenario_rankings" must include ALL non-baseline scenarios{rules_extra}
- Respond with ONLY the JSON object, no markdown fences or surrounding text"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_executive_summary(
    json_root: str,
    set_name: str,
    scenario_labels: dict[int, str],
    baseline: int,
    *,
    scenario_description: str | None = None,
    set_context: dict | None = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 8192,
    api_key: str | None = None,
    force: bool = False,
) -> dict:
    """Generate (or load cached) executive summary from all per-plot reviews.

    Parameters
    ----------
    json_root : str
        Root directory containing set_name subdirectory with JSON outputs.
    set_name : str
        Scenario set directory name.
    scenario_labels : dict
        Mapping of scenario ID (int) -> label string.
    baseline : int
        Baseline scenario ID.
    scenario_description : str, optional
        Analyst-provided description of what this scenario set is testing.
    set_context : dict, optional
        Rich comparison context from ``review_config.load_set_context()``.
        When provided, the summary prompt is augmented with domain expert
        expectations and targeted questions, and the resulting JSON includes
        ``expectations_assessment`` and ``questions_answered`` fields.
    model : str
        Claude model to use.
    max_tokens : int
        Max response tokens.
    api_key : str, optional
        Anthropic API key (falls back to env var).
    force : bool
        If True, regenerate even if cached summary exists.

    Returns
    -------
    dict with keys: overall_findings, key_tradeoffs, unexpected_results,
    section_highlights, scenario_rankings, _meta (prompt tokens, model, etc.)
    """
    cache_path = os.path.join(json_root, set_name, "executive_summary.json")

    if not force and os.path.isfile(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"Loaded cached executive summary: {cache_path}")
        return cached

    # Collect all findings
    findings = _collect_findings(json_root, set_name)
    if not findings:
        raise ValueError(f"No JSON outputs found in {os.path.join(json_root, set_name)}")

    n_warnings = sum(1 for f in findings if f["warnings"])
    findings_text = _format_findings_for_prompt(findings)

    print(f"Collected {len(findings)} plot findings ({n_warnings} with warnings)")
    print(f"Context size: ~{len(findings_text.split())} words")

    # Build prompt
    prompt = _build_summary_prompt(
        findings_text, scenario_labels, baseline, scenario_description,
        set_context=set_context,
    )

    # Call LLM
    from coeqwalpackage.llm_utils import _get_client

    client = _get_client(api_key)
    print(f"Calling {model} for executive summary...")

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text

    # Parse response
    parsed = _try_parse_json(raw_text.strip())
    if parsed is None:
        # Try extracting from code fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw_text, re.DOTALL)
        if fence_match:
            parsed = _try_parse_json(fence_match.group(1).strip())

    if parsed is None:
        # Last resort: find outermost braces
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last > first:
            parsed = _try_parse_json(raw_text[first:last + 1])

    if parsed is None:
        print("WARNING: Could not parse executive summary response as JSON")
        parsed = {"overall_findings": raw_text, "parse_error": True}

    # Add metadata
    parsed["_meta"] = {
        "model": model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "findings_count": len(findings),
        "warnings_count": n_warnings,
        "set_name": set_name,
    }
    if scenario_description:
        parsed["_meta"]["scenario_description"] = scenario_description

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"Executive summary cached: {cache_path}")
    print(f"  Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

    return parsed
