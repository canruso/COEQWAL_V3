"""Prompt templates for LLM-based Scenario Review.

Three layers combined at runtime:
1. MAIN_RULES: always applied (format, tone, honesty)
2. PLOT_CONTEXT: per plot type (what the plot shows, what to focus on)
3. SCENARIO_CONTEXT: optional user-provided descriptions of what each scenario represents
"""

MAIN_RULES = ("Write 2-3 concise sentences integrating what you see in the plot with the provided data. "
              "Lead with the most important finding. Only cite numbers from the provided statistics - "
              "do not estimate or read values from the plot image. "
              "If the plot looks indistinguishable but data shows differences, say so directly. "
              "Rank non-baseline scenarios where meaningful. No headers, labels, or bullet points. "
              "Reference scenarios only by their ID (e.g., s0028). Do not mention line colors. "
              "Calibrate qualifier words to the actual magnitude of differences: "
              "under 5% is minor/slight, 5-15% is moderate/notable, 15-30% is significant, "
              "over 30% is substantial/dramatic. Let the numbers speak - avoid exaggeration.")

PLOT_CONTEXT = {
    "mon_ts": (
        "This is a monthly time series of {var_label}. "
        "Focus on whether scenarios visually separate or overlap, and use the data to quantify the differences."),

    "exceed_all": (
        "This is an exceedance probability plot of {var_label} (all years). "
        "Exceedance curves smooth out monthly noise, so even small visible gaps between curves "
        "represent persistent probabilistic differences. Note where curves separate and by how much, "
        "where they cross, and how tails (extremes) differ."),

    "exceed_tucp": (
        "This is an exceedance probability plot of {var_label} filtered to TUCP years. "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "Compare scenario performance during these drought emergency periods."),

    "exceed_wet": (
        "This is an exceedance probability plot of {var_label} for wet water years. "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "Compare scenario behavior in wet conditions."),

    "exceed_dry": (
        "This is an exceedance probability plot of {var_label} for dry water years. "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "Compare scenario behavior in dry conditions - critical for reliability."),

    "exceed_oct": (
        "This is an exceedance probability plot of {var_label} for October only. "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "October marks the start of the water year."),

    "moy_all": (
        "This is a month-of-year average plot of {var_label} (all years). "
        "Compare seasonal patterns -which months show the largest differences?"),

    "moy_tucp": (
        "This is a month-of-year average plot of {var_label} for TUCP years. "
        "Compare seasonal patterns during drought emergency periods."),

    "moy_wet": (
        "This is a month-of-year average plot of {var_label} for wet water years. "
        "Compare seasonal patterns in wet conditions."),

    "moy_dry": (
        "This is a month-of-year average plot of {var_label} for dry water years. "
        "Compare seasonal patterns in dry conditions."),

    "ann_exceed_all": (
        "This is an annual exceedance probability plot of {var_label}. "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "Compare annual distributions - focus on the dry-year tail (right side)."),

    "ann_exceed_tucp": (
        "This is an annual exceedance plot of {var_label} for TUCP years. "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "Compare annual performance during drought emergencies."),

    "ann_exceed_apr": (
        "This is an April annual exceedance plot of {var_label} (storage). "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "April storage reflects water supply after snowmelt. Compare distributions."),

    "ann_exceed_sept": (
        "This is a September annual exceedance plot of {var_label} (storage). "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "September storage is critical for carry-over into next water year."),

    "ann_exceed_apr_tucp": (
        "This is an April annual exceedance plot of {var_label} for TUCP years. "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "Compare April storage during drought emergencies."),

    "ann_exceed_sept_tucp": (
        "This is a September annual exceedance plot of {var_label} for TUCP years. "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "Compare September carry-over storage during drought emergencies."),

    "ann_tot": (
        "This is an annual totals time series of {var_label}. "
        "Compare annual volumes -note years with large divergences."),

    "ann_exceed_mar": (
        "This is a March annual exceedance plot of {var_label} (applied water). "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "Compare March distributions across scenarios."),

    "ann_exceed_mar_dry": (
        "This is a March annual exceedance plot of {var_label} for dry water years. "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "Compare applied water in dry conditions."),

    "ann_exceed_mar_wet": (
        "This is a March annual exceedance plot of {var_label} for wet water years. "
        "Exceedance curves smooth out noise - visible gaps are meaningful. "
        "Compare applied water in wet conditions."),
}

DEFAULT_CONTEXT = ("This is a plot of {var_label}. "
                   "Describe key observable differences between scenarios.")

SCENARIO_CONTEXT = {
    # 20: "Baseline 2023 DCR operations with TUCP actions",
    # 28: "CV-wide ag acreage reductions with TUCP",
    # 30: "Removing existing flow requirements",
    # 40: "Alt3 with 35% unimpaired Delta outflow",
    # 65: "Delta conveyance project (DCP)",
}


def format_set_context(set_context):
    """Format a scenario set context dict into a text block for the prompt.

    Accepts the dict returned by ``review_config.load_set_context()`` which
    has keys: ``description``, ``scenarios``, ``expectations``, ``questions``.
    Returns an empty string if the context is ``None`` or has nothing useful.
    """
    if not set_context:
        return ""

    parts = []

    description = set_context.get("description", "")
    if description:
        parts.append(f"Comparison context: {description}")

    scenarios = set_context.get("scenarios", {})
    if scenarios:
        lines = ["Scenario descriptions:"]
        for sid, desc in sorted(scenarios.items()):
            if isinstance(sid, int):
                lines.append(f"  s{sid:04d}: {desc}")
            else:
                lines.append(f"  {sid}: {desc}")
        parts.append("\n".join(lines))

    expectations = set_context.get("expectations", [])
    if expectations:
        lines = ["Expected patterns (domain expert hypotheses for this comparison):"]
        for exp in expectations:
            lines.append(f"  - {exp}")
        parts.append("\n".join(lines))

    questions = set_context.get("questions", [])
    if questions:
        lines = ["Targeted questions for this comparison:"]
        for q in questions:
            lines.append(f"  - {q}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def extract_plot_type(filename):
    stem = filename.rsplit(".", 1)[0]
    for key in sorted(PLOT_CONTEXT.keys(), key=len, reverse=True):
        if stem.endswith(key):
            return key
    return stem


def build_prompt(plot_type, var_label, scenario_names, baseline,
                 scenario_context=None, stats_text=None, set_context=None):
    """Build a full prompt by combining all layers.

    Parameters
    ----------
    plot_type : str
        Plot type key used to select PLOT_CONTEXT template.
    var_label : str
        Human-readable variable label.
    scenario_names : str
        Comma-separated list of non-baseline scenario labels.
    baseline : str
        Baseline scenario label.
    scenario_context : dict or None
        Legacy: simple {sid: description} mapping. Kept for backward compat.
    stats_text : str or None
        Statistics block to inject into the prompt.
    set_context : dict or None
        Rich scenario set context with keys: ``description``, ``scenarios``,
        ``expectations``, ``questions``. If provided, takes precedence over
        ``scenario_context``. The LLM is instructed to note whether the plot
        confirms or contradicts the listed expectations.
    """

    template = None
    for key in sorted(PLOT_CONTEXT.keys(), key=len, reverse=True):
        if key in plot_type:
            template = PLOT_CONTEXT[key]
            break
    if template is None:
        template = DEFAULT_CONTEXT

    body = template.format(var_label=var_label)
    header = f"Scenarios: {scenario_names}. Baseline: {baseline}."

    parts = [body, header]

    if set_context:
        ctx_block = format_set_context(set_context)
        if ctx_block:
            parts.append(ctx_block)
            parts.append(
                "When analyzing this plot, explicitly note whether the patterns you observe "
                "confirm or contradict any of the listed expectations. If the plot does not "
                "address any expectation, that is fine - focus on what the plot actually shows."
            )
    else:
        ctx = scenario_context if scenario_context is not None else SCENARIO_CONTEXT
        if ctx:
            ctx_lines = "Scenario descriptions:\n" + "\n".join(
                f"  s{sid:04d}: {desc}" for sid, desc in ctx.items()
            )
            parts.append(ctx_lines)

    if stats_text:
        parts.append(stats_text)
    parts.append(MAIN_RULES)

    return "\n\n".join(parts)
