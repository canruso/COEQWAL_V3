"""Data-driven document spec for LLM Scenario Review.

Instead of hard-coding variable lists, this module dynamically discovers all
variables and plot types by scanning the actual PNG files that exist under a
given scenario set's plot output folder. The only things that remain fixed are:

  - UNITS_MAP       : variables whose units are not TAF
  - SUBDIR_MAP      : plot-type key -> subfolder name (must stay in sync with
                      the plotting notebook's output structure)
  - PLOT_TYPE_ORDER : canonical ordering applied to plot types per variable
                      so that the same plot sequence appears for every variable
                      regardless of filesystem order

Usage
-----
The main entry point for the rest of the codebase is ``build_doc_spec(plots_dir)``
which replaces the old static ``DOC_SPEC`` list.  Call it once per scenario set:

    from coeqwalpackage.review_config import build_doc_spec
    doc_spec = build_doc_spec(os.path.join(plots_root, set_name))

The returned structure is identical in shape to the old DOC_SPEC so all
downstream consumers (``iter_doc_spec``, ``run_batch_review``, report
generator, etc.) work without modification.
"""

import os
from collections import defaultdict
import re

# ---------------------------------------------------------------------------
# Units map - variables that are NOT TAF
# ---------------------------------------------------------------------------
UNITS_MAP = {
    "X2_PRV_KM": "KM",
    "EM_EC_MONTH": "UMHOS/CM",
    "RS_EC_MONTH": "UMHOS/CM",
    "JP_EC_MONTH": "UMHOS/CM",
}

# ---------------------------------------------------------------------------
# Plot type -> subdirectory mapping
# Must stay in sync with the plotting notebook's output folder names.
# ---------------------------------------------------------------------------
SUBDIR_MAP = {
    "mon_ts":               "mon_ts",
    "exceed_all":           "exceedance",
    "exceed_tucp":          "exceedance",
    "exceed_wet":           "exceedance",
    "exceed_dry":           "exceedance",
    "exceed_oct":           "exceedance",
    "moy_all":              "moy_avg",
    "moy_tucp":             "moy_avg",
    "moy_wet":              "moy_avg",
    "moy_dry":              "moy_avg",
    "ann_tot":              "ann_tot",
    "ann_exceed_all":       "ann_exceed",
    "ann_exceed_tucp":      "ann_exceed",
    "ann_exceed_sept":      "ann_exceed",
    "ann_exceed_apr":       "ann_exceed",
    "ann_exceed_sept_tucp": "ann_exceed",
    "ann_exceed_apr_tucp":  "ann_exceed",
    "ann_exceed_mar":       "ann_exceed",
    "ann_exceed_mar_wet":   "ann_exceed",
    "ann_exceed_mar_dry":   "ann_exceed",
}

# ---------------------------------------------------------------------------
# Canonical plot-type ordering.
# All plots for every variable are sorted by this sequence so the report
# always presents plot types in the same order regardless of filesystem order.
# Any plot type found on disk that is NOT listed here is appended at the end
# in alphabetical order (future-proofing).
# ---------------------------------------------------------------------------
PLOT_TYPE_ORDER = [
    "mon_ts",
    "ann_tot",
    "moy_all",              "moy_tucp",
    "moy_wet",              "moy_dry",
    "exceed_all",           "exceed_tucp",
    "exceed_wet",           "exceed_dry",           "exceed_oct",
    "ann_exceed_all",       "ann_exceed_tucp",
    "ann_exceed_apr",       "ann_exceed_apr_tucp",
    "ann_exceed_sept",      "ann_exceed_sept_tucp",
    "ann_exceed_mar",       "ann_exceed_mar_wet",   "ann_exceed_mar_dry",
]

# Pre-compute a rank dict for fast sorting
_PLOT_TYPE_RANK = {pt: i for i, pt in enumerate(PLOT_TYPE_ORDER)}

# ---------------------------------------------------------------------------
# Suffix matching helpers
# ---------------------------------------------------------------------------

# Sort known plot-type suffixes longest-first so that e.g.
# "ann_exceed_apr_tucp" is matched before "ann_exceed_apr".
_SORTED_PLOT_TYPES = sorted(SUBDIR_MAP.keys(), key=lambda x: -len(x))


def _parse_filename(stem):
    """Given a PNG stem (filename without .png), return (varname, plot_type)
    or (None, None) if the stem does not match any known plot type suffix.

    Matching is done longest-suffix-first to avoid ambiguous partial matches.
    """
    for plot_type in _SORTED_PLOT_TYPES:
        suffix = "_" + plot_type
        if stem.endswith(suffix):
            varname = stem[: -len(suffix)]
            if varname:  # guard against empty varname
                return varname, plot_type
    return None, None


def _sort_plot_types(plot_types):
    """Sort a collection of plot-type strings by PLOT_TYPE_ORDER.
    Unknown types are appended alphabetically after all known types."""
    known   = [pt for pt in PLOT_TYPE_ORDER if pt in plot_types]
    unknown = sorted(pt for pt in plot_types if pt not in _PLOT_TYPE_RANK)
    return known + unknown


# ---------------------------------------------------------------------------
# Dynamic doc-spec builder
# ---------------------------------------------------------------------------

def build_doc_spec(set_plots_dir):
    """Scan *set_plots_dir* and build a DOC_SPEC-compatible list dynamically.

    Parameters
    ----------
    set_plots_dir : str
        Absolute path to the plot output folder for one scenario set, e.g.
        ``.../plots_output/s20_s35_s36_s37``.  The function walks all
        immediate subdirectories (one level deep) and collects every PNG file.

    Returns
    -------
    list of dict
        A single-section DOC_SPEC list with title "All Variables", where each
        variable entry has keys ``varname``, ``label``, and ``plots``.
        Variables are ordered alphabetically by varname; plot types within
        each variable follow PLOT_TYPE_ORDER.

    Notes
    -----
    - Only files whose plot-type suffix is recognised in SUBDIR_MAP are
      included; unrecognised files are silently ignored.
    - The function does NOT raise if *set_plots_dir* does not exist; it
      returns an empty spec so the caller can handle the missing folder
      gracefully.
    """
    if not os.path.isdir(set_plots_dir):
        return []

    # var -> set of plot_types found on disk
    var_plots = defaultdict(set)

    for subdir_name in os.listdir(set_plots_dir):
        subdir_path = os.path.join(set_plots_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        for fname in os.listdir(subdir_path):
            if not fname.lower().endswith(".png"):
                continue
            stem = fname[:-4]  # strip .png
            varname, plot_type = _parse_filename(stem)
            if varname is not None:
                var_plots[varname].add(plot_type)

    if not var_plots:
        return []

    variables = []
    for varname in sorted(var_plots.keys()):
        plots = _sort_plot_types(var_plots[varname])
        variables.append({
            "varname": varname,
            "label":   varname,   # use varname as label; override via LABEL_MAP if needed
            "plots":   plots,
        })

    return [{"title": "All Variables", "variables": variables}]


# ---------------------------------------------------------------------------
# Optional human-readable label overrides
# Add entries here to replace the default varname-as-label behaviour.
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "S_SHSTA":       "Shasta Reservoir Storage",
    "S_OROVL":       "Oroville Reservoir Storage",
    "S_FOLSM":       "Folsom Reservoir Storage",
    "S_MELON":       "New Melones Reservoir Storage",
    "S_TRNTY":       "Trinity Reservoir Storage",
    "S_SLUIS_SWP":   "San Luis Reservoir Storage (SWP)",
    "S_SLUIS_CVP":   "San Luis Reservoir Storage (CVP)",
    "DEL_CVP_PAG_N": "CVP Project Agriculture - NOD",
    "DEL_CVP_PAG_S": "CVP Project Agriculture - SOD",
    "DEL_CVP_PSC_N": "CVP Settlement Contractors - NOD",
    "DEL_CVP_PEX_S": "CVP Exchange Contractors - SOD",
    "DEL_SWP_TOTA":  "SWP Total Table A Deliveries",
    "DEL_SWP_PMI_S": "SWP Municipal - SOD",
    "DEL_SWP_PAG_N": "SWP Project Agriculture - NOD",
    "C_DMC000_TD_S": "CVP South Delta Exports",
    "C_CAA003_TD_S": "SWP South Delta Exports",
    "C_SAC000_S":    "Delta Outflow",
    "C_SAC041_S":    "Sacramento River at Freeport",
    "C_SJR070_S":    "San Joaquin River Inflow to the Delta",
    "X2_PRV_KM":     "X2 Position (KM)",
    "EM_EC_MONTH":   "Emmaton EC (umhos/cm)",
    "RS_EC_MONTH":   "Rock Slough EC (umhos/cm)",
    "JP_EC_MONTH":   "Jersey Point EC (umhos/cm)",
    "SG_SACAB":      "Sacramento River Above Bend Bridge",
    "SG_SACBB":      "Sacramento River Below Bend Bridge",
    "SG_SACFB":      "Sacramento River - Feather/Bear",
    "SG_SACAMR":     "Sacramento River - American",
    "AWOANN_ALL_DV": "Total Applied Water",
}


def build_doc_spec_with_labels(set_plots_dir):
    """Same as ``build_doc_spec`` but applies LABEL_MAP to each variable entry.
    Variables not in LABEL_MAP keep their varname as the label.
    This is the recommended function to call from the notebook.
    """
    spec = build_doc_spec(set_plots_dir)
    for section in spec:
        for var_entry in section["variables"]:
            var_entry["label"] = LABEL_MAP.get(var_entry["varname"], var_entry["varname"])
    return spec


# ---------------------------------------------------------------------------
# Utility functions (unchanged interface from original review_config)
# ---------------------------------------------------------------------------

def get_units(varname):
    """Return units for a variable. Defaults to TAF."""
    return UNITS_MAP.get(varname, "TAF")


def get_subdir(plot_type):
    """Return the plot subdirectory for a given plot type."""
    return SUBDIR_MAP.get(plot_type, plot_type)


def build_filename(varname, plot_type):
    """Build the expected PNG filename for a variable + plot type."""
    return f"{varname}_{plot_type}.png"


def iter_doc_spec(doc_spec):
    """Yield (section_title, varname, label, units, plot_type) for every entry.

    Parameters
    ----------
    doc_spec : list
        The list returned by ``build_doc_spec_with_labels``.
    """
    for section in doc_spec:
        for var_entry in section["variables"]:
            varname = var_entry["varname"]
            label   = var_entry["label"]
            units   = get_units(varname)
            for plot_type in var_entry["plots"]:
                yield section["title"], varname, label, units, plot_type


def _extract_s_tokens(text):
    """Extract all sNNNN tokens from a string, returning them as integers."""
    return [int(m) for m in re.findall(r's(\d+)', str(text))]


def parse_scenario_set_configs(plots_root: str, scenario_groupings_csv: str) -> list:
    """Scan plots_root for scenario set folders and match them to scenario_groupings.csv.

    Parameters
    ----------
    plots_root : str
        Path to the directory containing one subfolder per scenario set
        (e.g. ``plots_output/s20_s35_s36_s37``).
    scenario_groupings_csv : str
        Path to scenario_groupings.csv. Must contain a ``Scenarios`` column
        whose values are comma-separated sNNNN tokens (e.g. ``s0020, s0035``).

    Returns
    -------
    list of dict
        Each dict has keys: ``set_name``, ``plots_base``, ``baseline``,
        ``compare``, ``all_scenarios``.
    """
    import pandas as pd

    scen_groups_df = pd.read_csv(scenario_groupings_csv)

    if "Scenarios" not in scen_groups_df.columns:
        raise KeyError("Column 'Scenarios' not found in scenario_groupings.csv")

    # Build a map: frozenset(scenario_ids) -> scenario_ids list (preserves order)
    group_map = {}
    for _, row in scen_groups_df.iterrows():
        ids = _extract_s_tokens(row["Scenarios"])
        if ids:
            group_map[frozenset(ids)] = ids

    # Discover folders under plots_root
    if not os.path.isdir(plots_root):
        return []

    configs = []
    seen = set()

    for folder_name in sorted(os.listdir(plots_root)):
        folder_path = os.path.join(plots_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        folder_ids = set(_extract_s_tokens(folder_name))
        if not folder_ids:
            continue

        # Match folder to a group in the CSV
        matched_ids = None
        for key, ids in group_map.items():
            if key == folder_ids:
                matched_ids = ids
                break

        if matched_ids is None:
            # Fall back: use the order implied by the folder name itself
            matched_ids = _extract_s_tokens(folder_name)

        if not matched_ids:
            continue

        set_name = folder_name
        if set_name in seen:
            continue
        seen.add(set_name)

        configs.append({
            "set_name":      set_name,
            "plots_base":    folder_path,
            "baseline":      matched_ids[0],
            "compare":       matched_ids[1:],
            "all_scenarios": matched_ids,
        })

    return configs

def _build_static_doc_spec():
    """Build a static DOC_SPEC from LABEL_MAP and SUBDIR_MAP at import time."""
    from collections import defaultdict

    # Section assignment mirrors _SECTION_PATTERNS in scenario_review.py
    _SECTION_PATTERNS = [
        ("Reservoir Storage",  lambda v: v.startswith("S_")),
        ("Deliveries",         lambda v: v.startswith("DEL_") or v.startswith("C_DMC") or v.startswith("C_CAA")),
        ("Flows & Salinity",   lambda v: v.startswith("C_SAC") or v.startswith("C_SJR")
                                         or v.startswith("SP_SAC") or v.startswith("X2_")
                                         or v.endswith("_EC_MONTH")),
        ("Stream Gain",        lambda v: v.startswith("SG_")),
        ("Applied Water",      lambda v: v.startswith("AWO")),
    ]
    _SECTION_ORDER = [
        "Reservoir Storage", "Deliveries", "Flows & Salinity",
        "Stream Gain", "Applied Water", "Other",
    ]

    def _get_section(varname):
        for title, test in _SECTION_PATTERNS:
            if test(varname):
                return title
        return "Other"

    # All known plot types (use the full PLOT_TYPE_ORDER as the default set)
    all_plot_types = list(PLOT_TYPE_ORDER)

    # Group variables by section
    section_vars = defaultdict(list)
    for varname in sorted(LABEL_MAP.keys()):
        section_vars[_get_section(varname)].append(varname)

    spec = []
    for section_title in _SECTION_ORDER:
        variables = []
        for varname in section_vars.get(section_title, []):
            variables.append({
                "varname": varname,
                "label":   LABEL_MAP.get(varname, varname),
                "plots":   list(PLOT_TYPE_ORDER),   # all known plot types; missing files are skipped at load time
            })
        if variables:
            spec.append({"title": section_title, "variables": variables})

    return spec


DOC_SPEC = _build_static_doc_spec()


# ---------------------------------------------------------------------------
# Scenario set context loader
# ---------------------------------------------------------------------------
def load_set_context(set_name, context_dir=None):
    """Load scenario set context from a YAML file.

    Each YAML file lives at ``<context_dir>/<set_name>.yaml`` and has:

        description: "<one-line summary>"
        scenarios:
          s0020: "Baseline, for reference"
          s0035: "Prioritize HHS standards"
        expectations:
          - "Expected pattern 1"
        questions:
          - "Targeted question 1"

    Returns the normalized dict, or ``None`` if no file exists.
    Backward-compatible: downstream code falls back to generic prompts
    when this returns ``None``.
    """
    if context_dir is None:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(module_dir)
        context_dir = os.path.join(repo_root, "data", "scenario_contexts")

    yaml_path = os.path.join(context_dir, f"{set_name}.yaml")
    if not os.path.isfile(yaml_path):
        return None

    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load scenario contexts. Install with: pip install pyyaml"
        )

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return None

    data.setdefault("description", "")
    data.setdefault("scenarios", {})
    data.setdefault("expectations", [])
    data.setdefault("questions", [])

    # Coerce sNNNN string keys to int
    normalized = {}
    for key, val in data["scenarios"].items():
        if isinstance(key, str) and key.lower().startswith("s"):
            try:
                normalized[int(key[1:])] = val
                continue
            except ValueError:
                pass
        normalized[key] = val
    data["scenarios"] = normalized

    return data
