"""Data-driven document spec for LLM Scenario Review.

Defines the exact section order, variable order, and plot types per variable
that appear in each scenario review document. Mirrors the structure from the
original scenario_review.py template but as pure data.

Each section has a title and an ordered list of variable entries. Each entry
specifies the CalSim3 variable name, human-readable label, units, and which
plot types to include (in order). TUCP variants are included but handled
gracefully when TUCP data is unavailable.
"""

# ---------------------------------------------------------------------------
# Units map - variables that are NOT TAF
# ---------------------------------------------------------------------------
UNITS_MAP = {
    "X2_PRV_KM_": "KM",
    "EM_EC_MONTH_": "UMHOS/CM",
    "RS_EC_MONTH_": "UMHOS/CM",
    "JP_EC_MONTH_": "UMHOS/CM",
}

STORAGE_VARS = ["S_SHSTA_", "S_OROVL_", "S_FOLSM_", "S_MELON_", "S_TRNTY_", "S_SLUIS_s"]

# ---------------------------------------------------------------------------
# Plot type -> subdirectory mapping
# ---------------------------------------------------------------------------
SUBDIR_MAP = {
    "mon_ts": "mon_ts",
    "exceed_all": "exceedance",
    "exceed_tucp": "exceedance",
    "exceed_wet": "exceedance",
    "exceed_dry": "exceedance",
    "exceed_oct": "exceedance",
    "moy_all": "moy_avg",
    "moy_tucp": "moy_avg",
    "moy_wet": "moy_avg",
    "moy_dry": "moy_avg",
    "ann_tot": "ann_tot",
    "ann_exceed_all": "ann_exceed",
    "ann_exceed_tucp": "ann_exceed",
    "ann_exceed_sept": "ann_exceed",
    "ann_exceed_apr": "ann_exceed",
    "ann_exceed_sept_tucp": "ann_exceed",
    "ann_exceed_apr_tucp": "ann_exceed",
    "ann_exceed_mar": "ann_exceed",
    "ann_exceed_mar_wet": "ann_exceed",
    "ann_exceed_mar_dry": "ann_exceed",
}

# ---------------------------------------------------------------------------
# Standard plot sets reused across variable types
# ---------------------------------------------------------------------------
_STORAGE_PLOTS = [
    "mon_ts",
    "ann_exceed_apr", "ann_exceed_apr_tucp",
    "ann_exceed_sept", "ann_exceed_sept_tucp",
    "moy_all", "moy_tucp",
]

_DELIVERY_PLOTS = [
    "moy_all", "moy_tucp",
    "ann_exceed_all", "ann_exceed_tucp",
]

_FLOW_PLOTS = [
    "moy_all", "moy_tucp",
    "moy_dry", "moy_wet",
    "ann_exceed_all", "ann_exceed_tucp",
]

_SALINITY_PLOTS = _FLOW_PLOTS  # same set

_STREAM_GAIN_PLOTS = [
    "moy_all", "moy_tucp",
    "ann_exceed_all", "ann_exceed_tucp",
]

_APPLIED_WATER_PLOTS = [
    "ann_exceed_mar", "ann_exceed_mar_wet", "ann_exceed_mar_dry",
]


# ---------------------------------------------------------------------------
# Document spec - defines the review structure
# ---------------------------------------------------------------------------
DOC_SPEC = [
    {
        "title": "Reservoir Storage",
        "variables": [
            {"varname": "S_SHSTA_", "label": "Shasta Reservoir Storage", "plots": _STORAGE_PLOTS},
            {"varname": "S_OROVL_", "label": "Oroville Reservoir Storage", "plots": _STORAGE_PLOTS},
            {"varname": "S_FOLSM_", "label": "Folsom Reservoir Storage", "plots": _STORAGE_PLOTS},
            {"varname": "S_MELON_", "label": "New Melones Reservoir Storage", "plots": _STORAGE_PLOTS},
            {"varname": "S_TRNTY_", "label": "Trinity Reservoir Storage", "plots": _STORAGE_PLOTS},
            {"varname": "S_SLUIS_s", "label": "San Luis Reservoir Storage", "plots": _STORAGE_PLOTS},
        ],
    },
    {
        "title": "Deliveries",
        "variables": [
            {"varname": "DEL_CVP_PAG_N", "label": "CVP Project Agriculture - NOD", "plots": _DELIVERY_PLOTS},
            {"varname": "DEL_CVP_PAG_S", "label": "CVP Project Agriculture - SOD", "plots": _DELIVERY_PLOTS},
            {"varname": "DEL_CVP_PSC_N", "label": "CVP Settlement Contractors - NOD", "plots": _DELIVERY_PLOTS},
            {"varname": "DEL_CVP_PEX_S", "label": "CVP Exchange Contractors - SOD", "plots": _DELIVERY_PLOTS},
            {"varname": "DEL_SWP_TOTA_", "label": "SWP Total Table A Deliveries", "plots": _DELIVERY_PLOTS},
            {"varname": "DEL_SWP_PMI_S", "label": "SWP Municipal - SOD", "plots": _DELIVERY_PLOTS},
            {"varname": "DEL_SWP_PAG_N", "label": "SWP Project Agriculture - NOD", "plots": _DELIVERY_PLOTS},
            {"varname": "C_DMC000_TD_s", "label": "CVP South Delta Exports", "plots": _DELIVERY_PLOTS},
            {"varname": "C_CAA003_TD_s", "label": "SWP South Delta Exports", "plots": _DELIVERY_PLOTS},
        ],
    },
    {
        "title": "Flows & Salinity",
        "variables": [
            {"varname": "C_SAC000_s", "label": "Delta Outflow", "plots": _FLOW_PLOTS},
            {"varname": "C_SAC041_s", "label": "Sacramento River at Freeport", "plots": _FLOW_PLOTS},
            {"varname": "C_SJR070_s", "label": "San Joaquin River Inflow to the Delta", "plots": _FLOW_PLOTS},
            {"varname": "X2_PRV_KM_", "label": "X2 Position (KM)", "plots": _SALINITY_PLOTS},
            {"varname": "EM_EC_MONTH_", "label": "Emmaton EC (umhos/cm)", "plots": _SALINITY_PLOTS},
            {"varname": "RS_EC_MONTH_", "label": "Rock Slough EC (umhos/cm)", "plots": _SALINITY_PLOTS},
            {"varname": "JP_EC_MONTH_", "label": "Jersey Point EC (umhos/cm)", "plots": _SALINITY_PLOTS},
        ],
    },
    {
        "title": "Stream Gain",
        "variables": [
            {"varname": "SG_SACAB", "label": "Sacramento River Above Bend Bridge", "plots": _STREAM_GAIN_PLOTS},
            {"varname": "SG_SACBB", "label": "Sacramento River Below Bend Bridge", "plots": _STREAM_GAIN_PLOTS},
            {"varname": "SG_SACFB", "label": "Sacramento River - Feather/Bear", "plots": _STREAM_GAIN_PLOTS},
            {"varname": "SG_SACAMR", "label": "Sacramento River - American", "plots": _STREAM_GAIN_PLOTS},
        ],
    },
    {
        "title": "Applied Water",
        "variables": [
            {"varname": "AWOANN_ALL_DV", "label": "Total Applied Water", "plots": _APPLIED_WATER_PLOTS},
        ],
    },
]


def get_units(varname):
    """Return units for a variable. Defaults to TAF."""
    return UNITS_MAP.get(varname, "TAF")


def get_subdir(plot_type):
    """Return the plot subdirectory for a given plot type."""
    return SUBDIR_MAP.get(plot_type, plot_type)


def build_filename(varname, plot_type):
    """Build the expected PNG filename for a variable + plot type."""
    return f"{varname}_{plot_type}.png"


def iter_doc_spec():
    """Yield (section_title, varname, label, units, plot_type) for every entry."""
    for section in DOC_SPEC:
        for var_entry in section["variables"]:
            varname = var_entry["varname"]
            label = var_entry["label"]
            units = get_units(varname)
            for plot_type in var_entry["plots"]:
                yield section["title"], varname, label, units, plot_type
