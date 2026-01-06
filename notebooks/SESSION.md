# Session Notes: Plotting.ipynb Refactoring

## Overview

This document tracks the changes made to `Plotting.ipynb` to support:
1. Multiple scenario comparison sets
2. TUCP (Temporary Urgency Change Petition) filtering modes

---

## Change 1: Multiple Scenario Sets

### Problem
Previously, `Plotting.ipynb` only supported a single baseline/comparison set:
```python
BASELINE = 20
COMPARE = [39, 42]
```

This required manual editing and re-running the notebook for each comparison.

### Solution
Replaced with `SCENARIO_SETS` - a list of dictionaries:
```python
SCENARIO_SETS = [
    {"name": "baseline_vs_compare", "baseline": 20, "compare": [39, 42]},
    # Add more sets as needed:
    # {"name": "alt_comparison", "baseline": 27, "compare": [47, 56]},
]
```

### Implementation Details

**Cell 8 (Config):** Added `SCENARIO_SETS` configuration.

**Cell 12 (Metadata):** Collects all unique scenarios across all sets, computes TUCP years for each:
```python
all_scenarios = set()
for scenario_set in SCENARIO_SETS:
    all_scenarios.add(scenario_set["baseline"])
    all_scenarios.update(scenario_set["compare"])

tucp_years_by_scenario = {}
for sid in sorted(all_scenarios):
    years = cu.selected_tucp_years(df, scenario=sid, tucp_var_base=TUCP_VAR_BASE)
    tucp_years_by_scenario[sid] = years
```

**Cell 19 (Main Loop):** Wrapped entire plotting loop in outer iteration over `SCENARIO_SETS`.

**Cell 23 (Parallel Plots):** Same outer loop structure.

### Output Directory Structure
```
plots_output/
├── baseline_vs_compare/
│   ├── exceedance/
│   ├── moy_avg/
│   ├── mon_ts/
│   ├── ann_exceed/
│   └── ann_tot/
├── alt_comparison/
│   └── ...
└── parallel_plots/
    ├── baseline_vs_compare/
    └── alt_comparison/
```

Each scenario set gets its own isolated output directory.

---

## Change 2: TUCP Filtering Modes

### Background

TUCP years are drought years when California implemented Temporary Urgency Change Petitions, relaxing certain water quality and flow requirements. When comparing scenarios, we often want to filter data to these critical drought periods.

### The Problem

Different scenarios may have different TUCP trigger years (based on their `TUCP_TRIGGER_DV` variable). This creates a conceptual issue when comparing scenarios:

| Mode | What it means |
|------|---------------|
| **per_scenario** | Each scenario uses its own TUCP years |
| **baseline** | All scenarios in a set use the baseline's TUCP years |

### Configuration
```python
TUCP_MODE = "baseline"  # Options: "baseline" | "per_scenario"
```

### Implementation (Cell 19)
```python
if TUCP_MODE == "baseline":
    baseline_tucp_years = tucp_years_by_scenario.get(baseline, [])
    tucp_years_for_set = {s: baseline_tucp_years for s in scenarios}
else:  # "per_scenario"
    tucp_years_for_set = {s: tucp_years_by_scenario.get(s, []) for s in scenarios}
```

---

## Conceptual Analysis: Which Plots Make Sense?

### "baseline" Mode (Recommended)

All scenarios filtered to the **same** years (baseline's TUCP years).

| Plot Type | Valid? | Reasoning |
|-----------|--------|-----------|
| Exceedance curves | YES | Same underlying years, apples-to-apples |
| Monthly averages | YES | Same underlying years |
| Time series | YES | Same underlying years |
| Annual exceedance | YES | Same underlying years |
| Annual totals | YES | Same underlying years |
| Parallel plots (absolute) | YES | Same years, fair comparison |
| Parallel plots (relative %) | YES | Same years, meaningful % change |

### "per_scenario" Mode (Use with caution)

Each scenario filtered to its **own** TUCP years.

| Plot Type | Valid? | Reasoning |
|-----------|--------|-----------|
| Exceedance curves | QUESTIONABLE | Different years being compared |
| Monthly averages | QUESTIONABLE | Different years being compared |
| Time series | QUESTIONABLE | Different years being compared |
| Annual exceedance | QUESTIONABLE | Different years being compared |
| Annual totals | QUESTIONABLE | Different years being compared |
| Parallel plots (absolute) | MISLEADING | Each point based on different years |
| Parallel plots (relative %) | **NONSENSICAL** | % change between means over different year sets |

### Decision

For "per_scenario" mode:
- **SKIP** relative parallel plots entirely (they are mathematically meaningless)
- **WARN** on absolute parallel plots (label indicates different TUCP years used)
- Standard plots remain available but should be interpreted carefully

---

## Implementation Complete

### Changes to `coeqwalpackage/cqwlutils.py`

**Added `filter_by_water_years()` function:**
```python
def filter_by_water_years(df: pd.DataFrame, water_years: list[int]) -> pd.DataFrame:
    """Filter DataFrame to include only rows within the specified water years."""
```

**Modified `per_scenario_series()` to accept `tucp_years` parameter:**
```python
def per_scenario_series(
        ...,
        tucp_years: dict[int, list[int]] | None = None,  # NEW PARAMETER
        ...
) -> dict[int, pd.Series]:
```

Logic (TUCP filtering only applies when `use_tucp=True`):
- `use_tucp=False`: No TUCP filtering (tucp_years ignored) → "All Years" plots
- `use_tucp=True` + `tucp_years` provided: Use explicit years from dict
- `use_tucp=True` + `tucp_years=None`: Compute per-scenario TUCP years internally

### Changes to `coeqwalpackage/plotting.py`

**Updated `_multi` plotting functions to accept `tucp_years` parameter:**
- `plot_exceedance_multi()` - added `tucp_years: dict[int, list[int]] | None = None`
- `plot_moy_averages_multi()` - added `tucp_years: dict[int, list[int]] | None = None`
- `annualize_exceedance_multi()` - added `tucp_years: dict[int, list[int]] | None = None`

Each passes `tucp_years` through to `cu.per_scenario_series()`.

### Changes to `Plotting.ipynb`

**Cell 19 (Main Plotting Loop):**
- `tucp_args` now includes `tucp_years=tucp_years_for_set`
- This respects `TUCP_MODE`:
  - `"baseline"`: all scenarios use baseline's TUCP years
  - `"per_scenario"`: each scenario uses its own TUCP years

**Cell 23 (Parallel Plots):**
- Added `skip_relative_tucp` guard
- When `TUCP_MODE == "per_scenario"`:
  - Skips relative TUCP parallel plots (plots 5 & 6)
  - Prints warning: "Skipping relative TUCP plots in per_scenario mode (comparison over different years is nonsensical)"
  - Progress bar count adjusted to 4 per pair instead of 6

---

## Files Modified

| File | Changes |
|------|---------|
| `cqwlutils.py` | Added `filter_by_water_years()`, added `tucp_years` param to `per_scenario_series()` |
| `plotting.py` | Added `tucp_years` param to `plot_exceedance_multi()`, `plot_moy_averages_multi()`, `annualize_exceedance_multi()` |
| `Plotting.ipynb` Cell 8 | Added `SCENARIO_SETS`, `TUCP_MODE` |
| `Plotting.ipynb` Cell 12 | Collect all scenarios, compute TUCP years |
| `Plotting.ipynb` Cell 13 | Create base output directory |
| `Plotting.ipynb` Cell 15 | Update plot count calculation |
| `Plotting.ipynb` Cell 19 | Outer loop over sets, `tucp_years=tucp_years_for_set` in tucp_args |
| `Plotting.ipynb` Cell 22 | Create base parallel plots directory |
| `Plotting.ipynb` Cell 23 | Outer loop for parallel plots, skip relative TUCP in per_scenario mode |

---

## Backward Compatibility

Default configuration matches previous behavior:
```python
SCENARIO_SETS = [
    {"name": "baseline_vs_compare", "baseline": 20, "compare": [39, 42]},
]
TUCP_MODE = "per_scenario"  # Matches previous behavior
```

To use the new "baseline" mode (recommended for apples-to-apples comparisons):
```python
TUCP_MODE = "baseline"
```

---

## Date
2026-01-02
