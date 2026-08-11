import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from contextlib import redirect_stdout
import sys

import datetime as dt
from coeqwalpackage.metrics import *
from coeqwalpackage import cqwlutils as cu
from coeqwalpackage import plotting as pu
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from coeqwalpackage.cqwlutils import find_calsim_model_root

""" INDELTA TIER CALCULATION FUNCTION """
    
def calc_indelta_tiers(df, scenario_ids, stations, thresholds, tier_rules):
    """
    Compute in-delta discrete and continuous salinity tiers.

    Continuous mapping:
        Tier 1 -> [1,2)
        Tier 2 -> [2,3)
        Tier 3 -> [3,4)
        Tier 4 -> [4,5)

    Lower value:
        stronger within tier

    Higher value:
        weaker within tier
    """
    idx = pd.IndexSlice
    max_tier = max(tier_rules.keys())

    results = {}

    # ============================================================
    # Direction metadata
    # ============================================================
    # LT metrics: higher is better
    # GT metrics: lower is better
    metric_direction = {
        "LT_A": "high_good",
        "LT_B": "high_good",
        "GT_C": "low_good"
    }

    # ============================================================
    # Normalization helper
    # ============================================================
    def interpolate(obs, better, worse, direction, interior = True):
        """
        Returns position in [0,1] or (1,1)

        0 -> exactly at better boundary
        1 -> exactly at worse boundary
        """
        if interior:
            eps = 0.001
        else:
            eps = 0.000
            
        if better is None or worse is None:
            return np.nan

        if better == worse:
            return 0.0

        if direction == "high_good":
            # higher is better
            p = (better - obs) / (better - worse)

        else:
            # lower is better
            p = (obs - better) / (worse - better)

        return np.clip(p, eps, 1 - eps)

    # ============================================================
    # Main loop
    # ============================================================
    for scenID in scenario_ids:

        # --------------------------------------------------------
        # Extract scenario columns
        # --------------------------------------------------------
        selcols = [c for c in df.columns if scenID in c[1]]

        if len(selcols) < len(stations):
            raise ValueError(f"Missing salinity columns for scenario {scenID}")

        thisdat = df.loc[:, selcols]

        # --------------------------------------------------------
        # Compute fractions
        # --------------------------------------------------------
        fracs = {}

        for var in stations:

            col = idx[:, f"{var}_{scenID}"]

            values = thisdat.loc[:, col].values.flatten()
            values = values[~np.isnan(values)]

            if len(values) == 0:
                continue

            fracs[var] = {
                "LT_A": np.mean(values < thresholds["Low"]),
                "LT_B": np.mean(values < thresholds["Mid"]),
                "GT_C": np.mean(values > thresholds["Top"]),
            }

        if not fracs:
            results[scenID] = {
                "Salinity_Tier": np.nan,
                "Salinity_Tier_continuous": np.nan
            }
            continue

        metrics = {
            "LT_A": min(v["LT_A"] for v in fracs.values()),
            "LT_B": min(v["LT_B"] for v in fracs.values()),
            "GT_C": max(v["GT_C"] for v in fracs.values())
        }

        # ========================================================
        # DISCRETE ASSIGNMENT
        # ========================================================
        discrete_tier = np.nan

        for tier in sorted(tier_rules.keys()):

            rule = tier_rules[tier]

            cond_A = rule["LT_A"] is None or metrics["LT_A"] >= rule["LT_A"]
            cond_B = rule["LT_B"] is None or metrics["LT_B"] >= rule["LT_B"]
            cond_C = rule["GT_C"] is None or metrics["GT_C"] < rule["GT_C"]

            if cond_A and cond_B and cond_C:
                discrete_tier = tier
                break

        if pd.isna(discrete_tier):
            discrete_tier = max_tier

        discrete_tier = int(discrete_tier)

        # ========================================================
        # CONTINUOUS
        # ========================================================
        
        eps = 0.001

        progresses = []

        # -------------------------
        # Tier 1
        # -------------------------
        if discrete_tier == 1:

            better = {
                "LT_A": 1.0,
                "LT_B": 1.0,
                "GT_C": 0.0
            }

            worse = tier_rules[1]

        # -------------------------
        # Tier 4
        # -------------------------
        elif discrete_tier == max_tier:

            better = tier_rules[max_tier - 1]

            ### CHANGED to allow a "worse boundary to be passed in
            # worse = {
            #     "LT_A": 0.0,
            #     "LT_B": 0.0,
            #     "GT_C": 1.0
            # }
            worse = tier_rules[max_tier]
            
        # -------------------------
        # Middle tiers
        # -------------------------
        else:

            better = tier_rules[discrete_tier - 1]
            worse = tier_rules[discrete_tier]

        # --------------------------------------------------------
        # Compute progress
        # --------------------------------------------------------
        for metric, obs in metrics.items():

            p = interpolate(
                obs=obs,
                better=better[metric],
                worse=worse[metric],
                direction=metric_direction[metric]
            )

            if not np.isnan(p):
                progresses.append(p)

        progress = np.mean(progresses) if progresses else 0.0

        continuous = discrete_tier + progress

        # hard guarantee
        continuous = min(
            max(continuous, discrete_tier + eps),
            discrete_tier + 1 - eps
        )

        # ========================================================
        # Store
        # ========================================================
        results[scenID] = {
            "Salinity_Tier": discrete_tier,
            "Salinity_Tier_continuous": continuous
        }

    result_df = pd.DataFrame.from_dict(results, orient="index")
    result_df.index.name = "ScenarioID"

    return result_df
    

""" EXPORT TIER CALCULATION FUNCTION """

##### OLD VERSION: #####
# def generate_salinity_tier_assignment_matrix(df, station_list, thresholds, start_date):
#     # extract scenario id from column name
#     def extract_scenario_id(colname):
#         name = "_".join(colname) if isinstance(colname, tuple) else str(colname)
#         match = re.search(r's\d{4}', name)
#         return match.group(0) if match else None

#     # extract station name from column name
#     def extract_station_name(colname):
#         name = "_".join(colname) if isinstance(colname, tuple) else str(colname)
#         for st in station_list:
#             if name.startswith(st + "_") or f"_{st}_" in name:
#                 return st
#         return None

#     # function to assign tiers to scenarios
#     def assign_tiers_by_scenario(df, date_series):
#         tier_rows = []
#         scenario_map = {}

#         # adds scenarios to scenario_map dictionary, prints list of all scenarios found
#         for col in df.columns:
#             sid = extract_scenario_id(col)
#             station = extract_station_name(col)
#             if sid and station:
#                 scenario_map.setdefault(sid, {})[station] = col

#         print(f"Found {len(scenario_map)} scenarios: {list(scenario_map.keys())}")

#         #iterate over each scenario in scenario_map
#         for sid, col_dict in scenario_map.items():

#             # skip scenario if missing station columns
#             if not all(st in col_dict for st in station_list):
#                 print(f" Skipping {sid}: missing one or more station columns")
#                 continue

#             # create scenario data frame with "Year" column, use dates as index
#             df_scenario = pd.DataFrame(
#                 {st: df[col_dict[st]] for st in station_list},
#                 index=date_series
#             )
#             df_scenario["Year"] = df_scenario.index.year

#             # drop rows with no data, skip scenario if all rows empty
#             valid_rows = df_scenario.dropna(subset=station_list)
#             if valid_rows.empty:
#                 print(f" Skipping {sid}: all data is NaN")
#                 continue

#             # group rows to get one row for each year
#             yearly = valid_rows.groupby("Year")
#             valid_years = list(yearly.groups.keys())
#             total_years = len(valid_years)

#             # initialize tier values
#             tier4_flag = False
#             tier3_flag = False
#             tier3_years_with_1month_over_mid = 0
#             tier2_valid_years = 0
#             tier1_valid_years = 0
#             any_year_exceeds_mid = False

#             # set flags for tier assignment, iterate over each year
#             for year, group in yearly:
#                 readings = {st: group[st] for st in station_list}

#                 if any((r > thresholds["Top"]).sum() >= 2 for r in
#                        readings.values()):  # set tier 4 if there are 2+ values greater than top threshold
#                     tier4_flag = True
#                     break

#                 if any((r > thresholds["Mid"]).sum() >= 2 for r in
#                        readings.values()):  # set tier 3 if there are 2+ values greater than mid threshold
#                     tier3_flag = True

#                 if any((r > thresholds["Mid"]).any() for r in
#                        readings.values()):  # add number of values greater than mid threshold to the variable
#                     tier3_years_with_1month_over_mid += 1

#                 if any((r > thresholds["Mid"]).any() for r in
#                        readings.values()):  # set variable to true if any value exceeds mid threshold
#                     any_year_exceeds_mid = True

#                 else:
#                     # sum up number of values in between low and mid thresholds (inclusive)
#                     in_range_counts = [((r >= thresholds["Low"]) & (r <= thresholds["Mid"])).sum() for r in
#                                        readings.values()]
#                     # add to tier2_valid_years if there are 10+ counts between the low and mid thresholds (inclusive)
#                     if all(count >= 10 for count in in_range_counts):
#                         tier2_valid_years += 1

#                 # if all 12 values are below low threshold, add to tier1_valid_years
#                 if all(((r < thresholds["Low"]).sum() == 12) for r in readings.values()):
#                     tier1_valid_years += 1

#             # skip scenario if no valid years with complete data
#             if total_years == 0:
#                 print(f" Scenario {sid}: No valid years with complete data.")
#                 continue

#             # assign tiers based on values found above
#             if tier4_flag:  # tier 4 if flag is true
#                 tier = 4
#             # tier 3 if flag is true or if fraction of years with 1 month over mid threshold is greater than 0.5
#             elif tier3_flag or (tier3_years_with_1month_over_mid / total_years > 0.05):
#                 tier = 3
#             # tier 2 if no years exceed mid threshold and if fraction of tier 2 valid years is greater than or equal to 0.95
#             elif not any_year_exceeds_mid and (tier2_valid_years / total_years >= 0.95):
#                 tier = 2
#             # tier 1 if fraction of tier 1 valid years is greater than or equal to 0.95
#             elif tier1_valid_years / total_years >= 0.95:
#                 tier = 1
#             else:  # no tier if none match
#                 tier = None
#                 print(f" Scenario {sid} did not match any tier.")
#                 # print summary
#                 print(
#                     f"   Summary: tier3_flag={tier3_flag}, tier3_pct={tier3_years_with_1month_over_mid / total_years:.2f}, "
#                     f"tier2_pct={tier2_valid_years / total_years:.2f}, tier1_pct={tier1_valid_years / total_years:.2f}, "
#                     f"any_year_exceeds_mid={any_year_exceeds_mid}")
#                 continue

#             # print tier assigned to each scenario
#             print(f"→ Scenario {sid} assigned Tier {tier}")
#             # add scenario and salinity tier to tier_rows
#             tier_rows.append({
#                 "Scenario": sid,
#                 "Salinity_Tier": tier
#             })

#         # return data frame with scenario and salinity tier columns
#         return pd.DataFrame(tier_rows, columns=["Scenario", "Salinity_Tier"])

#     # make copy of data frame, set date as index
#     df = df.copy()
#     if not pd.api.types.is_datetime64_any_dtype(df.index):
#         df.index = pd.date_range(start=start_date, periods=len(df), freq="MS")

#     # assign tiers using function above
#     date_series = df.index
#     tier_df = assign_tiers_by_scenario(df, date_series)

#     # print statement if data frame is empty
#     if tier_df.empty:
#         print(" No valid scenario-station pairs were found.")
#         return pd.DataFrame(columns=["Salinity_Tier"])

#     # return data frame with tier assignments, set scenario as index
#     return tier_df.set_index("Scenario")


def compute_export_quality_volume(salinity_df, volume_df, salinity_var, volume_var,
                                  thresholds, scenario_ids):
    """Compute annual water-year sums of quality-adjusted export volume per scenario."""
    ht, lt = thresholds["Top"], thresholds["Low"]
    annual_qv = {}

    for sid in scenario_ids:
        sal_cols = [c for c in salinity_df.columns if salinity_var in c[1] and sid in c[1]]
        vol_cols = [c for c in volume_df.columns if volume_var in c[1] and sid in c[1]]
        if not sal_cols or not vol_cols:
            continue

        salinity = salinity_df[sal_cols[0]]
        volume = volume_df[vol_cols[0]]
        weight = ((ht - salinity) / (ht - lt)).clip(0, 1)
        quality_volume = volume * weight
        annual_qv[sid] = quality_volume.resample("YS-OCT").sum()

    return pd.DataFrame(annual_qv) if annual_qv else pd.DataFrame()


def build_export_quality_detail(salinity_df, volume_df, thresholds, scenario_ids,
                                stations):
    """
    Build per-scenario DataFrames with salinity, volume, weight, and adjusted volume.

    Parameters
    ----------
    salinity_df : pd.DataFrame
        DataFrame with salinity columns (multi-index).
    volume_df : pd.DataFrame
        DataFrame with volume columns (multi-index).
    thresholds : dict
        {"Top": ..., "Low": ...} salinity thresholds for weight calculation.
    scenario_ids : list of str
        Scenario identifiers (e.g. ["s0001", "s0020"]).
    stations : list of dict
        Each dict has keys: "salinity_var", "volume_var", "label".
        Example: [
            {"salinity_var": "BANKSEC", "volume_var": "C_CAA003", "label": "Banks"},
            {"salinity_var": "TRACYEC", "volume_var": "C_DMC000", "label": "Tracy"},
        ]

    Returns
    -------
    detail_dict : dict[str, pd.DataFrame]
        Per-scenario DataFrames indexed by time. Columns per station:
        Salinity_{label}, Volume_{label}, Weight_{label}, AdjVolume_{label}.
    detail_wide : pd.DataFrame
        Combined wide DataFrame with multi-level columns (scenario, variable).
    """
    ht, lt = thresholds["Top"], thresholds["Low"]
    detail_dict = {}

    for sid in scenario_ids:
        parts = {}
        for st in stations:
            sal_var, vol_var, label = st["salinity_var"], st["volume_var"], st["label"]
            sal_cols = [c for c in salinity_df.columns if sal_var in c[1] and sid in c[1]]
            vol_cols = [c for c in volume_df.columns if vol_var in c[1] and sid in c[1]]
            if not sal_cols or not vol_cols:
                continue
            salinity = salinity_df[sal_cols[0]]
            volume = volume_df[vol_cols[0]]
            weight = ((ht - salinity) / (ht - lt)).clip(0, 1)
            adj_volume = volume * weight

            parts[f"Salinity_{label}"] = salinity
            parts[f"Volume_{label}"] = volume
            parts[f"Weight_{label}"] = weight
            parts[f"AdjVolume_{label}"] = adj_volume

        if parts:
            detail_dict[sid] = pd.DataFrame(parts)

    # build wide DataFrame with multi-level columns (scenario, variable)
    wide_parts = {}
    for sid, df_detail in detail_dict.items():
        for col in df_detail.columns:
            wide_parts[(sid, col)] = df_detail[col]
    detail_wide = pd.DataFrame(wide_parts)
    if not detail_wide.empty:
        detail_wide.columns = pd.MultiIndex.from_tuples(
            detail_wide.columns, names=["Scenario", "Variable"])

    return detail_dict, detail_wide


def build_export_quality_summary(detail_dict):
    """
    Build summary DataFrame with total volumes per scenario (one row per scenario).

    Parameters
    ----------
    detail_dict : dict[str, pd.DataFrame]
        Output from build_export_quality_detail().

    Returns
    -------
    pd.DataFrame
        Index: Scenario. Columns: Total_Volume_{label}, Total_AdjVolume_{label}
        for each station found in the detail DataFrames.
    """
    rows = []
    for sid, df_detail in detail_dict.items():
        row = {"Scenario": sid}
        # find all labels from column names (pattern: Volume_{label}, AdjVolume_{label})
        labels = set()
        for col in df_detail.columns:
            if col.startswith("Volume_"):
                labels.add(col.replace("Volume_", ""))
        for label in sorted(labels):
            row[f"Total_Volume_{label}"] = df_detail[f"Volume_{label}"].sum()
            row[f"Total_AdjVolume_{label}"] = df_detail[f"AdjVolume_{label}"].sum()
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.set_index("Scenario")
    return summary

def assign_export_tiers(summary_df, detail_dict, bounds_config):
    """
    Assign export tiers.

    Discrete:
        Based on total adjusted export volume.

    Continuous:
        Interpolated strictly within assigned discrete tier.

        Tier 1 -> [1, 2)
        Tier 2 -> [2, 3)
        Tier 3 -> [3, 4)
        Tier 4 -> [4, 5)

    Lower continuous value:
        Closer to better tier boundary

    Higher continuous value:
        Closer to worse tier boundary
    """

    result = summary_df.copy()
    eps = 0.001
    
    for label, bounds in bounds_config.items():

        # Ensure descending:
        # b1 > b2 > b3
        bounds = sorted(bounds, reverse=True)

        if len(bounds) != 3:
            raise ValueError(
                f"{label}: bounds_config must contain exactly 3 bounds"
            )

        b1, b2, b3 = bounds

        total_col = f"Total_AdjVolume_{label}"
        tier_col = f"Export_Tier_{label}"
        cont_col = f"{tier_col}_continuous"

        # ============================================================
        # DISCRETE
        # ============================================================
        def discrete_assign(v):
            if v >= b1:
                return 1
            elif v >= b2:
                return 2
            elif v >= b3:
                return 3
            else:
                return 4

        result[tier_col] = result[total_col].apply(discrete_assign)

        # ============================================================
        # CONTINUOUS
        # ============================================================
        continuous_vals = {}

        for scen, df in detail_dict.items():

            if scen not in result.index:
                continuous_vals[scen] = np.nan
                continue

            df = df.copy()
            df.index = pd.to_datetime(df.index)

            # --------------------------------------------------------
            # Select export series
            # --------------------------------------------------------
            if label == "Total":
                series = (
                    df["AdjVolume_Banks"].fillna(0)
                    + df["AdjVolume_Tracy"].fillna(0)
                )
            else:
                col = f"AdjVolume_{label}"

                if col not in df.columns:
                    continuous_vals[scen] = np.nan
                    continue

                series = df[col].fillna(0)

            # Annual aggregation
            annual = series.resample("YE").sum()

            if annual.empty:
                continuous_vals[scen] = np.nan
                continue

            total_export = annual.sum()

            discrete_tier = int(result.loc[scen, tier_col])

            # get max export from data
            tier1_max = result.loc[result[tier_col] == 1, total_col].max()
            
            if discrete_tier == 1:
            
                denom = tier1_max - b1
            
                if denom <= 0:
                    progress = 0
                else:
                    progress = (tier1_max - total_export) / denom
            
                continuous = 1 + np.clip(progress, eps, 1 - eps)
            
            # Tier 2: [2,3)
            elif discrete_tier == 2:

                denom = b1 - b2

                if denom == 0:
                    progress = 0
                else:
                    progress = (b1 - total_export) / denom

                continuous = 2 + np.clip(progress, eps, 1 - eps)

            # Tier 3: [3,4)
            elif discrete_tier == 3:

                denom = b2 - b3

                if denom == 0:
                    progress = 0
                else:
                    progress = (b2 - total_export) / denom

                continuous = 3 + np.clip(progress, eps, 1 - eps)

            # Tier 4: [4,5)
            else:

                if b3 == 0:
                    progress = 0
                else:
                    progress = (b3 - total_export) / b3

                continuous = 4 + np.clip(progress, eps, 1 - eps)

            continuous_vals[scen] = continuous

        result[cont_col] = pd.Series(continuous_vals)

    return result
    

    
""" Storage """


# Builds a tier assignment matrix by comparing CalSim storage data against historical thresholds
def generate_storage_tier_assignment_matrix(
    df, cdec_df, 
    tiers_output_dir, metrics_output_dir, tiers_output_filename, probabilities_output_filename,
    start_date="1921-10-01",
    percentiles=[0.5, 0.33, 0.25], tier_thresholds=(0.75, 0.67, 0.33),
    saveprobs=False, verbose=False, continuous = False, DropScenarios = []
):
    """
    Generate tier assignment matrix for reservoir storage.

    Parameters
    ----------
    df : pd.DataFrame
        CalSim data with flattened column names.
    cdec_df : pd.DataFrame
        CDEC metadata with CalSim_Variable and filename columns.
    start_date : str
        Start date for index if not datetime.
    percentiles : list
        Percentiles for historical thresholds.
    tier_thresholds : tuple
        Thresholds for tier assignment (tt1, tt2, tt3).
    saveprobs : bool
        If True, save probability matrix to CSV.
    verbose : bool
        If True, print detailed processing info.

    Returns
    -------
    tuple of pd.DataFrame
        (tier_matrix, prob_matrix)
    """
    def load_historical_storage_csv(filepath):
        df_raw = pd.read_csv(filepath, header=None)
        start_row = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains('RESERVOIR STORAGE').any(), axis=1)].index[0]
        df_data = pd.read_csv(filepath, skiprows=start_row)
        df_data.columns = df_data.columns.str.strip()
        df_data["DATE"] = pd.to_datetime(df_data.iloc[:, 0], format="%Y-%m-%d", errors="coerce")
        df_data = df_data.dropna(subset=["DATE"])
        storage_col = next((col for col in df_data.columns if "RESERVOIR STORAGE" in col.upper()), None)
        df_data["STORAGE"] = pd.to_numeric(df_data[storage_col], errors="coerce")
        df_data = df_data.dropna(subset=["STORAGE"])
        return df_data[["DATE", "STORAGE"]]
    # RETIRED 061726
    # def extract_historical_thresholds(df, percentiles):
    #     may = df[df["DATE"].dt.month == 5]
    #     may_1 = may.groupby(may["DATE"].dt.year).first()
    #     thresholds = may_1["STORAGE"].quantile(percentiles)
    #     return thresholds / 1000  # Convert AF to TAF
        
    def extract_historical_thresholds(df, percentiles, start_year=None, end_year=None):
        apr = df[df["DATE"].dt.month == 4]
        if start_year is not None:
            apr = apr[apr["DATE"].dt.year >= start_year]
        if end_year is not None:
            apr = apr[apr["DATE"].dt.year <= end_year]
        apr_by_year = apr.groupby(apr["DATE"].dt.year).first()
        return apr_by_year["STORAGE"].quantile(percentiles) / 1000 # AF to TAF
        
    def extract_variable_by_scenario(df, variable):
        return df[
            [col for col in df.columns
             if variable in col and "_STORAGE_" in col and "LEVEL" not in col.upper()]
        ]
# RETIRING THIS VERSION 061226
    # def assign_tiers_from_calsim(
    #     var_df,
    #     thresholds,
    #     date_series,
    #     var,
    #     tier_thresholds,
    #     saveprobs=None,
    #     verbose=False,
    #     continuous=False
    # ):
    #     """
    #     Assign storage tiers from CalSim output.
    
    #     Continuous interpretation:
    
    #         Tier 1 -> [1,2)
    #         Tier 2 -> [2,3)
    #         Tier 3 -> [3,4)
    #         Tier 4 -> [4,5)
    
    #     Lower value = stronger within tier
    #     Higher value = weaker within tier
    #     """
    
    #     tier_rows = []
    
    #     tt1, tt2, tt3 = tier_thresholds
    #     eps = 1e-6
    
    #     for col in var_df.columns:
    
    #         match = re.search(r's\d{4}', col)
    #         if not match:
    #             continue
    
    #         sid = match.group(0)
    
    #         series = var_df[col].copy()
    
    #         if not pd.api.types.is_datetime64_any_dtype(series.index):
    #             series.index = date_series
    
    #         april_series = series[series.index.month == 4]
    #         april_by_year = april_series.groupby(april_series.index.year).last()
    
    #         if april_by_year.empty:
    #             continue
    
    #         # This is reversed!
    #         # low_thresh = thresholds[percentiles[0]]
    #         # mid_thresh = thresholds[percentiles[1]]
    #         # high_thresh = thresholds[percentiles[2]]
    #         # Should be:
    #         high_thresh = thresholds[percentiles[0]]
    #         mid_thresh = thresholds[percentiles[1]]
    #         low_thresh = thresholds[percentiles[2]]
    
    #         top = (april_by_year >= high_thresh).sum()
    #         mid = ((april_by_year >= mid_thresh) &
    #                (april_by_year < high_thresh)).sum()
    #         low = ((april_by_year >= low_thresh) &
    #                (april_by_year < mid_thresh)).sum()
    #         bot = (april_by_year < low_thresh).sum()
    
    #         total = len(april_by_year)
    
    #         top_frac = top / total
    #         mid_frac = mid / total
    #         low_frac = low / total
    #         bot_frac = bot / total
    
    #         cumulative_top = top_frac
    #         cumulative_mid = top_frac + mid_frac
    #         cumulative_low = top_frac + mid_frac + low_frac
            
    #         # print(f"\nScenario {sid} ({var}):")
    #         # print("top_frac: " + str(top_frac) + ", mid_frac: " + str(mid_frac) + ", low_frac: " + str(low_frac) + "bot_frac: " + str(bot_frac))
    #         # print("cumulative_top: " + str(cumulative_top) + ", cumulative_mid: " + str(cumulative_mid) + ", low_frac: " + str(cumulative_low))
    
    #         if verbose:
    #             print(f"\nScenario {sid} ({var}):")
    #             print(
    #                 f"top_frac={top_frac:.3f}, "
    #                 f"mid_frac={mid_frac:.3f}, "
    #                 f"low_frac={low_frac:.3f}, "
    #                 f"bot_frac={bot_frac:.3f}"
    #             )
    #             print(
    #                 f"cum_top={cumulative_top:.3f}, "
    #                 f"cum_mid={cumulative_mid:.3f}"
    #             )
    
    #         # ============================================================
    #         # DISCRETE
    #         # ============================================================
    #         if cumulative_top >= tt1:
    #             discrete_tier = 1
    #         elif cumulative_mid >= tt2:
    #             discrete_tier = 2
    #         # elif cumulative_mid >= tt3:
    #         elif cumulative_low >= tt3:
    #             discrete_tier = 3
    #         else:
    #             discrete_tier = 4
    
    #         # ============================================================
    #         # CONTINUOUS
    #         # ============================================================
    #         if continuous:
    
    #             # if discrete_tier == 1:
    
    #             #     denom = 1 - tt1
    #             #     progress = 0 if denom <= 0 else (1 - cumulative_top) / denom
    
    #             # elif discrete_tier == 2:
    
    #             #     denom = tt1 - tt2
    #             #     progress = 0 if denom <= 0 else (tt1 - cumulative_mid) / denom
    
    #             # elif discrete_tier == 3:
    
    #             #     denom = tt2 - tt3
    #             #     # progress = 0 if denom <= 0 else (tt2 - cumulative_mid) / denom
    #             #     progress = 0 if denom <= 0 else (tt2 - cumulative_low) / denom
    
    #             # else:
    
    #             #     # progress = 0 if tt3 <= 0 else (tt3 - cumulative_mid) / tt3
    #             #     progress = 0 if tt3 <= 0 else (tt3 - cumulative_low) / tt3

    #             if discrete_tier == 1:
    #                 deficit = 1 - cumulative_top
    #                 surplus = cumulative_top - tt1
    #                 progress = deficit / (deficit + surplus)
    #             elif discrete_tier == 2:
    #                 deficit = tt1 - cumulative_top
    #                 surplus = cumulative_mid - tt2
    #                 progress = deficit / (deficit + surplus)
    #            	elif discrete_tier == 3:
    #                 deficit = tt2 - cumulative_mid
    #                 surplus = cumulative_low - tt3
    #                 progress = deficit / (deficit + surplus)
    #             else:
    #                 deficit = tt3 - cumulative_low
    #                 surplus = cumulative_low 
    #                 progress = deficit / (deficit + surplus)

    #             # Strict interior clipping
    #             progress = np.clip(progress, eps, 1 - eps)
    
    #             tier = discrete_tier + progress
    
    #             if verbose:
    #                 print(
    #                     f"discrete={discrete_tier}, "
    #                     f"progress={progress:.6f}, "
    #                     f"tier={tier:.6f}"
    #                 )
    
    #         else:
    #             tier = discrete_tier
    
    #         # print("progress: " + str(progress) + ", tier: " + str(tier))
            
    #         tier_rows.append({
    #             "Scenario": sid,
    #             "Variable": var,
    #             "TopProb": round(top_frac, 3),
    #             "MidProb": round(mid_frac, 3),
    #             "LowProb": round(low_frac, 3),
    #             "BotProb": round(bot_frac, 3),
    #             "StorageTier": tier
    #         })
    
    #     return pd.DataFrame(tier_rows).drop_duplicates(
    #         subset=["Scenario", "Variable"]
    #     )

# NEW VERSION 061226:
    def assign_tiers_from_calsim(
        var_df,
        thresholds,
        date_series,
        var,
        tier_thresholds,
        saveprobs=None,
        verbose=False,
        continuous=False
    ):
        """
        Assign storage tiers - v3 two-threshold combined distance (JG 2026-06-12).
    
        D = (1 - frac_top) + frac_bot is the combined distance from the ideal
        corner (every April at/above the high bar, never at/below the low bar).
        Tier cutpoints on the D axis are derived from ``tier_thresholds`` =
        3 (min_frac_above_high, max_frac_below_low) pairs, best tier first;
        tier 4 runs from the last cutpoint to the worst feasible corner.
    
        Continuous interpretation:
    
            Tier 1 -> [1,2)
            Tier 2 -> [2,3)
            Tier 3 -> [3,4)
            Tier 4 -> [4,5)
    
        Lower value = stronger within tier. The retired v2 logic (three-bar
        ladder + deficit/surplus continuous) is archived in comments at the
        bottom of this function.
        """
    
        tier_rows = []
        eps = 1e-6
    
        # with the default standards ((0.50, 0.20), (0.33, 0.25), (0.25, 0.33)):
        (top1, bot1), (top2, bot2), (top3, bot3) = tier_thresholds
    
        tier1_cutoff = (1.0 - top1) + bot1 # (1 - 0.50) + 0.20 = 0.70
        tier2_cutoff = (1.0 - top2) + bot2 # (1 - 0.33) + 0.25 = 0.92
        tier3_cutoff = (1.0 - top3) + bot3 # (1 - 0.25) + 0.33 = 1.08
    
    
        worst_frac_top = 0.0
        worst_frac_bot = 1.0
        d_worst = (1.0 - worst_frac_top) + worst_frac_bot   # (1 - 0) + 1 = 2.0  worst feasible D
    
        tier1_width = tier1_cutoff - 0.0 # 0.70
        tier2_width = tier2_cutoff - tier1_cutoff # 0.92 - 0.70 = 0.22
        tier3_width = tier3_cutoff - tier2_cutoff # 1.08 - 0.92 = 0.16
        tier4_width = d_worst - tier3_cutoff # 2.0 - 1.08 = 0.92
    
        if min(tier1_width, tier2_width, tier3_width, tier4_width) <= 0:
            raise ValueError(
                f"assign_tiers_from_calsim: tier standards {tier_thresholds} yield "
                f"non-increasing cutoffs ({tier1_cutoff}, {tier2_cutoff}, {tier3_cutoff}).")
    
        high_thresh = thresholds[percentiles[0]]
        low_thresh = thresholds[percentiles[-1]]
    
        for col in var_df.columns:
    
            match = re.search(r's\d{4}', col)
            if not match:
                continue
            sid = match.group(0)
    
            series = var_df[col].copy()
            if not pd.api.types.is_datetime64_any_dtype(series.index):
                series.index = date_series
    
            april_series = series[series.index.month == 4]
            april_by_year = april_series.groupby(april_series.index.year).last().dropna()
            if april_by_year.empty:
                print(f"WARNING [assign_tiers_from_calsim]: {var} {sid}: no April values after dropna - skipped.")
                continue
    
            frac_top = (april_by_year >= high_thresh).mean()
            frac_bot = (april_by_year <= low_thresh).mean()
            frac_mid = 1.0 - frac_top - frac_bot
    
            # combined distance from the best corner (frac_top = 1, frac_bot = 0)
            # combined penalty: shortfall from always-above-high + excess below low
            # (JG's dist1 = 1 - frac_top + frac_bot)
            d_combined = (1.0 - frac_top) + frac_bot
    
            # distance past each tier boundary (negative = boundary not reached)
            dist1 = d_combined # 1 - frac_top + frac_bot
            dist2 = d_combined - tier1_cutoff # (0.5 - frac_top) + (frac_bot - 0.20)
            dist3 = d_combined - tier2_cutoff  # (0.33 - frac_top) + (frac_bot - 0.25)
            dist4 = d_combined - tier3_cutoff  #(0.25 - frac_top) + (frac_bot - 0.33)
    
            # first segment that contains D wins; progress = position within that
            # segment as a fraction of its width (0 = good edge, 1 = bad edge)
            if dist1 <= tier1_width + eps: # dist1 <= 0.70: at least as good as the T1 standard
                discrete_tier = 1
                progress = dist1 / tier1_width
            elif dist2 <= tier2_width + eps: # dist2 <= 0.22
                discrete_tier = 2
                progress = dist2 / tier2_width
            elif dist3 <= tier3_width + eps: # dist3 <= 0.16
                discrete_tier = 3
                progress = dist3 / tier3_width
            else:
                discrete_tier = 4
                progress = dist4 / tier4_width
    
            if continuous:
                progress = np.clip(progress, eps, 1 - eps)
                tier = discrete_tier + progress
            else:
                tier = discrete_tier
    
            if verbose:
                print(f"\nScenario {sid} ({var}):")
                print(
                    f"frac_top={frac_top:.3f}, "
                    f"frac_bot={frac_bot:.3f}, "
                    f"frac_mid={frac_mid:.3f}, "
                    f"D={d_combined:.3f}, "
                    f"tier={tier:.6f}" if continuous else f"tier={tier}"
                )
    
            tier_rows.append({
                "Scenario": sid,
                "Variable": var,
                "FracTop": round(frac_top, 3),
                "FracBot": round(frac_bot, 3),
                "FracMid": round(frac_mid, 3),
                "StorageTier": tier
            })
    
        return pd.DataFrame(tier_rows).drop_duplicates(
            subset=["Scenario", "Variable"]
        )
    
    try:
        base_model_dir = find_calsim_model_root()
    except FileNotFoundError as e:
        print(e)
        return pd.DataFrame(), pd.DataFrame()

    hist_data_dir = os.path.join(base_model_dir, "Scenarios", "CDEC_Historical_Monthly_Storage")
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.date_range(start=start_date, periods=len(df), freq="MS")
    df["DATE"] = df.index

    tier_matrix = pd.DataFrame()
    prob_matrix = pd.DataFrame()

    for _, row in cdec_df.iterrows():
        var = row["CalSim_Variable"]
        file = row["filename"]
        label = f"{var}_Storage"

        if verbose:
            print(f"\n Processing reservoir: {row['ReservoirName']}")
            print(f"  ↳ CalSim variable: {var}")
            print(f"  ↳ Historical file: {file}")

        try:
            hist_path = os.path.join(hist_data_dir, file)
            hist_df = load_historical_storage_csv(hist_path)
            # Entire record:
            # thresholds = extract_historical_thresholds(hist_df, percentiles)
            # Alternative:
            thresholds = extract_historical_thresholds(hist_df, percentiles, start_year = 1990, end_year = 2024)
            if verbose:
                print(f"  ↳ Historical thresholds: {thresholds.to_dict()}")

            var_df = extract_variable_by_scenario(df, var)
            if verbose:
                print(f"  ↳ Matched CalSim columns: {var_df.columns.tolist()}")

            if var_df.empty:
                print(f" No CalSim data found for variable {var}")
                continue

            tier_df = assign_tiers_from_calsim(var_df, thresholds, df["DATE"], var, tier_thresholds, continuous = continuous)

            # for _, r in tier_df.iterrows():
            #     sid = r["Scenario"]
            #     prob_matrix.loc[sid, f"{label}_TopProb"] = r["TopProb"]
            #     prob_matrix.loc[sid, f"{label}_MidProb"] = r["MidProb"]
            #     prob_matrix.loc[sid, f"{label}_LowProb"] = r["LowProb"]
            #     prob_matrix.loc[sid, f"{label}_BotProb"] = r["BotProb"]
            #     tier_matrix.loc[sid, f"{label}_Tier"] = r["StorageTier"]
            for _, r in tier_df.iterrows():
                sid = r["Scenario"]
                prob_matrix.loc[sid, f"{label}_FracTop"] = r["FracTop"]
                prob_matrix.loc[sid, f"{label}_FracBot"] = r["FracBot"]
                prob_matrix.loc[sid, f"{label}_FracMid"] = r["FracMid"]
                tier_matrix.loc[sid, f"{label}_Tier"] = r["StorageTier"]
                
        except Exception as e:
            print(f" Failed to process {var}: {e}")
            continue

    tier_matrix.index.name = "Scenario"
    prob_matrix.index.name = "Scenario"

    # DROP UNWANTED SCENARIOS            
    # Only drop if the list contains items
    if DropScenarios:
        tier_matrix = tier_matrix.drop(DropScenarios, errors='ignore')
        prob_matrix = prob_matrix.drop(DropScenarios, errors='ignore')

    if verbose:
        print("tier_matrix:")
        print(tier_matrix.head(2))
        print("prob_matrix:")
        print(prob_matrix.head(2))

    # check if output directory exists
    if not os.path.exists(tiers_output_dir):
        print("Warning: directory " + tiers_output_dir + " does not exist and will be created")
        os.makedirs(tiers_output_dir)

    if not os.path.exists(metrics_output_dir):
        print("Warning: directory " + metrics_output_dir + " does not exist and will be created")
        os.makedirs(metrics_output_dir)

    tiers_output_path = os.path.join(tiers_output_dir, tiers_output_filename)
    metrics_output_path = os.path.join(metrics_output_dir, probabilities_output_filename)
    tier_matrix.to_csv(tiers_output_path)
    print(f"\n Tier assignment CSV saved to:\n{tiers_output_path}")
    if saveprobs:
        prob_matrix.to_csv(metrics_output_path)
        print(f"\n Level probabilities CSV saved to:\n{metrics_output_path}")

    return tier_matrix, prob_matrix


""" Groundwater """


# Parses WRESL mappings to link Subregion IDs (SRxx) to Water Balance Areas (WBAxx)
def parse_wresl_mappings(wresl_path: str):
    with open(wresl_path, "r") as f:
        wresl_lines = f.readlines()

    sr_to_wba_map = {}
    mapping_records = []

    for line in wresl_lines:
        raw = line.strip()
        if not raw or raw.startswith("!") or raw.startswith("#"):
            continue
        match_eq = re.search(r"indxWBA_([0-9A-Za-z]+)\s*=\s*(SR\d+)", raw, re.IGNORECASE)
        if match_eq:
            wba_id, sr_key = match_eq.groups()
            sr_to_wba_map[sr_key] = f"WBA{wba_id}"
            mapping_records.append({"Subregion_ID": sr_key, "WBA_ID": f"WBA{wba_id}"})
            continue
        match_def = re.search(r"define\s+indxWBA_([0-9A-Za-z]+)\s*\{\s*value\s+(\d+)\s*\}", raw, re.IGNORECASE)
        if match_def:
            wba_id, sr_num = match_def.groups()
            sr_key = f"SR{int(sr_num):02d}"
            sr_to_wba_map[sr_key] = f"WBA{wba_id}"
            mapping_records.append({"Subregion_ID": sr_key, "WBA_ID": f"WBA{wba_id}"})
            continue
        # Handling DETAW direct assignment
        match_det_eq = re.search(r"indxDETAW\s*=\s*(SR\d+)", raw, re.IGNORECASE)
        if match_det_eq:
            sr_key = match_det_eq.group(1)
            sr_to_wba_map[sr_key] = "DETAW"
            mapping_records.append({"Subregion_ID": sr_key, "WBA_ID": "DETAW"})
            continue
        match_det_def = re.search(r"define\s+indxDETAW\s*\{\s*value\s+(\d+)\s*\}", raw, re.IGNORECASE)
        if match_det_def:
            sr_num = match_det_def.group(1)
            sr_key = f"SR{int(sr_num):02d}"
            sr_to_wba_map[sr_key] = "DETAW"
            mapping_records.append({"Subregion_ID": sr_key, "WBA_ID": "DETAW"})
            continue

    # Build DataFrames from parsed mappings
    if sr_to_wba_map:
        mapping_df = pd.DataFrame([{"Subregion_ID": k, "WBA_ID": v} for k, v in sr_to_wba_map.items()])
    else:
        mapping_df = pd.DataFrame(columns=["Subregion_ID", "WBA_ID"])

    if mapping_records:
        wresl_df = pd.DataFrame(mapping_records)
        if "Subregion_ID" in wresl_df.columns:
            wresl_df["Subregion_ID"] = wresl_df["Subregion_ID"].astype(str)
            try:
                wresl_df = wresl_df.sort_values(
                    by=["Subregion_ID"],
                    key=lambda s: s.str.extract(r"(\d+)")[0].astype(int)
                )
            except Exception:
                wresl_df = wresl_df.sort_values(by=["Subregion_ID"])
    else:
        print(" Warning: No mappings found. Check your WRESL file formatting.")
        wresl_df = pd.DataFrame(columns=["Subregion_ID", "WBA_ID"])

    print(f" Parsed {len(mapping_records)} mappings from {wresl_path}")
    print(wresl_df.head())
    return mapping_df, wresl_df, sr_to_wba_map


def load_gw1_df(csv_path: Path) -> pd.DataFrame:
    """Load GW1 dataframe with multi-index columns if present."""
    na_vals = ["", "NA", "NaN", "nan", "-", "--"]
    # Read the multi-level column headers
    try:
        df = pd.read_csv(
            csv_path,
            header=[0, 1, 2, 3, 4],
            index_col=0,
            parse_dates=True,
            low_memory=False,
            na_values=na_vals,
        )
    except Exception:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True, na_values=na_vals)
        if len(df.columns) and isinstance(df.columns[0], str) and "|" in df.columns[0]:
            tuples = [tuple(str(c).split("|")) for c in df.columns]
            if all(len(t) == 5 for t in tuples):
                df.columns = pd.MultiIndex.from_tuples(
                    tuples, names=["Model", "VarTag", "Type", "Timestep", "Unit"]
                )
    return df


# Normalization helpers
def normalize_id(s: str) -> str:
    """Normalize WBA_ID or SR number to two-digit uppercase format."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().upper()
    if s.startswith("WBA"):
        s = s[3:]
    return s.zfill(2) if s.isdigit() else s


def normalize_storage_col(col: str) -> Optional[str]:
    """Extract WBA ID or DETAW identifier from storage column names."""
    if not isinstance(col, str):
        return None
    if col.upper().startswith("WBA"):
        s = col.replace("WBA", "", 1)
        for suffix in ("_STORAGE_AF", "_Storage_AF", "_storage_af"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                break
        s = s.strip().upper()
        return s.zfill(2) if s.isdigit() else s
    elif col.upper().startswith("DETAW"):
        return "DETAW"
    return None


# Function to compute depth (ft) timeseries and combine outputs
def build_combined_storage_timeseries(
        gw_csv_path: Path,
        wba_csv_path: Path,
        wba_storage_csv_path: Path,
        mapping_df: pd.DataFrame,
        window_start: str,
        window_end: str,
        start_year: int,
        data_output_dir: Path,
        monthly_filename: str = "combined_monthly.csv",
        annual_filename: str = "combined_annual.csv",
        detaw_area_acres: Optional[float] = None,
        use_total_wba_acres_for_detaw: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine groundwater and storage data into consistent FT (depth) units.
    Returns monthly and annual DataFrames.
    """

    # Load inputs
    gw1_df = load_gw1_df(gw_csv_path)
    wba_df = pd.read_csv(wba_csv_path)

    required_cols = {"fid", "GIS_Acres", "WBA_ID"}
    missing = required_cols - set(wba_df.columns)
    if missing:
        raise ValueError(f"{wba_csv_path.name} missing columns: {missing}")

    # Handle either naming style for mapping_df
    if {"SR_number", "WBA_name"}.issubset(mapping_df.columns):
        mapping_df = mapping_df.rename(columns={"SR_number": "Subregion_ID", "WBA_name": "WBA_ID"})

    if not {"Subregion_ID", "WBA_ID"}.issubset(mapping_df.columns):
        raise KeyError(
            f"mapping_df missing required columns Subregion_ID/WBA_ID. "
            f"Got columns: {mapping_df.columns.tolist()}"
        )

    # Build lookup dictionaries
    sr_to_fid = {f"SR{int(fid):02d}": fid for fid in wba_df["fid"]}
    fid_to_acres = dict(zip(wba_df["fid"], wba_df["GIS_Acres"]))
    sr_to_wba = dict(zip(mapping_df["Subregion_ID"], mapping_df["WBA_ID"]))

    # Build WBA/DETAW series from GW1 
    series_map = {}
    for col in gw1_df.columns:
        if not isinstance(col, tuple) or len(col) < 5:
            continue
        model, var_tag, var_type, timestep, unit = col[:5]

        # WBA series: convert TAF → ft
        if isinstance(var_tag, str) and re.match(r"^SR\d+:TOT_s\d{4}$", var_tag):
            sr, rest = var_tag.split(":")
            scen_raw = rest.split("_s")[-1]
            scenario = f"s{int(scen_raw):04d}"

            if sr not in sr_to_wba:
                continue
            wba_name = sr_to_wba[sr]
            fid = sr_to_fid.get(sr)
            area_acres = fid_to_acres.get(fid)
            if area_acres is None or float(area_acres) == 0.0:
                continue

            series = pd.to_numeric(gw1_df[col], errors="coerce")
            series = series[series.index >= pd.Timestamp(f"{start_year}-01-01")]
            values_ft = (series / float(area_acres)) * 1000.0  # TAF → AF → FT
            # Truncate negative values (indicating missing or invalid data)
            neg_idx = np.where(values_ft < 0)[0]
            if len(neg_idx) > 0:
                values_ft = values_ft.iloc[: neg_idx[0]]

            series_map[f"{wba_name}_{scenario}"] = values_ft

        # DETAW series: convert AF → ft
        elif isinstance(var_tag, str) and re.match(r"^DETAW:TOT_s\d{4}$", var_tag):
            scen_raw = var_tag.split("_s")[-1]
            scenario = f"s{int(scen_raw):04d}"

            series_af = pd.to_numeric(gw1_df[col], errors="coerce")
            series_af = series_af[series_af.index >= pd.Timestamp(f"{start_year}-01-01")]

            if detaw_area_acres is not None:
                detaw_area = float(detaw_area_acres)
            elif use_total_wba_acres_for_detaw:
                detaw_area = float(pd.to_numeric(wba_df["GIS_Acres"], errors="coerce").sum())
            else:
                raise ValueError("DETAW area unknown - set detaw_area_acres or enable use_total_wba_acres_for_detaw.")

            values_ft = series_af / detaw_area
            series_map[f"DETAW_{scenario}"] = values_ft

    # Fallback: if no TOT columns matched, sum L1+L2+L3 layers per (SR, scenario)
    if not series_map:
        layer_data = {}  # key: (sr_normalized, scenario) -> list of series
        detaw_layers = {}  # key: scenario -> list of series
        for col in gw1_df.columns:
            if not isinstance(col, tuple) or len(col) < 5:
                continue
            _model, var_tag, _var_type, _timestep, _unit = col[:5]
            if not isinstance(var_tag, str):
                continue

            # SR layer columns (e.g. SR1:L1_s0002)
            if re.match(r"^SR\d+:L[123]_s\d{4}$", var_tag):
                sr_raw, rest = var_tag.split(":")
                sr_num = re.search(r"\d+", sr_raw).group()
                sr = f"SR{int(sr_num):02d}"  # normalize: SR1 -> SR01
                scen_raw = rest.split("_s")[-1]
                scenario = f"s{int(scen_raw):04d}"
                series = pd.to_numeric(gw1_df[col], errors="coerce")
                series = series[series.index >= pd.Timestamp(f"{start_year}-01-01")]
                layer_data.setdefault((sr, scenario), []).append(series)

            # DETAW layer columns (e.g. DETAW:L1_s0002)
            elif re.match(r"^DETAW:L[123]_s\d{4}$", var_tag):
                scen_raw = var_tag.split("_s")[-1]
                scenario = f"s{int(scen_raw):04d}"
                series = pd.to_numeric(gw1_df[col], errors="coerce")
                series = series[series.index >= pd.Timestamp(f"{start_year}-01-01")]
                detaw_layers.setdefault(scenario, []).append(series)

        # Sum L1+L2+L3 for each (SR, scenario)
        for (sr, scenario), series_list in layer_data.items():
            if len(series_list) < 3:
                continue  # require all 3 layers
            summed = series_list[0].add(series_list[1], fill_value=0).add(series_list[2], fill_value=0)
            if sr not in sr_to_wba:
                continue
            wba_name = sr_to_wba[sr]
            fid = sr_to_fid.get(sr)
            area_acres = fid_to_acres.get(fid)
            if area_acres is None or float(area_acres) == 0.0:
                continue
            series_ft = (summed / float(area_acres)) * 1000.0  # TAF -> FT
            neg_idx = np.where(series_ft < 0)[0]
            if len(neg_idx) > 0:
                series_ft = series_ft.iloc[:neg_idx[0]]
            series_map[f"{wba_name}_{scenario}"] = series_ft

        # Sum DETAW layers
        for scenario, series_list in detaw_layers.items():
            if len(series_list) < 3:
                continue
            summed = series_list[0].add(series_list[1], fill_value=0).add(series_list[2], fill_value=0)
            if detaw_area_acres is not None:
                detaw_area = float(detaw_area_acres)
            elif use_total_wba_acres_for_detaw:
                detaw_area = float(pd.to_numeric(wba_df["GIS_Acres"], errors="coerce").sum())
            else:
                raise ValueError("DETAW area unknown - set detaw_area_acres or enable use_total_wba_acres_for_detaw.")
            series_map[f"DETAW_{scenario}"] = summed / detaw_area

    monthly_from_tot = (
        pd.concat(series_map, axis=1).loc[window_start:window_end]
        if series_map
        else pd.DataFrame(index=pd.date_range(window_start, window_end, freq="MS"))
    )

    # Build s0000 (storage baseline)
    storage_df = pd.read_csv(wba_storage_csv_path, index_col=0, parse_dates=True)
    wba_df["WBA_ID_norm"] = wba_df["WBA_ID"].apply(normalize_id)
    acres_map = (
        wba_df[["WBA_ID_norm", "GIS_Acres"]]
        .dropna()
        .drop_duplicates(subset=["WBA_ID_norm"])
        .set_index("WBA_ID_norm")["GIS_Acres"]
        .astype("float64")
        .to_dict()
    )

    s0000_result = {}
    for raw_col in storage_df.columns:
        norm = normalize_storage_col(raw_col)
        if norm is None:
            continue

        af_series = pd.to_numeric(storage_df[raw_col], errors="coerce")

        if norm == "DETAW":
            # Compute DETAW area 
            if detaw_area_acres is not None:
                detaw_area = float(detaw_area_acres)
            elif use_total_wba_acres_for_detaw:
                detaw_area = float(pd.to_numeric(wba_df["GIS_Acres"], errors="coerce").sum())
            else:
                raise ValueError("Missing DETAW area for s0000 conversion.")
            ft_series = af_series / detaw_area
            s0000_result["DETAW_s0000"] = ft_series
        else:
            if norm not in acres_map:
                continue
            acres = float(acres_map[norm])
            if acres <= 0.0:
                continue
            ft_series = af_series / acres
            s0000_result[f"WBA{norm}_s0000"] = ft_series

    s0000_df = (
        pd.DataFrame(s0000_result, index=storage_df.index).loc[window_start:window_end]
        if s0000_result
        else pd.DataFrame(index=storage_df.index)
    )

    # Combine and export
    combined_monthly = pd.concat([monthly_from_tot, s0000_df], axis=1).sort_index(axis=1)
    combined_monthly = combined_monthly.loc[window_start:window_end]
    combined_monthly.to_csv(Path(data_output_dir) / monthly_filename)

    combined_annual = (
        combined_monthly.resample("YE").mean() if not combined_monthly.empty else pd.DataFrame()
    )
    combined_annual.index = combined_annual.index.year
    combined_annual.to_csv(Path(data_output_dir) / annual_filename)

    return combined_monthly, combined_annual


def build_gw_timeseries(
    gw_csv_path: Path,
    wba_csv_path: Path,
    wba_storage_csv_path: Path,
    mapping_df: pd.DataFrame,
    window_start: str,
    window_end: str,
    start_year: int,
    detaw_area_acres: Optional[float] = None,
    use_total_wba_acres_for_detaw: bool = True,
    use_cv2sim = False
) -> tuple:
    """
    Build FT and AF timeseries from GW data.

    Returns
    -------
    tuple of (ft_monthly, ft_annual, af_monthly, af_annual)
    """
    gw1_df = load_gw1_df(gw_csv_path)
    wba_df = pd.read_csv(wba_csv_path)

    required_cols = {"fid", "GIS_Acres", "WBA_ID"}
    missing = required_cols - set(wba_df.columns)
    if missing:
        raise ValueError(f"WBA CSV missing columns: {missing}")

    # Handle mapping_df column names
    if {"SR_number", "WBA_name"}.issubset(mapping_df.columns):
        mapping_df = mapping_df.rename(columns={"SR_number": "Subregion_ID", "WBA_name": "WBA_ID"})

    # Build lookup dictionaries
    sr_to_fid = {f"SR{int(fid):02d}": fid for fid in wba_df["fid"]}
    fid_to_acres = dict(zip(wba_df["fid"], wba_df["GIS_Acres"]))
    sr_to_wba = dict(zip(mapping_df["Subregion_ID"], mapping_df["WBA_ID"]))

    # Compute DETAW area
    if detaw_area_acres is not None:
        detaw_area = float(detaw_area_acres)
    elif use_total_wba_acres_for_detaw:
        detaw_area = float(pd.to_numeric(wba_df["GIS_Acres"], errors="coerce").sum())
    else:
        raise ValueError("DETAW area unknown")

    ft_map = {}
    af_map = {}

    # Process scenario data from gw1_df
    for col in gw1_df.columns:
        if not isinstance(col, tuple) or len(col) < 5:
            continue
        model, var_tag, var_type, timestep, unit = col[:5]

        # WBA series (TAF -> FT and AF)
        if isinstance(var_tag, str) and re.match(r"^SR\d+:TOT_s\d{4}$", var_tag):
            sr, rest = var_tag.split(":")
            scen_raw = rest.split("_s")[-1]
            scenario = f"s{int(scen_raw):04d}"

            if sr not in sr_to_wba:
                continue
            wba_name = sr_to_wba[sr]
            fid = sr_to_fid.get(sr)
            area_acres = fid_to_acres.get(fid)
            if area_acres is None or float(area_acres) == 0.0:
                continue

            series = pd.to_numeric(gw1_df[col], errors="coerce")
            series = series[series.index >= pd.Timestamp(f"{start_year}-01-01")]

            # Truncate at first negative value
            neg_idx = np.where(series < 0)[0]
            if len(neg_idx) > 0:
                series = series.iloc[:neg_idx[0]]

            col_name = f"{wba_name}_{scenario}"
            af_map[col_name] = series * 1000.0  # TAF -> AF
            ft_map[col_name] = (series / float(area_acres)) * 1000.0  # TAF -> FT

        # DETAW series
        elif isinstance(var_tag, str) and re.match(r"^DETAW:TOT_s\d{4}$", var_tag):
            scen_raw = var_tag.split("_s")[-1]
            scenario = f"s{int(scen_raw):04d}"

            series_af = pd.to_numeric(gw1_df[col], errors="coerce")
            series_af = series_af[series_af.index >= pd.Timestamp(f"{start_year}-01-01")]

            col_name = f"DETAW_{scenario}"
            af_map[col_name] = series_af
            ft_map[col_name] = series_af / detaw_area

    # Fallback: if no TOT columns matched, sum L1+L2+L3 layers per (SR, scenario)
    if not ft_map:
        layer_data = {}  # key: (sr_normalized, scenario) -> list of series
        detaw_layers = {}  # key: scenario -> list of series
        for col in gw1_df.columns:
            if not isinstance(col, tuple) or len(col) < 5:
                continue
            _model, var_tag, _var_type, _timestep, _unit = col[:5]
            if not isinstance(var_tag, str):
                continue

            # SR layer columns (e.g. SR1:L1_s0002)
            if re.match(r"^SR\d+:L[123]_s\d{4}$", var_tag):
                sr_raw, rest = var_tag.split(":")
                sr_num = re.search(r"\d+", sr_raw).group()
                sr = f"SR{int(sr_num):02d}"  # normalize: SR1 -> SR01
                scen_raw = rest.split("_s")[-1]
                scenario = f"s{int(scen_raw):04d}"
                series = pd.to_numeric(gw1_df[col], errors="coerce")
                series = series[series.index >= pd.Timestamp(f"{start_year}-01-01")]
                layer_data.setdefault((sr, scenario), []).append(series)

            # DETAW layer columns (e.g. DETAW:L1_s0002)
            elif re.match(r"^DETAW:L[123]_s\d{4}$", var_tag):
                scen_raw = var_tag.split("_s")[-1]
                scenario = f"s{int(scen_raw):04d}"
                series = pd.to_numeric(gw1_df[col], errors="coerce")
                series = series[series.index >= pd.Timestamp(f"{start_year}-01-01")]
                detaw_layers.setdefault(scenario, []).append(series)

        # Sum L1+L2+L3 for each (SR, scenario)
        for (sr, scenario), series_list in layer_data.items():
            if len(series_list) < 3:
                continue  # require all 3 layers
            summed = series_list[0].add(series_list[1], fill_value=0).add(series_list[2], fill_value=0)
            if sr not in sr_to_wba:
                continue
            wba_name = sr_to_wba[sr]
            fid = sr_to_fid.get(sr)
            area_acres = fid_to_acres.get(fid)
            if area_acres is None or float(area_acres) == 0.0:
                continue
            neg_idx = np.where(summed < 0)[0]
            if len(neg_idx) > 0:
                summed = summed.iloc[:neg_idx[0]]
            col_name = f"{wba_name}_{scenario}"
            af_map[col_name] = summed * 1000.0  # TAF -> AF
            ft_map[col_name] = (summed / float(area_acres)) * 1000.0  # TAF -> FT

        # Sum DETAW layers
        for scenario, series_list in detaw_layers.items():
            if len(series_list) < 3:
                continue
            summed = series_list[0].add(series_list[1], fill_value=0).add(series_list[2], fill_value=0)
            col_name = f"DETAW_{scenario}"
            af_map[col_name] = summed
            ft_map[col_name] = summed / detaw_area

    # Add s0000 baseline from storage file
    if use_cv2sim:
        storage_df = pd.read_csv(wba_storage_csv_path, index_col=0, parse_dates=True)
        wba_df["WBA_ID_norm"] = wba_df["WBA_ID"].apply(normalize_id)
        acres_map = (
            wba_df[["WBA_ID_norm", "GIS_Acres"]]
            .dropna()
            .drop_duplicates(subset=["WBA_ID_norm"])
            .set_index("WBA_ID_norm")["GIS_Acres"]
            .astype("float64")
            .to_dict()
        )

        for raw_col in storage_df.columns:
            norm = normalize_storage_col(raw_col)
            if norm is None:
                continue
    
            af_series = pd.to_numeric(storage_df[raw_col], errors="coerce")
    
            if norm == "DETAW":
                ft_map["DETAW_s0000"] = af_series / detaw_area
                af_map["DETAW_s0000"] = af_series
            else:
                if norm not in acres_map:
                    continue
                acres = float(acres_map[norm])
                if acres <= 0.0:
                    continue
                ft_map[f"WBA{norm}_s0000"] = af_series / acres
                af_map[f"WBA{norm}_s0000"] = af_series

    # Combine into DataFrames
    ft_monthly = pd.concat(ft_map, axis=1).loc[window_start:window_end].sort_index(axis=1)
    af_monthly = pd.concat(af_map, axis=1).loc[window_start:window_end].sort_index(axis=1)

    ft_annual = ft_monthly.resample("YE").mean()
    ft_annual.index = ft_annual.index.year
    af_annual = af_monthly.resample("YE").mean()
    af_annual.index = af_annual.index.year

    return ft_monthly, ft_annual, af_monthly, af_annual


def compute_baseline_percent(df: pd.DataFrame, baseline_scenario: str = "s0000") -> pd.DataFrame:
    """
    Compute values as percentage of baseline scenario.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns like WBA2_s0001, WBA2_s0002, etc.
    baseline_scenario : str
        Scenario ID to use as baseline (default "s0000").

    Returns
    -------
    pd.DataFrame
        Percentage values (value / baseline * 100).
    """
    pct_map = {}

    for col in df.columns:
        if "_s" not in col:
            continue
        wba_part = col.rsplit("_s", 1)[0]
        baseline_col = f"{wba_part}_{baseline_scenario}"

        if baseline_col not in df.columns:
            continue

        baseline_series = df[baseline_col]
        pct_series = (df[col] / baseline_series) * 100
        pct_map[col] = pct_series

    if not pct_map:
        return pd.DataFrame()

    return pd.concat(pct_map, axis=1)


# Normalize WBA names for labeling consistency
def normalize_wba_name(wba: str) -> str:
    """
    Normalize WBA identifiers by removing leading zeros while preserving suffixes.
    Examples:
        WBA02   -> WBA2
        WBA07N  -> WBA7N
        WBA17S  -> WBA17S
        DETAW   -> DETAW (unchanged)
    """
    if not isinstance(wba, str):
        return str(wba)

    s = wba.strip().upper()
    if not s.startswith("WBA"):
        # Non-WBA names (e.g., DETAW) are returned unchanged
        return s

    core = s[3:]
    # Digits (with optional trailing letters)
    m = re.match(r"^(\d+)([A-Z].*)?$", core)
    if m:
        num = str(int(m.group(1)))  # drop leading zeros
        tail = m.group(2) or ""
        return f"WBA{num}{tail}"

    # Handle edge case like "WBA0N" -> "WBA N"
    m2 = re.match(r"^0+([A-Z].*)$", core)
    if m2:
        return f"WBA{m2.group(1)}"

    # Fallback (leave unchanged if unrecognized)
    return s


# Compute linear groundwater storage trends for each scenario
def compute_wba_trends(
        combined_monthly: pd.DataFrame,
        trends_output_dir: str,
        trend_filename: str = "GroundWater_Trends_ft_per_month.csv",
        diff_filename: str = "GroundWater_AvgEndStartDiff_ft.csv",
        years_to_average: int = 10
) -> tuple:
    """
    Compute linear trends (ft/month) and end-start diffs for each WBA_s#### timeseries.
    Saves and returns pivoted matrices (scenarios × WBAs) for both trends and diffs.

    Parameters
    ----------
    combined_monthly : pd.DataFrame
        Monthly FT timeseries with columns like WBA2_s0001, DETAW_s0002, etc.
    trends_output_dir : str
        Directory to save output CSVs.
    trend_filename : str
        Filename for trend matrix CSV.
    diff_filename : str
        Filename for diff matrix CSV.
    years_to_average : int
        Number of years to average at start and end for diff calculation.

    Returns
    -------
    tuple of (trend_matrix, diff_matrix)
    """
    time_numeric = np.arange(len(combined_monthly)).reshape(-1, 1)

    records = []
    for col in combined_monthly.columns:
        if "_s" not in col:
            continue
        wba_raw, scen_raw = col.split("_s", 1)
        scenario = f"s{scen_raw}"
        wba_norm = normalize_wba_name(wba_raw)

        ts = combined_monthly[col].dropna()
        y = combined_monthly[col].to_numpy(dtype=float)
        mask = ~np.isnan(y)

        slope = np.nan
        diff_ft = np.nan

        if mask.sum() > 1:
            # Compute trend (slope)
            model = LinearRegression().fit(time_numeric[mask], y[mask])
            slope = model.coef_[0]

            # Compute end-start diff
            ts = ts.sort_index()
            years = ts.index.year
            first_years = ts[years <= years.min() + years_to_average - 1]
            last_years = ts[years >= years.max() - years_to_average + 1]

            if len(first_years) > 0 and len(last_years) > 0:
                mean_start = first_years.mean()
                mean_end = last_years.mean()
                diff_ft = mean_end - mean_start

        records.append({
            "scenario": scenario,
            "WBA": wba_norm,
            "slope_ft_per_month": slope,
            "diff_last_vs_first_ft": diff_ft
        })

    records_df = pd.DataFrame.from_records(records)
    if records_df.empty:
        print(" No valid trend data found.")
        return pd.DataFrame(), pd.DataFrame()

    # Aggregate duplicates
    trends_df = (
        records_df
        .groupby(["scenario", "WBA"], as_index=False)["slope_ft_per_month"]
        .mean()
    )
    diff_df = (
        records_df
        .groupby(["scenario", "WBA"], as_index=False)["diff_last_vs_first_ft"]
        .mean()
    )

    # Pivot to matrix form
    trend_matrix = trends_df.pivot(index="scenario", columns="WBA", values="slope_ft_per_month")
    diff_matrix = diff_df.pivot(index="scenario", columns="WBA", values="diff_last_vs_first_ft")

    # Sort helper functions
    def scen_key(s):
        try:
            return int(str(s).lstrip("sS"))
        except Exception:
            return 10 ** 9

    def wba_key(w):
        w = str(w).upper()
        if w.startswith("WBA"):
            core = w[3:]
            m = re.match(r"^(\d+)([A-Z].*)?$", core)
            if m:
                return (int(m.group(1)), m.group(2) or "")
        return (10 ** 9, w)

    # Sort matrices
    trend_matrix = trend_matrix.reindex(sorted(trend_matrix.index, key=scen_key))
    trend_matrix = trend_matrix.reindex(sorted(trend_matrix.columns, key=wba_key), axis=1)
    diff_matrix = diff_matrix.reindex(sorted(diff_matrix.index, key=scen_key))
    diff_matrix = diff_matrix.reindex(sorted(diff_matrix.columns, key=wba_key), axis=1)

    # Save
    os.makedirs(trends_output_dir, exist_ok=True)
    trends_out_path = os.path.join(trends_output_dir, trend_filename)
    diff_out_path = os.path.join(trends_output_dir, diff_filename)
    trend_matrix.to_csv(trends_out_path)
    diff_matrix.to_csv(diff_out_path)

    print(f"Trends saved: {trends_out_path}")
    print(f"Diffs saved: {diff_out_path}")

    return trend_matrix, diff_matrix


# Assign groundwater storage tiers based on trends
def assign_tiers_from_trends(trend_matrix, baseline, output_dir, filename, severe_decline_threshold=-0.015, continuous = False, DropScenarios = []):
    if baseline not in trend_matrix.index:
        raise ValueError(f"Baseline scenario {baseline} not found in trend_matrix")

    tier_matrix = pd.DataFrame(index=trend_matrix.index, columns=trend_matrix.columns)

    for wba_col in trend_matrix.columns:
        baseline_slope = trend_matrix.loc[baseline, wba_col]

        for scenario in trend_matrix.index:
            slope = trend_matrix.loc[scenario, wba_col]
            # print("wba_col: " + str(wba_col) + ", scenario: " + str(scenario))
            # Determine tier category based on slope relative to baseline
            if pd.isna(slope) or pd.isna(baseline_slope):
                tier = np.nan
            elif scenario == baseline:
                tier = 0  # baseline tier
            #OUR VERSION (WITH A FIX)
            elif continuous:
                if slope >= 0:
                    if slope >= baseline_slope:
                        # print("slope positive, slope >= baseline (" + str(baseline_slope) + ")")
                        if baseline_slope >= 0:
                            tier = 1 + (baseline_slope / slope) # problem: what if baseline is negative
                            # Example to comare with below
                            # slope = 1.5, baseline slope = 0.5
                            # 1 + 0.5 / 1.5 = 1.33
                            # slope = 2, baseline slope = 0.5
                            # 1 + 0.5 / 2 = 1.25
                            # print("baseline_slope / slope = " + str(baseline_slope / slope) + ", tier = " + str(tier))
                        else:
                            # tier = 1 + (abs(baseline_slope)/(abs(baseline_slope)-baseline_slope)+slope) # Problem: is this not 1 + 0.5 + slope????
                            # tier = 1 + abs(baseline_slope)/(abs(baseline_slope)+slope) 
                            # this works but has issues in some cases
                            # let's say slope is 0.5 and baseline -0.5.
                            # this gives:
                            # 1 + 0.5 / (0.5 + 0.5) = 1 + 0.5 / 1 = 1.5
                            # if slope is 1 and baseline is -1 then:
                            # 1 + 1 / (1 + 1) = 1 + 1/2 = 1.5
                            # The second case should yield a number that is less than the first case
                            # Alternative:
                            tier = 1 + abs(baseline_slope)/(abs(baseline_slope)-baseline_slope+slope) 
                            # this seems to have the same issue
                            # let's say slope is 0.5 and baseline -0.5.
                            # this gives:
                            # 1 + 0.5 / (0.5 + 0.5 + 0.5) = 1 + 0.5 / 1.5 = 1.33
                            # if slope is 1 and baseline is -1 then:
                            # 1 + 1 / (1 + 1 + 1) = 1 + 1/3 = 1.33
                            # The second case should yield a number that is less than the first case.
                            # Simply because it is a lower number, we will go with the second version

                    else:
                        # print("slope positive, slope < baseline")
                        tier = 2 + (1 - (slope / baseline_slope)) # seems correct
                        # print("1 - (slope / baseline_slope) = " + str(1 - (slope / baseline_slope)) + ", tier = " + str(tier))
                else:
                    if slope > severe_decline_threshold:
                        # print("slope negative, slope >= threshold")
                        tier = 3 + (slope / severe_decline_threshold) # seems correct
                        # print("slope / severe_decline_threshold = " + str(slope / severe_decline_threshold) + ", tier = " + str(tier))
                    else:
                        # print("slope negative, slope < threshold")
                        # tier = max(4.999, 4 + (1 - (severe_decline_threshold / slope))) # Problem: tier 4s will always be 4.999
                        tier = min(4.999, 4 + (1 - (severe_decline_threshold / slope))) # seems correct
                        # print("1 - (severe_decline_threshold / slope) = " + str(1 - (severe_decline_threshold / slope)) + ", tier = " + str(tier))
            # # NEW TEST VERSION
            # # CONTINUOUS
            # elif continuous:
            
            #     # ------------------------------------------------
            #     # TIER 1:
            #     # Positive slope and >= reference
            #     # ------------------------------------------------
            #     if slope >= 0 and slope >= baseline_slope:
            
            #         improvement = slope - baseline_slope
            
            #         # bounded between 0 and 0.999999
            #         # large improvement -> closer to 1
            #         # small improvement -> closer to 2
            #         fraction = np.exp(-improvement)
            
            #         tier = 1 + min(0.999999, fraction)
                    # The problem is that we get values always near 1.99 because we are exponentiating very small numbers
            
            
            #     # ------------------------------------------------
            #     # TIER 2:
            #     # Positive slope but below reference
            #     # ------------------------------------------------
            #     elif slope >= 0:
            
            #         if baseline_slope > 0:
            
            #             # slope = baseline -> tier 2 boundary
            #             # slope = 0 -> approaches tier 3
            #             fraction = (baseline_slope - slope) / baseline_slope
            
            #             tier = 2 + min(0.999999, fraction)
            
            #         else:
            #             # If reference is negative, there is no positive Tier 2 range
            #             tier = 1.999999
            
            
            #     # ------------------------------------------------
            #     # TIER 3:
            #     # Negative slope but above severe decline threshold
            #     # ------------------------------------------------
            #     elif slope > severe_decline_threshold:
            
            #         # 0 -> tier 3
            #         # severe threshold -> approaches tier 4
            
            #         fraction = (
            #             0 - slope
            #         ) / (
            #             0 - severe_decline_threshold
            #         )
            
            #         tier = 3 + min(0.999999, fraction)
            
            
            #     # ------------------------------------------------
            #     # TIER 4:
            #     # Negative slope worse than severe threshold
            #     # ------------------------------------------------
            #     else:
            
            #         excess = (
            #             severe_decline_threshold - slope
            #         )
            
            #         # asymptotically approaches 4.999999
            #         fraction = 1 - np.exp(-excess)
            
            #         tier = 4 + min(0.999999, fraction)
        
            # DISCRETE
            else:
                if slope >= 0:
                    diff = slope - baseline_slope
                    tier = 1 if diff >= 0 else 2
                elif slope >= severe_decline_threshold:
                    tier = 3
                else:
                    tier = 4
            
            tier_matrix.loc[scenario, wba_col] = tier
            
    # DROP UNWANTED SCENARIOS            
    # Only drop if the list contains items
    if DropScenarios:
        tier_matrix = tier_matrix.drop(DropScenarios, errors='ignore')

    out_path = os.path.join(output_dir, filename)
    tier_matrix.to_csv(out_path)
    print("✓ Tier assignment saved to:", out_path)
    print(tier_matrix.head())

    return tier_matrix


""" FLOOD RISK TIER CLASSIFICATION """


def classify_flood_tier(prob, thresh1=0.10, thresh2=0.40):
    """
    Classify flood risk probability into tiers.

    Parameters
    ----------
    prob : float
        Probability that storage >= flood pool level.
    thresh1 : float
        Threshold between Tier 1 and Tier 2 (default 0.10).
    thresh2 : float
        Threshold between Tier 2 and Tier 3 (default 0.40).

    Returns
    -------
    str or np.nan
        Tier classification: "1", "2", "3", or np.nan if input is NaN.
    """
    if np.isnan(prob):
        return np.nan
    elif prob < thresh1:
        return "1"
    elif prob < thresh2:
        return "2"
    else:
        return "3"
