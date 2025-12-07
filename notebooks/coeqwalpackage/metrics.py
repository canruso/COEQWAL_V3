"""IMPORTS"""
import os
import sys
import importlib
import datetime as dt
import time
from pathlib import Path
from contextlib import redirect_stdout
import calendar
from typing import Sequence, Tuple, Dict, List

# Import data manipulation libraries
import numpy as np
import pandas as pd

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


def read_in_df(df_path, names_path):
    df = pd.read_csv(df_path, header=[0, 1, 2, 3, 4, 5, 6], index_col=0, parse_dates=True)
    dss_names = pd.read_csv(names_path)["0"].tolist()
    return df, dss_names


def load_metadata_df(extract_path, all_data, metadata_file, nrows=200):
    metadata_df = pd.read_excel(extract_path + metadata_file, engine='openpyxl', skiprows=7, usecols="B:K", nrows=nrows)
    metadata_df.columns = ['Pathnames', 'Part A', 'Part B', 'Part C', 'UNITS', 'Part F', 'Empty1', 'Col', 'Empty2',
                           'Description']

    metadata_df.drop(['Empty1', 'Empty2'], axis=1, inplace=True)
    df = pd.read_csv(extract_path + all_data, header=[0, 1, 2, 3, 4, 5, 6], index_col=0, parse_dates=True)
    return metadata_df, df


def convert_cfs_to_taf(df, metadata_df):
    units_mapping = (metadata_df.set_index("Part B")["UNITS"].dropna().to_dict())

    print("\nUnits Mapping:")
    for key, value in list(units_mapping.items()):
        print(f"{key}: {value}")

    date_column = df.index
    months = date_column.strftime('%m')
    years = date_column.strftime('%Y')

    days_in_month = np.zeros(len(df))
    for i in range(len(months)):
        if months[i] in {"01", "03", "05", "07", "08", "10", "12"}:
            days_in_month[i] = 31
        elif months[i] == "02":
            year = int(years[i])
            if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
                days_in_month[i] = 29
            else:
                days_in_month[i] = 28
        elif months[i] in {"04", "06", "09", "11"}:
            days_in_month[i] = 30

    columns_to_convert = []
    columns_to_skip = []

    for col in df.columns:
        part_a = col[0]  # e.g. "CALCULATED" or "CALSIM"
        part_b = col[1]  # e.g. "DEL_NOD_AG"
        data_unit = col[6]  # e.g. "CFS"

        matched_part_b = None
        for meta_part_b in units_mapping.keys():
            if meta_part_b in part_b:
                matched_part_b = meta_part_b
                break

        if matched_part_b:
            desired_unit = units_mapping.get(matched_part_b, data_unit)
            if data_unit == "CFS" and desired_unit == "TAF":
                columns_to_convert.append((col, "TAF"))
            else:
                columns_to_skip.append(col)

        if matched_part_b is None:
            if part_a == "CALCULATED" and data_unit == "CFS":
                if ("DEL" in part_b) or ("TOTAL_EXPORTS" in part_b):
                    columns_to_convert.append((col, "TAF"))
                else:
                    columns_to_skip.append(col)
            else:
                columns_to_skip.append(col)

    print("\nColumns to Convert:")
    for col, desired_unit in columns_to_convert:
        print(f"{col}: Data Unit = {col[6]}, Desired Unit = {desired_unit}")

    print("\nColumns to Skip:")
    for col in columns_to_skip:
        print(f"{col}: Data Unit = {col[6]}, "
              f"Desired Unit = {units_mapping.get(col[1], 'No Unit Information')}")

    for col, desired_unit in columns_to_convert:
        if col[6] == 'CFS' and desired_unit == 'TAF':
            print(f"\nConverting column: {col} from CFS to TAF")

            new_values = df[col].values * 0.001984 * days_in_month
            new_col = list(col)
            new_col[6] = 'TAF'
            new_col = tuple(new_col)
            df[new_col] = new_values

            print(f"Updated column units to 'TAF' for {new_col}")
        else:
            print(f"No defined conversion rule for {col[6]} to {desired_unit}. Skipping.")

    return df


def add_water_year_column(df):
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)

    water_year = np.where(idx.month >= 10, idx.year + 1, idx.year).astype('int32')

    if 'WaterYear' in out.columns:
        out = out.drop(columns=['WaterYear'])
    out.insert(0, 'WaterYear', water_year)

    return out


def create_subset_var(df, varname, water_year_type=None, month=None):
    filtered_columns = df.columns.get_level_values(1).str.contains(varname)

    if water_year_type is not None:
        if month is None:
            raise ValueError("If 'water_year_type' is provided, 'month' must also be provided.")

        wyt_filter = df.columns.get_level_values(1).str.contains('WYT_SAC_')
        wy_filter = df.columns.get_level_values(0).str.contains("WaterYear")
        combined_filter = filtered_columns | wyt_filter | wy_filter
        filtered_df = df.loc[:, combined_filter].copy()
        df_wyt_filtered = df.loc[:, filtered_df.columns.get_level_values(1).str.contains('WYT_SAC_') | (
                filtered_df.columns.get_level_values(0) == 'WaterYear')]
        month_values = df_wyt_filtered[df_wyt_filtered.index.month == month].groupby('WaterYear').first()
        df_wyt_filtered = df_wyt_filtered.merge(month_values, left_on='WaterYear', right_index=True, how='left',
                                                suffixes=('_df', ''))
        filtered_df.update(df_wyt_filtered)
        df_wyt = filtered_df.loc[:, filtered_df.columns.get_level_values(1).str.contains('WYT_SAC_')]
        filtered_df.loc[:, df_wyt.columns] = df_wyt.map(lambda x: x if x in water_year_type else np.nan)
        df_var = filtered_df.loc[:, filtered_df.columns.get_level_values(1).str.contains(varname)]
        df_copy = filtered_df.loc[:, filtered_df.columns.get_level_values(1).str.contains('WYT_SAC_')]
        for i in range(len(df_var.columns)):
            na = df_copy[df_copy.columns[i]].isna()
            df_var.loc[na, df_var.columns[i]] = np.nan
        return df_var
    return df.loc[:, filtered_columns]


def create_subset_unit(df, varname, units, water_year_type=None, month=None):
    var_filter = df.columns.get_level_values(1).str.contains(varname)
    unit_filter = df.columns.get_level_values(6).str.contains(units)
    filtered_columns = var_filter & unit_filter

    if water_year_type is not None:
        if month is None:
            raise ValueError("If 'water_year_type' is provided, 'month' must also be provided.")

        wyt_filter = df.columns.get_level_values(1).str.contains('WYT_SAC_')
        wy_filter = df.columns.get_level_values(0).str.contains("WaterYear")

        combined_filter = (var_filter & unit_filter) | wyt_filter | wy_filter
        filtered_columns = df.loc[:, combined_filter]

        df_wyt_filtered = filtered_columns.loc[:,
                          filtered_columns.columns.get_level_values(1).str.contains('WYT_SAC_') | (
                                  filtered_columns.columns.get_level_values(0) == 'WaterYear')]
        df_wyt_filtered = df_wyt_filtered.sort_index(axis=1)
        month_values = df_wyt_filtered[df_wyt_filtered.index.month == month].groupby('WaterYear').first()
        df_wyt_filtered = df_wyt_filtered.merge(month_values, left_on='WaterYear', right_index=True, how='left',
                                                suffixes=('_df', ''))
        filtered_columns.update(df_wyt_filtered)
        df_wyt = filtered_columns.loc[:, filtered_columns.columns.get_level_values(1).str.contains('WYT_SAC_')]
        filtered_columns.loc[:, df_wyt.columns] = df_wyt.map(lambda x: x if x in water_year_type else np.nan)
        df_var = filtered_columns.loc[:, filtered_columns.columns.get_level_values(1).str.contains(varname)]
        filtered_columns = filtered_columns.loc[:,
                           filtered_columns.columns.get_level_values(1).str.contains('WYT_SAC_')]
        for i in range(len(df_var.columns)):
            df_nan = filtered_columns[filtered_columns.columns[i]].isna()
            df_var.loc[df_nan, df_var.columns[i]] = np.nan
        return df_var
    return df.loc[:, filtered_columns]


def create_subset_list(df, var_names, water_year_type=None, month=None):
    filtered_columns = df.columns.get_level_values(1).str.contains('|'.join(var_names))
    if water_year_type is not None:
        if month is None:
            raise ValueError("If 'water_year_type' is provided, 'month' must also be provided.")

        wyt_filter = df.columns.get_level_values(1).str.contains('WYT_SAC_')
        wy_filter = df.columns.get_level_values(0).str.contains("WaterYear")
        combined_filter = filtered_columns | wyt_filter | wy_filter
        filtered_df = df.loc[:, combined_filter].copy()
        df_wyt_filtered = filtered_df.loc[:, filtered_df.columns.get_level_values(1).str.contains('WYT_SAC_') | (
                filtered_df.columns.get_level_values(0) == 'WaterYear')]
        month_values = df_wyt_filtered[df_wyt_filtered.index.month == month].groupby('WaterYear').first()
        df_wyt_filtered = df_wyt_filtered.merge(month_values, left_on='WaterYear', right_index=True, how='left',
                                                suffixes=('_df', ''))
        filtered_df.update(df_wyt_filtered)
        df_wyt = filtered_df.loc[:, filtered_df.columns.get_level_values(1).str.contains('WYT_SAC_')]
        filtered_df.loc[:, df_wyt.columns] = df_wyt.map(lambda x: x if x in water_year_type else np.nan)
        df_var = filtered_df.loc[:, filtered_df.columns.get_level_values(1).str.contains('|'.join(var_names))]
        df_copy = filtered_df.loc[:, filtered_df.columns.get_level_values(1).str.contains('WYT_SAC_')]
        for i in range(len(df_var.columns)):
            na = df_copy[df_copy.columns[i]].isna()
            df_var.loc[na, df_var.columns[i]] = np.nan
        return df_var
    return df.loc[:, filtered_columns]


def set_index(df, dss_names):
    scenario_names = []
    for i in range(len(dss_names)):
        scenario_names.append(dss_names[i][:5])
    df.index = scenario_names
    return df


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='first')]
    return df


def compute_annual_means(df, var, study_lst=None, units="TAF", months=None):
    subset_df = create_subset_unit(df, var, units)
    if study_lst is not None:
        subset_df = subset_df.iloc[:, study_lst]

    subset_df = add_water_year_column(subset_df)

    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]

    subset_df = _ensure_lexsorted_axes(subset_df)
    annual_mean = subset_df.groupby('WaterYear').mean()
    return annual_mean


def compute_mean(df, variable_list, study_lst, units="TAF", months=None):
    df = compute_annual_means(df, variable_list, study_lst, units, months)
    len_nonnull_yrs = df.dropna().shape[0]
    return (df.sum() / len_nonnull_yrs).iloc[-1]


def compute_sd(df, variable_list, varname, months=None, units="TAF"):
    subset_df = create_subset_unit(df, variable_list, units)
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]

    standard_deviation = subset_df.std().to_frame(name=varname).reset_index(drop=True)
    return standard_deviation


def compute_cv(df, variable, varname, months=None, units="TAF"):
    subset_df = create_subset_unit(df, variable, units)

    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]

    cv = (subset_df.std(axis=0) / subset_df.mean(axis=0)).to_frame(name=varname)
    cv.index = [col[1][-5:] if isinstance(col, tuple) else col[:5] for col in cv.index]

    return cv


def compute_iqr(df, variable, units, varname, upper_quantile=0.75, lower_quantile=0.25, months=None):
    subset_df = create_subset_unit(df, variable, units)
    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]
    iqr_values = subset_df.apply(lambda x: x.quantile(upper_quantile) - x.quantile(lower_quantile), axis=0)
    iqr_df = pd.DataFrame(iqr_values, columns=['IQR']).reset_index()[["IQR"]].rename(columns={"IQR": varname})
    return iqr_df


def compute_iqr_value(df, iqr_value, variable, units, varname, study_list, months=None, annual=True):
    if annual:
        subset_df = compute_annual_means(create_subset_unit(df, variable, units), variable, study_list, units, months)
    else:
        subset_df = create_subset_unit(df, variable, units)
        if months is not None:
            subset_df = subset_df[subset_df.index.month.isin(months)]
    iqr_values = subset_df.apply(lambda x: x.quantile(iqr_value), axis=0)
    iqr_df = pd.DataFrame(iqr_values, columns=['IQR']).reset_index()[["IQR"]].rename(columns={"IQR": varname})
    return iqr_df


def calculate_monthly_average(flow_data):
    flow_data = flow_data.reset_index()
    flow_data['Date'] = pd.to_datetime(flow_data.iloc[:, 0])
    flow_data.loc[:, 'Month'] = flow_data['Date'].dt.strftime('%m')
    flow_data.loc[:, 'Year'] = flow_data['Date'].dt.strftime('%Y')
    flow_values = flow_data.iloc[:, 1:]
    monthly_avg = flow_values.groupby(flow_data['Month']).mean().reset_index()
    monthly_avg.rename(columns={'Month': 'Month'}, inplace=True)
    return monthly_avg


def compute_annual_sums(df, var, study_lst=None, units="TAF", months=None):
    subset_df = create_subset_unit(df, var, units).iloc[:, study_lst]
    subset_df = add_water_year_column(subset_df)

    if months is not None:
        subset_df = subset_df[subset_df.index.month.isin(months)]

    subset_df = _ensure_lexsorted_axes(subset_df)
    annual_sum = subset_df.groupby('WaterYear').sum()
    return annual_sum


def compute_sum(df, variable_list, study_lst, units, months=None):
    df = compute_annual_sums(df, variable_list, study_lst, units, months)
    return (df.sum()).iloc[-1]


def count_exceedance_days(data, threshold):
    exceedance_counts = pd.DataFrame(np.nan, index=[0], columns=data.columns)

    for col in data.columns:
        exceedance_counts.loc[0, col] = (data[col] > threshold).sum()
    return exceedance_counts


def calculate_flow_sum_per_year(flow_data):
    """
    :NOTE: This was translated from Abhinav's code and is only used in the exceedance_metric function
    """
    flow_data = add_water_year_column(flow_data)
    flow_sum_per_year = flow_data.groupby('WaterYear').sum(numeric_only=True).reset_index()
    return flow_sum_per_year


def calculate_exceedance_probabilities(df):
    exceedance_df = pd.DataFrame(index=df.index)
    for column in df.columns:
        sorted_values = df[column].dropna().sort_values(ascending=False)
        exceedance_probs = sorted_values.rank(method='first', ascending=False) / (1 + len(sorted_values))
        exceedance_df[column] = exceedance_probs.reindex(df.index)

    new_columns = pd.MultiIndex.from_tuples([
        col if isinstance(col, tuple) else (col,)
        for col in exceedance_df.columns])
    exceedance_df.columns = new_columns
    return exceedance_df


def exceedance_probability(df, var, threshold, month, vartitle):
    var_df = create_subset_var(df, var)
    var_month_df = var_df[var_df.index.month.isin([month])].dropna()
    result_df = count_exceedance_days(var_month_df, threshold) / len(var_month_df) * 100
    reshaped_df = result_df.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]
    return reshaped_df


def exceedance_metric(df, var, exceedance_percent, vartitle, unit):
    var_df = create_subset_unit(df, var, unit)
    annual_flows = calculate_flow_sum_per_year(var_df).iloc[:, 1:].dropna()
    exceedance_probs = calculate_exceedance_probabilities(annual_flows)
    annual_flows_sorted = annual_flows.apply(np.sort, axis=0)[::-1]
    exceedance_prob_baseline = exceedance_probs.apply(np.sort, axis=0)
    if not exceedance_prob_baseline.empty:
        exceedance_prob_baseline = exceedance_prob_baseline.iloc[:, 0].to_frame()
        exceedance_prob_baseline.columns = ["Exceedance Sorted"]
    else:
        raise ValueError("No data available for exceedance probability calculation")
    if 'Exceedance Sorted' not in exceedance_prob_baseline.columns:
        raise KeyError("Column 'Exceedance Sorted' not found in DataFrame")
    filtered_indices = exceedance_prob_baseline.loc[
        exceedance_prob_baseline['Exceedance Sorted'] >= exceedance_percent].index
    if len(filtered_indices) == 0:
        raise ValueError("No values found meeting the exceedance criteria")
    exceeding_index = filtered_indices[0]
    baseline_threshold = annual_flows_sorted.iloc[len(annual_flows_sorted) - exceeding_index - 1, 0]
    result_df = count_exceedance_days(annual_flows, baseline_threshold).dropna() / len(annual_flows) * 100
    reshaped_df = result_df.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]
    return reshaped_df


def ann_avg(df, dss_names, var_name, units="TAF", months=None):
    metrics = []
    for study_index in np.arange(0, len(dss_names)):
        metric_value = compute_mean(df, var_name, [study_index], units, months=months)
        metrics.append(metric_value)

    ann_avg_delta_df = pd.DataFrame(metrics, columns=['Ann_Avg_' + var_name + units])
    ann_avg_delta_df = set_index(ann_avg_delta_df, dss_names)
    return ann_avg_delta_df


def ann_percentile(df, dss_names, pct, var_name, units="TAF"):
    study_list = np.arange(0, len(dss_names))
    df_title = 'Percentile_' + var_name + units
    iqr_df = compute_iqr_value(df, pct, var_name, units, df_title, study_list, months=None, annual=True)
    iqr_df = set_index(iqr_df, dss_names)
    return iqr_df


def mnth_avg(df, dss_names, var_name, mnth_num, units="TAF"):
    metrics = []
    for study_index in np.arange(0, len(dss_names)):
        metric_value = compute_mean(df, var_name, [study_index], units, months=[mnth_num])
        metrics.append(metric_value)

    mnth_str = calendar.month_abbr[mnth_num]
    mnth_avg_df = pd.DataFrame(metrics, columns=[mnth_str + '_Avg_' + var_name + units])
    mnth_avg_df = set_index(mnth_avg_df, dss_names)
    return mnth_avg_df


def moy_avgs(df, var_name, dss_names, units="TAF"):
    var_df = create_subset_var(df, varname=var_name)

    all_months_avg = {}
    for mnth_num in range(1, 13):
        metrics = []

        for study_index in np.arange(0, len(dss_names)):
            metric_val = compute_mean(var_df, var_name, [study_index], units, months=[mnth_num])
            metrics.append(metric_val)

        mnth_str = calendar.month_abbr[mnth_num]
        all_months_avg[mnth_str] = np.mean(metrics)

    moy_df = pd.DataFrame(list(all_months_avg.items()), columns=['Month', f'moy_Avg_{var_name}_{units}'])
    return moy_df


def mnth_percentile(df, dss_names, pct, var_name, mnth_num, units="TAF"):
    study_list = np.arange(0, len(dss_names))
    mnth_str = calendar.month_abbr[mnth_num]
    df_title = mnth_str + '_Percentile_' + var_name + units
    iqr_df = compute_iqr_value(df, pct, var_name, units, df_title, study_list, months=[mnth_num], annual=True)
    iqr_df = set_index(iqr_df, dss_names)
    return iqr_df


def annual_totals(df, var_name, units):
    df = create_subset_unit(df, var_name, units)
    annualized_df = pd.DataFrame()
    var = '_'.join(df.columns[0][1].split('_')[:-1])
    studies = [col[1].split('_')[-1] for col in df.columns]

    i = 0
    for study in studies:
        study_cols = [col for col in df.columns if col[1].endswith(study)]
        for col in study_cols:
            with redirect_stdout(open(os.devnull, 'w')):
                temp_df = df.loc[:, [df.columns[i]]]
                temp_df["Year"] = df.index.year
                df_ann = temp_df.groupby("Year").sum()
                annualized_df = pd.concat([annualized_df, df_ann], axis=1)
                i += 1

    return annualized_df


def frequency_hitting_var_const_level(df, dss_names, var_res, var_fldzn, units, vartitle, floodzone=True, months=None,
                                      threshold=None):

    def _get_subset_or_constant(var, ref_df):
        if isinstance(var, (int, float)):
            return pd.DataFrame(var, index=ref_df.index, columns=ref_df.columns)
        else:
            return create_subset_unit(df, var, units)

    if isinstance(var_res, (int, float)):
        ref_df = create_subset_unit(df, var_fldzn, units) if not isinstance(var_fldzn, (int, float)) else df
        subset_df_res = pd.DataFrame(var_res, index=ref_df.index, columns=ref_df.columns)
    else:
        subset_df_res = create_subset_unit(df, var_res, units)

    if isinstance(var_fldzn, (int, float)):
        subset_df_floodzone = pd.DataFrame(var_fldzn, index=subset_df_res.index, columns=subset_df_res.columns)
    else:
        subset_df_floodzone = create_subset_unit(df, var_fldzn, units)

    if months is not None:
        subset_df_res = subset_df_res[subset_df_res.index.month.isin(months)]
        subset_df_floodzone = subset_df_floodzone[subset_df_floodzone.index.month.isin(months)]

    multiindex_columns = subset_df_res.columns
    subset_df_res_comp_values = subset_df_res.values - subset_df_floodzone.values

    if floodzone:
        subset_df_res_comp_values += 0.000001

    subset_df_res_comp = pd.DataFrame(subset_df_res_comp_values, index=subset_df_res.index, columns=multiindex_columns)

    if threshold is not None:
        days_within_threshold = (abs(subset_df_res_comp_values) <= threshold).sum().sum()

    exceedance_days = count_exceedance_days(subset_df_res_comp, 0)
    exceedance_days_fraction = exceedance_days / len(subset_df_res_comp)

    if not floodzone:
        exceedance_days = 100 - exceedance_days

    exceedance_days = exceedance_days.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]
    exceedance_days = set_index(exceedance_days, dss_names)

    exceedance_days_fraction = exceedance_days_fraction.melt(value_name=vartitle).reset_index(drop=True)[[vartitle]]
    exceedance_days_fraction = set_index(exceedance_days_fraction, dss_names)

    if threshold is not None:
        return exceedance_days, exceedance_days_fraction, days_within_threshold
    else:
        return exceedance_days, exceedance_days_fraction


def compute_storage_thresholds(df: pd.DataFrame, dss_names: Sequence[str], variables_storage: Sequence[str],
                               variables_deadpool: Sequence[str], variables_floodpool: Sequence[str], *,
                               mlrtn_level5constant: float | None = None, mlrtn_level1constant: float | None = None,
                               smelon_level1constant: float | None = None) -> Tuple[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], List[pd.DataFrame]]:

    thresholds: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    threshold_fractions: List[pd.DataFrame] = []

    print("Variables:")
    print(list(variables_storage))

    for var, var_lvl1, var_lvl5 in zip(variables_storage, variables_deadpool, variables_floodpool):
        print("Current variable: " + var)
        if var == "S_MLRTN_":
            if mlrtn_level5constant is None or mlrtn_level1constant is None:
                raise ValueError("MLRTN constants must be provided for S_MLRTN_.")
            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, var, mlrtn_level5constant, "TAF", f"All_Prob_{var}flood")
            thresholds[f"{var}frequency_hitting_level5"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, var, mlrtn_level5constant, "TAF", f"Sept_Prob_{var}flood", months=[9])
            thresholds[f"Sept_{var}frequency_hitting_level5"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, mlrtn_level1constant, var, "TAF", f"All_Prob_{var}dead")
            thresholds[f"{var}frequency_hitting_level1"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, mlrtn_level1constant, var, "TAF", f"Sept_Prob_{var}dead", months=[9])
            thresholds[f"Sept_{var}frequency_hitting_level1"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

        elif var == "S_MELON_":
            if smelon_level1constant is None:
                raise ValueError("S_MELON_ level1 constant must be provided.")
            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, var, var_lvl5, "TAF", f"All_Prob_{var}flood")
            thresholds[f"{var}frequency_hitting_level5"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, var, var_lvl5, "TAF", f"Sept_Prob_{var}flood", months=[9])
            thresholds[f"Sept_{var}frequency_hitting_level5"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, smelon_level1constant, var, "TAF", f"All_Prob_{var}dead")
            thresholds[f"{var}frequency_hitting_level1"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, smelon_level1constant, var, "TAF", f"Sept_Prob_{var}dead", months=[9])
            thresholds[f"Sept_{var}frequency_hitting_level1"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

        else:
            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, var, var_lvl5, "TAF", f"All_Prob_{var}flood")
            thresholds[f"{var}frequency_hitting_level5"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, var_lvl1, var, "TAF", f"All_Prob_{var}dead")
            thresholds[f"{var}frequency_hitting_level1"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, var, var_lvl5, "TAF", f"Sept_Prob_{var}flood", months=[9])
            thresholds[f"Sept_{var}frequency_hitting_level5"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

            exceedance_days, exceedance_days_fraction = frequency_hitting_var_const_level(
                df, dss_names, var_lvl1, var, "TAF", f"Sept_Prob_{var}dead", months=[9])
            thresholds[f"Sept_{var}frequency_hitting_level1"] = (exceedance_days, exceedance_days_fraction)
            threshold_fractions.append(exceedance_days_fraction)

    return thresholds, threshold_fractions


def compute_metrics_suite(df: pd.DataFrame, dss_names: Sequence[str],
                          variables: Sequence[str]) -> Tuple[Dict[str, pd.DataFrame], List[pd.DataFrame]]:

    metrics: Dict[str, pd.DataFrame] = {}
    study_list = np.arange(0, len(dss_names))

    for var in variables:
        if var in ["C_SAC041_", "C_SJR070_", "C_SAC000_", "C_SJR070_", "C_DMC000_TD_", "C_CAA003_TD_", "NDO_",
                   "D_TOTAL_"]:
            units = "CFS"
        elif var == "X2_PRV_KM_":
            units = "KM"
        elif var in ["EM_EC_MONTH_", "JP_EC_MONTH"]:
            units = "UMHOS/CM"
        else:
            units = "TAF"

        if var in ["S_RESTOT_NOD_", "S_RESTOT_s"]:
            metrics[f"Apr_{var}mnth_avg"] = mnth_avg(df, dss_names, var, 4, units)
            metrics[f"Sept_{var}mnth_avg"] = mnth_avg(df, dss_names, var, 9, units)
            metrics[f"{var}ann_avg"] = ann_avg(df, dss_names, var, units)

        if var in ["S_SHSTA_", "S_OROVL_", "S_TRNTY_", "S_FOLSM_", "S_MELON_", "S_MLRTN_", "S_SLUIS_SWP",
                   "S_SLUIS_CVP"]:
            metrics[f"Apr_{var}_mnth_avg"] = mnth_avg(df, dss_names, var, 4, units)
            metrics[f"Sept_{var}_mnth_avg"] = mnth_avg(df, dss_names, var, 9, units)
            metrics[f"Apr{var}_CV"] = compute_cv(df, var, f"Apr{var}_CV", [4], units)
            metrics[f"Sept_{var}_CV"] = compute_cv(df, var, f"Sept{var}_CV", [9], units)

        if var in ["DEL_SWP_TOTAL_", "DEL_SWP_PMI_", "DEL_SWP_PAG_", "DEL_CVP_TOTAL_", "DEL_CVP_PAG_TOTAL_",
                   "DEL_CVP_PSCEX_TOTAL_", "DEL_CVP_PRF_TOTAL_", "D_TOTAL_", "NDO_"]:
            metrics[f"{var}_ann_avg"] = ann_avg(df, dss_names, var, units)

        if var == "X2_PRV_KM_":
            metrics[f"Fall_{var}_ann_avg"] = ann_avg(df, dss_names, var, units, months=[9, 10, 11]).rename(
                columns={f"Ann_Avg_{var}{units}": f"Fall_Ann_Avg_{var}{units}"})
            metrics[f"Spring_{var}_ann_avg"] = ann_avg(df, dss_names, var, units, months=[3, 4, 5]).rename(
                columns={f"Ann_Avg_{var}{units}": f"Spring_Ann_Avg_{var}{units}"})
            metrics[f"Fall_{var}_CV"] = compute_cv(df, var, f"Fall_{var}_CV", [9, 10, 11], units)
            metrics[f"Spring_{var}_CV"] = compute_cv(df, var, f"Spring_{var}_CV", [3, 4, 5], units)

        if var in ["EM_EC_MONTH_", "JP_EC_MONTH"]:
            metrics[f"Fall_{var}_ann_avg"] = ann_avg(df, dss_names, var, units, months=[9, 10, 11]).rename(
                columns={f"Ann_Avg_{var}{units}": f"Fall_Ann_Avg_{var}{units}"})
            metrics[f"Spring_{var}_ann_avg"] = ann_avg(df, dss_names, var, units, months=[3, 4, 5]).rename(
                columns={f"Ann_Avg_{var}{units}": f"Spring_Ann_Avg_{var}{units}"})
            metrics[f"Fall_{var}_CV"] = compute_cv(df, var, f"Fall_{var}_CV", [9, 10, 11], units)
            metrics[f"Spring_{var}_CV"] = compute_cv(df, var, f"Spring_{var}_CV", [3, 4, 5], units)

    metric_frames = list(metrics.values())
    return metrics, metric_frames


def probability_var1_lt_var2_for_scenario(df, var1_name, var2_name, units="CFS", tolerance=1e-6):

    df_var1 = create_subset_unit(df, var1_name, units)
    df_var2 = create_subset_unit(df, var2_name, units)

    if df_var1.empty or df_var2.empty:
        return np.nan

    series_var1 = df_var1.iloc[:, 0].reindex(df_var2.index).dropna()
    series_var2 = df_var2.iloc[:, 0].reindex(df_var1.index).dropna()
    common_idx = series_var1.index.intersection(series_var2.index)
    if len(common_idx) == 0:
        return np.nan

    series_var1 = series_var1.loc[common_idx]
    series_var2 = series_var2.loc[common_idx]

    count_less = (series_var1 < series_var2).sum()
    prob_less = count_less / len(series_var1)
    return prob_less


def probability_var1_eq_var2_for_scenario(df, var1_name, var2_name, units="CFS", tolerance=1e-6):

    df_var1 = create_subset_unit(df, var1_name, units)
    df_var2 = create_subset_unit(df, var2_name, units)

    if df_var1.empty or df_var2.empty:
        return np.nan

    series_var1 = df_var1.iloc[:, 0].reindex(df_var2.index).dropna()
    series_var2 = df_var2.iloc[:, 0].reindex(df_var1.index).dropna()
    common_idx = series_var1.index.intersection(series_var2.index)
    if len(common_idx) == 0:
        return np.nan

    series_var1 = series_var1.loc[common_idx]
    series_var2 = series_var2.loc[common_idx]

    count_equal = (np.abs(series_var1 - series_var2) < tolerance).sum()
    prob_equal = count_equal / len(series_var1)
    return prob_equal


def probability_var1_gte_var2_for_scenario(df, var1_name, var2_name, units="CFS", tolerance=1e-6):

    df_var1 = create_subset_unit(df, var1_name, units)
    df_var2 = create_subset_unit(df, var2_name, units)

    if df_var1.empty or df_var2.empty:
        return np.nan

    series_var1 = df_var1.iloc[:, 0].reindex(df_var2.index).dropna()
    series_var2 = df_var2.iloc[:, 0].reindex(df_var1.index).dropna()
    common_idx = series_var1.index.intersection(series_var2.index)
    if len(common_idx) == 0:
        return np.nan

    series_var1 = series_var1.loc[common_idx]
    series_var2 = series_var2.loc[common_idx]
    count_gte = (series_var1 >= series_var2).sum()
    prob_less = count_gte / len(series_var1)
    return prob_less


def probability_var1_gte_const_for_scenario(df, var1_name, const, units="CFS"):

    df_var1 = create_subset_unit(df, var1_name, units)
    if df_var1.empty:
        return np.nan

    series_var1 = df_var1.iloc[:, 0].dropna()

    if len(series_var1) == 0:
        return np.nan

    count_gte = (series_var1 >= const).sum()
    prob_less = count_gte / len(series_var1)
    return prob_less


def create_subset_tucp(df: pd.DataFrame, scenario: int, tucp_var: str, tucp_wy_month_count: int = 1) -> pd.DataFrame:
    suffix = f"s{int(scenario):04d}"
    lvl1 = df.columns.get_level_values(1).astype(str)

    name_mask = lvl1.str.contains(tucp_var, regex=False)
    scen_mask = lvl1.str.endswith(suffix)
    mask = name_mask & scen_mask

    if mask.sum() == 0:
        mask = name_mask

    trigger_df = df.loc[:, mask]
    trigger_series = pd.to_numeric(trigger_df.iloc[:, 0], errors="coerce")
    triggered = trigger_series >= 1
    wy_index = trigger_series.index.year + (trigger_series.index.month >= 10)
    months_per_wy = triggered.groupby(wy_index).sum(min_count=1)
    selected_wys = months_per_wy[months_per_wy >= int(tucp_wy_month_count)].index

    if len(selected_wys) == 0:
        return df.iloc[0:0].copy()

    all_wy = df.index.year + (df.index.month >= 10)
    row_mask = np.isin(all_wy, selected_wys)
    return df.loc[row_mask].copy()


def _ensure_lexsorted_axes(df: pd.DataFrame) -> pd.DataFrame:

    if isinstance(df.index, pd.MultiIndex):
        df = df.sort_index()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.sort_index(axis=1)
    return df


def percent_change_from_baseline(df: pd.DataFrame, baseline_label: str) -> pd.DataFrame:

    base = df.loc[baseline_label]

    prob_cols = [c for c in df.columns if str(c).startswith(('All_Prob_', 'Sept_Prob_'))]
    other_cols = [c for c in df.columns if c not in prob_cols]

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    if prob_cols:
        out[prob_cols] = df[prob_cols].sub(base[prob_cols], axis=1)

    if other_cols:
        denom = base[other_cols].replace(0, np.nan)
        out[other_cols] = df[other_cols].sub(base[other_cols], axis=1).div(denom, axis=1) * 100.0

    return out.replace([np.inf, -np.inf], np.nan)


def percent_change_from_baseline_by_index(df: pd.DataFrame, baseline_index: int = 0) -> pd.DataFrame:

    baseline = df.iloc[baseline_index]
    return ((df - baseline) / baseline.replace(0, np.nan)) * 100


def process_scenario_dataframe(df: pd.DataFrame, column_mapping: dict = None, desired_order: list = None,
                               final_names: list = None) -> pd.DataFrame:

    if column_mapping is None:
        column_mapping = {
            'NDO': 'NDO', 'X2_APR': 'X2_APR', 'X2_OCT': 'X2_OCT',
            'SAC_IN': 'SAC_IN', 'SJR_IN': 'SJR_IN', 'ES_YBP_IN': 'ES_YBP_IN',
            'TOTAL_DELTA_IN': 'TOTAL_DELTA_IN', 'CVP_SWP_EXPORTS': 'CVP_SWP_EXPORTS',
            'OTHER_EXPORTS': 'OTHER_EXPORTS', 'ADJ_CVP_SWP_EXPORTS': 'ADJ_CVP_SWP_EXPORTS',
            'DEL_NOD_TOTAL': 'DEL_NOD_TOTAL', 'DEL_NOD_AG_TOTAL': 'DEL_NOD_AG_TOTAL',
            'DEL_NOD_MI_TOTAL': 'DEL_NOD_MI_TOTAL', 'DEL_SJV_AG_TOTAL': 'DEL_SJV_AG_TOTAL',
            'DEL_SJV_MI_TOTAL': 'DEL_SJV_MI_TOTAL', 'DEL_SJV_TOTAL': 'DEL_SJV_TOTAL',
            'DEL_SOCAL_MI_TOTAL': 'DEL_SOCAL_MI_TOTAL', 'DEL_CCOAST_MI_TOTAL': 'DEL_CCOAST_MI_TOTAL',
            'STO_NOD_TOTAL_APR': 'STO_NOD_TOTAL_APR', 'STO_NOD_TOTAL_OCT': 'STO_NOD_TOTAL_OCT',
            'STO_SOD_TOTAL_APR': 'STO_SOD_TOTAL_APR', 'STO_SOD_TOTAL_OCT': 'STO_SOD_TOTAL_OCT'}

    if desired_order is None:
        desired_order = ['DEL_NOD_AG_TOTAL', 'DEL_SJV_AG_TOTAL', 'DEL_NOD_MI_TOTAL', 'DEL_SJV_MI_TOTAL',
            'DEL_SOCAL_MI_TOTAL', 'CVP_SWP_EXPORTS', 'NDO', 'SAC_IN', 'SJR_IN', 'X2_APR',
            'X2_OCT', 'STO_NOD_TOTAL_OCT', 'STO_SOD_TOTAL_OCT']

    if final_names is None:
        final_names = ["Sac Valley AG Deliveries", "SJ Valley AG Deliveries", "Sac Valley Municipal Deliveries",
            "SJ Valley Municipal Deliveries", "SoCal Municipal Deliveries", "Delta Exports",
            "Delta Outflows", "Sac River Inflows", "SJ River Inflows", "X2 Salinity (Apr)",
            "X2 Salinity (Oct)", "North of Delta Storage (Sep)", "South of Delta Storage (Sep)"]

    selected_columns = [col for col in df.columns if col[1] in column_mapping]
    selected_df = df[selected_columns]
    new_columns = [column_mapping[col[1]] for col in selected_df.columns]
    selected_df.columns = new_columns
    ordered_df = selected_df[desired_order]
    ordered_df.columns = final_names
    return ordered_df


def calculate_scenario_statistics(dataframes: list, scenario_names: list, process_func=None) -> tuple:

    if process_func is None:
        process_func = process_scenario_dataframe

    medians, std_devs, percentiles_90, percentiles_10 = [], [], [], []

    for df, name in zip(dataframes, scenario_names):
        processed_df = process_func(df)

        median_values = processed_df.median()
        std_dev_values = processed_df.std()
        percentile_90_values = processed_df.quantile(0.90)
        percentile_10_values = processed_df.quantile(0.10)
        median_values.name = name
        std_dev_values.name = name
        percentile_90_values.name = name
        percentile_10_values.name = name
        medians.append(median_values)
        std_devs.append(std_dev_values)
        percentiles_90.append(percentile_90_values)
        percentiles_10.append(percentile_10_values)

    return pd.DataFrame(medians), pd.DataFrame(std_devs), pd.DataFrame(percentiles_90), pd.DataFrame(percentiles_10)

def compute_cv_df(df: pd.DataFrame) -> pd.DataFrame:

    records = []

    for col in df.columns:
        if "_s" not in col:
            continue
        wba_id, scen_raw = col.split("_s", 1)
        scenario = f"s{scen_raw}"

        series = df[col].dropna()
        if len(series) == 0:
            continue

        mean_val = series.mean()
        std_val = series.std()
        cv = std_val / mean_val if mean_val != 0 else np.nan

        records.append({"scenario": scenario, "WBA": wba_id, "CV": cv})

    cv_df = (pd.DataFrame(records).pivot(index="scenario", columns="WBA", values="CV").sort_index(axis=1).
             sort_index(axis = 0))

    return cv_df
