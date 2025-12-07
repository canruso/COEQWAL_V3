from __future__ import annotations
# import metrics library
from metrics import *

# Import visualization libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm
from matplotlib.cm import get_cmap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import seaborn as sns

import os
import numpy as np
import pandas as pd
import cqwlutils as cu


def enable_headless_mode() -> None:
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass

    plt.ioff()

    def _silent_show(*args, **kwargs):
        plt.close("all")

    plt.show = _silent_show


def plot_ts(df, varname, units='TAF', pTitle='Time Series', xLab='Date', lTitle='Studies', fTitle='mon_tot', pSave=True,
            fPath='fPath', study_list=None, start_date=None, end_date=None, scenario_styles=None, dpi=300):
    df_plot = create_subset_unit(df, varname, units)

    if start_date is not None:
        df_plot = df_plot.loc[df_plot.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot.loc[df_plot.index <= pd.to_datetime(end_date)]

    if study_list is not None:
        suffixes = [f"s{str(st).zfill(4)}" for st in study_list]
        new_cols = []
        for col in df_plot.columns:
            if any(col[1].endswith(sfx) for sfx in suffixes):
                new_cols.append(col)
        df_plot = df_plot[new_cols]

    if df_plot.empty:
        print("[plot_ts] WARNING: No data to plot after subsetting!")
        return

    var = '_'.join(df_plot.columns[0][1].split('_')[:-1])
    colormap = plt.cm.tab20
    colors = [colormap(i) for i in range(df_plot.shape[1])]
    if len(colors) > 0:
        colors[-1] = (0, 0, 0, 1)

    plt.figure(figsize=(14, 8))
    default_font_size = plt.rcParams['font.size']
    scaled_font_size = 1.5 * default_font_size
    scaled_line_width = 1.5 * plt.rcParams['lines.linewidth']

    studies = [col[1].split('_')[-1] for col in df_plot.columns]
    scenario_labeled = set()
    count = 0

    for col, study in zip(df_plot.columns, studies):
        numeric_study = int(study.replace('s', ''))
        if scenario_styles and numeric_study in scenario_styles:
            style_dict = scenario_styles[numeric_study]
            this_color = style_dict.get('color', colors[count])
            this_linestyle = style_dict.get('linestyle', '-')
            this_label = style_dict.get('label', study) if study not in scenario_labeled else None
            this_lw = style_dict.get('linewidth', scaled_line_width)
        else:
            this_color = colors[count]
            this_linestyle = '-'
            this_label = study if study not in scenario_labeled else None
            this_lw = scaled_line_width

        sns.lineplot(data=df_plot, x=df_plot.index, y=col, label=this_label, color=this_color, linewidth=this_lw,
                     linestyle=this_linestyle)

        if this_label is not None:
            scenario_labeled.add(study)
        count += 1

    plt.title(var + ' ' + pTitle, fontsize=scaled_font_size * 2)
    plt.xlabel(xLab, fontsize=scaled_font_size * 1.5)
    first_col_units = df_plot.columns[0][6]
    plt.ylabel(var + "\nUnits: " + str(first_col_units), fontsize=scaled_font_size * 1.5)

    plt.legend(title=lTitle, title_fontsize=scaled_font_size * 1.5, fontsize=scaled_font_size * 1.25,
               bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.xticks(rotation=45, fontsize=scaled_font_size)
    plt.yticks(fontsize=scaled_font_size)
    plt.tight_layout()

    if pSave:
        plt.savefig(f'{fPath}/{var}_{fTitle}.png', format='png', bbox_inches='tight', dpi=dpi, transparent=False)

    plt.show()


def plot_annual_totals(df, varname, units='TAF', xLab='Date', pTitle='Annual Totals', lTitle='Studies',
                       fTitle='ann_tot', pSave=True, fPath='fPath', study_list=None, start_date=None, end_date=None,
                       scenario_styles=None, months=None, dpi=300):
    df_plot = create_subset_unit(df, varname, units)

    if start_date is not None:
        df_plot = df_plot.loc[df_plot.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_plot = df_plot.loc[df_plot.index <= pd.to_datetime(end_date)]

    if months is not None:
        df_plot = df_plot[df_plot.index.month.isin(months)]

    if study_list is not None:
        suffixes = [f"s{str(st).zfill(4)}" for st in study_list]
        new_cols = []
        for col in df_plot.columns:
            if any(col[1].endswith(sfx) for sfx in suffixes):
                new_cols.append(col)
        df_plot = df_plot[new_cols]

    if df_plot.empty:
        print("[plot_annual_totals] WARNING: No data after subsetting!")
        return pd.DataFrame()

    var = '_'.join(df_plot.columns[0][1].split('_')[:-1])
    colormap = plt.cm.tab20
    colors = [colormap(i) for i in range(df_plot.shape[1])]
    if len(colors) > 0:
        colors[-1] = (0, 0, 0, 1)

    plt.figure(figsize=(14, 8))
    default_font_size = plt.rcParams['font.size']
    scaled_font_size = 1.5 * default_font_size
    scaled_line_width = 1.5 * plt.rcParams['lines.linewidth']

    annualized_df = pd.DataFrame()

    studies = [col[1].split('_')[-1] for col in df_plot.columns]
    scenario_labeled = set()
    count = 0

    for col, study in zip(df_plot.columns, studies):
        numeric_study = int(study.replace('s', ''))
        if scenario_styles and numeric_study in scenario_styles:
            style_dict = scenario_styles[numeric_study]
            this_color = style_dict.get('color', colors[count])
            this_linestyle = style_dict.get('linestyle', '-')
            this_label = style_dict.get('label', study) if (study not in scenario_labeled) else None
            this_lw = style_dict.get('linewidth', scaled_line_width)
        else:
            this_color = colors[count]
            this_linestyle = '-'
            this_label = study if study not in scenario_labeled else None
            this_lw = scaled_line_width

        from contextlib import redirect_stdout
        with redirect_stdout(open(os.devnull, 'w')):
            single_col_df = df_plot[[col]]
            df_ann = annualize_ts(single_col_df, freq='YS-OCT')

        annualized_df = pd.concat([annualized_df, df_ann], axis=1)

        sns.lineplot(data=df_ann, x=df_ann.index, y=df_ann.columns[0], label=this_label, color=this_color,
                     linewidth=this_lw, linestyle=this_linestyle, drawstyle='steps-post')

        if this_label is not None:
            scenario_labeled.add(study)
        count += 1

    plt.title(f"{var} {pTitle}", fontsize=scaled_font_size * 2)
    plt.xlabel(xLab, fontsize=scaled_font_size * 1.5)
    first_col_units = df_plot.columns[0][6]
    plt.ylabel(var + "\nUnits: " + str(first_col_units), fontsize=scaled_font_size * 1.5)

    plt.legend(title=lTitle, title_fontsize=scaled_font_size * 1.5, fontsize=scaled_font_size * 1.25,
               bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.xticks(rotation=45, fontsize=scaled_font_size)
    plt.yticks(fontsize=scaled_font_size)
    plt.tight_layout()

    if pSave:
        plt.savefig(f'{fPath}/{var}_{fTitle}.png', format='png', bbox_inches='tight', dpi=dpi, transparent=False)

    plt.show()
    return annualized_df


def plot_exceedance(df, varname, units='TAF', xLab='Probability', pTitle='Exceedance Probability', lTitle='Studies',
                    fTitle='exceed', pSave=True, fPath='fPath', study_list=None, scenario_styles=None, months=None,
                    dpi=300):
    df_plot = create_subset_unit(df, varname, units)

    if months is not None:
        df_plot = df_plot[df_plot.index.month.isin(months)]

    if study_list is not None:
        suffixes = [f"s{str(st).zfill(4)}" for st in study_list]
        new_cols = []
        for col in df_plot.columns:
            if any(col[1].endswith(sfx) for sfx in suffixes):
                new_cols.append(col)
        df_plot = df_plot[new_cols]

    if df_plot.empty:
        print("[plot_exceedance] WARNING: No data to plot after subsetting!")
        return

    var = '_'.join(df_plot.columns[0][1].split('_')[:-1])
    studies = [col[1].split('_')[-1] for col in df_plot.columns]

    colormap = plt.cm.tab20
    colors = [colormap(i) for i in range(df_plot.shape[1])]
    if len(colors) > 0:
        colors[-1] = (0, 0, 0, 1)

    plt.figure(figsize=(14, 8))
    default_font_size = plt.rcParams['font.size']
    scaled_font_size = 1.5 * default_font_size
    scaled_line_width = 1.5 * plt.rcParams['lines.linewidth']

    scenario_labeled = set()

    for i, (col, study) in enumerate(zip(df_plot.columns, studies)):
        numeric_study = int(study.replace('s', ''))
        if scenario_styles and numeric_study in scenario_styles:
            style_dict = scenario_styles[numeric_study]
            this_color = style_dict.get('color', colors[i])
            this_linestyle = style_dict.get('linestyle', '-')
            this_label = style_dict.get('label', study) if (study not in scenario_labeled) else None
            this_lw = style_dict.get('linewidth', scaled_line_width)
        else:
            this_color = colors[i]
            this_linestyle = '-'
            this_label = study if study not in scenario_labeled else None
            this_lw = scaled_line_width

        df_ex = single_exceed_alternative(df_plot[col])
        ex_col_name = df_ex.columns[0]

        sns.lineplot(data=df_ex, x=df_ex.index, y=ex_col_name, label=this_label, color=this_color, linewidth=this_lw,
                     linestyle=this_linestyle)

        if this_label:
            scenario_labeled.add(study)

    plt.title(f"{var} {pTitle}", fontsize=scaled_font_size * 2)
    plt.xlabel(xLab, fontsize=scaled_font_size * 1.5)
    first_col_units = df_plot.columns[0][6]
    plt.ylabel(f"{var}\nUnits: {first_col_units}", fontsize=scaled_font_size * 1.5)

    plt.legend(title=lTitle, title_fontsize=scaled_font_size * 1.5, fontsize=scaled_font_size * 1.25,
               bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=45, fontsize=scaled_font_size)
    plt.yticks(fontsize=scaled_font_size)
    plt.tight_layout()

    if pSave:
        plt.savefig(f'{fPath}/{var}_{fTitle}.png', format='png', bbox_inches='tight', dpi=dpi, transparent=False)

    plt.show()


def single_exceed_alternative(series):
    s_clean = series.dropna()
    s_sorted = s_clean.sort_values(ascending=False)
    n = len(s_sorted)
    ranks = np.arange(1, n + 1)
    exceed_probs = ranks / (n + 1.0)
    out_df = pd.DataFrame(data=s_sorted.values, index=exceed_probs, columns=[series.name])
    return out_df


def annualize_exceedance_plot(df, varname, units="TAF", freq="YS-OCT", pTitle='Annual Exceedance', xLab='Probability',
                              lTitle='Studies', fTitle='annual_exceed', pSave=True, fPath='fPath', study_list=None,
                              scenario_styles=None, months=None, dpi=300):
    df_subset = create_subset_unit(df, varname, units)

    if months is not None:
        df_subset = df_subset[df_subset.index.month.isin(months)]

    if study_list is not None:
        suffixes = [f"s{str(st).zfill(4)}" for st in study_list]
        keep_cols = []
        for col in df_subset.columns:
            if any(col[1].endswith(sfx) for sfx in suffixes):
                keep_cols.append(col)
        df_subset = df_subset[keep_cols]

    if df_subset.empty:
        print("[annualize_exceedance_plot] WARNING: No data after subsetting!")
        return pd.DataFrame()

    annual_cols = []
    for col in df_subset.columns:
        one_col = df_subset[[col]]
        ann_col = one_col.resample(freq).sum(min_count=1)
        annual_cols.append(ann_col)

    annual_df = pd.concat(annual_cols, axis=1)

    plot_exceedance(annual_df, varname=varname, units=units, xLab=xLab, pTitle=pTitle, lTitle=lTitle, fTitle=fTitle,
                    pSave=pSave, fPath=fPath, study_list=None, scenario_styles=scenario_styles, months=None, dpi=dpi)

    return annual_df


def annualize_ts(df_col, freq='YS-OCT'):
    if df_col.shape[1] != 1:
        raise ValueError("annualize_ts expects exactly ONE column in df_col")
    df_annual = df_col.resample(freq).sum(min_count=1)
    return df_annual


def plot_moy_averages(df, varname, units='TAF', xLab='Month of Year', pTitle='Month of Year Average Totals',
                      lTitle='Studies', fTitle='moy_avg', pSave=True, fPath='fPath', study_list=None,
                      scenario_styles=None, dpi=300):
    df_plot = create_subset_unit(df, varname, units)

    if study_list is not None:
        suffixes = [f"s{str(st).zfill(4)}" for st in study_list]
        new_cols = []
        for col in df_plot.columns:
            if any(col[1].endswith(sfx) for sfx in suffixes):
                new_cols.append(col)
        df_plot = df_plot[new_cols]

    if df_plot.empty:
        print("[plot_moy_averages] WARNING: No data after subsetting!")
        return

    df_copy = df_plot.copy()
    df_copy["Month"] = df_copy.index.month
    df_moy = df_copy.groupby("Month").mean(numeric_only=True)

    plot_ts(df_moy, varname=varname, units=units, pTitle=pTitle, xLab=xLab, lTitle=lTitle, fTitle=fTitle, pSave=pSave,
            fPath=fPath, study_list=None, start_date=None, end_date=None, scenario_styles=scenario_styles, dpi=dpi)


def get_difference_from_baseline(df):
    df_diff = df.copy()
    baseline_column = df_diff.iloc[:, 0]

    for i in range(1, df_diff.shape[1]):
        df_diff.iloc[:, i] = df_diff.iloc[:, i].sub(baseline_column)
    df_diff = df_diff.iloc[:, 1:]

    return df_diff


def difference_from_baseline(df, plot_type, pTitle='Difference from Baseline ', xLab='Date', lTitle='Studies',
                             fTitle="___", pSave=True, fPath='fPath'):
    pTitle += plot_type.__name__
    diff_df = get_difference_from_baseline(df)
    plot_type(diff_df, pTitle=pTitle, fTitle=fTitle, fPath='fPath')


def slice_with_baseline(df, var, study_lst):
    subset_df = create_subset_var(df, var)
    df_baseline = subset_df.iloc[:, [0]]
    df_rest = subset_df.iloc[:, study_lst]
    return pd.concat([df_baseline, df_rest], axis=1)


"""PARALLEL PLOTS"""
#source: https://reedgroup.github.io/FigureLibrary/ParallelCoordinatesPlots.html

figsize = (18, 6)
fontsize = 14
main_data_dir = "../output/metrics/"
data_dir_knobs = "../data/parallelplots/"
fig_dir = '../output/parallelplots/'

"""Functions for flexible parallel coordinates plots"""


def reorganize_objs(objs, columns_axes, ideal_direction, minmaxs):
    if minmaxs is None:
        minmaxs = ['max'] * len(columns_axes)
    assert len(minmaxs) == len(columns_axes)

    objs_reorg = objs[columns_axes].copy()

    tops_vals = np.empty(len(columns_axes), dtype=float)
    bottoms_vals = np.empty(len(columns_axes), dtype=float)

    for i, col in enumerate(columns_axes):
        s = objs_reorg.iloc[:, i]
        cmin = float(s.min())
        cmax = float(s.max())

        if ideal_direction == 'top':
            top_val, bot_val = cmax, cmin
        else:
            top_val, bot_val = cmin, cmax

        if minmaxs[i] == 'min':
            top_val, bot_val = bot_val, top_val

        tops_vals[i] = top_val
        bottoms_vals[i] = bot_val
        denom = top_val - bot_val

        if np.isclose(denom, 0.0):
            objs_reorg.iloc[:, i] = np.nan
        else:
            if denom > 0:
                objs_reorg.iloc[:, i] = (s - bot_val) / denom
            else:
                objs_reorg.iloc[:, i] = (bot_val - s) / (-denom)

    tops = pd.Series(tops_vals, index=columns_axes, dtype=float)
    bottoms = pd.Series(bottoms_vals, index=columns_axes, dtype=float)
    return objs_reorg, tops, bottoms


def get_color(value, color_by_continuous, color_palette_continuous, color_by_categorical, color_dict_categorical):
    if color_by_continuous is not None:
        color = plt.get_cmap(color_palette_continuous)(value)
    elif color_by_categorical is not None:
        color = color_dict_categorical[value]
    return color


def get_zorder(norm_value, zorder_num_classes, zorder_direction):
    xgrid = np.arange(0, 1.001, 1 / zorder_num_classes)
    if zorder_direction == 'ascending':
        return 4 + np.sum(norm_value > xgrid)
    elif zorder_direction == 'descending':
        return 4 + np.sum(norm_value < xgrid)


def custom_parallel_coordinates_highlight_scenarios(objs, columns_axes=None, axis_labels=None, ideal_direction='top',
                                                    minmaxs=None, color_by_continuous=None,
                                                    color_palette_continuous=None, color_by_categorical=None,
                                                    color_palette_categorical=None, colorbar_ticks_continuous=None,
                                                    color_dict_categorical=None, zorder_by=None, zorder_num_classes=10,
                                                    zorder_direction='ascending', alpha_base=0.8, brushing_dict=None,
                                                    alpha_brush=0.05, lw_base=1.5, fontsize=14, figsize=(22, 10),
                                                    save_fig_filename=None, cluster_column_name='Cluster', title=None,
                                                    highlight_indices=None, highlight_colors=None,
                                                    highlight_descriptions=None, dpi=300):

    assert ideal_direction in ['top', 'bottom']
    assert zorder_direction in ['ascending', 'descending']

    if minmaxs is not None:
        for minmax in minmaxs:
            assert minmax in ['max', 'min']
    assert color_by_continuous is None or color_by_categorical is None

    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    objs_reorg, tops, bottoms = reorganize_objs(objs, columns_axes, ideal_direction, minmaxs)

    for i in range(objs_reorg.shape[0]):
        idx_val = objs.index[i]
        if (highlight_indices is None) or (idx_val not in highlight_indices):
            for j in range(objs_reorg.shape[1] - 1):
                y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j + 1]]
                x = [j, j + 1]
                ax.plot(x, y, c='lightgrey', alpha=0.5, zorder=2, lw=0.5)

    if brushing_dict is not None:
        satisfice = np.zeros(objs.shape[0]) == 0.
        for col_idx, (threshold, operator) in brushing_dict.items():
            if operator == '<':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] < threshold)
            elif operator == '<=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] <= threshold)
            elif operator == '>':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] > threshold)
            elif operator == '>=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] >= threshold)

            threshold_norm = (threshold - bottoms.iloc[col_idx]) / (tops.iloc[col_idx] - bottoms.iloc[col_idx])
            rect = Rectangle([col_idx - 0.05, 0], 0.1, threshold_norm)
            pc = PatchCollection([rect], facecolor='grey', alpha=0.5, zorder=3)
            ax.add_collection(pc)

    if highlight_indices is not None:
        highlight_labels = (highlight_descriptions if highlight_descriptions
                            else [f"Scenario {i + 1}" for i in range(len(highlight_indices))])
        for i in range(objs_reorg.shape[0]):
            idx_value = objs.index[i]
            if idx_value in highlight_indices:
                color = highlight_colors[highlight_indices.index(idx_value)]
                zorder = 15
                lw = 3
                label = highlight_labels[highlight_indices.index(idx_value)]
                for j in range(objs_reorg.shape[1] - 1):
                    y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j + 1]]
                    x = [j, j + 1]
                    ax.plot(x, y, c=color, alpha=alpha_base, zorder=zorder, lw=lw,
                            label=label if j == 0 else "")

    for j in range(len(columns_axes)):
        span = float(abs(tops.iloc[j] - bottoms.iloc[j]))
        if span >= 10:
            fmt = "{:.0f}"
        elif span >= 1:
            fmt = "{:.1f}"
        elif span >= 0.1:
            fmt = "{:.2f}"
        else:
            fmt = "{:.3f}"

        ax.annotate(fmt.format(tops.iloc[j]), [j, 1.02], ha='center', va='bottom',
                    zorder=5, fontsize=fontsize, color='black')
        ax.annotate(fmt.format(bottoms.iloc[j]), [j, -0.02], ha='center', va='top',
                    zorder=5, fontsize=fontsize, color='black')
        ax.plot([j, j], [0, 1], c='black', alpha=0.3, zorder=1)

    ax.set_xticks(range(len(columns_axes)))
    ax.set_xticklabels(axis_labels, rotation=45, ha='center', va='top', fontsize=fontsize)
    ax.tick_params(axis='x', colors='black', pad=10)
    ax.set_yticks([])

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    ax.set_xlim(-0.4, len(columns_axes) - 0.6)
    ax.set_ylim(-0.1, 1.1)

    if color_by_continuous is not None:
        mappable = cm.ScalarMappable(cmap=color_palette_continuous)
        mappable.set_clim(vmin=objs[columns_axes[color_by_continuous]].min(),
                          vmax=objs[columns_axes[color_by_continuous]].max())
        cb = plt.colorbar(mappable, ax=ax, orientation='horizontal', shrink=0.4, label=axis_labels[color_by_continuous],
                          pad=0.03, alpha=alpha_base)
        if colorbar_ticks_continuous is not None:
            _ = cb.ax.set_xticks(colorbar_ticks_continuous, colorbar_ticks_continuous,
                                 fontsize=fontsize)
        _ = cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize + 2, color='black', pad=20)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_height = max([label.get_window_extent(renderer).height for label in ax.get_xticklabels()])
    bottom_margin = max_label_height / fig.get_figheight() / fig.dpi
    legend_height = 0.15
    plt.subplots_adjust(bottom=bottom_margin + legend_height)

    leg = []
    if highlight_indices is not None:
        for idx, color, label in zip(highlight_indices, highlight_colors, highlight_labels):
            leg.append(Line2D([0], [0], color=color, lw=3, alpha=alpha_base, label=label))
    if leg:
        leg = ax.legend(handles=leg, loc='upper center',
                        bbox_to_anchor=(0.5, -bottom_margin / (bottom_margin + legend_height)),
                        ncol=len(highlight_indices), frameon=False, fontsize=fontsize)
        for text in leg.get_texts():
            text.set_color('black')

    plt.tight_layout()
    if save_fig_filename is not None:
        plt.savefig(save_fig_filename, bbox_inches='tight', dpi=dpi, facecolor='white')

    return fig, ax


def custom_parallel_coordinates_highlight_variability(objs, variability_data, columns_axes=None, axis_labels=None,
                                                      alpha_base=0.8, alpha_shade=0.2, lw_base=1.5,
                                                      fontsize=14, figsize=(22, 10), save_fig_filename=None,
                                                      title=None, highlight_indices=None,
                                                      highlight_colors=None, highlight_descriptions=None, dpi=300):
    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    y_mins = objs - variability_data
    y_maxs = objs + variability_data
    bottoms = y_mins.min()
    tops = y_maxs.max()
    objs_norm = (objs - bottoms) / (tops - bottoms)
    var_norm = variability_data / (tops - bottoms)

    if highlight_indices is not None:
        highlight_labels = highlight_descriptions if highlight_descriptions else [f"Scenario {i + 1}" for i in
                                                                                  range(len(highlight_indices))]
        for i, idx in enumerate(highlight_indices):
            if i >= len(highlight_colors) or i >= len(highlight_labels):
                break
            color = highlight_colors[i]
            label = highlight_labels[i]

            y = objs_norm.loc[idx]
            var = var_norm.loc[idx]
            ax.plot(range(len(columns_axes)), y, c=color, alpha=alpha_base, zorder=15, lw=2, label=label)

            for j in range(len(columns_axes) - 1):
                y_lower = [y.iloc[j] - var.iloc[j], y.iloc[j + 1] - var.iloc[j + 1]]
                y_upper = [y.iloc[j] + var.iloc[j], y.iloc[j + 1] + var.iloc[j + 1]]
                x = [j, j + 1]
                ax.fill_between(x, y_lower, y_upper, color=color, alpha=alpha_shade, zorder=10)

    ax.set_xlim(-0.5, len(columns_axes) - 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(range(len(columns_axes)))
    ax.set_xticklabels(axis_labels, rotation=45, ha='right', va='top', fontsize=fontsize)
    ax.tick_params(axis='x', pad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_visible(False)

    for i in range(len(columns_axes)):
        ax.axvline(x=i, color='grey', linestyle=':', alpha=0.5, zorder=1, ymin=-0.1, ymax=1.1)

    for j, (col, bot, top) in enumerate(zip(columns_axes, bottoms, tops)):
        ax.annotate(f'{bot:.0f}', (j, -0.1), xytext=(0, 5), textcoords='offset points', ha='center',
                    va='bottom', fontsize=fontsize - 2, color='black')
        ax.annotate(f'{top:.0f}', (j, 1.1), xytext=(0, -5), textcoords='offset points', ha='center', va='top',
                    fontsize=fontsize - 2, color='black')

    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=20)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_height = max([label.get_window_extent(renderer).height for label in ax.get_xticklabels()])
    bottom_margin = max_label_height / fig.get_figheight() / fig.dpi
    legend_height = 0.05

    plt.subplots_adjust(bottom=bottom_margin + legend_height)
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_margin / (bottom_margin + legend_height)),
                    ncol=len(highlight_indices), frameon=False, fontsize=fontsize)
    plt.tight_layout()

    if save_fig_filename:
        plt.savefig(save_fig_filename, bbox_inches='tight', dpi=dpi, facecolor='white')

    return fig, ax


def custom_parallel_coordinates_highlight_quantile(objs, lower_bound_data, upper_bound_data, columns_axes=None,
                                                   axis_labels=None, alpha_base=0.8, alpha_shade=0.2, lw_base=1.5,
                                                   fontsize=14, figsize=(22, 10), save_fig_filename=None, title=None,
                                                   highlight_indices=None,  highlight_colors=None,
                                                   highlight_descriptions=None, dpi=300):
    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    y_mins = lower_bound_data
    y_maxs = upper_bound_data
    bottoms = y_mins.min()
    tops = y_maxs.max()
    objs_norm = (objs - bottoms) / (tops - bottoms)
    lower_norm = (lower_bound_data - bottoms) / (tops - bottoms)
    upper_norm = (upper_bound_data - bottoms) / (tops - bottoms)

    if highlight_indices is not None:
        highlight_labels = highlight_descriptions if highlight_descriptions else [f"Scenario {i + 1}" for i in
                                                                                  range(len(highlight_indices))]
        for i, idx in enumerate(highlight_indices):
            if i >= len(highlight_colors) or i >= len(highlight_labels):
                break
            color = highlight_colors[i]
            label = highlight_labels[i]

            y = objs_norm.loc[idx]
            y_lower = lower_norm.loc[idx]
            y_upper = upper_norm.loc[idx]
            ax.plot(range(len(columns_axes)), y, c=color, alpha=alpha_base, zorder=15, lw=2, label=label)

            for j in range(len(columns_axes) - 1):
                y_lower_values = [y_lower.iloc[j], y_lower.iloc[j + 1]]
                y_upper_values = [y_upper.iloc[j], y_upper.iloc[j + 1]]
                x = [j, j + 1]
                ax.fill_between(x, y_lower_values, y_upper_values, color=color, alpha=alpha_shade, zorder=10)

    ax.set_xlim(-0.5, len(columns_axes) - 0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(range(len(columns_axes)))
    ax.set_xticklabels(axis_labels, rotation=45, ha='right', va='top', fontsize=fontsize)
    ax.tick_params(axis='x', pad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_visible(False)

    for i in range(len(columns_axes)):
        ax.axvline(x=i, color='grey', linestyle=':', alpha=0.5, zorder=1, ymin=-0.1, ymax=1.1)

    for j, (col, bot, top) in enumerate(zip(columns_axes, bottoms, tops)):
        ax.annotate(f'{bot:.0f}', (j, -0.1), xytext=(0, 5), textcoords='offset points', ha='center',
                    va='bottom', fontsize=fontsize - 2, color='black')
        ax.annotate(f'{top:.0f}', (j, 1.1), xytext=(0, -5), textcoords='offset points', ha='center', va='top',
                    fontsize=fontsize - 2, color='black')

    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=20)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_height = max([label.get_window_extent(renderer).height for label in ax.get_xticklabels()])
    bottom_margin = max_label_height / fig.get_figheight() / fig.dpi
    legend_height = 0.05

    plt.subplots_adjust(bottom=bottom_margin + legend_height)

    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_margin / (bottom_margin + legend_height)),
                    ncol=len(highlight_indices), frameon=False, fontsize=fontsize)

    plt.tight_layout()

    if save_fig_filename:
        plt.savefig(save_fig_filename, bbox_inches='tight', dpi=dpi, facecolor='white')

    return fig, ax


def custom_parallel_coordinates_highlight_scenarios_baseline_at_zero(objs, columns_axes=None, axis_labels=None,
                                                                     color_dict_categorical=None, alpha_base=0.8,
                                                                     lw_base=1.5, fontsize=14, figsize=(22, 8),
                                                                     save_fig_filename=None, title=None,
                                                                     highlight_indices=None, highlight_colors=None,
                                                                     highlight_descriptions=None, dpi=300):

    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    global_min = objs[columns_axes].min().min()
    global_max = objs[columns_axes].max().max()

    for idx in objs.index:
        if highlight_indices is not None and idx not in highlight_indices:
            ax.plot(range(len(columns_axes)), objs.loc[idx, columns_axes],
                    c='grey', alpha=0.1, zorder=5, lw=0.5)

    if highlight_indices is not None:
        highlight_labels = (highlight_descriptions if highlight_descriptions
                            else [f"Scenario {i + 1}" for i in range(len(highlight_indices))])

        for i, idx in enumerate(highlight_indices):
            color = highlight_colors[i]
            label = highlight_labels[i]
            ax.plot(range(len(columns_axes)), objs.loc[idx, columns_axes], c=color, alpha=alpha_base, zorder=15, \
                    lw=3, label=label)

    ax.set_xlim(-0.5, len(columns_axes) - 0.5)
    _safe_set_ylim(ax, global_min, global_max)

    ax.set_xticks(range(len(columns_axes)))
    ax.set_xticklabels(axis_labels, rotation=45, ha='right', fontsize=fontsize)

    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.tick_params(axis='y', colors='black', labelsize=fontsize)

    for i in range(len(columns_axes)):
        ax.axvline(x=i, color='gray', linestyle=':', alpha=0.3, zorder=1)

    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('black')

    ax.yaxis.grid(True, linestyle=':', alpha=0.3, color='gray')

    if title is not None:
        ax.set_title(title, fontsize=fontsize + 2, color='black', pad=20)
    ax.set_ylabel('Percentage Change from Baseline', fontsize=fontsize, color='black')

    plt.tight_layout()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_height = max([label.get_window_extent(renderer).height for label in ax.get_xticklabels()])

    bottom_margin = max_label_height / fig.get_figheight() / fig.dpi
    legend_height = 0.15
    plt.subplots_adjust(bottom=bottom_margin + legend_height)

    if highlight_indices is not None:
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_margin / (bottom_margin + legend_height)),
                        ncol=len(highlight_indices), frameon=False, fontsize=fontsize)

    if save_fig_filename is not None:
        plt.savefig(save_fig_filename, bbox_inches='tight', dpi=dpi, facecolor='white')

    return fig, ax


def custom_parallel_coordinates_relative_with_baseline_values(objs_rel, baseline_abs, axis_label_map, columns_axes=None,
                                                              axis_labels=None, alpha_base=0.8, lw_base=1.5,
                                                              fontsize=14, figsize=(22, 8), save_fig_filename=None,
                                                              title=None, highlight_indices=None, highlight_colors=None,
                                                              highlight_descriptions=None, dpi=300):

    def get_units_for_axis_label(label):
        if label in axis_label_map:
            return axis_label_map[label]['units']
        return ""

    if columns_axes is None:
        columns_axes = objs_rel.columns
    if axis_labels is None:
        axis_labels = columns_axes

    if isinstance(baseline_abs, pd.DataFrame):
        if baseline_abs.shape[0] == 1:
            baseline_abs = baseline_abs.iloc[0]
        else:
            raise ValueError("baseline_abs DataFrame must have exactly 1 row")

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    global_min = objs_rel[columns_axes].min().min()
    global_max = objs_rel[columns_axes].max().max()

    for idx in objs_rel.index:
        if highlight_indices is not None and idx not in highlight_indices:
            ax.plot(range(len(columns_axes)), objs_rel.loc[idx, columns_axes],
                    c='grey', alpha=0.1, lw=0.5, zorder=5)

    if highlight_indices is not None:
        if highlight_descriptions is None:
            highlight_descriptions = highlight_indices
        for i, idx in enumerate(highlight_indices):
            color = highlight_colors[i]
            label = highlight_descriptions[i]
            ax.plot( range(len(columns_axes)), objs_rel.loc[idx, columns_axes], c=color, alpha=alpha_base, lw=3,
                     label=label, zorder=10)

    ax.set_xlim(-0.5, len(columns_axes) - 0.5)
    _safe_set_ylim(ax, global_min, global_max)
    ax.set_xticks(range(len(columns_axes)))
    ax.set_xticklabels(axis_labels, rotation=45, ha='right', fontsize=fontsize)

    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.tick_params(axis='y', labelsize=fontsize)

    for i in range(len(columns_axes)):
        ax.axvline(x=i, color='gray', linestyle=':', alpha=0.3)

    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('black')

    ax.yaxis.grid(True, linestyle=':', alpha=0.3, color='gray')

    if title:
        ax.set_title(title, fontsize=fontsize + 2, color='black', pad=20)
    ax.set_ylabel('Percentage Change from Baseline', fontsize=fontsize)
    plt.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_height = max(label.get_window_extent(renderer).height for label in ax.get_xticklabels())
    bottom_margin = max_label_height / fig.get_figheight() / fig.dpi
    legend_height = 0.15
    plt.subplots_adjust(bottom=bottom_margin + legend_height)

    if highlight_indices is not None:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_margin / (bottom_margin + legend_height)),
                  ncol=len(highlight_indices), frameon=False, fontsize=fontsize)

    margin = 0.05 * (global_max - global_min)
    label_y = global_min - margin

    for i, col in enumerate(columns_axes):
        base_val = baseline_abs[col]
        if pd.isna(base_val):
            txt_label = "NaN"
        else:
            units_str = get_units_for_axis_label(col)
            txt_label = f"{base_val:,.0f} {units_str} (Baseline)"

        ax.text(i, label_y, txt_label, ha='center', va='top', fontsize=fontsize * 0.8, rotation=45, color='black')

    if save_fig_filename:
        plt.savefig(save_fig_filename, bbox_inches='tight', dpi=dpi, facecolor='white')

    return fig, ax


def get_scenario_styles(studies: tuple[int, int] | list[int], scenario_labels: dict[int, str], baseline_id: int,
                        label_style: str = "label_only", baseline_label_override: str | None = None,
                        colors: tuple[str, str] = ("black", "red"),
                        linestyles: tuple[str, str] = ("-", "--"),) -> dict[int, dict]:

    s1, s2 = (int(studies[0]), int(studies[1]))
    base = int(baseline_id) if int(baseline_id) in (s1, s2) else s1
    comp = s2 if s1 == base else s1

    def code(s: int) -> str:
        return f"s{s:04d}"

    base_label = baseline_label_override or scenario_labels.get(base, code(base))
    comp_label = scenario_labels.get(comp, code(comp))
    if label_style == "label_plus_code":
        base_label = f"{base_label} ({code(base)})"
        comp_label = f"{comp_label} ({code(comp)})"

    return {base: {"color": colors[0], "linestyle": linestyles[0], "label": base_label},
            comp: {"color": colors[1], "linestyle": linestyles[1], "label": comp_label}}


def infer_units(varname: str, units_map: dict[str, str] | None = None, default: str = "TAF") -> str:
    if units_map and varname in units_map:
        return units_map[varname]
    return default


def get_mean_for_axis_label(df: pd.DataFrame, axis_label: str, scenario: int, axis_label_map: dict[str, dict],
                            subset_years: list[int] | None = None,) -> float:

    info = axis_label_map[axis_label]
    calsim_vars = info["calsim_vars"]
    units = info["units"]
    months = info.get("months", None)

    combined_series = None
    suffix = f"s{int(scenario):04d}"

    for var in calsim_vars:
        df_sub = create_subset_unit(df, var, units)

        if hasattr(df_sub.columns, "levels") and len(getattr(df_sub.columns, "levels", [])) >= 2:
            keep_cols = [col for col in df_sub.columns if str(col[1]).endswith(suffix)]
        else:
            keep_cols = [col for col in df_sub.columns if str(col).endswith(suffix)]

        if not keep_cols:
            continue

        df_sub = df_sub[keep_cols]

        if months is not None:
            df_sub = df_sub[df_sub.index.month.isin(months)]

        series_sum = df_sub.sum(axis=1)

        combined_series = series_sum if combined_series is None else (combined_series + series_sum)

    if combined_series is None or combined_series.empty:
        return np.nan

    if subset_years is not None:
        wy_df = add_water_year_column(combined_series.to_frame())
        wy_df = wy_df.loc[wy_df["WaterYear"].isin(subset_years)]
        wy_df.drop(columns="WaterYear", inplace=True)
        combined_series = wy_df.iloc[:, 0]

    return float(combined_series.mean())


def build_parallel_df_absolute(df: pd.DataFrame, axis_label_map: dict[str, dict], scenario1: int, scenario2: int,
                               subset_years_s1: list[int] | None = None,
                               subset_years_s2: list[int] | None = None) -> pd.DataFrame:

    row_s1, row_s2 = {}, {}
    for label in axis_label_map.keys():
        row_s1[label] = get_mean_for_axis_label(df, label, scenario1, axis_label_map, subset_years=subset_years_s1)
        row_s2[label] = get_mean_for_axis_label(df, label, scenario2, axis_label_map, subset_years=subset_years_s2)

    return pd.DataFrame([row_s1, row_s2], index=[f"Scen{scenario1}", f"Scen{scenario2}"])


def build_parallel_df_relative(df: pd.DataFrame, axis_label_map: dict[str, dict], scenario1: int, scenario2: int,
                               subset_years_s1: list[int] | None = None,
                               subset_years_s2: list[int] | None = None) -> pd.DataFrame:

    baseline_vals, compare_vals = {}, {}
    for label in axis_label_map.keys():
        v1 = get_mean_for_axis_label(df, label, scenario1, axis_label_map, subset_years=subset_years_s1)
        v2 = get_mean_for_axis_label(df, label, scenario2, axis_label_map, subset_years=subset_years_s2)
        baseline_vals[label] = 0.0
        compare_vals[label] = np.nan if (pd.isna(v1) or v1 == 0) else (100.0 * (v2 - v1) / v1)

    return pd.DataFrame([baseline_vals, compare_vals], index=[f"Scen{scenario1}", f"Scen{scenario2}"])


def get_scenario_styles_multi(scenarios: list[int], *, scenario_labels: dict[int, str] | None = None,
                              baseline_id: int | None = None, baseline_label_override: str | None = None,
                              label_style: str = "label_only",
                              scenario_color_map: dict[int, str] | None = None) -> dict[int, dict]:

    styles: dict[int, dict] = {}
    cmap = get_cmap("tab20")

    def code(s: int) -> str:
        return f"s{s:04d}"

    def make_label(s: int) -> str:
        base_label = scenario_labels.get(s, code(s)) if scenario_labels else code(s)
        if s == baseline_id and baseline_label_override:
            base_label = baseline_label_override
        if label_style == "label_plus_code":
            return f"{base_label} ({code(s)})"
        return base_label

    used_colors = 0
    if baseline_id is not None and baseline_id in scenarios:
        baseline_color = scenario_color_map.get(baseline_id, "black") if scenario_color_map else "black"
        styles[baseline_id] = {"color": baseline_color, "linestyle": "-", "linewidth": 2.0,
                               "label": make_label(baseline_id)}

    for s in scenarios:
        if s == baseline_id:
            continue
        color = scenario_color_map.get(s) if scenario_color_map else None
        if color is None:
            color = cmap(used_colors % 20)
            used_colors += 1
        styles[s] = {"color": color, "linestyle": "-", "linewidth": 1.8, "label": make_label(s)}

    return styles

def plot_ts_multi(df: pd.DataFrame, *, varname: str, units: str = "TAF", scenarios: list[int],
                  scenario_labels: dict[int, str] | None = None, baseline_id: int | None = None,
                  baseline_label_override: str | None = None, label_style: str = "label_only",
                  scenario_color_map: dict[int, str] | None = None, months: list[int] | None = None,
                  start_date: str | None = None, end_date: str | None = None, pTitle: str = "Monthly Time Series",
                  xLab: str = "Date", lTitle: str = "Scenarios", fTitle: str = "ts_multi", fPath: str = "fPath",
                  pSave: bool = True, dpi: int = 300):

    series_map = cu.per_scenario_series(df, varname=varname, units=units, scenarios=scenarios, use_tucp=False,
                                        use_wyt=False, months=months)

    for sid, s in series_map.items():
        if start_date is not None:
            s = s.loc[s.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            s = s.loc[s.index <= pd.to_datetime(end_date)]
        series_map[sid] = s

    styles = get_scenario_styles_multi(scenarios, scenario_labels=scenario_labels, baseline_id=baseline_id,
                                       baseline_label_override=baseline_label_override, label_style=label_style,
                                       scenario_color_map=scenario_color_map)

    plt.figure(figsize=(14, 8))
    for sid in scenarios:
        s = series_map[sid]
        st = styles.get(sid, {})
        plt.plot(s.index, s.values, color=st.get("color", None), linestyle=st.get("linestyle", "-"),
                 linewidth=st.get("linewidth", 1.8), label=st.get("label", f"s{sid:04d}"))

    plt.title(f"{varname} {pTitle}", fontsize=18)
    plt.xlabel(xLab, fontsize=14)
    plt.ylabel(f"{varname}\nUnits: {units}", fontsize=14)
    plt.legend(title=lTitle, fontsize=12, title_fontsize=13, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if pSave:
        os.makedirs(fPath, exist_ok=True)
        plt.savefig(os.path.join(fPath, f"{varname}_{fTitle}.png"), dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_exceedance_multi(df: pd.DataFrame, *, varname: str, units: str = "TAF", scenarios: list[int],
                          scenario_labels: dict[int, str] | None = None, baseline_id: int | None = None,
                          baseline_label_override: str | None = None, label_style: str = "label_only",
                          scenario_color_map: dict[int, str] | None = None, use_tucp: bool = False,
                          tucp_var_base: str = "TUCP_TRIGGER_DV", tucp_wy_month_count: int = 1, use_wyt: bool = False,
                          wyt: list[int] | None = None, wyt_month: int | None = None, months: list[int] | None = None,
                          pTitle: str = "Exceedance Probability", xLab: str = "Probability", lTitle: str = "Scenarios",
                          fTitle: str = "exceed_multi", fPath: str = "fPath", pSave: bool = True, dpi: int = 300):

    series_map = cu.per_scenario_series(df, varname=varname, units=units, scenarios=scenarios, use_tucp=use_tucp,
                                        tucp_var_base=tucp_var_base, tucp_wy_month_count=tucp_wy_month_count,
                                        use_wyt=use_wyt, wyt=wyt, wyt_month=wyt_month, months=months)

    styles = get_scenario_styles_multi(scenarios, scenario_labels=scenario_labels, baseline_id=baseline_id,
                                       baseline_label_override=baseline_label_override, label_style=label_style,
                                       scenario_color_map=scenario_color_map)

    plt.figure(figsize=(14, 8))
    for sid in scenarios:
        s = series_map[sid].dropna()
        if s.empty:
            continue
        ex_df = single_exceed_alternative(s)
        ycol = ex_df.columns[0]
        st = styles.get(sid, {})
        plt.plot(ex_df.index, ex_df[ycol].values, color=st.get("color", None), linestyle=st.get("linestyle", "-"),
                 linewidth=st.get("linewidth", 1.8), label=st.get("label", f"s{sid:04d}"))

    plt.title(f"{varname} {pTitle}", fontsize=18)
    plt.xlabel(xLab, fontsize=14)
    plt.ylabel(f"{varname}\nUnits: {units}", fontsize=14)
    plt.legend(title=lTitle, fontsize=12, title_fontsize=13, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if pSave:
        os.makedirs(fPath, exist_ok=True)
        plt.savefig(os.path.join(fPath, f"{varname}_{fTitle}.png"), dpi=dpi, bbox_inches="tight")

    plt.close()


def plot_moy_averages_multi(df: pd.DataFrame, *, varname: str, units: str = "TAF", scenarios: list[int],
                            scenario_labels: dict[int, str] | None = None, baseline_id: int | None = None,
                            baseline_label_override: str | None = None, label_style: str = "label_only",
                            scenario_color_map: dict[int, str] | None = None, use_tucp: bool = False,
                            tucp_var_base: str = "TUCP_TRIGGER_DV", tucp_wy_month_count: int = 1,
                            use_wyt: bool = False, wyt: list[int] | None = None, wyt_month: int | None = None,
                            pTitle: str = "MOY Averages", xLab: str = "Month", lTitle: str = "Scenarios",
                            fTitle: str = "moy_multi", fPath: str = "fPath", pSave: bool = True, dpi: int = 300):

    series_map = cu.per_scenario_series(df, varname=varname, units=units, scenarios=scenarios, use_tucp=use_tucp,
                                        tucp_var_base=tucp_var_base, tucp_wy_month_count=tucp_wy_month_count,
                                        use_wyt=use_wyt, wyt=wyt, wyt_month=wyt_month, months=None)

    styles = get_scenario_styles_multi(scenarios, scenario_labels=scenario_labels, baseline_id=baseline_id,
                                       baseline_label_override=baseline_label_override, label_style=label_style,
                                       scenario_color_map=scenario_color_map)

    plt.figure(figsize=(14, 8))
    month_numbers = np.arange(1, 13)

    for sid in scenarios:
        s = series_map[sid].dropna()
        if s.empty:
            continue
        moy = s.groupby(s.index.month).mean()
        moy = moy.reindex(month_numbers)
        st = styles.get(sid, {})
        plt.plot(month_numbers, moy.values, marker="o", color=st.get("color", None),
                 linestyle=st.get("linestyle", "-"), linewidth=st.get("linewidth", 1.8),
                 label=st.get("label", f"s{sid:04d}"))

    plt.title(f"{varname} {pTitle}", fontsize=18)
    plt.xlabel(xLab, fontsize=14)
    plt.xticks(month_numbers, [str(m) for m in month_numbers])
    plt.ylabel(f"{varname}\nUnits: {units}", fontsize=14)
    plt.legend(title=lTitle, fontsize=12, title_fontsize=13, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()

    if pSave:
        os.makedirs(fPath, exist_ok=True)
        plt.savefig(os.path.join(fPath, f"{varname}_{fTitle}.png"), dpi=dpi, bbox_inches="tight")

    plt.close()

def plot_annual_totals_ts_multi(df: pd.DataFrame, *, varname: str, units: str = "TAF", scenarios: list[int],
                                scenario_labels: dict[int, str] | None = None, baseline_id: int | None = None,
                                baseline_label_override: str | None = None, label_style: str = "label_only",
                                scenario_color_map: dict[int, str] | None = None, months: list[int] | None = None,
                                freq: str = "YS-OCT", pTitle: str = "Annual Totals", xLab: str = "Water Year",
                                lTitle: str = "Scenarios", fTitle: str = "ann_tot_ts_multi", fPath: str = "fPath",
                                pSave: bool = True, dpi: int = 300):

    series_map = cu.per_scenario_series(df, varname=varname, units=units, scenarios=scenarios, use_tucp=False,
                                        use_wyt=False, months=months)

    styles = get_scenario_styles_multi(scenarios, scenario_labels=scenario_labels, baseline_id=baseline_id,
                                       baseline_label_override=baseline_label_override, label_style=label_style,
                                       scenario_color_map=scenario_color_map)

    plt.figure(figsize=(14, 8))
    for sid in scenarios:
        s = series_map[sid].dropna()
        if s.empty:
            continue
        ann = s.resample(freq).sum(min_count=1)
        st = styles.get(sid, {})
        plt.plot(ann.index, ann.values, drawstyle="steps-post", color=st.get("color", None),
                 linestyle=st.get("linestyle", "-"), linewidth=st.get("linewidth", 1.8),
                 label=st.get("label", f"s{sid:04d}"))

    plt.title(f"{varname} {pTitle}", fontsize=18)
    plt.xlabel(xLab, fontsize=14)
    plt.ylabel(f"{varname}\nUnits: {units}", fontsize=14)
    plt.legend(title=lTitle, fontsize=12, title_fontsize=13, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if pSave:
        os.makedirs(fPath, exist_ok=True)
        plt.savefig(os.path.join(fPath, f"{varname}_{fTitle}.png"), dpi=dpi, bbox_inches="tight")

    plt.close()


def annualize_exceedance_multi(df: pd.DataFrame, *, varname: str, units: str = "TAF", scenarios: list[int],
                               scenario_labels: dict[int, str] | None = None, baseline_id: int | None = None,
                               baseline_label_override: str | None = None, label_style: str = "label_only",
                               scenario_color_map: dict[int, str] | None = None, use_tucp: bool = False,
                               tucp_var_base: str = "TUCP_TRIGGER_DV", tucp_wy_month_count: int = 1,
                               use_wyt: bool = False, wyt: list[int] | None = None, wyt_month: int | None = None,
                               months: list[int] | None = None, freq: str = "YS-OCT", pTitle: str = "Annual Exceedance",
                               xLab: str = "Exceedance Probability", lTitle: str = "Scenarios",
                               fTitle: str = "ann_exceed_multi", fPath: str = "fPath", pSave: bool = True,
                               dpi: int = 300):

    series_map = cu.per_scenario_series(df, varname=varname, units=units, scenarios=scenarios, use_tucp=use_tucp,
                                        tucp_var_base=tucp_var_base, tucp_wy_month_count=tucp_wy_month_count,
                                        use_wyt=use_wyt, wyt=wyt, wyt_month=wyt_month, months=months)

    styles = get_scenario_styles_multi(scenarios, scenario_labels=scenario_labels, baseline_id=baseline_id,
                                       baseline_label_override=baseline_label_override, label_style=label_style,
                                       scenario_color_map=scenario_color_map)

    plt.figure(figsize=(14, 8))
    for sid in scenarios:
        s = series_map[sid].dropna()
        if s.empty:
            continue
        ann = s.resample(freq).sum(min_count=1).dropna()
        if ann.empty:
            continue
        ex_df = single_exceed_alternative(ann)
        ycol = ex_df.columns[0]
        st = styles.get(sid, {})
        plt.plot(ex_df.index, ex_df[ycol].values, color=st.get("color", None), linestyle=st.get("linestyle", "-"),
                 linewidth=st.get("linewidth", 1.8), label=st.get("label", f"s{sid:04d}"))

    plt.title(f"{varname} {pTitle}", fontsize=18)
    plt.xlabel(xLab, fontsize=14)
    plt.ylabel(f"{varname}\nUnits: {units}", fontsize=14)
    plt.legend(title=lTitle, fontsize=12, title_fontsize=13, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if pSave:
        os.makedirs(fPath, exist_ok=True)
        plt.savefig(os.path.join(fPath, f"{varname}_{fTitle}.png"), dpi=dpi, bbox_inches="tight")
    plt.close()


def _safe_set_ylim(ax, y_min, y_max, frac_margin=0.02, abs_margin=1.0):

    try:
        y_min = float(y_min)
        y_max = float(y_max)
    except Exception:
        ax.autoscale()
        return

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        ax.autoscale()
        return

    if y_min == y_max:
        pad = max(abs(y_min) * frac_margin, abs_margin if y_min == 0 else abs(y_min) * frac_margin)
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        ax.set_ylim(y_min, y_max)


def custom_parallel_coordinates_highlight_cluster(objs, columns_axes=None, axis_labels=None, ideal_direction='top',
                                                  minmaxs=None, color_by_continuous=None, color_palette_continuous=None,
                                                  color_by_categorical=None, color_palette_categorical=None,
                                                  colorbar_ticks_continuous=None, color_dict_categorical=None,
                                                  zorder_by=None, zorder_num_classes=10, zorder_direction='ascending',
                                                  alpha_base=0.8, brushing_dict=None, alpha_brush=0.05, lw_base=1.5,
                                                  fontsize=14, figsize=(11, 6), save_fig_filename=None,
                                                  cluster_column_name='Cluster', title=None, highlight_indices=None,
                                                  highlight_colors=None, dpi=600):

    assert ideal_direction in ['top', 'bottom']
    assert zorder_direction in ['ascending', 'descending']
    if minmaxs is not None:
        for minmax in minmaxs:
            assert minmax in ['max', 'min']
    assert color_by_continuous is None or color_by_categorical is None
    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    fig, ax = plt.subplots(1, 1, figsize=figsize, gridspec_kw={'hspace': 0.1, 'wspace': 0.1})

    objs_reorg, tops, bottoms = reorganize_objs(objs, columns_axes, ideal_direction, minmaxs)

    if brushing_dict is not None:
        satisfice = np.zeros(objs.shape[0]) == 0.
        for col_idx, (threshold, operator) in brushing_dict.items():
            if operator == '<':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] < threshold)
            elif operator == '<=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] <= threshold)
            elif operator == '>':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] > threshold)
            elif operator == '>=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] >= threshold)

            threshold_norm = (threshold - bottoms[col_idx]) / (tops[col_idx] - bottoms[col_idx])
            rect = Rectangle([col_idx - 0.05, 0], 0.1, threshold_norm)
            pc = PatchCollection([rect], facecolor='grey', alpha=0.5, zorder=3)
            ax.add_collection(pc)

    baseline_present = 0 in objs.index
    highlight_labels = [f"median {i + 1}" for i in range(len(highlight_indices))] if highlight_indices else []
    highlight_colors = highlight_colors or []

    if baseline_present and highlight_indices is not None:
        highlight_indices = [0] + list(highlight_indices)
        highlight_labels = ["baseline"] + highlight_labels
        highlight_colors = ["black"] + list(highlight_colors)

    for i in range(objs_reorg.shape[0]):
        idx_value = objs.index[i]
        if idx_value == 0 and baseline_present:
            color = "black"
            zorder = 20
            lw = 4
            label = "baseline"
        elif highlight_indices and idx_value in highlight_indices:
            color = highlight_colors[highlight_indices.index(idx_value)]
            zorder = 15
            lw = 4
            label = highlight_labels[highlight_indices.index(idx_value)]
        elif color_by_categorical is not None and cluster_column_name in objs.columns:
            cluster_value = objs[cluster_column_name].iloc[i]
            color = color_dict_categorical.get(cluster_value, 'grey')
            zorder = 4
            lw = lw_base
            label = None
        else:
            color = color_dict_categorical[1] if color_dict_categorical else 'grey'
            zorder = 4
            lw = lw_base
            label = None

        alpha = alpha_base

        for j in range(objs_reorg.shape[1] - 1):
            y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j + 1]]
            x = [j, j + 1]
            ax.plot(x, y, c=color, alpha=alpha, zorder=zorder, lw=lw)

    for j in range(len(columns_axes)):
        ax.annotate(str(round(tops[j])), [j, 1.02], ha='center', va='bottom', zorder=5, fontsize=fontsize)
        ax.annotate(str(round(bottoms[j])), [j, -0.02], ha='center', va='top', zorder=5, fontsize=fontsize)
        ax.plot([j, j], [0, 1], c='k', zorder=1)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    if ideal_direction == 'top':
        ax.arrow(-0.15, 0.1, 0, 0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    elif ideal_direction == 'bottom':
        ax.arrow(-0.15, 0.9, 0, -0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    ax.annotate('Direction of preference', xy=(-0.3, 0.5), ha='center', va='center', rotation=90,
                fontsize=fontsize)

    ax.set_xlim(-0.4, len(columns_axes) - 0.6)
    ax.set_ylim(-0.4, 1.1)

    for i, l in enumerate(axis_labels):
        ax.annotate(l, xy=(i, -0.12), ha='center', va='top', fontsize=fontsize)

    if color_by_continuous is not None:
        mappable = cm.ScalarMappable(cmap=color_palette_continuous)
        mappable.set_clim(vmin=objs[columns_axes[color_by_continuous]].min(),
                          vmax=objs[columns_axes[color_by_continuous]].max())
        cb = plt.colorbar(mappable, ax=ax, orientation='horizontal', shrink=0.4, label=axis_labels[color_by_continuous],
                          pad=0.03, alpha=alpha_base)
        if colorbar_ticks_continuous is not None:
            _ = cb.ax.set_xticks(colorbar_ticks_continuous, colorbar_ticks_continuous, fontsize=fontsize)
        _ = cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize)
    elif color_by_categorical is not None or highlight_indices is not None:
        leg = []
        if color_by_categorical is not None and color_dict_categorical:
            for label, color in color_dict_categorical.items():
                leg.append(Line2D([0], [0], color=color, lw=3, alpha=alpha_base, label=label))
        if highlight_indices is not None:
            for idx, color, label in zip(highlight_indices, highlight_colors, highlight_labels):
                leg.append(Line2D([0], [0], color=color, lw=3, alpha=alpha_base, label=label))

        if leg and color_dict_categorical:
            _ = ax.legend(handles=leg, loc='lower center', ncol=max(3, len(color_dict_categorical)),
                          bbox_to_anchor=[0.5, -0.07], frameon=False, fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    if save_fig_filename is not None:
        plt.savefig(save_fig_filename, bbox_inches='tight', transparent=True, dpi=dpi)

    return fig, ax


def custom_parallel_coordinates_highlight_iqr(objs, columns_axes=None, axis_labels=None, ideal_direction='top',
                                              minmaxs=None, color_by_continuous=None, color_palette_continuous=None,
                                              color_by_categorical=None, color_palette_categorical=None,
                                              colorbar_ticks_continuous=None, color_dict_categorical=None,
                                              zorder_by=None, zorder_num_classes=10, zorder_direction='ascending',
                                              alpha_base=0.8, brushing_dict=None, alpha_brush=0.05, lw_base=1.5,
                                              fontsize=14, figsize=(11, 6), save_fig_filename=None,
                                              cluster_column_name='Cluster', title=None, highlight_indices=None,
                                              highlight_colors=None, filter_indices=None, iqr_data=None, dpi=600):

    assert ideal_direction in ['top', 'bottom']
    assert zorder_direction in ['ascending', 'descending']
    if minmaxs is not None:
        for minmax in minmaxs:
            assert minmax in ['max', 'min']
    assert color_by_continuous is None or color_by_categorical is None
    if columns_axes is None:
        columns_axes = objs.columns
    if axis_labels is None:
        axis_labels = columns_axes

    if filter_indices is not None:
        objs = objs.loc[filter_indices]
        if iqr_data is not None:
            iqr_data = iqr_data.loc[filter_indices]

    fig, ax = plt.subplots(1, 1, figsize=figsize, gridspec_kw={'hspace': 0.1, 'wspace': 0.1})

    objs_reorg, tops, bottoms = reorganize_objs(objs, columns_axes, ideal_direction, minmaxs)

    if brushing_dict is not None:
        satisfice = np.zeros(objs.shape[0]) == 0.
        for col_idx, (threshold, operator) in brushing_dict.items():
            if operator == '<':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] < threshold)
            elif operator == '<=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] <= threshold)
            elif operator == '>':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] > threshold)
            elif operator == '>=':
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] >= threshold)

            threshold_norm = (threshold - bottoms[col_idx]) / (tops[col_idx] - bottoms[col_idx])
            rect = Rectangle([col_idx - 0.05, 0], 0.1, threshold_norm)
            pc = PatchCollection([rect], facecolor='grey', alpha=0.5, zorder=3)
            ax.add_collection(pc)

    if iqr_data is not None and color_dict_categorical is not None:
        for i, idx in enumerate(objs.index):
            cluster_value = objs[cluster_column_name].loc[idx]
            color = color_dict_categorical.get(cluster_value, 'lightgrey')

            for j, col in enumerate(columns_axes[:-1]):
                iqr_bottom = objs[col].iloc[i] - (iqr_data[col].iloc[i] / 2)
                iqr_top = objs[col].iloc[i] + (iqr_data[col].iloc[i] / 2)

                iqr_bottom_norm = (iqr_bottom - bottoms[j]) / (tops[j] - bottoms[j])
                iqr_top_norm = (iqr_top - bottoms[j]) / (tops[j] - bottoms[j])

                rect = Rectangle([j - 0.05, iqr_bottom_norm], 0.1, iqr_top_norm - iqr_bottom_norm)
                pc = PatchCollection([rect], facecolor=color, alpha=0.3, zorder=2)
                ax.add_collection(pc)

    baseline_present = 0 in objs.index
    highlight_labels = [f"median {i + 1}" for i in range(len(highlight_indices))] if highlight_indices else []
    highlight_colors = highlight_colors or []

    if baseline_present and highlight_indices is not None:
        highlight_indices = [0] + list(highlight_indices)
        highlight_labels = ["baseline"] + highlight_labels
        highlight_colors = ["black"] + list(highlight_colors)

    for i in range(objs_reorg.shape[0]):
        idx_value = objs.index[i]
        if idx_value == 0 and baseline_present:
            color = "black"
            zorder = 20
            lw = 4
            label = "baseline"
        elif highlight_indices and idx_value in highlight_indices:
            color = highlight_colors[highlight_indices.index(idx_value)]
            zorder = 15
            lw = 4
            label = highlight_labels[highlight_indices.index(idx_value)]
        elif color_by_categorical is not None and cluster_column_name in objs.columns:
            cluster_value = objs[cluster_column_name].iloc[i]
            color = color_dict_categorical.get(cluster_value, 'grey')
            zorder = 4
            lw = lw_base
            label = None
        else:
            color = color_dict_categorical[1] if color_dict_categorical else 'grey'
            zorder = 4
            lw = lw_base
            label = None

        alpha = alpha_base

        for j in range(objs_reorg.shape[1] - 1):
            y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j + 1]]
            x = [j, j + 1]
            ax.plot(x, y, c=color, alpha=alpha, zorder=zorder, lw=lw)

    for j in range(len(columns_axes)):
        ax.annotate(str(round(tops[j])), [j, 1.02], ha='center', va='bottom', zorder=5, fontsize=fontsize)
        ax.annotate(str(round(bottoms[j])), [j, -0.02], ha='center', va='top', zorder=5, fontsize=fontsize)
        ax.plot([j, j], [0, 1], c='k', zorder=1)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    if ideal_direction == 'top':
        ax.arrow(-0.15, 0.1, 0, 0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    elif ideal_direction == 'bottom':
        ax.arrow(-0.15, 0.9, 0, -0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    ax.annotate('Direction of preference', xy=(-0.3, 0.5), ha='center', va='center', rotation=90, fontsize=fontsize)

    ax.set_xlim(-0.4, len(columns_axes) - 0.6)
    ax.set_ylim(-0.4, 1.1)

    for i, l in enumerate(axis_labels):
        ax.annotate(l, xy=(i, -0.12), ha='center', va='top', fontsize=fontsize)

    if color_by_continuous is not None:
        mappable = cm.ScalarMappable(cmap=color_palette_continuous)
        mappable.set_clim(vmin=objs[columns_axes[color_by_continuous]].min(),
                          vmax=objs[columns_axes[color_by_continuous]].max())
        cb = plt.colorbar(mappable, ax=ax, orientation='horizontal', shrink=0.4, label=axis_labels[color_by_continuous],
                          pad=0.03, alpha=alpha_base)
        if colorbar_ticks_continuous is not None:
            _ = cb.ax.set_xticks(colorbar_ticks_continuous, colorbar_ticks_continuous,
                                 fontsize=fontsize)
        _ = cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize)
    elif color_by_categorical is not None or highlight_indices is not None:
        leg = []
        if color_by_categorical is not None and color_dict_categorical:
            for label, color in color_dict_categorical.items():
                leg.append(Line2D([0], [0], color=color, lw=3, alpha=alpha_base, label=label))
        if highlight_indices is not None:
            for idx, color, label in zip(highlight_indices, highlight_colors, highlight_labels):
                leg.append(Line2D([0], [0], color=color, lw=3, alpha=alpha_base, label=label))

        if leg and color_dict_categorical:
            _ = ax.legend(handles=leg, loc='lower center', ncol=max(3, len(color_dict_categorical)),
                          bbox_to_anchor=[0.5, -0.07], frameon=False, fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    if save_fig_filename is not None:
        plt.savefig(save_fig_filename, bbox_inches='tight', transparent=True, dpi=dpi)

    return fig, ax
