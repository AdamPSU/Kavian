import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from kavian.eda.config import NUM, CAT
from kavian.eda.utils import subset_handler

sns.set_style('whitegrid')

def _barplot_framework(ax, title):
    """
    General styles such as titles, axis formatting, and
    labeling present in all Kavian barplots.
    """
    ax.xaxis.grid(True, linestyle='--', which='major', color='black', alpha=0.10)
    ax.axvline(50, color='black', alpha=0.10)
    ax.tick_params(axis='y', rotation=20)

    xticks = np.arange(0, 101, 10)
    xlabels = [f'{x:.0f}%' for x in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontstyle='italic')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(f'{title} Barchart', fontdict={'fontsize': 24, 'fontfamily': 'serif'})

    for bar in ax.patches:
        bar.set_linewidth(1)
        bar.set_edgecolor('black')


def mode_barplot(dataframe, palette='kavian', subset=None, sort=True):
    """
    Plots the percentages of the most common value in each column.
    """

    dataframe = subset_handler(dataframe, subset)

    mode_percents = []
    size = len(dataframe)

    for col in dataframe:
        mode_size = dataframe[col].value_counts().iloc[0]
        mode_percent = mode_size / size * 100

        mode_percents.append(mode_percent)

    mode_df = pd.DataFrame({
        'Feature': dataframe.columns,
        'Mode %': mode_percents
    })

    if sort:
        mode_df = mode_df.sort_values(by='Mode %', ascending=False)

    if palette == 'kavian':
        palette = ['#e85440' if p > 50 else '#17aab5' for p in mode_df['Mode %']]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=mode_df, x='Mode %', y='Feature', orient='h',
                palette=palette, ax=ax)

    _barplot_framework(ax, 'Mode%')

    annot = True
    if len(mode_df) > 15:
        annot = False

    if annot:
        small_percents = [f'{p:.2f}%' if p < 50 else '' for p in mode_df['Mode %']]
        large_percents = [f'{p:.2f}%' if p >= 50 else '' for p in mode_df['Mode %']]

        for container in ax.containers:
            ax.bar_label(container, labels=small_percents,
                         padding=5, color='black', fontweight='bold', fontstyle='italic')
            ax.bar_label(container, labels=large_percents,
                         padding=-50, color='white', fontweight='bold', fontstyle='italic')

    plt.tight_layout()
    plt.show()


def null_barplot(dataframe, palette='kavian', subset=None, sort=True):
    """
    Plots the percentages of the missing values in the dataframe.
    """

    dataframe = subset_handler(dataframe, subset)

    missing_series = dataframe.isna().sum()
    missing_series = missing_series[missing_series > 0] / len(dataframe) * 100

    if sort:
        missing_series = missing_series.sort_values(ascending=False)

    missing_df = pd.DataFrame({
        'Feature': missing_series.index,
        'Null %': missing_series.values
    })

    if palette == 'kavian':
        palette = ['#e85440' if p > 10 else '#17aab5' for p in missing_df['Null %']]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=missing_df, x='Null %', y='Feature', orient='h',
                palette=palette, ax=ax)

    _barplot_framework(ax, 'Null%')

    annot = True
    if len(missing_df) > 15:
        annot = False

    if annot:
        small_percents = [f'{p:.2f}%' if p < 10 else '' for p in missing_df['Null %']]
        large_percents = [f'{p:.2f}%' if p >= 10 else '' for p in missing_df['Null %']]

        for container in ax.containers:
            ax.bar_label(container, labels=small_percents,
                         padding=5, color='black', fontweight='bold', fontstyle='italic')
            ax.bar_label(container, labels=large_percents,
                         padding=-50, color='white', fontweight='bold', fontstyle='italic')

    plt.tight_layout()
    plt.show()

def mode_null_barplot(dataframe, palette='kavian', subset=None):
    """
    Plots the percentages of both the most common value in each
    column and the missing values in the dataframe.
    """
    dataframe_copy = dataframe.copy()
    dataframe_copy = subset_handler(dataframe, subset)

    null_cols = dataframe_copy.columns[dataframe_copy.isna().any()].tolist()
    non_null_cols = dataframe_copy.columns.difference(null_cols).tolist()
    dataframe_copy = dataframe_copy.reindex(columns=null_cols + non_null_cols)

    null_series = dataframe_copy.isna().sum() / len(dataframe_copy) * 100

    mode_percents = []
    size = len(dataframe_copy)
    for col in dataframe_copy:
        mode_size = dataframe_copy[col].value_counts().iloc[0]
        mode_percent = mode_size / size * 100
        mode_percents.append(mode_percent)

    combined_df = pd.DataFrame({
        'Feature': dataframe_copy.columns,
        'Null %': null_series,
        'Mode %': mode_percents
    })

    melted_df = combined_df.melt(id_vars='Feature', value_vars=['Null %', 'Mode %'], var_name='Metric',
                                 value_name='Percent')

    if palette == 'kavian':
        palette = ['#e85440', '#17aab5']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted_df, x='Percent', y='Feature', hue='Metric',
                orient='h', palette=palette, ax=ax)

    _barplot_framework(ax, 'EDA')

    annot = True
    if len(combined_df) > 8:
        annot = False

    if annot:
        for i, container in enumerate(ax.containers):
            small_percents, large_percents = [], []

            for j, rect in enumerate(container):
                bar = rect.get_width()

                if bar == 0:
                    small_percents.append('')
                    large_percents.append('')
                elif bar < 50:
                    small_percents.append(f'{bar:.2f}%')
                    large_percents.append('')
                elif bar >= 50:
                    large_percents.append(f'{bar:.2f}%')
                    small_percents.append('')

            ax.bar_label(container, labels=small_percents, padding=5,
                         color='black', fontweight='bold', fontstyle='italic')
            ax.bar_label(container, labels=large_percents, padding=-50,
                         color='white', fontweight='bold', fontstyle='italic')

    plt.tight_layout()
    plt.show()


def heatmap(dataframe, palette='kavian', subset=None):
    if subset:
        dataframe = dataframe[subset]

    fig, ax = plt.subplots(figsize=(10, 6))

    numerical = dataframe.select_dtypes(NUM)
    corr = numerical.corr()

    font_size = 18 - len(numerical.columns)
    font_size = 8 if font_size < 8 else font_size
    cbar_kws = {'pad': 0.01}

    if palette == 'kavian':
        palette = sns.diverging_palette(18, 240, s=80, l=50, n=19, center="dark")

    too_many_cols = len(numerical.columns) >= 12

    # Too many values for annotations to be legible
    if too_many_cols:
        sns.heatmap(corr, ax=ax, cmap=palette, cbar_kws=cbar_kws,
                    fmt='.2f', linecolor='black', linewidth=0.5, square=True)

    else:
        annot_kws = {'size': font_size, 'fontweight': 'bold', 'fontstyle': 'italic'}
        sns.heatmap(corr, ax=ax, cmap=palette, annot=True, annot_kws=annot_kws,
                    cbar_kws=cbar_kws, fmt='.2f', linecolor='black', linewidth=0.5, square=True)

    xticklabels = '' if too_many_cols else ax.get_xticklabels()

    ax.tick_params(rotation=20)
    ax.set_xticklabels(xticklabels, fontweight='bold', fontstyle='italic', fontsize=font_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold', fontstyle='italic', fontsize=font_size)
    ax.set_title(f'EDA Heatmap', fontdict={'fontsize': 24, 'fontfamily': 'serif'})

    plt.tight_layout()
    plt.show()


def _format_chart_data(col):
    min_val = col.min()
    median_val = col.median()
    max_val = col.max()

    # Check if median is too close to min or max
    min_max_distance = max_val - min_val
    if abs(median_val - min_val) <= 0.15 * min_max_distance or \
            abs(median_val - max_val) <= 0.15 * min_max_distance:

        ticks = [min_val, max_val]
        labels = [_format_value(min_val), _format_value(max_val)]
        labels = [f'{min_val:.2f}', f'{max_val:.2f}']

    else:
        ticks = [min_val, median_val, max_val]
        labels = [_format_value(min_val), _format_value(median_val), _format_value(max_val)]

    return ticks, labels


def _format_value(value):
    if abs(value) >= 100_000:
        exponent = int(np.log10(abs(value)))
        mantissa = value / 10 ** exponent

        return f'{mantissa:.0f}x10^{exponent}'
    else:

        return f'{value:.0f}'


def numerical_plot(col, palette='kavian', flip_colors=False):
    sns.set_style('ticks')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharex=True)

    if palette == 'kavian':
        palette = ['#17aab5', '#e85440']

    histplot_params = {'x': col,
                       'ax': axes[0],
                       'kde': True,
                       'line_kws': {'linewidth': 2},
                       'fill': False,
                       }

    boxplot_params = {'x': col,
                      'ax': axes[1],
                      'notch': True,
                      'linewidth': 2,
                      'width': 0.2,
                      'boxprops': {'edgecolor': 'black'},
                      'medianprops': {'color': 'black'},
                      'whiskerprops': {'color': 'black'},
                      'capprops': {'color': 'black'},
                      'flierprops': {'marker': 'x'},
                      }

    if not flip_colors:
        sns.histplot(color=palette[0], **histplot_params)
        sns.boxplot(color=palette[1], **boxplot_params)
    else:
        sns.histplot(color=palette[1], **histplot_params)
        sns.boxplot(color=palette[0], **boxplot_params)

    # Histplot
    axes[0].tick_params(axis='y', rotation=20)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontweight='bold', fontstyle='italic')
    axes[0].set_ylabel('Count', fontstyle='italic', size=16, fontfamily='serif')

    ticks, labels = _format_chart_data(col)

    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(labels, fontstyle='italic')
    axes[0].set_xlabel('')

    # Add relevant statistics to the histplot graph
    text_params = {'x': 0.50,
                   'transform': axes[0].transAxes,
                   'fontdict': {'weight': 'bold', 'style': 'italic'},
                   'bbox': {'facecolor': 'white', 'edgecolor': palette[1], 'boxstyle': 'roundtooth'}}

    axes[0].text(y=0.93, s=f"Skewness: {col.skew():.2f}", **text_params)
    axes[0].text(y=0.87, s=f"Null Count: {col.isna().sum()}", **text_params)

    # Boxplot
    axes[1].set_xticklabels(axes[1].get_xticklabels(), fontstyle='italic')
    axes[1].axvline(col.median(), color="black", linewidth=2, dashes=(2, 2))
    axes[1].set_xlabel('')

    plt.suptitle(f'{col.name} Distribution Analysis', size=24, fontfamily='serif')
    plt.tight_layout()
    plt.show()


def gen_numerical_plots(dataframe, palette='kavian'):
    numerical = dataframe.select_dtypes(include=NUM)

    flip_switch = False
    for col in numerical:
        numerical_plot(numerical[col], palette=palette, flip_colors=flip_switch)
        flip_switch = not flip_switch


