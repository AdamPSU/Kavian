import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

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

    if subset:
        dataframe = dataframe[subset]

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

    if subset:
        dataframe = dataframe[subset]

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

def eda_barplot(dataframe, palette='kavian', subset=None):
    """
    Plots the percentages of both the most common value in each
    column and the missing values in the dataframe.
    """
    dataframe_copy = dataframe.copy()

    if subset:
        dataframe_copy = dataframe_copy[subset]

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
