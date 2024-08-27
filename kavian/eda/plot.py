import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')

def null_barplot(dataframe, palette='kavian_defaults', orient='h', subset=None):
    if subset:
        dataframe = dataframe[subset]

    missing_series = dataframe.isna().sum()
    missing_series = missing_series[missing_series > 0] / len(dataframe) * 100
    missing_series = missing_series.sort_values(ascending=False)

    missing_df = pd.DataFrame({
        'Feature': missing_series.index,
        'Null %': missing_series.values
    })

    annot = True
    if len(missing_df) > 10:
        annot = False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.xaxis.grid(True, linestyle='--', which='major', color='black', alpha=0.10)
    ax.axvline(50, color='black', alpha=0.10)
    ax.tick_params(axis='y', rotation=20)

    xticks = np.arange(0, 101, 10)
    xlabels = [f'{x:.0f}%' for x in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel('')
    ax.set_xlabel('')

    if palette == 'kavian_defaults':
        palette = ['#e85440' if p > 10 else '#17aab5' for p in missing_df['Null %']]

    sns.barplot(data=missing_df, x='Null %', y='Feature', orient=orient,
                palette=palette, ax=ax)

    if annot:
        small_percents = [f'{p:.2f}' if p < 10 else '' for p in missing_df['Null %']]
        large_percents = [f'{p:.2f}' if p >= 10 else '' for p in missing_df['Null %']]

        for container in ax.containers:
            ax.bar_label(container, labels=small_percents,
                         padding=5, color='black', fontweight='bold', fontstyle='italic')
            ax.bar_label(container, labels=large_percents,
                         padding=-50, color='white', fontweight='bold', fontstyle='italic')

    for bar in ax.patches:
        bar.set_linewidth(1)
        bar.set_edgecolor('black')

    plt.tight_layout()
    plt.show()