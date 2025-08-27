import pandas as pd
import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

west = pd.read_pickle('/home/catherinej/BarrierIslandBreach/PaperFigures/west_fgmax_points.pkl.gz')
central = pd.read_pickle('/home/catherinej/BarrierIslandBreach/PaperFigures/central_fgmax_pts.pkl.gz')
east = pd.read_pickle('/home/catherinej/BarrierIslandBreach/PaperFigures/east_fgmax_pts.pkl.gz')

subsets = [central[central.columns[1]], west[west.columns[1]], east[east.columns[1]]]
names = ['west', 'central', 'east']

hist_stats = {}

for idx, subset in enumerate(subsets):
    x = {}
    y = {}
    z = {}
    k = {}
    l = {}
    m = {}

    wd = subset[subset.index.str.contains('dw')]
    rall = subset[subset.index.str.contains('loc')]
    depth = subset[~subset.index.str.contains('|'.join(['w', 'loc','east', 'west']))]
    width = subset[~subset.index.str.contains('|'.join(['d', 'loc', 'east', 'west']))]
    east = subset[subset.index.str.contains('east')]
    west = subset[subset.index.str.contains('west')]
    hist_stats[idx] = {'width' : width,
                       'depth' : depth,
                       'width and depth': wd,
                       'location' : rall,
                       'east': east,
                       'west': west}


# Function to map keys to regions
def map_to_region(key):
    regions = {0: "west", 1: "central", 2: "east"}
    return regions[key]

# Create empty lists to store statistics and density peaks
regions = []
series_names = []
min_values = []
max_values = []
mean_values = []
median_values = []
variance_values = []
density_peak_values = []
std_dev_values = []

# Loop through the dictionary and calculate statistics for each series
for key, series_data in hist_stats.items():
    region = map_to_region(key)
    for series_name, series in series_data.items():
        regions.append(region)
        series_names.append(series_name.capitalize())
        min_values.append(series.min())
        max_values.append(series.max())
        mean_values.append(series.mean())
        median_values.append(series.median())
        variance_values.append(series.var())
        std_dev_values.append(series.std())

        # Calculate KDE for density estimation
        kde = sns.kdeplot(series)
        density_peak_value = kde.get_lines()[0].get_data()[1].max()
        density_peak_x = kde.get_lines()[0].get_data()[0][kde.get_lines()[0].get_data()[1].argmax()]
        density_peak_values.append(density_peak_x)
        plt.clf()

# Create a DataFrame to store the statistics and density peaks
data = {
    "Region": regions,
    "Series Name": series_names,
    "Min": min_values,
    "Max": max_values,
    "Mean": mean_values,
    "Median": median_values,
    'Standard Deviation': std_dev_values,
    "Variance": variance_values,
    "Density Peak": density_peak_values,
}

statistics_df = pd.DataFrame(data)

# Round the values for better presentation
statistics_df = statistics_df.round(2)

# Convert the DataFrame to LaTeX table format
latex_table = statistics_df.to_latex(index=False)

# Save the LaTeX table to a file
with open('statistics_table.tex', 'w') as f:
    f.write(latex_table)

# Print a message to indicate the successful save
print("LaTeX table saved as 'statistics_table.tex'")

statistics_df.to_csv('table2.csv')

statistics_df = statistics_df.set_index('Series Name')
west_df = statistics_df[statistics_df['Region'] == 'west'].drop(columns=['Region'])
east_df = statistics_df[statistics_df['Region'] == 'east'].drop(columns=['Region'])
central_df = statistics_df[statistics_df['Region'] == 'central'].drop(columns=['Region'])
west_df = west_df.T
east_df = east_df.T
central_df = central_df.T

tables = [west_df, central_df, east_df]
column_order = ['Width', 'Depth', 'Width and Depth', 'Location', 'East', 'West']
for idx, table in enumerate(tables):
    print(table)
    latex_table = table.to_latex(index=True)
    table.to_csv(f'{idx}_stats_table.csv')
    with open(f'{idx}_stats.tex', 'w') as f:
        
        f.write(latex_table)