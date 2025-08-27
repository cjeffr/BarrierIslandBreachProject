import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

plt.style.use('./mystyle.mplstyle')
# Configure matplotlib to use pgf for latex integration
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def hist_nine(data_dicts, bins, ax):
    from matplotlib.ticker import FormatStrFormatter, LinearLocator
    for data_dict in data_dicts:
        ax.hist(data_dict['data'], bins, weights=data_dict['weight'], color=data_dict['color'],
                histtype='stepfilled', alpha=0.2, label=data_dict['name'])
        ax.hist(data_dict['data'], bins, weights=data_dict['weight'], color=data_dict['color'],
                histtype='step', alpha=0.7)
    ax.xaxis.set_major_locator(LinearLocator(6))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

def fg_pt_histograms(subsets, fig_file):
    
    names = ['west', 'central', 'east']

    # Initialize the figure and axes for plotting
    plt.figure(figsize=(8,2.33))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    # interior axes
    fig, axes = plt.subplots(1,3, figsize=(10,3))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    for idx, subset in enumerate(subsets):
        ax = axes[idx]
        ax.set_title(names[idx].capitalize())

        # Extract different categories of data
        wd = subset[subset.index.str.contains('dw')]
        rall = subset[subset.index.str.contains('loc')]
        depth = subset[~subset.index.str.contains('|'.join(['w', 'loc']))]
        width = subset[~subset.index.str.contains('|'.join(['d', 'loc']))]
        east = subset[subset.index.str.contains('east')]
        west = subset[subset.index.str.contains('west')]

        # Calculate weights for histograms
        wweights = np.ones_like(width)/len(width)
        dweights = np.ones_like(depth)/len(depth)
        rweights = np.ones_like(rall)/len(rall)
        wd_weights = np.ones_like(wd)/len(wd)
        west_weights = np.ones_like(west)/len(west)
        east_weights = np.ones_like(east)/len(east)

        # determine histogram bin sizes
        bins=np.histogram(np.hstack((wd, width, depth, rall, 
                                     west_weights, east_weights)), bins=20)[1]

        data_dicts = [
            {'data': wd, 'weight': wd_weights, 'color': '#D55E00', 'name': 'Width and Depth'},
            {'data': width, 'weight': wweights, 'color': '#009E73', 'name': 'Width'},
            {'data': depth, 'weight': dweights, 'color': '#56B4E9', 'name': 'Depth'},
            {'data': rall, 'weight': rweights, 'color': '#CC79A7', 'name': 'Locations'},
        {'data': west, 'weight': west_weights, 'color': 'grey' , 'name': 'West'},
        {'data': east, 'weight': east_weights, 'color': '#E69F00' , 'name': 'East'}]

        hist_nine(data_dicts, bins, ax)
        plt.setp(ax, ylim=(0.0, 0.80), xlim=(0.0, 4.0))


    # Add Legend and labels
    plt.legend(loc=(.75,.75))
    plt.title = names[idx]
    fig.text(0.5, 0.0001, 'Storm Surge (m)', ha='center', va='center', fontsize=16)
    fig.text(0.06, 0.5, 'Density', ha='center', va='center', rotation='vertical', fontsize=16)
    fig.subplots_adjust(bottom=0.15)
    plt.show()
    plt.savefig(f'{os.path.join("/home/catherinej/temp_cleanup/figures", fig_file + ".pdf")}', bbox_inches='tight')
    
def fg_region_histograms(subsets, fig_file):
    
    names = ['west', 'central', 'east']

    # Initialize the figure and axes for plotting
    plt.figure(figsize=(8,2.33))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    # interior axes
    fig, axes = plt.subplots(1,3, figsize=(10,3))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    for idx, subset in enumerate(subsets):
        ax = axes[idx]
        ax.set_title(names[idx].capitalize())

        # Extract different categories of data
        wd = subset.T[subset.T.index.str.contains('dw')]
        wd = wd.values[~np.isnan(wd.values)]
        rall = subset.T[subset.T.index.str.contains('loc')]
        rall = rall.values[~np.isnan(rall.values)]
        depth = subset.T[~subset.T.index.str.contains('|'.join(['w', 'loc', 'east']))]
        depth = depth.values[~np.isnan(depth.values)]
        width = subset.T[~subset.T.index.str.contains('|'.join(['d', 'loc', 'west', 'east']))]
        width = width.values[~np.isnan(width.values)]
        east = subset.T[subset.T.index.str.contains('east')]
        east = east.values[~np.isnan(east.values)]
        west = subset.T[subset.T.index.str.contains('west')]
        west = west.values[~np.isnan(west.values)]
        
        # Calculate weights for histograms
        wweights = np.ones_like(width) / len(width)
        dweights = np.ones_like(depth) / len(depth)
        rweights = np.ones_like(rall) / len(rall)
        wd_weights = np.ones_like(wd) / len(wd)
        eweights = np.ones_like(east)/len(east)
        west_weights = np.ones_like(west)/len(west)
        bins = np.histogram(np.hstack((wd, width, depth, rall, east, west)), bins=20)[1]

        data_dicts = [
            {'data': wd, 'weight': wd_weights, 'color': '#D55E00', 'name': 'Width and Depth'},
            {'data': width, 'weight': wweights, 'color': '#009E73', 'name': 'Width'},
            {'data': depth, 'weight': dweights, 'color': '#56B4E9', 'name': 'Depth'},
            {'data': rall, 'weight': rweights, 'color': '#CC79A7', 'name': 'Locations'},
            {'data': east, 'weight': eweights, 'color': '#E69F00', 'name': 'East'},
            {'data': west, 'weight': west_weights, 'color': 'grey', 'name': 'West'}]

        hist_nine(data_dicts, bins, ax)
        ax.spines[['right', 'top']].set_visible(False)
        plt.setp(ax, ylim=(0.0, 0.80), xlim=(0.55, 4.0))


    # Add Legend and labels
    plt.legend(loc=(.75,.55))
    plt.title = names[idx]
    fig.text(0.5, 0.0001, 'Storm Surge (m)', ha='center', va='center', fontsize=16)
    fig.text(0.06, 0.5, 'Density', ha='center', va='center', rotation='vertical', fontsize=16)
    fig.subplots_adjust(bottom=0.15)
    plt.show()
    plt.savefig(f'{os.path.join("/home/catherinej/temp_cleanup/figures", fig_file + ".pdf")}', format='pdf', bbox_inches='tight')
    

# For point histograms
west = pd.read_pickle('/home/catherinej/temp_cleanup/data/processed/west_fgmax_points.pkl.gz')
central = pd.read_pickle('/home/catherinej/temp_cleanup/data/processed/central_fgmax_pts.pkl.gz')
east = pd.read_pickle('/home/catherinej/temp_cleanup/data/processed/east_fgmax_pts.pkl.gz')

# Prepare subsets and names for plotting
subsets = [ west[west.columns[1]], central[central.columns[1]], east[east.columns[1]]]
fig_file = 'fig3_v2'
fg_pt_histograms(subsets, fig_file)

# For region fgmax histograms
west = pd.read_pickle('/home/catherinej/temp_cleanup/data/processed/west_fg_max.pkl.gz')
central = pd.read_pickle('/home/catherinej/temp_cleanup/data/processed/central_fg_max.pkl.gz')
east = pd.read_pickle('/home/catherinej/temp_cleanup/data/processed/east_fg_max.pkl.gz')

def fg_max_prep(west, central, east):
    w = west.drop(['lat', 'lon'], axis=1)
    c = central.drop(['lat', 'lon'], axis=1)
    e = east.drop(['lat', 'lon'], axis=1)

    return w, c, e

w, c, e = fg_max_prep(west, central, east)
subsets = [w, c, e]
fg_region_histograms(subsets, 'fig4_v2')


