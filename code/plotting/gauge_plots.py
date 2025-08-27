import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
mpl.use('pgf')

def get_columns(df):
    loc_cols = [col for col in df if 'loc' in col]
    wcols = [col for col in df if 'd' not in col and 'w' in col]
    dcols = [col for col in df if 'w' not in col and 'd' in col]
    wd_cols = [col for col in df if 'dw' in col]

    w = df[wcols].mean(axis=1)
    d = df[dcols].mean(axis=1)
    dw = df[wd_cols].mean(axis=1)
    loc = df[loc_cols].mean(axis=1)
    nob = df['no_breach']
    return w, d, loc, dw, nob

def apply_style():
    style_file = os.path.join(os.path.dirname(__file__), 'mystyle.mplstyle')
    plt.style.use(style_file)
    
    
def create_figure(nrows=3, ncols=1):
    fig = plt.figure(constrained_layout=True, figsize=(10,6))
    subfigs = fig.subfigures(nrows=nrows, ncols=ncols)
    return fig, subfigs
    

def create_axes(subfigs, idx, ncols=2):
    axes = subfigs[idx].subplots(nrows=1, ncols=ncols)
    return axes

def add_row_title(fig, subfigs, i):
    mid = (fig.subplotpars.right + fig.subplotpars.left)/2
    subfigs[i].suptitle(f'{loc.capitalize()}', x=mid)

apply_style()

# Define the figure and subplots
# fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
fig = plt.figure(constrained_layout=True,figsize=(10,6))

# create 3x1 subfigs
subfigs = fig.subfigures(nrows=3, ncols=1)

def gauge_category(width, depth, locs, wd, no_breach):
    categories = {'nob':
                     {'name': 'no breach',
                     'data': no_breach,
                     'color': 'black'},
                    'width':
                      {'name': 'width',
                      'data': width,
                      'color': '#009E73'},
                      'depth':
                      {'name': 'depth',
                       'data': depth,
                       'color': '#56B4E9'},
                      'wd':
                      {'name': 'width/depth',
                       'data': wd,
                       'color': '#D55E00'},
                      'loc': 
                      {'name': 'vary everything',
                       'data': locs,
                       'color': '#CC79A7'},
                      }
    return categories

def read_pickle(PATH, loc, gauge_num):
    df = pd.read_pickle(os.path.join(PATH, f'{loc}_gauge1{gauge_num:04}.pkl.gz'))
    # remove any duplicate columns
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def plot_gauges(df, ax, categories):
    for cat in categories:
        data = categories[cat]
        if data['name'] == 'no_breach':
            linestyle='dashed'
        else:
            linestyle = 'solid'
        ax.plot(df.index, data['data'], label=data['name'], 
                color=data['color'], linestyle=linestyle)
        
        # Add legend and subplot labels
        ax.set_title(f'Gauge {chr(ord("`")+(j+1))}')
        # ax.legend(loc='upper left')
        ax.set_xlim(-2,4)
        if i == 2:
            ax.set_xlabel('Hours from landfall')
        if j == 0:
            ax.set_ylabel('Surge height (m)')
            
    
def legend_create(ax, fig):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', 
               bbox_to_anchor=(0,-0.1,1,0.25),  
               ncol=5, fontsize=12)  
    
    
if __name__ == '__main__':
    fig, subfigs = create_figure()
    PATH = '/home/catherinej/BarrierIslandBreachProject/data/processed/'

    gauges = {'west': [84, 82],
              'central': [45, 133],
              'east': [11, 119]}
    
    # # Loop through each location/gauge number
    for i, (loc, nums) in enumerate(gauges.items()):
        # Create the figures, axes, add title to rows
        axs = create_axes(subfigs, i, ncols=2)
        add_row_title(fig, subfigs, i)
        
        for j, n in enumerate(nums):
            # Load the data for this location/gauge number
            df = read_pickle(PATH, loc, n)
            
            # Get the columns for each category
            w, d, l, wd, nob = get_columns(df)
            
            # Define the categories
            categories = gauge_category(w, d, l, wd, nob)
            
            # Plot each category on the same axis
            ax = axs[j]
            plot_gauges(df, ax, categories)
            
            legend_create(ax, fig)

    plt.show()
    plt.savefig('gauges_west_central_east.pgf', format='pgf', bbox_inches='tight')
    plt.savefig('gauges_west_central_east.png', format='png')

