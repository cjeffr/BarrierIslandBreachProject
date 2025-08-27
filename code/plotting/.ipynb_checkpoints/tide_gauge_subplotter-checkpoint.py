import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import datetime

plt.style.use('./mystyle.mplstyle')

def get_columns(df):
    loc_cols = [col for col in df if 'loc' in col]
    wcols = [col for col in df if 'd' not in col and 'w' in col and not 'west' in col]
    dcols = [col for col in df if 'w' not in col and 'd' in col]
    wd_cols = [col for col in df if 'dw' in col]
    east_cols = [col for col in df if 'east' in col]
    west_cols = [col for col in df if 'west' in col]

    w = df[wcols]
    d = df[dcols]
    dw = df[wd_cols]
    loc = df[loc_cols]
    west = df[west_cols]
    east = df[east_cols]
    nob = df['no_breach']
    return w, d, loc, dw, nob, east, west

def lock_ticks(ax):
    # X in HOURS from landfall  
    ax.set_xlim(-2, 4)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))    # every 1 hr
    # ax.xaxis.set_minor_locator(MultipleLocator(0.25))   # every 15 min
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    # Y in meters (adjust if you like)
    ax.set_ylim(-0.5, 2.0)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.minorticks_on()
    ax.tick_params(which="both", direction="out")
    
def calc_perc(df, min_per=0.05, max_per=0.95):
    # Calculate the mean, 5th, and 95th percentiles
    mean_values = df.mean(axis=1).values
    med_values = df.median(axis=1).values
    percentile_5 = df.quantile(min_per, axis=1).values
    percentile_95 = df.quantile(max_per, axis=1).values
    return mean_values, med_values, percentile_5, percentile_95

def gauge_data_plotting(data):
    # Define the grid size
    num_rows = 3
    num_cols = 2

    # Create a figure and axes for the subplots
    fig = plt.figure(constrained_layout=True, figsize=(7, 7))
    subfigs = fig.subfigures(nrows=num_rows, ncols=1)
    letters = ['a', 'b', 'c', 'd', 'e', 'f']
    for i, (loc, loc_data) in enumerate(data.items()):
        axs = subfigs[i].subplots(nrows=1, ncols=2)
        mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
        subfigs[i].suptitle(f'{loc.capitalize()}', x=mid)

        # Iterate over the gauge numbers and plot each category's data
        for j, (n, gauge_data) in enumerate(loc_data.items()):
            ax = axs[j]
            for category, category_data in gauge_data['data'].items():
                if category_data['name'] == 'No breach':
                    ax.plot(category_data['data'].index, category_data['data'],
                            label=category_data['name'], color=category_data['color'], linestyle='--')
                else:
                    mean, medi, minp, maxp = calc_perc(category_data['data'])
                    print(mean, medi, minp, maxp)
                    ax.plot(category_data['data'].index, mean, color=category_data['color'], label=category_data['name'])
                    ax.plot(category_data['data'].index, medi, color=category_data['color'], linestyle='--')
                    ax.fill_between(category_data['data'].index, minp, maxp, color=category_data['color'], alpha=0.3,
                                    rasterized=True)
            ax.set_title(f'Gauge {letters[i * 2 + j]}')
            ax.set_xlim(-2, 4)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.get_major_formatter()._usetex = False
            ax.yaxis.get_major_formatter()._usetex = False
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # ax.tick_params(axis='both', which='major', labelsize=12, font='Times New Roman')

            if i == 2:
                ax.set_xlabel('Hours from landfall')
            if j == 0:
                ax.set_ylabel('Surge height (m)')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0, -0.1, 1, 0.25), ncol=5, fontsize=12,frameon=False)
    plt.savefig('/home/catherinej/fig4.pdf', bbox_inches='tight')
    plt.show()

def simple_gauge_plotting():
    sandy_hook = pd.read_csv('CO-OPS_8531680_met.csv',
                             parse_dates=[['Date', 'Time (GMT)']])
    sandy_hook['storm'] = sandy_hook['Verified (m)'] - sandy_hook['Predicted (m)']
    sandy_hook['Date_Time (GMT)'] = pd.to_datetime(sandy_hook['Date_Time (GMT)'])
    sandy_hook['Seconds'] = ((sandy_hook['Date_Time (GMT)'] - datetime.datetime(1938, 9, 21, 19, 20,
                                                                                00)).dt.total_seconds()) / 3600.
    sandy_hook[sandy_hook['Seconds'] >= -0.1]

    sim_sandy = pd.read_csv('gauge85310.txt', skiprows=4, header=None,
                               delim_whitespace=True, usecols=[1,5], names=['Time', 'Eta'])
    sim_sandy['Time'] -= 216000
    sim_sandy['Time'] /= 3600
    fig, ax = plt.subplots(constrained_layout=True, figsize=(7.0, 4.66))

    ax.plot(sandy_hook['Seconds'], sandy_hook['storm'],
            label='Sandy Hook 1938', color='black')
    ax.plot(sandy_hook['Seconds'], sandy_hook['storm'] + 0.239,
            label='Sandy Hook Adjusted MSL (m)', color='black', linestyle='--')
    smooth = sim_sandy.Eta.rolling(2000).mean()
    ax.plot(sim_sandy.Time, smooth-0.073, label='Simulation Adjusted to MSL (m)', color='#D55E00')

    ax.set_xlim(-10, 8)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # ax.tick_params(axis='both', which='major', labelsize=12, font='Times New Roman')
    mt = sandy_hook.loc[sandy_hook['storm'].idxmax(), 'Seconds']
    print('Max Surge height', sandy_hook['storm'].max()+0.239, 'max time', mt )
    print('max sim height', smooth.max()-0.073, 'max time', sim_sandy.loc[smooth.idxmax(), 'Time'])
    print(sim_sandy.Time)
    # ax.axvline(x=0.3378888888888873, color='red')
    # ax.axvline(x=1.66666667, color='green')
    ax.set_xlabel('Hours from landfall')

    ax.set_ylabel('Surge height (m)')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(-0.18, 0.12, 1, 0.25), ncol=1, fontsize=8, frameon=False) #(0, -0.35, 1, 0.25)
    plt.savefig('sandy_hook.pdf', bbox_inches='tight')
    plt.show()

PATH = '/home/catherinej/BarrierIslandBreachProject/data/processed/'

gauges = {'west': [84, 82],
          'central': [45, 133],
          'east': [11, 119]}


data = {}
for loc, nums in gauges.items():
    loc_data = {}
    for n in nums:
        df = pd.read_pickle(os.path.join(PATH, f'{loc}_gauge1{n:04}.pkl.gz'))
        df = df.loc[:, ~df.columns.duplicated()].copy()
        w, d, l, wd, nob, east, west = get_columns(df)

        categories = {
            'nob': {'name': 'No breach', 'data': nob, 'color': 'black'},
            'width': {'name': 'Width', 'data': w, 'color': '#009E73'},
            'depth': {'name': 'Depth', 'data': d, 'color': '#56B4E9'},
            'wd': {'name': 'Width and Depth', 'data': wd, 'color': '#D55E00'},
            'loc': {'name': 'Locations', 'data': l, 'color': '#CC79A7'},
            'east': {'name': 'East', 'data': east, 'color': '#E69F00'},
            'west': {'name': 'West', 'data': west, 'color': 'grey'},

        }

        loc_data[n] = {'data': categories}

    data[loc] = loc_data
gauge_data_plotting(data)
  
# Load the dataset
ds = xr.open_dataset("gauges_ensemble.nc", decode_times=False)
import numpy as np
import xarray as xr

ds = xr.open_dataset("gauges_ensemble.nc", decode_times=False)  # or True if you fixed CF time

# 1) Verify current gauge order
print("gauge coord:", ds["gauge"].values)  # e.g., ['g001' 'g002' 'g003' 'g004' 'g005' 'g006']

# 2) Attach your numeric labels IN THAT SAME ORDER:
gauge_labels_in_file_order = np.array([84, 82, 45, 133, 11, 119], dtype="int32")
ds = ds.assign_coords(gauge_label=("gauge", gauge_labels_in_file_order))

# (optional) also attach regions so you can ds.sel(gauge_region="west")
region_map = {84:"west", 82:"west", 45:"central", 133:"central", 11:"east", 119:"east"}
ds = ds.assign_coords(gauge_region=("gauge", [region_map[i] for i in gauge_labels_in_file_order]))

# 3) Register gauge_label for label-based selection (do this ONCE, not inside the loop)
ds = ds.set_xindex("gauge_label")        # or: ds = ds.swap_dims({"gauge": "gauge_label"})
sim_names = ds["simulation_name"].values.astype(str)
idx = np.where(sim_names == "no_breach")[0]
if idx.size == 0:
    raise ValueError("'no_breach' not found in simulation_name")
nob_idx = int(idx[0])

# Find the index of the 'no_breach' simulation (by name)
sim_names = ds["simulation_name"].values
nob_idx = int(np.where(sim_names == "no_breach")[0][0])

gauges = {'west': [84, 82],
          'central': [45, 133],
          'east': [11, 119]}

# match your colors & display names
cat_style = {
    "w":  {"name": "Width",            "color": "#009E73"},
    "d":  {"name": "Depth",            "color": "#56B4E9"},
    "dw": {"name": "Width and Depth",  "color": "#D55E00"},
    "loc":{"name": "Locations",        "color": "#CC79A7"},
    "no_breach": {"name": "No breach", "color": "black"},
}

plt.style.use('BarrierIslandBreachProject/notebooks/mystyle.mplstyle')
fig = plt.figure(constrained_layout=True, figsize=(7, 7))
subfigs = fig.subfigures(nrows=3, ncols=1)
letters = ['a','b','c','d','e','f']

# If your time index is seconds in the file but you want hours on x-axis:
t = ds["time"].values
# t_hours = t / 3600.0  # uncomment if needed; otherwise use t directly

for i, (region, nums) in enumerate(gauges.items()):
    axs = subfigs[i].subplots(nrows=1, ncols=2)
    mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
    subfigs[i].suptitle(f"{region.capitalize()}", x=mid)

    for j, n in enumerate(nums):
        ax = axs[j]
        
        # --- No breach (single simulation series)
       
        nob = ds["water_level_raw"].sel(gauge_label=n, simulation=nob_idx)
        # nob = ds["water_level_raw"].sel(gauge_label=n, simulation=nob_idx)
        ax.plot(t, nob, label=cat_style["no_breach"]["name"],
                color=cat_style["no_breach"]["color"], linestyle="--")

        # --- Category bands (mean + 5–95%)
        for cat in ["w","d","dw","loc"]:
            if "water_level_mean_bycat" not in ds:
                # fall back to all-simulation stats if by-category not stored
                mu = ds["water_level_mean"].sel(gauge_label=n)
                p05 = ds["water_level_p05"].sel(gauge_label=n)
                p95 = ds["water_level_p95"].sel(gauge_label=n)
            else:
                mu  = ds["water_level_mean_bycat"].sel(gauge_label=n, category=cat)
                p05 = ds["water_level_p05_bycat"].sel(gauge_label=n, category=cat)
                p95 = ds["water_level_p95_bycat"].sel(gauge_label=n, category=cat)
            # make sure you're plotting hours, not seconds or datetimes
            t_hours = (ds["time_rel_hours"].values
                       if "time_rel_hours" in ds.coords
                       else ds["time"].values / 3600.0)



            lock_ticks(ax)
            ax.plot(t, mu, color=cat_style[cat]["color"], label=cat_style[cat]["name"])
            ax.fill_between(t, p05, p95, color=cat_style[cat]["color"], alpha=0.3)

        ax.set_title(f"Gauge {letters[i*2 + j]}")
        ax.set_xlim(-2, 4)      # adjust if using seconds vs hours
        ax.set_ylim(-0.5, 2.0)
        if i == 2:
            ax.set_xlabel("Hours from landfall")
        if j == 0:
            ax.set_ylabel("Surge height (m)")


handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0,-0.1,1,0.25), ncol=5, fontsize=12)
plt.show()

