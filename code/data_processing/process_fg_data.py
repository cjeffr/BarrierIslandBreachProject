import numpy as np
import pandas as pd
import glob
import xarray as xr
import sys
import os
import warnings
import matplotlib.pyplot as plt

# Add custom module path
sys.path.insert(1, '/home/catherinej/temp_cleanup/code/utilities')
import waveforms # import custom waveform module

# Define paths to mask files
MASK_FILES = {
    'ocean': '/home/catherinej/temp_cleanup/data/mask_ocean.npz',
    'land': '/home/catherinej/temp_cleanup/data/land_mask.npz',
    'no_breach': '/home/catherinej/temp_cleanup/data/no_breach_mask.npz',
    'island': '/home/catherinej/temp_cleanup/data/island_only_mask.npz'
}

def load_mask(file_path):
    with np.load(file_path) as npz:
        return np.ma.MaskedArray(**npz)

# Load all masks
ocean_mask = load_mask(MASK_FILES['ocean'])
land_mask = load_mask(MASK_FILES['land'])
no_breach_mask = load_mask(MASK_FILES['no_breach'])
masked_island = load_mask(MASK_FILES['island'])


# Load breach data
bdata_path = '/home/catherinej/BarrierBreach/data/breach_data_updated.pkl.gz'
bdata = pd.read_pickle(bdata_path)

# Process breach data to calculate mean depth and mean distance
sim_names = bdata['key'].unique()
breach_data = {}
for sim in sim_names:
    if 'r' not in sim:
        breach_data[sim] = {
            'mean_depth': bdata.loc[bdata['key'] == sim]['Depth'].mean(),
            'mean_distance': bdata.loc[bdata['key'] == sim]['breach_width (m)'].mean()
        }
breach_stats = pd.DataFrame(breach_data)


# Directory containing fgmax_data
fgmax_data_dir = '/home/catherinej/temp_cleanup/data/raw/fg_nc/'

def process_fgmax_data(sim, masked_island):
    path = os.path.join(fgmax_data_dir, sim)
    try: 
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            ds = xr.load_dataset(path)
    except Exception as e:
        return None, (e, sim)
    
    mask_wet_cells = np.ma.masked_where(ds.eta == 0, ds.eta)
    mask_breach = np.ma.masked_where(ds.eta > 0, ds.eta)
    wet_cell_count = mask_wet_cells.count()
    diff_topo = wet_cell_count - land_mask.count()
    diff_no_breach = wet_cell_count - no_breach_mask.count()
    num_breaches = int(sim.rsplit('_', 1)[0].rsplit('_', 1)[-1])

    return {
        'wet_cell_count': wet_cell_count,
        'diff_topo': diff_topo,
        'diff_no_breach': diff_no_breach,
        'num_breaches': num_breaches
    }, None
    
# Process all fgmax data
wet_cells = {}
bad_sets = []
sims = os.listdir(fgmax_data_dir)
for sim in sims:
    if 'r' not in sim and '.ipynb' not in sim:
        result, error = process_fgmax_data(sim, masked_island)
        if error:
            bad_sets.append(error)
        else:
            wet_cells[sim.rsplit('_', 1)[0]] = result
        
wet = pd.DataFrame(wet_cells)
data = pd.concat([wet, breach_stats], axis=0)
data = data.T
data['mean_breach_area'] = abs(data['mean_depth'] * data['mean_distance'])
data['total_breach_area'] = abs(data['mean_depth'] * data['mean_distance'] * data['num_breaches'])
data['inundation_per_breach (m**2)'] = abs(data['diff_no_breach'] / data['num_breaches'])*18**2
data = data.dropna()


# Filter and save data by simulation types
sim_types = {
    'width_sims': [row for row in data.index if 'w' in row and not 'd' in row and not 'west' in row],
    'depth_sims': [row for row in data.index if 'd' in row and not 'w' in row],
    'rall_sims': [row for row in data.index if 'loc' in row],
    'width_depth_sims': [row for row in data.index if 'dw' in row],
    'east_sims': [row for row in data.index if 'east' in row],
    'west_sims': [row for row in data.index if 'west' in row]
}

for sim_type, sim_list in sim_types.items():
    df = data[data.index.isin(sim_list)]
    df.to_csv(f'{sim_type}_wet_cells.csv')

