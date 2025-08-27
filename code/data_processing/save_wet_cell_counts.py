import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import glob
import xarray as xr
import sys
import os
sys.path.insert(1, '/home/catherinej/claw_code/src/claw_code/pre')
sys.path.insert(0, '/home/catherinej/claw_code/src/claw_code/post')
import bathymetry_adulterant
import waveforms
import warnings


# Load all masks - land, ocean, island
ocean_mask_file = '/home/catherinej/BarrierBreach/src/mask_ocean.npz'
with np.load(ocean_mask_file) as npz:
    ocean_mask = np.ma.MaskedArray(**npz)
land_mask_file = '/home/catherinej/BarrierBreach/src/land_mask.npz'
with np.load(land_mask_file) as npz:
    land_mask = np.ma.MaskedArray(**npz)
no_breach_mask_file = '/home/catherinej/BarrierBreach/src/no_breach_mask.npz'
with np.load(no_breach_mask_file) as npz:
    no_breach_mask = np.ma.MaskedArray(**npz)


# Load all breach data
bdata_path = '/home/catherinej/BarrierBreach/data/breach_data.pkl.gz'
bdata = pd.read_pickle(bdata_path)
# clean up the data, save into the final formatsim_names = bdata['key'].unique()
sim_names = bdata['key'].unique()
breach_data = {}
for sim in sim_names:
    if 'r' not in sim:
        breach_data[sim] = {'mean_depth': bdata.loc[bdata['key'] == sim]['Depth'].mean(),
                            'mean_distance': bdata.loc[bdata['key'] == sim]['breach_width (m)'].mean()}
breach_stats = pd.DataFrame(breach_data)

#load all fgmax_data
sims = os.listdir('/home/catherinej/BarrierBreach/data/fgmax_data/fg_nc')

#clean and combine data with breach data and save
wet_cells = {}
bad_sets = []
for sim in sims:
    if 'r' not in sim and '.ipynb' not in sim:
        print(sim)
        path = os.path.join('/home/catherinej/BarrierBreach/data/fgmax_data/fg_nc/', sim)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                ds = xr.load_dataset(path)
        except Exception as e:
            bad_sets.append((e, sim))
        mask = np.ma.masked_where(ds.eta == 0, ds.eta)
        # print(sim.rsplit('_',1)[0])
        wet_cells[sim.rsplit('_',1)[0]] = {'wet_cell_count': mask.count(),
                                            'diff_topo': (mask.count() - land_mask.count()),
                                            'diff_no_breach': (mask.count() - no_breach_mask.count()),
                                            'num_breaches': int(sim.rsplit('_',1)[0].rsplit('_',1)[-1])}
print(bad_sets)
wet = pd.DataFrame(wet_cells)
data = pd.concat([wet, breach_stats], axis=0)
data = data.T
data['mean_breach_area'] = abs(data['mean_depth'] * data['mean_distance'])
data['total_breach_area'] = abs(data['mean_depth'] * data['mean_distance'] * data['num_breaches'])
data['inundation_per_breach (m**2)'] = abs(data['diff_no_breach'] / data['num_breaches'])*18**2
data = data.dropna()

width_sims = [row for row in data.index if 'w' in row and not 'd' in row]
depth_sims = [row for row in data.index if 'd' in row and not 'w' in row]
rall_sims = [row for row in data.index if 'loc' in row]
width_depth_sims = [row for row in data.index if 'dw' in row]

                                    
# split data into width and depth
df = data[data.index.isin(width_sims)]
df.to_csv('width_wet_cells.csv')
df = data[data.index.isin(depth_sims)]
df.to_csv('depth_wet_cells.csv')
df = data[data.index.isin(width_depth_sims)]
df.to_csv('width_depth_wet_cells.csv')
df = data[data.index.isin(rall_sims)]
df.to_csv('rall_wet_cells.csv')
# save so it can be plotted easily with different programs
