# Import all the things
import pandas as pd
import numpy as np
import os
import sys
import waveforms
import xarray as xr
import pygmt

#Load a single fgmax
path = '/home/catherinej/BarrierBreach/data/fgmax_data/fg_nc'
fg_files = [os.path.join(path, x) for x in os.listdir(path) if '.ipynb' not in x]

filter_data = xr.open_dataset(fg_files[0])
df = filter_data.eta.to_dataframe()
    
# Filter by bay/ocean
locs = [waveforms.gauges_in_bay(x,y) for x,y in zip(df.index.get_level_values(1),
                                                    df.index.get_level_values(0))]
df['locations'] = locs
bay_idxs = df.index[df['locations'] == 'bay']
bay = df.loc[bay_idxs].reset_index()             
# Divide into west, east, central
west_pt = df.index.get_level_values(1).min()
east_pt = df.index.get_level_values(1).max()
central_pt = (east_pt - west_pt)/3
print(west_pt, east_pt, central_pt)
fig = pygmt.Figure()
fig.basemap(region=[-72.89, -72.64, 40.71, 40.83],
            projection='M3', frame=True)
fig.plot(x=west_pt, y=df.index.get_level_values(0).min(), style='c0.05c', color='red', label='west')
fig.plot(x=west_pt + central_pt, y=df.index.get_level_values(0).min(),style='c0.05c', color='blue', label='central')
fig.plot(x=east_pt - central_pt, y=df.index.get_level_values(0).min(), style='c0.05c', color='green', label='east')
fig.coast(shorelines=True)
# fig.legend()
fig.savefig('test_boundaries.png')

wbay = bay[bay.lon.between(west_pt, west_pt + central_pt)]
cbay = bay[bay.lon.between(west_pt + central_pt, east_pt - central_pt)]
ebay = bay[bay.lon.between(east_pt - central_pt, east_pt)]

#get random locations
n = 20 # number of random points
wsubset = wbay.sample(n)
esubset = ebay.sample(n)
csubset = cbay.sample(n)

#select 3 random locations using these for consistency, can change
widx = [105695, 528355, 56117]
eidx = [1356806, 552806, 1068642]
cidx = [439911, 1147411, 1249304]

# Load all the others, sort, get the individual locations saved
w = []
e = []
c = []
for file in fg_files:
    ds = xr.open_dataset(file)
    df = ds.eta.to_dataframe().rename(
        columns={'eta': file.split('/')[-1].rsplit('_',1)[0]})
    
    bay = df.loc[bay_idxs]
    
    w.append(bay.iloc[widx])
    c.append(bay.iloc[cidx])
    e.append(bay.iloc[eidx])

west = pd.concat(w, axis=1)
east = pd.concat(e, axis=1)
central = pd.concat(c, axis=1)

datapath = '/home/catherinej/BarrierBreach/data/plotdata'
if not os.path.exists(datapath):
    os.mkdir(datapath)
west.T.to_pickle(os.path.join(datapath, 'west_fgmax_points.pkl.gz'), 
               compression='gzip')
east.T.to_pickle(os.path.join(datapath, 'east_fgmax_pts.pkl.gz'), 
               compression='gzip')
central.T.to_pickle(os.path.join(datapath, 'central_fgmax_pts.pkl.gz'), 
                  compression='gzip')
