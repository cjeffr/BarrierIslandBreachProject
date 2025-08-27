import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from clawpack.geoclaw import topotools
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
from matplotlib.colors import LogNorm


# Data building 
bathy_path = '/home/catherinej/bathymetry/'
STATIONS = '/home/catherinej/NACCS_data/NAC2014_R01_Station_Depths.dat'
MAX_ELEV = '/home/catherinej/NACCS_data/BASE_Table_Maxele_V5.csv'
elev_df = pd.read_csv(MAX_ELEV, sep=',', header=None)
sta_df = pd.read_fwf(STATIONS, sep='\t',header=None)
sta_df = sta_df.drop(0, axis=1)
storms =  {'Moriches': {'storm_nums':[453, 470, 481, 409, 367],
                        'west': -72.89,
                        'east': -72.63,
                        'south': 40.71,
                        'north': 40.83,
                        'bathy': os.path.join(bathy_path, 'm_test.nc')},
           'Chincoteague': {'storm_nums':[131, 92, 123, 86, 169],
                            'west': -75.5,
                            'east': -74.97,
                            'south': 37.73,
                            'north': 38.42,
                            'bathy': os.path.join(bathy_path, 'chinc.nc')},
           'Barnegat':{'storm_nums':[433, 180, 349, 99, 100],
                       'west': -74.43,
                        'east': -74.02,
                        'south': 39.48,
                        'north': 40.07,
                       'bathy': os.path.join(bathy_path, 'barn.nc')}}
region = [-73.0, -72.5, 40.75, 41.00]

# Find the data for each bay adn storm in the different 
# csv files and add to dictionary for easy looping
for key in storms:
    stations = sta_df.loc[(sta_df.iloc[:,0] >= storms[key]['west']) &
                      (sta_df.iloc[:,0] <= storms[key]['east']) &
                      (sta_df.iloc[:,1] >= storms[key]['south']) & 
                      (sta_df.iloc[:,1] <= storms[key]['north'])]
    storms[key]['stations'] = stations
    max_elev_data = elev_df.iloc[:,1:].loc[elev_df.iloc[:,0].isin(stations.index)]
    max_elev_data[max_elev_data > 100] = np.nan
    max_elev_data[max_elev_data < 0] = np.nan
    max_elev_data = max_elev_data.dropna(axis=1, how='all')
    max_elev_data = max_elev_data[storms[key]['storm_nums']]
    storms[key]['max_elev_data'] = max_elev_data

storms['Moriches']['stations']
moriches_gauges = storms['Moriches']['stations'].reset_index(drop=True)
moriches_gauges = moriches_gauges.rename(columns={1:'lon', 2:'lat', 3:'depth'})
moriches_gauges.to_csv('m_gauges.csv')