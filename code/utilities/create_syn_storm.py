import os
import numpy as np
import xarray as xr
from dataclasses import dataclass
import holland_wind_fields as hwf
import pygmt

# Path to storm data file
storm_file = '/home/catherinej/geoclaw.old_020623/examples/storm-surge/isaac/isaac.storm'

# Load storm data from the specified file
storm = hwf.load_storm_data(storm_file)

# Define boundaries of basin for grid
xlower = -99.0
xupper = -70.
ylower = 8.00
yupper = 32.0

# Define grid sizes
mx = (xupper - xlower) * 4
my = (yupper - ylower) * 4
dx = 0.25
dy = 0.25

uu, vv, pressure, speed = [], [], [], []

print(storm.num_casts)
# Generate coordinate arrays for lat and lon
lon = [xlower + (i - 0.5) * dx for i in range(int(mx))]
lat = [ylower + (i - 0.5) * dy for i in range(int(my))]
time = [i - storm.track[0][0] for i in storm.track[:,0]]

for i in range(storm.num_casts): #, 1):
    t = storm.track[i, 0]
    # Calculate Holland parameters for the current time
    aux = hwf.calculate_holland_param(mx, my, xlower, ylower, dx, dy, t,
                            wind_index=0, pressure_index=2, storm)
    # Append computed data to the lists
    uu.append(aux[0,:,:].T) # u component of wind
    vv.append(aux[1,:,:].T) # v component of wind
    pressure.append(aux[2,:,:].T) # pressure data
    speed.append(np.sqrt(aux[0,:,:]**2 + aux[1,:,:]**2).T) # wind speed
    
ds = xr.Dataset(data_vars={
    'u': (('time', 'lat', 'lon'), uu),
    'v': (('time', 'lat', 'lon'), vv),
    'speed': (('time', 'lat', 'lon'), speed),
    'pressure':(('time', 'lat', 'lon'), pressure),
    'eye_loc': (('time', 'loc'), storm.track[:, 1:3]),
    'radius': (('time'), storm.radius[:])
    },
    coords={'lat':lat,
            'lon':lon,
            'time': time})

# Save the dataset to a netcdf file
ds.to_netcdf('isaac_syn_with_radii.nc')
