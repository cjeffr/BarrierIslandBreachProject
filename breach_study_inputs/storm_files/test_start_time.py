import xarray as xr
import numpy as np

storm_file = 'test_480.nc'

ds = xr.open_dataset(storm_file)
eye = ds.eye_loc.data
lat_row = np.where(eye[:,1] > 22.0)
lon_row = np.where(eye[:,0] < -58.0)
idx = max(min(lat_row[0]), min(lon_row[0]))
print('idx is: ', idx)
print( int(ds.time.isel(time=idx).values))
