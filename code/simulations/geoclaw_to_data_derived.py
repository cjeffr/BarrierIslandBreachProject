import os
import xarray as xr
import pandas as pd
import holland_wind_fields as hwf
import read_OWI

def create_storm_arrays(storm, xlow, ylow, xhi, yhi, savename, dx=0.25, dy=0.25):
    mx = int((xhi - xlow) * 4)
    my = int((yhi - ylow) * 4)
    
    aux = np.empty(shape=(2, mx, my))
    wind_index = 0
    pressure_index = 2
    auxs = []
    speed = []
    uu = []
    vv = []
    pressure = []
    
    lon, lat = lon_lat_arrays(mx, my, xlow, ylow, dx, dy)
    time = calc_time_arrays(storm)
    
    for i in range(0, storm.num_casts, 1):
        t = storm.track[i,0]
        aux = hwf.calculate_holland_param(mx, my, xlower, ylower, dx, dy,
                                          t, wind_index, pressure_index, storm)
        uu.append(aux[0,:,:].T)
        vv.append(aux[1,:,:].T)
        pressure.append(aux[2,:,:].T)
        speed.append(np.sqrt(aux[0,:,:]**2 + aux[1,:,:]**2).T)
        
    ds = arrange_dataset(time, lat, lon, uu, vv, speed, pressure, storm, eye_locs)
    ds.to_netcdf(savename)
    
def arrange_dataset(time, lat, lon, uu, vv, speed, pressure, storm):
    xr.Dataset(data_vars={'u': (('time', 'lat', 'lon'), uu),
                           'v': (('time', 'lat', 'lon'), vv),
                           'speed': (('time', 'lat', 'lon'), speed),
                           'pressure':(('time', 'lat', 'lon'), pressure),
                           'eye_loc': (('time', 'loc'), storm.track[:, 1:3]),
                           'radius': (('time'), storm.radius[:])},
                coords={'lat':lat,
                        'lon':lon,
                        'time': time})
    
    
def lon_lat_arrays(mx, my, xlow, ylow, dx, dy):
    lon = [xlow + (i - 0.5) * dx for i in range(mx)]
    lat = [ylow + (i - 0.5) * dy for i in range(my)]
    
    return lon, lat

def calc_time_arrays(storm):
    time = [i - storm.track[0][0] for i in storm.track[:,0]]
    return time

def find_eye(pressure):
    first = None
    eye_locs = []
    for idx, p in enumerate(pressure):
        min_p = np.min(p.values)
        if min_p < 100000:
            if first is None:
                first = idx
            eye_locs.append(read_OWI.eye_from_pressure(p.values))
    return first, eye_locs



# Function to find the first transition index in a given direction from the center
def find_first_pressure_transition_index(pressure_array, target_value, center_point, direction):
    rows, cols = pressure_array.shape
    center_j, center_i = center_point

    if direction == 'north':
        for i in range(center_i, -1, -1):
            if pressure_array[i, center_j] < target_value:
                continue
            if pressure_array[i+1, center_j] >= target_value:
                return (i+2, center_j)
                break
    elif direction == 'south':
        for i in range(center_i + 1, rows):
            if pressure_array[i, center_j] < target_value:
                continue
            if pressure_array[i, center_j] >= target_value:
                return (i-1, center_j)
                break
    elif direction == 'east':
        for j in range(center_j, cols):
            if pressure_array[center_i, j] < target_value:
                continue
            if pressure_array[center_i, j] >= target_value:
                return (center_i, j-1)
                break
    elif direction == 'west':
        for j in range(center_j, -1, -1):
            if pressure_array[center_i, j-1] < target_value:
                continue
            if pressure_array[center_i, j] >= target_value:
                return (center_i, j+1)
                break
    
    return None

 def calc_rmw(data, eye_loc):
    y, x = np.unravel_index(data.speed.argmax(), data.speed.shape)
    lat1 = data.lat[eye_loc[1]]
    lon1 = data.lon[eye_loc[0]]
    lat2 = data.lat[y]
    lon2 = data.lon[x]
    dist = haversine(lat1, lon1, lat2, lon2)
    return dist

for idx in range(len(isaac.time)):
    data = isaac.isel(time=idx)
    eye = read_OWI.eye_from_pressure(data.pressure.values)
    
    
   

for idx in range(len(isaac.time)):
    data = isaac.isel(time=idx)
    if data.time >= 345600:
        eye_loc = read_OWI.eye_from_pressure(data.pressure.values)
        dist = calc_rmw(data, eye_loc)
        print(data.time.values,  dist)