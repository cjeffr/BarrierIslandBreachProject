import os
import numpy as np
import pandas as pd
import xarray as xr
import datetime


"""
This script is designed to choose a random location from a masked array
test whether it is a viable breach location, randomize the width/depth
choose a time related to max dune height for that location
and write this out to a breach.data file in the following format/example

6
breach_trigger, south, north, west, east, mu, sigma, time_factor, start_time, end_time
0 0 0 0 0 0
40.770121 40.767005 40.765269 40.764085 40.759 40.758
40.77683 40.771566 40.770601 40.77 40.766711 40.765
-72.730156 -72.741864 -72.74672 -72.757493 -72.766651 -72.772045
-72.722699 -72.736314 -72.742147 -72.756182 -72.764132 -72.769959
-72.728378 -72.739811 -72.744832 -72.758208 -72.765685 -72.770963
1.0
0.002 0.002 0.002 0.002 0.002 0.002
68400 68400 68400 68400 68400 68400
72000 72000 72000 72000 72000 72000
"""

def get_random_location():
    with np.load('/home/catherinej/BarrierBreach/data/masked_island.npz') as npz:
        island = np.ma.MaskedArray(**npz)
    island_indices = list(zip(*np.where(island.mask == False)))
    lons = np.unique([lon for lat, lon in island_indices])
    random_lon = np.random.choice(lons)
    lats = [lat for lat, lon in island_indices if lon == random_lon]
    return lats[0], lats[-1], random_lon


def calc_random_distance(lat, lon):
    total_width = np.random.randint(25,300)
    m_per_deg = np.pi/180 * 6371000 * np.cos(lat * np.pi/180)
    dist = total_width / m_per_deg
    east = lon - dist/2
    west = lon + dist/2  
    return east, west



def random_num_breaches():
    num_breaches = np.random.randint(0,10)
    return num_breaches


def get_depth(breach_num):
    depth = np.random.uniform(0,-2.0)
    return depth


def find_nearest_bathy_val(topo_data, breach_lat, breach_lon):
    X, Y = np.meshgrid(topo_data.x, topo_data.y)
    abslat = np.abs(Y- breach_lat)
    abslon = np.abs(X - breach_lon)
    c = np.maximum(abslon, abslat)
    y, x = np.where(c==np.min(c))
    x_deg = topo_data.x[x]
    y_deg = topo_data.y[y]
    return x, y, x_deg, y_deg


def get_bearing():
    import math
    # Upper adn lower points along moriches barrier island
    llon = -72.875
    ulon = -72.59
    llat = 40.73
    ulat = 40.81
    dLon = (ulon - llon)
    x = math.cos(math.radians(ulat)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(llat)) * math.sin(math.radians(ulat)) - \
        math.sin(math.radians(llat)) * math.cos(math.radians(ulat)) * \
        math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)

    return brng



def get_vector_points(lat, lon, bearing, distance):
    import math

    R = 6371 * 1000 #Radius of the Earth
    brng = math.radians(bearing) #Bearing is 90 degrees converted to radians.
    d = distance #Distance in m

    #lat2  52.20444 - the lat result I'm hoping for
    #lon2  0.36056 - the long result I'm hoping for.

    lat1 = math.radians(lat) #Current lat point converted to radians
    lon1 = math.radians(lon) #Current long point converted to radians

    lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
         math.cos(lat1)*math.sin(d/R)*math.cos(brng))

    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
                 math.cos(d/R)-math.sin(lat1)*math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return lat2, lon2


def calc_distance(lat1, lat2, lon1, lon2):
    from math import sin, cos, sqrt, atan2, radians
    R = 6371*1000
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1)*cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

def gauges_locs_semicircle(gauge_x, gauge_y, center_x, 
                           center_y, radius, lon1, lat1):
    v1 = [lon1 - center_x, lat1 - center_y] # vect1 = b - center_pt
    v2 = [gauge_x - center_x, gauge_y - center_y] # p - center_pt
    xp = v1[0]*v2[1] - v1[1]*v2[0]
    dist_to_center = calc_distance(gauge_y, center_y, gauge_x, center_x)
    if xp < 0: # gauge is to right of line (ie ocean)
        if dist_to_center <= radius:
            return dist_to_center

        
        
def check_breach_location_viability():
    
    # get max dune height
    max_dune = max_dune_height(topo_data, ylow, yhigh, x)
    
    # Get bearing for island
    bearing = get_bearing()
    
    # get half circle lat/lon (point b) distance in m
    distance = 1000 # meters
    ulat, ulon = get_vector_points(llat, llon, bearing, distance)
    
    # get all gauges inside the half circle
    gauge_df = pd.read_csv('/home/catherinej/BarrierBreach/src/visualization/ocean_gauges.csv', usecols[1,2])
    gauge_df['dist'] = [gauges_locs_semicircle(x, y, true_breach_loc[1],
                                         true_breach_loc[0], 1000,
                                         lon2, lat2) for x, y in zip(gauges['lon'],
                                                                     gauges['lat'])]
    df = gauge_df[~gauge_df['dist'].isnull()]
    
    # load all gauge_data for that subset of gauges
    gauge_data_df = load_gauge_data(df.index.values)
    
    # get point where gauge_data exceeds x% dune height and index (for time stamp)
    x_percent = max_dune * .30
    
    # is the max gauge x% of dune height?
    cols_greater = (gauge_data_df >= x_percent).any()
    all_breach_times = first_greater(df, n, col):

    
    breach_indx = [first_greater(gauge_data_df, x_percent, col) for col in cols_greater.index]
    first_index = min([x for x in breach_indx if type(x) == np.float64])
    if isinstance(first_index, float):
        
        # yes proceed
        # get random width/depth
        # breach stop time is start + 2hours?
        
    else:
        with open(f'failed_location_{x}/{y}.data', 'rw') as f:
            f.write(f'Breach location failed: {} {}'# no write to file as failed location

                    
                    
def max_dune_height(topo_data, ylow, yhigh, x):
    return topo_data.Z[ylow:yhigh, x].max()

                    
def first_greater(df, n, col):
    m = df[col].gt(n)
    return m.any() and m.idxmax()
                    
                    
                    