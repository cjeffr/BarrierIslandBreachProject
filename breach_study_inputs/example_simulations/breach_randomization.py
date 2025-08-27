import numpy as np
import pandas as pd
import os

def get_random_location():
    with np.load('/home/catherinej/BarrierBreach/data/masked_island.npz') as npz:
        island = np.ma.MaskedArray(**npz)
    island_indices = list(zip(*np.where(island.mask == False)))
    lons = np.unique([lon for lat, lon in island_indices])
    random_lon = np.random.choice(lons)
    lats = [lat for lat, lon in island_indices if lon == random_lon]
    return lats[0], lats[-1], random_lon

def random_breach_width(lat, lon):
    total_width = np.random.randint(25,630)
    m_per_deg = np.pi/180 * 6371000 * np.cos(lat * np.pi/180)
    dist = total_width / m_per_deg
    east = lon + dist/2
    west = lon - dist/2  
    return east, west


def random_num_breaches():
    num_breaches = np.random.randint(0,10)
    return num_breaches


def get_depth(breach_num):
    depth = np.random.uniform(0,-2.02)
    return depth


def find_nearest_bathy_val(topo_data, breach_lat, breach_lon):
    X, Y = np.meshgrid(topo_data.x, topo_data.y)
    abslat = np.abs(Y- breach_lat)
    abslon = np.abs(X - breach_lon)
    c = np.maximum(abslon, abslat)
    y, x = np.where(c==np.min(c))
    x_deg = topo_data.x[x]
    y_deg = topo_data.y[y]
    return x[0], y[0], x_deg, y_deg


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

    R = 6378.1 * 1000 #Radius of the Earth (m)
    brng = math.radians(bearing) #Bearing is 90 degrees converted to radians.
    d = distance #Distance in km

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


def gauges_locs_semicircle(gauge_x, gauge_y, center_x, 
                           center_y, radius, lon1, lat1):
    v1 = [lon1 - center_x, lat1 - center_y] # vect1 = b - center_pt
    v2 = [gauge_x - center_x, gauge_y - center_y] # p - center_pt
    xp = v1[0]*v2[1] - v1[1]*v2[0]
    dist_to_center = calc_distance(gauge_y, center_y, gauge_x, center_x)
    if xp < 0: # gauge is to right of line (ie ocean)
        if dist_to_center <= radius:
            return dist_to_center


        
def calc_distance(lat1, lat2, lon1, lon2):
    from math import sin, cos, sqrt, atan2, radians
    R = 6373.0 * 1000 # convert km to m
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


def max_dune_height(topo_data, ylow, yhigh, x):
    return topo_data.Z[ylow:yhigh, x].max()


def find_nearest_gauges(breach_lon, breach_lat, distance):
    df = pd.read_csv('/projects/weiszr_lab/catherine/ocean_gauges.csv')
    df = df.drop(['Unnamed: 0', 'dist'], axis=1)
    bearing = get_bearing()
    lat2, lon2 = get_vector_points(breach_lat, breach_lon, bearing, distance)
    df['dist'] = [gauges_locs_semicircle(x, y, breach_lon, breach_lat, distance,
                                        lon2, lat2) for x, y in zip(df['lon'],
                                                                    df['lat'])]
    df = df[~df['dist'].isnull()]
    return df


def load_gauge_data(gauge_names):
    import os
    gauge_data_list = []
    for gauge in gauge_names:
        if gauge > 40:
            cols = ['Time', f'{gauge}_eta']
            gauge_path = os.path.join('../reference_gauges', f'gauge5{gauge:04}.txt')
            df = pd.read_csv(gauge_path, skiprows=3, header=None, delim_whitespace=True,
                             usecols=[1,5], index_col='Time', names=cols)
            gauge_data_list.append(df)
    gauge_data_df = pd.concat(gauge_data_list, axis=1)
    return gauge_data_df


def first_greater(df, n, col):
    m = df[col].gt(n)
    return m.any() and m.idxmax()
