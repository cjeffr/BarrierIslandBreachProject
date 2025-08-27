import numpy as np
import pandas as pd
import os
import breach_randomization as br
from clawpack.geoclaw import topotools

def set_breach_location():
    # Set breach locations as indicated by 1938 storm
    breach_center = [-72.728378, -72.739811, -72.744832,
                         -72.758208, -72.765685, -72.770963]
    breach_south = [40.770121, 40.767005, 40.765269, 
                        40.764085, 40.759, 40.758]
    breach_north = [40.77683, 40.771566, 40.770601,
                        40.77, 40.766711, 40.765]
    return breach_center, breach_south, breach_north
    

    
def load_topo_data():
    topo_file = os.path.join('/home/catherinej/bathymetry', 'moriches.nc') # /projects/weiszr_lab/catherine/bathymetry/
    topo = topotools.read_netcdf(topo_file)
    crop_region = [-72.88560186665958, -72.63430557035839,
                   40.71837964443896, 40.828287051848505]
    topo_data = topo.crop(crop_region)
    return topo_data


def get_dune_heights(topo_data, south, north, lon):
    lon_idx, south_idx, _, _ = br.find_nearest_bathy_val(topo_data, south, lon)
    _, north_idx, _, _ = br.find_nearest_bathy_val(topo_data, north, lon)
    dune_height = br.max_dune_height(topo_data, south_idx, north_idx, lon_idx)
    return dune_height


def load_gauges(lon, lat):
    distance = 1000 # meters
    gauge_list = br.find_nearest_gauges(lon, lat, distance)
    return gauge_list


def load_gauge_data(gauge_list, dune_height):
    df = br.load_gauge_data(gauge_list.index.values)
    x_percent = dune_height * .24
    cols_greater = (df >= x_percent).any()
    all_breach_times = [br.first_greater(df, x_percent, col) for col in cols_greater.index]
    breach_idx = min([x for x in all_breach_times if type(x) == np.float64])
    return breach_idx


def randomize_width(lat, center):
    depth = -2.0
    e, w = br.random_breach_width(lat, center)
    print('returning east, west', e, w)
    return e, w, depth


def randomize_depth(index):
    east = [-72.722699, -72.736314, -72.742147, 
            -72.756182, -72.764132, -72.769959]
    west = [-72.730156, -72.741864, -72.74672,
            -72.757493, -72.766651, -72.772045]
    depth = br.get_depth(index)
    
    return east[index], west[index], depth



def write_breach_data(breach_data, num_breaches, write_path):
    comment_str = 'breach_trigger, south, north, west, east, mu, sigma, time_factor, start_time, end_time'
    write_order = ['breach_trigger', 'south', 'north', 'west', 'east', 
                   'mu', 'sigma', 'time_factor', 'start_time',
                   'end_time', 'depth']
    with open(os.path.join(write_path, 'breach.data'), 'w') as f:
        f.write(f'{num_breaches}' + '\n')
        f.write(comment_str + '\n')
        for key in write_order:
            if key == 'sigma':
                f.write(f'{breach_data[key]}' + '\n')
            else:
                f.write(' '.join(map(str, breach_data[key])) + '\n')
              
              

def runit(write_path, vary_width=False, vary_depth=False):
    num_breaches = 6
    depth = []
    east = []
    west = []
    sigma = 1.0
    time_ratio = 0.002
    dune_heights = []
    breach_gauge_list = []
    breach_time = []
    
    breach_data = {}
    
    mu, south, north = set_breach_location()
    topo_data = load_topo_data()
    
    for i in range(num_breaches):
        lat = (south[i] + north[i]) / 2 
        dune_height = (get_dune_heights(topo_data, south[i], north[i], mu[i]))
        gauge_list = load_gauges(mu[i], lat)
        breach_time.append(load_gauge_data(gauge_list, dune_height))
        if vary_width:
            e, w, d = randomize_width(lat, mu[i])
            east.append(e)
            west.append(w)
            depth.append(d)
        if vary_depth:
            e, w, d = randomize_depth(i)
            east.append(e)
            west.append(w)
            depth.append(d)
    
    breach_trigger = np.ones_like(breach_time, dtype='int')
    time_ratio = np.ones_like(breach_time) * .002
    breach_end = [x + 7200.0 for x in breach_time]
    print(vary_width, vary_depth)
    breach_data['east'] = east
    breach_data['west'] = west
    breach_data['mu'] = mu
    breach_data['breach_trigger'] = breach_trigger
    breach_data['south'] = south
    breach_data['north'] = north
    breach_data['sigma'] = sigma
    breach_data['start_time'] = breach_time
    breach_data['end_time'] = breach_end
    breach_data['depth'] = depth
    breach_data['time_factor'] = time_ratio
    write_breach_data(breach_data, num_breaches, write_path)
    
if __name__ == '__main__':
    runit(write_path='./')
    
        
                            
        
        
        
    