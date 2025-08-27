import xarray as xr
import pandas as pd
import os
import sys
sys.path.insert(0, '/home/catherinej/temp_cleanup/code/utilities/')
import waveforms


def bay_division(fg):
    data = xr.open_dataset(fg)
    df = data.eta.to_dataframe()
    
    locs = [waveforms.gauges_in_bay(x,y) for x,y in
            zip(df.index.get_level_values(1),
                df.index.get_level_values(0))]
    
    df['locations'] = locs
    
    bay_idx = df.index[df['locations'] == 'bay']
    bay_locs = df.loc[bay_idx].reset_index()
    
    west_pt = df.index.get_level_values(1).min()
    east_pt = df.index.get_level_values(1).max()
    central_pt = (east_pt - west_pt / 3)
    
    return west_pt, east_pt, central_pt, bay_locs, bay_idx


def save_selected_surge_locs(fg_files, bay_indx, datapath='/home/catherinej/temp_cleanup/data/processed/'):
    # select 3 random locations using these for consistency, can change
    widx = [105695, 528355, 56117]
    eidx = [1356806, 552806, 1068642]
    cidx = [439911, 1147411, 1249304]

    # Load all the others, sort, get the individual locations saved
    west, central, east = [], [], []

    for file in fg_files:
        ds = xr.open_dataset(file)
        df = ds.eta.to_dataframe().rename(
            columns={'eta': file.split('/')[-1].rsplit('_', 1)[0]})

        bay = df.loc[bay_indx]

        west.append(bay.iloc[widx])
        central.append(bay.iloc[cidx])
        east.append(bay.iloc[eidx])

    if not os.path.exists(datapath):
        os.mkdir(datapath)
    
    pd.concat(west, axis=1).T.to_pickle(os.path.join(datapath,
                                                     'west_fgmax_points_v2.pkl.gz'),
                                        compression='gzip')
    pd.concat(east, axis=1).T.to_pickle(os.path.join(datapath,
                                                     'east_fgmax_pts_v2.pkl.gz'),
                     
                                        compression='gzip')
    pd.concat(central, axis=1).T.to_pickle(os.path.join(datapath, 
                                                        'central_fgmax_pts_v2.pkl.gz'),
                        
                                           compression='gzip')
    
    

# def bay_section_max_surge(fg_files, bay_indx, west_pt, east_pt, central_pt, datapath='/home/catherinej/temp_cleanup/data/processed'):
#     w_max, c_max, e_max = [], [], []
    
#     # precompute teh central boundaries
#     west_central = west_pt + central_pt
#     east_central = east_pt - central_pt
    
#     for file in fg_files:
#         ds = xr.open_dataset(file)
#         df = ds.eta.to_dataframe().rename(
#             columns={'eta': file.split('/')[-1].rsplit('_', 1)[0]})

#         bay_locations = df.loc[bay_indx].reset_index()
#         w_max.append(bay_locations[bay_locations.lon.between(west_pt, west_central)].max())
#         c_max.append(bay_locations[bay_locations.lon.between(west_central, east_central)].max())
#         e_max.append(bay_locations[bay_locations.lon.between(east_central, east_pt)].max())

       
#     west = pd.concat(w_max, axis=1).T
#     east = pd.concat(e_max, axis=1).T
#     central = pd.concat(c_max, axis=1).T

#     # Save for easy replotting
#     west.to_pickle(os.path.join(datapath, 'west_fg_max_v2.pkl.gz'),
#                      compression='gzip')
#     east.to_pickle(os.path.join(datapath, 'east_fg_max_v2.pkl.gz'),
#                      compression='gzip')
#     central.to_pickle(os.path.join(datapath, 'central_fg_max_v2.pkl.gz'),
#                         compression='gzip')
    

import xarray as xr
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

def process_file(file, bay_indx, west_pt, east_pt, central_pt):
    ds = xr.open_dataset(file)
    df = ds.eta.to_dataframe().rename(
        columns={'eta': file.split('/')[-1].rsplit('_', 1)[0]})

    bay_locations = df.loc[bay_indx].reset_index()
    
    west_central = west_pt + central_pt
    east_central = east_pt - central_pt
    
    w_max = bay_locations[bay_locations.lon.between(west_pt, west_central)].max()
    c_max = bay_locations[bay_locations.lon.between(west_central, east_central)].max()
    e_max = bay_locations[bay_locations.lon.between(east_central, east_pt)].max()
    
    return w_max, c_max, e_max

def bay_section_max_surge(fg_files, bay_indx, west_pt, east_pt, central_pt, datapath='/home/catherinej/temp_cleanup/data/processed'):
    w_max, c_max, e_max = [], [], []
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, fg_files, [bay_indx]*len(fg_files), [west_pt]*len(fg_files), [east_pt]*len(fg_files), [central_pt]*len(fg_files)))
    
    for w, c, e in results:
        w_max.append(w)
        c_max.append(c)
        e_max.append(e)
    
    west = pd.concat(w_max, axis=1).T
    east = pd.concat(e_max, axis=1).T
    central = pd.concat(c_max, axis=1).T

    # Save for easy replotting
    west.to_pickle(os.path.join(datapath, 'west_fg_max_v2.pkl.gz'), compression='gzip')
    east.to_pickle(os.path.join(datapath, 'east_fg_max_v2.pkl.gz'), compression='gzip')
    central.to_pickle(os.path.join(datapath, 'central_fg_max_v2.pkl.gz'), compression='gzip')

# Example usage
# bay_section_max_surge(fg_files, bay_indx, west_pt, east_pt, central_pt)
    

if __name__ == '__main__':
    path = '/home/catherinej/temp_cleanup/data/raw/fg_nc/'
    fg_files = [os.path.join(path, x) for x in os.listdir(path) if '.ipynb' not in x]
# west, east, central, bay_locations, bay_indx = bay_division(fg_files[0])
# save_selected_surge_locs(fg_files, bay_indx)
    bay_section_max_surge(fg_files, bay_indx, west, east, central)
    # total_bay_max_surge(fg_files, bay_indx)