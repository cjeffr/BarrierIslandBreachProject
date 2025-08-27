import os
import pandas as pd
import xarray as xr
import glob
import numpy as np
from clawpack.geoclaw import fgmax_tools

# Add list of simulation folders from which to extract the output
folders = ['no_breach_5.9.1', 'test_original_updates', 'test_486_geoclaw_updates']
path = '/projects/weiszr_lab/catherine/486/geoclaw_changes_validation'
# Functions that read specific gauge files and compile to a single dataframe
def load_gauge_data(allfiles):
    """Loads the data from a list of locations into a single dataframe by folder name

    """
    df_list = []
    for file in allfiles:
        sim_name = file.split('/')[-3]
        cols = ['Time', f'{sim_name}']
        df = pd.read_csv(file, skiprows=4, header=None,
                         delim_whitespace=True, usecols=[1,5],
                         names = cols, index_col='Time')
    
        if sim_name == 'no_breach':
            no_breach = df
        df_list.append(df)
    df_list.append(no_breach)
        
    df_list = reindex_df(df_list, no_breach)
    data_df = merge_clean_data(df_list, 486)
    return data_df

def merge_clean_data(data_dict, storm_id):
    """
    Takes a dictionary of dataframes, concatenates them together and
    Changes time from seconds to hours from landfall
    """
    df = pd.concat(data_dict, axis=1)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    landfall_time = get_landfall_time(storm_id)
    df.index -= landfall_time
    df.index /= 3600.0
    return df


def get_landfall_time(storm_id):
    import xarray as xr
    storm_data = xr.open_dataset(os.path.join('/projects/weiszr_lab/catherine/storm_files', 
                                              f'NACCS_TP_{storm_id:04}_SYN_L2.nc'))
    landfall_loc =  [-73.31, 40.68]
    eye = storm_data.eye_loc.data
    lat_row = np.where(eye[:,1] >= 40.68)
    lon_row = np.where(eye[:,0] <= -73.31)
    idx = max(min(lat_row[0]), min(lon_row[0]))
    landfall_time = storm_data.time[idx].values
    return landfall_time

    
def reindex_df(df_list, res_15):
    df_l = []
    for df in df_list:
        df_re = df.reindex(df.index.union(res_15.index)).interpolate('index').reindex(res_15.index)
        df_l.append(df_re)
    return df_l
def save_specific_gauges(path):
    gauges_to_save = {'west' : [84, 82],
                      'central' : [133, 45],
                      'east' : [11, 119]}
    gauge_data = {}
    for location in gauges_to_save:
        for gauge in gauges_to_save[location]:
#             no_breach = no_breach.rename(columns={'Eta': 'no_breach'})
            gaugefile = f'gauge1{gauge:04}.txt'
#             
            files = glob.glob(os.path.join(path, '**', gaugefile), recursive=True)
            
            gaugefiles = [file for file in files if 'r' not in file.split('/')[-1]]
            print(gaugefiles)
            data = load_gauge_data(gaugefiles)
            if 'no_breach' in data.columns:
                print('exists')
            data.to_pickle(f'/home/catherinej/geoclaw_changes_validation_output/{location}_gauge1{gauge:04}.pkl.gz', compression='gzip')
            
            
    
save_specific_gauges(path)
# Extract Fgmax_data for simulations
def load_fgmax_save_netcdf(path, fgno, savename):
    os.chdir(path)
    fg = fgmax_tools.FGmaxGrid()
    fg.read_fgmax_grids_data(fgno=fgno)
    fg.read_output(fgno=fgno, outdir=os.path.join(path, '_output'),
                   verbose=False)
    fg.eta = np.where(fg.B > 0, fg.h, fg.h + fg.B)
    print(fg.eta.shape, len(fg.y), len(fg.x))
    ds = xr.Dataset(data_vars={
        "B": (["lat", "lon"], fg.B),
        'h' :(["lat", "lon"], fg.h),
        'eta': (['lat', 'lon'], fg.eta)
        },
        coords={'lat': fg.Y[:,0],
                'lon': fg.X[0,:]})
    ds.to_netcdf(savename)
#     return fg
    
for simulation in folders:
    fh = os.path.join(path, simulation)
    savename = f'/home/catherinej/geoclaw_changes_validation_output/{simulation}_fgmax.nc'
    fg = load_fgmax_save_netcdf(fh, 1, savename)