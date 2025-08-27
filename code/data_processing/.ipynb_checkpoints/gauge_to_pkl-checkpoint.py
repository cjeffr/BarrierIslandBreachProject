import numpy as np
import pandas as pd
import os
import glob

def load_gauge_data(allfiles, no_breach):
    df_list = []
    for file in allfiles:
        sim_name = file.split('/')[-3]
        cols = ['Time', f'{sim_name}']
        df = pd.read_csv(file, skiprows=3, header=None,
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


def resave_gauges_pickle(PATH, gauges_to_save, no_breach_path):
    gauge_data = {}
    for location in gauges_to_save:
        for gauge in gauges_to_save[location]:

            gaugefile = f'gauge1{gauge:04}.txt'
            no_breach = pd.read_pickle(os.path.join(no_breach_path, 
                                                    f'gauge1{gauge:04}.pkl.gz'))
            no_breach = no_breach.rename(columns={'Eta': 'no_breach'})
            files = glob.glob(os.path.join(PATH, '**', gaugefile), recursive=True)
            gaugefiles = [file for file in files if 'r' not in file.split('/')[-1]]
            data = load_gauge_data(gaugefiles, no_breach)
            data.to_pickle(f'/home/catherinej/{location}_gauge1{gauge:04}.pkl.gz', compression='gzip')


if __name__ == '__main__':
    PATH = '/projects/weiszr_lab/catherine/breach_sims/'
    gauges_to_save = {'west' : [84, 82],
                      'central' : [133, 45],
                      'east' : [11, 119]}
    no_breach_path = '/home/catherinej/no_breach/_output'
    resave_gauges_pickle(PATH, gauges_to_save, no_breach_path)
