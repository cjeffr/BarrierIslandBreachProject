import pandas as pd
import os


def reindex_df(df_list, resample_gauge):
    """
    Re-indexes all gauge data to match the same length and time steps as the 
    example gauge, in most cases the no breach gauge data
    """
    df_l = []
    for df in df_list:
        df_re = df.reindex(df.index.union(resample_gauge.index)).interpolate('index').reindex(resample_gauge.index)
        df_l.append(df_re)
    return df_l


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


def load_gauge_data(allfiles):
    """
    Load all gauge files in file list (allfiles) and put into a dict to be 
    cleaned and merged into a single dataframe for plotting
    """
    df_list = []
    for file in allfiles:
        sim_name = file.split('/')[4]
        cols = ['Time', f'{sim_name}']
        df = pd.read_csv(file, skiprows=3, header=None,
                         delim_whitespace=True, usecols=[1,5],
                         names = cols, index_col='Time')
        if sim_name == 'no_breach':
            no_breach = df
        df_list.append(df)
    df_list = reindex_df(df_list, no_breach)
    data_df = merge_clean_data(df_list, 486)
    return data_df


PATH = '/home/catherinej/width_depth'
for g in bay_gauge_numbers:
    filename = f'gauge1{g:04}.txt' # Bay Gauges
    allfiles = glob.glob(os.path.join(PATH, '**', filename), 
                                     recursive=True)
    gauge_data = load_gauge_data(allfiles)
    # do stuff like plot gauges