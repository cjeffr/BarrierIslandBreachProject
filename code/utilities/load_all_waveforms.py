import os
import numpy as np
import pandas as pd

def load_waveforms_pandas(SOURCE_DIR):
    """
    This function will load all simulations gauge data into a dictionary with 
    key names for the simulation name.
    All simulations must be in the same parent directory
    """
    from pathlib import Path
    
    #Create an empty dictionary to hold all the data 
    # get folder names inside SOURCEDIR
    df_dict = {}
    sim_folders = [f.name for f in os.scandir(SOURCE_DIR) if f.is_dir()]
    df_dict = {key:[] for key in sim_folders}
    df_dict
    data = {}
    # Loop over simulation folders to get all gauge data loaded into the dictionary
    for fldr in df_dict:
        for path in Path(os.path.join(SOURCE_DIR, fldr)).rglob('gauge*.txt'):
            file = str(path)
            
            # Gets header information from the gauge file includes lat/lon not being used currently
            with open(file) as f:
                header = f.readline().split()
                gauge_id = int(header[2][-3:])
                cols = ['Time', f'{gauge_id}']
                lat = float(header[5])
                lon = float(header[4])

            data[gauge_id] = pd.read_csv(file, skiprows=3, header=None, 
                                       delim_whitespace=True, usecols=[1,5], index_col='Time', 
                                         names=cols)

        df_dict[fldr] = data

    return df_dict


def merge_clean_data(data_dict, landfall_time):
    """
    Takes a dictionary of dataframes, concatenates them together and
    Changes time from seconds to hours from landfall
    """
    df = pd.concat(data_dict, axis=1)
    df.columns = df.columns.droplevel(0)
    df.index -= 216000.0
    df.index /= 3600.0
    return df

if __name__ == "__main__":
    SOURCE_DIR = '/home/catherinej/486'
    landfall_time = 216000.0
    gauge_df_dict = load_waveforms_pandas(SOURCE_DIR)
    for key in gauge_df_dict:
        print(merge_clean_data(gauge_df_dict[key], landfall_time))
    