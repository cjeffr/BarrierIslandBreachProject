import pandas as pd
import os
import glob
import sys
import shutil

def pickle_gauges():
    folder_path = '_output'  # Replace with the actual path to your folder
    gauge_files = glob.glob('_output/gauge1*.txt')
    dfs = []

    for file in gauge_files:
        gauge_number = int(file.split('/')[-1].split('.')[0].replace('gauge', ''))  # Extract gauge_number from the file name
        cols = ['Time', f'gauge_{gauge_number}']
        df = pd.read_csv(file, skiprows=4, header=None,
                         delim_whitespace=True, usecols=[1, 5],
                         names=cols, index_col='Time')
        dfs.append(df)

    # Combine all DataFrames into a single DataFrame
    result_df = pd.concat(dfs, axis=1)
    result_df.to_pickle(os.path.join('results/', 
                                     os.getcwd().split('/')[-1]) + '.pkl.gz', compression='gzip')
    
# Extract Fgmax_data for simulations
def load_fgmax_save_netcdf(savename, fgno=1):
    from clawpack.geoclaw import fgmax_tools
    import numpy as np
    import xarray as xr
    fg = fgmax_tools.FGmaxGrid()
    fg.read_fgmax_grids_data(fgno=fgno)
    fg.read_output(fgno=fgno, outdir='_output',
                   verbose=False)
    fg.eta = np.where(fg.B > 0, fg.h, fg.h + fg.B)
    ds = xr.Dataset(data_vars={
        "B": (["lat", "lon"], fg.B),
        'h' :(["lat", "lon"], fg.h),
        'eta': (['lat', 'lon'], fg.eta)
        },
        coords={'lat': fg.Y[:,0],
                'lon': fg.X[0,:]})
    ds.to_netcdf(savename)
    
if __name__=='__main__':
    pickle_gauges()
    sim_name = os.getcwd().split('/')[-1]
    savename = os.path.join('results', f'{sim_name}_fgmax.nc')
    load_fgmax_save_netcdf(savename)
    #shutil.rmtree('_output')
