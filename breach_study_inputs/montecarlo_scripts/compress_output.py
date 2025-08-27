import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import tarfile
from clawpack.geoclaw import fgmax_tools
import xarray as xr
import sys

def load_fgmax_save_netcdf(path,  savename, fgno=1):
    os.chdir(path)
    fg = fgmax_tools.FGmaxGrid()
    fg.read_fgmax_grids_data(fgno=fgno)
    fg.read_output(fgno=fgno, outdir=os.path.join(path, '_output'),
                   verbose=False)
    fg.eta = np.ma.masked_where(fg.h <= 0.001,
                                np.where(fg.B<0, fg.B+fg.h,fg.h))
    da = xr.DataArray(data=fg.eta,
                      coords={'lat': fg.y,
                              'lon': fg.x})
    ds = da.to_dataset(name=f'{savename.split("/")[-1]}')
                             # 'name': savename.split('/')[-1]})
    ds.to_netcdf(savename)
    
    
def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        

def pickle_gauge_data(DATAPATH, OUTPATH):
    all_gauges = glob.glob(os.path.join(DATAPATH, '_output', 'gauge*.txt'), 
                           recursive=True)
    cols = ['Time', 'Eta']
    if all_gauges:
        for gauge in all_gauges:
            name = gauge.split('/')[-1].split('.')[0]
            df = pd.read_csv(gauge, skiprows=3, header=None,
                             delim_whitespace=True, usecols=[1,5],
                             index_col='Time', names=cols)
            df.to_pickle(os.path.join(OUTPATH, name + '.pkl.gz'),
                         compression='gzip')
            os.remove(gauge)
    if not os.path.exists(OUTPATH):
        print(f'Making tar file: {OUTPATH}')
        make_tarfile(OUTPATH)
        
        
if __name__ == '__main__':
    print(sys.argv)
    DATAPATH = os.path.join('/projects/weiszr_lab/catherine/breach_sims/rall_10423'
                            ,sys.argv[1])
    OUTPATH = f'/home/catherinej/gz_output/{sys.argv[1]}'
    if not os.path.exists(OUTPATH):
        os.mkdir(OUTPATH)
    savename = os.path.join(DATAPATH.split('/')[-1] + '_fgmax.nc')
    savename = os.path.join(OUTPATH, savename)
    load_fgmax_save_netcdf(DATAPATH, savename)
    pickle_gauge_data(DATAPATH, OUTPATH)
