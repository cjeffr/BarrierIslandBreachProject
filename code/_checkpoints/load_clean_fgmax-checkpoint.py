%config InlineBackend.figure_format = 'retina'
import pygmt
import pandas as pd
import xarray as xr
import numpy as np
from clawpack.geoclaw import fgmax_tools
import os
import glob


def load_fgmax_grid_data(fname, fgno,  path, verbose=False):
    os.chdir(path)
    fg = fgmax_tools.FGmaxGrid()
    fg.read_fgmax_grids_data(fgno=fgno)
    fg.read_output(fgno=fgno, outdir=os.path.join(path, '_output'), verbose=False)
    zeta = np.ma.masked_where(fg.h<= 0.001, np.where(fg.B<0, fg.B+fg.h, fg.h))
    return {'X': fg.X,
            'x': fg.x,
            'y': fg.y,
            'Y': fg.Y,
            'zeta': zeta}


def convert_to_da(fg_data):
    max_value = 2.5
    clines_zeta = np.linspace(0.0, max_value, 10)
    z = fg_data['zeta']
    zeta1 = np.empty_like(z, dtype='float')
    zeta1[:,:] = z[:,:]
    nnnx, nnny = zeta1.shape
    print(np.nanmax(zeta1))
    # clines_zeta = np.linspace(0.0, max_value, 11
    mmax = 0.0
    for i in range(nnnx):
        for j in range(nnny):
            if zeta1[i,j] > mmax and zeta1[i,j] != 'NaN':
                mmax = zeta1[i,j]
            if zeta1[i,j] > max_value and zeta1[i,j] != 'NaN':
                zeta1[i,j] = max_value
            if zeta1[i,j] < 0.0 and zeta1[i,j] != 'NaN':
                zeta1[i,j] = 0.0    
    print(mmax)
    da = xr.DataArray(data=zeta1, coords=[("y",fg_data['y']), ("x",fg_data['x'])])
    return da


SOURCE_DIR = '/home/catherinej/width_depth/'
filename = 'fgmax_grids.data'
allfiles = glob.glob(os.path.join(SOURCE_DIR, '**', filename), recursive=True)

compare_path = '/home/catherinej/width_depth/no_breach'
dirs = [os.path.dirname(f) for f in allfiles if os.path.dirname(f) != compare_path and '_rand_' in os.path.dirname(f)]
source_dirs = [d for d in dirs if not d.endswith('_output')]
compare = load_fgmax_grid_data('moriches_fgmax2.data', 1, compare_path)
compare['da'] = convert_to_da(compare)
compare['name'] = 'No Breach'

for d in source_dirs:
    print(d + '\n')
    name = d.split('/')[4]
    data = plot_fgmax_grid('moriches_fgmax2.data', 1, d)
    data['da'] = convert_to_da(data)
    data['name'] = d.split('/')[4]
    # Then plot things below here at some point
