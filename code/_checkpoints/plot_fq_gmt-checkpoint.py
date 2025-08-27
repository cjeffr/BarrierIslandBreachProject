import pandas as pd
import xarray as xr
import numpy as np
import os
from plot_fortq import read_fortq
from gmt_tools import GmtTools
import subprocess

bathy = '/mnt/c/RData/Bathy/entire_world_srtm30.grd'
[patch_dict, water, h, X, Y, eta]=read_fortq('/mnt/c/Projects/GMT/_output',4)

gmt = '/usr/bin/gmt/bin/gmt'
region = '-R-126.0/-123.0/44.0/48.0'
projection = '-JM3.0i'
output = 'test.ps'
B1 = ['-BneSW','-Bxa2f1','-Bya1f1']  # Boundaries
B2 = '-Bxa2f1'
B3 = '-Bya2f1'
infile = 'in.png'
outfile = 'test.ps'
moo = GmtTools(region, projection ,infile, outfile, bathy, B1)
moo.bmap()

moo.image()
patch = 5
print(patch_dict[patch]['amr_level'])
lat = patch_dict[patch]['lat']
long = patch_dict[patch]['long']
water = patch_dict[patch]['water']

df = pd.DataFrame(data=water, index=[i for i in long],
                  columns=[i for i in lat])  # 'Latitude':lat, 'Longitude':long})
print(df)
df.index.name = 'Longitude'
df.columns.name = "Latitude"
# print(df)
# df_new =
da = xr.DataArray(data=df.T, dims=('Latitude', 'Longitude'),
                  attrs={'actual_range':[np.nanmin(water), np.nanmax(water)]})
f = 'tmp.nc'
da.to_netcdf(f)
subprocess.run([gmt, 'begin', 'testing'])
subprocess.run([gmt, 'figure', 'testing', 'png'])
# # subprocess.run([gmt, 'grdmask', region, '-I0.01',  f,'-NZNaN', '-Gmask.grd'])
subprocess.run([gmt, 'grdinfo', f])
moo.water_img(f)
#
# subprocess.run([gmt, 'grdimage', bathy, region, projection,  B2, B3, '-Cgeo'])
# subprocess.run([gmt, 'grdimage', 'tmp.nc', region, projection, '-Cpolar'])
#
moo.convert_to_png()
subprocess.run([gmt, 'end'])
# P = '-P'
# T = '-TG'
# A = '-A0/2/0/0'  # .2c/3c'
# file = '-F{}.png'.format('test')
# G = '-G/home/cat/.pyenv/shims/gs'
#
# subprocess.run([gmt, 'psconvert', output, P, A, T, file])