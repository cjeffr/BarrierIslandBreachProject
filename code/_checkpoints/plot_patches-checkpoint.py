import numpy as np
from clawpack.pyclaw import Solution
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy as sp
import cartopy.crs as ccrs
import cartopy
import pandas as pd
import xarray as xr


def read_fortq(frame):
    """
    Import fort.q files to get x,y,z data
    """
    fortq = Solution(frame, file_format='ascii')
    patch_dict = {}
    for stateno, state in enumerate(fortq.states):
        patch = state.patch
        this_level = patch.level
        Xc, Yc = state.grid.c_centers
        mask_coarse = np.empty(Xc.shape, dtype=bool)
        mask_coarse.fill(False)
        for stateno_fine, state_fine in enumerate(fortq.states):
            patch_fine = state_fine.patch
            if patch_fine.level != this_level + 1:
                continue
            xlower_fine = patch_fine.dimensions[0].lower
            xupper_fine = patch_fine.dimensions[0].upper
            ylower_fine = patch_fine.dimensions[1].lower
            yupper_fine = patch_fine.dimensions[1].upper

            m1 = (Xc > xlower_fine) & (Xc < xupper_fine)
            m2 = (Yc > ylower_fine) & (Yc < yupper_fine)
            mask_coarse = (m1 & m2) | mask_coarse

        h = state.q[0, :, :]
        eta = state.q[3, :, :]
        drytol_default = 0.001
        water = np.copy(eta)
        idx = np.where(h <= drytol_default)
        water[idx] = np.NaN

        #         water[mask_coarse == True] = np.NaN

        # Save variables to dictionary
        long = Xc[:, 0]
        lat = Yc[0]
        patch_dict[stateno] = {"lat": lat, 'long': long, 'eta': eta, 'amr_level': this_level, 'Xc': Xc, 'Yc': Yc,
                               'water': water}
    return patch_dict, water, h, Xc, Yc, eta

[patch_dict, water, h, X, Y, eta]=read_fortq(4)
def combine_patches(patch):
    lat = patch['Yc'][0]
    lon = patch['Xc'][:,0]
    water = patch['water']
    df = pd.DataFrame(data=water, index=[i for i in lon], columns=[i for i in lat])# 'Latitude':lat, 'Longitude':long})
    df.index.name='Longitude'
    df.columns.name="Latitude"
    return df

da1 = []
da2 = []
da3 = []
da4 = []
da5 = []
da6 = []

for patch_no in patch_dict:
    if patch_dict[patch_no]['amr_level'] == 1:
        da1 = combine_patches(patch_dict[patch_no])
    if  patch_dict[patch_no]['amr_level'] == 2:
        df = combine_patches(patch_dict[patch_no])
        da2.append(df)
    if patch_dict[patch_no]['amr_level'] == 3:
        df = combine_patches(patch_dict[patch_no])
        da3.append(df)
    if patch_dict[patch_no]['amr_level'] == 4:
        df = combine_patches(patch_dict[patch_no])
        da4.append(df)
    if patch_dict[patch_no]['amr_level'] == 5:
        df = combine_patches(patch_dict[patch_no])
        da5.append(df)
    if patch_dict[patch_no]['amr_level'] == 6:
        df = combine_patches(patch_dict[patch_no])
        da6.append(df)


def clean_patches(patch_list):
    df_comb = pd.concat(patch_list)
    df_sort = df_comb.groupby(level=0).sum()
    df_sort.sort_index(axis=0)
    df_sort.replace(0.00, np.nan, inplace=True)
    return df_sort


def convert_to_xarray(df):
    da = xr.DataArray(df.T)
    return da


def plot_patches(da):
    map_limits = [-130.0, -120.0, 40.0, 51.0]
    ax = fig.add_subplot(121, projection=ccrs.PlateCarree(map_limits))
    ax.set_extent(map_limits)
    ax.coastlines(resolution='10m')

    da.plot.pcolormesh(x='Longitude', y='Latitude', add_colorbar=False, vmin=-0.1, vmax=0.1, ax=ax)

patch_vars = [da1, da2, da3, da4, da5, da6]
fig = plt.figure(figsize=[10,7])
for i in range(6):
    patch_list = patch_vars[i]
    if i == 0:
        da = convert_to_xarray(patch_list)
        plot_patches(da)
    else:
        df = clean_patches(patch_list)
        da = convert_to_xarray(df)
        plot_patches(da)

class South(cartopy.crs.Projection):
    def __init__(self):

        # see: http://www.spatialreference.org/ref/epsg/3408/
        proj4_params = {'proj': 'stere',
            'lat_0': 35.,
            'lon_0': -130,
            'lat_ts':-124.0,
            'x_0': 0,
            'y_0': 0,
            #'a': 6371228,
            #'b': 6371228,
            'a':6378137,
            'b':298.2572235630016,
            'units': 'm',
            'datum':'WGS84',
            #'ellipse':'WGS84',
            'no_defs': ''}

        super(South, self).__init__(proj4_params)

    @property
    def boundary(self):
        coords = ((self.x_limits[0], self.y_limits[0]),(self.x_limits[1], self.y_limits[0]),
                  (self.x_limits[1], self.y_limits[1]),(self.x_limits[0], self.y_limits[1]),
                  (self.x_limits[0], self.y_limits[0]))

        return cartopy.crs.sgeom.Polygon(coords).exterior

    @property
    def threshold(self):
        return 1e5

    @property
    def x_limits(self):
        return (-12400000,12400000)

    @property
    def y_limits(self):
        return (-12400000, 12400000)

import gdal
import matplotlib.colors as mcolors
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
gtif = gdal.Open('/mnt/c/RData/Bathy/NA_CAS.grd')
arr = gtif.ReadAsArray()
trans = gtif.GetGeoTransform()
extent = (-130.0, -120.0, 40.0, 50.0)
# extent = (trans[0], trans[0] + gtif.RasterXSize*trans[1],
#             trans[3] + gtif.RasterYSize*trans[5], trans[3])
arr[:, :] = -arr[:, :] * (arr > -1e20)
c = mcolors.ColorConverter().to_rgb

color1 = mcolors.ColorConverter.to_rgba('white')
color2 = mcolors.ColorConverter.to_rgba('black')

# make the colormaps
cmap2 = mcolors.LinearSegmentedColormap.from_list('my_cmap2', [color1, color2], 256)
cmap2._init()  # create the _lut array, with rgba values

# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
cmap2._lut[0:10, -1] = 0.0  # We made transparent de 10 first levels of hillshade,
cmap2._lut[10:cmap2.N + 3 - 1, -1] = 0.99
ax.imshow(arr[:, :], extent=extent, cmap='gist_earth',
          alpha=0.7)


# lons = load_da.variables['lon']
# lats = load_da.variables['lat']
# d = load_da.variables['z']
# load_da.close()
extent
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(lons, lats, d)

bathy = xr.open_dataset('/mnt/c/RData/Bathy/entire_world_srtm30.grd')
X, Y = np.meshgrid(bathy['lon'], bathy['lat'])
z = bathy['z']
plt.pcolor(X, Y,z)