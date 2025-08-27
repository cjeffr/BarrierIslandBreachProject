#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import os
import numpy as np
import pygmt

def combine_patches(patch):
    lat = patch['Yc'][0]
    lon = patch['Xc'][:,0]
    wind_x = patch['wind_x'].T
    wind_y = patch['wind_y'].T 
    water = patch['water'].T
    land = patch['land'].T
    ds = xr.Dataset(data_vars={'u':(('lat', 'lon'), wind_x), 
                              'v':(('lat', 'lon'), wind_y),
                              'land':(('lat', 'lon'), land),
                              'water':(('lat','lon'), water)},
                   coords={'lat': lat,
                          'lon': lon})
    return ds

def organize_patches(patch_dict):
    ds_dict = {}
    amr_max = max(int(d['amr_level']) for d in patch_dict.values())
    for i in range(1, amr_max + 1, 1):
        ds_dict[i] = []
    for patch_no in patch_dict: 
        t = patch_dict[patch_no]['time']
        amr = patch_dict[patch_no]['amr_level']
        timestep = round(t/3600,2)
        ds = combine_patches(patch_dict[patch_no])
        ds_dict[patch_dict[patch_no]['amr_level']].append(ds) 
    return ds_dict, timestep

def get_corners(lon, lat, xdelta, ydelta):
    x = [x + 0.5*xdelta for x in lon]
    y = [y + 0.5*ydelta for y in lat]
    xs = [x[0], x[-1], x[-1], x[0], x[0]]
    ys = [y[0], y[0], y[-1], y[-1], y[0]]
    corners = np.array([[x[0], y[0], x[-1], y[-1]]])
    return xs, ys, corners

def read_fortq(frame, path):
    """
    Import fort.q files to get x,y,z data
    """
    from clawpack.pyclaw import solution
    fortq = solution.Solution(frame, path=path, file_format='ascii')
    patch_dict = {}
    for stateno, state in enumerate(fortq.states):
        patch = state.patch
        time = state.t
        this_level = patch.level
        Xc, Yc = state.grid.c_centers
        X_edge, Y_edge = state.grid.c_edges
        delta = patch.delta
        mask_coarse = np.empty(Xc.shape, dtype=bool)
        mask_coarse.fill(False)
        for stateno_fine, state_fine in enumerate(fortq.states):
            patch_fine = state_fine.patch
            if patch_fine.level != this_level+1:
                continue
            xlower_fine = patch_fine.dimensions[0].lower
            xupper_fine = patch_fine.dimensions[0].upper
            ylower_fine = patch_fine.dimensions[1].lower
            yupper_fine = patch_fine.dimensions[1].upper

            m1 = (Xc > xlower_fine) & (Xc < xupper_fine)
            m2 = (Yc > ylower_fine) & (Yc < yupper_fine)
            mask_coarse = (m1 & m2) | mask_coarse
        # s = speed(state)
        h = state.q[0,:,:]
        eta = state.q[3,:,:]
        wind_x = state.aux[4,:,:]
        wind_y = state.aux[5,:,:]
        print(wind_x.shape)
        drytol_default = 0.001
        topo = eta - h
        b = state.aux[0,:,:]
        water = np.ma.masked_where(h<= drytol_default, np.where(topo<0, eta, h))
        land = np.ma.masked_where(h>drytol_default, eta)
        patch_dict[stateno] = {'amr_level':this_level, 'Xc':Xc, 'Yc':Yc, 'water':water,
                               'land':land, 'delta':delta,  'time':time, 'wind_x': wind_x, 'wind_y': wind_y}
    return patch_dict

SOURCE_DIR = '/home/catherinej/geoclaw.old_020623/examples/storm-surge/isaac/'
patch_dict = read_fortq(4, os.path.join(SOURCE_DIR, '_output'))

ds_dict, ts = organize_patches(patch_dict)
for amr in ds_dict:
    ds_list = ds_dict[amr]
    for ds in ds_list:
        xs, ys, corners = get_corners(ds.lon.values, ds.lat.values, ds.lon[1]-ds.lon[0], ds.lat[1] - ds.lat[0])
        
# %%
grid = pygmt.datasets.load_earth_relief(resolution='30s', region=[-99, -60, 5, 45])
fig = pygmt.Figure()
pygmt.config(FONT_ANNOT='18p,Helvetica')
pygmt.config(FORMAT_GEO_MAP='ddd')
pygmt.config(MAP_FRAME_PEN='.4p,black')
pygmt.makecpt(cmap='lajolla', series=[0,2,.25], reverse=False)
fig.basemap(region=[-97.0, -77.0, 15, 32.2], projection='M20c', frame=['WSne', 'af'] )#'xa1.5f.5', 'ya1f.5'])
# fig.grdimage(a, nan_transparent=True, transparency=25)
for amr in ds_dict:
# [[-91.75  20.25 -84.5   32.  ]]
# [[-91.125  26.625 -86.     31.25 ]
    Q11 = [[-91.75, 20.25]]
    Q12 = [[-91.75, 32.0]]
    Q21 = [[-84.5, 20.25]]
    Q22 = [[-84.5, 32.]]
    
    x = np.array([-91.75-3, -84.5+3])
    y = np.array([20.25-1.5, 32.0+1.5])
    y1 = np.array([20.25, 20.25])
    y2 = np.array([32., 32.0])
    x1 = np.array([-91.75, -91.75])
    x2 = np.array([-84.5, -84.5])
    xx = np.random.uniform(-91.125, -86.00)
    yy = np.random.uniform(26.625, 31.25)
    ds_list = ds_dict[amr]
    for ds in ds_list:
       
        xs, ys, corners = get_corners(ds.lon.values, ds.lat.values, ds.lon[1]-ds.lon[0], ds.lat[1] - ds.lat[0])
        fig.grdimage(ds.water, nan_transparent=True, verbose='q')
        fig.plot(corners, style='r+s', pen='.3p,blue')
        
fig.coast(shorelines='thinnest,black')
fig.colorbar(frame=['x+l"Storm Surge Depth (m)"'])
fig.show()
# fig.savefig('amr_surge.png')
# %%
data_storm_path = '/home/catherinej/geoclaw.old_020623/examples/storm-surge/isaac_syn/'
data_patch_dict = read_fortq(4, os.path.join(data_storm_path, '_output'))
data_ds_dict, ts = organize_patches(data_patch_dict)
grid = pygmt.datasets.load_earth_relief(resolution='30s', region=[-99, -60, 5, 45])
fig = pygmt.Figure()
pygmt.config(FONT_ANNOT='18p,Helvetica')
pygmt.config(FORMAT_GEO_MAP='ddd')
pygmt.config(MAP_FRAME_PEN='.4p,black')
pygmt.makecpt(cmap='lajolla', series=[0,30,5], reverse=False)

#Map elements
region=[-97.0, -77.0, 15, 32]
fig.basemap(region=region, projection='M10c', frame=['Wsne', 'af'])
for amr in ds_dict:

    ds_list = ds_dict[amr]

    for ds1 in ds_list:
        wind_speed = np.sqrt(ds1.u**2 + ds1.v**2)
        fig.grdimage(wind_speed, nan_transparent=True, verbose='q')
fig.coast(shorelines='thinnest,black')
# fig.colorbar(frame=['x+l"Storm Surge Depth (m)"'])
fig.shift_origin(yshift='-h-0.55c')
fig.basemap(region=region, projection='M10c', frame=['WSne', 'af'])
for amr in data_ds_dict:
    data_ds_list = data_ds_dict[amr]
    for ds2 in data_ds_list:
        wind_speed = np.sqrt(ds2.u**2 + ds2.v**2)
        fig.grdimage(wind_speed, nan_transparent=True, verbose='q')
fig.coast(shorelines='thinnest,black')
fig.colorbar(frame=['x+l"Wind Speed (m/s)"']) #, position='JMR+o1.55c/0c+w18c/1c'
fig.show()
# fig.grdimage(a, nan_transparent=True, transparency=25)

# %%
# isaac_syn = xr.open_dataset('/home/catherinej/storm_files/isaac_syn2.nc')
data_storm_path = '/home/catherinej/geoclaw.old_020623/examples/storm-surge/isaac_syn/'
data_patch_dict = read_fortq(6, os.path.join(data_storm_path, '_output'))
data_ds_dict, ts = organize_patches(data_patch_dict)
grid = pygmt.datasets.load_earth_relief(resolution='30s', region=[-99, -60, 5, 45])
fig = pygmt.Figure()
pygmt.config(FONT_ANNOT='18p,Helvetica')
pygmt.config(FORMAT_GEO_MAP='ddd')
pygmt.config(MAP_FRAME_PEN='.4p,black')
pygmt.makecpt(cmap='lajolla', series=[0,30], reverse=False)

#Map elements
region=[-97.0, -77.0, 15, 32]
fig.basemap(region=region, projection='M10c', frame=['Wsne', 'af'])
for amr in ds_dict:

    ds_list = ds_dict[amr]

    for ds in ds_list:
        wind_speed = np.sqrt(ds.u**2 + ds.v**2)
        fig.grdimage(wind_speed, nan_transparent=True, verbose='q')
fig.coast(shorelines='thinnest,black')
# fig.colorbar(frame=['x+l"Storm Surge Depth (m)"'])
fig.shift_origin(yshift='-h-0.55c')
fig.basemap(region=region, projection='M10c', frame=['WSne', 'af'])
for amr in data_ds_dict:
    data_ds_list = data_ds_dict[amr]
    for ds in data_ds_list:
        wind_speed = np.sqrt(ds.u**2 + ds.v**2)
        fig.grdimage(wind_speed, nan_transparent=True, verbose='q')
# fig.grdimage(isaac_syn.sel(time=604800).speed, nan_transparent=True, verbose='q')
fig.coast(shorelines='thinnest,black')
fig.colorbar(frame=['x+l"Wind Speed (m/s)"']) #, position='JMR+o1.55c/0c+w18c/1c'
fig.show()
# %%
#648000, 669600
# %%
