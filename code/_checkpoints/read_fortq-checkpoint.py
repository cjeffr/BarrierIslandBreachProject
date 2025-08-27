from clawpack.pyclaw import Solution
import numpy as np
import xarray as xr


def read_fortq(frame, path, file_format='ascii'):
    """
    Import fort.q files to get x,y,z data
    """
    fortq = Solution(frame, path=path, file_format=file_format)
    patch_dict = {}
    for stateno, state in enumerate(fortq.states):
        patch = state.patch
        time = state.t
        this_level = patch.level
        Xc, Yc = state.grid.c_centers
        delta = patch.delta
        mask_coarse = np.empty(Xc.shape, dtype=bool)
        mask_coarse.fill(False)
        # for stateno_fine, state_fine in enumerate(fortq.states):
        #     patch_fine = state_fine.patch
        #     if patch_fine.level != this_level+1:
        #         continue
        #     xlower_fine = patch_fine.dimensions[0].lower
        #     xupper_fine = patch_fine.dimensions[0].upper
        #     ylower_fine = patch_fine.dimensions[1].lower
        #     yupper_fine = patch_fine.dimensions[1].upper
        #
        #     m1 = (Xc > xlower_fine) & (Xc < xupper_fine)
        #     m2 = (Yc > ylower_fine) & (Yc < yupper_fine)
        #     mask_coarse = (m1 & m2) | mask_coarse
        s = speed(state)
        h = state.q[0, :, :]
        eta = state.q[3, :, :]
        drytol_default = 0.001
        topo = eta - h
        wind_x = state.aux[4, :, :]
        wind_y = state.aux[5, :, :]
        water = np.ma.masked_where(h <= drytol_default, np.where(topo < 0, eta, h))
        land = np.ma.masked_where(h > drytol_default, eta)
        # Save variables to dictionary

        patch_dict[stateno] = {'amr_level': this_level, 'Xc': Xc, 'Yc': Yc,
                               'water': water, 'land': land, 'delta': delta,
                               'speed': s, 'time': time, 'wind_x': wind_x,
                               'wind_y': wind_y}
    return patch_dict


def speed(state):
    from pylab import sqrt, zeros
    from numpy.ma import masked_where
    q = state.q
    h = q[0, :, :]
    hs = sqrt(q[1, :, :]**2, q[2, :, :]**2)
    where_hpos = (h > 1e-3)
    s = zeros(h.shape)
    s[where_hpos] = hs[where_hpos]/h[where_hpos]
    s = masked_where(h < 1e-3, s)
    return s


def combine_patches(patch):
    lat = patch['Yc'][0]
    lon = patch['Xc'][:, 0]
    wind_x = patch['wind_x'].T
    wind_y = patch['wind_y'].T
    water = patch['water'].T
    land = patch['land'].T
    speed = patch['speed'].T
    ds = xr.Dataset(data_vars={'u': (('lat', 'lon'), wind_x),
                               'v': (('lat', 'lon'), wind_y),
                               'land': (('lat', 'lon'), land),
                               'water': (('lat', 'lon'), water),
                               'speed': (('lat', 'lon'), speed)},
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
        timestep = round(t/3600, 2)
        ds = combine_patches(patch_dict[patch_no])
        ds_dict[patch_dict[patch_no]['amr_level']].append(ds)
    return ds_dict, timestep
