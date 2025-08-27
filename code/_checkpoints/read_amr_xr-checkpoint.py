__author__ = 'Catherine Jeffries'

import os
import numpy as np
import xarray as xr

class ReadFort(object):
    """
    Read fort.q and fort.aux files into xarray

    Attributes:

    """
    def __init__(self, path, frame_no, file_format):
        self.drytol_default = 0.001
        self.data = self.read_fortq_solution(frame_no, path, file_format)
        self.dataset_list, self.timestep = self.organize_patches(self.data)

    def read_fortq_solution(self, frame_no, path, file_format):
        from clawpack.pyclaw import solution
        fortq = solution.Solution(frame_no, path=path, file_format=file_format)
        patch_dict = {}
        for stateno, state in enumerate(fortq.states):
            patch = state.patch
            time = state.t
            this_level = patch.level
            Xc, Yc = state.grid.c_centers
            X_edge, Y_edge = state.grid.c_edges
            delta = patch.delta
            height = state.q[0,:,:]
            eta = state.q[3,:,:]
            wind_u = state.aux[4, :, :]
            wind_v = state.aux[5, :, :]
            topo = eta - height
            water = np.ma.masked_where(height <= self.drytol_default, eta)
            land = np.ma.masked_where(height > self.drytol_default, eta)
            patch_dict[stateno] = {'amr_level': this_level,
                                   'Xc': Xc,
                                   'Yc': Yc,
                                   'X_edge': X_edge,
                                   'Y_edge': Y_edge,
                                   'water': water,
                                   'land': land,
                                   'delta': delta,
                                   'time': time,
                                   'wind_u': wind_u,
                                   'wind_v': wind_v}
            return patch_dict


    def organize_patches(self, patch_dict):
        ds_dict = {}
        amr_max = max(int(d['amr_level']) for d in patch_dict.values())
        for i in range(1, amr_max + 1, 1):
            ds_dict[i] = []
        for patch_no in patch_dict:
            t = patch_dict[patch_no]['time']
            amr = patch_dict[patch_no]['amr_level']
            timestep = round(t / 3600, 2)
            ds = self.combine_patches(patch_dict[patch_no])
            ds_dict[patch_dict[patch_no]['amr_level']].append(ds)
        return ds_dict, timestep

    def combine_patches(self, patch_data):
        lat = patch_data['Yc'][0]
        lon = patch_data['Xc'][:,0]
        wind_u = patch_data['wind_u'].T
        wind_v = patch_data['wind_v'].T
        water = patch_data['water'].T
        land = patch_data['land'].T
        ds = xr.Dataset(data_vars={'u': (('lat', 'lon'), wind_u),
                                   'v': (('lat', 'lon'), wind_v),
                                   'land': (('lat', 'lon'),land),
                                   'water': (('lat', 'lon'), water)},
                        coords={'lat': lat,
                                'lon': lon})
        return ds


if __name__ == '__main__':
    moo = ReadFort(path='/home/catherinej/multi/moriches_6/_output', frame_no=19, file_format='ascii')
    baa = moo.dataset_list
    print(baa)
