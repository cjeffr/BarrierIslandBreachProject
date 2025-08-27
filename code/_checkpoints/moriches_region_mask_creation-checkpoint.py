import fgmax_refinement_setup as fgrr
from clawpack.geoclaw import topotools
import numpy as np
import barrier_island_operations as bsearch

def read_crop_topo(topofile, region=[-72.88, -72.65, 40.72, 40.83]):
    topo = topotools.read_netcdf(topofile)
    topo_crop = topo.crop(region)
    topo_crop.plot()
    return topo_crop

def create_mask(topo_data, depth=-9, height=7):
    """
    Creates an array of the data in the topography that is > -9m of depth and less than 7m in elevation
    pts_chosen: array of the indices that satisfy the criteria above
    masked_region: masked_array with the pts_chosen unmasked and all the rest of the pts masked
    Default depth and height values create's a boundary around the barrier island system that can be used for GeoClaw's refinement region
    """
    pts_chosen, masked_region = fgrr.create_refine_region(topo_data, depth, height)
    return pts_chosen, masked_region

def save_mask(masked_region, savename):
    np.savez_compresssed(f'data/{savename}.npz', data=masked_region.data, mask=shallow_mask.mask)


def create_half_island_mask(topo_data, points_indices, save_path, island_name='moriches', direction='west', specific_point=None):
    """
    Search through island indices and mask out the island.
    Clarify the mask to include everything less than a specific point.
    """
    indices = []
    for data in points_indices:
        for i in range(data[1], data[2] + 1):
            indices.append((data[0], i))

    mask_array = np.zeros(shape=topo_data.Z.shape)
    inlety, inletx = specific_point
    
    if direction=='west':
        for locs in indices:
            col, row = locs
            if (topo_data.x[col] < inletx) & (topo_data.y[row] < inlety):
                mask_array[row, col] = 1
    else:
        for locs in indices:
            col, row = locs
            if (topo_data.x[col] > inletx) & (topo_data.y[row] > inlety):
                mask_array[row, col] = 1

    
    masked_data = np.ma.masked_array(topo_data.Z, np.logical_not(mask_array))
    np.savez(save_path + f'{direction}_masked_{island_name}.npz', 
             data=masked_data, mask=masked_data.mask)
    return masked_data
    

# Example usage:
# specify the specific point's latitude and longitude
specific_point = (n, w)

# call the function with the specific_point argument
mask_west = create_half_island_mask(topo_data, island_coords['inds'], save_path='./', island_name='moriches', direction='west', specific_point=specific_point)
specific_point = (s, e)
mask_east = create_half_island_mask(topo_data, island_coords['inds'], save_path='./', island_name='moriches', direction='east', specific_point=specific_point)

# Create a mask for all land at Moriches:
# Takes two steps: Create a mask to only get the water
# Then create a mask that only masks land
# Combined them for all land + ocean masked
topo_data = read_crop_topo('/home/catherinej/bathymetry/moriches/moriches.nc')
water_idxs, water_mask = create_refine_region(topo_data, depth=0, height=0)
ocean_idxs, ocean_mask = create_refine_region(topo_data, depth=-9, height=7)
save_mask(ocean_mask, 'ocean_mask_v2.npz')
combined_mask = np.ma.masked_array(topo_data.Z, np.ma.mask_or(water_mask.mask, ocean_mask.mask))
save_mask(combined_mask, 'land_ocean_mask.npz')

# Create island mask
# SEE barrier_island_operations.py

