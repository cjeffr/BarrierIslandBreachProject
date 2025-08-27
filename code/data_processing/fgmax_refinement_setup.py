import os, sys
import numpy as np
from clawpack.geoclaw import topotools, marching_front
from clawpack.amrclaw import region_tools
from clawpack.visclaw import colormaps, plottools
import matplotlib.pyplot as plt

def setup_colormaps(zmin=-60, zmax=40):
    # Land colormap definition
    land_cmap = colormaps.make_colormap({
        0.0: [0.1, 0.4, 0.0],
        0.25: [0.0, 1.0, 0.0],
        0.5: [0.8, 1.0, 0.5],
        1.0: [0.8, 0.5, 0.2]
    })

    # Sea colormap definition
    sea_cmap = colormaps.make_colormap({
        0.0: [0, 0, 1],
        1.0: [.8, .8, 1]
    })

    # Combine land and sea colormaps
    cmap, norm = colormaps.add_colormaps(
        (land_cmap, sea_cmap),
        data_limits=(zmin, zmax),
        data_break=0.
    )

    # Dry sea colormap definition
    sea_cmap_dry = colormaps.make_colormap({
        0.0: [1.0, 0.7, 0.7],
        1.0: [1.0, 0.7, 0.7]
    })

    # Combine land and dry sea colormaps
    cmap_dry, norm_dry = colormaps.add_colormaps(
        (land_cmap, sea_cmap_dry),
        data_limits=(zmin, zmax),
        data_break=0.
    )

    return cmap, norm, cmap_dry, norm_dry


# def plot_topo(X, Y, Z):
#     """
#     Creates a quick plot to ensure the region is correct using the 2d
#     X, Y, Z arrays from the topography object
#     """
#     cmap, norm, _, _ = setup_colormaps()

#     # plot the bathymetry
#     plottools.pcolorcells(X, Y, Z, cmap=cmap, norm=norm)
#     plt.colorbar(extend='both')
#     plt.gca().set_aspect(1./np.cos(48*np.pi/180.))
#     return plt.gca()
    
def plot_topo(X, Y, Z, ax=None):
    """
    Creates a quick plot to ensure the region is correct using the 2d
    X, Y, Z arrays from the topography object
    
    Parameters:
    X : 2D array
        X-coordinate data.
    Y : 2D array
        Y-coordinate data.
    Z : 2D array
        Z-coordinate data.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The axes on which to draw the topography plot. If not provided, a new figure will be created.
    
    Returns:
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes on which the topography plot is drawn.
    """
    cmap, norm, _, _ = setup_colormaps()

    if ax is not None:
        # Plot the bathymetry on the provided axes
        plottools.pcolorcells(X, Y, Z, cmap=cmap, norm=norm, ax=ax)
        ax.set_aspect(1. / np.cos(48 * np.pi / 180.))
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Topography Plot')
        return ax
    else:
        # Create a new figure and plot the bathymetry
        fig = plt.figure(figsize=(12, 6))
        plottools.pcolorcells(X, Y, Z, cmap=cmap, norm=norm)
        plt.colorbar(extend='both')
        plt.gca().set_aspect(1. / np.cos(48 * np.pi / 180.))
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Topography Plot')
        plt.show()


def create_refine_region(topo, z_depth, z_height, z_mid=0):
    """
    Use the marching_front algorithm to create a refinement region
    with provided depths for a given bathymetry
    
    topo = Bathymetric data read in with topotools.Topography class
    z_depth = Deepest part of refinement region
    z_height = Highest topography of refinment region
    z_mid = midpoint of refinement region, usually 0 for mean sea level
    max_iters = how many iterations to perform to obtain the region
    """
    # First pass of the region to get dry points near the shoreline
    pts_chosen = marching_front.select_by_flooding(
        topo.Z, Z1=z_mid, Z2=z_height, max_iters=None)
    Zmasked = np.ma.masked_array(topo.Z, np.logical_not(pts_chosen))
    
    # Plot results to see initial region
    plot_topo(topo.X, topo.Y, Zmasked)
    
    
    # Second pass of the region to obtain the depths
    pts_chosen_shallow = marching_front.select_by_flooding(
        topo.Z, Z1=z_mid, Z2=z_depth, max_iters=None)
    Zshallow = np.ma.masked_array(topo.Z, np.logical_not(pts_chosen_shallow))
    
    # Plot to see results of second region
    plot_topo(topo.X, topo.Y, Zshallow)
    
    # combine the two regions to get the nearshore region 
    # with both height and depth
    pts_chosen_nearshore = np.logical_and(pts_chosen, pts_chosen_shallow)
    Znearshore = np.ma.masked_array(topo.Z,
                                 np.logical_not(pts_chosen_nearshore))
    
    # Plot combined region
    plot_topo(topo.X, topo.Y, Znearshore)
    return pts_chosen_nearshore, Zshallow


def create_fgmax_file(pts_chosen_nearshore, topo_data, filename):
    """
    Create an fgmax file to get maximum eta during the simulation 
    Uses the pre-defined region from create_refine_region
    pts_chosen_nearshore: defines all of the points chosen given 
                          selected height and depths
    topo_data: Bathymetry for the region
    filename: name of the output fgmax file
    """
    # Create new instance of topography class and point to existing data
    topo_fgmax_mask = topotools.Topography()
    topo_fgmax_mask._x = topo_data.x
    topo_fgmax_mask._y = topo_data.y
    
    # change boolean to 1/0
    topo_fgmax_mask._Z = np.where(pts_chosen_nearshore, 1, 0)
    topo_fgmax_mask.generate_2d_coordinates()
    # Write output fgmax data file
    fgmax_fname = filename + '.data'
    topo_fgmax_mask.write(
        fgmax_fname, topo_type=3, Z_format='%1i')
    print(f'Created {fgmax_fname}')

def compare_ixy(topo, pts_chosen_nearshore):
    """
    Compare the two different marching front directions
    'x' and 'y' to see which best contains the area
    
    pts_chosen_nearshore: The previously determined region defined 
                          by depth and height
    """
    # Setup axes for comparison
    # Plot the total regions comparing ixy methods
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    
    # Select points by 'x'
    rr = region_tools.ruledrectangle_covering_selected_points(
        topo.X, topo.Y, pts_chosen_nearshore, ixy='x', method=0,
    padding=0, verbose=True)
    xv, yv = rr.vertices()
    ax1 = plot_topo(topo.X, topo.Y, pts_chosen_nearshore, ax=ax1)
    ax1.plot(xv, yv, 'r')
    ax1.set_title("With ixy = 'x'")

    
    rr = region_tools.ruledrectangle_covering_selected_points(
        topo.X, topo.Y, pts_chosen_nearshore, ixy='y', method=0,
        padding=0, verbose=True)
    xv,yv = rr.vertices()
    ax2 = plot_topo(topo.X, topo.Y, pts_chosen_nearshore, ax=ax2)
    ax2.plot(xv, yv, 'r')
    ax2.set_title("With ixy = 'y'")
    
    plt.show()
    
def write_rr_file(topo, pts_chosen_nearshore, outfile, ixy_method):
    """
    Writes the output of the marching front algorithm to a 
    ruled rectangle file with whichever method is better 'x' or 'y'
    topo: Bathymetry data in topotools.Topography() class
    pts_chosen_nearshore: depth to height points for consideration
    outfile: name of output file
    ixy_method: 'x' or 'y' whichever better outlines the region per above
    
    
    """
    
    rr = region_tools.ruledrectangle_covering_selected_points(
        topo.X, topo.Y, pts_chosen_nearshore, 
        ixy=ixy_method, method=0, padding=0, verbose=True)
    rr_name = outfile + '.data'
    rr.write(rr_name)