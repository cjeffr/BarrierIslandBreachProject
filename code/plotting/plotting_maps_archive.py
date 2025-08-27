#!/usr/bin/env python
# coding: utf-8

"""
TITLE: Archive Data Plotter for Storm Surge Figures

Purpose: Plot figures 8, 9, and 10 using archived datasets
Dependencies: pygmt, xarray, pandas, numpy, yaml, pathlib

This script replaces the original fg_surge_plotter.py to work with 
the simplified archive data structure.
"""

import pygmt
import xarray as xr
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from collections import OrderedDict

# Configuration
def load_config():
    """Load GMT configuration from yaml file"""
    try:
        with open("gmt_config.yml", "r") as ymlfile:
            return yaml.safe_load(ymlfile)
    except FileNotFoundError:
        # Fallback configuration if file doesn't exist
        return {
            'region_map': {},
            'moriches_map': {
                'region': [-72.88, -72.65, 40.72, 40.83],
                'basemap_frame': 'lrtb'
            }
        }

# Data loading functions
def load_archive_dataset(figure_name, dataset_name, archive_dir='figure_data_archive'):
    """
    Load a specific dataset from the archive
    
    Parameters:
    -----------
    figure_name : str
        Name of figure directory (e.g., 'figure8', 'figure9', 'figure10')
    dataset_name : str  
        Name of dataset file without extension (e.g., 'no_breach', 'min_surge')
    archive_dir : str
        Path to archive directory
    
    Returns:
    --------
    xarray.Dataset
        Loaded dataset
    """
    archive_path = Path(archive_dir) / figure_name / f"{dataset_name}.nc"
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive file not found: {archive_path}")
    return xr.open_dataset(archive_path)

def load_figure_data(figure_name, dataset_order=None, archive_dir='figure_data_archive'):
    """
    Load all datasets for a specific figure
    
    Parameters:
    -----------
    figure_name : str
        Name of figure directory
    dataset_order : list, optional
        Specific order for datasets. If None, uses alphabetical order
    archive_dir : str
        Path to archive directory
        
    Returns:
    --------
    OrderedDict
        Dictionary of datasets in specified order
    """
    figure_dir = Path(archive_dir) / figure_name
    if not figure_dir.exists():
        raise FileNotFoundError(f"Figure directory not found: {figure_dir}")
    
    # Get all .nc files
    available_datasets = {f.stem: f for f in figure_dir.glob('*.nc')}
    
    # Order datasets
    if dataset_order:
        ordered_names = [name for name in dataset_order if name in available_datasets]
    else:
        ordered_names = sorted(available_datasets.keys())
    
    # Load datasets in order
    datasets = OrderedDict()
    for name in ordered_names:
        datasets[name] = xr.open_dataset(available_datasets[name])
    
    return datasets

def prepare_plotting_data(datasets):
    """
    Convert archive datasets to format expected by plotting functions
    
    Parameters:
    -----------
    datasets : OrderedDict
        Loaded datasets from archive
        
    Returns:
    --------
    OrderedDict
        Data formatted for plotting functions
    """
    fgdata = OrderedDict()
    
    for key, dataset in datasets.items():
        # Check if we have breach mesh data
        mesh_path = None
        if 'mesh_file' in dataset.attrs:
            mesh_file = dataset.attrs['mesh_file']
            if Path(mesh_file).exists():
                mesh_path = mesh_file
        
        fgdata[key] = {
            'data': dataset.eta_positive,  # This is equivalent to dataset.eta.where(dataset.eta > 0)
            'name': key,
            'mesh': mesh_path
        }
    
    return fgdata

# Utility functions
def re_scale(cbar_len):
    """Calculate scaling factor for colorbar elements"""
    return np.sqrt(cbar_len / 15)

def add_region_labels(fig, region):
    """Add west/central/east region labels and dividing lines"""
    third = 0.07663580247030193
    west = -72.87995371851132 + third
    east = -72.65004631110041 - third
    
    # Add text labels
    with pygmt.config(FONT_ANNOT_PRIMARY='12p'):
        fig.text(x=west - (third / 2), y=40.724, text='west', font='black')
        fig.text(x=east - (third / 2), y=40.724, text='central', font='black')
        fig.text(x=east + (third / 2), y=40.724, text='east', font='black')
    
    # Add dividing lines
    west_x = np.linspace(west, west, 20)
    y = np.linspace(region[2], region[3], 20)
    east_x = np.linspace(east, east, 20)
    fig.plot(x=west_x, y=y, pen='1p,black,-')
    fig.plot(x=east_x, y=y, pen='1p,black,-')

def setup_bathymetry_and_colormaps():
    """Set up bathymetry grid and color palettes"""
    grid = '../../bathymetry/moriches.nc'  # Update path as needed
    grd = pygmt.grdclip(grid=grid, below=[0.0, -50])
    
    # Set up colormaps
    pygmt.makecpt(cmap='gray', series=[-50, 50], reverse=True)  # Bathymetry
    return grd

# Main plotting functions
def plot_figure8(archive_dir='figure_data_archive', savepath='figure8_archive.pdf'):
    """
    Plot Figure 8: 2x2 subplot layout showing no_breach, min_surge, max_surge, mean_surge
    
    Parameters:
    -----------
    archive_dir : str
        Path to archive directory
    savepath : str
        Output file path
    """
    cfg = load_config()
    region = cfg['moriches_map']['region']
    
    # Load data in specific order
    dataset_order = ['no_breach', 'min_surge', 'max_surge', 'mean_surge']
    datasets = load_figure_data('figure8', dataset_order, archive_dir)
    fgdata = prepare_plotting_data(datasets)
    
    # Calculate subplot layout
    n_subplots = len(fgdata)
    if n_subplots == 2:
        nrows, ncols = 1, 2
        margins = '0.5c'
    else:
        ncols = 2
        nrows = -(-n_subplots // ncols)  # Ceiling division
        margins = '0.5c' if nrows > 1 else '1c'
    
    subsize = '8.25c'
    fig = pygmt.Figure()
    
    with pygmt.config(**cfg['region_map']):
        with fig.subplot(
            nrows=nrows,
            ncols=ncols,
            subsize=[subsize],
            frame='lrtb',
            autolabel='(a)+gwhite',
            sharex='b',
            sharey='l',
            region=region,
            margins=margins,
            projection='M8.25c'
        ):
            for i, key in enumerate(fgdata):
                with fig.set_panel(panel=i):
                    # Set up bathymetry
                    grd = setup_bathymetry_and_colormaps()
                    fig.grdimage(region=region, projection='M?', grid=grd, cmap=True, shading=True)
                    
                    # Plot surge data
                    pygmt.makecpt(cmap='lajolla', series=[0, 2.30, 0.10], continuous=False, reverse=True)
                    fig.grdimage(grid=fgdata[key]['data'], cmap=True, nan_transparent=True)
                    
                    # Plot breach mesh if available
                    if fgdata[key]['mesh'] and Path(fgdata[key]['mesh']).exists():
                        pygmt.makecpt(cmap='acton', series=[-2.0, 0, 0.10])
                        fig.plot(data=fgdata[key]['mesh'], fill='+z', pen="0.1p", close=True, cmap=True)
                    
                    # Add region labels
                    add_region_labels(fig, region)
        
        # Add colorbars
        cbar_len = 11
        scaling = re_scale(cbar_len)
        
        if nrows == 1:
            surge_cbar_position = 'JMR+o1c/0c+w5.5c/0.5c+mc'
            breach_cbar_position = 'JBC+o0c/1.15c+w15c/.5ch+mc'
        else:
            surge_cbar_position = 'JMR+o1c/0c+w11c/0.5c+mc'
            breach_cbar_position = 'JBC+o-0c/1.15c+w15c/.5ch+mc'
        
        # Surge colorbar
        with pygmt.config(FONT_ANNOT=f"{16/scaling}p", FONT_LABEL=f"{16/scaling}p"):
            pygmt.makecpt(cmap='lajolla', series=[0, 2.30, 0.10], continuous=False, reverse=True)
            fig.colorbar(position=surge_cbar_position, frame=['xa.3f.10+lSea Surface (m)'])
        
        # Breach colorbar
        with pygmt.config(FONT_ANNOT_PRIMARY='16p,Times-Roman'):
            pygmt.makecpt(cmap='acton', series=[-2.0, 0, 0.10])
            fig.colorbar(position=breach_cbar_position, frame=['xa.3f.10+lBreach Depth (m)'])
    
    fig.savefig(savepath, transparent=False, dpi=300)
    fig.show(verbose='i')
    print(f"Figure 8 saved to: {savepath}")

def plot_figure9(archive_dir='figure_data_archive', savepath='figure9_archive.pdf'):
    """
    Plot Figure 9: 2x1 subplot with insets
    
    Parameters:
    -----------
    archive_dir : str
        Path to archive directory
    savepath : str
        Output file path
    """
    cfg = load_config()
    region = cfg['moriches_map']['region']
    
    # Load data
    datasets = load_figure_data('figure9', archive_dir=archive_dir)
    fgdata = prepare_plotting_data(datasets)
    
    # Inset region
    r1 = [-72.879, (-72.84700 + -72.839850) / 2, 40.73363, 40.753449]
    
    fig = pygmt.Figure()
    
    with pygmt.config(**cfg['region_map']):
        with fig.subplot(
            nrows=1,
            ncols=2,
            subsize=['8.25c'],
            frame='lrtb',
            autolabel='(a)+gwhite',
            sharex='b',
            sharey='l',
            region=region,
            margins='0.5c',
            projection='M8.25c'
        ):
            for i, key in enumerate(fgdata):
                with fig.set_panel(panel=i):
                    # Set up bathymetry and plot surge data
                    grd = setup_bathymetry_and_colormaps()
                    fig.grdimage(region=region, projection='M?', grid=grd, cmap=True, shading=True)
                    
                    pygmt.makecpt(cmap='lajolla', series=[0, 2.30, 0.10], continuous=False, reverse=True)
                    fig.grdimage(grid=fgdata[key]['data'], cmap=True, nan_transparent=True)
                    
                    # Plot breach mesh if available
                    if fgdata[key]['mesh'] and Path(fgdata[key]['mesh']).exists():
                        pygmt.makecpt(cmap='acton', series=[-2.0, 0, 0.10])
                        fig.plot(data=fgdata[key]['mesh'], fill='+z', pen="0.1p", close=True, cmap=True)
                    
                    # Add region labels
                    add_region_labels(fig, region)
                    
                    # Add inset rectangle
                    rectangle = [[r1[0], r1[2], r1[1], r1[3]]]
                    fig.plot(data=rectangle, style='r+s', pen='2p,black')
                    
                    # Create inset
                    with pygmt.config(MAP_FRAME_TYPE='plain'):
                        with fig.inset(position='jTL+w2.7c/2c', box='+pblack'):
                            fig.basemap(region=r1, frame=True, projection='M?')
                            
                            # Plot inset data
                            pygmt.makecpt(cmap='gray', series=[-50, 50], reverse=True)
                            fig.grdimage(grid=grd, cmap=True, shading=True)
                            
                            pygmt.makecpt(cmap='lajolla', series=[0, 2.30, 0.10], continuous=False, reverse=True)
                            fig.grdimage(grid=fgdata[key]['data'], cmap=True, nan_transparent=True)
                            
                            if fgdata[key]['mesh'] and Path(fgdata[key]['mesh']).exists():
                                pygmt.makecpt(cmap='acton', series=[-2.0, 0, 0.10])
                                fig.plot(data=fgdata[key]['mesh'], fill='+z', pen="0.1p", close=True, cmap=True)
        
        # Add colorbars
        cbar_len = 5.35
        scaling = re_scale(cbar_len)
        
        with pygmt.config(
            FONT_ANNOT=f"{16/scaling}p",
            FONT_LABEL=f"{16/scaling}p",
            MAP_TICK_LENGTH_PRIMARY=f"{5/scaling}p",
            MAP_TICK_LENGTH_SECONDARY=f"{2.5/scaling}p",
            MAP_ANNOT_OFFSET=f"{5/scaling}p"
        ):
            # Surge colorbar
            surge_cbar_position = f'JMR+w{cbar_len}c/.5c+o0.95c/0c+mc'
            pygmt.makecpt(cmap='lajolla', series=[0, 2.30, 0.10], continuous=False, reverse=True)
            fig.colorbar(position=surge_cbar_position, frame=['xa.3f.10+lSea Surface (m)'])
        
        # Breach colorbar
        with pygmt.config(FONT_ANNOT_PRIMARY='16p,Times-Roman'):
            breach_cbar_position = 'JBC+o0c/1.15c+w15c/.5c+mc'
            pygmt.makecpt(cmap='acton', series=[-2.0, 0, 0.10])
            fig.colorbar(position=breach_cbar_position, frame=['xa.3f.10+lBreach Depth (m)'])
    
    fig.savefig(savepath, transparent=False, dpi=300)
    fig.show(verbose='i')
    print(f"Figure 9 saved to: {savepath}")

def plot_figure10(archive_dir='figure_data_archive', savepath='figure10_archive.pdf'):
    """
    Plot Figure 10: 2x1 subplot layout
    
    Parameters:
    -----------
    archive_dir : str
        Path to archive directory  
    savepath : str
        Output file path
    """
    cfg = load_config()
    region = cfg['moriches_map']['region']
    
    # Load data
    datasets = load_figure_data('figure10', archive_dir=archive_dir)
    fgdata = prepare_plotting_data(datasets)
    
    fig = pygmt.Figure()
    
    with pygmt.config(**cfg['region_map']):
        with fig.subplot(
            nrows=int(round(len(fgdata) / 2)),
            ncols=2,
            subsize=['8.25c'],
            frame='lrtb',
            autolabel='(a)+gwhite',
            sharex='b',
            sharey='l',
            region=region,
            margins='0.5c',
            projection='M8.25c'
        ):
            for i, key in enumerate(fgdata):
                with fig.set_panel(panel=i):
                    # Set up bathymetry and plot surge data
                    grd = setup_bathymetry_and_colormaps()
                    fig.grdimage(region=region, projection='M?', grid=grd, cmap=True, shading=True)
                    
                    pygmt.makecpt(cmap='lajolla', series=[0, 2.30, 0.10], continuous=False, reverse=True)
                    fig.grdimage(grid=fgdata[key]['data'], cmap=True, nan_transparent=True)
                    
                    # Plot breach mesh if available
                    if fgdata[key]['mesh'] and Path(fgdata[key]['mesh']).exists():
                        pygmt.makecpt(cmap='acton', series=[-2.0, 0, 0.10])
                        fig.plot(data=fgdata[key]['mesh'], fill='+z', pen="0.1p", close=True, cmap=True)
                    
                    # Add region labels
                    add_region_labels(fig, region)
        
        # Add colorbars
        if int(len(fgdata) / 2) == 2:
            surge_cbar_position = 'JMR+o1c/0c+w11c/0.5c+mc'
            breach_cbar_position = 'JBC+o-0c/1.15c+w15c/.5ch+mc'
            cbar_len = 15
            scaling = re_scale(cbar_len)
        else:
            cbar_len = 5.35
            scaling = re_scale(cbar_len)
            surge_cbar_position = f'JMR+w{cbar_len}c/.5c+o0.95c/0c+mc'
            breach_cbar_position = 'JBC+o0c/1.15c+w15c/.5c+mc'
        
        # Surge colorbar
        with pygmt.config(
            FONT_ANNOT=f"{16/scaling}p",
            FONT_LABEL=f"{16/scaling}p",
            MAP_TICK_LENGTH_PRIMARY=f"{5/scaling}p",
            MAP_TICK_LENGTH_SECONDARY=f"{2.5/scaling}p",
            MAP_ANNOT_OFFSET=f"{5/scaling}p"
        ):
            pygmt.makecpt(cmap='lajolla', series=[0, 2.30, 0.10], continuous=False, reverse=True)
            fig.colorbar(position=surge_cbar_position, frame=['xa.3f.10+lSea Surface (m)'])
        
        # Breach colorbar
        with pygmt.config(FONT_ANNOT_PRIMARY='16p,Times-Roman'):
            pygmt.makecpt(cmap='acton', series=[-2.0, 0, 0.10])
            fig.colorbar(position=breach_cbar_position, frame=['xa.3f.10+lBreach Depth (m)'])
    
    fig.savefig(savepath, transparent=False, dpi=300)
    fig.show(verbose='i')
    print(f"Figure 10 saved to: {savepath}")

# Main execution
def main():
    """Main function to generate all figures"""
    print("Plotting figures from archive data...")
    
    # Check if archive directory exists
    archive_dir = '/home/catherinej/figure_data_archive/'
    if not Path(archive_dir).exists():
        print(f"Error: Archive directory '{archive_dir}' not found!")
        print("Please run the archive creation script first.")
        return
    
    try:
        # Plot all figures
        print("\n--- Plotting Figure 8 ---")
        plot_figure8(archive_dir)
        
        print("\n--- Plotting Figure 9 ---") 
        plot_figure9(archive_dir)
        
        print("\n--- Plotting Figure 10 ---")
        plot_figure10(archive_dir)
        
        print("\nAll figures completed successfully!")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        print("Please check that all archive files exist and paths are correct.")

if __name__ == "__main__":
    main()