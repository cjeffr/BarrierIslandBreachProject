"""
This code creates a figure to illustrate the study region for the Barrier
Island Breach project. Illustrating, the storm track, general location, and 
points of interest from the study
"""
import os
import numpy as np
import pandas as pd
import pygmt
import xarray as xr
import glob
import yaml

# Load map configuration from YAML file
with open('../../notebooks/gmt_config.yml', 'r') as ymlfile:
    map_cfg = yaml.safe_load(ymlfile)

# Load processed surge point data for the chosen surge locations
# in the west, central, and east locations
west = pd.read_pickle('../../data/processed/west_fgmax_points.pkl.gz')
central = pd.read_pickle('../../data/processed/central_fgmax_pts.pkl.gz')
east = pd.read_pickle('../../data/processed/east_fgmax_pts.pkl.gz')

# Extracts the lat/lon pairs for plotting surge locations
surge_locs = np.array([west.columns[1], 
                       central.columns[1], 
                       east.columns[1]])

# Helper function to split tuple strings into components
def split_tuple(value):
    location, number = value.strip('()').split(", ")
    return location, int(number)


# Load gauge locations, parsing the index column as a tuple
gauge_locs = pd.read_csv('../../data/plotting_gauges_locs.csv',
                        converters={'Unnamed: 0': split_tuple})
# Expand tuple into seperate columns
gauge_locs[['location', 'number']] = pd.DataFrame(gauge_locs['Unnamed: 0'].tolist(),
                                                  index=gauge_locs.index)

# Drop unneeded columns
gauge_locs.drop('Unnamed: 0', axis=1, inplace=True)
# Clean up quotes in location names
gauge_locs['location'] = gauge_locs['location'].str.strip("'\'")

# Filter to only keep specific gauge IDS
numbers_to_keep = [84,82,133, 45, 11, 119]
gauge_locs = gauge_locs[gauge_locs['number'].astype(int).isin(numbers_to_keep)]

# Define mapping of region names to label letters for plotting
location_mapping = {'west': ['a', 'b'],
                    'central': ['c', 'd'],
                    'east':['e', 'f']}

# Initialize new column for label
gauge_locs['new_column'] = ''
# ASsign each gauge a letter based on its region
for location in location_mapping.keys():
    indices = gauge_locs[gauge_locs['location'] == location].index
    if len(indices) > 0:
        location_values = location_mapping[location]
        for i, index in enumerate(indices):
            value_index = i % len(location_values)
            gauge_locs.at[index, 'new_column'] = location_values[value_index]                                           

# Load storm track for 1938 Hurricane
storm_track = pd.read_csv('../../data/1938_storm_track.csv')

# Predefined annotation positions for various landmarks
annot_locs = {'Forge River': [-72.83288, 40.80620986-0.025],
           'Seatuck Cove': [-72.7260 - .02, 40.8094],
           'Fire Island': [-72.832, 40.7253],
           'Westhampton Island': [-72.675, 40.767],
           'Moriches Inlet': [-72.7540, 40.75],
           'Moriches Bay': [-72.80, 40.77]
             }

central_pt = -72.87995371851132 - 0.07663580247030193
east_pt = -72.65004631110041 +  0.07663580247030193

# Set up main map using config parameters
# Create figure
fig = pygmt.Figure()
#Load inset annotation information
inset_annot = pd.read_csv('../../notebooks/moriches_map_annotations.csv')

# Set up main map using configuration parameters
with pygmt.config(**map_cfg['region_map']):
    region = map_cfg['moriches_map']['region']
    frame_params = map_cfg['moriches_map']['basemap_frame']
    grid = '/home/catherinej/bathymetry/moriches.nc'

    # Clip bathymetry grid to below -50 m depth for shading
    grd = pygmt.grdclip(grid=grid, below=[0.0, -50])
    # Compute gradient for hillshading
    dgrid = pygmt.grdgradient(grid=grd, radiance=[100, 0])

    # Draw the base map frame and border
    fig.basemap(
        region=region,
        projection='M16.50c',  # Mercator projection
        frame=frame_params
    )

    # Create a gray color palette for elevation
    pygmt.makecpt(cmap='gray', series=[-50, 50], reverse=True)
    # Plot the bathymetry grid with shading
    fig.grdimage(grid=grd, cmap=True, shading=True)

    # Plot surge measurement locations as green circles
    fig.plot(
        x=surge_locs[:, 1], y=surge_locs[:, 0],
        style='c0.45c', pen='black', fill='#009E73',
        label='surge location+S0.5c'
    )

    # Separate gauge data by region for plotting
    w = gauge_locs[gauge_locs['location'] == 'west']
    c = gauge_locs[gauge_locs['location'] == 'central']
    e = gauge_locs[gauge_locs['location'] == 'east']

    # Plot west gauges: labels and blue circles
    fig.text(
        text=w['new_column'], x=w.lon - 0.006, y=w.lat,
        font='Times-Roman,#0072B2', fill='white', pen='.25p,black'
    )
    fig.plot(
        x=w.lon, y=w.lat, style='c0.45c', pen='black', fill='#0072B2',
        label='west gauges+S0.5c'
    )

    # Plot central gauges: labels and orange circles
    fig.text(
        text=c['new_column'], x=c.lon - 0.006, y=c.lat,
        font='Times-Roman,#D55E00', fill='white', pen='.25p,black'
    )
    fig.plot(
        x=c.lon, y=c.lat, style='c0.45c', pen='black', fill='#D55E00',
        label='central gauges+S0.5c'
    )

    # Plot east gauges: labels and light-blue circles
    fig.text(
        text=e['new_column'], x=e.lon - 0.006, y=e.lat,
        font='Times-Roman,#56B4E9', fill='white', pen='.25p,black'
    )
    fig.plot(
        x=e.lon, y=e.lat, style='c0.45c', pen='black', fill='#56B4E9',
        label='east gauges+S0.5c'
    )

    # Draw vertical dashed lines marking central and east boundaries
    fig.plot(
        x=[central_pt, central_pt], y=[region[2], region[3]],
        pen='1p,--'
    )
    fig.plot(
        x=[east_pt, east_pt], y=[region[2], region[3]],
        pen='1p,--'
    )

    # Create inset map with a plain frame
    with pygmt.config(MAP_FRAME_TYPE='plain'):
        with fig.inset(position='jTL+w7/4.32+o0.25c', box='+pblack'):
            inset_grd = '/home/catherinej/bathymetry/gebco_2020_n45.0_s8.0_w-88.0_e-50.0.nc'
            inset_grid = pygmt.grdclip(grid=inset_grd, below=[0.0, -50])
            inset_region = [-75, -71.25, 40.25, 42]

            # Draw inset basemap
            fig.basemap(region=inset_region, frame=True, projection='M?')
            pygmt.makecpt(cmap='gray', series=[-50, 500], reverse=True)
            fig.grdimage(grid=inset_grid, cmap=True, shading=True, projection='M?')
            fig.coast(borders=['2/1p,black'], shorelines=True)

            # Highlight main map region on inset
            rectangle = [[region[0], region[2], region[1], region[3]]]
            fig.plot(data=rectangle, style='r+s', pen='2p,#D55E00')

            # Plot storm track on inset
            fig.plot(
                x=storm_track['Storm Longitude (deg)'],
                y=storm_track['Storm Latitude (deg)'],
                pen='3p,#56B4E9'
            )

            # Add annotations to inset from CSV file
            for idx, row in inset_annot.iterrows():
                name, lon, lat, font, pen, angle, fill = row
                fig.text(
                    x=lon, y=lat, text=f'{name}', angle=f'{angle}',
                    font=f'{font}', pen=f'{pen}', fill=f'{fill}'
                )

    # Plot textual annotations and arrows for key landmarks
    for key, data in annot_locs.items():
        x, y = data
        # Offset text/arrow orientation for certain keys
        if key in ['Fire Island', 'Westhampton Island', 'Moriches Inlet']:
            fig.text(
                x=x - 0.015, y=y, text=key,
                font='14p,Times-Roman,black', fill='white', pen='0.25p'
            )
            fig.plot(
                x=x, y=y + 0.005, style='v0.5c+e+h0.05', direction=([90],[1.05]),
                pen='0.6p', fill='black'
            )
        else:
            fig.text(
                x=x - 0.015, y=y, text=key,
                font='14p,Times-Roman,black', fill='white', pen='0.25p'
            )
            fig.plot(
                x=x + 0.005, y=y, style='v0.5c+e+h0.05', direction=([0],[1]),
                pen='0.6p', fill='black'
            )

    # Plot breach locations from data file
    fig.plot(
        data='../../notebooks/w_0486_6_mesh.dat',
        style='d0.45c', pen='black', fill='#CC79A7',
        label='1938 Breaches+S0.5c'
    )

    # Create color palette for colorbar (positive values only)
    pygmt.makecpt(
        cmap='gray', series=[0, 50], reverse=True, background='white'
    )

    # Add colorbar with custom position and styling
    breach_cbar_position = 'JBC+o-0c/1.15c+w15/.5h+mc'
    with pygmt.config(FONT_ANNOT_PRIMARY='16p,Times-Roman', FONT_LABEL='16p,Times-Roman'):
        fig.colorbar(position=breach_cbar_position, frame=['xa10+lElevation (m)'])
    # Add legend with custom position and box
    with pygmt.config(FONT_ANNOT_PRIMARY='16p,Times-Roman'):
        fig.legend(position='JBR+jBR+o.5c/.45c+w4.55/3.25', box=True)

# Save figure to PDF at high resolution
fig.savefig('fig1.pdf', transparent=False, dpi=600)
# Display figure interactively
fig.show(verbose='i')
