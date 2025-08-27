import pygmt
import xarray as xr
import pandas as pd
import numpy as np


def plot_fg_subplots(fgdata, savepath):
    annot_locs = {'Forge River': [-72.83288, 40.80620986],
                  'Seatuck Cove': [-72.7260 - .030, 40.8094],
                 }
    import yaml
    with open("/home/catherinej/claw_code/src/claw_code/post/gmt_config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    fig = pygmt.Figure()
    
    with pygmt.config(**cfg['region_map']):
        region = cfg['moriches_map']['region']
        frame_params = cfg['moriches_map']['basemap_frame']
        grid = '/home/catherinej/bathymetry/moriches.nc'
        grd = pygmt.grdclip(grid=grid, below=[0.0,-50])
        print(len(fgdata)/2)
        with fig.subplot(nrows=int(len(fgdata)/2), ncols=2, subsize=('15c', '7.5c'), frame='lrtb',
                         autolabel='+jTL+o1.65c/0.5c+gwhite', sharex='b', sharey='l', 
                         margins=['-.5c','-.5c', '.75c', '.75c']):
            for i, key in enumerate(fgdata):
                with fig.set_panel(panel=i):
                    fig.basemap(region=region, projection='M?', frame=frame_params)
                    pygmt.makecpt(cmap='gray', series=[-50,50], reverse=True)
                    fig.grdimage(grid=grd, cmap=True, shading=True)
                    pygmt.makecpt(cmap='lajolla', series=[0,3.0, 0.25],continuous=False, reverse=True)#, truncate='0.2/nan') #, truncate='nan/0.75')

                    # fig.grdview(grid=fg_data['da'], cmap=True)
                    fig.grdimage(grid=fgdata[key]['data'], cmap=True, nan_transparent=True)
                    # pygmt.makecpt(cmap='lajolla', series=[0,3.0,.25], color_model='+c0-3',output='c.cpt')
                    # fig.grdcontour(fg_data['da'], annotation=10, interval='c.cpt', pen=.0025)
                    # for key, data in annot_locs.items():
                    #     fig.text(x=data[0], y=data[1], text=key, font='12p,Helvetic-Bold,black', fill='white')
                    # fig.contour(fg_data['da'])
                    fig.colorbar(position='jBC+o1.65c/.15c+w6/.5h+mc',
                                frame=['xa.5f.25+l"Sea Surface (m)"'],
                                )#'jMR+o-1.75c/0c')
    fig.show()
    fig.savefig(savepath)
                        
# map of gauge locations and fgmax locations for histograms
import yaml
west = pd.read_pickle('/home/catherinej/BarrierBreach/data/plotdata/west_fgmax_points.pkl.gz')
central = pd.read_pickle('/home/catherinej/BarrierBreach/data/plotdata/central_fgmax_pts.pkl.gz')
east = pd.read_pickle('/home/catherinej/BarrierBreach/data/plotdata/east_fgmax_pts.pkl.gz')
surge_locs = np.array([west.columns[1],central.columns[1], east.columns[1]])

 #Custom function to split the tuple into separate columns
def split_tuple(value):
    location, number = value.strip("()").split(", ")
    return location, int(number)

gauge_locs = pd.read_csv('/home/catherinej/BarrierBreach/data/plotdata/plotting_gauges_locs.csv',
                        converters={'Unnamed: 0': split_tuple})
gauge_locs[['location', 'number']] = pd.DataFrame(gauge_locs['Unnamed: 0'].tolist(), index=gauge_locs.index)

# Drop the original "Unnamed: 0" column
gauge_locs.drop('Unnamed: 0', axis=1, inplace=True)
# Remove quotes from the location values
gauge_locs['location'] = gauge_locs['location'].str.strip("'\"")

numbers_to_keep = [84, 82, 133, 45, 11, 119]

gauge_locs = gauge_locs[gauge_locs['number'].astype(int).isin(numbers_to_keep)]
# Define the mapping of locations to values
location_mapping = {
    'west': ['a', 'b'],
    'central': ['a', 'b'],
    'east': ['a', 'b']
}
# Add a new column for the first entry in each location
gauge_locs['new_column'] = ''
for location in location_mapping.keys():
    indices = gauge_locs[gauge_locs['location'] == location].index
    if len(indices) > 0:
        location_values = location_mapping[location]
        for i, index in enumerate(indices):
            value_index = i % len(location_values)
            gauge_locs.at[index, 'new_column'] = location_values[value_index]

with open("/home/catherinej/claw_code/src/claw_code/post/gmt_config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
fig = pygmt.Figure()
with pygmt.config(**cfg['region_map']):

    region = cfg['moriches_map']['region']
    frame_params = cfg['moriches_map']['basemap_frame']
    # frame_params.append(f'+t"Gauge and surge locations"')
    # if name == 'No Breach':
    grid = '/home/catherinej/bathymetry/moriches.nc'
    grd = pygmt.grdclip(grid=grid, below=[0.0,-50])
    dgrid = pygmt.grdgradient(grid=grd, radiance=[100, 0])
    fig.basemap(region = region,
                projection='M20c',
                frame=frame_params)
    pygmt.makecpt(cmap='gray', series=[-50,50], reverse=True)
    fig.grdimage(grid=grd, cmap=True, shading=True)
    
    fig.plot(x=surge_locs[:,1], y=surge_locs[:,0], style='c0.45',pen='black', color='#009E73', label=f'surge location+S0.5c')
    dottext = ['west', 'central', 'east']
    fig.text(text=dottext,x=surge_locs[:,1], y=surge_locs[:,0]+0.005, font='#009E73', fill='white',pen='.25p,black')
    
    
    w = gauge_locs.loc[gauge_locs['location'] == 'west']
    c = gauge_locs.loc[gauge_locs['location'] == 'central']
    e = gauge_locs.loc[gauge_locs['location'] == 'east']
    fig.text(text=w['new_column'], x=w.lon-0.006, y=w.lat, 
             font='#0072B2', fill='white', pen='.25p,black')
    fig.plot(x=w.lon, y=w.lat, style='c0.45c', pen='black', fill='#0072B2',  label='west gauges+S0.5c')
    
    fig.text(text=c['new_column'], x=c.lon-0.006, y=c.lat, 
             font='#D55E00', fill='white', pen='.25p,black')
    fig.plot(x=c.lon, y=c.lat, style='c0.45c', pen='black', fill='#D55E00', label='central gauges+S0.5c')
    
    fig.text(text=e['new_column'], x=e.lon-0.006, y=e.lat, 
             font='#56B4E9', fill='white', pen='.25p,black')
    fig.plot(x=e.lon, y=e.lat, style='c0.45c', pen='black', fill='#56B4E9', label='east gauges+S0.5c')
    
with pygmt.config(FONT_ANNOT_PRIMARY='18p'):
    fig.legend(position='JBR+jBR+o.5c/.45c+w6/3.25',  
              box=True)
fig.show()
fig.savefig('/home/catherinej/BarrierBreach/visualization/dot_map.pdf')

datadir = '/home/catherinej/BarrierBreach/data/fgmax_data/'
max_inundation = 'loc_0247_259_fgmax.nc'
min_inundation = 'dw_0283_1_fgmax.nc'
mean_inundation = 'w_0435_6_fgmax.nc'

minfg = xr.open_dataset(os.path.join(datadir, 'fg_nc', min_inundation))
maxfg = xr.open_dataset(os.path.join(datadir, 'fg_nc', max_inundation))
meanfg = xr.open_dataset(os.path.join(datadir, 'fg_nc', mean_inundation))
no_breach = xr.open_dataset(os.path.join(datadir, 'no_breach_fgmax.nc'))
w_0019_6 = xr.open_dataset(os.path.join(datadir, 'fg_nc', 'w_0019_6_fgmax.nc'))
loc_0213_11 = xr.open_dataset(os.path.join(datadir, 'fg_nc', 'loc_0213_11_fgmax.nc'))
loc_0026_0011 = xr.open_dataset(os.path.join(datadir, 'fg_nc', 'loc_0026_0011_fgmax.nc'))
d_0284_6 = xr.open_dataset(os.path.join(datadir, 'fg_nc', 'd_0284_6_fgmax.nc'))