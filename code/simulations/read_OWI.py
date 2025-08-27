import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import re
import xarray as xr
from dataclasses import dataclass
import pandas as pd
import gzip

@dataclass
# Creates a dataclass for holding the OWI data
class StormData:
    iLat: int  # number of latitude points
    iLong: int  # number of longitude points
    DX: float  # resolution in x direction
    DY: float  # resolution in y direction
    SWLat: float  # Initial Latitude point in SW corner
    SWLon: float  # Initial Longitude point in SW corner
    DT: datetime  # datestamp of current wind/pressure array
    matrix: list  # placeholder for wind or pressure array

    def __post_init__(self):
        # Put everything in correct format
        self.iLat = int(self.iLat)
        self.iLong = int(self.iLong)
        self.DX = float(self.DX)
        self.DY = float(self.DY)
        self.SWLat = float(self.SWLat)
        self.SWLon = float(self.SWLon)
        self.DT = datetime.strptime(self.DT, '%Y%m%d%H%M')
        
# class ConvertOWIData():
#     """
#     Takes Oceanweather style data and converts to necessary format to be used in GeoClaw's data
#     derived storm module
#     """
#     def __init__(self):
#         self.storm_data = 


def read_oceanweather(PATH, filename):
    """
    Reads in Oceanweather files and puts them into a dataclass for ease of data cleanup
    """
    fh = os.path.join(PATH, filename)
    subset = None
    all_data = []

    # Open file and use regex matching for parsing data
    with gzip.open(fh, 'rt') as f:
        input = f.readlines()
        print(type(input))
        for line in input:
            if not line.startswith('Oceanweather'):  # Skip the file header
                # Find the header lines containing this pattern of characters
                # Example from file: iLat= 105iLong=  97DX=0.2500DY=0.2500SWLat=22.00000SWLon=-82.0000DT=200007121200
                # \w+: any unicode string
                # \s*: any repetition of whitespace characters
                # \d+: decimal digit of length +=1
                # \.?: matches anything but a new line in minimal fashion
                # \d*: decimal digit with +=0 repetitions
                header = re.findall("\\w+=-?\\s*\\d+\\.?\\d*", line)
                if header:
                    if subset:
                        # put data into dataclass
                        storm_data = StormData(**subset)
                        all_data.append(storm_data)
                    # Split apart the header data into separate values rather than the string
                    subset = {
                        x.replace(' ', '').split('=')[0]: x.replace(' ', '').split('=')[1]
                        for x in header
                    }
                    subset["matrix"] = []
                else:
                    nums = list(map(float, line.split()))
                    subset["matrix"].append(nums)
                storm_data = StormData(**subset)
        all_data.append(storm_data)

    return all_data
class OwiStormData():
    def __init__(self):
        self.wind_data = read_oceanweather(PATH, wind_file)
        self.pressure_data = read_oceanweather(PATH, pressure_file)
        self.time_location_arrays()
    
    def time_location_arrays(self):
        self.timesteps = [] 
        self.lats_per_timestep = []
        self.lons_per_timestep = []
        t0 = None

        for idx, d in enumerate(data):
            if not t0:
                t0 = d.DT
            t = d.DT
            seconds_from_start = (t - t0).total_seconds()
            self.timesteps.append(seconds_from_start)

            self.lats_per_timestep.append([d.SWLat + i * d.DY for i in range(d.iLat)])
            self.lons_per_timestep.append([d.SWLon + i * d.DX for i in range(d.iLong)])
        
    def calculate_velocity_speed(self):
        xvel = []
        yvel = []
        pressure = []
        speed = []
        
        for idx, data in enumerate(self.wind_data):
            winds = [item for sublist in data.matrix for item in sublist]
            pressure = [item for sublist in self.pressure_data[idx].matrix for item in sublist]
            # Declare data arrays
            u = np.empty(shape=(data.iLong, data.iLat))
            v = np.empty(shape=(data.iLong, data.iLat))
            p = np.empty(shape=(data.iLong, data.iLat))

        

def format_data(wind_data, pres_data, STORM_INFO_FILE):
    time_array = []  # list of timesteps
    slat = []  # latitude list for each timestep
    slon = []  # longitude list for each timestep
    uu = []  # list of velocity arrays in x direction
    vv = []  # list of velocity arrays in y direction
    pp = []  # list of pressure arrays
    sp = []  # list of speed arrays
    eye_list = []  # eye location list
    max_wind_radius_list = []
    max_pressure_radius_list = []
    t0 = None

    try:
        df = pd.read_csv(STORM_INFO_FILE)
        file = True
        total_time = wind_data[-1].DT - wind_data[0].DT
        num_steps = int(round((total_time / len(wind_data)).total_seconds() / 60))
        print('num steps:', num_steps)
        storm_df = get_eye_info(df, num_steps)

    except FileNotFoundError:
        file = False

    for idx, data in enumerate(wind_data):
        if not t0:
            t0 = data.DT
        t = data.DT
        seconds_from_start = (t - t0).total_seconds()
        time_array.append(seconds_from_start)

        # Get latitude array
        slat.append([data.SWLat + i * data.DY for i in range(data.iLat)])

        # Get Longitude array
        slon.append([data.SWLon + i * data.DX for i in range(data.iLong)])

        # Get list of wind and pressure values from list of lists
        winds = [item for sublist in data.matrix for item in sublist]
        pressure = [item for sublist in pres_data[idx].matrix for item in sublist]

        # Declare data arrays
        u = np.empty(shape=(data.iLong, data.iLat))
        v = np.empty(shape=(data.iLong, data.iLat))
        p = np.empty(shape=(data.iLong, data.iLat))

        # Fill data arrays with values from wind and pressure lists
        for j in range(data.iLat):
            for i in range(data.iLong):
                u[i, j] = winds[j * data.iLong + i]
                p[i, j] = pressure[j * data.iLong + i]
                v[i, j] = winds[data.iLat * data.iLong + j * data.iLong + i]
        sp.append(np.sqrt(u.T**2 + v.T**2))
        uu.append(u.T)
        pp.append(p.T)
        vv.append(v.T)
        if file:
            eye_lon = storm_df['Storm Longitude (deg)'][idx]
            eye_lat = storm_df['Storm Latitude (deg)'][idx]
            wind_radius = storm_df['Radius Max Winds (km)'][idx]
            pressure_radius = storm_df['Radius Pressure 1 (km)'][idx]
        else:
            lon_idx, lat_idx = find_eye(np.sqrt(u.T**2 + v.T**2))
            eye_lon = slon[0][lon_idx]
            eye_lat = slat[0][lat_idx]
        eye_list.append([eye_lon, eye_lat])
        max_wind_radius_list.append(wind_radius)
        max_pressure_radius_list.append(300000)

    # convert list of arrays to arrays for entry into dataset
    windx = np.array(uu)
    windy = np.array(vv)
    pressure = np.array(pp)*100
    speed = np.array(sp)
    eyes = np.array(eye_list)
    max_pressure_radius = np.array(max_pressure_radius_list)
    max_wind_radius = np.array(max_wind_radius_list)*1000

    print(eyes.shape, windx.shape)

    # Create dataset includes wind in x & y directions, pressure, speed, and eye location (if applicable)
    ds = xr.Dataset(data_vars={'u': (('time', 'lat', 'lon'), windx),
                               'v': (('time', 'lat', 'lon'), windy),
                               'speed': (('time', 'lat', 'lon'), speed),
                               'pressure': (('time', 'lat', 'lon'), pressure),
                               'eye_loc': (('time', 'loc'), eyes),
                               'mwr': ('time', max_wind_radius),
                               'storm_radius':('time', max_pressure_radius)},
                    coords={'lat': slat[0],
                            'lon': slon[0],
                            'time': time_array})

    return ds


def find_eye(speed_array):
    """
    Find the location of the eye of the storm using the wind speed gradient
    """
    row_grad, col_grad = np.gradient(speed_array)
    lon_idx = np.unravel_index(np.argmax(row_grad), row_grad.shape)[1]
    lat_idx = np.unravel_index(np.argmax(col_grad), col_grad.shape)[0]

    return lon_idx, lat_idx

def eye_from_pressure(pressure_data):
    # Define the center as the minimum pressure point
    center = np.unravel_index(np.argmin(pressure_data), pressure_data.shape)
    return center[1], center[0]


def get_eye_info(df, num_steps):
    """
    Load dataframe of Storm data resample and interpolate to match wind/pressure data
    """
    storm_number = int(storm_id.split('_')[1])
    print(storm_number)
    df_storm = df[df['Storm ID'] == storm_number]
    df_storm['Time'] = [0 + x for x in range(len(df_storm))]
    df_storm = df_storm.drop('yyyymmddHHMM', axis=1)
    df_storm.Time = pd.to_timedelta(df_storm.Time, unit='h')
    df_storm = df_storm.set_index(df_storm.Time)
    res = df_storm.resample(f'{num_steps}T').mean()
    print('len res', len(res))
    new_res = res.interpolate(method='time')
    new_df = new_res.reset_index()
    print(len(new_df))
    return new_df

if __name__ == '__main__':
    
    PATH =  '486' #'/Users/catherinej/projects/storms/data/storms_for_jen/' 
    STORM_INFO_FILE = os.path.join(PATH,  'NACCS_TS_Sim0_Post0_ST_TROP_STcond.csv')
    
    storm_files = {}
    fldr_list = [f.path for f in os.scandir(PATH) if f.is_dir()]
    for fldr in fldr_list:
        # Get list of files in the directory and split them by name and extension
        file_list = [[fh, ext] for f, fh, ext, _  in ((f, *os.path.basename(f).split('.'))
                                                      for f in os.listdir(fldr) if not f.startswith('.'))]
        pre_files = [f for f in file_list if 'pre' in  f[1]]
        wind_files = [f for f in file_list if 'w' in f[1].lower() ]
        dir_name = os.path.split(fldr)
        file_entry = {}
        for f in wind_files:

            if f[0] in [row[0] for row in pre_files]:
                file_entry[f[0]] =  {'pressure_file': f'{f[0]}.pre.gz',
                                     'wind_file': f[0] + f'.{f[1]}.gz'}
            storm_files[dir_name[1]] = file_entry


    for fldr in fldr_list:
        storm_id = os.path.split(fldr)[1]


    for filename in storm_files[storm_id]:
        print(fldr, filename)
        #tmp_p = os.path.join(fldr, storm_files[storm_id][filename]['pressure_file'])
        #tmp_w = os.path.join(fldr, storm_files[storm_id][filename]['wind_file'])
        wind_data = read_oceanweather(fldr, storm_files[storm_id][filename]['wind_file'])
        pres_data = read_oceanweather(fldr, storm_files[storm_id][filename]['pressure_file'])
        ds = format_data(wind_data, pres_data, STORM_INFO_FILE)
        ds.to_netcdf(f'{filename}_300km.nc')
    # for speed in ds.speed:
    #     speed.plot(add_colorbar=False)
    #     plt.show()



