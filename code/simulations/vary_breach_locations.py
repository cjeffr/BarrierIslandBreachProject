import pandas as pd
import numpy as np
import sys
import os
import glob
import breach_randomization as br
import re
import shutil


class RandomBreach(): 
    def __init__(self, breach_data_path, gauge_location_path, gauge_data_path, topo_path, masked_path): 
        self.breach_data_path = breach_data_path
        self.gauge_location_path = gauge_location_path
        self.gauge_data_path = gauge_data_path
        self.topo_path = topo_path
        self.topo_data = br.load_topography(self.topo_path)
        self.masked_island = masked_path
        self.gauge_names = self.load_gauge_names()
        self.gauge_data = br.load_gauge_data(self.gauge_names.index.values, self.gauge_data_path)
        self.breach_data = self.load_existing_breach()    
    
        
    def load_gauge_names(self):
        df = pd.read_csv(self.gauge_location_path)
        df = df.drop(['Unnamed: 0', 'dist'], axis=1)
        return df
    
    
    def load_existing_breach(self):
        breach_data_files = glob.glob(os.path.join(self.breach_data_path, '**', 'breach.data'), recursive=True)
        data_files = [d for d in breach_data_files if not d.split('/')[-2] == '_output']
        breach_data = {}
        for file in data_files:
            directory = file.split('/')[-2]
            if directory not in ['no_breach', '15m']:
                with open(file) as f:
                    data = f.read()
                data = data.split('\n')
                data = [line.split(' ') for line in data]
                data.pop(0)
                names = data.pop(0)
                names.append('Depth')
                d = {k: v for k,v in zip(names, data) if k != 'sigma,'}
                df = pd.DataFrame(d)
                df.columns = [col.replace(',', '') for col in df.columns]
                df = df.apply(pd.to_numeric, errors='ignore')
                df.columns = [x.title() for x in df.columns]
                dist = [abs(west - east) for west, east
                        in zip(df['West'], df['East'])]
                df['Distance'] = dist
                breach_data[directory] = df
        return breach_data
        
        
    def check_location_viability(self, max_dune):
        """Checks to make sure that a breach location reaches the required minimum % of dune height

        Args:
            df (_dataframe_): data frame of all tide gauge timeseries
            max_dune (_type_): max dune height at chosen breach location
        """
        breach_time = []
        x_percent = max_dune * .24
        cols_greater = (self.gauge_data >= x_percent).any()
        if cols_greater.any():
            time_to_exceed_x_percent = [br.first_greater(self.gauge_data, x_percent, col) for col in cols_greater.index]
            breach_start = min([x for x in time_to_exceed_x_percent if type(x) == np.float64])
            if breach_start == 27000.0:
                print('these columns suck:', cols_greater, 'dune height is: ', x_percent)
            breach_stop = breach_start + 7200.0
            return breach_start, breach_stop
        else:
            print('This location is not viable')
            return False, False
            # WRite location to file and break to restart randomize location?
        
        
    def randomize_location(self):
        """Takes a collection of breach data and randomizes the locations

        Args:
            df (pandas dataframe): breach data, location, width, depth, timing
        """
        bad_breach = []
        breach_loc = br.get_random_location(self.masked_island)
        lat = (breach_loc['south'][1] + breach_loc['north'][1])/2
        
        max_dune = br.max_dune_height(self.topo_data, breach_loc['south'][0], breach_loc['north'][0],
                                    breach_loc['lon'][0])
        if max_dune == 0.0:
            print('Why is the dune at 0.0?,', breach_loc)

        tide_gauges = br.find_nearest_gauges(self.gauge_names, breach_loc['lon'][1], lat,  1000)
        
        
        breach_start, breach_stop = self.check_location_viability(max_dune)
        if breach_start:
            # print(breach_start, 'something isnt false')
            new_breach_data = {'south' : breach_loc['south'][1],
                            'north': breach_loc['north'][1],
                            'mu': breach_loc['lon'][1],
                            'start': breach_start,
                            'stop': breach_stop,
                            'bad_breach': bad_breach
                            }
            # print(new_breach_data)
            return new_breach_data
            # do more stuff
        else:
            bad_breach.append(breach_loc)
            return self.randomize_location()
        
    
    def arrange_data(self, new_breach, key):
        south = [x['south'] for x in new_breach]
        north = [x['north'] for x in new_breach]
        mu = [x['mu'] for x in new_breach]
        start = [x['start'] for x in new_breach]
        stop = [x['stop'] for x in new_breach]
        west = [x['mu'] - y/2 for x, y in zip(new_breach, self.breach_data[key]['Distance'])]
        east = [x['mu'] + y/2 for x, y in zip(new_breach, self.breach_data[key]['Distance'])]
        data = {'South': south,
                'North': north,
                'Mu': mu,
                'Start_Time': start,
                'End_Time': stop,
                'West': west,
                'East': east}
        return data
    
    
    def combine_data(self, key, data):
        df = pd.DataFrame(data)
        self.breach_data[key] = self.breach_data[key].drop(columns=list(data.keys()))
        self.breach_data[key] = pd.concat([self.breach_data[key], df], axis=1)
        
    
    
    def write_breach_data(self, num_breaches, write_path, key):
        comment_str = 'breach_trigger, south, north, west, east, mu, sigma, time_factor, start_time, end_time depth'
        write_order = ['Breach_Trigger', 'South', "North", 'West', 'East',
                       'Mu', 'Sigma', 'Time_Factor', 'Start_Time', 'End_Time', 
                       'Depth']
        with open(os.path.join(write_path, 'breach.data'), 'w') as f:
            f.write(f'{num_breaches}' + '\n')
            f.write(comment_str + '\n')
            for param in write_order:
                if param == 'Sigma':
                    f.write(f'1' + '\n')
                else:
                    f.write(' '.join(map(str, self.breach_data[key][param])) + '\n')
                    
def runit():
    num_breaches = 6
    randb = RandomBreach('/home/catherinej/width_depth',
                         '/home/catherinej/BarrierBreach/data/ocean_gauges.csv',
                         '/home/catherinej/width_depth/no_breach/_output',
                         '/home/catherinej/bathymetry/moriches.nc',
                         '/home/catherinej/BarrierBreach/data/')
    
    for key in randb.breach_data:
        sim = re.split(r'(\d+)', key)
        breach = randb.breach_data[key]
        new_breach = [randb.randomize_location() for row in breach.Breach_Trigger]
        data = randb.arrange_data(new_breach, key)
        randb.combine_data(key, data)
        sim_dir =  f'{sim[0]}_rand_{sim[1]}'
        if not os.path.exists(sim_dir):
            os.mkdir(sim_dir)
        randb.write_breach_data(num_breaches, sim_dir, key)
         
        
        

if __name__ == '__main__':
    runit()