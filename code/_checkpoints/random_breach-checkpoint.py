import pandas as pd
import numpy as np
import sys
import os
import glob
from clawpack.geoclaw import topotools
import breach_randomization as br
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
        #self.breach_data = self.load_existing_breach()    
    
        
    def load_gauge_names(self):
        df = pd.read_csv(self.gauge_location_path)
        df = df.drop(['Unnamed: 0', 'dist'], axis=1)
        return df
    
    
    def load_existing_breach(self):
        breach_data_file = os.path.join(self.breach_data_path,  'breach.data')
        with open(breach_data_file) as f:
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
        return df
        
        
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
        
    def randomize_width(self, new_breach_data):
        lat = (new_breach_data['south'] + new_breach_data['north'])/2
        lon = new_breach_data['mu']
        east, west = br.random_breach_width(lat, lon)
        new_breach_data['west'] = west
        new_breach_data['east'] = east
        return new_breach_data
    
    def randomize_depth(self, new_breach):
        depth = br.get_depth()
        new_breach['depth'] = depth
        return new_breach
    
    def arrange_data(self, new_breach):
        south = [x['south'] for x in new_breach]
        north = [x['north'] for x in new_breach]
        mu = [x['mu'] for x in new_breach]
        start = [x['start'] for x in new_breach]
        stop = [x['stop'] for x in new_breach]
        west = [x['west'] for x in new_breach]
        east = [x['east'] for x in new_breach]
        depth = [x['depth'] for x in new_breach]
        time_factor = np.ones_like(new_breach) * 0.002
        breach_trigger = np.ones_like(new_breach)
        data = {'South': south,
                'North': north,
                'Mu': mu,
                'Start_Time': start,
                'End_Time': stop,
                'West': west,
                'East': east,
                'Breach_Trigger': breach_trigger,
                'Time_Factor': time_factor,
                'Depth': depth}
        return data
    
    
    def combine_data(self, data):
        df = pd.DataFrame(data)
        #print(self.breach_data)
        self.breach_data = df #self.breach_data.drop(columns=list(data.keys()))
        #self.breach_data = pd.concat([self.breach_data, df], axis=1)
        
    
    
    def write_breach_data(self, num_breaches, write_path):
        comment_str = 'breach_trigger, south, north, west, east, mu, sigma, time_factor, start_time, end_time, depth'
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
                    f.write(' '.join(map(str, self.breach_data[param])) + '\n')
                    

def runit(sim_num):
    
    PATH = '/projects/weiszr_lab/catherine'
    breach_data_path = os.path.join(PATH, 'breach_sims/width_depth/d0048')
    gauge_location_path = os.path.join(PATH, 'ocean_gauges.csv')
    gauge_data_path = os.path.join(PATH, 'breach_sims/reference_gauges')
    topo_path = os.path.join(PATH, 'bathymetry/moriches.nc')
    masked_island_path = os.path.join(PATH, 'data') 
    randb = RandomBreach(breach_data_path, 
                         gauge_location_path,
                         gauge_data_path,
                         topo_path,
                         masked_island_path)
    
    #for i in range(1, num_sims +1, 1):
        #breach = randb.breach_data
    num_breaches = br.random_num_breaches()
    new_breach = [randb.randomize_location() for b in range(num_breaches)] #row in breach.Breach_Trigger]
    new_breach = [randb.randomize_depth(x) for x in new_breach]
    new_breach = [randb.randomize_width(x) for x in new_breach]
    data = randb.arrange_data(new_breach)
    randb.combine_data(data)
    write_path = f'r{sim_num:04}'
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    randb.write_breach_data(num_breaches, write_path)
    copy_source = '../req_geoclaw'
    for file in os.listdir(copy_source):
        shutil.copy(os.path.join(copy_source, file), os.path.join(write_path, file))
        
if __name__ == '__main__':
    import sys
    sim_num = int(sys.argv[-1])
    runit(sim_num)
