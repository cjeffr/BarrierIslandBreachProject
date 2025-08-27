import os
import subprocess
import shutil
import sensitivity_study as ss

num_sims = 16
num_bools = 2

for i in range(num_bools):
    if i == 0:
        for j in range(1, num_sims+1, 1):
            dir_name = os.path.join('./', 'vary_depth')
            sim_dir = os.path.join(dir_name, f'{j}')
            data_dir = 'req_geoclaw/'
            if not os.path.exists(sim_dir):
                shutil.copytree(data_dir, sim_dir)
            ss.runit(sim_dir, vary_depth=True)
        
    elif i == 1:
        for j in range(1, num_sims+1, 1):
            dir_name = os.path.join('./', 'vary_width')
            sim_dir = os.path.join(dir_name, f'{j}')
            data_dir = 'req_geoclaw'
            if not os.path.exists(sim_dir):
                shutil.copytree(data_dir, sim_dir)
            ss.runit(sim_dir, vary_width=True)

        
