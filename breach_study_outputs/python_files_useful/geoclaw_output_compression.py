import pandas as pd
import read_amr
import os
import glob
import subprocess
import shutil
import tarfile


ABS_PATH = '/projects/weiszr_lab/catherine/486/depth_width_sensitivity'
fldr = [ 'vary_depth']

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        

num_sims = 16
for folder in fldr:
    SOURCE_DIR = os.path.join(ABS_PATH, folder)
    for i in range(2, num_sims + 1):
        if folder == 'vary_width':
            i = 16
        print(folder, i)
        path = os.path.join(SOURCE_DIR, f'{i}', '_output')
        new_output_fldr = os.path.join(ABS_PATH, f'{folder}_{i}__output')
        if not os.path.exists(new_output_fldr):
            os.mkdir(new_output_fldr)
        
        
        fort_files = glob.glob(os.path.join(path, 'fort.q*'))
        gauge_files = glob.glob(os.path.join(path, 'gauge1*.txt'))
        for file in gauge_files:
            shutil.copy(file, new_output_fldr)
        shutil.copy(os.path.join(path, 'fgmax_grids.data'), new_output_fldr)
        shutil.copy(os.path.join(path, 'fgmax0001.txt'), new_output_fldr)
        
        num_output = len(fort_files)
        for j in range(num_output):
            data_df = read_amr.ReadAmr(path, j).pandas_dataframe
            out_file = os.path.join(new_output_fldr, f'frame_{j}.pkl.gz')
            data_df.to_pickle(out_file, compression='gzip')
        
        make_tarfile(f'{folder}_{i}', new_output_fldr)
