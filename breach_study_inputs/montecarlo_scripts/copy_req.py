import os
import shutil
import re

dirs = [f.path for f in os.scandir(os.getcwd()) if f.is_dir()] #x[0] for x in os.walk(os.getcwd())]
for d in dirs:
    if not d.endswith('req_geoclaw'):
        [shutil.copy(os.path.join('req_geoclaw', f), os.path.join(d, f)) for f in os.listdir('req_geoclaw')]     
