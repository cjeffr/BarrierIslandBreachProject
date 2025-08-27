#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import os
import numpy as np


# In[ ]:


huh


# In[17]:


# def load_existing_breach():
PATH = '/projects/weiszr_lab/catherine/breach_sims/'
breach_data_files = glob.glob(os.path.join(PATH, '**', 'breach.data'), recursive=True)
data_files = [d for d in breach_data_files if not d.split('/')[-2] == '_output']
breach_data = {}
for file in data_files:
    directory = file.split('/')[-2]
    if directory not in ['no_breach', '15m', 'width_depth']:
        with open(file) as f:
            data = f.read()
        data = data.split('\n')
        data = [line.split(' ') for line in data]
        data.pop(0)
        names = data.pop(0)
        if 'depth' not in map(str.lower, names):
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
#     return breach_data


# In[18]:


breach_df = pd.concat(breach_data, axis=0).reset_index(level=0).rename({'level_0':'key'}, axis=1)


# In[21]:


breach_df.to_pickle('breach_data.pkl.gz', compression='gzip')


# In[11]:


if 'Depth' in map(str.lower, names):
    print(names)


# In[20]:


breach_df[breach_df.key.isin(['d1', 'd15'])]


# In[ ]:




