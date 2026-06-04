# -*- coding: utf-8 -*-
'''
Format lidar data through LIDARGO

Inputs (both hard-coded and available as command line inputs in this order):
    sdate [%Y-%m-%d]: start date in UTC
    edate [%Y-%m-%d]: end date in UTC
    delete [bool]: whether to delete raw data
    path_config: path to general config file
    mode [str]: serial or parallel
'''
import os
cd=os.path.dirname(__file__)
import sys
import warnings
import logging
from datetime import datetime
import yaml
from multiprocessing import Pool
import utils as utl
import glob
import re
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='1970-01-01' #start date
    edate='2070-01-01' #end date
    delete=False #delete input files?
    replace=False #replace existing files?
    path_config=os.path.join(cd,'configs/config_corsair.yaml') #config path
    mode='serial' #serial or parallel
else:
    sdate=sys.argv[1]
    edate=sys.argv[2]
    delete=sys.argv[3]=="True"
    replace=sys.argv[4]=="True"
    path_config=sys.argv[5]
    mode=sys.argv[6]
    
#%% Initalization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#initialize main logger
logfile_main=os.path.join(cd,'log',datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S'))+'_errors.log'
os.makedirs('log',exist_ok=True)
            
#%% Main
for channel in config['channels']:
        
    #format all files within date range
    files=sorted(glob.glob(os.path.join(config['path_data'],channel,'*hpl')))
    files=[f for f in files if (m:=re.search(r'20\d{6}',os.path.basename(f))) is not None and
           datetime.strptime(sdate,'%Y-%m-%d')<=datetime.strptime(m.group(0),'%Y%m%d')<=datetime.strptime(edate,'%Y-%m-%d')]
    save_path_raw=os.path.join(config['path_data'],channel.replace('raw','00'))
    if mode=='serial':
        for f in files:
            utl.format_file(f,save_path_raw,delete,config,logfile_main,replace)
    elif mode=='parallel':
        args = [(files[i],save_path_raw,delete,config,logfile_main+f'.{i}',replace) for i in range(len(files))]
        with Pool() as pool:
            pool.starmap(utl.format_file, args)
        for i in range(len(files)):
            wlog=logfile_main+f'.{i}'
            if os.path.isfile(wlog):
                with open(wlog,'r') as src, open(logfile_main,'a') as dst:
                    dst.write(src.read())
                os.remove(wlog)
    else:
        raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")
   
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)