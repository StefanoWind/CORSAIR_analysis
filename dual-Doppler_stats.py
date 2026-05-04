# -*- coding: utf-8 -*-
'''
Reconstruction of dual-Doppler statistics

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
from datetime import datetime
import yaml
from multiprocessing import Pool
import utils as utl
import xarray as xr
import numpy as np
import logging
import glob
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
    sdate='2026-04-14' #start date
    edate='2026-04-15' #end date
    delete=False #delete input files?
    replace=False #replace existing files?
    path_config=os.path.join(cd,'configs/config_corsair.yaml') #config path
    mode='serial'#serial or parallel
else:
    sdate=sys.argv[1]
    edate=sys.argv[2] 
    delete=sys.argv[3]=="True"
    replace=sys.argv[4]=="True"
    path_config=sys.argv[5]
    mode=sys.argv[6]#
    
#other inputs
max_time_diff=np.timedelta64(10,'s') #maximum time difference between synchronized files
    
#%% Initalization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#initialize main logger
logfile_main=os.path.join(cd,'log',datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S'))+'_errors.log'
os.makedirs('log',exist_ok=True)

#%% Main

#apply LiSBOA on indivudual files
for channel in config['channels_dual-doppler']:
        
    #standardize all files within date range
    files=sorted(glob.glob(os.path.join(config['path_data'],channel,'*nc')))
    if mode=='serial':
        for f in files:
              utl.lisboa_file(f,config['path_config_dual-doppler'],logfile_main,sdate,edate,delete,replace)
    elif mode=='parallel':
        args = [(files[i], config['path_config_dual-doppler'],logfile_main,sdate,edate,delete,replace) for i in range(len(files))]
        with Pool() as pool:
            pool.starmap(utl.lisboa_file, args)
    else:
        raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")

#list all LiSBOA files and dates
files={}
dates={}
for channel in config['channels_dual-doppler']:
        
    #standardize all files within date range
    files[channel]=sorted(glob.glob(os.path.join(config['path_data'],
                '.'.join(channel.split('.')[:-1])+'.c1','*nc')))
    
    dates[channel]=np.array([utl.date_from_file(f) for f in files[channel]])
   
#dual-Doppler reconstruction
for f1,d1 in zip(files[config['channels_dual-doppler'][0]],dates[config['channels_dual-doppler'][0]]):
    
    #find matching file
    time_diff=d1-dates[config['channels_dual-doppler'][1]]
    match=np.where(np.abs(time_diff)<max_time_diff)[0]
    
    if len(match)==1:
        
        #load matching data
        f2=files[config['channels_dual-doppler'][1]][match[0]]
        Data1=xr.open_dataset(f1)
        Data2=xr.open_dataset(f2)
        
        #compose filename
        save_path=os.path.join(config['path_data'],Data1.attrs['config_channel_name'], 
                               Data1.attrs['config_channel_name'].split('/')[-1]+f1.split('.c1')[-1])
        
        #wind map reconstruction
        Output=utl.dual_doppler_reconstruction(Data1=Data1,
                                               Data2=Data2,
                                               sigma_rws=Data1.attrs['config_sigma_rws'],
                                               sigma_w=Data1.attrs['config_sigma_w'],
                                               logfile_main=logfile_main,
                                               sdate=sdate,
                                               edate=edate,
                                               replace=replace,
                                               save_path=save_path)
        #plot wind map
        if Output is not None:
            utl.plot_wind_map(Output,
                              max_sigma=config['max_sigma_map'],
                              heights=Data1.attrs['config_plot_heights'],
                              save_path=save_path.replace('.nc','.wind_map.png'),
                              stride=Data1.attrs['config_stride_map'],
                              path_layout=config['path_layout'],
                              markers = config['markers_map'])
    else:
        continue
        
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
        
