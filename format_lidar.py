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
import traceback
import warnings
import logging
import lidargo as lg
from datetime import datetime
import yaml
from multiprocessing import Pool
import glob
warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    delete=False #delete input files?
    replace=False #replace existing files?
    path_config=os.path.join(cd,'configs/config_corsair.yaml') #config path
    mode='serial'#serial or parallel
else:
    delete=sys.argv[1]=="True"
    replace=sys.argv[2]=="True"
    path_config=sys.argv[3]
    mode=sys.argv[4]#
    
#%% Initalization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#initialize main logger
logfile_main=os.path.join(cd,'log',datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S'))+'_errors.log'
os.makedirs('log',exist_ok=True)

#%% Functions
def format_file(file,save_path,delete,config,logfile_main,replace):
    try:
        logfile=os.path.join(cd,'log',os.path.basename(file).replace('hpl','log'))
        lproc = lg.Format(file, config=config['path_config_format'], verbose=True,logfile=logfile)
        lproc.process_scan(replace=replace, save_file=True,save_path=save_path_raw)
        
        if delete:
            os.remove(file)
            
    except:
        with open(logfile_main, 'a') as lf:
            lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error formatting file {os.path.basename(file)}: \n")
            traceback.print_exc(file=lf)
            lf.write('\n --------------------------------- \n')
            
#%% Main
for channel in config['channels']:
        
    #format all files
    files=sorted(glob.glob(os.path.join(config['path_data'],channel,'*hpl')))
    save_path_raw=os.path.join(config['path_data'],channel.replace('raw','00'))
    if mode=='serial':
        for f in files:
            format_file(f,save_path_raw,delete,config,logfile_main,replace)
    elif mode=='parallel':
        args = [(files[i],save_path_raw,delete, config,logfile_main,replace) for i in range(len(files))]
        with Pool() as pool:
            pool.starmap(format_file, args)
    else:
        raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")
   
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)