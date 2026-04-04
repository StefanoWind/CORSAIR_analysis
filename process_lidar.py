# -*- coding: utf-8 -*-
'''
Format and standardize/qc of lidar data through LIDARGO

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
import lidargo as lg
from lisboa import statistics as stats
import socket
import getpass
from datetime import datetime
import yaml
import xarray as xr
import numpy as np
from multiprocessing import Pool
from utils import load_lisboa_config,visualization
import re
import logging
import glob
warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='2026-01-07' #start date
    edate='2026-10-08' #end date
    delete=False #delete input files?
    replace=False #replace existing files?
    path_config=os.path.join(cd,'configs/config_crosswind.yaml') #config path
    mode='serial'#serial or parallel
else:
    sdate=sys.argv[1]
    edate=sys.argv[2] 
    delete=sys.argv[3]=="True"
    replace=sys.argv[4]=="True"
    path_config=sys.argv[5]
    mode=sys.argv[6]#
    
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
            
def standardize_file(file,save_path_stand,config,logfile_main,sdate,edate):
    date=re.search(r'\d{8}.\d{6}',file).group(0)[:8]
    if datetime.strptime(date,'%Y%m%d')>=datetime.strptime(sdate,'%Y-%m-%d') and datetime.strptime(date,'%Y%m%d')<=datetime.strptime(edate,'%Y-%m-%d'):
        try:
            logfile=os.path.join(cd,'log',os.path.basename(file).replace('nc','log'))
            lproc = lg.Standardize(file, config=config['path_config_stand'], verbose=True,logfile=logfile)
            lproc.process_scan(replace=False, save_file=True, save_path=save_path_stand)
        except:
            with open(logfile_main, 'a') as lf:
                lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error standardizing file {os.path.basename(file)}: \n")
                traceback.print_exc(file=lf)
                lf.write('\n --------------------------------- \n')

def lisboa_file(file,save_path_stats,config_path,logfile_main,sdate,edate,delete,replace):
    date=re.search(r'\d{8}.\d{6}',file).group(0)[:8]
    if datetime.strptime(date,'%Y%m%d')>=datetime.strptime(sdate,'%Y-%m-%d') and datetime.strptime(date,'%Y%m%d')<=datetime.strptime(edate,'%Y-%m-%d'):
        try:
            logfile=os.path.join(cd,'log',os.path.basename(file).replace('nc','log'))
         
            #load config
            config,config_lisboa=load_lisboa_config(config_path,file)
            save_path=file.replace(config['data_level_in'],config['data_level_out'])
            if not os.path.isfile(save_path) or replace:
                
                if config is not None:
                    #load data
                    Data=xr.open_dataset(file)
                    time=Data.time
                    Data=Data.where(Data.qc_wind_speed==0)
                    
                    if len(config['Dn0'])==3:
                        x_exp=[Data.x.values.ravel(),Data.y.values.ravel(),Data.z.values.ravel()]
                    else:
                        x_exp=[Data.x.values.ravel(),Data.y.values.ravel()]
                    
                    #de-projection
                    proj=Data.x/Data.range
                    f=(Data.wind_speed/proj).values.ravel()
                    
                    #thresholding
                    f[f<config['u_limits'][0]]=np.nan
                    f[f>config['u_limits'][1]]=np.nan
                    
                    #run LiSBOA
                    lproc=stats.statistics(config_lisboa,logfile=logfile)
                    grid,Dd,excl,avg,hom=lproc.calculate_statistics(x_exp,f,2)
                    avg[avg<config['u_limits'][0]]=np.nan
                    avg[avg>config['u_limits'][1]]=np.nan
                    
                    #% output
                    Output=xr.Dataset()
                    if len(grid)==3:
                        coords={'x':grid[0],'y':grid[1],'z':grid[2]}
                    else:
                        coords={'x':grid[0],'y':grid[1]}
                    Output['u_avg']=xr.DataArray(avg,coords=coords,
                                                 attrs={'units':'m/s','description':'mean streamwise velocity'})
                    Output['u_std']=xr.DataArray(hom**0.5,coords=coords,
                                                 attrs={'units':'m/s','description':'std of streamwise velocity'})

                    Output.attrs['start_time']=str(time.isel(beamID=0,scanID=0).values)
                    Output.attrs['end_time']=  str(time.isel(beamID=-1,scanID=-1).values)
                    
                    for c in config:
                        Output.attrs[f'config_{c}']=config[c]
                    
                    Output.attrs["data_level"]=config['data_level_out']
                    Output.attrs['input_source']=os.path.basename(file)
                    Output.attrs["contact"]= "stefano.letizia@nrel.gov"
                    Output.attrs["institution"]= "NLR"
                    Output.attrs["description"]= "Statistics of de-projected wind speed calculated through LiSBOA"
                    Output.attrs["reference"]= "Letizia et al. LiSBOA (LiDAR Statistical Barnes Objective Analysis) for optimal design of lidar scans and retrieval of wind statistics – Part 1: Theoretical framework. AMT, 14, 2065–2093, 2021, 10.5194/amt-14-2065-2021"
                    Output.attrs["history"]= (
                        f"Generated by {getpass.getuser()} on {socket.gethostname()} on "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using {os.path.basename(sys.argv[0])}"
                    )
                    Output.attrs["code1"]="https://github.com/NREL/FIEXTA/tree/main/lisboa"
                    Output.attrs["code2"]="https://github.com/StefanoWind/awaken_lidar_processing" 
                    
                    os.makedirs(os.path.dirname(save_path),exist_ok=True)
                    Output.to_netcdf(save_path)
                    
                    visualization(Output,config,save_path)
                    
                    if delete:
                        os.remove(file)
        except:
            with open(logfile_main, 'a') as lf:
                lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error processing file {os.path.basename(file)}: \n")
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
        
    #standardize all files within date range
    files=sorted(glob.glob(os.path.join(config['path_data'],channel.replace('raw','a0'),'*a0*user5*nc')))
    save_path_stand=os.path.join(config['path_data'],channel.replace('raw','b0'))
    if mode=='serial':
        for f in files:
              standardize_file(f,save_path_stand,config,logfile_main,sdate,edate)
    elif mode=='parallel':
        args = [(files[i],save_path_stand, config,logfile_main,sdate,edate) for i in range(len(files))]
        with Pool() as pool:
            pool.starmap(standardize_file, args)
    else:
        raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")
    
    #apply lisboa all files
    files=sorted(glob.glob(os.path.join(config['path_data'],channel.replace('raw','b0'),'*b0*user5*nc')))
    if mode=='serial':
        for f in files:
              lisboa_file(f,None,config['path_config_lisboa'],logfile_main,sdate,edate,delete,replace)
    elif mode=='parallel':
        args = [(files[i],None, config['path_config_lisboa'],logfile_main,sdate,edate,delete,replace) for i in range(len(files))]
        with Pool() as pool:
            pool.starmap(lisboa_file, args)
    else:
        raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")
          
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
        
