# -*- coding: utf-8 -*-
'''
Calculate wind profiled from sxi-beam
'''
import os
cd=os.path.dirname(__file__)
import sys
import warnings
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool
import logging
import matplotlib.pyplot as plt
import re
from utils import _git_hash
import matplotlib.dates as mdates
from scipy import interpolate
from datetime import datetime
import glob
import xarray as xr
import numpy as np
import socket
import getpass
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='2026-01-01' #start date
    edate='2026-12-31' #end date
    delete=False #delete source files?
    replace=False #replace existing files"
    path_config=os.path.join(cd,'configs/config_corsair.yaml') #config path
    mode='serial' #processing mofe (serial or parallel)
    
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

#list of days to process
days=np.arange(np.datetime64(sdate+'T00:00:00'),np.datetime64(edate+'T00:00:00')+np.timedelta64(1,'D'),np.timedelta64(1,'D'))

#%% Functions
def cosd(x):
    '''
    Cosine in degrees
    '''
    return np.cos(x/180*np.pi)

def sind(x):
    '''
    Sine in degrees
    '''
    return np.sin(x/180*np.pi)

def vstack(a,b):
    '''
    Stack vertically vectors
    '''
    if len(a)>0:
        ab=np.vstack((a,b))
    else:
        ab=b
    return ab   

def interp_gap(xs,x,y,max_gap=1):
    '''
    Interpolate with max allowable data gap contraint
    '''
    
    #interpolate
    real=~np.isnan(x+y)
    f_int=interpolate.interp1d(x[real],y[real],bounds_error=False)
    y_int=f_int(xs)
    
    #find local gap
    f_gap=interpolate.interp1d(x[real],x[real],bounds_error=False,kind='nearest')
    gap=np.abs(f_gap(xs)-xs)
    
    #enforce gap limit
    y_int[gap>max_gap]=np.nan
    
    return y_int
    

def wind_retrieval(files, config, lidar_height, save_path, delete, replace):
    '''
    Wind reconstruction for list of files
    '''
    if len(files)==0:
        return

    #zeroing
    U=[]
    V=[]
    W=[] 
    uu=[]
    vv=[]
    ww=[]
    uv=[]
    uw=[]
    vw=[]
    time=np.array([],dtype='datetime64')
    
    #file naming
    files=sorted(files)
    match = re.match(r"^(.*\D)(\d{8}\.\d{6})(.*)$", os.path.basename(files[0]))
    filename=f'{match.group(1)}{str(match.group(2))[:8]}.000000{match.group(3)}'.replace('b0','c1')
    
    if save_path==None:
        save_path=os.path.dirname(files[0]).replace('b0','c1')
    os.makedirs(save_path,exist_ok=True)
    
    #initialize logger
    logfile=os.path.join(cd,'log',filename.replace('nc','log'))
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    os.makedirs('log',exist_ok=True)
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    #check file existance
    if os.path.isfile(os.path.join(save_path,filename)) and replace==False:
        logging.info(f'File {filename} already exists, skipping')
        return
    
    #define height
    height=np.arange(config['min_height'],config['max_height']+config['height_step'],config['height_step'])
        
    #loopg throigh files
    for f in files:
        logging.info(f'Processing {os.path.basename(f)}')
        data=xr.open_dataset(f)
        Nb=len(data.beamID)
        if Nb != 6:
            logging.error(f'Expected 6 beams, found {Nb} in {os.path.basename(f)}, skipping')
            continue

        #cartesian angles
        alpha=(90-data.azimuth.mean(dim='scanID').values)%360
        beta=data.elevation.mean(dim='scanID').values
        
        #average scan time
        time_start=data.time.isel(scanID=0,beamID=0).values
        time_end=data.time.isel(scanID=-1,beamID=-1).values
        time_avg=time_start+(time_end-time_start)/2
        
        #average and interpolate RWS
        z=data.z.values+lidar_height
        rws_avg=data.wind_speed.where(data.qc_wind_speed==0).mean(dim='scanID').values
        
        #interpolate RWS at nominal height
        rws_avg_int=np.zeros((len(height),Nb))
        for i in range(Nb):
            rws_avg_int[:,i]=interp_gap(height, z[:,i], rws_avg[:,i],max_gap=config['max_gap'])
        nan_beams=np.where(np.isnan(rws_avg_int).any(axis=0))[0].tolist()
        if nan_beams:
            logging.warning(f'Mean RWS interpolation failed for beams {nan_beams} in {os.path.basename(f)}')

        #velocity vector
        A=[]
        for a,b in zip(alpha,beta):
            A=vstack(A,np.array([cosd(b)*cosd(a),cosd(b)*sind(a),sind(b)]))
        A_inv=np.linalg.pinv(A)
            
        vel_vector=A_inv@rws_avg_int.T
        if np.isnan(vel_vector).any():
            logging.warning(f'Mean wind reconstruction produced NaN at {np.isnan(vel_vector).sum()} height(s) in {os.path.basename(f)}')

        #store wind velocity
        U=vstack(U,vel_vector[0,:])
        V=vstack(V,vel_vector[1,:])
        W=vstack(W,vel_vector[2,:])
        time=np.append(time,time_avg)
        
        #variance
        # rws_var includes measurement noise; noise-subtraction is not applied [Sathe et al., 2015]
        rws_var=data.wind_speed.where(data.qc_wind_speed==0).var(dim='scanID').values
        rws_var_int=np.zeros((len(height),Nb))
        for i in range(Nb):
            rws_var_int[:,i]=interp_gap(height, z[:,i], rws_var[:,i],max_gap=config['max_gap'])
        nan_beams=np.where(np.isnan(rws_var_int).any(axis=0))[0].tolist()
        if nan_beams:
            logging.warning(f'Variance RWS interpolation failed for beams {nan_beams} in {os.path.basename(f)}')
            
        #reynolds stresses
        B=[]
        for a,b in zip(alpha,beta):
            B=vstack(B,
            [cosd(b)**2*cosd(a)**2,
             cosd(b)**2*sind(a)**2, 
             sind(b)**2,
             2*cosd(b)**2*cosd(a)*sind(a),  
             2*cosd(b)*sind(b)*cosd(a),
             2*cosd(b)*sind(b)*sind(a)])               
        
        B_inv=np.linalg.pinv(B)
        RS=B_inv@rws_var_int.T
        if np.isnan(RS).any():
            logging.warning(f'Reynolds stress reconstruction produced NaN at {np.isnan(RS).sum()} height(s) in {os.path.basename(f)}')

        uu=vstack(uu,RS[0,:])
        vv=vstack(vv,RS[1,:])
        ww=vstack(ww,RS[2,:])
        uv=vstack(uv,RS[3,:])
        uw=vstack(uw,RS[4,:])
        vw=vstack(vw,RS[5,:])
        
        if delete:
            os.remove(f)
        
    #output
    Output=xr.Dataset()
    Output['U']=xr.DataArray(data=U,coords={'time':time,'height':height},
                             attrs={'units':'m/s','description':'average W-E wind component'})
    Output['V']=xr.DataArray(data=V,coords={'time':time,'height':height},
                             attrs={'units':'m/s','description':'average S-N wind component'})
    Output['W']=xr.DataArray(data=W,coords={'time':time,'height':height},
                             attrs={'units':'m/s','description':'average vertical wind component'})
    
    Output['WS']=(Output['U']**2+Output['V']**2)**0.5
    Output['WS'].attrs={'units':'m/s','description':'average horizontal wind speed'}
    
    Output['WD']=(270-np.degrees(np.arctan2(Output['V'],Output['U'])))%360
    Output['WD'].attrs={'units':'degrees','description':'average horizontal wind direction (0=N, 90=E)'}
    
    Output['uu']=xr.DataArray(data=uu,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'W-E velocity variance'})
    Output['vv']=xr.DataArray(data=vv,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'S-N velocity variance'})
    Output['ww']=xr.DataArray(data=ww,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'vertical velocity variance'})
    Output['uv']=xr.DataArray(data=uv,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'horizontal (W-E to S-N) Reynolds stress'})
    Output['uw']=xr.DataArray(data=uw,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'vertical Reynolds stress in W-E direction'})
    Output['vw']=xr.DataArray(data=vw,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'vertical Reynolds stress in S-N direction'})
    
    Output['tke']=xr.DataArray(data=(uu+vv+ww)/2,coords={'time':time,'height':height},
                               attrs={'units':'m^2/s^2','description':'turbulence kintic energy'})
    
    uu_rot= Output['uu']*cosd(270-Output['WD'])**2+\
          2*Output['uv']*cosd(270-Output['WD'])*sind(270-Output['WD'])+\
            Output['vv']*sind(270-Output['WD'])**2
            
    Output['ti']=uu_rot**0.5/Output['WS']*100
    Output['ti'].attrs={'units':'%','description':'streamwise turbulence intensity'}
    
    Output['u_star']=(Output['uw']**2+Output['vw']**2)**0.25
    Output['u_star'].attrs={'units':'m/s','description':'friction velocity'}
    
    #general attributes
    location_id=filename.split('.')[0]
    Output.attrs.update({
        'title':        'Six-beam wind statistics',
        'description':  ('10-minute wind statistics from six-beam scan using the methods in Sathe et. al, 2015(10.5194/amt-8-729-2015)',
                         ' and Letizia et al., 2024 (10.1063/5.0209729)'),
        'contact':      'stefano.letizia@nlr.gov',
        'institution':  'National Laboratory of the Rockies',
        'conventions':  'MHKiT-Cloud Data Standards v. 1.0',
        'location_id':  location_id,
        'history':      (f"Generated by {getpass.getuser()} on {socket.gethostname()} on "
                         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using "
                         f"{os.path.basename(sys.argv[0])}"),
        'code':             'https://github.com/StefanoWind/CORSAIR_analysis',
        'code_version': _git_hash(),
    })

    
    logging.info(f'Wind profiles saves as {os.path.join(save_path,filename)}')
    Output.to_netcdf(os.path.join(save_path,filename))
    
    #plots
    wind_map(Output,os.path.join(save_path,filename))
    
    return Output

    
def wind_map(data,filename):
    '''
    Draw wind speed ans TKE maps
    '''
    
    barb_stagger_time = 1#skipped time samples in barbs
    barb_stagger_height = 10#skipped height samples in barbs
    colorbar_fs = 14#colorbar fontsize
    label_fs = 14#colorbar fontsize
    tick_fs = 14#colorbar fontsize
    
    date=str(np.min(data.time.values))[:10]
    dtime=int(np.round(np.median(np.diff(data.time))/np.timedelta64(1,'m')))
    offset=int(np.round((np.min(data.time)-np.datetime64(date+'T00:00:00'))/np.timedelta64(1,'m')))
    data=data.resample(time=f'{dtime}min',offset=f'{offset-1}min').nearest(tolerance='2min')
    data['time']=data.time+np.timedelta64(1,'m')
    
    fig=plt.figure(figsize=(18,10))
    ax=plt.subplot(2,1,1)
    CS = ax.contourf(data.time, data.height, data.WS.T, np.round(np.arange(np.nanpercentile(data.WS,5), np.nanpercentile(data.WS,95)+0.5, 0.25),1), extend='both', cmap='coolwarm')
    ax.barbs(data.time[::barb_stagger_time], data.height[::barb_stagger_height], data.U.T[::barb_stagger_height,::barb_stagger_time]*1.94, data.V.T[::barb_stagger_height,::barb_stagger_time]*1.94,
        barbcolor='black', flagcolor='black', color='black', fill_empty=0, length=5.8, linewidth=1)
    ax.barbs(data.time[0]+np.timedelta64(1260,'s'), 1800, 10*np.cos(60), -10*np.sin(60), barbcolor='black',
        flagcolor='black', color='black', fill_empty=0, length=5.8, linewidth=1.4,zorder=11)
    ax.text(data.time[0]+np.timedelta64(600,'s'), 1710, '10 kts \n', fontsize=12, bbox=dict(facecolor='w', edgecolor='black', alpha=0.8))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '2%', pad=0.65)
    cb = fig.colorbar(CS, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=colorbar_fs)
    cb.set_label(r'Mean wind speed [m s$^{-1}$]', fontsize=colorbar_fs)
    ax.set_xlabel('Time (UTC)', fontsize=label_fs)
    ax.set_ylabel(r'$z$ [m a.g.l.]', fontsize=label_fs)
    ax.set_xlim(data.time.min()-np.timedelta64(300,'s'),data.time.max()+np.timedelta64(900,'s'))
    ax.set_ylim(0, np.max(data.height))
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title(str(data.time.values[0])[:10], fontsize=label_fs)
    ax.set_title(f'Wind speed and direction at {data.attrs["location_id"]} on {str(data.time.values[0])[:10]} \n')

    ax=plt.subplot(2,1,2)
    CS = ax.contourf(data.time, data.height, data.tke.T, np.round(np.arange(np.nanpercentile(data.tke,5), np.nanpercentile(data.tke,95)+0.5, 0.25),1), extend='both', cmap='coolwarm')
    ax.barbs(data.time[::barb_stagger_time], data.height[::barb_stagger_height], data.U.T[::barb_stagger_height,::barb_stagger_time]*1.94, data.V.T[::barb_stagger_height,::barb_stagger_time]*1.94,
        barbcolor='black', flagcolor='black', color='black', fill_empty=0, length=5.8, linewidth=1)
    ax.barbs(data.time[0]+np.timedelta64(1260,'s'), 1800, 10*np.cos(60), -10*np.sin(60), barbcolor='black',
        flagcolor='black', color='black', fill_empty=0, length=5.8, linewidth=1.4, zorder=11)
    ax.text(data.time[0]+np.timedelta64(600,'s'), 1710, '10 kts \n', fontsize=12, bbox=dict(facecolor='w', edgecolor='black', alpha=0.8))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '2%', pad=0.65)
    cb = fig.colorbar(CS, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=colorbar_fs)
    cb.set_label(r'Turbulent kinetic energy [m$^2$ s$^{-2}$]', fontsize=colorbar_fs)
    
    ax.set_xlabel('Time (UTC)', fontsize=label_fs)
    ax.set_ylabel(r'$z$ [m a.g.l.]', fontsize=label_fs)
    ax.set_xlim(data.time.min()-np.timedelta64(300,'s'),data.time.max()+np.timedelta64(900,'s'))
    ax.set_ylim(0, np.max(data.height))
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.savefig(filename.replace('.nc','.png'))
    plt.close()

#%% Main
for channel in config['channels_six_beam']:
    save_path=os.path.join(config['path_data'],channel.replace('b0','c1'))
    os.makedirs(save_path,exist_ok=True)
    
    files={}
    for d in days:
        files[d]=glob.glob(os.path.join(config['path_data'],channel,f'*{str(d)[:10].replace("-","")}*six.beam*nc'))
        
    if mode=='serial':
        for d in days:
            if len(files[d])>1:
                Output=wind_retrieval(files[d],config,config['lidar_height'][channel],save_path,delete,replace)
    
    elif mode=='parallel':
        args = [(files[d], config,config['lidar_height'][channel], save_path,delete,replace) for d in days]
        with Pool() as pool:
            pool.starmap(wind_retrieval, args)
    else:
        raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")
     