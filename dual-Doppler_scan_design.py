# -*- coding: utf-8 -*-
"""
Scan optimization for dual-Doppler mapping
"""

import os
cd=os.getcwd()
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
from lisboa import scan_optimizer as opt
import pandas as pd
import glob
import utm
from scipy import stats
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300


#%% Inputs
source_nwtc=os.path.join(cd,'data/FC_assets.xlsx')#source nwtc sites
source_m2=os.path.join(cd,'data/nwtc.m2.b0/*nc')#source of M2 wind data
min_range=100#[m] minimum range
max_range=2000#[m] maximum range

bins_ws=np.array([0,2.5,5,7.5,10,15,25])#[m/s] bins in wind speed
bins_wd=np.arange(0,360,30)#[deg] bins in wind direction
sites=['Site 4.2-4.3','Site 1.9']
site_ref='NextTracker'

#Pareto
azi1={'Site 4.2-4.3':[0],'Site 1.9':[-70]}
azi2={'Site 4.2-4.3':[70],'Site 1.9':[0]}
ele1={'Site 4.2-4.3':[0],'Site 1.9':[0]}
ele2={'Site 4.2-4.3':[10],'Site 1.9':[10]}
dazi={'Site 4.2-4.3':[2,3,4],'Site 1.9':[2,3,4]}
dele={'Site 4.2-4.3':[0.5,1],'Site 1.9':[0.5,1]}
num_azi=None
num_ele=None
full_scan_file=False

#lidar settings
coords='xyz'
path_config_lidar='C:/Users/sletizia/Software/FIEXTA/halo_suite/halo_suite/configs/config.217.yaml'
volumetric=True
mode='CSM'
azi_offset=0#[deg] difference between scan direction and x axis
ppr=1000#pulses per ray
dr=30#[m] gate length
rmin=100#minimum range
rmax=1000#maximum range

#time info
T=600#[s] scan duration
tau=4#[s] timescale in the wake

#lisboa
config={'sigma':0.25,
        'max_iter':5,
        'mins':[-1000,-1000,0],
        'maxs':[1000,1000,200],
        'Dn0':[200,200,50],
        'r_max':3,
        'dist_edge':1,
        'tol_dist':0.1,
        'grid_factor':0.25,
        'max_Dd':1}


#%% Initialization

# #read data
FC=pd.read_excel(source_nwtc).set_index('Site')
# M2=xr.open_mfdataset(glob.glob(source_m2))

# #Cartesianize sites
FC['x_utm'],FC['y_utm'],FC['zone_utm1'],FC['zone_utm2']=utm.from_latlon(FC['Lat'].values, FC['Lon'].values)

x0={}
y0={}
z0={}
for s in sites:
    x0[s]=FC['x_utm'].loc[s]-FC['x_utm'].loc[site_ref]
    y0[s]=FC['y_utm'].loc[s]-FC['y_utm'].loc[site_ref]
    z0[s]=0

# #climatology
# M2=M2.where(M2.WS_5m>0)
# N=     stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WS_5m.values,statistic='count',bins=[bins_ws,bins_wd])[0]
# ws_avg=stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WS_5m.values,statistic='mean', bins=[bins_ws,bins_wd])[0]
# wd_avg=stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WD_5m.values,statistic='mean', bins=[bins_ws,bins_wd])[0]
 

os.makedirs(os.path.join(cd,'figures','dual-Doppler'),exist_ok=True)

#%% Main

scopt=opt.scan_optimizer(config,save_path=os.path.join(cd,'data','Pareto'),logfile=os.path.join(cd,'log','test.log'))


Pareto=scopt.pareto(coords,x0,y0,z0,azi1,azi2,ele1,ele2,dazi,dele,num_azi,num_ele,
                    volumetric=volumetric,rmin=rmin,rmax=rmax, T=T,tau=tau,
                    mode=mode, ppr=ppr, dr=dr, path_config_lidar=path_config_lidar,
                    full_scan_file=full_scan_file)


       