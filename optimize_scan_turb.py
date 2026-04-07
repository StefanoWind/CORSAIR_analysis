# -*- coding: utf-8 -*-
"""
Scan optimization for dual-Doppler mapping of turbulent field
"""

import os
cd=os.getcwd()
from matplotlib import pyplot as plt
import warnings
import matplotlib
from lisboa import scan_optimizer as opt
import glob
import xarray as xr
import pandas as pd
import utm
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300

#%% Inputs
source_nwtc=os.path.join(cd,'data/FC_sensing.xlsx')#source nwtc sites
source_ws=os.path.join(cd,'data/nwtc.m2.b0/*nc')
parallel=False

sites=['Site 4.2','Site 1.9']#lidar sites
site_ref='NextTracker'#grid origin

#Pareto
coords='xy'
azi1={'Site 4.2':[-80],'Site 1.9':[-30]}
azi2={'Site 4.2':[30],'Site 1.9':[80]}
ele1={'Site 4.2':[0],'Site 1.9':[0]}
ele2={'Site 4.2':[0],'Site 1.9':[0]}
dazi={'Site 4.2':[0.5,1,2,3,4],'Site 1.9':[0.5,1,2,3,4]}
dele={'Site 4.2':[0,0,0,0,0],'Site 1.9':[0,0,0,0,0]}

num_azi=None
num_ele=None
full_scan_file=False

#lidar settings
path_config_lidar={'Site 4.2':os.path.join(cd,'configs','config.199.yaml'),
                   'Site 1.9':os.path.join(cd,'configs','config.200.yaml')}
volumetric=False
mode='CSM'
azi0={'Site 4.2':3.55,
      'Site 1.9':12}#azimuth offset [deg]

ppr=3000#pulses per ray
dr=30#[m] gate length
rmin=100#minimum range
rmax=1500#maximum range

#time info
T=600#[s] scan duration
tau=30#[s] timescale

#lisboa
config={'sigma':0.25,
        'max_iter':3,
        'mins':[-1000,-250],
        'maxs':[1000,1000],
        'Dn0':[100,100],
        'r_max':3,
        'dist_edge':1,
        'tol_dist':0.1,
        'grid_factor':0.25,
        'max_Dd':1}

#%% Initialization

#read data
FC=pd.read_excel(source_nwtc).set_index('Site')

Data_m2=xr.open_mfdataset(glob.glob(source_ws))
ws=Data_m2.WS_5m.values

#UTM locations
FC['x_utm'],FC['y_utm'],FC['zone_utm1'],FC['zone_utm2']=utm.from_latlon(FC['Latitude'].values, FC['Longitude'].values)

x0={}
y0={}
z0={}
for s in sites:
    x0[s]=FC['x_utm'].loc[s]-FC['x_utm'].loc[site_ref]
    y0[s]=FC['y_utm'].loc[s]-FC['y_utm'].loc[site_ref]
    z0[s]=0

os.makedirs(os.path.join(cd,'figures','dual-Doppler'),exist_ok=True)

#%% Main
scopt=opt.scan_optimizer(config,save_path=os.path.join(cd,'data','Pareto'))

Pareto=scopt.pareto(coords,x0,y0,z0,azi0,azi1,azi2,ele1,ele2,dazi,dele,num_azi,num_ele,
                    volumetric=volumetric,rmin=rmin,rmax=rmax, T=T,tau=tau,ws=ws,
                    mode=mode, ppr=ppr, dr=dr, path_config_lidar=path_config_lidar,
                    full_scan_file=full_scan_file,parallel=parallel)


       