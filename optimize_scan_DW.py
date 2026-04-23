# -*- coding: utf-8 -*-
"""
Scan optimization for dual-Doppler mapping of mean wind around distributed wind lidars
"""

import os
cd=os.getcwd()
from matplotlib import pyplot as plt
import warnings
import matplotlib
from lisboa import scan_optimizer as opt
import pandas as pd
import numpy as np
import utm
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300

#%% Inputs
source_nwtc=os.path.join(cd,'data/CORSAIR_layout.xlsx')#source nwtc lidars
parallel=False

lidars=['Halo 199','Halo 200']#lidar lidars
site_ref='QED'#grid origin

#Pareto
coords='xyz'
azi1={'Halo 199':[280],'Halo 200':[350]}
azi2={'Halo 199':[300],'Halo 200':[30]}
ele1={'Halo 199':[0],  'Halo 200':[0]}
ele2={'Halo 199':[6],  'Halo 200':[10]}
dazi={'Halo 199':[0.5 ,0.75, 1, 1.25,  1.5],'Halo 200':[1 ,1.5, 2, 2.5,  3]}
dele={'Halo 199':np.array(dazi['Halo 199'])/4,'Halo 200': np.array(dazi['Halo 200'])/4}

num_azi=None
num_ele=None
full_scan_file=False

#lidar settings
path_config_lidar={'Halo 199':os.path.join(cd,'configs','config.199.yaml'),
                   'Halo 200':os.path.join(cd,'configs','config.200.yaml')}
volumetric=True
mode='CSM'
azi0={'Halo 199':3.55,
      'Halo 200':12}#azimuth offset [deg]

ppr=3000#pulses per ray
dr=30#[m] gate length
rmin=300#minimum range
rmax=1000#maximum range

#time info
T=1200#[s] scan duration
tau=30#[s] timescale

#lisboa
config={'sigma':0.25,
        'max_iter':5,
        'mins':[-150,-100,0],
        'maxs':[ 150,100,80],
        'Dn0':[40,40,10],
        'r_max':3,
        'dist_edge':1,
        'tol_dist':0.1,
        'grid_factor':0.25,
        'max_Dd':1}

#%% Initialization

#read data
RS=pd.read_excel(source_nwtc,sheet_name='Instruments').set_index('Name')
FC=pd.read_excel(source_nwtc,sheet_name='Assets').set_index('Name')

#UTM locations
RS['x_utm'],RS['y_utm'],RS['zone_utm1'],RS['zone_utm2']=utm.from_latlon(RS['Latitude'].values, RS['Longitude'].values)
FC['x_utm'],FC['y_utm'],FC['zone_utm1'],FC['zone_utm2']=utm.from_latlon(FC['Latitude'].values, FC['Longitude'].values)

x0={}
y0={}
z0={}
for l in lidars:
    x0[l]=RS['x_utm'].loc[l]-FC['x_utm'].loc[site_ref]
    y0[l]=RS['y_utm'].loc[l]-FC['y_utm'].loc[site_ref]
    z0[l]=0

os.makedirs(os.path.join(cd,'figures','dual-Doppler'),exist_ok=True)

#%% Main
scopt=opt.scan_optimizer(config,save_path=os.path.join(cd,'data','Pareto'))

Pareto=scopt.pareto(coords,x0,y0,z0,azi0,azi1,azi2,ele1,ele2,dazi,dele,num_azi,num_ele,
                    volumetric=volumetric,rmin=rmin,rmax=rmax, T=T,tau=tau,ws=None,
                    mode=mode, ppr=ppr, dr=dr, path_config_lidar=path_config_lidar,
                    full_scan_file=full_scan_file,parallel=parallel)


       