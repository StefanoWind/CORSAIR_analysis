# -*- coding: utf-8 -*-
"""
Plot hard target results superimposed to aerial map
"""
import os
import utils as utl
import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import utm

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['savefig.dpi']=300
plt.close("all")

#%% Inputs
source=os.path.abspath(os.path.join('data','corsair','s42.lidar.z01.a0',
                                    's42.lidar.z01.a0.20260406.150921.user1.nc'))

source_nwtc=os.path.abspath('data/FC_sensing.xlsx')#source nwtc sites


min_SNR=10#[dB] minimum SNR for hard taget
max_SNR_bloc=-22.5#[dB] maximum mean near range SNR for blocked beams
max_range_bloc=500#[m] near range averaging distance for blocked beams
min_r=75#[s] minimum range

home_azimuth=3.55#[deg]

#graphics
zoom=18
xmin,xmax,ymin,ymax=-2000,500,-1000,1000

#%% Initialization

#site layout
FC=pd.read_excel(source_nwtc).set_index('Site')
    
site=f'Site {os.path.basename(source)[1]}.{os.path.basename(source)[2]}'
lat0, lon0 =  FC['Latitude'].loc[site], FC['Longitude'].loc[site] #location of the lidar    
x0,y0,_,_=utm.from_latlon(lat0, lon0)

#lidar data
Data=xr.open_dataset(source)

#%% Main
Data=Data.where(Data.distance<np.max(np.abs(np.array([xmin,xmax,ymin,ymax]))),drop=True)
Data['SNR']=Data.SNR.where(~np.isnan(Data.SNR),Data.SNR.min())#fill null SNR

#Cartesian coordinates
Data['x']=Data.distance*np.cos(np.radians(90-(Data.azimuth+home_azimuth)))*np.cos(np.radians(Data.elevation))
Data['y']=Data.distance*np.sin(np.radians(90-(Data.azimuth+home_azimuth)))*np.cos(np.radians(Data.elevation))

#remove blind zone
Data['SNR']=Data.SNR.where(Data.distance>=min_r)

#hard-target flag
Data['ht']=Data.SNR>min_SNR

#blocked beam flag
blocked=Data.SNR.where(Data.distance<=max_range_bloc).mean(dim='range_gate')<max_SNR_bloc

#rais hard target flag in nearest reange of blocked beam
Data['ht'].isel(range_gate=0)[:] = xr.where(blocked, True, Data.ht.isel(range_gate=0))

#extract hard targets
Data_ht=Data.where(Data.ht,drop=True)
x_ht=Data_ht.x.values.ravel()
y_ht=Data_ht.y.values.ravel()

#%% Main

#aerial map
fig,ax=utl.aerial_map(x_ht,y_ht,lat0,lon0,markersize=2,alpha=1,zoom=zoom,
                      xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
plt.title(str(os.path.basename(source))+r', $\Delta \theta='+str(home_azimuth)+'^\circ$')
plt.xlabel('E-W [m]')
plt.ylabel('S-N [m]')
plt.grid()

#SNR map
fig=plt.figure(figsize=(10,16))
pc=plt.pcolor(Data.x,Data.y,Data.SNR.T,vmin=-30,vmax=10,cmap='plasma')
plt.plot(x_ht,y_ht,'.r',markersize=1)
ax=plt.gca()
ax.set_aspect('equal')
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$y$ [m]')
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
plt.grid()
plt.title(str(os.path.basename(source))+r', $\Delta \theta='+str(home_azimuth)+'^\circ$')
plt.colorbar(pc,label='SNR [dB]')