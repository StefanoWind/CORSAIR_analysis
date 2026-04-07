
# -*- coding: utf-8 -*-
"""
Evaluate scanning sector for dual-Doppler mapping
"""

import os
cd=os.getcwd()
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
import glob
import utm
import utils as utl
from scipy import stats
from utils import cosd,sind,dual_Doppler
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_nwtc=os.path.join(cd,'data/CORSAIR_layout.xlsx')#source nwtc lidars
source_m2=os.path.join(cd,'data/nwtc.m2.b0/*nc')#source of M2 wind data
lidars=['Halo 199','Halo 200']#lidar lidars

#scan limits
azi1={'Halo 199':280,'Halo 200': -10}#initial azimuth [deg]
azi2={'Halo 199':300,'Halo 200': 30}#final azimuth [deg]
min_range=100#[m] minimum range
max_range=2000#[m] maximum range

#stats
bins_ws=np.array([0,2.5,5,7.5,10,15,25])#[m/s] bins in wind speed
bins_wd=np.arange(0,360,30)#[deg] bins in wind direction

#graphics
zoom=15
xmin,xmax,ymin,ymax=-2000,500,-1000,1000#[m] plot limits 

#%% Initialization

#site layout
FC=pd.read_excel(source_nwtc,sheet_name='Instruments').set_index('Name')

x0={}
y0={}
for l in lidars:
    lat, lon =  FC['Latitude'].loc[l], FC['Longitude'].loc[l] #location of the lidar
    x0[l],y0[l],_,_=utm.from_latlon(lat, lon)
    
lat0, lon0 =  FC['Latitude'].loc[lidars[0]], FC['Longitude'].loc[lidars[0]] #reference lat-lon

#wind data
M2=xr.open_mfdataset(glob.glob(source_m2))

#%% Main

#scan limits
x=[]
y=[]
for l in lidars:
    x=np.append(x,utl.cosd(90-azi1[l])*np.arange(max_range)+x0[l])
    y=np.append(y,utl.sind(90-azi1[l])*np.arange(max_range)+y0[l])
    x=np.append(x,utl.cosd(90-azi2[l])*np.arange(max_range)+x0[l])
    y=np.append(y,utl.sind(90-azi2[l])*np.arange(max_range)+y0[l])
    
#climatology
M2=M2.where(M2.WS_5m>0)
N=     stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WS_5m.values,statistic='count',bins=[bins_ws,bins_wd])[0]
ws_avg=stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WS_5m.values,statistic='mean', bins=[bins_ws,bins_wd])[0]
wd_avg=stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WD_5m.values,statistic='mean', bins=[bins_ws,bins_wd])[0]
 
#DD error
x_DD1,x_DD2,y_DD1,y_DD2=x0[lidars[0]],x0[lidars[1]],y0[lidars[0]],y0[lidars[1]]
DD=dual_Doppler(x_DD1, x_DD2, y_DD1, y_DD2,min_range,max_range)

DD['N']=xr.DataArray(N/np.sum(N),    coords={'ws':(bins_ws[1:]+bins_ws[:-1])/2,'wd':(bins_wd[1:]+bins_wd[:-1])/2})
DD['ws_avg']=xr.DataArray(ws_avg,    coords={'ws':(bins_ws[1:]+bins_ws[:-1])/2,'wd':(bins_wd[1:]+bins_wd[:-1])/2})
DD['wd_avg']=xr.DataArray(wd_avg,    coords={'ws':(bins_ws[1:]+bins_ws[:-1])/2,'wd':(bins_wd[1:]+bins_wd[:-1])/2})

DD['sigma_ws']=(((cosd(270-DD['wd_avg']))**2*DD['sigma_u']**2+\
                 (sind(270-DD['wd_avg']))**2*DD['sigma_v']**2)**0.5*DD['N']).sum(dim='ws').sum(dim='wd')
DD['sigma_ws']=DD['sigma_ws'].where(~np.isnan(DD['sigma_u']))
    
DD['sigma_wd']=(((sind(270-DD['wd_avg']))**2*DD['sigma_u']**2+\
                 (cosd(270-DD['wd_avg']))**2*DD['sigma_v']**2)**0.5/DD['ws_avg']*DD['N']).sum(dim='ws').sum(dim='wd')*180/np.pi
DD['sigma_wd']=DD['sigma_wd'].where(~np.isnan(DD['sigma_u']))
    
#%% Plots
#aerial map
fig,ax=utl.aerial_map(x-x0[lidars[0]],y-y0[lidars[0]],lat0,lon0,markersize=1,alpha=1,zoom=zoom,color='k',
                      xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
plt.xlabel('E-W [m]')
plt.ylabel('S-N [m]')
plt.grid()

#plots
#dual-Doppler error (u,v)
plt.figure(figsize=(18,10))
ax=plt.subplot(2,2,1)
cf=plt.contourf(DD.x,DD.y,DD.sigma_u,np.arange(0,10.2,0.1),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlim([xmin+x0[lidars[0]],xmax+x0[lidars[0]]])
plt.ylim([ymin+y0[lidars[0]],ymax+y0[lidars[0]]])
plt.ylabel('$y$ [m]')
ax.set_xticklabels([])
plt.plot(x,y,'.k',markersize=1)
plt.colorbar(cf,label=r'Error factor of $U$')
   
ax=plt.subplot(2,2,2)
cf=plt.contourf(DD.x,DD.y,DD.sigma_v,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlim([xmin+x0[lidars[0]],xmax+x0[lidars[0]]])
plt.ylim([ymin+y0[lidars[0]],ymax+y0[lidars[0]]])
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.plot(x,y,'.k',markersize=1)
plt.colorbar(cf,label=r'Error factor of $V$')

ax=plt.subplot(2,2,3)
cf=plt.contourf(DD.x,DD.y,DD.sigma_ws,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlim([xmin+x0[lidars[0]],xmax+x0[lidars[0]]])
plt.ylim([ymin+y0[lidars[0]],ymax+y0[lidars[0]]])
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.xticks(rotation=30) 
plt.plot(x,y,'.k',markersize=1)
plt.colorbar(cf,label=r'Error factor of $U_h$')

ax=plt.subplot(2,2,4)
cf=plt.contourf(DD.x,DD.y,DD.sigma_wd,np.arange(0,91),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlim([xmin+x0[lidars[0]],xmax+x0[lidars[0]]])
plt.ylim([ymin+y0[lidars[0]],ymax+y0[lidars[0]]])
plt.xlabel('$x$ [m]')
plt.xticks(rotation=30) 
ax.set_yticklabels([])
plt.plot(x,y,'.k',markersize=1)
plt.colorbar(cf,label=r'Error factor of $\theta_h$ [$^\circ$ s m$^{-1}$]')
plt.tight_layout()