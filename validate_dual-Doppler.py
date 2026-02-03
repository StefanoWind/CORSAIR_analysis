# -*- coding: utf-8 -*-
"""
Monte Carlo validation of dual-Doppler error formulas
"""

import os
cd=os.getcwd()
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
from utils import cosd,sind,dual_Doppler
from scipy import stats
import xarray as xr
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs

#grid [m]
x=np.arange(-1500,1501,25)
y=np.arange(-1000,1001,25)

#lidar locations
x1=-500 #[m]
y1=0#[m]
x2=500#[m]
y2=0#[m]

#stats
sigma_los=0.1#[m/s] error on LOS wind speed
L=100#MC draws
N_wind=1000#climatology samples
shape_ws=3#shape factor of wind speed distribution
scale_ws=10#[m] scale factor of wind speed distribution
min_ws=0.5 #[m/s] minimum wind speed
bins_ws=np.array([0,2.5,5,7.5,10,15,25])#[m/s] bins in wind speed
bins_wd=np.arange(0,361,30)#[deg] bins in wind direction

#%% Initialization
X,Y=np.meshgrid(x,y)

#climatology
ws=scale_ws*np.random.weibull(shape_ws, N_wind)
wd=np.random.uniform(0,360,N_wind)
N=     stats.binned_statistic_2d(ws,wd,ws,statistic='count',bins=[bins_ws,bins_wd])[0]
ws_avg=stats.binned_statistic_2d(ws,wd,ws,statistic='mean', bins=[bins_ws,bins_wd])[0]
wd_avg=stats.binned_statistic_2d(ws,wd,wd,statistic='mean', bins=[bins_ws,bins_wd])[0]

#dual-Doppler projections
AZI1=(90-np.degrees(np.arctan2(Y-y1,X-x1)))%360
AZI2=(90-np.degrees(np.arctan2(Y-y2,X-x2)))%360
C1=cosd(90-np.stack([AZI1]*L,axis=2))
S1=sind(90-np.stack([AZI1]*L,axis=2))
C2=cosd(90-np.stack([AZI2]*L,axis=2))
S2=sind(90-np.stack([AZI2]*L,axis=2))
 
#zeroing
U_rec_std=[]
V_rec_std=[]
WS_rec_std=[]
WD_rec_std=[]

#%% Main

ctr=0
for wsi,wdi in zip(ws,wd):
    if wsi>=min_ws:
        
        #true wind filed
        WS=np.zeros((len(y),len(x),L))+wsi
        WD=np.zeros((len(y),len(x),L))+wdi
        U=WS*cosd(270-WD)
        V=WS*sind(270-WD)
        
        #LOS velocities with uncertainty
        U_los1=U*C1+V*S1+np.random.normal(0,sigma_los,np.shape(U))
        U_los2=U*C2+V*S2+np.random.normal(0,sigma_los,np.shape(U))
        
        #wind reconstruction
        DET=C1*S2-C2*S1
        U_rec= (U_los1*S2-U_los2*S1)/DET
        V_rec=(-U_los1*C2+U_los2*C1)/DET
        WS_rec=(U_rec**2+V_rec**2)**0.5
        WD_rec=(270-np.degrees(np.arctan2(V_rec,U_rec)))%360
        
        #MC uncertainty
        U_rec_std+=[np.std(U_rec,axis=2)/sigma_los]
        V_rec_std+=[np.std(V_rec,axis=2)/sigma_los]
        WS_rec_std+=[np.std(WS_rec,axis=2)/sigma_los]
        WD_rec_std+=[np.std(WD_rec,axis=2)/sigma_los]
        
    ctr+=1
    print(ctr/N_wind)

#median MC uncertainty
U_rec_std=np.median(np.stack(U_rec_std,axis=2),axis=2)
V_rec_std=np.median(np.stack(V_rec_std,axis=2),axis=2)
WS_rec_std=np.median(np.stack(WS_rec_std,axis=2),axis=2)
WD_rec_std=np.median(np.stack(WD_rec_std,axis=2),axis=2)

#theoretical uncertainty
DD=dual_Doppler(x1, x2, y1, y2, min_range=0, max_range=2500,x=x,y=y)

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
plt.figure(figsize=(20,20))
ax=plt.subplot(2,2,1)
cf=plt.contourf(x,y,U_rec_std,np.arange(0,10.2,0.1),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
plt.ylabel('$y$ [m]')
ax.set_xticklabels([])
plt.colorbar(cf,label=r'Error factor of $u$')

ax=plt.subplot(2,2,2)
cf=plt.contourf(x,y,V_rec_std,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.colorbar(cf,label=r'Error factor of $v$')

ax=plt.subplot(2,2,3)
cf=plt.contourf(x,y,WS_rec_std,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.xticks(rotation=30) 
plt.colorbar(cf,label=r'Error factor of wind speed')

ax=plt.subplot(2,2,4)
cf=plt.contourf(x,y,WD_rec_std,np.arange(0,91),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.xticks(rotation=30) 
ax.set_yticklabels([])
plt.colorbar(cf,label=r'Error factor of wind direction [$^\circ$ s m$^{-1}$]')

plt.tight_layout()

#dual-Doppler error (u,v)
plt.figure(figsize=(20,20))
ax=plt.subplot(2,2,1)
cf=plt.contourf(DD.x,DD.y,DD.sigma_u,np.arange(0,10.2,0.1),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
plt.ylabel('$y$ [m]')
ax.set_xticklabels([])
plt.colorbar(cf,label=r'Error factor of $u$')
   
ax=plt.subplot(2,2,2)
cf=plt.contourf(DD.x,DD.y,DD.sigma_v,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.colorbar(cf,label=r'Error factor of $v$')

ax=plt.subplot(2,2,3)
cf=plt.contourf(DD.x,DD.y,DD.sigma_ws,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.xticks(rotation=30) 
plt.colorbar(cf,label=r'Error factor of wind speed')

ax=plt.subplot(2,2,4)
cf=plt.contourf(DD.x,DD.y,DD.sigma_wd,np.arange(0,91),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.xticks(rotation=30) 
ax.set_yticklabels([])
plt.colorbar(cf,label=r'Error factor of wind direction [$^\circ$ s m$^{-1}$]')

plt.tight_layout()



