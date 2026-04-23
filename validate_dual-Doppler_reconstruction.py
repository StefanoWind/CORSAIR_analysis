# -*- coding: utf-8 -*-
"""
Validate dual-Doppler reconstruction
"""
import numpy as np
import utils as utl
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs

#grid [m]
x=np.arange(-1000,1000,100)
y=np.arange(-250,1001,100)
z=np.arange(0,201,50)

WS=10
WD=260

#lidar locations
x1=-500 #[m]
y1=0#[m]
z1=0
x2=500#[m]
y2=0#[m]
z2=0

#stats
sigma_los=0.1#[m/s] error on LOS wind speed
L=100#MC draws

#%% Initialization

Data=xr.Dataset()

Data['WS']=xr.DataArray(np.zeros((len(x),len(y),len(z)))+WS,coords={'x':x,'y':y,'z':z})
Data['WD']=xr.DataArray(np.zeros((len(x),len(y),len(z)))+WD,coords={'x':x,'y':y,'z':z})


U=Data.WS*utl.cosd(270-Data.WD)
V=Data.WS*utl.sind(270-Data.WD)

Data['U']=U
Data['V']=V

#%% Main

#spherical coords
azi1=(90-np.degrees(np.arctan2(Data.y-y1,Data.x-x1)))%360
r1=((Data.x-x1)**2+(Data.y-y1)**2+(Data.z-z1)**2)**0.5
ele1=np.degrees(np.arcsin(Data.z/r1))

azi2=(90-np.degrees(np.arctan2(Data.y-y2,Data.x-x2)))%360
r2=((Data.x-x2)**2+(Data.y-y2)**2+(Data.z-z2)**2)**0.5
ele2=np.degrees(np.arcsin(Data.z/r2))

#LOS projection
rws1=utl.cosd(90-azi1)*utl.cosd(ele1)*U+utl.sind(90-azi1)*utl.cosd(ele1)*V
rws2=utl.cosd(90-azi2)*utl.cosd(ele2)*U+utl.sind(90-azi2)*utl.cosd(ele2)*V

#build synthetic datasets
err_U=np.zeros((len(x),len(y),len(z),L))
sigma_U2=np.zeros((len(x),len(y),len(z),L))
err_V=np.zeros((len(x),len(y),len(z),L))
sigma_V2=np.zeros((len(x),len(y),len(z),L))
err_WS=np.zeros((len(x),len(y),len(z),L))
sigma_WS2=np.zeros((len(x),len(y),len(z),L))
err_WD=np.zeros((len(x),len(y),len(z),L))
sigma_WD2=np.zeros((len(x),len(y),len(z),L))
for l in range(L):
    
    Data1=xr.Dataset()
    Data1['rws_avg']=rws1+np.random.normal(0,sigma_los)
    Data1.attrs['x_lidar']=x1
    Data1.attrs['y_lidar']=y1
    Data1.attrs['z_lidar']=z1
    
    Data2=xr.Dataset()
    Data2['rws_avg']=rws2+np.random.normal(0,sigma_los)
    Data2.attrs['x_lidar']=x2
    Data2.attrs['y_lidar']=y2
    Data2.attrs['z_lidar']=z2
    
    Output=utl.dual_doppler_reconstruction(Data1,Data2)
    
    err_U[:,:,:,l]=(Output.U-Data.U).values
    sigma_U2[:,:,:,l]=Output.sigma_U
    err_V[:,:,:,l]=(Output.V-Data.V).values
    sigma_V2[:,:,:,l]=Output.sigma_V
    err_WS[:,:,:,l]=(Output.WS-Data.WS).values
    sigma_WS2[:,:,:,l]=Output.sigma_WS
    err_WD[:,:,:,l]=(Output.WD-Data.WD).values
    sigma_WD2[:,:,:,l]=Output.sigma_WD
    
    print(l)
    
sigma_U=np.std(err_U,axis=3)
sigma_U2=np.mean(sigma_U2,axis=3)*sigma_los
sigma_V=np.std(err_V,axis=3)
sigma_V2=np.mean(sigma_V2,axis=3)*sigma_los
sigma_WS=np.std(err_WS,axis=3)
sigma_WS2=np.mean(sigma_WS2,axis=3)*sigma_los
sigma_WD=np.std(err_WD,axis=3)
sigma_WD2=np.mean(sigma_WD2,axis=3)*sigma_los

#%% Plots
fig=plt.figure(figsize=(18,10))
gs = GridSpec(nrows=2, ncols=len(z)+1, width_ratios=[1]*len(z)+[0.05], figure=fig)
for iz in range(len(z)):
    ax=fig.add_subplot(gs[0,iz])
    pc=plt.pcolor(x,y,sigma_U[:,:,iz].T,vmin=0,vmax=0.5,cmap='RdYlGn_r')
    if iz>z[0]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(r'$y$')
    ax.set_xticklabels([])
    ax.set_aspect('equal')
    plt.grid()
    
    ax=fig.add_subplot(gs[1,iz])
    plt.pcolor(x,y,sigma_U2[:,:,iz].T,vmin=0,vmax=0.5,cmap='RdYlGn_r')
    if iz>z[0]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(r'$y$')
    ax.set_xlabel(r'$x$')
    ax.set_aspect('equal')
    plt.grid()
cax=fig.add_subplot(gs[:,-1])
plt.colorbar(pc,cax,label='Error on $U$ [m s$^{-1}$]')

fig=plt.figure(figsize=(18,10))
gs = GridSpec(nrows=2, ncols=len(z)+1, width_ratios=[1]*len(z)+[0.05], figure=fig)
for iz in range(len(z)):
    ax=fig.add_subplot(gs[0,iz])
    pc=plt.pcolor(x,y,sigma_V[:,:,iz].T,vmin=0,vmax=0.5,cmap='RdYlGn_r')
    if iz>z[0]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(r'$y$')
    ax.set_xticklabels([])
    ax.set_aspect('equal')
    plt.grid()
    
    ax=fig.add_subplot(gs[1,iz])
    plt.pcolor(x,y,sigma_V2[:,:,iz].T,vmin=0,vmax=0.5,cmap='RdYlGn_r')
    if iz>z[0]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(r'$y$')
    ax.set_xlabel(r'$x$')
    ax.set_aspect('equal')
    plt.grid()
cax=fig.add_subplot(gs[:,-1])
plt.colorbar(pc,cax,label='Error on $V$ [m s$^{-1}$]')

fig=plt.figure(figsize=(18,10))
gs = GridSpec(nrows=2, ncols=len(z)+1, width_ratios=[1]*len(z)+[0.05], figure=fig)
for iz in range(len(z)):
    ax=fig.add_subplot(gs[0,iz])
    pc=plt.pcolor(x,y,sigma_WS[:,:,iz].T,vmin=0,vmax=0.5,cmap='RdYlGn_r')
    if iz>z[0]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(r'$y$')
    ax.set_xticklabels([])
    ax.set_aspect('equal')
    plt.grid()
    
    ax=fig.add_subplot(gs[1,iz])
    plt.pcolor(x,y,sigma_WS2[:,:,iz].T,vmin=0,vmax=0.5,cmap='RdYlGn_r')
    if iz>z[0]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(r'$y$')
    ax.set_xlabel(r'$x$')
    ax.set_aspect('equal')
    plt.grid()
cax=fig.add_subplot(gs[:,-1])
plt.colorbar(pc,cax,label='Error on wind speed [m s$^{-1}$]')


fig=plt.figure(figsize=(18,10))
gs = GridSpec(nrows=2, ncols=len(z)+1, width_ratios=[1]*len(z)+[0.05], figure=fig)
for iz in range(len(z)):
    ax=fig.add_subplot(gs[0,iz])
    pc=plt.pcolor(x,y,sigma_WD[:,:,iz].T,vmin=0,vmax=10,cmap='RdYlGn_r')
    if iz>z[0]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(r'$y$')
    ax.set_xticklabels([])
    ax.set_aspect('equal')
    plt.grid()
    
    ax=fig.add_subplot(gs[1,iz])
    plt.pcolor(x,y,sigma_WD2[:,:,iz].T,vmin=0,vmax=10,cmap='RdYlGn_r')
    if iz>z[0]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(r'$y$')
    ax.set_xlabel(r'$x$')
    ax.set_aspect('equal')
    plt.grid()
cax=fig.add_subplot(gs[:,-1])
plt.colorbar(pc,cax,label='Error on wind direction [$^\circ$ s m$^{-1}$]')