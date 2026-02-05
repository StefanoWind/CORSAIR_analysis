# -*- coding: utf-8 -*-
"""
Esimate error due to neglecting w
"""

import os
cd=os.getcwd()
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
from utils import cosd,sind

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs

#grid [m]
x=np.arange(-1500,1501,25)
y=np.arange(-1000,1001,25)

z=200#[m] height

#stats
L=100#MC draws
sigma_w=0.1# std of w

#lidar locations [m]
x1=-500
y1=0
z1=0
x2=500
y2=0
z2=0

#%% Initialization
X,Y=np.meshgrid(x,y)

#true wind field
WS=np.zeros((len(y),len(x),L))+1
WD=np.zeros((len(y),len(x),L))
U=WS*cosd(270-WD)
V=WS*sind(270-WD)

#random w component
w=np.random.normal(0,sigma_w,(len(y),len(x),L))

#%% Main

#scan geometry based on relative location
AZI1=(90-np.degrees(np.arctan2(Y-y1,X-x1)))%360
AZI2=(90-np.degrees(np.arctan2(Y-y2,X-x2)))%360
R1=((X-x1)**2+(Y-y1)**2)**0.5
ELE1=np.degrees(np.arctan2(z-z1,R1))
R2=((X-x2)**2+(Y-y2)**2)**0.5
ELE2=np.degrees(np.arctan2(z-z2,R2))

#component of projection matrix [A,B],[C,D]]
A=cosd(90-AZI1)*cosd(ELE1)
B=sind(90-AZI1)*cosd(ELE1)
C=cosd(90-AZI2)*cosd(ELE2)
D=sind(90-AZI2)*cosd(ELE2)
DET=A*D-B*C

#LOS velocities with uncertainty
U_los1=np.stack([A]*L,axis=2)*U+np.stack([B]*L,axis=2)*V+sind(np.stack([ELE1]*L,axis=2))*w
U_los2=np.stack([C]*L,axis=2)*U+np.stack([D]*L,axis=2)*V+sind(np.stack([ELE2]*L,axis=2))*w

#wind reconstruction
U_rec=( np.stack([D]*L,axis=2)*U_los1-np.stack([B]*L,axis=2)*U_los2)/np.stack([DET]*L,axis=2)
V_rec=(-np.stack([C]*L,axis=2)*U_los1+np.stack([A]*L,axis=2)*U_los2)/np.stack([DET]*L,axis=2)

#MC uncertainty
U_rec_std=np.nanstd(U_rec,axis=2)/sigma_w
V_rec_std=np.nanstd(V_rec,axis=2)/sigma_w

#theoretical uncertainty
U_rec_std_th=np.abs((D*sind(ELE1)-B*sind(ELE2))/DET)
V_rec_std_th=np.abs((-C*sind(ELE1)+A*sind(ELE2))/DET)

#%% Plots
plt.figure(figsize=(20,20))
ax=plt.subplot(2,2,1)
cf=plt.contourf(x,y,U_rec_std_th,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
ax.set_xticklabels([])
plt.ylabel('$y$ [m]')
plt.xticks(rotation=30) 
plt.colorbar(cf,label=r'Error factor of $u$ (Theory)',ticks=np.arange(11))

ax=plt.subplot(2,2,2)
cf=plt.contourf(x,y,V_rec_std_th,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
plt.xticks(rotation=30) 
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.colorbar(cf,label=r'Error factor of $v$ (Theory)',ticks=np.arange(11))

ax=plt.subplot(2,2,3)
cf=plt.contourf(x,y,U_rec_std,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.xticks(rotation=30) 
plt.text(-1400,850,s=r'$z='+str(z)+'$',bbox={'facecolor':(1,1,1,0.5),'edgecolor':'k'})
plt.colorbar(cf,label=r'Error factor of $u$ (MC)',ticks=np.arange(11))

ax=plt.subplot(2,2,4)
cf=plt.contourf(x,y,V_rec_std,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
plt.plot(x1,y1,'sk',markersize=10)
plt.plot(x2,y2,'sk',markersize=10)
ax.set_aspect('equal')
ax.set_yticklabels([])
plt.colorbar(cf,label=r'Error factor of $v$ (MC)',ticks=np.arange(11))
plt.xlabel('$x$ [m]')
plt.xticks(rotation=30) 
plt.tight_layout()