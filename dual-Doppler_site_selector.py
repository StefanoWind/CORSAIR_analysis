# -*- coding: utf-8 -*-
"""
Identify suitable pair of sites for dual-Doppler
"""

import os
cd=os.getcwd()
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import matplotlib.image as mpimg
import pandas as pd
import glob
import utm
from scipy import stats
from utils import cosd,sind,dual_Doppler,matrix_plt
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_nwtc=os.path.join(cd,'data/FC_assets.xlsx')#source nwtc sites
source_m2=os.path.join(cd,'data/nwtc.m2.b0/*nc')#source of M2 wind data
source_img=os.path.join(cd,'figures/FC.png')#image source
min_range=100#[m] minimum range
max_range=2000#[m] maximum range

bins_ws=np.array([0,2.5,5,7.5,10,15,25])#[m/s] bins in wind speed
bins_wd=np.arange(0,360,30)#[deg] bins in wind direction

#graphics
xmin, xmax = 480022-710, 480909+380    
ymin, ymax = 4417540-450,4418182+280 

#%% Initialization

#read data
FC=pd.read_excel(source_nwtc).set_index('Site')
M2=xr.open_mfdataset(glob.glob(source_m2))
img = mpimg.imread(source_img)

#UTM coordonates
FC['x_utm'],FC['y_utm'],FC['zone_utm1'],FC['zone_utm2']=utm.from_latlon(FC['Lat'].values, FC['Lon'].values)

#climatology
M2=M2.where(M2.WS_5m>0)
N=     stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WS_5m.values,statistic='count',bins=[bins_ws,bins_wd])[0]
ws_avg=stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WS_5m.values,statistic='mean', bins=[bins_ws,bins_wd])[0]
wd_avg=stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WD_5m.values,statistic='mean', bins=[bins_ws,bins_wd])[0]
 
#dual-Doppler locations
x_target=FC[FC['DD target']=='Yes']['x_utm']
y_target=FC[FC['DD target']=='Yes']['y_utm']
x_source=FC[FC['DD source']=='Yes']['x_utm']
y_source=FC[FC['DD source']=='Yes']['y_utm']

#zeroing
err_u=np.zeros((len(x_source),len(x_source)))+np.nan
err_v=np.zeros((len(x_source),len(x_source)))+np.nan
err_ws=np.zeros((len(x_source),len(x_source)))+np.nan
err_wd=np.zeros((len(x_source),len(x_source)))+np.nan

os.makedirs(os.path.join(cd,'figures','dual-Doppler'),exist_ok=True)

#%% Main
for i_s1 in range(len(x_source)):
    for i_s2 in range(i_s1+1,len(x_source)):

        #dual-Doppler maps
        x_DD1=x_source[i_s1]
        x_DD2=x_source[i_s2]
        y_DD1=y_source[i_s1]
        y_DD2=y_source[i_s2]

        #dual-Doppler error
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
            
        #evaluate at target points
        DD_interp=DD.interp(x=x_target,y=y_target)
        err_u[i_s1,i_s2]= np.mean(np.diag(DD_interp['sigma_u'].values))
        err_v[i_s1,i_s2]= np.mean(np.diag(DD_interp['sigma_v'].values))
        err_ws[i_s1,i_s2]=np.mean(np.diag(DD_interp['sigma_ws'].values))
        err_wd[i_s1,i_s2]=np.mean(np.diag(DD_interp['sigma_wd'].values))
            
        #plots
        #dual-Doppler error (u,v)
        plt.figure(figsize=(20,20))
        ax=plt.subplot(2,2,1)
        ax.imshow(img, extent=[xmin, xmax, ymin, ymax])
        plt.contourf(DD.x,DD.y,DD.sigma_u,np.arange(0,10.2,0.1),cmap='RdYlGn_r',extend='both',alpha=0.25)
        ax.set_aspect('equal')
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        plt.ylabel('$y$ [m]')
        ax.set_xticklabels([])
        cf=plt.scatter(x_target,y_target,s=20,c=np.diag(DD_interp['sigma_u'].values),cmap='RdYlGn_r',vmin=0,vmax=10,edgecolor='k')
        plt.plot([x_source[i_s1],x_source[i_s2]],[y_source[i_s1],y_source[i_s2]],'bs',markersize=5)
        plt.colorbar(cf,label=r'Error factor of $u$')
           
        ax=plt.subplot(2,2,2)
        ax.imshow(img, extent=[xmin, xmax, ymin, ymax])
        plt.contourf(DD.x,DD.y,DD.sigma_v,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both',alpha=0.25)
        ax.set_aspect('equal')
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        cf=plt.scatter(x_target,y_target,s=20,c=np.diag(DD_interp['sigma_v'].values),cmap='RdYlGn_r',vmin=0,vmax=10,edgecolor='k')
        plt.plot([x_source[i_s1],x_source[i_s2]],[y_source[i_s1],y_source[i_s2]],'bs',markersize=5)
        plt.colorbar(cf,label=r'Error factor of $v$')
        
        ax=plt.subplot(2,2,3)
        ax.imshow(img, extent=[xmin, xmax, ymin, ymax])
        plt.contourf(DD.x,DD.y,DD.sigma_ws,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both',alpha=0.25)
        ax.set_aspect('equal')
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        plt.xlabel('$x$ [m]')
        plt.ylabel('$y$ [m]')
        plt.xticks(rotation=30) 
        cf=plt.scatter(x_target,y_target,s=20,c=np.diag(DD_interp['sigma_ws'].values),cmap='RdYlGn_r',vmin=0,vmax=10,edgecolor='k')
        plt.plot([x_source[i_s1],x_source[i_s2]],[y_source[i_s1],y_source[i_s2]],'bs',markersize=5)
        plt.colorbar(cf,label=r'Error factor of wind speed')
        
        ax=plt.subplot(2,2,4)
        ax.imshow(img, extent=[xmin, xmax, ymin, ymax])
        plt.contourf(DD.x,DD.y,DD.sigma_wd,np.arange(0,91),cmap='RdYlGn_r',extend='both',alpha=0.25)
        ax.set_aspect('equal')
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        plt.xlabel('$x$ [m]')
        plt.xticks(rotation=30) 
        ax.set_yticklabels([])
        cf=plt.scatter(x_target,y_target,s=20,c=np.diag(DD_interp['sigma_wd'].values),cmap='RdYlGn_r',vmin=0,vmax=90,edgecolor='k')
        plt.plot([x_source[i_s1],x_source[i_s2]],[y_source[i_s1],y_source[i_s2]],'bs',markersize=5)
        plt.colorbar(cf,label=r'Error factor of wind direction [$^\circ$ s m$^{-1}$]')
        
        plt.tight_layout()
        plt.savefig(os.path.join(cd,'figures','dual-Doppler',f'{x_source.index[i_s1]}-{x_source.index[i_s2]}.png'))
        plt.close()
        
#%% Output
err=err_u/np.nanmedian(err_u)+err_v/np.nanmedian(err_v)+err_ws/np.nanmedian(err_ws)+err_wd/np.nanmedian(err_wd)

sites=x_source.index  
SITES1,SITES2=np.meshgrid(sites,sites)
Output=pd.DataFrame({'Site1':SITES1.ravel(),'Site2':SITES2.ravel(),'Total error':err.ravel()})
Output.to_excel(os.path.join('data','dual-Doppler_opt.xlsx'))

#%% Plots
fig=plt.figure(figsize=(18,10))
ax=plt.subplot(2,2,1)
pc=matrix_plt(sites,sites, err_u, cmap='RdYlGn_r', vmin=0, vmax=10)
plt.title('Mean error factor of $u$')
plt.ylabel('Lidar 2 site')
ax.set_xticklabels([])
plt.title(r'Mean error factor of $u$')
ax=plt.subplot(2,2,2)
pc=matrix_plt(sites,sites, err_v, cmap='RdYlGn_r', vmin=0, vmax=10)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.title(r'Mean error factor of $v$')
ax=plt.subplot(2,2,3)
pc=matrix_plt(sites,sites, err_ws, cmap='RdYlGn_r', vmin=0, vmax=10)
plt.xlabel('Lidar 1 site')
plt.ylabel('Lidar 2 site')
plt.xticks(rotation=30) 
plt.title(r'Mean error factor of wind speed')
ax=plt.subplot(2,2,4)
pc=matrix_plt(sites,sites, err_wd, cmap='RdYlGn_r', vmin=0, vmax=90)
plt.xlabel('Lidar 1 site')
ax.set_yticklabels([])
plt.xticks(rotation=30) 
plt.title(r'Mean error factor of wind direction [$^\circ$ s m$^{-1}$]')
plt.tight_layout()
     