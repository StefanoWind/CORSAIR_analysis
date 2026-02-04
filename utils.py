# -*- coding: utf-8 -*-
"""
Utilities
"""

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

#%% Functions
def angle_difference_deg(a1, a2):
    return (a2 - a1 + 180) % 360 - 180 

def cosd(x):
    return np.cos(np.radians(x))

def sind(x):
    return np.sin(np.radians(x))

def dual_Doppler(x1,x2,y1,y2,min_range,max_range,x=None,y=None):
    
    #define grid
    if x is None or y is None:
        x=np.arange(-2000,2011,10)+x1
        y=np.arange(-2000,2001,10)+y1
    
    DD=xr.Dataset()
    DD['x']=xr.DataArray(data=x,coords={'x':x})
    DD['y']=xr.DataArray(data=y,coords={'y':y})
    
    #define angles
    DD['chi1']=np.degrees(np.arctan2(DD.y-y1,DD.x-x1))
    DD['chi2']=np.degrees(np.arctan2(DD.y-y2,DD.x-x2))
    DD['dchi']=angle_difference_deg(DD['chi1'],DD['chi2'])
    DD['chi_avg']=np.degrees(np.arctan2(sind(DD.chi1)+sind(DD.chi2),cosd(DD.chi1)+cosd(DD.chi2)))
    DD['alpha_u']=angle_difference_deg(DD['chi_avg'],0)
    DD['alpha_v']=angle_difference_deg(DD['chi_avg'],90)

    #uncertainties
    Nu=(sind(DD['alpha_u']+DD['dchi']/2))**2+(sind(DD['alpha_u']-DD['dchi']/2))**2
    Nv=(sind(DD['alpha_v']+DD['dchi']/2))**2+(sind(DD['alpha_v']-DD['dchi']/2))**2
    D=sind(DD['dchi'])**2

    DD['sigma_u']=(Nu/(D+10**-10))**0.5
    DD['sigma_v']=(Nv/(D+10**-10))**0.5
    
    #exclude ranges
    DD['range1']=((DD.x-x1)**2+(DD.y-y1)**2)**0.5
    DD['range2']=((DD.x-x2)**2+(DD.y-y2)**2)**0.5
      
    DD=DD.where((DD['range1']>min_range)*(DD['range1']<max_range)*\
                (DD['range2']>min_range)*(DD['range2']<max_range))
     
    return DD


def matrix_plt(x,y,f,cmap,vmin,vmax):
    '''
    Plot matrix with color and display values
    '''
    pc=plt.pcolor(x,y,f.T,cmap=cmap,vmin=vmin,vmax=vmax)
    ax=plt.gca()
    for i in range(len(x)):
        for j in range(len(y)):
            if ~np.isnan(f[i,j]):
                ax.text(i,j, f"{f[i,j]:.1f}", 
                        ha='center', va='center', color='k', fontsize=10,fontweight='bold')
            
    return pc
