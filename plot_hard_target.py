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
import io
import json
import requests
from PIL import Image
from scipy import ndimage
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['savefig.dpi']=300
plt.close("all")

#%% Inputs
# source=os.path.abspath(os.path.join('data','corsair','s42.lidar.z01.a0',
#                                     's42.lidar.z01.a0.20260406.150921.user1.nc'))
source=os.path.abspath(os.path.join('data','corsair','s19.lidar.z01.a0',
                                    's19.lidar.z01.a0.20260403.202255.user1.nc'))

save_fig=True

source_nwtc=os.path.abspath('data/CORSAIR_layout.xlsx')#source nwtc sites


min_SNR=10#[dB] minimum SNR for hard target
min_rws_std_bloc=15 # minimum std of rws for blocked beams [m/s]
max_SNR_bloc=-25#[dB] maximum mean near range SNR for blocked beams
max_range_bloc=500#[m] near range averaging distance for blocked beams
min_r=90#[m] minimum range
range_tol=100#[m] maximum |pred-obs| blockage range for a hit

home_azimuth={'Halo 200':12, 'Halo 199':3.55}#[deg]

#line-of-sight blockage model (same as planet/index.html)
source_dsm=r'C:\Users\sletizia\Software\FIEXTA\planet\planet\data\dsm'#lidar point cloud surface (make_dsm.py output)
terrain_url='https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'#bare earth
terrain_zoom=14#zoom of bare earth tiles
lidar_height=1#[m] beam origin above ground
ds=2#[m] ray sampling step
r_excl=0#[m] surface ignored around the lidar (lidar platform/footprint)
R_eff=4/3*6371000#[m] effective Earth radius for beam bending [Doviak and Zrnic, 1993]

#graphics
zoom=16
xmin,xmax,ymin,ymax=-1000,1000,-1000,1000

#%% Functions
def merc_px(lat,lon,z):
    n=256*2**z
    return (lon+180)/360*n, (1-np.arcsinh(np.tan(np.radians(lat)))/np.pi)/2*n

def bare_earth(px,py,z):
    tx=np.arange(int(px.min()//256),int(px.max()//256)+1)
    ty=np.arange(int(py.min()//256),int(py.max()//256)+1)
    mosaic=np.zeros((len(ty)*256,len(tx)*256))
    for i,y in enumerate(ty):
        for j,x in enumerate(tx):
            r=requests.get(terrain_url.format(z=z,x=x,y=y),timeout=60)
            r.raise_for_status()
            rgb=np.asarray(Image.open(io.BytesIO(r.content)).convert('RGB'),dtype=float)
            mosaic[i*256:(i+1)*256,j*256:(j+1)*256]=rgb[...,0]*256+rgb[...,1]+rgb[...,2]/256-32768
    return ndimage.map_coordinates(mosaic,[py-ty[0]*256-0.5,px-tx[0]*256-0.5],order=1,mode='nearest')

def surface(px,py,z,fallback):
    with open(source_dsm+'.json') as f:
        meta=json.load(f)
    rgba=np.asarray(Image.open(source_dsm+'.png').convert('RGBA'),dtype=float)
    dsm=rgba[...,0]*256+rgba[...,1]+rgba[...,2]/256-32768
    dsm[rgba[...,3]==0]=np.nan
    f=2**(meta['zoom']-z)
    i=np.floor(px*f).astype(int)-meta['px0']
    j=np.floor(py*f).astype(int)-meta['py0']
    inside=(i>=0)&(j>=0)&(i<meta['width'])&(j<meta['height'])
    out=np.full(px.shape,np.nan)
    out[inside]=dsm[j[inside],i[inside]]
    return np.where(np.isnan(out),fallback,out)

def nearest_blockage(r_obs,r_pred):
    d=np.abs(r_pred-r_obs[:,None])
    d=np.where(np.isnan(d),np.inf,d)
    r=r_pred[np.arange(len(r_obs)),d.argmin(axis=1)]
    return np.where(np.isinf(d.min(axis=1)),np.nan,r)

def contingency(r_obs,r_pred):
    obs=~np.isnan(r_obs)
    pred=(~np.isnan(r_pred)).any(axis=1)
    match=np.abs(nearest_blockage(r_obs,r_pred)-r_obs)<=range_tol
    #a beam is a hit if any predicted blockage is within range_tol of the observed one, otherwise a range mismatch is both a miss and a false alarm
    hit=np.sum(match)
    miss=np.sum(obs&~match)
    fa=np.sum(pred&~match)
    cn=np.sum(~obs&~pred)
    #probability of detection and false alarm ratio [Wilks, 2011]
    return {'hits':hit,'misses':miss,'false alarms':fa,'correct negatives':cn,
            'POD':hit/(hit+miss) if hit+miss>0 else np.nan,
            'FAR':fa/(hit+fa) if hit+fa>0 else np.nan,
            'accuracy':(hit+cn)/obs.size}

#%% Initialization

#lidar data
Data=xr.open_dataset(source)

#site layout
FC=pd.read_excel(source_nwtc,sheet_name='Instruments').set_index('Name')

site=f'Halo {Data.attrs["source"].split("_")[1]}'
lat0, lon0 =  FC['Latitude'].loc[site], FC['Longitude'].loc[site] #location of the lidar

#%% Main
Data=Data.where(Data.distance<np.max(np.abs(np.array([xmin,xmax,ymin,ymax]))),drop=True)
Data['SNR']=Data.SNR.where(~np.isnan(Data.SNR),Data.SNR.min())#fill null SNR

#Cartesian coordinates
Data['x']=Data.distance*np.cos(np.radians(90-(Data.azimuth+home_azimuth[site])))*np.cos(np.radians(Data.elevation))
Data['y']=Data.distance*np.sin(np.radians(90-(Data.azimuth+home_azimuth[site])))*np.cos(np.radians(Data.elevation))

#remove blind zone
Data['wind_speed']=Data.wind_speed.where(Data.distance>=min_r)
Data['SNR']=Data.SNR.where(Data.distance>=min_r)

#hard-target flag
high_SNR=(Data.SNR>min_SNR).transpose('time','range_gate').values
rws=Data.wind_speed.transpose('time','range_gate').values
ht=np.zeros(high_SNR.shape,dtype=bool)
for i,(hi,w) in enumerate(zip(high_SNR,rws)):
    #high-SNR runs are gates [start,end), followed by the low-SNR segment [end,next start)
    edges=np.diff(np.concatenate([[0],hi.astype(int),[0]]))
    starts=np.where(edges==1)[0]
    ends=np.where(edges==-1)[0]
    next_starts=np.append(starts[1:],len(hi))
    for i1,i2,i3 in zip(starts,ends,next_starts):
        #valid hard target only if the rws behind it is noise (no aerosol signal past the obstacle)
        w_behind=w[i2:i3]
        if np.sum(~np.isnan(w_behind))>1 and np.nanstd(w_behind)>min_rws_std_bloc:
            ht[i,i1:i2]=True

Data['ht']=(('time','range_gate'),ht)

#blocked beam flag
blocked_rws=Data.wind_speed.where(Data.distance<=max_range_bloc).std(dim='range_gate')>=min_rws_std_bloc
blocked_snr=Data.SNR.where(Data.distance<=max_range_bloc).mean(dim='range_gate')<=max_SNR_bloc
blocked=blocked_rws*blocked_snr

#raise hard target flag of blocked beam at the first gate beyond the blind zone
i_min_r=int(np.argmax(Data.distance.values>=min_r))
Data['ht'][dict(range_gate=i_min_r)] = Data.ht.isel(range_gate=i_min_r) | blocked

#extract hard targets
Data_ht=Data.where(Data.ht,drop=True)
x_ht=Data_ht.x.values.ravel()
y_ht=Data_ht.y.values.ravel()

#observed blockage: first hard target along each beam
r_obs=Data.distance.where(Data.ht).min(dim='range_gate')

#predicted blockages: every sample where the beam centerline goes from above to below the surface (s is slant range, as lidar distance)
rmax=float(Data.distance.max())
s=np.arange(ds,rmax+ds,ds)
azi_deg=(Data.azimuth.isel(range_gate=0)+home_azimuth[site])%360
ele=Data.elevation.isel(range_gate=0)
azi=np.radians(azi_deg.values)[:,None]
ele_rad=np.radians(ele.values)[:,None]
px0,py0=merc_px(lat0,lon0,terrain_zoom)
mpp=40075016.686*np.cos(np.radians(lat0))/(256*2**terrain_zoom)
px=px0+np.sin(azi)*s*np.cos(ele_rad)/mpp
py=py0-np.cos(azi)*s*np.cos(ele_rad)/mpp
g0=float(bare_earth(np.array([px0]),np.array([py0]),terrain_zoom)[0])
#beam height for slant range s [Doviak and Zrnic, 1993]
beam=g0+lidar_height+s*np.sin(ele_rad)+s**2/(2*R_eff)

z_be=bare_earth(px,py,terrain_zoom)
z_dsm=surface(px,py,terrain_zoom,z_be)

r_pred={}
for name,z in zip(['Bare earth','Lidar point cloud'],[z_be,z_dsm]):
    cut=(z>beam)&(s>r_excl)
    entry=cut&~np.pad(cut[:,:-1],((0,0),(1,0)))
    #blockages inside the blind zone are placed at min_r
    r_pred[name]=np.where(entry,np.maximum(s,min_r),np.nan)

#skill
obs=~np.isnan(r_obs.values)
print(f'Ground at lidar: {g0:.1f} m, beam origin: {g0+lidar_height:.1f} m, file altitude: {Data.attrs["altitude"]} m')
print(f'Observed blocked beams within {rmax:.0f} m: {obs.mean()*100:.1f}%')
skill={}
for name,r in r_pred.items():
    stats=skill[name]=contingency(r_obs.values,r)
    err=xr.DataArray(nearest_blockage(r_obs.values,r)-r_obs.values)
    print(f'{name}: '+', '.join(f'{k}={v:.2f}' if isinstance(v,float) else f'{k}={v}' for k,v in stats.items())+
          f', range error (pred-obs) median={float(err.median()):.0f} m, IQR={float(err.quantile(0.75)-err.quantile(0.25)):.0f} m')

#%% Plots
plt.close('all')
fig_dir=os.path.join('figures','hard_target')
fig_name=os.path.basename(source).replace('.nc','')
if save_fig:
    os.makedirs(fig_dir,exist_ok=True)

#aerial map
fig,ax=utl.aerial_map(x_ht,y_ht,lat0,lon0,markersize=5,alpha=1,zoom=zoom,
                      xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,label='Lidar hard target')
plt.title(str(os.path.basename(source))+fr', lidar height = {lidar_height} m a.g.l., $\Delta \alpha='+str(home_azimuth[site])+r'^\circ$')
plt.xlabel('E-W [m]')
plt.ylabel('S-N [m]')
plt.grid()

#predicted blockage on aerial map (same local-to-Web-Mercator projection as utl.aerial_map)
to_web=Transformer.from_crs(f'+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m +datum=WGS84','EPSG:3857',always_xy=True)
for (name,r),m in zip(r_pred.items(),['.c','.w']):
    x_web,y_web=to_web.transform((r*np.cos(np.pi/2-azi)*np.cos(ele_rad)).ravel(),
                                 (r*np.sin(np.pi/2-azi)*np.cos(ele_rad)).ravel())
    ax.plot(x_web,y_web,m,markersize=2.5,label=name,zorder=4)
ax.legend(markerscale=5)
if save_fig:
    plt.savefig(os.path.join(fig_dir,f'{fig_name}_aerial_map.png'))

#RWS and SNR maps
fig=plt.figure(figsize=(18,8))
plt.subplot(1,2,1)
pc=plt.pcolor(Data.x,Data.y,Data.wind_speed.T,vmin=-15,vmax=15,cmap='viridis')
plt.plot(x_ht,y_ht,'.r',markersize=1)
ax=plt.gca()
ax.set_aspect('equal')
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$y$ [m]')
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
plt.grid()
plt.colorbar(pc,label=r'Radial wind speed [m s$^{-1}$]')

plt.subplot(1,2,2)
pc=plt.pcolor(Data.x,Data.y,Data.SNR.T,vmin=-30,vmax=10,cmap='plasma')
plt.plot(x_ht,y_ht,'.r',markersize=1)
ax=plt.gca()
ax.set_aspect('equal')
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$y$ [m]')
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
plt.grid()
plt.colorbar(pc,label='SNR [dB]')

plt.suptitle(str(os.path.basename(source))+fr', lidar height = {lidar_height} m a.g.l., $\Delta \alpha='+str(home_azimuth[site])+r'^\circ$')
if save_fig:
    plt.savefig(os.path.join(fig_dir,f'{fig_name}_rws_snr.png'))


#blockage range vs azimuth
plt.figure(figsize=(12,4))
plt.plot(azi_deg,r_obs,'.r',markersize=5,alpha=0.5,label='Observed (hard target)')
for (name,r),c in zip(r_pred.items(),['c','w']):
    plt.plot(np.broadcast_to(azi_deg.values[:,None],r.shape).ravel(),r.ravel(),'.',color=c,markersize=2.5,alpha=0.5,label=name)
plt.xlabel(r'$\alpha$ [$^\circ$]')
plt.ylabel('Blockage range [m]')
plt.gca().set_facecolor((0.5,0.5,0,0.5))
plt.xlim([0,360])
plt.ylim([0,rmax])
plt.grid()
plt.legend(markerscale=5)
plt.title(str(os.path.basename(source))+fr', lidar height = {lidar_height} m a.g.l., $\Delta \alpha='+str(home_azimuth[site])+r'^\circ$')
if save_fig:
    plt.savefig(os.path.join(fig_dir,f'{fig_name}_range_vs_azimuth.png'))

#contingency tables
fig,axs=plt.subplots(1,len(skill),figsize=(12,3.5))
for ax,(name,st) in zip(axs,skill.items()):
    ax.axis('off')
    tab=ax.table(cellText=[[st['hits'],st['false alarms']],[st['misses'],st['correct negatives']]],
                 rowLabels=['Pred. blocked','Pred. clear'],colLabels=['Obs. blocked','Obs. clear'],
                 cellLoc='center',loc='center')
    tab.scale(1,2.5)
    ax.set_title(f'{name}\nPOD={st["POD"]:.2f}, FAR={st["FAR"]:.2f}, accuracy={st["accuracy"]:.2f}')
plt.suptitle(str(os.path.basename(source))+fr', range tolerance = {range_tol} m')
plt.tight_layout()
if save_fig:
    plt.savefig(os.path.join(fig_dir,f'{fig_name}_contingency.png'))