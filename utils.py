# -*- coding: utf-8 -*-
"""
Utilities
"""

import numpy as np
import xarray as xr
import os
cd=os.path.dirname(__file__)
from matplotlib import pyplot as plt

#%% Geometry
def angle_difference_deg(a1, a2):
    return (a2 - a1 + 180) % 360 - 180 

def cosd(x):
    return np.cos(np.radians(x))

def sind(x):
    return np.sin(np.radians(x))

#%% Statistics
def filt_stat(x, func, perc_lim=[5, 95]):
    x = x.copy()
    x = x[np.isfinite(x)]
    x = x[(x >= np.nanpercentile(x, perc_lim[0])) & (x <= np.nanpercentile(x, perc_lim[1]))]
    return func(x)

def filt_BS_stat(x, func, p_value=5, M_BS=100, min_N=10, perc_lim=[5, 95]):
    x = x.copy()
    x = x[np.isfinite(x)]
    x = x[(x >= np.nanpercentile(x, perc_lim[0])) & (x <= np.nanpercentile(x, perc_lim[1]))]
    if len(x) < min_N or len(x) == 1:
        return np.nan
    x_BS = x[np.random.randint(0, len(x), size=(M_BS, len(x)))]
    return np.nanpercentile(func(x_BS, axis=1), p_value)

def mean_ci(f, max_ci, p_value=0.05, perc_lim=[5, 95]):
    f = np.asarray(f, dtype=float)
    f_avg = filt_stat(f, np.nanmean, perc_lim=perc_lim)
    f_low = filt_BS_stat(f, np.nanmean, perc_lim=perc_lim, p_value=p_value / 2 * 100)
    f_top = filt_BS_stat(f, np.nanmean, perc_lim=perc_lim, p_value=(1 - p_value / 2) * 100)
    if f_top - f_low > max_ci or np.isnan(f_top - f_low):
        f_avg = np.nan
    return f_avg, f_low, f_top

#%% Data processing
def format_file(file,save_path,delete,config,logfile_main,replace):
    '''
    Format file
    '''
    import lidargo as lg
    from datetime import datetime
    import traceback
    
    try:
        logfile=os.path.join(cd,'log',os.path.basename(file).replace('hpl','log'))
        lproc = lg.Format(file, config=config['path_config_format'], verbose=True,logfile=logfile)
        lproc.process_scan(replace=replace, save_file=True,save_path=save_path)
        
        if delete:
            os.remove(file)
            
    except:
        with open(logfile_main, 'a') as lf:
            lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error formatting file {os.path.basename(file)}: \n")
            traceback.print_exc(file=lf)
            lf.write('\n --------------------------------- \n')
            
def standardize_file(file,save_path_stand,config,logfile_main,sdate,edate):
    '''
    Standardize data file    
    '''
    import traceback
    import re
    from datetime import datetime
    import lidargo as lg
    
    date=re.search(r'\d{8}.\d{6}',file).group(0)[:8]
    if datetime.strptime(date,'%Y%m%d')>=datetime.strptime(sdate,'%Y-%m-%d') and datetime.strptime(date,'%Y%m%d')<=datetime.strptime(edate,'%Y-%m-%d'):
        try:
            logfile=os.path.join(cd,'log',os.path.basename(file).replace('nc','log'))
            lproc = lg.Standardize(file, config=config['path_config_stand'], verbose=True,logfile=logfile)
            lproc.process_scan(replace=False, save_file=True, save_path=save_path_stand)
        except:
            with open(logfile_main, 'a') as lf:
                lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error standardizing file {os.path.basename(file)}: \n")
                traceback.print_exc(file=lf)
                lf.write('\n --------------------------------- \n')
            
def lisboa_file(file,config_path,logfile_main,sdate,edate,delete,replace):
    
    '''
    Apply LiSBOA statistics reconstruction on RWS data for each file
    '''
    import re
    from datetime import datetime
    import utm
    from lisboa import statistics as stats
    import socket
    import getpass
    import traceback
    import sys
    date=re.search(r'\d{8}.\d{6}',file).group(0)[:8]
    if datetime.strptime(date,'%Y%m%d')>=datetime.strptime(sdate,'%Y-%m-%d') and datetime.strptime(date,'%Y%m%d')<=datetime.strptime(edate,'%Y-%m-%d'):
        try:
            logfile=os.path.join(cd,'log',os.path.basename(file).replace('nc','log'))
         
            #load config
            config,config_lisboa=load_config_from_file(config_path,file)
            save_path=file.replace(config['data_level_in'],'c0')
            if not os.path.isfile(save_path) or replace:
                
                if config is not None:
                    #load data
                    Data=xr.open_dataset(file)
                    time=Data.time
                    Data=Data.where(Data.qc_wind_speed==0)
                    
                    #origin
                    x0,y0,zone_num0,zone_str0=utm.from_latlon(config['origin_lat'], config['origin_lon'])
                    z0=config['origin_alt']
                    x_lidar,y_lidar,zone_num_lidar,zone_str_lidar=utm.from_latlon(Data.attrs['latitude'], Data.attrs['longitude'])
                    z_lidar=Data.attrs['altitude']
                    assert zone_num0==zone_num_lidar and zone_str0==zone_str_lidar, "Mismatiching UTM zones"
                    
                    #build LiSBOA input data
                    if len(config['Dn0'])==3:
                        x_exp=[Data.x.values.ravel()+x_lidar-x0,
                               Data.y.values.ravel()+y_lidar-y0,
                               Data.z.values.ravel()+z_lidar-z0]
                    else:
                        x_exp=[Data.x.values.ravel()+x_lidar-x0,
                               Data.y.values.ravel()+y_lidar-y0]
                    
                    f=Data.wind_speed.values.ravel()
                    
                    #thresholding
                    f[f<config['limits'][0]]=np.nan
                    f[f>config['limits'][1]]=np.nan
                    
                    #run LiSBOA
                    lproc=stats.statistics(config_lisboa,logfile=logfile)
                    grid,Dd,excl,avg,hom=lproc.calculate_statistics(x_exp,f,2)
                    avg[avg<config['limits'][0]]=np.nan
                    avg[avg>config['limits'][1]]=np.nan
                    
                    #% Output
                    Output=xr.Dataset()
                    if len(grid)==3:
                        coords={'x':grid[0],'y':grid[1],'z':grid[2]}
                    else:
                        coords={'x':grid[0],'y':grid[1]}
                    Output['rws_avg']=xr.DataArray(avg,coords=coords,
                                                 attrs={'units':'m/s','description':'mean LOS velocity'})
                    Output['rws_std']=xr.DataArray(hom**0.5,coords=coords,
                                                 attrs={'units':'m/s','description':'std of LOS velocity'})

                    #specific attributes
                    Output.attrs['start_time']=str(time.isel(beamID=0,scanID=0).values)
                    Output.attrs['end_time']=  str(time.isel(beamID=-1,scanID=-1).values)
                    
                    for c in config:
                        Output.attrs[f'config_{c}']=config[c]
                    
                    Output.attrs["x_lidar"]=x_lidar-x0
                    Output.attrs["y_lidar"]=y_lidar-y0
                    Output.attrs['input_source']=os.path.basename(file)
                    
                    #general attributes
                    Output.attrs.update({
                        'title':        'LiSBOA wind statistics',
                        'description':  'Statistics of de-projected wind speed calculated through LiSBOA based on Letizia et al., 2021 (10.5194/amt-14-2065-2021)',
                        'contact':      'stefano.letizia@nlr.gov',
                        'institution':  'National Laboratory of the Rockies',
                        'conventions':  'MHKiT-Cloud Data Standards v. 1.0',
                        'location_id':  Data.attrs['location_id'],
                        'history':      (f"Generated by {getpass.getuser()} on {socket.gethostname()} on "
                                         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using "
                                         f"{os.path.basename(sys.argv[0])}"),
                        'code':             'https://github.com/StefanoWind/CORSAIR_analysis',
                        'dependencies': 'https://github.com/NatLabRockies/FIEXTA/tree/main/lisboa',
                        'code_version': _git_hash(),
                    })
                    

                    os.makedirs(os.path.dirname(save_path),exist_ok=True)
                    Output.to_netcdf(save_path)
                    
                    visualize_volume(Output,config,save_path)
                    
                    if delete:
                        os.remove(file)
        except:
            with open(logfile_main, 'a') as lf:
                lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error processing file {os.path.basename(file)}: \n")
                traceback.print_exc(file=lf)
                lf.write('\n --------------------------------- \n')
    
def dual_doppler_reconstruction(Data1:xr.Dataset(),
                                Data2:xr.Dataset(),
                                sigma_rws: float=0.1,
                                sigma_w:float=1,
                                save_path:str='',
                                logfile_main:str=None,
                                sdate:str='1970-01-01',
                                edate:str='2070-01-01',
                                replace:bool=True):
    '''
    Perform dual-Doppler reconstruction from two LiSBOA outputs.
    '''
    import socket
    import getpass
    from datetime import datetime
    import traceback
    import sys
    
    if np.datetime64(Data1.attrs['start_time'])>=np.datetime64(sdate+'T00:00:00') and\
       np.datetime64(Data1.attrs['end_time'])  <=np.datetime64(edate+'T23:59:59'):
        try:
            if not os.path.isfile(save_path) or replace:

                if 'z' not in Data1.dims or 'z' not in Data2.dims:
                    msg = "dual_doppler_reconstruction requires 3D LiSBOA grids (z dimension required). Aborting."
                    if logfile_main is not None:
                        with open(logfile_main, 'a') as lf:
                            lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - {msg}\n")
                    else:
                        print(msg)
                    return None

                #check that coordinates match (use allclose for float robustness)
                if not (np.allclose(Data1.x.values, Data2.x.values, atol=1e-9) and
                        np.allclose(Data1.y.values, Data2.y.values, atol=1e-9) and
                        np.allclose(Data1.z.values, Data2.z.values, atol=1e-9)):

                    msg = "Mismatching coordinates, aborting dual-Doppler reconstruction. Aborting."
                    if logfile_main is not None:
                        with open(logfile_main, 'a') as lf:
                            lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - {msg}\n")
                    else:
                        print(msg)
                        
                    return None
                
                #get spherical coordinates for lidar 1
                x1=Data1.x-Data1.attrs['x_lidar']
                y1=Data1.y-Data1.attrs['y_lidar']
                z1=Data1.z
                r1=(x1**2+y1**2+z1**2)**0.5
                sin_ele1=(z1/(r1+10**-16)).transpose('x','y','z')
                cos_ele1=(1-sin_ele1**2)**0.5
                cos_azi1= x1/(r1+10**-16)/cos_ele1
                sin_azi1=(y1/(r1+10**-16)/cos_ele1).transpose('x','y','z')
                
                #get spherical coordinates for lidar 2
                x2=Data2.x-Data2.attrs['x_lidar']
                y2=Data2.y-Data2.attrs['y_lidar']
                z2=Data2.z
                r2=(x2**2+y2**2+z2**2)**0.5
                sin_ele2=(z2/(r2+10**-16)).transpose('x','y','z')
                cos_ele2=(1-sin_ele2**2)**0.5
                cos_azi2= x2/(r2+10**-16)/cos_ele2
                sin_azi2=(y2/(r2+10**-16)/cos_ele2).transpose('x','y','z')
                
                #build forward matrix
                a=cos_azi1*cos_ele1
                b=sin_azi1*cos_ele1
                c=cos_azi2*cos_ele2
                d=sin_azi2*cos_ele2
                
                #build inverse matrix
                det=(a*d-b*c)
                a_inv=+d/det
                b_inv=-b/det
                c_inv=-c/det
                d_inv=+a/det
                
                #reconstructed wind field
                U=a_inv*Data1.rws_avg+b_inv*Data2.rws_avg
                V=c_inv*Data1.rws_avg+d_inv*Data2.rws_avg
                WS=(U**2+V**2)**0.5
                WD=(270-np.degrees(np.arctan2(V,U)))%360
                
                #error factor
                sigma_U=(a_inv**2*(sigma_rws**2+sin_ele1**2*sigma_w**2)\
                        +b_inv**2*(sigma_rws**2+sin_ele2**2*sigma_w**2)\
                        +2*a_inv*b_inv*sin_ele1*sin_ele2   *sigma_w**2)**0.5
                sigma_V=(c_inv**2*(sigma_rws**2+sin_ele1**2*sigma_w**2)\
                        +d_inv**2*(sigma_rws**2+sin_ele2**2*sigma_w**2)\
                        +2*c_inv*d_inv*sin_ele1*sin_ele2   *sigma_w**2)**0.5
                
                sigma_UV=a_inv*c_inv*(sigma_rws**2+sin_ele1**2*sigma_w**2)\
                        +b_inv*d_inv*(sigma_rws**2+sin_ele2**2*sigma_w**2)\
                        +(a_inv*d_inv+b_inv*c_inv)*sin_ele1*sin_ele2*sigma_w**2
                    
                sigma_WS=((U/WS*sigma_U   )**2+(V/WS*sigma_V   )**2+2*U*V/WS**2*sigma_UV)**0.5
                sigma_WD=((V/WS**2*sigma_U)**2+(U/WS**2*sigma_V)**2-2*U*V/WS**4*sigma_UV)**0.5*180/np.pi
                
                #output
                Output=xr.Dataset()
                Output['U']=U
                Output['U'].attrs={'units':'m/s','description':'reconstructed mean W-E velocity'}
                Output['V']=V
                Output['V'].attrs={'units':'m/s','description':'reconstructed mean S-N velocity'}
                Output['WS']=WS
                Output['WS'].attrs={'units':'m/s','description':'reconstructed mean horizontal wind speed'}
                Output['WD']=WD
                Output['WD'].attrs={'units':'degrees','description':'reconstructed mean horizontal wind direction'}
                
                Output['sigma_U']=sigma_U
                Output['sigma_U'].attrs={'units':'','description':'error factor of reconstructed mean W-E velocity'}
                Output['sigma_V']=sigma_V
                Output['sigma_V'].attrs={'units':'','description':'error factor of reconstructed mean S-N velocity'}
                Output['sigma_WS']=sigma_WS
                Output['sigma_WS'].attrs={'units':'','description':'error factor of reconstructed mean horizontal wind speed'}
                Output['sigma_WD']=sigma_WD
                Output['sigma_WD'].attrs={'units':'degrees/(m/s)','description':'error factor of reconstructed mean horizontal wind direction'}
                
                #specific attributes
                Output.attrs['start_time']=Data1.attrs['start_time']
                Output.attrs['end_time']=  Data1.attrs['end_time']
                Output.attrs['location_id1']=Data1.attrs['location_id']
                Output.attrs['location_id2']=Data2.attrs['location_id']
                Output.attrs['origin_lat']=Data1.attrs['config_origin_lat']
                Output.attrs['origin_lon']=Data1.attrs['config_origin_lon']
                Output.attrs['origin_alt']=Data1.attrs['config_origin_alt']
                Output.attrs['sigma_rws']=Data1.attrs['config_sigma_rws']
                Output.attrs['sigma_w']=Data1.attrs['config_sigma_w']
                
                #general attributes
                Output.attrs.update({
                    'title':        'Dual-Doppler wind maps',
                    'description':  ('Wind statistics from dual-Doppler measurements interpolated on Cartesian grid'),
                    'contact':      'stefano.letizia@nlr.gov',
                    'institution':  'National Laboratory of the Rockies',
                    'conventions':  'MHKiT-Cloud Data Standards v. 1.0',
                    'history':      (f"Generated by {getpass.getuser()} on {socket.gethostname()} on "
                                     f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using "
                                     f"{os.path.basename(sys.argv[0])}"),
                    'code':             'https://github.com/StefanoWind/CORSAIR_analysis',
                    'code_version': _git_hash(),
                })
                
                if save_path != '':
                    os.makedirs(os.path.dirname(save_path),exist_ok=True)
                    Output.to_netcdf(save_path)
                  
                return Output

        except:
            if logfile_main is not None:
                with open(logfile_main, 'a') as lf:
                    lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error in creation of dual-Doppler file {os.path.basename(save_path)}: \n")
                    traceback.print_exc(file=lf)
            else:
                print(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error in creation of dual-Doppler file {os.path.basename(save_path)}: \n")
            return None
    else:
        if logfile_main is not None:
            with open(logfile_main, 'a') as lf:
                lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - INFO - Skipping dual-Doppler reconstruction: start_time outside [{sdate}, {edate}].\n")
        return None


def dual_doppler_error(x_lidar1: float, y_lidar1: float,
                       x_lidar2: float, y_lidar2: float,
                       x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       WS: float, WD: float,
                       sigma_rws: float = 1,
                       sigma_w: float = 1) -> xr.Dataset:

    coords = {'x': x, 'y': y, 'z': z}

    # wind components from speed and meteorological direction
    U = -WS * np.sin(np.radians(WD))
    V = -WS * np.cos(np.radians(WD))

    x_xr = xr.DataArray(x, dims='x', coords={'x': x})
    y_xr = xr.DataArray(y, dims='y', coords={'y': y})
    z_xr = xr.DataArray(z, dims='z', coords={'z': z})

    def forward_rws(x_lidar, y_lidar):
        dx = x_xr - x_lidar
        dy = y_xr - y_lidar
        r = (dx**2 + dy**2 + z_xr**2)**0.5
        sin_ele = (z_xr / (r + 1e-16)).transpose('x', 'y', 'z')
        cos_ele = (1 - sin_ele**2)**0.5
        cos_azi = dx / (r + 1e-16) / cos_ele
        sin_azi = (dy / (r + 1e-16) / cos_ele).transpose('x', 'y', 'z')
        return (cos_ele * (cos_azi * U + sin_azi * V)).transpose('x', 'y', 'z')

    dummy_attrs = {
        'start_time': '2000-01-01T00:00:00',
        'end_time':   '2000-01-01T00:00:00',
        'site': '',
        'config_origin_lat': 0,
        'config_origin_lon': 0,
    }

    Data1 = xr.Dataset({'rws_avg': forward_rws(x_lidar1, y_lidar1)}, coords=coords)
    Data1.attrs = {**dummy_attrs, 'x_lidar': float(x_lidar1), 'y_lidar': float(y_lidar1)}

    Data2 = xr.Dataset({'rws_avg': forward_rws(x_lidar2, y_lidar2)}, coords=coords)
    Data2.attrs = {**dummy_attrs, 'x_lidar': float(x_lidar2), 'y_lidar': float(y_lidar2)}

    Output = dual_doppler_reconstruction(Data1, Data2, sigma_rws=sigma_rws, sigma_w=sigma_w)

    if Output is None:
        return None
    return Output[['sigma_U', 'sigma_V', 'sigma_WS', 'sigma_WD']]


#%% Graphics
def matrix_plt(x,y,f,cmap,vmin,vmax):
    '''
    Plot matrix with color and display values
    '''
    # trim always-NaN borders: last Lidar-1 column and first Lidar-2 row
    x_trim = x[:-1]
    y_trim = y[1:]
    f_trim = f[:-1, 1:]
    nx, ny = len(x_trim), len(y_trim)

    pc = plt.pcolor(np.arange(nx), np.arange(ny), f_trim.T,
                    cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax = plt.gca()

    # separation lines between cells
    for k in range(1, nx):
        ax.axvline(k - 0.5, color='w', linewidth=1.5, zorder=3)
    for k in range(1, ny):
        ax.axhline(k - 0.5, color='w', linewidth=1.5, zorder=3)

    for i in range(nx):
        for j in range(ny):
            if ~np.isnan(f_trim[i, j]):
                ax.text(i, j, f"{f_trim[i, j]:.2f}",
                        ha='center', va='center', color='k', fontsize=10, fontweight='bold')

    ax.set_xticks(np.arange(nx))
    ax.set_xticklabels(x_trim)
    ax.set_yticks(np.arange(ny))
    ax.set_yticklabels(y_trim)

    return pc

def aerial_map(x,y,lat0,lon0,zoom=15,color='r',markersize=10,alpha=1,
               xmin=None,xmax=None,ymin=None,ymax=None,ax=None):
    '''
    Draw aerail map and superpose points
    '''
    import requests
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    
    # 1. Suppress the annoying warning messages
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    
    # 2. Force 'verify=False' for every single request made in this session
    old_merge_environment_settings = requests.Session.merge_environment_settings
    
    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False
        return settings
    
    requests.Session.merge_environment_settings = merge_environment_settings
    
    import contextily as cx
    import matplotlib.pyplot as plt
    import pandas as pd
    import geopandas as gpd
    
    # Create a custom PROJ string for AEQD centered on your origin
    custom_crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m +datum=WGS84"
    
    # Create the GeoDataFrame using your custom CRS
    df = pd.DataFrame({'x':x,'y':y})
    
    gdf = gpd.GeoDataFrame(df, 
    geometry=gpd.points_from_xy(df.x, df.y), 
    crs=custom_crs)
    
   
    # 2. Transform these corners to Web Mercator (just like your points)

    
    # Re-project for the aerial map background
    gdf_web = gdf.to_crs(epsg=3857)
   
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    # Plot your relative points
    gdf_web.plot(ax=ax, color=color, markersize=markersize, alpha=alpha,zorder=3)
    
    # Impose edges
    if xmin is not None:
        corners = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([xmin, xmax], [ymin, ymax]),
        crs=custom_crs
        )
        corners_web = corners.to_crs(epsg=3857)
        ax.set_xlim(list(corners_web.geometry.x))
        ax.set_ylim(list(corners_web.geometry.y))

    # Add high-res aerial imagery
    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery,zoom=zoom)
    
    return fig,ax

def visualize_volume(Data,config,save_path):
    '''
    Visualize volume of  mean RWS data
    '''
    from matplotlib.gridspec import GridSpec
    import os
    
    #Extract coordinates
    x=Data.x.values
    y=Data.y.values
    z=Data.z.values
        
    #contour levels
    rws_avg=Data['rws_avg'].values
    levels_u=np.unique(np.round(np.linspace(np.nanpercentile(rws_avg,5)-0.5, np.nanpercentile(rws_avg,95)+0.5, 20),1))
    
    #Plot mean LOS velocity at several height
    fig=plt.figure(figsize=(18,10))
    ncols=int(np.floor(len(config['plot_heights']))/2)+1
    gs = GridSpec(nrows=2, ncols=ncols, width_ratios=[1,1,0.05], figure=fig)
    
    ctr=0
    for iz in config['plot_heights']:
        ax = fig.add_subplot(gs[int(ctr/2),ctr%(ncols-1)])
        ax.set_facecolor((0,0,0,0.2))
    
        cf=plt.contourf(x,y,Data['rws_avg'].isel(z=iz).T,levels_u, cmap='coolwarm',extend='both')
        plt.contour(x,y,Data['rws_avg'].isel(z=iz).T,levels_u, colors='k',linewidths=1,alpha=0.25,extend='both')
        plt.xlim([config['mins'][0],config['maxs'][0]])
        plt.ylim([config['mins'][1],config['maxs'][1]])
        plt.grid(alpha=0.5)
        ax.set_aspect('equal')
        plt.xlabel(r'$x$ [m]')
        plt.ylabel(r'$y$ [m]')
        
        plt.title(r'$z='+str(z[iz]).replace('.0','')+'$ m a.g.l.')
        ctr+=1
        
    cax=fig.add_subplot(gs[:,-1])
    plt.colorbar(cf,cax,label=r'Mean radial wind speed [m s$^{-1}$]')
    
    plt.suptitle('Mean radial wind speed at '+Data.attrs['location_id']+' on '+Data.attrs['start_time'][:10]+'\n File: '+os.path.basename(Data.attrs['input_source'])\
              +'\n Time (UTC): '+Data.attrs['start_time'][11:19]+' - '+Data.attrs['end_time'][11:19])
   
    fig.savefig(save_path.replace('.nc','_rws_avg.png'))
    plt.close()
    
def plot_wind_map(Data,
                  max_sigma={'U':10,'V':10,'WS':10,'WD':50},
                  heights=[0,4,8,12],
                  save_path='',
                  stride=3,
                  path_layout='',
                  markers={},
                  levels=None,
                  perc_min=5,
                  perc_max=95,
                  topo=None,
                  topo_levels=10):
    '''
    Plot dual-Doppler wind map
    '''
    from matplotlib.gridspec import GridSpec
    import pandas as pd
    import utm
    from matplotlib.markers import MarkerStyle
    star_marker = MarkerStyle(three_point_star())
    
    #layout
    Layout=pd.read_excel(path_layout,sheet_name='Assets')
    Layout['x'],Layout['y'],_,_=utm.from_latlon(Layout['Latitude'].values, Layout['Longitude'].values)
    x0,y0,_,_=utm.from_latlon(Data.attrs['origin_lat'],Data.attrs['origin_lon'])
    
    #QC
    Data=Data.where(Data.sigma_U<max_sigma['U'])\
             .where(Data.sigma_V<max_sigma['V'])\
             .where(Data.sigma_WS<max_sigma['WS'])\
             .where(Data.sigma_WD<max_sigma['WD'])\
 
    
    #Plot mean LOS velocity at several heights
    fig=plt.figure(figsize=(18,10))
    ncols=int(np.floor(len(heights))/2)+1
    gs = GridSpec(nrows=2, ncols=ncols, width_ratios=[1,1,0.05], figure=fig)
      
    if levels is None:
        levels=np.unique(np.round(np.linspace(np.nanpercentile(Data.WS,perc_min)-0.5, 
                                              np.nanpercentile(Data.WS,perc_max)+0.5, 20),1))
    
    if len(levels)>=2:
    
        ctr=0
        for iz in heights:
            ax = fig.add_subplot(gs[int(ctr/2),ctr%(ncols-1)])
            ax.set_facecolor((0,0,0,0.2))
            
            #heatmap
            cf=plt.contourf(Data.x,Data.y,Data.WS.isel(z=iz).T,levels, cmap='coolwarm',extend='both')
            plt.contour(    Data.x,Data.y,Data.WS.isel(z=iz).T,levels, color='k',alpha=0.25,linewidths=1,extend='both')
            
            #quiver
            x_q=Data.x.values[::stride]
            y_q=Data.y.values[::stride]
            u_q = Data.U.isel(z=iz)[::stride, ::stride]
            v_q = Data.V.isel(z=iz)[::stride, ::stride]
            
            ax.quiver(x_q,y_q,         
                u_q.values.T,                     
                v_q.values.T,                   
                angles="xy",
                pivot='middle',
                scale=levels[-1]/100,                              
                scale_units="xy",
                width=0.003,
                color="k",
                alpha=0.8)
            
            #topography
            if topo is not None:
                ax.contour(topo.x.values, topo.y.values, topo.z.values.T,
                           colors='k', linewidths=0.5, alpha=0.4, levels=topo_levels)

            #layout
            for m in markers.keys():
                sel=Layout['Description']==m
                xp=Layout[sel]['x'].values-x0
                yp=Layout[sel]['y'].values-y0
                try:
                    plt.plot(xp,yp,'xk', marker=markers[m], markersize=10, color='g',label=m)
                except ValueError as e:
                    if "Unrecognized marker style" in str(e):
                        plt.plot(xp,yp,'xk', marker=eval(markers[m]), markersize=10, color='g',label=m)
                        
            #decorations
            plt.xlim([Data.x.min(),Data.x.max()])
            plt.ylim([Data.y.min(),Data.y.max()])
            
            plt.grid(alpha=0.5)
            ax.set_aspect('equal')
            plt.xlabel(r'$x$ [m]')
            plt.ylabel(r'$y$ [m]')
            
            if ctr==0:
                plt.legend()
            
            plt.title(r'$z='+str(Data.z.values[iz]).replace('.0','')+'$ m a.g.l.')
            ctr+=1
            
        cax=fig.add_subplot(gs[:,-1])
        plt.colorbar(cf,cax,label=r'LiSBOA-averaged horizontal wind speed [m s$^{-1}$]')
        
        plt.suptitle('LiSBOA-averaged horizontal velocity on '+Data.attrs['start_time'][:10]+\
                     '\n Synthesized from: '+Data.attrs['location_id1'] +' and ' + Data.attrs['location_id2'] \
                    +'\n Time (UTC): '+Data.attrs['start_time'][11:19]+' - '+Data.attrs['end_time'][11:19])
            
        plt.tight_layout()
            
        if save_path is not None:
             fig.savefig(save_path)
        plt.close()
    else:
        print('Not enough levels to generate wind map.',flush=True)
        return None


def three_point_star():
    ''' 
    Points of a 3-pointed star (scaled and centered)
    '''

    from matplotlib.path import Path
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points (3 outer, 3 inner)
    outer_radius = 1
    inner_radius = 0.1
    coords = []

    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        coords.append((x, y))

    coords.append(coords[0])  # close the shape
    return Path(coords)


#%% Others
def load_config_from_file(config_file: str, source: str):
    import pandas as pd
    import re
    import ast
    """
    Load configuration from an Excel file.

    Args:
        config_file (str): Path to Excel configuration file

    Returns:
        LidarConfig or None: Configuration parameters or None if loading fails
    """
    configs = pd.read_excel(config_file,header=None).set_index(0)
    date_source = np.int64(re.search(r"\d{8}", source).group(0))

    matches = []
    for c in configs.columns:
        regex=configs[c]['regex']
        if "start_date" not in  configs[c]:
            sdate=19700101
        else:
            sdate = configs[c]["start_date"]
        if "end_date" not in  configs[c]:
            edate=30000101
        else:
            edate = configs[c]["end_date"]
        
        match = re.findall(regex, source)
        if len(match) > 0 and sdate <= date_source <= edate:
            matches.append(c)

    if not matches:
        return None
        
    elif len(matches) > 1:
        return None
    
    config=configs[matches[0]].to_dict()
    
    #read literal lists
    for s in ['mins','maxs','Dn0','limits','plot_heights']:
        config[s]=list(np.array(ast.literal_eval(config[s])))
    
    config_lisboa=config.copy()
    del config_lisboa['regex']
    del config_lisboa['start_date']
    del config_lisboa['end_date']
    del config_lisboa['data_level_in']
    del config_lisboa['limits']
    del config_lisboa['origin_lat']
    del config_lisboa['origin_lon']
    del config_lisboa['plot_heights']
    del config_lisboa['channel_name']
    del config_lisboa['origin_alt']
    del config_lisboa['stride_map']
    del config_lisboa['sigma_rws']
    del config_lisboa['sigma_w']
    
    return config,config_lisboa
    
def date_from_file(file,pattern=r'(\d{8}\.\d{6})',fmt='%Y%m%d.%H%M%S'):
    '''
    Extract datetime from filename
    '''
    import re
    from datetime import datetime
    
    match=re.search(pattern, file)
    if match is not None:
        return np.datetime64(datetime.strptime(match.group(0), fmt))
    else:
        return None
    
def _git_hash():
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cd,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ''
    
  