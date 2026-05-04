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
            save_path=file.replace(config['data_level_in'],'c1')
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

                    Output.attrs['start_time']=str(time.isel(beamID=0,scanID=0).values)
                    Output.attrs['end_time']=  str(time.isel(beamID=-1,scanID=-1).values)
                    
                    for c in config:
                        Output.attrs[f'config_{c}']=config[c]
                    
                    Output.attrs["x_lidar"]=x_lidar-x0
                    Output.attrs["y_lidar"]=y_lidar-y0
                    Output.attrs['input_source']=os.path.basename(file)
                    Output.attrs["contact"]= "stefano.letizia@nlr.gov"
                    Output.attrs["institution"]= "NLR"
                    Output.attrs["description"]= "Statistics of de-projected wind speed calculated through LiSBOA"
                    Output.attrs["reference"]= "Letizia et al. LiSBOA (LiDAR Statistical Barnes Objective Analysis) for optimal design of lidar scans and retrieval of wind statistics – Part 1: Theoretical framework. AMT, 14, 2065–2093, 2021, 10.5194/amt-14-2065-2021"
                    Output.attrs["history"]= (
                        f"Generated by {getpass.getuser()} on {socket.gethostname()} on "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using {os.path.basename(sys.argv[0])}"
                    )
                    Output.attrs["code1"]="https://github.com/NREL/FIEXTA/tree/main/lisboa"
                    Output.attrs["code2"]="https://github.com/StefanoWind/CORSAIR_analysis" 
                    Output.attrs["site"]=Data.attrs['site']
                    Output.attrs["location_id"]=Data.attrs['location_id']

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
                                sigma_rws: float=1,
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
                
                #check that coordinates match
                if ((Data1.x==Data2.x)==False).any() or ((Data1.y==Data2.y)==False).any() or ((Data1.z==Data2.z)==False).any():
                    print("Mimatching coordinates, aborting dual-Doppler reconstruction")
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
                sin_ele2=(z2/(r2+20**-26)).transpose('x','y','z')
                cos_ele2=(1-sin_ele2**2)**0.5
                cos_azi2= x2/(r2+20**-26)/cos_ele2
                sin_azi2=(y2/(r2+20**-26)/cos_ele2).transpose('x','y','z')
                
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
                sigma_WS=((U/WS*sigma_U   )**2+(V/WS*sigma_V   )**2)**0.5
                sigma_WD=((V/WS**2*sigma_U)**2+(U/WS**2*sigma_V)**2)**0.5*180/np.pi
                
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
                
                #attributes
                Output.attrs['start_time']=Data1.attrs['start_time']
                Output.attrs['end_time']=  Data1.attrs['end_time']
                Output.attrs['site1']=Data1.attrs['site']
                Output.attrs['site2']=Data2.attrs['site']
                Output.attrs['origin_lat']=Data1.attrs['config_origin_lat']
                Output.attrs['origin_lon']=Data1.attrs['config_origin_lon']
                Output.attrs["contact"]= "stefano.letizia@nlr.gov"
                Output.attrs["institution"]= "NLR"
                Output.attrs["description"]= "Statistics of de-projected wind speed calculated through LiSBOA"
                Output.attrs["reference"]= "Letizia et al. LiSBOA (LiDAR Statistical Barnes Objective Analysis) for optimal design of lidar scans and retrieval of wind statistics – Part 1: Theoretical framework. AMT, 14, 2065–2093, 2021, 10.5194/amt-14-2065-2021"
                Output.attrs["history"]= (
                    f"Generated by {getpass.getuser()} on {socket.gethostname()} on "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using {os.path.basename(sys.argv[0])}"
                )
                Output.attrs["code1"]="https://github.com/NREL/FIEXTA/tree/main/lisboa"
                Output.attrs["code2"]="https://github.com/StefanoWind/CORSAIR_analysis" 
                
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
        return None
        
   

#%% Graphics
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

def aerial_map(x,y,lat0,lon0,zoom=15,color='r',markersize=10,alpha=1,
               xmin=None,xmax=None,ymin=None,ymax=None):
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
   
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
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
    
    plt.suptitle('Mean radial wind speed on '+Data.attrs['start_time'][:10]+'\n File: '+os.path.basename(Data.attrs['input_source'])\
              +'\n Time (UTC): '+Data.attrs['start_time'][11:19]+' - '+Data.attrs['end_time'][11:19])
   
    fig.savefig(save_path.replace('.nc','_rws_avg.png'))
    plt.close()
    
def plot_wind_map(Data,
                  max_sigma={'U':10,'V':10,'WS':10,'WD':50},
                  heights=[0,4,8,12],
                  save_path='',
                  stride=3,
                  path_layout='',
                  markers={}):
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
      
    levels=np.unique(np.round(np.linspace(np.nanpercentile(Data.WS,5)-0.5, 
                                          np.nanpercentile(Data.WS,95)+0.5, 20),1))
    
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
                 '\n Synthesized from: '+Data.attrs['site1'] +' and ' + Data.attrs['site2'] \
                +'\n Time (UTC): '+Data.attrs['start_time'][11:19]+' - '+Data.attrs['end_time'][11:19])
        
    plt.tight_layout()
        
    if save_path is not None:
         fig.savefig(save_path)
    plt.close()


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
    
    
    
  