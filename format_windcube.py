# -*- coding: utf-8 -*-
'''
Format Windcube .sta.7z and .rtd.7z files to NetCDF and create wind speed/direction plots.

Inputs (hard-coded or command line in this order):
    sdate [%Y-%m-%d]: start date
    edate [%Y-%m-%d]: end date
    replace [bool]: replace existing files
    path_config: path to YAML config
    mode [str]: serial or parallel

Config requires:
    path_data: root data directory
    channels_windcube: list of channel subpaths containing .sta.7z and .rtd.7z files
                       (must end in .00, e.g. 'corsair/s40.lidar.z02.00')
'''
import io
import os
cd = os.path.dirname(__file__)
import sys
import re
import lzma
import warnings
import logging
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import yaml
from datetime import datetime
from multiprocessing import Pool
from utils import _git_hash
import matplotlib
import socket
import getpass
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
if len(sys.argv) == 1:
    sdate = '1970-01-01' #start date
    edate = '2070-01-01' #end date
    delete=False #delete input files?
    replace = False #replace existing files?
    path_config = os.path.join(cd, 'configs/config_corsair.yaml') #config path
    mode = 'serial' #serial or parallel
else:
    sdate=sys.argv[1]
    edate=sys.argv[2]
    delete=sys.argv[3]=="True"
    replace=sys.argv[4]=="True"
    path_config=sys.argv[5]
    mode=sys.argv[6]#

#%% Initialization
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

logfile = os.path.join(cd, 'log', datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S') + '_windcube.log')
os.makedirs(os.path.join(cd, 'log'), exist_ok=True)
logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

#%% Constants
# Per-height column block: 12 cols (11 fields + 1 empty separator), starting at base col 7
FIELD_OFFSETS = {
    'wind_speed': 0,
    'horizontal_dispersion': 1,
    'min_wind_speed': 2,
    'max_wind_speed': 3,
    'wind_direction': 4,
    'vertical_wind_speed': 5,
    'vertical_dispersion': 6,
    'CNR': 7,
    # offset 8 is CNR min — not in output standard
    'doppler_spectral_broadening': 9,
    'data_availability': 10,
}

LONG_NAMES = {
    'wind_speed':                ('Wind Speed',                   'm/s',     'Average wind speed over 10 minute interval'),
    'wind_direction':            ('Wind Direction',               'degrees', None),
    'CNR':                       ('Carrier to Noise Ratio (CNR)', 'dB',      None),
    'min_wind_speed':            ('Min Wind Speed',               'm/s',     'Minimum wind speed measured during 10 minute interval'),
    'max_wind_speed':            ('Max Wind Speed',               'm/s',     'Maximum wind speed measured during 10 minute interval'),
    'horizontal_dispersion':     ('Horizontal Dispersion',        'm/s',     None),
    'vertical_wind_speed':       ('Vertical Wind Speed',          'm/s',     None),
    'vertical_dispersion':       ('Vertical Dispersion',          'm/s',     None),
    'doppler_spectral_broadening':('Doppler Spectral Broadening', 'm/s',     None),
    'data_availability':         ('Availability',                 '%',       None),
}

VALID_RANGES = {
    'internal_temperature':       (-40,   80),
    'air_temperature':            (-40,   60),
    'pressure':                   (800, 1100),
    'relative_humidity':          (  0,  100),
    'wind_speed':                 (  0,   60),
    'wind_direction':             (  0,  360),
    'CNR':                        (-35,   20),
    'min_wind_speed':             (  0,   60),
    'max_wind_speed':             (  0,   60),
    'horizontal_dispersion':      (  0,   10),
    'vertical_wind_speed':        (-10,   10),
    'vertical_dispersion':        (  0,   10),
    'doppler_spectral_broadening':(  0,   10),
    'data_availability':          (  0,  100),
    'latitude':                   ( -90,   90),
    'longitude':                  (-180,  180),
    'altitude':                   (  0, 4000),
    # rtd-specific
    'position':                   ( -1,  360),
    'wiper_count':                (  0, 1000),
    'radial_wind_speed':          (-60,   60),
    'radial_dispersion':          (  0,   10),
    'wind_speed_x':               (-60,   60),
    'wind_speed_y':               (-60,   60),
    'wind_speed_z':               (-10,   10),
}

QC_ATTRS = {
    'units': '1',
    'flag_masks': np.array([1, 2, 4], dtype=np.uint32),
    'flag_meanings': ['Value is equal to _FillValue or NaN',
                      'Value is less than the valid_min.',
                      'Value is greater than the valid_max.'],
    'flag_assessments': ['Bad', 'Bad', 'Bad'],
    'standard_name': 'quality_flag',
}

# Per-height column block for rtd: 8 fields + 1 empty separator, starting at base col 4
RTD_FIELD_OFFSETS = {
    'wind_speed':        3,
    'wind_direction':    4,
    'CNR':               0,
    'radial_wind_speed': 1,
    'radial_dispersion': 2,
    'wind_speed_x':      5,
    'wind_speed_y':      6,
    'wind_speed_z':      7,
}
RTD_BLOCK_STRIDE = 9
RTD_BASE_COLS = 4

RTD_LONG_NAMES = {
    'wind_speed':        ('Wind Speed',                   'm/s',     None),
    'wind_direction':    ('Wind Direction',               'degrees', None),
    'CNR':               ('Carrier-to-noise ratio',       'dB',      None),
    'radial_wind_speed': ('Radial Wind Speed',            'm/s',     None),
    'radial_dispersion': ('Radial Wind Speed Dispersion', 'm/s',     None),
    'wind_speed_x':      ('x-wind',                      'm/s',     None),
    'wind_speed_y':      ('y-wind',                      'm/s',     None),
    'wind_speed_z':      ('z-wind',                      'm/s',     None),
}

#%% Functions
def compute_qc(values, name):
    vmin, vmax = VALID_RANGES[name]
    arr = np.array(values, dtype=float)
    flags = np.zeros(arr.shape, dtype=np.int32)
    flags[np.isnan(arr)] |= 1
    flags[arr < vmin] |= 2
    flags[arr > vmax] |= 4
    return flags

def parse_gps(gps_str):
    lat_m = re.search(r'Lat:([\d.]+)deg([NS])', gps_str)
    lon_m = re.search(r'Long:([\d.]+)deg([EW])', gps_str)
    lat = float(lat_m.group(1)) * (1 if lat_m.group(2) == 'N' else -1) if lat_m else np.nan
    lon = float(lon_m.group(1)) * (-1 if lon_m.group(2) == 'W' else 1) if lon_m else np.nan
    return lat, lon

def format_sta(f, save_path, delete, replace):
    bn = os.path.basename(f)
    # Input:  s40.lidar.z01.0020260520.001000.sta.7z
    # Output: s40.lidar.z01.c0.20260520.001000.sta.nc
    m = re.match(r'^(.+\.[a-z]\d+)\.(0\d).(\d{8})\.(\d{6})\.sta\.7z$', bn)
    if m is None:
        logging.error(f'Cannot parse filename: {bn}')
        return
    prefix, date_str, time_str = m.group(1), m.group(3), m.group(4)
    out_name = f'{prefix}.c0.{date_str}.{time_str}.sta.nc'
    out_nc = os.path.join(save_path, out_name)

    if os.path.isfile(out_nc) and not replace:
        logging.info(f'{out_name} exists, skipping')
        return

    os.makedirs(save_path, exist_ok=True)

    # Decompress (files are lzma-compressed text, not actual 7z archives)
    with lzma.open(f) as fid:
        content = fid.read().decode('latin-1')
    lines = content.split('\n')

    # Parse header key=value pairs until the column-header row
    header = {}
    data_idx = None
    for i, line in enumerate(lines):
        if line.startswith('**') or not line.strip():
            continue
        if 'Timestamp' in line:
            data_idx = i
            break
        if '=' in line:
            k, v = line.split('=', 1)
            header[k.strip()] = v.strip()

    if data_idx is None:
        logging.error(f'Data section not found in {bn}')
        return

    alts = np.array([float(a) for a in header.get('Altitudes AGL (m)', '').split('\t') if a.strip()])
    n_alts = len(alts)
    install_offset = float(header.get('Installation offset AGL (m)', 0))
    distance = alts + install_offset
    lat, lon = parse_gps(header.get('GPS Location', ''))

    # Read data table: tab-separated, NaN / empty → np.nan
    data_text = '\n'.join(line for line in lines[data_idx + 1:] if line.strip())
    df = pd.read_csv(io.StringIO(data_text), sep='\t', header=None, na_values=['NaN', ''])
    if df.empty:
        logging.error(f'No data rows in {bn}')
        return

    times = pd.to_datetime(df.iloc[:, 0], format='%Y/%m/%d %H:%M').values
    n_times = len(df)

    int_temp = df.iloc[:, 1].values.astype(float)
    air_temp = df.iloc[:, 2].values.astype(float)
    pressure = df.iloc[:, 3].values.astype(float)
    humidity = df.iloc[:, 4].values.astype(float)
    # col 5: Wiper count, col 6: Vbatt — not in output standard

    # Extract 2D arrays: base col 7, stride 12 per height
    data_2d = {}
    for name, offset in FIELD_OFFSETS.items():
        arr = np.full((n_times, n_alts), np.nan)
        for h in range(n_alts):
            col = 7 + h * 12 + offset
            if col < df.shape[1]:
                arr[:, h] = pd.to_numeric(df.iloc[:, col], errors='coerce').values
        data_2d[name] = arr

    # Build Dataset
    coords = {'time': times, 'distance': distance}
    ds = xr.Dataset(coords=coords)
    ds['distance'].attrs = {'units': 'm', 'long_name': 'distance AGL'}

    for name, data, long_name, units in [
        ('internal_temperature', int_temp, 'Internal Temperature', 'degC'),
        ('air_temperature',      air_temp, 'Air Temperature',      'degC'),
        ('pressure',             pressure, 'Pressure',             'hPa'),
        ('relative_humidity',    humidity, 'Relative Humidity',    '%'),
    ]:
        ds[name] = xr.DataArray(data, dims=['time'],
                                attrs={'units': units, 'long_name': long_name,
                                       'ancillary_variables': f'qc_{name}'})
        ds[f'qc_{name}'] = xr.DataArray(compute_qc(data, name), dims=['time'],
                                         attrs={'long_name': f'Quality check results on field: {long_name}',
                                                **QC_ATTRS})

    for name, (long_name, units, comment) in LONG_NAMES.items():
        attrs = {'units': units, 'long_name': long_name, 'ancillary_variables': f'qc_{name}'}
        if comment:
            attrs['comment'] = comment
        ds[name] = xr.DataArray(data_2d[name], dims=['time', 'distance'], attrs=attrs)
        ds[f'qc_{name}'] = xr.DataArray(compute_qc(data_2d[name], name), dims=['time', 'distance'],
                                         attrs={'long_name': f'Quality check results on field: {long_name}',
                                                **QC_ATTRS})

    ds['latitude']    = xr.DataArray(float(lat),  attrs={'units': 'degN', 'long_name': 'Latitude',
                                                          'description': '', 'ancillary_variables': 'qc_latitude'})
    ds['qc_latitude'] = xr.DataArray(np.int32(compute_qc(np.array([lat]), 'latitude')[0]),
                                      attrs={'long_name': 'Quality check results on field: Latitude', **QC_ATTRS})
    ds['longitude']    = xr.DataArray(float(lon), attrs={'units': 'degE', 'long_name': 'Longitude',
                                                          'description': '', 'ancillary_variables': 'qc_longitude'})
    ds['qc_longitude'] = xr.DataArray(np.int32(compute_qc(np.array([lon]), 'longitude')[0]),
                                       attrs={'long_name': 'Quality check results on field: Longitude', **QC_ATTRS})
    ds['altitude']    = xr.DataArray(np.nan,      attrs={'units': 'm', 'long_name': 'Altitude',
                                                          'ancillary_variables': 'qc_altitude'})
    ds['qc_altitude'] = xr.DataArray(np.int32(1), attrs={'long_name': 'Quality check results on field: Altitude',
                                                          **QC_ATTRS})
    ds['time'].attrs['long_name'] = 'Time (UTC)'
    
    # Global attrs: copy header params, clean degree symbol and slash from keys [Windcube .sta convention]
    def clean_key(k):
        return k.replace('\xb0', 'deg').replace('/', ',')
    location_id = prefix.split('.')[0]

    ds.attrs = {clean_key(k): v for k, v in header.items() if k != 'HeaderSize'}
    
    #general attributes
    ds.attrs.update({
        'title':            'Wind Cube Profile AVG',
        'description':      'Wind statistics from WindCube v2.1',
        'contact':          'stefano.letizia@nlr.gov',
        'institution':      'National Laboratory of the Rockies',
        'conventions':      'MHKiT-Cloud Data Standards v. 1.0',
        'location_id':      location_id,
        'history':          (f"Generated by {getpass.getuser()} on {socket.gethostname()} on "
                             f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using "
                             f"{os.path.basename(sys.argv[0])}"),
        'code':             'https://github.com/StefanoWind/CORSAIR_analysis',
        'code_version':     _git_hash(),
    })


    ds.to_netcdf(out_nc)
    logging.info(f'Saved {out_name}')

    plot_ws_wd(ds, out_nc.replace('.nc', '.ws_wd.png'),f)
    
    if delete:
        os.remove(f)


def plot_ws_wd(ds, out_path, in_path):
    distances = ds.distance.values
    n_d = len(distances)
    cmap = cm.coolwarm
    colors = [cmap(i / max(n_d - 1, 1)) for i in range(n_d)]
    times = ds.time.values
    ws = ds['wind_speed'].values
    wd = ds['wind_direction'].values

    fig = plt.figure(figsize=(15, 8))
    gs = plt.GridSpec(2, 2, figure=fig, width_ratios=[30, 1], hspace=0.08, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[:, 1])  # spans full height of both subplots

    for i, c in enumerate(colors):
        ax1.plot(times, ws[:, i], color=c, linewidth=0.8)
        ax2.plot(times, wd[:, i], color=c, linewidth=0.8)

    ax1.set_ylabel(r'Wind speed [m s$^{-1}$]')
    ax1.set_ylim(0, 30)
    ax1.grid(True, alpha=0.3)
    ax1.set_title((f'Wind speed and direction at {ds.attrs["location_id"]} on {str(times[0])[:10]} \n File: {os.path.basename(in_path)}'))
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.set_ylabel(r'Wind direction [$^\circ$]')
    ax2.set_ylim(0, 360)
    ax2.set_yticks([0, 90, 180, 270, 360])
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.set_xlabel('Time (UTC)')

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(distances[0], distances[-1]))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=r'$z$ [m a.g.l.]')

    plt.savefig(out_path, dpi=100)
    plt.close()
    logging.info(f'Saved plot {os.path.basename(out_path)}')


def format_rtd(f, save_path, delete, replace):
    bn = os.path.basename(f)
    # Input:  s40.lidar.z02.00.20260531.000000.rtd.7z
    # Output: s40.lidar.z02.b0.20260531.000000.rtd.nc
    m = re.match(r'^(.+\.[a-z]\d+)\.(0\d)\.(\d{8})\.(\d{6})\.rtd\.7z$', bn)
    if m is None:
        logging.error(f'Cannot parse filename: {bn}')
        return
    prefix, date_str, time_str = m.group(1), m.group(3), m.group(4)
    out_name = f'{prefix}.b0.{date_str}.{time_str}.rtd.nc'
    out_nc = os.path.join(save_path, out_name)

    if os.path.isfile(out_nc) and not replace:
        logging.info(f'{out_name} exists, skipping')
        return

    os.makedirs(save_path, exist_ok=True)

    with lzma.open(f) as fid:
        content = fid.read().decode('latin-1')
    lines = content.split('\n')

    header = {}
    data_idx = None
    for i, line in enumerate(lines):
        if line.startswith('**') or not line.strip():
            continue
        if 'Timestamp' in line:
            data_idx = i
            break
        if '=' in line:
            k, v = line.split('=', 1)
            header[k.strip()] = v.strip()

    if data_idx is None:
        logging.error(f'Data section not found in {bn}')
        return

    alts = np.array([float(a) for a in header.get('Altitudes AGL (m)', '').split('\t') if a.strip()])
    n_alts = len(alts)
    install_offset = float(header.get('Installation offset AGL (m)', 0))
    distance = alts + install_offset
    lat, lon = parse_gps(header.get('GPS Location', ''))

    data_text = '\n'.join(line for line in lines[data_idx + 1:] if line.strip())
    df = pd.read_csv(io.StringIO(data_text), sep='\t', header=None, na_values=['NaN', ''])
    if df.empty:
        logging.error(f'No data rows in {bn}')
        return

    times = pd.to_datetime(df.iloc[:, 0], format='%Y/%m/%d %H:%M:%S.%f').values

    pos_raw = df.iloc[:, 1].astype(str).str.strip()
    position = np.where(pos_raw == 'V', -1.0,
                        pd.to_numeric(pos_raw, errors='coerce')).astype(float)
    air_temp    = df.iloc[:, 2].values.astype(float)
    wiper_count = df.iloc[:, 3].values.astype(float)

    data_2d = {}
    for name, offset in RTD_FIELD_OFFSETS.items():
        arr = np.full((len(df), n_alts), np.nan)
        for h in range(n_alts):
            col = RTD_BASE_COLS + h * RTD_BLOCK_STRIDE + offset
            if col < df.shape[1]:
                arr[:, h] = pd.to_numeric(df.iloc[:, col], errors='coerce').values
        data_2d[name] = arr

    coords = {'time': times, 'distance': distance}
    ds = xr.Dataset(coords=coords)
    ds['distance'].attrs = {'units': 'm', 'long_name': 'distance ASL'}

    ds['position'] = xr.DataArray(position, dims=['time'],
                                   attrs={'units': '1',
                                          'description': 'Azimuthal position of scan. V is replaced with -1 for a vertical scan.',
                                          'ancillary_variables': 'qc_position'})
    ds['qc_position'] = xr.DataArray(compute_qc(position, 'position'), dims=['time'],
                                      attrs={'long_name': 'Quality check results for position', **QC_ATTRS})

    for name, data, long_name, units in [
        ('air_temperature', air_temp,    'Ext Temp',    'C'),
        ('wiper_count',     wiper_count, 'Wiper Count', 'count'),
    ]:
        ds[name] = xr.DataArray(data, dims=['time'],
                                attrs={'units': units, 'long_name': long_name,
                                       'ancillary_variables': f'qc_{name}'})
        ds[f'qc_{name}'] = xr.DataArray(compute_qc(data, name), dims=['time'],
                                         attrs={'long_name': f'Quality check results on field: {long_name}',
                                                **QC_ATTRS})

    for name, (long_name, units, comment) in RTD_LONG_NAMES.items():
        attrs = {'units': units, 'long_name': long_name, 'ancillary_variables': f'qc_{name}'}
        if comment:
            attrs['comment'] = comment
        ds[name] = xr.DataArray(data_2d[name], dims=['time', 'distance'], attrs=attrs)
        ds[f'qc_{name}'] = xr.DataArray(compute_qc(data_2d[name], name), dims=['time', 'distance'],
                                         attrs={'long_name': f'Quality check results on field: {long_name}',
                                                **QC_ATTRS})

    ds['latitude']     = xr.DataArray(float(lat),  attrs={'units': 'degN', 'long_name': 'Latitude',
                                                            'standard_name': 'latitude',
                                                            'comment': 'Recorded lattitude at the instrument location',
                                                            'ancillary_variables': 'qc_latitude'})
    ds['qc_latitude']  = xr.DataArray(np.int32(compute_qc(np.array([lat]), 'latitude')[0]),
                                       attrs={'long_name': 'Quality check results on field: Latitude', **QC_ATTRS})
    ds['longitude']    = xr.DataArray(float(lon),  attrs={'units': 'degE', 'long_name': 'Longitude',
                                                            'comment': 'Recorded longitude at the instrument location',
                                                            'ancillary_variables': 'qc_longitude'})
    ds['qc_longitude'] = xr.DataArray(np.int32(compute_qc(np.array([lon]), 'longitude')[0]),
                                       attrs={'long_name': 'Quality check results on field: Longitude', **QC_ATTRS})
    ds['altitude']     = xr.DataArray(np.nan,       attrs={'units': 'm', 'long_name': 'Altitude',
                                                            'comment': 'Recorded altitude at the instrument location',
                                                            'ancillary_variables': 'qc_altitude'})
    ds['qc_altitude']  = xr.DataArray(np.int32(1),  attrs={'long_name': 'Quality check results on field: Altitude',
                                                             **QC_ATTRS})
    ds['time'].attrs['long_name'] = 'Time (UTC)'
    
    def clean_key(k):
        return k.replace('\xb0', 'deg').replace('/', ',')
    location_id = prefix.split('.')[0]
    ds.attrs = {clean_key(k): v for k, v in header.items() if k != 'HeaderSize'}
    
    #general attributes
    ds.attrs.update({
        'title':        'Wind Cube Profile RTD',
        'description':  'High-frequency winds from WindCube v2.1',
        'contact':      'stefano.letizia@nlr.gov',
        'institution':  'National Laboratory of the Rockies',
        'conventions':  'MHKiT-Cloud Data Standards v. 1.0',
        'location_id':  location_id,
        'history':      (f"Generated by {getpass.getuser()} on {socket.gethostname()} on "
                         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using "
                         f"{os.path.basename(sys.argv[0])}"),
        'code':             'https://github.com/StefanoWind/CORSAIR_analysis',
        'code_version': _git_hash(),
    })

    ds.to_netcdf(out_nc)
    logging.info(f'Saved {out_name}')

    plot_ws_wd(ds, out_nc.replace('.nc', '.ws_wd.png'),f)
    
    if delete:
        os.remove(f)


#%% Main
def _filter_by_date(files, sdate, edate):
    return [f for f in files
            if (dm := re.search(r'(\d{8})', os.path.basename(f))) is not None
            and datetime.strptime(sdate, '%Y-%m-%d')
            <= datetime.strptime(dm.group(1), '%Y%m%d')
            <= datetime.strptime(edate, '%Y-%m-%d')]

for channel in config.get('channels_windcube', []):
    in_path = os.path.join(config['path_data'], channel)

    sta_files = _filter_by_date(sorted(glob.glob(os.path.join(in_path, '*.sta.7z'))), sdate, edate)
    rtd_files = _filter_by_date(sorted(glob.glob(os.path.join(in_path, '*.rtd.7z'))), sdate, edate)

    sta_save = os.path.join(config['path_data'], re.sub(r'\.00$', '.c0', channel))
    rtd_save = os.path.join(config['path_data'], re.sub(r'\.00$', '.b0', channel))

    if mode == 'serial':
        for f in sta_files:
            format_sta(f, sta_save, delete, replace)
            if delete:
                os.remove(f)
        for f in rtd_files:
            format_rtd(f, rtd_save, delete, replace)
            if delete:
                os.remove(f)
    elif mode == 'parallel':
        with Pool() as pool:
            pool.starmap(format_sta, [(f, sta_save, delete, replace) for f in sta_files])
            pool.starmap(format_rtd, [(f, rtd_save, delete, replace) for f in rtd_files])
    else:
        raise ValueError(f'{mode} is not a valid processing mode (must be serial or parallel)')
