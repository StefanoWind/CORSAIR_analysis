# -*- coding: utf-8 -*-
"""
Compare dual-Doppler lidar wind speed and direction to M2 tower observations.
Time series at each tower height, with M2 as 1-min background and lidar as scan-averaged points.
"""

import os
cd = os.getcwd()
import glob
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import utm
import warnings

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_lidar  = os.path.join(cd, 'data/corsair/fc.ddoppler.z01.c1/*.nc')
source_m2     = os.path.join(cd, 'data/m2.20260401.20260601.csv')
source_layout = os.path.join(cd, 'data/CORSAIR_layout.xlsx')

tower_name = 'M2'
heights_m2 = np.array([2, 5, 10, 20, 50, 80], dtype=float)  # [m] M2 measurement heights
utc_offset = 7  # M2 timestamps are MST = UTC-7

#%% Initialization

# Tower location from layout
FC = pd.read_excel(source_layout, sheet_name='Assets').set_index('Name')
lat_tower = float(FC.loc[tower_name, 'Latitude'])
lon_tower = float(FC.loc[tower_name, 'Longitude'])

# M2 tower data — parse MST timestamps and convert to UTC
M2 = pd.read_csv(source_m2)
M2['time'] = (pd.to_datetime(M2['DATE (MM/DD/YYYY)'] + ' ' + M2['MST'],
                              format='%m/%d/%Y %H:%M')
              + pd.Timedelta(hours=utc_offset))
M2 = M2.set_index('time').sort_index()

# Pre-compute Cartesian wind components for vector averaging of WD
for h in heights_m2.astype(int):
    ws = M2[f'Avg Wind Speed @ {h}m [m/s]']
    wd = M2[f'Avg Wind Direction @ {h}m [deg]']
    M2[f'U_{h}m'] = -ws * np.sin(np.radians(wd))
    M2[f'V_{h}m'] = -ws * np.cos(np.radians(wd))

# Lidar files
files = sorted(glob.glob(source_lidar))

# Tower position in lidar local coordinates (origin from first file attributes)
ds0 = xr.open_dataset(files[0])
origin_lat = float(ds0.attrs['origin_lat'])
origin_lon = float(ds0.attrs['origin_lon'])
ds0.close()

x0, y0, zone_num, zone_str = utm.from_latlon(origin_lat, origin_lon)
x_t, y_t, _, _ = utm.from_latlon(lat_tower, lon_tower)
x_tower = x_t - x0
y_tower = y_t - y0
print(f"{tower_name} local coords: x = {x_tower:.1f} m, y = {y_tower:.1f} m")

#%% Main — interpolate lidar at tower location and match M2 scan-averages

t_lidar  = []
WS_lidar = np.full((len(files), len(heights_m2)), np.nan)
WD_lidar = np.full((len(files), len(heights_m2)), np.nan)
WS_m2    = np.full((len(files), len(heights_m2)), np.nan)
WD_m2    = np.full((len(files), len(heights_m2)), np.nan)

for i, f in enumerate(files):
    ds = xr.open_dataset(f)
    t_start = pd.Timestamp(ds.attrs['start_time'])
    t_end   = pd.Timestamp(ds.attrs['end_time'])
    t_lidar.append(t_start + (t_end - t_start) / 2)

    # Interpolate U, V at tower (x, y); then interpolate in z at M2 heights
    # Two-step to avoid potential issues with mixed scalar/array interp
    U_xy = ds['U'].interp(x=float(x_tower), y=float(y_tower), method='linear')
    V_xy = ds['V'].interp(x=float(x_tower), y=float(y_tower), method='linear')
    U_pt = U_xy.interp(z=heights_m2, method='linear').values  # shape (nh,)
    V_pt = V_xy.interp(z=heights_m2, method='linear').values

    WS_lidar[i] = (U_pt**2 + V_pt**2)**0.5
    WD_lidar[i] = (270 - np.degrees(np.arctan2(V_pt, U_pt))) % 360

    # Average M2 data over the lidar scan window (vector average for WD)
    M2_win = M2.loc[t_start:t_end]
    if len(M2_win) > 0:
        for j, h in enumerate(heights_m2.astype(int)):
            WS_m2[i, j] = M2_win[f'Avg Wind Speed @ {h}m [m/s]'].mean()
            U_avg = M2_win[f'U_{h}m'].mean()
            V_avg = M2_win[f'V_{h}m'].mean()
            WD_m2[i, j] = (270 - np.degrees(np.arctan2(V_avg, U_avg))) % 360

    ds.close()

t_lidar = np.array(t_lidar)

#%% Plots — time series of WS and WD at each tower height

os.makedirs(os.path.join(cd, 'figures', 'validate_DD_M2'), exist_ok=True)

nh    = len(heights_m2)
ncols = 3
nrows = int(np.ceil(nh / ncols))

# M2 time window: ±1 h around the lidar scan period
t0 = t_lidar.min() - pd.Timedelta(hours=1)
t1 = t_lidar.max() + pd.Timedelta(hours=1)
M2_plot = M2.loc[t0:t1]

plot_cfg = [
    ('WS', WS_lidar, 'Avg Wind Speed @ {}m [m/s]', r'$U_h$ [m s$^{-1}$]', None),
    ('WD', WD_lidar, 'Avg Wind Direction @ {}m [deg]', r'$\theta$ [deg]', [0, 360]),
]

for varname, lidar_vals, m2_col_fmt, ylabel, ylim in plot_cfg:
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows),
                             sharex=True, squeeze=False)

    for j, h in enumerate(heights_m2.astype(int)):
        ax = axes.ravel()[j]
        m2_col = m2_col_fmt.format(h)

        # M2 1-min time series
        ax.plot(M2_plot.index, M2_plot[m2_col],
                color='gray', lw=0.8, label='M2 (1-min)')

        # Lidar scan-mean scatter
        ax.scatter(t_lidar, lidar_vals[:, j],
                   color='C0', s=60, zorder=5, label='Lidar DD')

        ax.set_title(f'z = {h} m')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.4)

        if ylim is not None:
            ax.set_ylim(ylim)
            ax.set_yticks([0, 90, 180, 270, 360])

        if j == 0:
            ax.legend(fontsize=11)

    # Hide unused panels
    for k in range(nh, nrows * ncols):
        axes.ravel()[k].set_visible(False)

    # x-axis labels on bottom row only
    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))

    fig.suptitle(f'Lidar DD vs {tower_name} — {varname}', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(cd, 'figures', 'validate_DD_M2', f'{varname}_timeseries.png'),
                dpi=150, bbox_inches='tight')
    # plt.close(fig)
