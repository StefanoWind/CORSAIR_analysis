# -*- coding: utf-8 -*-
"""
Compare dual-Doppler lidar wind speed and direction to M2 tower observations.
Scan-averaged comparison at all M2 heights and vertical profile analysis.
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
source_dd  = os.path.join(cd, 'data/corsair/fc.ddoppler.z01.c1/*.nc')
source_m2     = os.path.join(cd, 'data/m2.20260401.20260601.csv')
source_layout = os.path.join(cd, 'data/CORSAIR_layout.xlsx')
source_topo   = os.path.join(cd, 'data/FC_topo_v2.nc')

utc_offset = 7    # M2 timestamps are MST = UTC-7
avg_hours  = 6   # [h] bin width for the vertical profile plot
max_ci     = 1    # [m/s] maximum bootstrap CI width; wider bins set to NaN
heights_plot=[50,80]

#%% Functions
def filt_stat(x, func, perc_lim=[5, 95]):
    x = x.copy()
    x = x[(x >= np.nanpercentile(x, perc_lim[0])) & (x <= np.nanpercentile(x, perc_lim[1]))]
    return func(x)

def filt_BS_stat(x, func, p_value=5, M_BS=100, min_N=10, perc_lim=[5, 95]):
    x = x.copy()
    x = x[(x >= np.nanpercentile(x, perc_lim[0])) & (x <= np.nanpercentile(x, perc_lim[1]))]
    if len(x) < min_N:
        return np.nan
    x_BS = x[np.random.randint(0, len(x), size=(M_BS, len(x)))]
    return np.nanpercentile(func(x_BS, axis=1), p_value)

def mean_ci(f, max_ci, p_value=0.05, perc_lim=[5, 95]):
    f_avg = filt_stat(f, np.nanmean, perc_lim=perc_lim)
    f_low = filt_BS_stat(f, np.nanmean, perc_lim=perc_lim, p_value=p_value / 2 * 100)
    f_top = filt_BS_stat(f, np.nanmean, perc_lim=perc_lim, p_value=(1 - p_value / 2) * 100)
    if f_top - f_low > max_ci or np.isnan(f_top-f_low):
        f_avg = np.nan
    return f_avg, f_low, f_top

#%% Initialization

FC        = pd.read_excel(source_layout, sheet_name='Assets').set_index('Name')
lat_tower = float(FC.loc['M2', 'Latitude'])
lon_tower = float(FC.loc['M2', 'Longitude'])

TOPO = xr.open_dataset(source_topo)

M2 = pd.read_csv(source_m2)
M2['time'] = (pd.to_datetime(M2['DATE (MM/DD/YYYY)'] + ' ' + M2['MST'],
                              format='%m/%d/%Y %H:%M')
              + pd.Timedelta(hours=utc_offset))
M2 = M2.set_index('time').sort_index()

# All M2 measurement heights from column names
heights_m2_all = np.array(sorted([
    int(col.split('@ ')[1].split('m ')[0])
    for col in M2.columns
    if col.startswith('Avg Wind Speed @ ') and 'm [m/s]' in col
]), dtype=float)

# Cartesian components for vector-averaging WD over scan windows
for h in heights_m2_all.astype(int):
    ws = M2[f'Avg Wind Speed @ {h}m [m/s]']
    wd = M2[f'Avg Wind Direction @ {h}m [deg]']
    M2[f'U_{h}m'] = -ws * np.sin(np.radians(wd))
    M2[f'V_{h}m'] = -ws * np.cos(np.radians(wd))

files = sorted(glob.glob(source_dd))

# Lidar origin → tower Cartesian offset; also read native z grid
ds0        = xr.open_dataset(files[0])
origin_lat = float(ds0.attrs['origin_lat'])
origin_lon = float(ds0.attrs['origin_lon'])
x0, y0, *_ = utm.from_latlon(origin_lat, origin_lon)
z0          = TOPO.z.interp(x=x0, y=y0).values
x_t, y_t, *_ = utm.from_latlon(lat_tower, lon_tower)
z_t         = TOPO.z.interp(x=x_t, y=y_t).values
x_tower     = x_t - x0
y_tower     = y_t - y0
z_tower     = z_t - z0
print(f"{'M2'} local coords: x={x_tower:.1f} m, y={y_tower:.1f} m, z={z_tower:.1f} m")

#DD height
z_dd     = ds0.z.values
ds0.close()
z_agl_dd = z_dd - z_tower

#%% Main — single pass through lidar files

n_files, n_dd, n_m2 = len(files), len(z_dd), len(heights_m2_all)

t_dd          = []
t_starts, t_ends = [], []
ws_dd_arr        = np.full((n_files, n_dd),  np.nan)
wd_dd_arr        = np.full((n_files, n_dd),  np.nan)
ws_dd_interp_arr = np.full((n_files, n_m2), np.nan)
wd_dd_interp_arr = np.full((n_files, n_m2), np.nan)
ws_m2_arr        = np.full((n_files, n_m2), np.nan)
wd_m2_arr        = np.full((n_files, n_m2), np.nan)

for i, f in enumerate(files):
    ds      = xr.open_dataset(f)
    t_start = pd.Timestamp(ds.attrs['start_time'])
    t_end   = pd.Timestamp(ds.attrs['end_time'])
    t_dd.append(t_start + (t_end - t_start) / 2)
    t_starts.append(t_start)
    t_ends.append(t_end)

    U_xy = ds['U'].interp(x=float(x_tower), y=float(y_tower), method='linear')
    V_xy = ds['V'].interp(x=float(x_tower), y=float(y_tower), method='linear')
    ds.close()

    # Native lidar z levels
    ws_dd_arr[i] = np.sqrt(U_xy.values**2 + V_xy.values**2)
    wd_dd_arr[i] = (270 - np.degrees(np.arctan2(V_xy.values, U_xy.values))) % 360

    # Interpolated at all M2 heights
    U_m2 = U_xy.interp(z=heights_m2_all + z_tower, method='linear').values
    V_m2 = V_xy.interp(z=heights_m2_all + z_tower, method='linear').values
    ws_dd_interp_arr[i] = np.sqrt(U_m2**2 + V_m2**2)
    wd_dd_interp_arr[i] = (270 - np.degrees(np.arctan2(V_m2, U_m2))) % 360

    # M2 averaged over scan window (vector average for WD)
    M2_win = M2.loc[t_start:t_end]
    if len(M2_win) > 0:
        for j, h in enumerate(heights_m2_all.astype(int)):
            ws_m2_arr[i, j] = M2_win[f'Avg Wind Speed @ {h}m [m/s]'].mean()
            wd_m2_arr[i, j] = (270 - np.degrees(np.arctan2(
                M2_win[f'V_{h}m'].mean(), M2_win[f'U_{h}m'].mean()))) % 360

t_dd = np.array(t_dd)

# Build xarrays — two coordinate systems
coords_dd = {'time': t_dd, 'height': z_agl_dd}
coords_m2 = {'time': t_dd, 'height': heights_m2_all}

WS_dd        = xr.DataArray(ws_dd_arr,        dims=['time', 'height'], coords=coords_dd)
WD_dd        = xr.DataArray(wd_dd_arr,        dims=['time', 'height'], coords=coords_dd)
WS_dd_interp = xr.DataArray(ws_dd_interp_arr, dims=['time', 'height'], coords=coords_m2)
WD_dd_interp = xr.DataArray(wd_dd_interp_arr, dims=['time', 'height'], coords=coords_m2)
WS_m2        = xr.DataArray(ws_m2_arr,        dims=['time', 'height'], coords=coords_m2)
WD_m2        = xr.DataArray(wd_m2_arr,        dims=['time', 'height'], coords=coords_m2)

hours_of_day = np.array([(pd.Timestamp(t).hour - utc_offset) % 24 for t in t_dd])
both_valid   = (WS_dd_interp.notnull().any('height').values &
                WS_m2.notnull().any('height').values)

#%% Plot 1 — Scan-averaged time series at each M2 height

os.makedirs(os.path.join(cd, 'figures', 'validate_DD_M2'), exist_ok=True)

plot_cfg = [
    ('WS', WS_dd_interp, WS_m2, r'WS [m s$^{-1}$]', [0, 25]),
    ('WD', WD_dd_interp, WD_m2, r'WD [$^\circ$]',   [0, 360]),
]

for varname, dd_vals, m2_vals, ylabel, ylim in plot_cfg:
    fig, axes = plt.subplots(len(heights_plot), 1, figsize=(18, 10), sharex=True, squeeze=False)
    for j, h in enumerate(heights_plot):
        ax = axes[j, 0]
        col = (f'Avg Wind Speed @ {h}m [m/s]' if varname == 'WS'
               else f'Avg Wind Direction @ {h}m [deg]')
        if col in M2.columns:
            ax.plot(M2.index, M2[col].values, '-', color='k', lw=0.5,
                    alpha=0.3, zorder=1, label='M2 (1 min)')
        ax.scatter(t_dd, m2_vals.sel(height=h).values, color='k', s=10, zorder=2,alpha=0.5, label='M2 (30 min)')
        ax.scatter(t_dd, dd_vals.sel(height=h).values, color='b', s=10, zorder=3, label='Lidar DD (30 min)')
        ax.set_title(f'z = {h} m')
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.4)
        if j == 0:
            ax.legend(fontsize=11)
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.suptitle(f'Lidar DD vs {'M2'} — {varname}', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(cd, 'figures', 'validate_DD_M2', f'{varname}_timeseries.png'),
                dpi=150, bbox_inches='tight')

#%% Plot 2 — Error histograms at each M2 height

dWS = (WS_dd_interp - WS_m2)
dWD = (((WD_dd_interp - WD_m2) + 180) % 360 - 180)

fig_h, axes_h = plt.subplots(2, len(heights_plot), figsize=(15, 9), squeeze=False)

for j, h in enumerate(heights_plot):
    for row, (d, xlabel, color, unit, fmt) in enumerate([
        (dWS.sel(height=h).values, r'$\Delta$WS [m s$^{-1}$]', 'steelblue', ' m/s', '.2f'),
        (dWD.sel(height=h).values, r'$\Delta$WD [$^\circ$]',    'coral',     '°',    '.1f'),
    ]):
        ok   = np.isfinite(d)
        bias = np.mean(d[ok])
        rms  = np.sqrt(np.mean(d[ok]**2))
        ax   = axes_h[row, j]
        ax.hist(d[ok], bins=30, color=color, edgecolor='k', linewidth=0.5)
        ax.axvline(0, color='k', lw=1, ls='--')
        ax.set_title(f'z = {h} m')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.text(0.97, 0.95,
                f'Bias = {bias:+{fmt}}{unit}\nRMS  = {rms:{fmt}}{unit}',
                transform=ax.transAxes, ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        ax.grid(alpha=0.3)

fig_h.suptitle(f'Lidar DD vs {'M2'} — Error distributions', fontsize=16)
fig_h.tight_layout()
fig_h.savefig(os.path.join(cd, 'figures', 'validate_DD_M2', 'error_histograms.png'),
              dpi=150, bbox_inches='tight')

#%% Plot 3 — avg_hours-averaged vertical WS profiles (same axes, colored by time)

bin_hour        = hours_of_day // avg_hours
unique_bins = np.unique(bin_hour[both_valid])

# Matrices indexed by scan; NaN where both_valid is False
WS_dd_mat = WS_dd.values.copy()
WS_m2_mat    = WS_m2.values.copy()
WS_dd_mat[~both_valid] = np.nan
WS_m2_mat[~both_valid]    = np.nan

WS_dd_avg, WS_dd_low, WS_dd_top={},{},{}
WS_m2_avg, WS_m2_low, WS_m2_top={},{},{}
for b in unique_bins:
    mask = bin_hour == b
    WS_dd_avg[b] = np.array([mean_ci(WS_dd_mat[mask, k], max_ci)[0] for k in range(n_dd)])
    WS_dd_low[b] = np.array([mean_ci(WS_dd_mat[mask, k], max_ci)[1] for k in range(n_dd)])
    WS_dd_top[b] = np.array([mean_ci(WS_dd_mat[mask, k], max_ci)[2] for k in range(n_dd)])
    
    WS_m2_avg[b] = np.array([mean_ci(WS_m2_mat[mask, k], max_ci)[0] for k in range(n_m2)])
    WS_m2_low[b] = np.array([mean_ci(WS_m2_mat[mask, k], max_ci)[1] for k in range(n_m2)])
    WS_m2_top[b] = np.array([mean_ci(WS_m2_mat[mask, k], max_ci)[2] for k in range(n_m2)])

norm_p   = plt.Normalize(unique_bins[0] * avg_hours, unique_bins[-1] * avg_hours)
cmap_p   = matplotlib.cm.inferno
colors_p = [cmap_p(norm_p(b * avg_hours)) for b in unique_bins]
valid_z  = np.isfinite(z_agl_dd) & (z_agl_dd >= 0)

fig_p, ax_p = plt.subplots(figsize=(8, 10))

for b, c in zip(unique_bins, colors_p):
    label = f'{b * avg_hours:02d}:00–{(b + 1) * avg_hours:02d}:00 MST'

    ws    = WS_dd_avg[b]
    ws_lo = WS_dd_low[b]
    ws_hi = WS_dd_top[b]
    valid = valid_z & np.isfinite(ws)
    ax_p.fill_betweenx(z_agl_dd[valid], ws_lo[valid], ws_hi[valid],
                       color=c, alpha=0.2, zorder=1)
    ax_p.plot(ws[valid], z_agl_dd[valid], color=c, lw=1.5, label=label, zorder=2)

    ws_m2    = WS_m2_avg[b]
    ws_m2_lo = WS_m2_low[b]
    ws_m2_hi = WS_m2_top[b]
    ok = np.isfinite(ws_m2)
    ax_p.errorbar(ws_m2[ok], heights_m2_all[ok],
                  xerr=[ws_m2[ok] - ws_m2_lo[ok], ws_m2_hi[ok] - ws_m2[ok]],
                  fmt='o', color=c, ms=6, mec='k', mew=0.5,
                  elinewidth=1, capsize=2, zorder=2)

ax_p.set_xlabel(r'WS [m s$^{-1}$]')
ax_p.set_ylabel('Height AGL [m]')
ax_p.set_title(f'{avg_hours}-hourly WS profiles — Lidar DD (line) vs M2 (dots)')
ax_p.grid(alpha=0.3)
ax_p.legend(fontsize=10, title='Local time')

fig_p.tight_layout()
fig_p.savefig(os.path.join(cd, 'figures', 'validate_DD_M2', 'WS_profiles_avgourly.png'),
              dpi=150, bbox_inches='tight')
