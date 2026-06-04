# -*- coding: utf-8 -*-
"""
Composite normalized wind speed map from dual-Doppler scans selected by
volume-average WS range, WD sector, and spatial uniformity criteria.
WS (and U, V) are normalized by each scan's volume average before compositing.
"""

import os
cd = os.getcwd()
import glob
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
import warnings
import utils as utl

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_dd     = os.path.join(cd, 'data/corsair/fc.ddoppler.z01.c1/*.nc')
source_layout = os.path.join(cd, 'data/CORSAIR_layout.xlsx')

ws_range   = [5, 15]    # [m/s] volume-avg WS selection range
wd_sector  = [200, 300] # [deg] volume-avg WD sector (handles wrap-around)
max_ws_std = 3.0        # [m/s] max spatial std of WS (uniformity criterion)
max_wd_std = 30.0       # [deg] max circular std of WD (uniformity criterion)
heights_plot = [0, 4, 8, 12]  # height indices for plot_wind_map
stride       = 3
max_sigma    = {'U': 100, 'V': 100, 'WS': 100, 'WD': 360}  # no QC on composite
markers      = {'Wind turbine': 'star_marker', 'Met tower': '^'}

#%% Pass 1: per-scan volume statistics

files   = sorted(glob.glob(source_dd))
n_files = len(files)

ws_avgs = np.full(n_files, np.nan)
wd_avgs = np.full(n_files, np.nan)
ws_stds = np.full(n_files, np.nan)
wd_stds = np.full(n_files, np.nan)

for i, f in enumerate(files):
    ds = xr.open_dataset(f)
    ws = ds['WS'].values.ravel()
    u  = ds['U'].values.ravel()
    v  = ds['V'].values.ravel()
    ds.close()

    ok = np.isfinite(ws) & np.isfinite(u) & np.isfinite(v) & (ws > 0)
    if ok.sum() < 10:
        continue

    ws_avgs[i] = np.mean(ws[ok])
    ws_stds[i] = np.std(ws[ok])

    # Circular mean and std via unit wind vectors [Mardia & Jupp, 2000]
    u_mn = np.mean(u[ok] / ws[ok])
    v_mn = np.mean(v[ok] / ws[ok])
    wd_avgs[i] = (270 - np.degrees(np.arctan2(v_mn, u_mn))) % 360
    R = np.sqrt(u_mn**2 + v_mn**2)
    wd_stds[i] = np.degrees(np.sqrt(-2 * np.log(np.clip(R, 1e-10, 1))))

#%% Scan selection

wd_in_sector = (
    (wd_avgs >= wd_sector[0]) & (wd_avgs <= wd_sector[1])
    if wd_sector[0] <= wd_sector[1]
    else (wd_avgs >= wd_sector[0]) | (wd_avgs <= wd_sector[1])
)

sel = (
    np.isfinite(ws_avgs) &
    (ws_avgs >= ws_range[0]) & (ws_avgs <= ws_range[1]) &
    wd_in_sector &
    (ws_stds <= max_ws_std) &
    (wd_stds <= max_wd_std)
)

sel_files   = [f for f, s in zip(files, sel) if s]
sel_ws_avgs = ws_avgs[sel]
print(f"Selected {len(sel_files)} / {n_files} scans")

if len(sel_files) == 0:
    raise RuntimeError("No scans passed the selection criteria.")

#%% Pass 2: normalize and composite

ds0 = xr.open_dataset(sel_files[0])
x, y, z = ds0.x.values, ds0.y.values, ds0.z.values
attrs0   = ds0.attrs
ds0.close()

ds_last  = xr.open_dataset(sel_files[-1])
end_time = ds_last.attrs['end_time']
ds_last.close()

n_sel = len(sel_files)
U_stack  = np.full((n_sel, len(x), len(y), len(z)), np.nan)
V_stack  = np.full((n_sel, len(x), len(y), len(z)), np.nan)
WS_stack = np.full((n_sel, len(x), len(y), len(z)), np.nan)

for i, (f, ws_s) in enumerate(zip(sel_files, sel_ws_avgs)):
    ds = xr.open_dataset(f)
    U_stack[i]  = ds['U'].values  / ws_s
    V_stack[i]  = ds['V'].values  / ws_s
    WS_stack[i] = ds['WS'].values / ws_s
    ds.close()

U_comp  = np.nanmean(U_stack,  axis=0)
V_comp  = np.nanmean(V_stack,  axis=0)
WS_comp = np.nanmean(WS_stack, axis=0)
WD_comp = (270 - np.degrees(np.arctan2(V_comp, U_comp))) % 360

sigma_U_comp  = np.nanstd(U_stack,  axis=0)
sigma_V_comp  = np.nanstd(V_stack,  axis=0)
sigma_WS_comp = np.nanstd(WS_stack, axis=0)

# Circular std of WD across scans [Mardia & Jupp, 2000]
with np.errstate(invalid='ignore', divide='ignore'):
    u_unit = np.where(WS_stack > 0, U_stack / WS_stack, np.nan)
    v_unit = np.where(WS_stack > 0, V_stack / WS_stack, np.nan)
u_mn = np.nanmean(u_unit, axis=0)
v_mn = np.nanmean(v_unit, axis=0)
R    = np.sqrt(u_mn**2 + v_mn**2)
sigma_WD_comp = np.degrees(np.sqrt(-2 * np.log(np.clip(R, 1e-10, 1))))

#%% Build composite Dataset and plot

coords = {'x': x, 'y': y, 'z': z}
dims   = ['x', 'y', 'z']

Comp = xr.Dataset(coords=coords)
Comp['U']        = xr.DataArray(U_comp,        dims=dims, coords=coords)
Comp['V']        = xr.DataArray(V_comp,        dims=dims, coords=coords)
Comp['WS']       = xr.DataArray(WS_comp,       dims=dims, coords=coords)
Comp['WD']       = xr.DataArray(WD_comp,       dims=dims, coords=coords)
Comp['sigma_U']  = xr.DataArray(sigma_U_comp,  dims=dims, coords=coords)
Comp['sigma_V']  = xr.DataArray(sigma_V_comp,  dims=dims, coords=coords)
Comp['sigma_WS'] = xr.DataArray(sigma_WS_comp, dims=dims, coords=coords)
Comp['sigma_WD'] = xr.DataArray(sigma_WD_comp, dims=dims, coords=coords)
Comp.attrs['start_time'] = attrs0['start_time']
Comp.attrs['end_time']   = end_time
Comp.attrs['site1']      = attrs0['site1']
Comp.attrs['site2']      = attrs0['site2']
Comp.attrs['origin_lat'] = attrs0['origin_lat']
Comp.attrs['origin_lon'] = attrs0['origin_lon']

os.makedirs(os.path.join(cd, 'figures', 'wind_map_stats'), exist_ok=True)
save_path = os.path.join(cd, 'figures', 'wind_map_stats', 'composite_wind_map.png')

utl.plot_wind_map(Comp,
                  max_sigma=max_sigma,
                  heights=heights_plot,
                  save_path=save_path,
                  stride=stride,
                  path_layout=source_layout,
                  markers=markers)
