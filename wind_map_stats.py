# -*- coding: utf-8 -*-
"""
Composite normalized wind speed map from dual-Doppler scans, stratified by
wind direction sector. For each sector, scans are selected by volume-average
WS range, WD, and spatial uniformity criteria. WS (and U, V) are normalized
by each scan's volume average before compositing.
"""

import os
cd = os.getcwd()
import glob
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
import warnings
import utm
import utils as utl

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_dd     = os.path.join(cd, 'data/corsair/fc.ddoppler.z01.c1/*.nc')
source_layout = os.path.join(cd, 'data/CORSAIR_layout.xlsx')
source_topo   = os.path.join(cd, 'data/FC_topo_v2.nc')

ws_range     = [2, 20]   # [m/s] volume-avg WS selection range
wd_width     = 30        # [deg] width of each wind direction sector
max_sigma_ws = 1         # [m/s] max error in wind speed
max_sigma_wd = 10        # [deg] max error in wind direction
max_ws       = 2         # [normalized] max normalized WS (outlier filter)
max_ws_std   = 3.0       # [m/s] max spatial std of WS (uniformity criterion)
max_wd_std   = 30.0      # [deg] max circular std of WD (uniformity criterion)
max_ci       = 0.2         # [normalized] max bootstrap CI width; wider grid cells set to NaN
heights_plot = [0, 2, 4, 8]  # height indices for plot_wind_map
stride       = 3
max_sigma    = {'U': 100, 'V': 100, 'WS': 100, 'WD': 360}  # no QC on composite
markers      = {'Wind turbine': 'star_marker', 'Met tower': '^'}

#%% Pass 1: per-scan volume statistics

files   = sorted(glob.glob(source_dd))
n_files = len(files)

ws_avgs    = np.full(n_files, np.nan)
wd_avgs    = np.full(n_files, np.nan)
ws_stds    = np.full(n_files, np.nan)
wd_stds    = np.full(n_files, np.nan)
t_starts   = []
t_ends     = []

for i, f in enumerate(files):
    ds = xr.open_dataset(f)
    t_starts.append(ds.attrs.get('start_time', ''))
    t_ends.append(ds.attrs.get('end_time', ''))
    ds = ds.where(ds.sigma_WS < max_sigma_ws).where(ds.sigma_WD < max_sigma_wd)
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

t_starts = np.array(t_starts)
t_ends   = np.array(t_ends)

#%% Pre-load all valid files (normalized by scan-average WS)

ds0 = xr.open_dataset(files[0])
x, y, z = ds0.x.values, ds0.y.values, ds0.z.values
attrs0   = ds0.attrs
ds0.close()

# Topography in local (lidar-origin) coordinates
x0_utm, y0_utm, *_ = utm.from_latlon(attrs0['origin_lat'], attrs0['origin_lon'])
TOPO       = xr.open_dataset(source_topo)
TOPO_local = TOPO.assign_coords(x=TOPO.x - x0_utm, y=TOPO.y - y0_utm)

valid_file = np.isfinite(ws_avgs)

U_stack_all  = np.full((n_files, len(x), len(y), len(z)), np.nan)
V_stack_all  = np.full((n_files, len(x), len(y), len(z)), np.nan)
WS_stack_all = np.full((n_files, len(x), len(y), len(z)), np.nan)

for i, f in enumerate(files):
    if not valid_file[i]:
        continue
    ds      = xr.open_dataset(f)
    ws_norm = ds['WS'].values / ws_avgs[i]
    bad     = ws_norm > max_ws
    U_stack_all[i]  = np.where(~bad, ds['U'].values  / ws_avgs[i], np.nan)
    V_stack_all[i]  = np.where(~bad, ds['V'].values  / ws_avgs[i], np.nan)
    WS_stack_all[i] = np.where(~bad, ws_norm,                       np.nan)
    ds.close()

#%% Loop over wind direction sectors

os.makedirs(os.path.join(cd, 'figures', 'wind_map_stats'), exist_ok=True)

coords = {'x': x, 'y': y, 'z': z}
dims   = ['x', 'y', 'z']

for wd_center in np.arange(0, 360, wd_width):
    wd_lo = (wd_center - wd_width / 2) % 360
    wd_hi = (wd_center + wd_width / 2) % 360

    wd_in_sector = (
        (wd_avgs >= wd_lo) & (wd_avgs <= wd_hi)
        if wd_lo <= wd_hi
        else (wd_avgs >= wd_lo) | (wd_avgs <= wd_hi)
    )

    sel = (
        valid_file &
        (ws_avgs >= ws_range[0]) & (ws_avgs <= ws_range[1]) &
        wd_in_sector &
        (ws_stds <= max_ws_std) &
        (wd_stds <= max_wd_std)
    )

    n_sel = sel.sum()
    print(f"WD {wd_lo:.0f}–{wd_hi:.0f}°: {n_sel} scans selected")
    if n_sel == 0:
        continue

    U_sel  = U_stack_all[sel]
    V_sel  = V_stack_all[sel]
    WS_sel = WS_stack_all[sel]

    nx_g, ny_g, nz_g = len(x), len(y), len(z)
    n_sp   = nx_g * ny_g * nz_g
    U_flat  = U_sel.reshape(n_sel, n_sp)
    V_flat  = V_sel.reshape(n_sel, n_sp)
    WS_flat = WS_sel.reshape(n_sel, n_sp)

    U_avg_flat  = np.array([utl.mean_ci(U_flat[:,  k], max_ci)[0] for k in range(n_sp)])
    V_avg_flat  = np.array([utl.mean_ci(V_flat[:,  k], max_ci)[0] for k in range(n_sp)])
    WS_avg_flat = np.array([utl.mean_ci(WS_flat[:, k], max_ci)[0] for k in range(n_sp)])

    U_comp  = U_avg_flat.reshape(nx_g, ny_g, nz_g)
    V_comp  = V_avg_flat.reshape(nx_g, ny_g, nz_g)
    WS_comp = WS_avg_flat.reshape(nx_g, ny_g, nz_g)
    WD_comp = (270 - np.degrees(np.arctan2(V_comp, U_comp))) % 360

    sigma_U_comp  = np.nanstd(U_sel,  axis=0)
    sigma_V_comp  = np.nanstd(V_sel,  axis=0)
    sigma_WS_comp = np.nanstd(WS_sel, axis=0)

    # Circular std of WD across scans [Mardia & Jupp, 2000]
    with np.errstate(invalid='ignore', divide='ignore'):
        u_unit = np.where(WS_sel > 0, U_sel / WS_sel, np.nan)
        v_unit = np.where(WS_sel > 0, V_sel / WS_sel, np.nan)
    u_mn = np.nanmean(u_unit, axis=0)
    v_mn = np.nanmean(v_unit, axis=0)
    R    = np.sqrt(u_mn**2 + v_mn**2)
    sigma_WD_comp = np.degrees(np.sqrt(-2 * np.log(np.clip(R, 1e-10, 1))))

    Comp = xr.Dataset(coords=coords)
    Comp['U']        = xr.DataArray(U_comp,        dims=dims, coords=coords)
    Comp['V']        = xr.DataArray(V_comp,        dims=dims, coords=coords)
    Comp['WS']       = xr.DataArray(WS_comp,       dims=dims, coords=coords)
    Comp['WD']       = xr.DataArray(WD_comp,       dims=dims, coords=coords)
    Comp['sigma_U']  = xr.DataArray(sigma_U_comp,  dims=dims, coords=coords)
    Comp['sigma_V']  = xr.DataArray(sigma_V_comp,  dims=dims, coords=coords)
    Comp['sigma_WS'] = xr.DataArray(sigma_WS_comp, dims=dims, coords=coords)
    Comp['sigma_WD'] = xr.DataArray(sigma_WD_comp, dims=dims, coords=coords)
    Comp.attrs['start_time'] = t_starts[sel][0]
    Comp.attrs['end_time']   = t_ends[sel][-1]
    Comp.attrs['site1']      = attrs0['site1']
    Comp.attrs['site2']      = attrs0['site2']
    Comp.attrs['origin_lat'] = attrs0['origin_lat']
    Comp.attrs['origin_lon'] = attrs0['origin_lon']

    save_path = os.path.join(cd, 'figures', 'wind_map_stats',
                             f'WD{wd_lo:03.0f}-{wd_hi:03.0f}.png')

    utl.plot_wind_map(Comp,
                      max_sigma=max_sigma,
                      heights=heights_plot,
                      save_path=save_path,
                      stride=stride,
                      path_layout=source_layout,
                      markers=markers,
                      levels=np.arange(0.4,1.21,0.05),
                      topo=TOPO_local, 
                      topo_levels=100)
