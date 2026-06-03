# -*- coding: utf-8 -*-
"""
Dual-Doppler error maps for a selected site pair with scan sector overlay
"""

import os
cd = os.getcwd()
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
import utm
from pyproj import Transformer
from utils import cosd, sind, dual_doppler_error, aerial_map

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_layout = os.path.join(cd, 'data/CORSAIR_layout.xlsx')

site1 = 'Site 4.2'   # name of first lidar site (must match layout)
site2 = 'Site 1.9'   # name of second lidar site

azi1 = {site1: -95, site2: -80}  # [deg] start azimuth CW from North
azi2 = {site1:  65, site2:  80}  # [deg] end   azimuth CW from North
max_range = 2000     # [m] lidar maximum range

WS = 10                          # [m/s] representative wind speed
WD_bins = np.arange(0, 360, 10)  # [deg] wind directions for omnidirectional averaging
sigma_rws = .1  # [m/s] error on LOS velocity
sigma_w = 1     # [m/s] error on vertical velocity
zoom = 16       # aerial imagery zoom level

# analysis grid (local coordinates)
x = np.arange(-1000, 1000, 10)
y = np.arange(-250, 1001, 10)
z = np.arange(0, 201, 50)

vmaxes = {'sigma_U': 1, 'sigma_V': 1, 'sigma_WS': 1, 'sigma_WD': 5}
clabels = {
    'sigma_U':  r'$\sigma_U$ [m s$^{-1}$]',
    'sigma_V':  r'$\sigma_V$ [m s$^{-1}$]',
    'sigma_WS': r'$\sigma_\text{WS}$ [m s$^{-1}$]',
    'sigma_WD': r'$\sigma_\text{WD}$ [deg]',
}

colors_s = {site1: 'k', site2: 'k'}  # per-lidar sector colors

#%% Initialization
FC = pd.read_excel(source_layout, sheet_name='Assets').set_index('Name')
_, _, zone_num, zone_str = utm.from_latlon(float(FC['Latitude'].values[0]),
                                           float(FC['Longitude'].values[0]))
FC['x_utm'], FC['y_utm'], _, _ = utm.from_latlon(FC['Latitude'].values, FC['Longitude'].values)

# local coordinate origin = midpoint of the selected pair
x0 = float(FC.loc[[site1, site2], 'x_utm'].mean())
y0 = float(FC.loc[[site1, site2], 'y_utm'].mean())
lat0, lon0 = utm.to_latlon(x0, y0, zone_num, zone_str)

x1_loc = float(FC.loc[site1, 'x_utm']) - x0
y1_loc = float(FC.loc[site1, 'y_utm']) - y0
x2_loc = float(FC.loc[site2, 'x_utm']) - x0
y2_loc = float(FC.loc[site2, 'y_utm']) - y0

x_target_loc = FC[FC['DD target'] == 'Yes']['x_utm'] - x0
y_target_loc = FC[FC['DD target'] == 'Yes']['y_utm'] - y0

# CRS transformer: local AEQD → Web Mercator (coordinate arithmetic only, no network)
aeqd_crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m +datum=WGS84"
transformer = Transformer.from_crs(aeqd_crs, "EPSG:3857", always_xy=True)

XX, YY = np.meshgrid(x, y, indexing='ij')
XX_web, YY_web = transformer.transform(XX.ravel(), YY.ravel())
XX_web = XX_web.reshape(XX.shape)
YY_web = YY_web.reshape(YY.shape)

x_tgt_web, y_tgt_web = transformer.transform(x_target_loc.values, y_target_loc.values)

# scan sector intersection mask on the (nx, ny) grid
def in_sector(azi, a1_deg, a2_deg):
    a1, a2 = a1_deg % 360, a2_deg % 360
    if a2 > a1:
        return (azi >= a1) & (azi <= a2)
    else:  # sector crosses North (0 / 360°)
        return (azi >= a1) | (azi <= a2)

azi_from_1 = (90 - np.degrees(np.arctan2(YY - y1_loc, XX - x1_loc))) % 360
azi_from_2 = (90 - np.degrees(np.arctan2(YY - y2_loc, XX - x2_loc))) % 360
mask = (in_sector(azi_from_1, azi1[site1], azi2[site1]) &
        in_sector(azi_from_2, azi1[site2], azi2[site2]))

# sector polygon vertices in Web Mercator (computed once, reused per panel)
sector_webs = {}
for site, x_l, y_l in [(site1, x1_loc, y1_loc), (site2, x2_loc, y2_loc)]:
    a1, a2 = azi1[site] % 360, azi2[site] % 360
    if a2 > a1:
        azi_arc = np.linspace(a1, a2, 300)
    else:
        azi_arc = np.linspace(a1, a2 + 360, 300) % 360
    x_arc = x_l + max_range * cosd(90 - azi_arc)
    y_arc = y_l + max_range * sind(90 - azi_arc)
    x_poly = np.concatenate([[x_l], x_arc, [x_l]])
    y_poly = np.concatenate([[y_l], y_arc, [y_l]])
    xw, yw = transformer.transform(x_poly, y_poly)
    sector_webs[site] = (xw, yw)

os.makedirs(os.path.join(cd, 'figures', 'dual-Doppler_scan_sector'), exist_ok=True)

#%% Main
x_pts = xr.DataArray(x_target_loc.values, dims='target')
y_pts = xr.DataArray(y_target_loc.values, dims='target')

acc_map = {v: [] for v in clabels}
acc_tgt = {v: [] for v in clabels}

for WD_i in WD_bins:
    Errors = dual_doppler_error(
        x1_loc, y1_loc, x2_loc, y2_loc,
        x, y, z,
        WS=WS, WD=WD_i,
        sigma_rws=sigma_rws, sigma_w=sigma_w
    )

    if Errors is None:
        continue

    Errors_2d  = Errors.mean(dim='z')
    Errors_tgt = Errors.interp(x=x_pts, y=y_pts).mean(dim='z')

    for v in clabels:
        acc_map[v].append(Errors_2d[v].values)
        acc_tgt[v].append(Errors_tgt[v].values)

# WD- and z-averaged fields
err_map = {v: np.nanmean(acc_map[v], axis=0) for v in clabels}
err_tgt = {v: np.nanmean(acc_tgt[v], axis=0) for v in clabels}

# mask error maps to dual-scan sector intersection
for v in clabels:
    err_map[v] = np.where(mask, err_map[v], np.nan)

#%% Plots
x_ldr_loc = np.array([x1_loc, x2_loc])
y_ldr_loc = np.array([y1_loc, y2_loc])

fig, axes = plt.subplots(2, 2, figsize=(17, 10))
for ax, var in zip(axes.ravel(), clabels):

    # aerial background with lidar positions (handles SSL internally)
    aerial_map(
        x_ldr_loc, y_ldr_loc,
        lat0, lon0,
        zoom=zoom, color='b', markersize=20,
        xmin=float(x.min()), xmax=float(x.max()),
        ymin=float(y.min()), ymax=float(y.max()),
        ax=ax
    )

    # scan sector polygons (filled + boundary)
    for site in [site1, site2]:
        xw, yw = sector_webs[site]
        ax.fill(xw, yw, alpha=0.15, color=colors_s[site], zorder=2)
        ax.plot(xw, yw, color=colors_s[site], linewidth=1.5, zorder=2)

    # error contour, clipped to dual-sector intersection
    cf = ax.contourf(XX_web, YY_web, err_map[var],
                     levels=np.linspace(0, vmaxes[var], 21),
                     cmap='RdYlGn_r', extend='both', alpha=0.6, zorder=3)
    plt.colorbar(cf, ax=ax, label=clabels[var], shrink=0.85,
                 ticks=np.linspace(0, vmaxes[var],6))

    # target sites colored by local error
    ax.scatter(x_tgt_web, y_tgt_web, c=err_tgt[var], cmap='RdYlGn_r',
               vmin=0, vmax=vmaxes[var], s=80,
               edgecolors='k', linewidths=0.8, zorder=5)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.suptitle(f'{site1} – {site2}', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(cd, 'figures', 'dual-Doppler_scan_sector', f'{site1}-{site2}.png'),
            dpi=150, bbox_inches='tight')

