# -*- coding: utf-8 -*-
"""
Identify suitable pair of sites for dual-Doppler
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
from utils import matrix_plt, dual_doppler_error, aerial_map

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_layout = os.path.join(cd, 'data/CORSAIR_layout.xlsx')

WS = 10                          # [m/s] representative wind speed
WD_bins = np.arange(0, 360, 10)  # [deg] wind directions for omnidirectional averaging
sigma_rws = .1  # [m/s] error on LOS velocity
sigma_w = 1     # [m/s] error on vertical velocity
zoom = 16       # aerial imagery zoom level

# analysis grid (local coordinates relative to source-site centroid)
x = np.arange(-1000, 1000, 10)
y = np.arange(-250, 1001, 10)
z = np.arange(0, 201, 50)

# colormap limits [units match sigma output]
vmaxes = {'sigma_U': 1, 'sigma_V': 1, 'sigma_WS': 1, 'sigma_WD': 5}
clabels = {
    'sigma_U':  r'$\sigma_U$ [m s$^{-1}$]',
    'sigma_V':  r'$\sigma_V$ [m s$^{-1}$]',
    'sigma_WS': r'$\sigma_\text{WS}$ [m s$^{-1}$]',
    'sigma_WD': r'$\sigma_\text{WD}$ [deg]',
}


#%% Initialization
FC = pd.read_excel(source_layout, sheet_name='Assets').set_index('Name')
_, _, zone_num, zone_str = utm.from_latlon(float(FC['Latitude'].values[0]),
                                           float(FC['Longitude'].values[0]))
FC['x_utm'], FC['y_utm'], _, _ = utm.from_latlon(FC['Latitude'].values, FC['Longitude'].values)

x_source = FC[FC['DD source'] == 'Yes']['x_utm']
y_source = FC[FC['DD source'] == 'Yes']['y_utm']
x_target = FC[FC['DD target'] == 'Yes']['x_utm']
y_target = FC[FC['DD target'] == 'Yes']['y_utm']

# local coordinate origin at centroid of source sites
x0 = float(x_source.mean())
y0 = float(y_source.mean())
lat0, lon0 = utm.to_latlon(x0, y0, zone_num, zone_str)

x_source_loc = x_source - x0
y_source_loc = y_source - y0
x_target_loc = x_target - x0
y_target_loc = y_target - y0

n_sites = len(x_source)
err_u  = np.full((n_sites, n_sites), np.nan)
err_v  = np.full((n_sites, n_sites), np.nan)
err_ws = np.full((n_sites, n_sites), np.nan)
err_wd = np.full((n_sites, n_sites), np.nan)

os.makedirs(os.path.join(cd, 'figures', 'dual-Doppler'), exist_ok=True)
os.makedirs(os.path.join(cd, 'figures', 'dual-Doppler_error'), exist_ok=True)

# CRS transformer: local AEQD (metres from lat0/lon0) → Web Mercator
# used only for coordinate arithmetic, no network access
aeqd_crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m +datum=WGS84"
transformer = Transformer.from_crs(aeqd_crs, "EPSG:3857", always_xy=True)

# error grid in Web Mercator (shape nx × ny, 'ij' indexing)
XX, YY = np.meshgrid(x, y, indexing='ij')
XX_web, YY_web = transformer.transform(XX.ravel(), YY.ravel())
XX_web = XX_web.reshape(XX.shape)
YY_web = YY_web.reshape(YY.shape)

# target positions in Web Mercator
x_tgt_web, y_tgt_web = transformer.transform(x_target_loc.values, y_target_loc.values)

#%% Main
x_pts = xr.DataArray(x_target_loc.values, dims='target')
y_pts = xr.DataArray(y_target_loc.values, dims='target')

for i_s1 in range(n_sites):
    for i_s2 in range(i_s1 + 1, n_sites):

        acc_map = {v: [] for v in clabels}
        acc_tgt = {v: [] for v in clabels}

        for WD_i in WD_bins:
            Errors = dual_doppler_error(
                x_source_loc.iloc[i_s1], y_source_loc.iloc[i_s1],
                x_source_loc.iloc[i_s2], y_source_loc.iloc[i_s2],
                x, y, z,
                WS=WS, WD=WD_i,
                sigma_rws=sigma_rws, sigma_w=sigma_w
            )

            if Errors is None:
                continue

            Errors_2d = Errors.mean(dim='z')
            Errors_tgt = Errors.interp(x=x_pts, y=y_pts).mean(dim='z')

            for v in clabels:
                acc_map[v].append(Errors_2d[v].values)
                acc_tgt[v].append(Errors_tgt[v].values)

        if not acc_map['sigma_U']:
            continue

        # WD- and z-averaged error maps and target values
        err_map = {v: np.nanmean(acc_map[v], axis=0) for v in clabels}
        err_tgt = {v: np.nanmean(acc_tgt[v], axis=0) for v in clabels}

        err_u[i_s1, i_s2]  = np.nanmean(err_tgt['sigma_U'])
        err_v[i_s1, i_s2]  = np.nanmean(err_tgt['sigma_V'])
        err_ws[i_s1, i_s2] = np.nanmean(err_tgt['sigma_WS'])
        err_wd[i_s1, i_s2] = np.nanmean(err_tgt['sigma_WD'])

        site1 = x_source.index[i_s1]
        site2 = x_source.index[i_s2]

        fig, axes = plt.subplots(2, 2, figsize=(17, 10))
        for ax, var in zip(axes.ravel(), clabels):
            # aerial background on the subplot axes via utils.aerial_map
            aerial_map(
                np.array([x_source_loc.iloc[i_s1], x_source_loc.iloc[i_s2]]),
                np.array([y_source_loc.iloc[i_s1], y_source_loc.iloc[i_s2]]),
                lat0, lon0,
                zoom=zoom, color='b', markersize=20,
                xmin=float(x.min()), xmax=float(x.max()),
                ymin=float(y.min()), ymax=float(y.max()),
                ax=ax
            )

            # height- and WD-averaged error contour (semi-transparent overlay)
            cf = ax.contourf(XX_web, YY_web, err_map[var],
                             levels=np.linspace(0, vmaxes[var], 21),
                             cmap='RdYlGn_r', extend='both', alpha=0.5, zorder=1)
            plt.colorbar(cf, ax=ax, label=clabels[var], shrink=0.85,
                         ticks=np.linspace(0, vmaxes[var],6))

            # target sites colored by their local error value
            ax.scatter(x_tgt_web, y_tgt_web, c=err_tgt[var], cmap='RdYlGn_r',
                       vmin=0, vmax=vmaxes[var], s=80,
                       edgecolors='k', linewidths=0.8, zorder=4)

            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.suptitle(f'{site1} – {site2}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(cd, 'figures', 'dual-Doppler_error',
                                 f'{site1}-{site2}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

#%% Output
err = (err_u  / np.nanmedian(err_u)  +
       err_v  / np.nanmedian(err_v)  +
       err_ws / np.nanmedian(err_ws) +
       err_wd / np.nanmedian(err_wd))

sites = x_source.index
SITES1, SITES2 = np.meshgrid(sites, sites)
Output = pd.DataFrame({'Site1': SITES1.ravel(), 'Site2': SITES2.ravel(), 'Total error': err.ravel()})
Output.to_excel(os.path.join('data', 'dual-Doppler_opt.xlsx'))

#%% Plots
fig = plt.figure(figsize=(18, 10))

ax = plt.subplot(2, 2, 1)
matrix_plt(sites, sites, err_u, cmap='RdYlGn_r', vmin=0, vmax=vmaxes['sigma_U'])
plt.ylabel('Lidar 2 site')
ax.set_xticklabels([])
plt.title(r'Mean error factor of $U$ [m s$^{-1}$]')

ax = plt.subplot(2, 2, 2)
matrix_plt(sites, sites, err_v, cmap='RdYlGn_r', vmin=0, vmax=vmaxes['sigma_V'])
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.title(r'Mean error factor of $V$ [m s$^{-1}$]')

ax = plt.subplot(2, 2, 3)
matrix_plt(sites, sites, err_ws, cmap='RdYlGn_r', vmin=0, vmax=vmaxes['sigma_WS'])
plt.xlabel('Lidar 1 site')
plt.ylabel('Lidar 2 site')
plt.xticks(rotation=30)
plt.title(r'Mean error factor of $U_h$ [m s$^{-1}$]')

ax = plt.subplot(2, 2, 4)
matrix_plt(sites, sites, err_wd, cmap='RdYlGn_r', vmin=0, vmax=vmaxes['sigma_WD'])
plt.xlabel('Lidar 1 site')
ax.set_yticklabels([])
plt.xticks(rotation=30)
plt.title(r'Mean error factor of $\theta_h$ [$^\circ$]')

plt.tight_layout()
