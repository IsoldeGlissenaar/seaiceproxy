# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:03:23 2022

@author: zq19140
"""

import numpy as np
import xarray as xr
from scipy import stats
import calendar
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m')

m = 4
year_1 = 2000
year_2 = 2020

direc = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/Projects/icecharts_thickness/data/processed/predicted_sit/'
sit = xr.open_dataset(direc+'predic_sit_19932020_'+f"{m:02}"+'.nc')

y1 = np.where(sit.year==year_1)[0][0]
y2 = np.where(sit.year==year_2)[0][0]

p_value = np.zeros(sit.dims['n'])
trend_sit = np.zeros(sit.dims['n'])
for i in range(sit.dims['n']):
    x=sit.year[y1:y2+1]
    y=sit.sit_mean[y1:y2+1,i]
    
    trend_sit[i], intercept, r_value, p_value[i], std_err = stats.linregress(x, y)

fig=plt.figure(dpi=200)
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-99, central_latitude=70, globe=None))
ax.coastlines(resolution='50m',linewidth=0.5)
ax.set_extent([-140,-57,62,84],crs=ccrs.PlateCarree())
ax.gridlines(linewidth=0.3, color='k', alpha=0.5, linestyle=':')
im = plt.scatter(sit.lon, sit.lat, c=trend_sit, cmap='RdBu',vmin=-0.05,vmax=0.05,s=6,transform=ccrs.PlateCarree())
# plt.scatter(sit.lon[p_value<0.05], sit.lat[p_value<0.05], c='black', s=0.1, transform=ccrs.PlateCarree())
plt.scatter(sit.lon[p_value<0.05], sit.lat[p_value<0.05], c=trend_sit[p_value<0.05], cmap='RdBu', vmin=-0.05, vmax=0.05,
            s=7, edgecolor='black', linewidth=0.2, transform=ccrs.PlateCarree())
ax.add_feature(land_50m, facecolor='#eeeeee')
cbar = fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04,extend='both')
cbar.ax.locator_params(nbins=4)
cbar.ax.tick_params(labelsize=7)
cbar.set_label(label='m/yr',fontsize=8,fontname='Arial')
plt.title(f'SIT trend ({calendar.month_name[int(m)]} {str(year_1)}-{str(year_2)})', fontsize=10, fontname='Arial')

