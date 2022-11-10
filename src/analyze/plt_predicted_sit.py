# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:50:03 2022

@author: Isolde Glissenaar
"""

import numpy as np
import xarray as xr
import calendar
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m')

m = 4
year = 2020

direc = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/Projects/icecharts_thickness/data/processed/predicted_sit_edit/'
sit = xr.open_dataset(direc+'predic_sit_19932020_'+f"{m:02}"+'.nc')

# sit.sit_mean.values[(sit.location.values!='WesternArctic')&(sit.location.values!='Both')] = np.nan
y = np.where(sit.year==year)[0][0]

def plot_CAA(plot):
    fig=plt.figure(dpi=200)
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-99, central_latitude=70, globe=None))
    ax.coastlines(resolution='50m',linewidth=0.5)
    ax.set_extent([-140,-57,62,84],crs=ccrs.PlateCarree())
    ax.gridlines(linewidth=0.3, color='k', alpha=0.5, linestyle=':')
    im = plt.scatter(sit.lon, sit.lat, c=plot, cmap='Spectral_r',vmin=0,vmax=3,s=6, transform=ccrs.PlateCarree())
    ax.add_feature(land_50m, facecolor='#eeeeee')
    cbar = fig.colorbar(im, ax=ax, label='m',fraction=0.046, pad=0.04)
    cbar.ax.locator_params(nbins=4)

# plot_CAA(sit.sit_c[y,:])
# plt.title(f'Predicted sea ice thickness ({calendar.month_name[m]} {str(year)}, C)')

# plot_CAA(sit.sit_ku[y,:])
# plt.title(f'Predicted sea ice thickness ({calendar.month_name[m]} {str(year)}, Ku)')

plot_CAA(sit.sit_mean[y,:].values)
plt.title(f'Predicted sea ice thickness ({calendar.month_name[m]} {str(year)}, mean)')



