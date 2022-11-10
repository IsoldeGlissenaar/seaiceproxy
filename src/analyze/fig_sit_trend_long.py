# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:27:41 2022

@author: zq19140
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m')
import shapefile as shp  # Requires the pyshp package
import matplotlib.pyplot as plt
import calendar
from shapely.geometry import Point, shape

    
#%%

fig = plt.figure(figsize=(9,13),dpi=200)

m = 4
year_1 = 1992
year_2 = 2020

direc = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/Projects/icecharts_thickness/data/processed/predicted_sit_edit/'
sit = xr.open_dataset(direc+'predic_sit_19932020_'+f"{m:02}"+'.nc')
sit = sit.drop(1997, dim='year')
sit['myi_conc'] = (['year', 'n'], np.nansum(sit.dataset_X[:,11:14,:],axis=1))
sit['fyi_conc'] = (['year', 'n'], np.nansum(sit.dataset_X[:,5:11,:],axis=1))

y1 = np.where(sit.year==year_1)[0][0]
y2 = np.where(sit.year==year_2)[0][0]

p_value = np.zeros(sit.dims['n'])
trend_sit = np.zeros(sit.dims['n'])
for i in range(sit.dims['n']):
    x=sit.year[y1:y2+1]
    y=sit.sit_mean[y1:y2+1,i]
    trend_sit[i], intercept, r_value, p_value[i], std_err = stats.linregress(x, y)


#SIT correction
if m==4:
    corr = 3.2 #cm/yr
    corr_f = 1.8 #cm/yr
elif m==11:
    corr = 2.8 #cm/yr
    corr_f = 0
elif m==12:
    corr = 3.2
    corr_f = 1.3
elif m==1:
    corr = 2.4
    corr_f = 1.2
elif m==2:
    corr = 2.9
    corr_f = 1.5
elif m==3:
    corr = 3.4
    corr_f = 1.7
    
corr_sit = np.zeros((sit.sit_mean.values.shape))
for y in range(len(sit.year)):
    yr = sit.year.values[y]
    yr_dif = 2010-yr
    if yr_dif<0:
        yr_dif=0
    corr_sit[y,:] = sit.sit_mean[y,:] + corr/100*yr_dif*sit.myi_conc[y,:] + corr_f/100*yr_dif*sit.fyi_conc[y,:]

sit['sit_mean_corr'] = (['year', 'n'], corr_sit)

p_value_corr = np.zeros(sit.dims['n'])
trend_sit_corr = np.zeros(sit.dims['n'])
for i in range(sit.dims['n']):
    x=sit.year[y1:y2+1]
    y=sit.sit_ku[y1:y2+1,i]
    trend_sit_corr[i], intercept, r_value, p_value_corr[i], std_err = stats.linregress(x, y)


regions_f = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Howell/v200_CISIRR_Regions/v200_CISIRR_Regions_4326_merge.shp'
sf = shp.Reader(regions_f)
regions = ['cwa04_00','cea12_00','cwa01_00','tew02_00']

ax1 = plt.subplot2grid((4,2), (0,0), colspan=2, rowspan=2,projection=ccrs.Orthographic(central_longitude=-99, central_latitude=70, globe=None))
ax1.coastlines(resolution='50m',linewidth=0.5)
ax1.set_extent([-140,-57,62,84],crs=ccrs.PlateCarree())
ax1.gridlines(linewidth=0.3, color='k', alpha=0.5, linestyle=':')
im = ax1.scatter(sit.lon, sit.lat, c=trend_sit*100, cmap='RdBu',vmin=-3,vmax=3,s=18,transform=ccrs.PlateCarree())
ax1.scatter(sit.lon[p_value<0.05], sit.lat[p_value<0.05], facecolors='none',
            s=18, edgecolor='black', linewidth=0.4, transform=ccrs.PlateCarree())
ax1.scatter(sit.lon[p_value_corr<0.05], sit.lat[p_value_corr<0.05], facecolors='none',
            s=18, edgecolor='black', linewidth=0.1, transform=ccrs.PlateCarree())
for j in range(len(sf.records())):
    shape1 = sf.shapeRecords()[j]
    idx = sf.records()[j][3]
    if idx in regions:
        x = [i[0] for i in shape1.shape.points[:]]
        y = [i[1] for i in shape1.shape.points[:]]
        ax1.plot(x,y, c='black', linewidth=0.5, transform=ccrs.PlateCarree())
ax1.add_feature(land_50m, facecolor='#eeeeee')
cbar = fig.colorbar(im,fraction=0.026, pad=0.04,extend='both')
cbar.ax.locator_params(nbins=7)
cbar.ax.tick_params(labelsize=8)
cbar.set_label(label='cm/yr',fontsize=10,fontname='Arial')
ax1.text(-160,71, 'a',fontname='Arial', fontsize=14, transform=ccrs.PlateCarree())
ax1.text(-132,76, 'b',fontname='Arial', fontsize=14, transform=ccrs.PlateCarree()) 
ax1.text(-120,73, 'c',fontname='Arial', fontsize=14, transform=ccrs.PlateCarree())
ax1.text(-66,76.8, 'd',fontname='Arial', fontsize=14, transform=ccrs.PlateCarree())
# ax.text(-66,77, 'e',fontname='Arial', transform=ccrs.PlateCarree())
# plt.title(f'Sea ice thickness trend ({calendar.month_name[int(m)]} {str(year_1)}-{str(year_2)})', fontsize=10, fontname='Arial')

#%%

#Predicted SIT 1980-1992
sit_old = xr.open_dataset('C:/Users/zq19140/OneDrive - University of Bristol/Documents/Projects/icecharts_thickness/data/processed/predicted_sit_edit/predic_sit_19802020_'+f"{m:02}"+'.nc')
if m==4:
    sit_old = sit_old.sel(year=slice(1982,1991))  #Because too much removed in 1980&1981 for not being in training data (FYI)
else:
    sit_old = sit_old.sel(year=slice(1980,1991))
    
# Check in which CISIRR Region the grid cells are
idx = []
shp1 = shp.Reader('C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Howell/v200_CISIRR_Regions/v200_CISIRR_Regions_4326_merge.shp') #open the shapefile
all_shapes = shp1.shapes() # get all the polygons
all_records = shp1.records()     
len_f = sit.dims['n']
location = []; p=0
for i in range(len_f):
    pt = (sit.lon.values[i], sit.lat.values[i])  
    for k in range (len(all_shapes)):             
        boundary = all_shapes[k]
        if Point(pt).within(shape(boundary)):
            location.append(all_records[k][3])
            p=1
    if p==0:
        location.append(' ')
    p=0
sit['location'] = (['n'], np.array(location))

   

def plt_trend(sit, name, region, c, subfig, ylim, j,k):
    idx = []
    for i in range(len(sit.location)):
        if sit.location[i]==region:
            idx.append(i)
    
    x = sit.year
    y = np.nanmean(sit.sit_mean[:,idx],axis=1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trend_sit = slope
    print('SIT trend: '+str(np.round(slope*100,4))+' cm/yr')
    print('p-value: '+str(np.round(p_value,4)))
    
    x = sit.year
    y = np.nanmean(sit.sit_mean_corr[:,idx],axis=1)
    slope_corr, intercept_corr, r_value, p_value_corr, std_err = stats.linregress(x, y)
    trend_sit = slope
    print('Corrected SIT trend: '+str(np.round(slope_corr*100,4))+' cm/yr')
    print('p-value: '+str(np.round(p_value_corr,4)))

    ax = plt.subplot2grid((4,2), (j,k)) 
    
    ax.plot(sit_old.year,np.nanmean(sit_old.sit_mean[:,idx],axis=1), c=c,alpha=0.4)
    ax.plot([sit_old.year[-1],sit.year[0]],[np.nanmean(sit_old.sit_mean[-1,idx],axis=0),np.nanmean(sit.sit_mean[0,idx],axis=0)], c=c, alpha=0.4)
    ax.plot(sit.year, np.nanmean(sit.sit_mean[:,idx],axis=1), c=c, marker='o', markersize=4)
    ax.plot(sit.year, slope*sit.year+intercept, c=c, linestyle='--', linewidth=1)
    ax.plot(sit.year, np.nanmean(sit.sit_mean_corr[:,idx],axis=1), c=c, marker='o', alpha=0.5, markersize=4)
    ax.plot(sit.year, slope_corr*sit.year+intercept_corr, c=c, linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(linestyle=':')
    ax.set_ylabel('sea ice thickness [m]', fontsize=8)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', labelsize=7)
    plt.title(name, fontsize=10)
    ax.text(0.03,0.05,subfig, fontsize=12, transform=ax.transAxes)
    if p_value<0.05:
        ax.text(0.95,0.085,'pred trend: '+str(np.round(trend_sit*100,2))+' cm/yr', fontsize=8,horizontalalignment='right',transform=ax.transAxes)
    else:
        ax.text(0.95,0.085,'pred trend: '+str(np.round(trend_sit*100,2))+' cm/yr', fontsize=8,horizontalalignment='right',alpha=0.5,transform=ax.transAxes)
    if p_value_corr<0.05:
        ax.text(0.95,0.025,'corr trend:  '+str(np.round(slope_corr*100,2))+' cm/yr', fontsize=8,horizontalalignment='right',transform=ax.transAxes)
    else:
        ax.text(0.95,0.025,'corr trend:  '+str(np.round(slope_corr*100,2))+' cm/yr', fontsize=8,horizontalalignment='right',alpha=0.5,transform=ax.transAxes)
        
     
            

plt_trend(sit, 'Beaufort Sea', 'cwa01_00', '#86CA38', '(a)', [1,3], 2,0)
plt_trend(sit, 'Arctic Ocean Periphery', 'tew02_00', '#ED72B2', '(b)', [1,4], 2,1)
# plt_trend(sit, 'Western Arctic Waterways', 'cwa05_00', '#ED72B2', '(b)', [1,2], 2,1)
plt_trend(sit, 'Parry Channel', 'cwa04_00', '#55D0EF', '(c)', [1,3], 3,0)
plt_trend(sit, 'Baffin Bay', 'cea12_00', '#D0AE78', '(d)', [0.5,2], 3,1)
                         

fig.tight_layout()       

 #%%
 
fig = plt.figure(dpi=200)
c='grey'
ylim=[0,3]
name='all '+str(m)
subfig='()'

x = sit.year
y = np.nanmean(sit.sit_mean[:,:],axis=1)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
trend_sit = slope
print('SIT trend: '+str(np.round(slope*100,4))+' cm/yr')
print('p-value: '+str(np.round(p_value,4)))    
print('Sum of squares error (SSE): '+str(np.round(np.sum((slope*sit.year.values+intercept - np.nanmean(sit.sit_mean[:,:],axis=1))**2),4)))
    

x = sit.year
y = np.nanmean(sit.sit_mean_corr[:,:],axis=1)
slope_corr, intercept_corr, r_value, p_value_corr, std_err = stats.linregress(x, y)
trend_sit = slope
print('Corrected SIT trend: '+str(np.round(slope_corr*100,4))+' cm/yr')
print('p-value: '+str(np.round(p_value_corr,4)))

ax = plt.axes() 

# ax.plot(sit_old.year,np.nanmean(sit_old.sit_mean[:,:],axis=1), c=c,alpha=0.4)
# ax.plot([sit_old.year[-1],sit.year[0]],[np.nanmean(sit_old.sit_mean[-1,:],axis=0),np.nanmean(sit.sit_mean[0,:],axis=0)], c=c, alpha=0.4)
ax.plot(sit.year, np.nanmean(sit.sit_mean[:,:],axis=1), c=c, marker='o', markersize=4)
ax.plot(sit.year, slope*sit.year+intercept, c=c, linestyle='--', linewidth=1)
ax.plot(sit.year, np.nanmean(sit.sit_mean_corr[:,:],axis=1), c=c, marker='o', alpha=0.5, markersize=4)
ax.plot(sit.year, slope_corr*sit.year+intercept_corr, c=c, linestyle='--', linewidth=1, alpha=0.5)
ax.grid(linestyle=':')
ax.set_ylabel('sea ice thickness [m]', fontsize=8)
ax.set_ylim(ylim)
ax.tick_params(axis='both', labelsize=7)
plt.title(name, fontsize=10)
ax.text(0.03,0.05,subfig, fontsize=12, transform=ax.transAxes)
if p_value<0.05:
    ax.text(0.95,0.085,'pred trend: '+str(np.round(trend_sit*100,2))+' cm/yr', fontsize=8,horizontalalignment='right',transform=ax.transAxes)
else:
    ax.text(0.95,0.085,'pred trend: '+str(np.round(trend_sit*100,2))+' cm/yr', fontsize=8,horizontalalignment='right',alpha=0.5,transform=ax.transAxes)
if p_value_corr<0.05:
    ax.text(0.95,0.025,'corr trend:  '+str(np.round(slope_corr*100,2))+' cm/yr', fontsize=8,horizontalalignment='right',transform=ax.transAxes)
else:
    ax.text(0.95,0.025,'corr trend:  '+str(np.round(slope_corr*100,2))+' cm/yr', fontsize=8,horizontalalignment='right',alpha=0.5,transform=ax.transAxes)
    