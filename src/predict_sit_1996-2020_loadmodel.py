# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:18:19 2022

@author: Isolde Glissenaar

# Date: 14/02/2022
# Name: predict_sit_1996-2020.py
# Description: Uses Random Forest Regression model to predict sea ice thickness
#              for 1996-2020 in the Canadian Arctic.
# Input requirements: Training dataset_X and dataset_y, 1996-2020 dataset for both C-band and Ku-band
# Output: Predicted sea ice thickness for 1996-2020
"""

import sys
sys.path.append('/functions')
sys.path.append('C:/Users/zq19140/OneDrive - University of Bristol/Documents/Projects/icecharts_thickness/src/model/functions')
import func_dataset

import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
import calendar
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m')

month = '01'
location = 'all_locations'
mintime = 1996

#%%
"""Predict sea ice thickness with C-band scatterometer data"""
#%%
#Get training dataset
scat = 'C'
dataset_X_train, dataset_y_train = func_dataset.one_month_training(location, scat, month)
dataset_X_train, dataset_y_train = func_dataset.removeCAA(dataset_X_train, dataset_y_train)

#Load model
filename = '../model/RFR_'+scat+'_'+month+'.sav'
forest_reg = pickle.load(open(filename, 'rb'))


#%%
#Open dataset 1990-2020
dataset_X, dataset_y = func_dataset.open_1992_2020_data(scat, month)

#Drop before start year
dataset_X = dataset_X.drop(np.where(dataset_y.year<mintime)[0]).reset_index(drop=True)
dataset_y = dataset_y.drop(np.where(dataset_y.year<mintime)[0]).reset_index(drop=True)
dataset_y['location'][dataset_y['location']=='WesternArctic'] = 0
dataset_y['location'][dataset_y['location']=='EasternArctic'] = 1

#Predict sea ice thickness
predictions = forest_reg.predict(dataset_X)

#%%
#Remove instances where no training
idx_notrain = np.where(np.nansum(dataset_X_train,axis=0)==0)[0]
for i in idx_notrain:
    idx_pres = np.where(dataset_X.iloc[:,i]>0.5)[0]
    predictions[idx_pres] = np.nan

#%%
#Sort predictions for each date
predictions_all = np.full((200,4,2000), np.nan)
dataset_X_arr = np.full((200,24,2000), np.nan)
date = np.zeros((200,2))
years = np.unique(dataset_y.year)
i = 0
for y in range(len(years)):
    yr = years[y]
    dataset_y_year = dataset_y.drop(np.where(dataset_y.year!=yr)[0]).reset_index(drop=True)    
    dataset_X_year = dataset_X.drop(np.where(dataset_y.year!=yr)[0]).reset_index(drop=True)
    predictions_year = predictions[np.where(dataset_y.year==yr)[0]]
    days = np.unique(dataset_y_year.day)
    for d1 in range(0,len(days)):
        d = days[d1] 
        dataset_y_d = dataset_y_year.drop(np.where(dataset_y_year.day!=d)[0]).reset_index(drop=True)
        dataset_X_d = dataset_X_year.drop(np.where(dataset_y_year.day!=d)[0]).reset_index(drop=True)
        predictions_d = predictions_year[np.where(dataset_y_year.day==d)[0]]
        
        predictions_all[i,2,:len(predictions_d)] = predictions_d
        predictions_all[i,0,:len(predictions_d)] = dataset_y_d.lon.values
        predictions_all[i,1,:len(predictions_d)] = dataset_y_d.lat.values
        predictions_all[i,3,:len(predictions_d)] = dataset_y_d.location.values

        dataset_X_arr[i,:,:len(predictions_d)] = np.transpose(np.array(dataset_X_d.iloc[:,:]))
        
        date[i,0] = dataset_y_d.year[0]
        date[i,1] = dataset_y_d.day[0]

        i = i+1
        
        
#%%
#Remove no data
predictions_all = predictions_all[:np.where(date[:,0]==0)[0][0],:,:]
dataset_X_arr = dataset_X_arr[:np.where(date[:,0]==0)[0][0],:,:]
date = date[:np.where(date[:,0]==0)[0][0],:]

#%%
#Make predictions into lat,lon grid
lon = predictions_all[:,0,:].flatten()
lat = predictions_all[:,1,:].flatten()
coords = np.transpose(np.array([lon,lat]))
unique = np.unique(coords,axis=0)
unique = unique[:np.where(np.isnan(unique))[0][0],:]

sit_grid = np.zeros((predictions_all.shape[0],4,len(unique))); sit_grid[:,:,:] = np.nan
dataset_X_grid = np.zeros((predictions_all.shape[0],24,len(unique))); dataset_X_grid[:,:,:] = np.nan
for i in range(len(unique)):
    idx = np.where((predictions_all[:,0,:]==unique[i,0])&(predictions_all[:,1,:]==unique[i,1]))
    sit_grid[idx[0],:,i] = predictions_all[idx[0],:,idx[1]]
    dataset_X_grid[idx[0],:,i] = dataset_X_arr[idx[0],:,idx[1]]
    
    
#%%
# Get monthly mean from all charts results
years_C = np.unique(date[:,0])
sit_yr_grid_C = np.full((len(years_C),4,sit_grid.shape[-1]), np.nan)
data_X_grid_C = np.full((len(years_C),24,sit_grid.shape[-1]), np.nan)
for yr in range(len(years_C)):
    idx = np.where(date[:,0]==years_C[yr])[0]
    sit_yr_grid_C[yr,:,:] = np.nanmean(sit_grid[idx,:,:], axis=0)
    data_X_grid_C[yr,:,:] = np.nanmean(dataset_X_grid[idx,:,:], axis=0)

#%%
"""Predict sea ice thickness with Ku-band scatterometer data"""
#%%
#Get training dataset
scat = 'Ku'
dataset_X_train, dataset_y_train = func_dataset.one_month_training(location, scat, month)
dataset_X_train, dataset_y_train = func_dataset.removeCAA(dataset_X_train, dataset_y_train)

#Load model
filename = '../model/RFR_'+scat+'_'+month+'.sav'
forest_reg = pickle.load(open(filename, 'rb'))

#%%
#Open dataset 1990-2019
dataset_X, dataset_y = func_dataset.open_1992_2020_data(scat, month)

#Drop before start year
dataset_X = dataset_X.drop(np.where(dataset_y.year<mintime)[0]).reset_index(drop=True)
dataset_y = dataset_y.drop(np.where(dataset_y.year<mintime)[0]).reset_index(drop=True)
dataset_y['location'][dataset_y['location']=='WesternArctic'] = 0
dataset_y['location'][dataset_y['location']=='EasternArctic'] = 1

#Predict sea ice thickness
predictions = forest_reg.predict(dataset_X)

#%%
#Remove instances where no training
idx_notrain = np.where(np.nansum(dataset_X_train,axis=0)==0)[0]
for i in idx_notrain:
    idx_pres = np.where(dataset_X.iloc[:,i]>0.5)[0]
    predictions[idx_pres] = np.nan

#%%

#Sort predictions by date
predictions_all = np.full((100,4,2000), np.nan)
dataset_X_arr = np.full((100,24,2000),np.nan)
date = np.zeros((100,2))
years = np.unique(dataset_y.year)
i = 0
for y in range(len(years)):
    yr = years[y]
    dataset_y_year = dataset_y.drop(np.where(dataset_y.year!=yr)[0]).reset_index(drop=True)
    dataset_X_year = dataset_X.drop(np.where(dataset_y.year!=yr)[0]).reset_index(drop=True)
    predictions_year = predictions[np.where(dataset_y.year==yr)[0]]
    days = np.unique(dataset_y_year.day)
    for d1 in range(0,len(days)):
        d = days[d1] 
        dataset_y_d = dataset_y_year.drop(np.where(dataset_y_year.day!=d)[0]).reset_index(drop=True)
        dataset_X_d = dataset_X_year.drop(np.where(dataset_y_year.day!=d)[0]).reset_index(drop=True)
        predictions_d = predictions_year[np.where(dataset_y_year.day==d)[0]]
        
        predictions_all[i,2,:len(predictions_d)] = predictions_d
        predictions_all[i,0,:len(predictions_d)] = dataset_y_d.lon.values
        predictions_all[i,1,:len(predictions_d)] = dataset_y_d.lat.values
        predictions_all[i,3,:len(predictions_d)] = dataset_y_d.location.values
        
        dataset_X_arr[i,:,:len(predictions_d)] = np.transpose(np.array(dataset_X_d.iloc[:,:]))
        
        date[i,0] = dataset_y_d.year[0]
        date[i,1] = dataset_y_d.day[0]

        i = i+1

        
#%%
#Remove no data
predictions_all = predictions_all[:np.where(date[:,0]==0)[0][0],:,:]
dataset_X_arr = dataset_X_arr[:np.where(date[:,0]==0)[0][0],:,:]
date = date[:np.where(date[:,0]==0)[0][0],:]

#%%
#Make predictions into lat,lon grid
lon = predictions_all[:,0,:].flatten()
lat = predictions_all[:,1,:].flatten()
coords = np.transpose(np.array([lon,lat]))
unique = np.unique(coords,axis=0)
unique = unique[:np.where(np.isnan(unique))[0][0],:]

sit_grid = np.zeros((predictions_all.shape[0],4,len(unique))); sit_grid[:,:,:] = np.nan
dataset_X_grid = np.zeros((predictions_all.shape[0],24,len(unique))); dataset_X_grid[:,:,:] = np.nan
for i in range(len(unique)):
    idx = np.where((predictions_all[:,0,:]==unique[i,0])&(predictions_all[:,1,:]==unique[i,1]))
    sit_grid[idx[0],:,i] = predictions_all[idx[0],:,idx[1]]
    dataset_X_grid[idx[0],:,i] = dataset_X_arr[idx[0],:,idx[1]]

#%%
# Get monthly mean from all charts results
years_Ku = np.unique(date[:,0])
sit_yr_grid_Ku = np.zeros((len(years_Ku),4,sit_grid.shape[-1])); sit_yr_grid_Ku[:,:,:] = np.nan
data_x_grid_Ku = np.zeros((len(years_Ku),24,sit_grid.shape[-1])); data_x_grid_Ku[:,:,:] = np.nan
for yr in range(len(years_Ku)):
    idx = np.where(date[:,0]==years_Ku[yr])[0]
    sit_yr_grid_Ku[yr,:,:] = np.nanmean(sit_grid[idx,:,:], axis=0)
    data_x_grid_Ku[yr,:] = np.nanmean(dataset_X_grid[idx,:,:], axis=0)


#%%
'''Combine Ku and C'''
#%%
lon = np.copy(sit_yr_grid_Ku[:,0,:])
lon = np.append(lon, sit_yr_grid_C[:,0,:])
lat = np.copy(sit_yr_grid_Ku[:,1,:])
lat = np.append(lat, sit_yr_grid_C[:,1,:])
coords = np.transpose(np.array([lon,lat]))
unique = np.unique(np.round(coords,decimals=5),axis=0)
unique = unique[:np.where(np.isnan(unique))[0][0],:]

years = np.append(np.copy(years_Ku), years_C)
years_un = np.unique(years)

sit_grid = np.zeros((len(years_un),4,len(unique),2)); sit_grid[:,:,:,:] = np.nan
data_x_grid = np.zeros((len(years_un),24,len(unique),2)); data_x_grid[:,:,:,:] = np.nan
for i in range(len(unique)):
    idx = np.where((np.round(sit_yr_grid_Ku[:,0,:],decimals=5)==unique[i,0])&
                   (np.round(sit_yr_grid_Ku[:,1,:],decimals=5)==unique[i,1]))
    yr_idx = np.zeros(len(idx[0])).astype(int)
    for j in range(len(yr_idx)):
        yr_idx[j] = np.where((years_un==years_Ku[idx[0][j]]))[0]
    sit_grid[yr_idx,:,i,0] = sit_yr_grid_Ku[idx[0],:,idx[1]]
    data_x_grid[yr_idx,:,i,0] = data_x_grid_Ku[idx[0],:,idx[1]]
    
    idx = np.where((np.round(sit_yr_grid_C[:,0,:],decimals=5)==unique[i,0])&
                   (np.round(sit_yr_grid_C[:,1,:],decimals=5)==unique[i,1]))
    yr_idx = np.zeros(len(idx[0])).astype(int)
    for j in range(len(yr_idx)):
        yr_idx[j] = np.where((years_un==years_C[idx[0][j]]))[0]
    sit_grid[yr_idx,:,i,1] = sit_yr_grid_C[idx[0],:,idx[1]]
    data_x_grid[yr_idx,:,i,1] = data_X_grid_C[idx[0],:,idx[1]]

sit_grid_mean = np.nanmean(sit_grid, axis=3)
data_x_grid_mean = np.nanmean(data_x_grid, axis=3)

#%%

location = np.zeros((sit_grid_mean.shape[0],sit_grid_mean.shape[2])).astype(str)
location[sit_grid_mean[:,3,:]==0] = 'WesternArctic'
location[sit_grid_mean[:,3,:]==1] = 'EasternArctic'
location[(sit_grid_mean[:,3,:]>0)&(sit_grid_mean[:,3,:]<1)] = 'Both'
location[np.isnan(sit_grid_mean[:,3,:])] = 'NaN'

#%%
# Plot predicted sea ice thickness for given year
y = 0

fig=plt.figure(dpi=200)
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-100, central_latitude=60, globe=None))
ax.coastlines(resolution='50m',linewidth=0.5)
ax.set_extent([-140,-60,55,84],crs=ccrs.PlateCarree())
ax.gridlines(linewidth=0.3, color='k', alpha=0.5, linestyle=':')
im = plt.scatter(sit_grid_mean[y,0,:], sit_grid_mean[y,1,:], c=sit_grid_mean[y,2,:], cmap='Spectral_r',vmin=0,vmax=3,s=3,transform=ccrs.PlateCarree())
ax.add_feature(land_50m, facecolor='#eeeeee')
cbar = fig.colorbar(im, ax=ax, label='m',fraction=0.046, pad=0.04)
cbar.ax.locator_params(nbins=4)
plt.title(f'Predicted thickness ({calendar.month_name[int(month)]}) '+str(int(years_un[y])))

#%%
# #Remove year with 100% old ice (December-Apr 1996/1997)
if month=='12':
    idx = np.where(years_un==1996)[0]
    sit_grid[idx,2,:,:] = np.nan
    sit_grid_mean[idx,2,:] = np.nan
elif int(month)<5:
    idx = np.where(years_un==1997)[0]
    sit_grid[idx,2,:,:] = np.nan
    sit_grid_mean[idx,2,:] = np.nan
    

#%%
#Export predicted sea ice thickness dataset

data = xr.Dataset(
    data_vars = dict(
        sit_ku          =(["year","n"], sit_grid[:,2,:,0]),
        sit_c           =(["year","n"], sit_grid[:,2,:,1]),
        sit_mean        =(["year","n"], sit_grid_mean[:,2,:]),
        location        =(["year","n"], location),
        dataset_X       =(["year","p","n"], data_x_grid_mean)
        ),
    coords=dict(
        lon=(["n"],unique[:,0]),
        lat=(["n"],unique[:,1]),
        year = years_un,
        dataset_X_head=(["p"],dataset_X.columns)
        ),
    attrs=dict(description="Predicted sea ice thickness from RandomForestRegression Model (1996-2020)"))
    
direc = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/Projects/icecharts_thickness/data/processed/predicted_sit_edit/'
data.to_netcdf(direc+'predic_sit_19962020_'+month+'.nc')

