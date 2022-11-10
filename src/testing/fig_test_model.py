# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:49:55 2022

@author: zq19140
"""

import sys
sys.path.append('../functions')
import func_dataset

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m')

#%%
location = 'all_locations'
all_months = 'off'
scat = 'Ku'

#%%
'''November'''
month = '11'
yr_compare = '2017'

#%%
# Get training dataset

if all_months=='on':
    trainmonth='all'
    dataset_X_train, dataset_y_train = func_dataset.all_months_training(location, scat)
elif all_months=='off':
    trainmonth=month    
    dataset_X_train, dataset_y_train = func_dataset.one_month_training(location, scat, month)
    
dataset_X_train, dataset_y_train = func_dataset.removeCAA(dataset_X_train, dataset_y_train)

#%%
# Fit model and print training and testing error

forest_reg = RandomForestRegressor(n_estimators=95, max_depth=15, max_features=5)
forest_reg.fit(dataset_X_train, dataset_y_train.sit)

predictions = forest_reg.predict(dataset_X_train)

forest_mse = mean_squared_error(dataset_y_train.sit, predictions)
forest_rmse = np.sqrt(forest_mse)
print(f"Training error - RMSE: {forest_rmse:.3f} m")

dataset_X_shuf = dataset_X_train.sample(frac=1,random_state=42).reset_index(drop=True)
dataset_y_shuf = dataset_y_train.sample(frac=1,random_state=42).reset_index(drop=True)
lin_scores = cross_val_score(forest_reg, dataset_X_shuf, dataset_y_shuf.sit,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(f"Testing error - K-fold cross-validation mean: {np.round(np.nanmean(lin_rmse_scores),3)} m")

#%%
# Drop year for testing

if all_months=='on':
    dataset_X_train = dataset_X_train.drop(np.where((dataset_y_train.year==int(yr_compare))&(dataset_X_train.month==int(month)))[0])                        
    dataset_y_train = dataset_y_train.drop(np.where((dataset_y_train.year==int(yr_compare))&(dataset_X_train.month==int(month)))[0])
elif all_months=='off':
    dataset_X_train = dataset_X_train.drop(np.where((dataset_y_train.year==int(yr_compare)))[0])                        
    dataset_y_train = dataset_y_train.drop(np.where((dataset_y_train.year==int(yr_compare)))[0])
    
forest_reg = RandomForestRegressor(n_estimators=95, max_depth=15, max_features=5)
forest_reg.fit(dataset_X_train, dataset_y_train.sit)
    
#%%
# Test on year left out from training

if all_months=='on':
    trainmonth='all'
    dataset_X, dataset_y = func_dataset.all_months_training(location, scat)
elif all_months=='off':
    trainmonth=month    
    dataset_X, dataset_y = func_dataset.one_month_training(location, scat, month)

dataset_X_test, dataset_y_test = func_dataset.get_testing_data(dataset_X, dataset_y, yr_compare, month, all_months=all_months)
predict_thickness = forest_reg.predict(dataset_X_test)

#Get mean of multiple charts
p = np.array(dataset_y_test)[:,:2]
coords = np.unique(p,axis=0)
sit = np.zeros(len(coords))
observed = np.zeros(len(coords))

for i in range(len(coords)):
    idx = np.where((dataset_y_test.lat == coords[i,0])&(dataset_y_test.lon == coords[i,1]))
    sit[i] = np.nanmean(predict_thickness[idx])
    observed[i] = np.nanmean(dataset_y_test.sit.values[idx])

#%%
'''April'''
month = '04'
yr_compare = '2018'

#%%
# Get training dataset

if all_months=='on':
    trainmonth='all'
    dataset_X_train, dataset_y_train = func_dataset.all_months_training(location, scat)
elif all_months=='off':
    trainmonth=month    
    dataset_X_train, dataset_y_train = func_dataset.one_month_training(location, scat, month)
    
dataset_X_train, dataset_y_train = func_dataset.removeCAA(dataset_X_train, dataset_y_train)


#%%
# Fit model and print training and testing error

forest_reg = RandomForestRegressor(n_estimators=95, max_depth=15, max_features=5)
forest_reg.fit(dataset_X_train, dataset_y_train.sit)

predictions = forest_reg.predict(dataset_X_train)

forest_mse = mean_squared_error(dataset_y_train.sit, predictions)
forest_rmse = np.sqrt(forest_mse)
print(f"Training error - RMSE: {forest_rmse:.3f} m")

lin_scores = cross_val_score(forest_reg, dataset_X_train, dataset_y_train.sit,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(f"Testing error - K-fold cross-validation mean: {np.round(np.nanmean(lin_rmse_scores),3)} m")

#%%
# Drop year for testing

if all_months=='on':
    dataset_X_train = dataset_X_train.drop(np.where((dataset_y_train.year==int(yr_compare))&(dataset_X_train.month==int(month)))[0])                        
    dataset_y_train = dataset_y_train.drop(np.where((dataset_y_train.year==int(yr_compare))&(dataset_X_train.month==int(month)))[0])
elif all_months=='off':
    dataset_X_train = dataset_X_train.drop(np.where((dataset_y_train.year==int(yr_compare)))[0])                        
    dataset_y_train = dataset_y_train.drop(np.where((dataset_y_train.year==int(yr_compare)))[0])
    
forest_reg = RandomForestRegressor(n_estimators=95, max_depth=15, max_features=5)
forest_reg.fit(dataset_X_train, dataset_y_train.sit)

#%%
# Test on year left out from training
if all_months=='on':
    trainmonth='all'
    dataset_X, dataset_y = func_dataset.all_months_training(location, scat)
elif all_months=='off':
    trainmonth=month    
    dataset_X, dataset_y = func_dataset.one_month_training(location, scat, month)

dataset_X_test, dataset_y_test = func_dataset.get_testing_data(dataset_X, dataset_y, yr_compare, month, all_months=all_months)
predict_thickness = forest_reg.predict(dataset_X_test)

#Get mean of multiple charts
p = np.array(dataset_y_test)[:,:2]
coords1 = np.unique(p,axis=0)
sit1 = np.zeros(len(coords1))
observed1 = np.zeros(len(coords1))

for i in range(len(coords1)):
    idx = np.where((dataset_y_test.lat == coords1[i,0])&(dataset_y_test.lon == coords1[i,1]))
    sit1[i] = np.nanmean(predict_thickness[idx])
    observed1[i] = np.nanmean(dataset_y_test.sit.values[idx])

#%%
#Plot predicted and observed SIT

def setup_plot(number, letter):
    ax = plt.subplot(gs1[number], projection=ccrs.Orthographic(central_longitude=-99, central_latitude=70, globe=None))
    ax.coastlines(resolution='50m',linewidth=0.5)
    ax.set_extent([-140,-57,62,84],crs=ccrs.PlateCarree()) 
    ax.gridlines(linewidth=0.3, color='k', alpha=0.5, linestyle=':')
    ax.add_feature(land_50m, facecolor='#eeeeee')
    ax.text(-133,59,letter, transform=ccrs.PlateCarree())  
    return ax

fig = plt.figure(figsize=(10.8,10),dpi=200)
gs1 = gridspec.GridSpec(3,2)
gs1.update(wspace=0.02, hspace=0.05)

#November
setup_plot(0, '(a)')
im = plt.scatter(coords[:,1], coords[:,0], c=sit, cmap='Spectral_r',vmin=0,vmax=3,s=6,transform=ccrs.PlateCarree())

setup_plot(2, '(c)')
im = plt.scatter(coords[:,1], coords[:,0], c=observed, cmap='Spectral_r',vmin=0,vmax=3,s=6,transform=ccrs.PlateCarree())

setup_plot(4, '(e)')
im = plt.scatter(coords[:,1], coords[:,0], c=sit-observed,cmap='RdBu',vmin=-1.5,vmax=1.5,s=6,transform=ccrs.PlateCarree())

#April
ax = setup_plot(1, '(b)')
im = plt.scatter(coords1[:,1], coords1[:,0], c=sit1,cmap='Spectral_r',vmin=0,vmax=3,s=6,transform=ccrs.PlateCarree())
cbar = fig.colorbar(im, ax=ax, label='predicted sea ice thickness [m]',extend='max', fraction=0.046, pad=0.03)
cbar.ax.locator_params(nbins=4)

ax = setup_plot(3, '(d)')
im = plt.scatter(coords1[:,1], coords1[:,0], c=observed1, cmap='Spectral_r',vmin=0,vmax=3,s=6,transform=ccrs.PlateCarree())
cbar = fig.colorbar(im, ax=ax, label='observed sea ice thickness [m]',extend='max', fraction=0.046, pad=0.03)
cbar.ax.locator_params(nbins=4)

ax = setup_plot(5, '(f)')
im = plt.scatter(coords1[:,1], coords1[:,0], c=sit1-observed1,cmap='RdBu',vmin=-1.5,vmax=1.5,s=6,transform=ccrs.PlateCarree())
ax.add_feature(land_50m, facecolor='#eeeeee')
cbar = fig.colorbar(im, ax=ax, label='predicted - observed [m]',extend='both', fraction=0.046, pad=0.03)



