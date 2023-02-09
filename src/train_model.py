# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 13:54:14 2023

@author: zq19140
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

month = '11'
location = 'all_locations'
mintime = 1996

#%%
"""Train model with C-band scatterometer data"""
#%%
#Get training dataset
scat = 'C'
dataset_X_train, dataset_y_train = func_dataset.one_month_training(location, scat, month)
dataset_X_train, dataset_y_train = func_dataset.removeCAA(dataset_X_train, dataset_y_train)

#%%
#Fit model
forest_reg = RandomForestRegressor(n_estimators=95, max_depth=15, max_features=5)
forest_reg.fit(dataset_X_train, dataset_y_train.sit.values)
#Save the model to disk
filename = '../model/RFR_'+scat+'_'+month+'.sav'
pickle.dump(forest_reg, open(filename, 'wb'))


#%%
"""Train model with Ku-band scatterometer data"""
#%%
#Get training dataset
scat = 'Ku'
dataset_X_train, dataset_y_train = func_dataset.one_month_training(location, scat, month)
dataset_X_train, dataset_y_train = func_dataset.removeCAA(dataset_X_train, dataset_y_train)

#%%
#Fit model
forest_reg = RandomForestRegressor(n_estimators=95, max_depth=15, max_features=5)
forest_reg.fit(dataset_X_train, dataset_y_train.sit.values)
#Save the model to disk
filename = '../model/RFR_'+scat+'_'+month+'.sav'
pickle.dump(forest_reg, open(filename, 'wb'))


