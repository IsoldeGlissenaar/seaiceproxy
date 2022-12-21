# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 13:50:47 2022

@author: zq19140
"""

import sys
sys.path.append('../model/functions')
import func_dataset

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde

cs2_period = 'monthly'
location = 'all_locations'
all_months = 'off'
scat = 'C'

x=[]
y=[]

months = ['11','12','01','02','03','04']
scats = ['Ku','C']

for scat in scats:
    for month in months:
        #Get training dataset
    
        if all_months=='on':
            trainmonth='all'
            dataset_X, dataset_y = func_dataset.all_months_training(location, scat)
        elif all_months=='off':
            trainmonth=month    
            dataset_X, dataset_y = func_dataset.one_month_training(location, scat, month)
            
        dataset_X, dataset_y = func_dataset.removeCAA(dataset_X, dataset_y)
        
    
        #Select train + test data
        dataset_X_train, dataset_X_test = train_test_split(dataset_X, test_size=0.2, random_state=42)
        dataset_y_train, dataset_y_test = train_test_split(dataset_y, test_size=0.2, random_state=42)
        
        dataset_X_train = dataset_X_train.reset_index(drop=True)
        dataset_y_train = dataset_y_train.reset_index(drop=True)
    
        '''Train on 80%, test on 20%'''
        
        dataset_X_test = dataset_X_test.reset_index(drop=True)
        dataset_y_test = dataset_y_test.reset_index(drop=True)
        
        forest_reg = RandomForestRegressor(n_estimators=95, max_depth=15, max_features=5)
        forest_reg.fit(dataset_X_train, dataset_y_train.sit)
        predictions_t = forest_reg.predict(dataset_X_test)
        
        dataset_y_t = dataset_y_test.copy()
        dataset_X_t = dataset_X_test.copy()
        
        forest_mse = mean_squared_error(dataset_y_t.sit, predictions_t)
        forest_rmse = np.sqrt(forest_mse)
        print(f"RMSE: {forest_rmse:.3f} m")
        
        lin_scores = cross_val_score(forest_reg, dataset_X_t, dataset_y_t.sit,
                                     scoring="neg_mean_squared_error", cv=10)
        lin_rmse_scores = np.sqrt(-lin_scores)
        print(f"K-fold cross-validation mean: {np.nanmean(lin_rmse_scores)}")
        
        score = forest_reg.score(dataset_X_test, dataset_y_test.sit)
        
        x.extend(predictions_t.copy())
        y.extend(dataset_y_t.sit.values)

#%%
# Scatterplot predictions vs observed
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig = plt.figure(dpi=400)
plt.scatter(x, y, c=z, s=0.1)
plt.plot(np.arange(0,5),slope*np.arange(0,5)+intercept, color='darkblue', linestyle='--')
plt.xlabel('Model proxy sea ice thickness [m]')
plt.ylabel('Observed CryoSat-2 sea ice thickness [m]')
plt.plot([-1,5.5],[-1,5.5],linestyle=':',c='k',linewidth=1)
plt.xlim([-1,5.5])
plt.ylim([-1,5.5])
plt.grid(linestyle=':')
plt.show()

    
