# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:35:40 2022

@author: Isolde Glissenaar

# Date: 23/02/2022
# Name: combine_dataset_locations_1992-2020.py
# Description: Combine locations WA, EA, HB into one dataset
# Input requirements: Dataset_X and dataset_y with C-band or Ku-band scatterometer
# Output: Dataset_X and dataset_y for all_locations
"""

import pandas as pd

cs2_period = 'monthly'
month = '04'
scat = 'Ku'
locations = ['WesternArctic', 'EasternArctic']#, 'HudsonBay']

for i in range(len(locations)):
    location = locations[i]
    dataset_X = pd.read_csv('../../data/dataset/'+scat+'-band/'+location+'_dataset_x_1992-2019_'+month+'.csv') 
    dataset_y = pd.read_csv('../../data/dataset/'+scat+'-band/'+location+'_dataset_y_1992-2019_'+month+'.csv')

    dataset_y['location'] = location    

    if i==0:
        dataset_X_tot = dataset_X.copy()
        dataset_y_tot = dataset_y.copy()
    elif i>0:
        dataset_X_tot = pd.concat([dataset_X_tot, dataset_X], ignore_index=True)
        dataset_y_tot = pd.concat([dataset_y_tot, dataset_y], ignore_index=True)
    
    

dataset_X_tot.to_csv('../../data/dataset/'+scat+'-band/dataset_x_1992-2019_'+month+'.csv',
                    index=False)
dataset_y_tot.to_csv('../../data/dataset/'+scat+'-band/dataset_y_1992-2019_'+month+'.csv',
                    index=False)


