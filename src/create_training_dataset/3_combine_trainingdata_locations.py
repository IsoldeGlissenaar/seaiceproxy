# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 13:36:08 2022

@author: Isolde Glissenaar

# Date: 06/01/2022
# Name: combine_trainingdata_locations.py
# Description: Combine locations WA, EA, HB into one dataset
# Input requirements: Training dataset_X and dataset_y with C-band or Ku-band scatterometer
# Output: Training dataset_X and dataset_y for all_locations
"""

import pandas as pd

cs2_period = 'monthly'
month = '04'
scat = 'Ku'
locations = ['WesternArctic', 'EasternArctic']#, 'HudsonBay']

for i in range(len(locations)):
    location = locations[i]
    dataset_X = pd.read_csv('C:/Users/zq19140/OneDrive - University of Bristol/Documents/Projects/icecharts_thickness/data/interim/datasetsML/'+cs2_period+'/'+locations[i]+'/incl_'+scat+'/dataset_x_2011-2019_'+month+'.csv') 
    dataset_y = pd.read_csv('C:/Users/zq19140/OneDrive - University of Bristol/Documents/Projects/icecharts_thickness/data/interim/datasetsML/'+cs2_period+'/'+locations[i]+'/incl_'+scat+'/dataset_y_2011-2019_'+month+'.csv')

    if i==0:
        dataset_X_tot = dataset_X.copy()
        dataset_y_tot = dataset_y.copy()
    elif i>0:
        dataset_X_tot = pd.concat([dataset_X_tot, dataset_X], ignore_index=True)
        dataset_y_tot = pd.concat([dataset_y_tot, dataset_y], ignore_index=True)
    

dataset_X_tot.to_csv('C:/Users/zq19140/OneDrive - University of Bristol/Documents/Projects/icecharts_thickness/data/interim/datasetsML/'+cs2_period+'/all_locations/incl_'+scat+'/dataset_x_2011-2019_'+month+'.csv',
                    index=False)
dataset_y_tot.to_csv('C:/Users/zq19140/OneDrive - University of Bristol/Documents/Projects/icecharts_thickness/data/interim/datasetsML/'+cs2_period+'/all_locations/incl_'+scat+'/dataset_y_2011-2019_'+month+'.csv',
                    index=False)


