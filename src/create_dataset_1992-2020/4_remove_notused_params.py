# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:08:30 2022

@author: Isolde Glissenaar

# Date: 29/09/2022
# Name: 4_remove_notused_params.py
# Description: Remove iceberg parameter and combine old ice types into one category
# for months that SYI is not used in training dataset.
"""

import numpy as np
import pandas as pd
import cartopy.feature as cfeature
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m')

month = '04'
scat = 'C'

#%%
"""Predict sea ice thickness with C-band scatterometer data"""
#%%
dataset_X = pd.read_csv('../../data/interim/datasetsML/no_CS2/all_locations/incl_'+scat+'/dataset_x_1992-2019_'+month+'.csv')
dataset_y = pd.read_csv('../../data/interim/datasetsML/no_CS2/all_locations/incl_'+scat+'/dataset_y_1992-2019_'+month+'.csv')

dataset_X_train = pd.read_csv('../../data/interim/datasetsML/monthly/all_locations/incl_'+scat+'/dataset_x_2011-2019_'+month+'.csv')
dataset_y_train = pd.read_csv('../../data/interim/datasetsML/monthly/all_locations/incl_'+scat+'/dataset_y_2011-2019_'+month+'.csv')

#%%

dataset_X = dataset_X.drop(['Num_Conc_IceBerg'],'columns')
dataset_X_train = dataset_X_train.drop(['Num_Conc_IceBerg'],'columns')

months_noSYI = ['01','02','03','04']
if np.isin(month,months_noSYI):
    dataset_X['Num_Conc_OldIce'] = dataset_X['Num_Conc_OldIce'] + dataset_X['Num_Conc_SYI'] + dataset_X['Num_Conc_MYI']
    # dataset_X_train['Num_Conc_OldIce'] = dataset_X_train['Num_Conc_OldIce'] + dataset_X_train['Num_Conc_SYI'] + dataset_X_train['Num_Conc_MYI']
    dataset_X['Num_Conc_SYI'] = 0
    dataset_X['Num_Conc_MYI'] = 0



#%%
dataset_X.to_csv('../../data/interim/datasetsML/no_CS2/all_locations/incl_'+scat+'/edit/dataset_x_1992-2019_'+month+'.csv',
                    index=False)
dataset_y.to_csv('../../data/interim/datasetsML/no_CS2/all_locations/incl_'+scat+'/edit/dataset_y_1992-2019_'+month+'.csv',
                    index=False)

dataset_X_train.to_csv('../../data/interim/datasetsML/monthly/all_locations/incl_'+scat+'/edit/dataset_x_2011-2019_'+month+'.csv',
                       index=False)
dataset_y_train.to_csv('../../data/interim/datasetsML/monthly/all_locations/incl_'+scat+'/edit/dataset_y_2011-2019_'+month+'.csv',
                       index=False)
