# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:58:20 2022

@author: zq19140
"""


import pandas as pd
import numpy as np
import shapefile
from shapely.geometry import Point # Point class
from shapely.geometry import shape # shape() is a function to convert geo objects through the interface
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m')

def removeCAA(dataset_X, dataset_y):
    idx = []
    shp = shapefile.Reader('location MAISIE shapefile CAA') #open the shapefile
    all_shapes = shp.shapes() # get all the polygons   
    boundary = all_shapes[1] # get a boundary polygon
    for i in range(len(dataset_y)):
        point_to_check = (dataset_y.lon[i],dataset_y.lat[i]) # an x,y tuple
        if Point(point_to_check).within(shape(boundary)):
            idx.append(i)
        
    dataset_X = dataset_X.drop(idx).reset_index(drop=True)
    dataset_y = dataset_y.drop(idx).reset_index(drop=True)
    
    return dataset_X, dataset_y


def one_month_training(location, scat, month):    
    '''Get training data 2011-2020 for one given month
    '''
    #Open training data
    dataset_X_train = pd.read_csv('../../data/training_dataset/'+scat+'-band/dataset_x_2011-2020_'+month+'.csv') 
    dataset_y_train = pd.read_csv('../../data/training_dataset/'+scat+'-band/dataset_y_2011-2020_'+month+'.csv')
    dataset_X_train = dataset_X_train.drop(np.where(dataset_y_train.lat<60)[0]).reset_index(drop=True)
    dataset_y_train = dataset_y_train.drop(np.where(dataset_y_train.lat<60)[0]).reset_index(drop=True)
    
    return dataset_X_train, dataset_y_train
    
    
    
    
def get_testing_data(dataset_X, dataset_y, yr_compare, month, all_months='off'):
    '''Get dataset and take one month+year to test on
    '''
    dataset_X_train2, dataset_X_test = train_test_split(dataset_X, test_size=0.2, random_state=42)
    dataset_y_train2, dataset_y_test = train_test_split(dataset_y, test_size=0.2, random_state=42)
    dataset_X_train2 = dataset_X_train2.reset_index(drop=True)
    dataset_y_train2 = dataset_y_train2.reset_index(drop=True)
        
    dataset_X_2018 = dataset_X_train2.drop(np.where(dataset_y_train2.year!=int(yr_compare))[0]).reset_index(drop=True)                        
    dataset_y_2018 = dataset_y_train2.drop(np.where(dataset_y_train2.year!=int(yr_compare))[0]).reset_index(drop=True)
    
    if all_months=='on':                                 
        dataset_y_2018 = dataset_y_2018.drop(np.where(dataset_X_2018.month!=int(month))[0]).reset_index(drop=True)
        dataset_X_2018 = dataset_X_2018.drop(np.where(dataset_X_2018.month!=int(month))[0]).reset_index(drop=True)
    
    return dataset_X_2018, dataset_y_2018


def open_1992_2020_data(scat, month):
    #Open dataset 1990-2019
    dataset_X = pd.read_csv('../../data/dataset/'+scat+'-band/dataset_x_1992-2020_'+month+'.csv') 
    dataset_y = pd.read_csv('../../data/dataset/'+scat+'-band/dataset_y_1992-2020_'+month+'.csv') #lat, lon, yr_data, dataset_y[:,0]
    
    land = np.where(np.nansum(dataset_X.drop('Scatterometer', axis=1), axis=1)==0)[0]
    dataset_y = dataset_y.drop(land).reset_index(drop=True)
    dataset_X = dataset_X.drop(land).reset_index(drop=True)
    
    return dataset_X, dataset_y


def plot_sit(lon, lat, sit, location, title='',
             cmap='Spectral_r', vmin=0, vmax=3):
    if location=='WesternArctic':
        clon = -120
        clat = 70
        minlon = -160
        maxlon = -100
        minlat = 66
        maxlat = 80
        s=20
    elif location=='EasternArctic':
        clon = -60
        clat = 60
        minlon = -80
        maxlon = -43
        minlat = 58
        maxlat = 82
        s=15
    elif location=='HudsonBay':
        clon = -75
        clat = 60
        minlon = -95
        maxlon = -65
        minlat = 50
        maxlat = 70
        s=10
    elif location=='all_locations':
        clon = -99
        clat = 70
        minlon = -140
        maxlon = -57
        minlat = 62
        maxlat = 84
        s=5 
        

    fig=plt.figure(dpi=200)
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=clon, central_latitude=clat, globe=None))
    ax.coastlines(resolution='50m',linewidth=0.5)
    ax.set_extent([minlon,maxlon,minlat,maxlat],crs=ccrs.PlateCarree()) 
    ax.gridlines(linewidth=0.3, color='k', alpha=0.5, linestyle=':')
    im = plt.scatter(lon, lat, c=sit,cmap=cmap,vmin=vmin,vmax=vmax,s=s,transform=ccrs.PlateCarree())
    ax.add_feature(land_50m, facecolor='#eeeeee')
    cbar = fig.colorbar(im, ax=ax, label='',extend='max', fraction=0.046, pad=0.04)
    cbar.ax.locator_params(nbins=4)
    plt.title(title)
