# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:15:56 2021

@author: Isolde Glissenaar

# Date: 11/11/2021
# Name: create_dataset_1992-2020.py
# Description: Script that creates dataset_X and dataset_y with information from ice charts 
#              used for predicting sea ice thickness in the Canadian Arctic. Can be run for 
#              months Nov-Apr and CIS ice chart regions WesternArctic, EasternArctic, and HudsonBay. 
# Input requirements: Canadian Ice Service ice charts, one CryoSat-2 monthly gridded sea ice thickness
# Output: dataset_X and dataset_y for predicting sea ice thickness
"""

import glob
import numpy as np
import geopandas
import pandas as pd
from shapely.geometry import Point # Point class
import xarray as xr

def reproject(x,y):
    from pyproj import Proj, transform
    inProj = Proj('epsg:4326')
    outProj = Proj('epsg:3413')
    x1,y1 = x, y
    xnew,ynew = transform(inProj,outProj,x1,y1)
    return xnew, ynew

def CS2_coordinates(yr, m):
    """ Get CryoSat-2 grid coordinates """
    direc = 'insert direc source to CryoSat-2 SIT'
    cs = xr.open_dataset(direc+'/CryoSat-2 Sea Ice Thickness Grids/ubristol_cryosat2_seaicethickness_nh25km_'+yr+'_'+m+'_v1.nc')
    cs = cs.where(cs.x!=cs.x[-1], drop=True); cs = cs.where(cs.y!=cs.y[-1], drop=True)
    cs = cs.coarsen(x=2).mean().coarsen(y=2).mean()
    
    lon = cs.Longitude.values.flatten()
    lat = cs.Latitude.values.flatten()
    return(lon, lat)


def chart_filenames(yr, m, location):
    src = 'insert direc source to CIS ice charts'
    direc = src+'/Canadian_ice_service_charts/'+location+'/Ice Chart Shapefiles/' 
    filenames = glob.glob(direc+'*_'+yr+m+'*_polygon.shp')
    return filenames


def open_shapefile(filename):
    data = geopandas.read_file(filename)
    data = data.to_crs(epsg=3413)
    # data = data.to_crs(epsg=4326)
    return(data)


def pnts_in_shape(pnts, shape):
    all_shapes = shape.geometry
    all_shapes = all_shapes.buffer(0)
    # all_shapes = all_shapes[all_shapes.is_valid]
    shape_idx = np.zeros(len(pnts)); shape_idx[:]=np.nan
    for key,geom in all_shapes.items():
        p = pnts.within(geom)
        idx = np.where(p==True)
        if len(idx[0])>0:
            shape_idx[idx[0]] = key
    return(shape_idx)


def get_partialconc(shapes, shape_idx):
    start = np.where(shapes.columns == 'E_CT')[0][0]
    shapes_cut = shapes.iloc[:,start:start+16]
    shp_arr = np.array(shapes_cut)
    shp_arr[shp_arr=='9+']='10'; shp_arr[shp_arr=='X']='0'
    #Fix '1.'->'10'
    for i in range(shp_arr.shape[0]):
        for j in range(shp_arr.shape[1]):
            if shp_arr[i,j] is not None:
                if '.' in shp_arr[i,j]:
                    shp_arr[i,j] = shp_arr[i,j][0]+'0'
                    
    shp_arr = np.where(shp_arr=='L', 0, shp_arr)   
    shp_arr = np.where(shp_arr=='L.', 0, shp_arr)
    shp_arr = np.where(shp_arr=='L0', 0, shp_arr)
  
    shp_arr = shp_arr.astype(float)
    shp_arr[:,:5] = shp_arr[:,:5]/10
    shp_arr[:,6:10][shp_arr[:,6:10]<=10] = shp_arr[:,6:10][shp_arr[:,6:10]<=10]-1
    shp_arr[:,6:10][shp_arr[:,6:10]==40] = 10
    shp_arr[:,6:10][shp_arr[:,6:10]==70] = 11        
    shp_arr[:,6:10][shp_arr[:,6:10]==80] = 12
    shp_arr[:,6:10][shp_arr[:,6:10]==90] = 13
    return shp_arr


def get_icetypes(shp_arr, shape_idx):
    icetype = np.zeros((shp_arr.shape[0],14))   

    for l in range(len(shp_arr[:,0])):
        stages = np.where(~np.isnan(shp_arr[l,6:10]))[0]
        if len(stages)>1:
            for m in stages:
                icetype[l,int(shp_arr[l,6+m])] = icetype[l,int(shp_arr[l,6+m])] + shp_arr[l,1+m]
        elif len(stages)==1:
            icetype[l,int(shp_arr[l,6])] = shp_arr[l,0]
        elif len(stages)==0:
            icetype[l,:] = 0
    
    Num_Conc_icetype = icetype[shape_idx,:]
    return Num_Conc_icetype


def get_floesize(shp_arr, shape_idx):
    floesize = np.zeros((shp_arr.shape[0],10))   

    for l in range(len(shp_arr[:,0])):
        stages = np.where(~np.isnan(shp_arr[l,11:15]))[0]
        if len(stages)>1:
            for m in stages:
                floesize[l,int(shp_arr[l,11+m])] = floesize[l,int(shp_arr[l,11+m])] + shp_arr[l,1+m]
        elif len(stages)==1:
            floesize[l,int(shp_arr[l,11])] = shp_arr[l,0]
        elif len(stages)==0:
            floesize[l,:] = 0
            
    Num_Conc_floesize = floesize[shape_idx,:]
    return Num_Conc_floesize


def create_datasetX(arr):
    header = ['Num_Conc_NewIce', 'Num_Conc_Nilas', 'Num_Conc_YoungIce', 'Num_Conc_GreyIce',
              'Num_Conc_GreyWhiteIce', 'Num_Conc_FYI', 'Num_Conc_FYI_Thin', 'Num_Conc_FYI_Thin_First',
              'Num_Conc_FYI_Thin_Second', 'Num_Conc_FYI_Med', 'Num_Conc_FYI_Thick',
              'Num_Conc_OldIce', 'Num_Conc_SYI', 'Num_Conc_MYI',
              'Num_Conc_Pancake', 'Num_Conc_SIceCake', 'Num_Conc_IceCake', 
              'Num_Conc_SmallFloe', 'Num_Conc_MediumFloe', 'Num_Conc_BigFloe',
              'Num_Conc_VastFloe', 'Num_Conc_GiantFloe', 'Num_Conc_FastIce', 'Num_Conc_IceBerg']
    
    dataset_X = pd.DataFrame(arr,
                             columns=header)
    return dataset_X

def create_datasety(lat, lon, yr, day): 
    yr_data = np.full(len(lat), int(yr))
    no = np.full(len(lat), int(day))
    data_y = {'lat':lat,
              'lon':lon,
              'year':yr_data,
              'day':no
              }
    
    dataset_y = pd.DataFrame(data_y)
    return dataset_y


#%%

def main():
    #Get CS2 coordinates
    lon, lat = CS2_coordinates('2013', '01')
    lon_c, lat_c = reproject(lat,lon)
    point_to_check = [Point(lon_c[i], lat_c[i]) for i in range(len(lon_c))]
    pnts = geopandas.GeoDataFrame(geometry=point_to_check)
        
    m = '04'
    location = 'EasternArctic'
    
    years_arr = np.arange(1992,2021,1)
    
    first = True
    for y in range(len(years_arr)):
        yr = str(years_arr[y])
        #Loop over charts in that month
        shp_filenames = chart_filenames(yr, m, location)
        for c in range(len(shp_filenames)):
            #Get ice chart
            day = shp_filenames[c][133+len(location):135+len(location)]
            shapes = open_shapefile(shp_filenames[c])
            if int(yr)<2020:
                shapes = shapes.drop(np.where(shapes.A_LEGEND=='No data')[0]).reset_index(drop=True)
                shapes = shapes.drop(np.where(shapes.A_LEGEND=='Land')[0]).reset_index(drop=True)
            elif int(yr)==2020:
                if (int(m)==1 and int(day)<14):
                    shapes = shapes.drop(np.where(shapes.A_LEGEND=='No data')[0]).reset_index(drop=True)
                    shapes = shapes.drop(np.where(shapes.A_LEGEND=='Land')[0]).reset_index(drop=True)
                elif (int(m)==1 and int(day)>14) or (int(m)==2) or (int(m)==3 and int(day)<3):
                    shapes = shapes.drop(np.where(shapes.SGD_POLY_T=='L')[0]).reset_index(drop=True)
                elif (int(m)==3 and int(day)>4) or (int(m)==4):
                    shapes = shapes.drop(np.where(shapes.POLY_TYPE=='L')[0]).reset_index(drop=True)
            #Project ice chart on grid
            shape_idx = pnts_in_shape(pnts, shapes)
            
            #Remove land and NaNs
            shape_idx[shape_idx==21] = np.nan
            lon_t = lon[~np.isnan(shape_idx)]; lat_t = lat[~np.isnan(shape_idx)]
            shape_idx = shape_idx[~np.isnan(shape_idx)].astype(int)
            
            #Get partial concentrations from ice chart
            shp_arr = get_partialconc(shapes, shape_idx)
            #Get ice type + floe size concentrations
            Num_Conc_icetype = get_icetypes(shp_arr, shape_idx)
            Num_Conc_floesize = get_floesize(shp_arr, shape_idx)
            Num_Conc = np.concatenate((Num_Conc_icetype, Num_Conc_floesize), axis=1)
            Num_Conc[np.isnan(Num_Conc)] = 0
            
            #Create dataset
            dataset_X_t = create_datasetX(Num_Conc)
            dataset_y_t = create_datasety(lat_t, lon_t, yr, day)
            
            if first:
                dataset_X = dataset_X_t
                dataset_y = dataset_y_t
                first = False
            else:
                dataset_X = pd.concat((dataset_X, dataset_X_t), axis=0)
                dataset_y = pd.concat((dataset_y, dataset_y_t), axis=0)
        print(yr)
    
    dataset_X = dataset_X.reset_index(drop=True)
    dataset_y = dataset_y.reset_index(drop=True)
    
    #Save dataset
    idx_nan = np.where(np.isnan(dataset_y))[0]
    dataset_X = dataset_X.drop(idx_nan).reset_index(drop=True)
    dataset_y = dataset_y.drop(idx_nan).reset_index(drop=True)
    dataset_X.to_csv('../../data/dataset/'+location+'_dataset_x_1992-2019_'+m+'_noCS2.csv',
                        index=False)
    dataset_y.to_csv('../../data/dataset/'+location+'_dataset_y_1992-2019_'+m+'_noCS2.csv',
                        index=False)



if __name__=="__main__":
    main()
    
    