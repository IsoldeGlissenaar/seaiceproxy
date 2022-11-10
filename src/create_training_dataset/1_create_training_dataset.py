# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:42:51 2021

@author: Isolde Glissenaar

# Date: 18/11/2021
# Name: create_training_dataset.py
# Description: Script that creates training dataset_X and dataset_y with information from ice charts
#              and the observed SIT from CryoSat-2. Can be run for months Nov-Apr and CIS ice chart
#              regions WesternArctic, EasternArctic, and HudsonBay. 
# Input requirements: CryoSat-2 monthly gridded sea ice thickness, Canadian Ice Service ice charts.
# Output: dataset_X and dataset_y for the training of a machine learning model
"""

import glob
import numpy as np
import xarray as xr
import geopandas
import pandas as pd
from shapely.geometry import Point # Point class

def reproject(x,y):
    from pyproj import Proj, transform
    inProj = Proj('epsg:4326')
    outProj = Proj('epsg:3413')
    x1,y1 = x, y
    xnew,ynew = transform(inProj,outProj,x1,y1)
    return xnew, ynew

def CS2_coordinates():
    """ Get CryoSat-2 grid coordinates """
    cs = xr.open_dataset('C:/Users/zq19140/OneDrive - University of Bristol/Documents/Jack_onedrive/Existing Sea Ice Thickness Datasets/CryoSat-2 Sea Ice Thickness Grids/ubristol_cryosat2_seaicethickness_nh25km_2011_04_v1.nc')
    cs = cs.where(cs.x!=cs.x[-1], drop=True); cs = cs.where(cs.y!=cs.y[-1], drop=True)
    cs = cs.coarsen(x=2).mean().coarsen(y=2).mean()
    
    lon = cs.Longitude.values.flatten()
    lat = cs.Latitude.values.flatten()
    return(lon, lat)


def chart_filenames(yr, m, location):
    """ Find available ice charts for given year, month and region """
    direc = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Canadian_ice_service_charts/'+location+'/Ice Chart Shapefiles/' 
    filenames = glob.glob(direc+'*_'+yr+m+'*_polygon.shp')
    return filenames


def open_shapefile(filename):
    data = geopandas.read_file(filename)
    data = data.to_crs(epsg=3413)
    # data = data.to_crs(epsg=4326)
    return(data)


def pnts_in_shape(pnts, shape):
    """ Compares point locations [lon, lat] with a shapefile and returns shape 
        index for each point """
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


def get_sit(yr, m, day):
    """ Loads CryoSat-2 gridded sea ice thickness product and regrids to 
        50km grid. Removes 5% data with highest uncertainty and returns SIT """
    cs = xr.open_dataset('C:/Users/zq19140/OneDrive - University of Bristol/Documents/Jack_onedrive/Existing Sea Ice Thickness Datasets/CryoSat-2 Sea Ice Thickness Grids/ubristol_cryosat2_seaicethickness_nh25km_'+yr+'_'+m+'_v1.nc')
    cs = cs.where(cs.x!=cs.x[-1], drop=True); cs = cs.where(cs.y!=cs.y[-1], drop=True)
    cs = cs.coarsen(x=2).mean().coarsen(y=2).mean()

    sit = cs.Sea_Ice_Thickness.values
    sit_uncer = cs.Sea_Ice_Thickness_Uncertainty.values
    sit[sit_uncer>0.48] = np.nan
    sit[np.isnan(sit_uncer)] = np.nan
    return(sit.flatten())


def get_partialconc(shapes, shape_idx):
    """ Retrieve gridded partial concentrations from ice charts' egg codes """
    start = np.where(shapes.columns == 'E_CT')[0][0]
    shapes_cut = shapes.iloc[:,start:start+16]
    part_conc = np.array(shapes_cut)
    part_conc[part_conc=='9+']='10'; part_conc[part_conc=='X']='0'
    #Fix '1.'->'10'
    for i in range(part_conc.shape[0]):
        for j in range(part_conc.shape[1]):
            if part_conc[i,j] is not None:
                if '.' in part_conc[i,j]:
                    part_conc[i,j] = part_conc[i,j][0]+'0'
                    
    part_conc = np.where(part_conc=='L', 0, part_conc)   
    part_conc = np.where(part_conc=='L.', 0, part_conc)
  
    part_conc = part_conc.astype(float)
    part_conc[:,:5] = part_conc[:,:5]/10
    part_conc[:,6:10][part_conc[:,6:10]<=10] = part_conc[:,6:10][part_conc[:,6:10]<=10]-1
    part_conc[:,6:10][part_conc[:,6:10]==40] = 10
    part_conc[:,6:10][part_conc[:,6:10]==70] = 11        
    part_conc[:,6:10][part_conc[:,6:10]==80] = 12
    part_conc[:,6:10][part_conc[:,6:10]==90] = 13
    return part_conc


def get_icetypes(part_conc, shape_idx):
    """ Retrieve partial concentration of the ice types from the ice charts """
    icetype = np.zeros((part_conc.shape[0],14))   

    for l in range(len(part_conc[:,0])):
        stages = np.where(~np.isnan(part_conc[l,6:10]))[0]
        if len(stages)>1:
            for m in stages:
                icetype[l,int(part_conc[l,6+m])] = icetype[l,int(part_conc[l,6+m])] + part_conc[l,1+m]
        elif len(stages)==1:
            icetype[l,int(part_conc[l,6])] = part_conc[l,0]
        elif len(stages)==0:
            icetype[l,:] = 0
    
    Num_Conc_icetype = icetype[shape_idx,:]
    return Num_Conc_icetype


def get_floesize(part_conc, shape_idx):
    """ Retrieve partial concentration of floe sizes from the ice charts """
    floesize = np.zeros((part_conc.shape[0],10))   

    for l in range(len(part_conc[:,0])):
        stages = np.where(~np.isnan(part_conc[l,11:15]))[0]
        if len(stages)>1:
            for m in stages:
                floesize[l,int(part_conc[l,11+m])] = floesize[l,int(part_conc[l,11+m])] + part_conc[l,1+m]
        elif len(stages)==1:
            floesize[l,int(part_conc[l,11])] = part_conc[l,0]
        elif len(stages)==0:
            floesize[l,:] = 0
            
    Num_Conc_floesize = floesize[shape_idx,:]
    return Num_Conc_floesize


def create_datasetX(arr):
    """ Create dataset_X with partial concentration of ice types and floe size """
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

def create_datasety(lat, lon, yr, day, sit): 
    """ Create dataset_y of observed sea ice thickness """
    yr_data = np.full(len(lat), int(yr))
    no = np.full(len(lat), int(day))
    data_y = {'lat':lat,
              'lon':lon,
              'year':yr_data,
              'day':no,
              'sit':sit
              }
    
    dataset_y = pd.DataFrame(data_y)
    return dataset_y

#%%

def main():
    #Select month and location to run this code for
    m = '04'  
    location = 'WesternArctic'
    
    #Get CS2 coordinates
    lon, lat = CS2_coordinates()
    lon_c, lat_c = reproject(lat,lon)
    point_to_check = [Point(lon_c[i], lat_c[i]) for i in range(len(lon_c))]
    pnts = geopandas.GeoDataFrame(geometry=point_to_check)
    
    if int(m) < 5:    
        years_arr = np.array([2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
    elif int(m) > 10:
        years_arr = np.array([2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])
    
    first = True
    for y in range(len(years_arr)):
        yr = str(years_arr[y])
        #Loop over charts in that month
        shp_filenames = chart_filenames(yr, m, location)
        
        for c in range(len(shp_filenames)):
            #Get ice 
            idx = [pos for pos, char in enumerate(shp_filenames[c]) if char=='/'][-1]
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
                    
            #Get CS2 SIT
            sit = get_sit(yr, m, day)

            #Project ice chart on grid
            shape_idx = pnts_in_shape(pnts, shapes)
            
            #Remove land and NaNs
            shape_idx[shape_idx==21] = np.nan
            lon_t = lon[~np.isnan(shape_idx)]; lat_t = lat[~np.isnan(shape_idx)]; sit_t = sit[~np.isnan(shape_idx)]
            shape_idx = shape_idx[~np.isnan(shape_idx)].astype(int)
            
            #Get partial concentrations from ice chart
            part_conc = get_partialconc(shapes, shape_idx)
            #Get ice type + floe size concentrations
            Num_Conc_icetype = get_icetypes(part_conc, shape_idx)
            Num_Conc_floesize = get_floesize(part_conc, shape_idx)
            Num_Conc = np.concatenate((Num_Conc_icetype, Num_Conc_floesize), axis=1)
            Num_Conc[np.isnan(Num_Conc)] = 0
            
            #Create dataset
            dataset_X_t = create_datasetX(Num_Conc)
            dataset_y_t = create_datasety(lat_t, lon_t, yr, day, sit_t)
            
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
    dataset_X.to_csv('../../../data/interim/datasetsML/monthly/'+location+'/raw/dataset_x_2011-2019_'+m+'.csv',
                        index=False)
    dataset_y.to_csv('../../../data/interim/datasetsML/monthly/'+location+'/raw/dataset_y_2011-2019_'+m+'.csv',
                        index=False)



if __name__=="__main__":
    main()
    
    
