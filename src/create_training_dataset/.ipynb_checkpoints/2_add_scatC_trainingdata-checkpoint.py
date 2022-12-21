# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:48:12 2022

@author: Isolde Glissenaar

# Date: 28/02/2022
# Name: add_scatC_trainingdata.py
# Description: Add scatterometer data to training dataset
# Input requirements: Training dataset_X and dataset_y, ASCAT scatterometer data, CryoSat-2 SIT grid
# Output: dataset_X and dataset_y for training including C band scatterometer data
"""

import sys
sys.path.append('../../functions')
from sir_io import loadsir
from sir_geom import pix2latlon
import numpy as np
import pandas as pd
from scipy import spatial
import xarray as xr
from julian_todate import date_to_jd

# Select month and region to run for
m = 4
location = 'EasternArctic'

# Import training dataset
dataset_X = pd.read_csv('../../data/training_dataset/'+location+'_dataset_x_2011-2019_'+str("{:02d}".format(m))+'.csv')
dataset_y = pd.read_csv('../../data/training_dataset/'+location+'_dataset_y_2011-2019_'+str("{:02d}".format(m))+'.csv')

days_chart = np.unique(dataset_y.day)


if m < 5:    
    years = np.array([2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
elif m > 9:
    years = np.array([2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])
   
# Import ASCAT scatterometer data
days = np.arange(1,32,1)
ascat = np.zeros((23409,len(years),31))
for i in range(len(years)):
    for j in range(len(days)):
        yr = years[i]
        d = days[j]
        
        jd = date_to_jd(-4712, m, d)
        direc = 'insert direc source to scatterometer data'
        try:
            f = 'msfa-a-Arc'+str(yr)[2:]+'-' + str("{:03d}".format(int(jd+1.5))) + '-' + str("{:03d}".format(int(jd+2.5))) + '.grd'
            sir_fname = direc+'/Scatterometer/ASCAT/grd/'+f
            image, head, descrip, iaopt = loadsir(sir_fname)
        except FileNotFoundError:
            try:
                f = 'msfa-a-Arc'+str(yr)[2:]+'-' + str("{:03d}".format(int(jd+2.5))) + '-' + str("{:03d}".format(int(jd+3.5))) + '.grd'
                sir_fname = direc+'/Scatterometer/ASCAT/grd/'+f
                image, head, descrip, iaopt = loadsir(sir_fname)
            except FileNotFoundError:
                f = 'msfa-a-Arc'+str(yr)[2:]+'-' + str("{:03d}".format(int(jd-0.5))) + '-' + str("{:03d}".format(int(jd+0.5))) + '.grd'
                sir_fname = direc+'/Scatterometer/ASCAT/grd/'+f
                image, head, descrip, iaopt = loadsir(sir_fname)
           
        ascat[:,i,j] = image.flatten()

ers_fname = direc+'/Scatterometer/ERS2/grd/ers2-a-Arc00-001-006.grd'
__, head, __, __ = loadsir(ers_fname)
    
x = np.arange(0,153,1)+1
y = np.arange(0,153,1)+1
x,y = np.meshgrid(x,y)

alon1,alat1 = pix2latlon(x, y, head)
alon = alon1[0,:,:].flatten()
alat = alat1[0,:,:].flatten()
coord_ascat = np.transpose(np.array([alat, alon]))

#%%

# Project ASCAT data on CS2 grid
direc = 'insert direc source to CryoSat-2 SIT'
cs = xr.open_dataset(direc+'/CryoSat-2 Sea Ice Thickness Grids/ubristol_cryosat2_seaicethickness_nh25km_'+str(yr)+'_'+str("{:02d}".format(m))+'_v1.nc')
cs = cs.where(cs.x!=cs.x[-1], drop=True); cs = cs.where(cs.y!=cs.y[-1], drop=True)
cs = cs.coarsen(x=2).mean().coarsen(y=2).mean()
    
cs2_thickness = cs.Sea_Ice_Thickness.values.flatten() 

coord = np.transpose(np.array([cs.Latitude.values.flatten(), cs.Longitude.values.flatten()]))
MdlKDT = spatial.KDTree(coord_ascat)
dist, closest = MdlKDT.query(coord, k=1, distance_upper_bound=0.5)
closest[closest==len(coord_ascat)] = 0
ascat_cs = ascat[closest,:,:]


#%%
# Project to dataset_X
ascat_x = np.zeros((0,1))
for j in range(len(years)):
    for k in range(len(days)):
        loc_yr = np.where((dataset_y.year==years[j])&(dataset_y.day==days[k]))
        if len(loc_yr[0])>0:
            dataset_x_yr = dataset_X.iloc[loc_yr[0],:]
            dataset_y_yr = dataset_y.iloc[loc_yr[0],:]
            ascat_x_yr = np.zeros((len(dataset_y_yr),1))
            for i in range(len(dataset_y_yr)):
                loc = np.where((coord[:,0]==dataset_y_yr.iloc[i,0])&(coord[:,1]==dataset_y_yr.iloc[i,1]))
                if len(loc[0])==1:
                    ascat_x_yr[i,0] = ascat_cs[loc[0],j,k]
                else:
                    ascat_x_yr[i,0] = np.nan
            ascat_x = np.append(ascat_x, ascat_x_yr, axis=0)

#%%
# Add C-band scatterometer data to dataset_X
dataset_X['Scatterometer'] = ascat_x

dataset_y = dataset_y.drop(np.where(np.isnan(dataset_X))[0])
dataset_y = dataset_y.reset_index(drop=True)
dataset_X = dataset_X.drop(np.where(np.isnan(dataset_X))[0])
dataset_X = dataset_X.reset_index(drop=True)


dataset_X.to_csv('../../data/training_dataset/C-band/'+location+'_dataset_x_2011-2019_'+str("{:02d}".format(m))+'.csv',
                    index=False)
dataset_y.to_csv('../../data/training_dataset/C-band/'+location+'dataset_y_2011-2019_'+str("{:02d}".format(m))+'.csv',
                    index=False)







