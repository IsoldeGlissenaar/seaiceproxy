# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:18:19 2022

@author: Isolde Glissenaar

# Date: 14/02/2022
# Name: add_scatKu_trainingdata.py
# Description: Add scatterometer data to training dataset
# Input requirements: Training dataset_X and dataset_y, OSCAT-1 and OSCAT-2 scatterometer data, CryoSat-2 SIT grid
# Output: dataset_X and dataset_y for training including Ku band scatterometer data
"""

import sys
sys.path.append('../../functions')
from sir_io import loadsir
from sir_geom import pix2latlon

from func_reproj import backproject
import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy import spatial
from julian_todate import date_to_jd

# Select month and region to run for
m = 4
location = 'EasternArctic'

# Import training dataset
dataset_X = pd.read_csv('../../../data/interim/datasetsML/monthly/'+location+'/raw/dataset_x_2011-2019_'+str("{:02d}".format(m))+'.csv')
dataset_y = pd.read_csv('../../../data/interim/datasetsML/monthly/'+location+'/raw/dataset_y_2011-2019_'+str("{:02d}".format(m))+'.csv')

days_chart = np.unique(dataset_y.day)

#%%
# Import OSCAT-1 scatterometer data
if m < 2:
    years_os = np.array([2011,2012,2013,2014])
elif (m>1)*(m<5):    
    years_os = np.array([2011,2012,2013])
elif m > 9:
    years_os = np.array([2011,2012,2013])
days = np.arange(1,32,1)
oscat = np.zeros((93636,len(years_os),31))

for i in range(len(years_os)):
    for j in range(len(days)):
        yr = years_os[i]
        d = days[j]
        
        jd = date_to_jd(-4712, m, d)

        try:
            f = 'oueh-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd+0.5)))+'-'+str("{:03d}".format(int(jd+0.5)))+'.grd'
            sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/OSCAT/'+str(yr)+'/grd/'+f
            image, head, descrip, iaopt = loadsir(sir_fname)
        except FileNotFoundError:
            try:
                f = 'oueh-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd-0.5)))+'-'+str("{:03d}".format(int(jd-0.5)))+'.grd'
                sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/OSCAT/'+str(yr)+'/grd/'+f
                image, head, descrip, iaopt = loadsir(sir_fname)
            except FileNotFoundError:
                try:
                    f = 'oueh-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd+1.5)))+'-'+str("{:03d}".format(int(jd+1.5)))+'.grd'
                    sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/OSCAT/'+str(yr)+'/grd/'+f
                    image, head, descrip, iaopt = loadsir(sir_fname)                
                except (FileNotFoundError, IndexError):
                    print('FileNotFoundError: no OSCAT-1 file available '+str(yr)+' '+str(m)+' '+str(d)+' jd: '+str(yr)+str("{:03d}".format(int(jd+0.5))))
           
        oscat[:,i,j] = image.flatten()
 
    
x = np.arange(0,306,1)
y = np.arange(0,306,1)
x,y = np.meshgrid(x,y)

alon1,alat1 = pix2latlon(x, y, head)
alon = alon1[0,:,:].flatten()
alat = alat1[0,:,:].flatten()
coord_oscat = np.transpose(np.array([alat, alon]))

#%%
# Import OSCAT-2 scatterometer data
if m < 5:
    years_ss = np.array([2017,2018,2019,2020])
elif m > 10:
    years_ss = np.array([2016,2017,2018,2019])
days = np.arange(1,32,1)
scatsat = np.zeros((9006001,len(years_ss),31), dtype=int)

for i in range(len(years_ss)):
    for j in range(len(days)):
        yr = years_ss[i]
        d = days[j]
        
        jd = date_to_jd(-4712, m, d)

        try:
            f = 'S1L4SH_'+str(yr)+str("{:03d}".format(int(jd+0.5)))+'_BTH_NP_v1.1.*_1.*.tif'
            sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/ScatSat-1/'+f
            fs = glob.glob(sir_fname)
            ss = xr.open_rasterio(fs[-1])
        except (FileNotFoundError, IndexError):
            try:
                f = 'S1L4SH_'+str(yr)+str("{:03d}".format(int(jd-0.5)))+'_BTH_NP_v1.1.*_1.*.tif'
                sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/ScatSat-1/'+f
                fs = glob.glob(sir_fname)
                ss = xr.open_rasterio(fs[-1])
            except (FileNotFoundError, IndexError):
                try:
                    f = 'S1L4SH_'+str(yr)+str("{:03d}".format(int(jd+1.5)))+'_BTH_NP_v1.1.*_1.*.tif'
                    sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/ScatSat-1/'+f
                    fs = glob.glob(sir_fname)
                    ss = xr.open_rasterio(fs[-1])
                except (FileNotFoundError, IndexError):
                    print('FileNotFoundError: no OSCAT-2 file available '+str(yr)+' '+str(m)+' '+str(d)+' jd: '+str(yr)+str("{:03d}".format(int(jd+0.5))))
                
        scatsat[:,i,j] = ss.values.flatten()
 
xs,ys = np.meshgrid(ss.x.values, ss.y.values)
lat, lon = backproject(xs,ys)
lat = lat.flatten()
lon = lon.flatten()
    
coord_scatsat = np.transpose(np.array([lat, lon]))



#%%
# Regrid scatterometer data to training dataset grid

cs = xr.open_dataset('C:/Users/zq19140/OneDrive - University of Bristol/Documents/Jack_onedrive/Existing Sea Ice Thickness Datasets/CryoSat-2 Sea Ice Thickness Grids/ubristol_cryosat2_seaicethickness_nh25km_'+str(yr)+'_'+str("{:02d}".format(m))+'_v1.nc')
cs = cs.where(cs.x!=cs.x[-1], drop=True); cs = cs.where(cs.y!=cs.y[-1], drop=True)
cs = cs.coarsen(x=2).mean().coarsen(y=2).mean()
    
cs2_thickness = cs.Sea_Ice_Thickness.values.flatten()[~np.isnan(cs.Sea_Ice_Thickness.values.flatten())]

coord = np.transpose(np.array([cs.Latitude.values.flatten(), cs.Longitude.values.flatten()]))[~np.isnan(cs.Sea_Ice_Thickness.values.flatten())]
MdlKDT = spatial.KDTree(coord_oscat)
dist, closest = MdlKDT.query(coord, k=1, distance_upper_bound=0.5)
closest[closest==len(coord_oscat)] = 0
oscat_cs = oscat[closest,:,:]
oscat_cs[oscat_cs<-32] = np.nan

MdlKDT = spatial.KDTree(coord_scatsat)
dist, closest = MdlKDT.query(coord, k=1, distance_upper_bound=0.5)
closest[closest==len(coord_scatsat)] = 0
scatsat_cs = scatsat[closest,:,:]*0.001-50         #Correction, data scale=0.001, data offset=-50
scatsat_cs[scatsat_cs>0] = np.nan

#%%
# Combine scatterometer data and training dataset
if m < 5:    
    years = np.array([2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
elif m > 9:
    years = np.array([2010,2011,2012,2013,2014,2015,2016,2017,2018,2019])

scat_x = np.zeros((0,1))
for j in range(len(years)):
    for k in range(len(days)):
        loc_yr = np.where((dataset_y.year==years[j])&(dataset_y.day==days[k]))
        if len(loc_yr[0])>0:
            dataset_x_yr = dataset_X.iloc[loc_yr[0],:]
            dataset_y_yr = dataset_y.iloc[loc_yr[0],:]
            scat_x_yr = np.zeros((len(dataset_y_yr),1))
            for i in range(len(dataset_y_yr)):
                loc = np.where((coord[:,0]==dataset_y_yr.iloc[i,0])&(coord[:,1]==dataset_y_yr.iloc[i,1]))
                if len(loc[0])==1:
                    if np.isin(years[j],years_os):
                        idx = np.where(years[j]==years_os)[0]
                        scat_x_yr[i,0] = oscat_cs[loc[0],idx,k]
                    elif np.isin(years[j],years_ss):
                        idx = np.where(years[j]==years_ss)[0]
                        scat_x_yr[i,0] = scatsat_cs[loc[0],idx,k]
                    else:
                        scat_x_yr[i,0] = np.nan
                else:
                    scat_x_yr[i,0] = np.nan
            scat_x = np.append(scat_x, scat_x_yr, axis=0)

#%%

dataset_X['Scatterometer'] = scat_x

dataset_y = dataset_y.drop(np.where(np.isnan(dataset_X))[0])
dataset_y = dataset_y.reset_index(drop=True)
dataset_X = dataset_X.drop(np.where(np.isnan(dataset_X))[0])
dataset_X = dataset_X.reset_index(drop=True)


dataset_X.to_csv('../../../data/interim/datasetsML/monthly/'+location+'/incl_Ku/dataset_x_2011-2019_'+str("{:02d}".format(m))+'.csv',
                    index=False)
dataset_y.to_csv('../../../data/interim/datasetsML/monthly/'+location+'/incl_Ku/dataset_y_2011-2019_'+str("{:02d}".format(m))+'.csv',
                    index=False)







