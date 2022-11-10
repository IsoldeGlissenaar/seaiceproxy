# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:44:33 2022

@author: Isolde Glissenaar

# Date: 16/02/2022
# Name: add_scatKu_1992-2020.py
# Description: Add scatterometer data to 1992-2020 dataset
# Input requirements: dataset_X and dataset_y 1992-2020, OSCAT-1, OSCAT-2, and QuickScat 
#                     scatterometer data, CryoSat-2 SIT grid
# Output: dataset_X and dataset_y including Ku band scatterometer data
"""

import sys
sys.path.append('../functions')
from sir_io import loadsir
from sir_geom import pix2latlon

from scipy import spatial
from func_reproj import backproject
import glob
import numpy as np
import pandas as pd
import xarray as xr
from julian_todate import date_to_jd

# Select month and region to run for
m = 4
location = 'EasternArctic'

# Import dataset
dataset_X = pd.read_csv('../../data/interim/datasetsML/no_CS2/'+location+'/raw/'+location+'_dataset_x_1992-2019_'+str("{:02d}".format(m))+'_noCS2.csv')
dataset_y = pd.read_csv('../../data/interim/datasetsML/no_CS2/'+location+'/raw/'+location+'_dataset_y_1992-2019_'+str("{:02d}".format(m))+'_noCS2.csv')

days_chart = np.unique(dataset_y.day)

#%%
#OSCAT-1 (Nov 2009-Feb 2014)

if m < 3:
    years_os = np.array([2010,2011,2012,2013,2014])
elif (m>2)*(m<5):    
    years_os = np.array([2010,2011,2012,2013])
elif m > 9:
    years_os = np.array([2009,2010,2011,2012,2013])
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
                    oscat[:,i,j] = np.nan
                    continue 
                
        oscat[:,i,j] = image.flatten()
 
    
x = np.arange(0,306,1)
y = np.arange(0,306,1)
x,y = np.meshgrid(x,y)

alon1,alat1 = pix2latlon(x, y, head)
alon = alon1[0,:,:].flatten()
alat = alat1[0,:,:].flatten()
coord_oscat = np.transpose(np.array([alat, alon]))

#%%
#OSCAT-2 (Oct 2016-2020)

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
                    scatsat[:,i,j] = np.nan
                    continue
                
        #Correction, data scale=0.001, data offset=-50
        # ss.values = ss.values*0.001-50
        scatsat[:,i,j] = ss.values.flatten()
 
xs,ys = np.meshgrid(ss.x.values, ss.y.values)
lat, lon = backproject(xs,ys)
lat = lat.flatten()
lon = lon.flatten()
    
coord_scatsat = np.transpose(np.array([lat, lon]))

#%%
# QuickScat (Aug 1999- Oct 2009)

if m < 5:
    years_qs = np.arange(2000,2010,1)
elif m > 9:
    years_qs = np.arange(1999,2009,1)
days = np.arange(1,32,1)
qscat = np.zeros((93636,len(years_qs),31))

for i in range(len(years_qs)):
    for j in range(len(days)):
        yr = years_qs[i]
        d = days[j]
        
        jd = date_to_jd(-4712, m, d)

        try:
            f = 'queh-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd+0.5)))+'-'+str("{:03d}".format(int(jd+0.5)))+'.grd'
            sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/QuickSCAT/grd/'+f
            image, head, descrip, iaopt = loadsir(sir_fname)
        except FileNotFoundError:
            try:
                f = 'queh-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd-0.5)))+'-'+str("{:03d}".format(int(jd-0.5)))+'.grd'
                sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/QuickSCAT/grd/'+f
                image, head, descrip, iaopt = loadsir(sir_fname)
            except FileNotFoundError:
                try:
                    f = 'queh-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd+1.5)))+'-'+str("{:03d}".format(int(jd+1.5)))+'.grd'
                    sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/QuickSCAT/grd/'+f
                    image, head, descrip, iaopt = loadsir(sir_fname)                
                except (FileNotFoundError, IndexError):
                    print('FileNotFoundError: no QuickSCAT file available '+str(yr)+' '+str(m)+' '+str(d)+' jd: '+str(yr)+str("{:03d}".format(int(jd+0.5))))
                    qscat[:,i,j] = np.nan
                    continue
                
        qscat[:,i,j] = image.flatten()
 
    
x = np.arange(0,306,1)
y = np.arange(0,306,1)
x,y = np.meshgrid(x,y)

alon1,alat1 = pix2latlon(x, y, head)
alon = alon1[0,:,:].flatten()
alat = alat1[0,:,:].flatten()
coord_qscat = np.transpose(np.array([alat, alon]))


#%%
'''Regrid to CS2'''

yr = 2011

cs = xr.open_dataset('C:/Users/zq19140/OneDrive - University of Bristol/Documents/Jack_onedrive/Existing Sea Ice Thickness Datasets/CryoSat-2 Sea Ice Thickness Grids/ubristol_cryosat2_seaicethickness_nh25km_'+str(yr)+'_'+str("{:02d}".format(m))+'_v1.nc')
cs = cs.where(cs.x!=cs.x[-1], drop=True); cs = cs.where(cs.y!=cs.y[-1], drop=True)
cs = cs.coarsen(x=2).mean().coarsen(y=2).mean()

coord = np.transpose(np.array([cs.Latitude.values.flatten(), cs.Longitude.values.flatten()]))
MdlKDT = spatial.KDTree(coord_oscat)
dist, closest = MdlKDT.query(coord, k=1, distance_upper_bound=0.5)
closest[closest==len(coord_oscat)] = 0
oscat_cs = oscat[closest,:,:]
oscat_cs[oscat_cs<-32] = np.nan

MdlKDT = spatial.KDTree(coord_scatsat)
dist, closest = MdlKDT.query(coord, k=1, distance_upper_bound=0.5)
closest[closest==len(coord_scatsat)] = 0
scatsat_cs = scatsat[closest,:,:]*0.001-50
scatsat_cs[scatsat_cs>0] = np.nan

MdlKDT = spatial.KDTree(coord_qscat)
dist, closest = MdlKDT.query(coord, k=1, distance_upper_bound=0.5)
closest[closest==len(coord_qscat)] = 0
qscat_cs = qscat[closest,:,:]
qscat_cs[qscat_cs<-32] = np.nan

#%%
years = np.arange(1992,2021,1)

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
                    elif np.isin(years[j],years_qs):
                        idx = np.where(years[j]==years_qs)[0]
                        scat_x_yr[i,0] = qscat_cs[loc[0],idx,k]
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


dataset_X.to_csv('../../data/interim/datasetsML/no_CS2/'+location+'/incl_Ku/dataset_x_1992-2019_'+str("{:02d}".format(m))+'.csv',
                    index=False)
dataset_y.to_csv('../../data/interim/datasetsML/no_CS2/'+location+'/incl_Ku/dataset_y_1992-2019_'+str("{:02d}".format(m))+'.csv',
                    index=False)







