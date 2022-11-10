# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:57:04 2022

@author: Isolde Glissenaar

# Date: 25/02/2022
# Name: add_scatC_1992-2020.py
# Description: Add scatterometer data to 1992-2020 dataset
# Input requirements: dataset_X and dataset_y 1992-2020, ERS-1, ERS-2 , and ASCAT 
#                     scatterometer data, CryoSat-2 SIT grid
# Output: dataset_X and dataset_y including C band scatterometer data
"""

import sys
sys.path.append('../functions/')
from sir_io import loadsir
from sir_geom import pix2latlon

from scipy import spatial
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
#ERS-1 (April 1992-April 1996)

if m < 4:
    years_e1 = np.array([1993,1995,1996])
elif m==4:    
    years_e1 = np.array([1992,1993,1994,1995,1996])
elif m > 9:
    years_e1 = np.array([1992,1993,1994,1995])
days = np.arange(1,32,1)
ers1 = np.zeros((23716,len(years_e1),31))

for i in range(len(years_e1)):
    for j in range(len(days)):
        yr = years_e1[i]
        
        esfile = xr.open_dataset('C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/ERS1/monthly/ers1_monthly_'+str(yr)+'_'+str("{:02d}".format(m))+'.nc')
        ers1[:,i,j] = esfile.image.values.flatten()
 
    
alat = esfile.lat.values.flatten()
alon = esfile.lon.values.flatten()

coord_ers1 = np.transpose(np.array([alat, alon]))

#%%
#ERS-2 (October 1996 - December 2000)

if m < 5:
    years_e2 = np.array([1997,1998,1999,2000])
elif m > 10:
    years_e2 = np.array([1996,1997,1998,1999,2000])
days = np.arange(1,32,1)
ers2 = np.zeros((23716,len(years_e2),31), dtype=int)

for i in range(len(years_e2)):
    for j in range(len(days)):
        yr = years_e2[i]
        d = days[j]
        
        jd = date_to_jd(-4712, m, d)

        try:
            f = 'ers2-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd+0.5)))+'-'+str("{:03d}".format(int(jd+5.5)))+'.grd'
            sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/ERS2/grd/'+f
            image, head, descrip, iaopt = loadsir(sir_fname)
        except FileNotFoundError:
            try:
                f = 'ers2-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd-0.5)))+'-'+str("{:03d}".format(int(jd+4.5)))+'.grd'
                sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/ERS2/grd/'+f
                image, head, descrip, iaopt = loadsir(sir_fname)
            except FileNotFoundError:
                try:
                    f = 'ers2-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd-1.5)))+'-'+str("{:03d}".format(int(jd+3.5)))+'.grd'
                    sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Scatterometer/ERS2/grd/'+f
                    image, head, descrip, iaopt = loadsir(sir_fname)                
                except (FileNotFoundError, IndexError):
                    print('FileNotFoundError: no ERS2 file available '+str(yr)+' '+str(m)+' '+str(d)+' jd: '+str(yr)+str("{:03d}".format(int(jd+0.5))))
                    ers2[:,i,j] = np.nan
                    continue
                
        ers2[:,i,j] = image.flatten()
 
    
x = np.arange(0,154,1)
y = np.arange(0,154,1)
x,y = np.meshgrid(x,y)

alon1,alat1 = pix2latlon(x, y, head)
alon = alon1[0,:,:].flatten()
alat = alat1[0,:,:].flatten()
coord_ers2 = np.transpose(np.array([alat, alon]))

#%%
# ASCAT (Jan 2007 - December 2020)

years_as = np.arange(2007,2021,1)
days = np.arange(1,32,1)
ascat = np.zeros((23409,len(years_as),31))

for i in range(len(years_as)):
    for j in range(len(days)):
        yr = years_as[i]
        d = days[j]
        
        jd = date_to_jd(-4712, m, d)

        try:
            f = 'msfa-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd+0.5)))+'-'+str("{:03d}".format(int(jd+1.5)))+'.grd'
            sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/ASCAT/grd/'+f
            image, __, descrip, iaopt = loadsir(sir_fname)
        except FileNotFoundError:
            try:
                f = 'msfa-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd-0.5)))+'-'+str("{:03d}".format(int(jd+0.5)))+'.grd'
                sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/ASCAT/grd/'+f
                image, __, descrip, iaopt = loadsir(sir_fname)
            except FileNotFoundError:
                try:
                    f = 'msfa-a-Arc'+str(yr)[2:]+'-'+str("{:03d}".format(int(jd+1.5)))+'-'+str("{:03d}".format(int(jd+2.5)))+'.grd'
                    sir_fname = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/ASCAT/grd/'+f
                    image, __, descrip, iaopt = loadsir(sir_fname)                
                except (FileNotFoundError, IndexError):
                    print('FileNotFoundError: no ASCAT file available '+str(yr)+' '+str(m)+' '+str(d)+' jd: '+str(yr)+str("{:03d}".format(int(jd+0.5))))
                    ascat[:,i,j] = np.nan
                    continue
                
        ascat[:,i,j] = image.flatten()
 
    
x = np.arange(0,153,1)
y = np.arange(0,153,1)
x,y = np.meshgrid(x,y)

alon1,alat1 = pix2latlon(x, y, head)
alon = alon1[0,:,:].flatten()
alat = alat1[0,:,:].flatten()
coord_ascat = np.transpose(np.array([alat, alon]))


#%%
'''Regrid to CS2'''

yr = 2011

cs = xr.open_dataset('C:/Users/zq19140/OneDrive - University of Bristol/Documents/Jack_onedrive/Existing Sea Ice Thickness Datasets/CryoSat-2 Sea Ice Thickness Grids/ubristol_cryosat2_seaicethickness_nh25km_'+str(yr)+'_'+str("{:02d}".format(m))+'_v1.nc')
cs = cs.where(cs.x!=cs.x[-1], drop=True); cs = cs.where(cs.y!=cs.y[-1], drop=True)
cs = cs.coarsen(x=2).mean().coarsen(y=2).mean()

coord = np.transpose(np.array([cs.Latitude.values.flatten(), cs.Longitude.values.flatten()])) 
MdlKDT = spatial.KDTree(coord_ers1)
dist, closest = MdlKDT.query(coord, k=1, distance_upper_bound=2)
closest[closest==len(coord_ers1)] = 0
ers1_cs = ers1[closest,:,:]
ers1_cs[ers1_cs==-33] = np.nan

MdlKDT = spatial.KDTree(coord_ers2)
dist, closest = MdlKDT.query(coord, k=1, distance_upper_bound=2)
closest[closest==len(coord_ers2)] = 0
ers2_cs = (ers2[closest,:,:]).astype(float)
ers2_cs[ers2_cs==-33] = np.nan

MdlKDT = spatial.KDTree(coord_ascat)
dist, closest = MdlKDT.query(coord, k=1, distance_upper_bound=2)
closest[closest==len(coord_ascat)] = 0
ascat_cs = ascat[closest,:,:]
ascat_cs[ascat_cs==-33] = np.nan

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
                    if np.isin(years[j],years_e1):
                        idx = np.where(years[j]==years_e1)[0]
                        scat_x_yr[i,0] = ers1_cs[loc[0],idx,k]
                    elif np.isin(years[j],years_e2):
                        idx = np.where(years[j]==years_e2)[0]
                        scat_x_yr[i,0] = ers2_cs[loc[0],idx,k]
                    elif np.isin(years[j],years_as):
                        idx = np.where(years[j]==years_as)[0]
                        scat_x_yr[i,0] = ascat_cs[loc[0],idx,k]
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


dataset_X.to_csv('../../data/interim/datasetsML/no_CS2/'+location+'/incl_C/dataset_x_1992-2019_'+str("{:02d}".format(m))+'.csv',
                    index=False)
dataset_y.to_csv('../../data/interim/datasetsML/no_CS2/'+location+'/incl_C/dataset_y_1992-2019_'+str("{:02d}".format(m))+'.csv',
                    index=False)







