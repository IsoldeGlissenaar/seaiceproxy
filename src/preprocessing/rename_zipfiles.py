# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:34:17 2022

@author: zq19140
"""

import os
import shutil

# % Download zipped CIS chart .shp files from:
# % https://iceweb1.cis.ec.gc.ca/Archive/page1.xhtml?lang=en


# %%% Main working directory %%%
region = 'EasternArctic'
region_short = 'EA'

folder = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Canadian_ice_service_charts/'+region+'/'

# Extract CIS .shp files to:
tempfolder = folder+'temp/'

#%%
# % Remove temporary directories and move shapefiles to working directory
for filename in os.listdir(tempfolder):
    yr = filename[4:8]
    m = filename[2:4]
    d = filename[0:2]
    extension = os.path.splitext(filename)[1]
    new_name = tempfolder+region_short+'_'+yr+m+d+'_polygon'+extension

    # Renaming the file
    os.rename(tempfolder+filename, new_name)
    shutil.move(os.path.join(tempfolder, new_name), folder+'Ice Chart Shapefiles/')

# Remove tempfolder

