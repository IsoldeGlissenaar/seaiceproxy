# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:48:43 2021

@author: Jack Landy & Isolde Glissenaar

# Name: process_e00_to_shapefile.py
# Description: Script that processes the Canadian Ice Service ice charts from their e00 format
#              into shapefiles. The e00 format was used for ice charts until 8 January 2020.
# Input requirements: Zipped CIS chart .e00 files from the Canadian Ice Service
#                     Script requires ArcGIS to run the convert_e00_shp.py script.
# Output: Ice chart in shapefile format
"""

import os
import glob
import shutil

# % (c) J.C. Landy, University of Bristol, 2019
# % (Requires ArcGIS and Python for file conversion)

# % Refer to following for stage of development and floe size codes:
# % https://www.ec.gc.ca/glaces-ice/?lang=En&n=503E8E74-1

# % Download zipped CIS chart .e00 files from:
# % https://iceweb1.cis.ec.gc.ca/Archive/page1.xhtml?lang=en


# %%% Main working directory %%%
region = 'WesternArctic'
region_short = 'WA'

folder = 'C:/Users/zq19140/OneDrive - University of Bristol/Documents/SatelliteData/Canadian_ice_service_charts/'+region+'/'

# arcpy does not support long file paths, so create low level temporary
# directories
os.mkdir('C:/in/')
os.mkdir('C:/out/')

# Extract CIS .e00 files to:
tempfolder = 'C:/in/';

# %% Rename .e00 files from CIS archive

datfiles = glob.glob(tempfolder+'*.e00')
for i in range(len(datfiles)):
    outname = region_short+datfiles[i][13:22]+'.e00'
    os.rename(tempfolder+datfiles[i][6:], tempfolder+outname)

# %% Convert .e00 files to shapefiles externally with arcpy

# % EXTERNAL %
# % open arcmap
# % open python window
# % load & run 'convert_e00_shp.py'
# % the 'out' directory should contain a set of converted .shp files
# % close arcmap

#%%
# % Remove temporary directories and move shapefiles to working directory
for filename in os.listdir('C:/out/'):
    shutil.move(os.path.join('C:/out/', filename), folder+'Ice Chart Shapefiles/')

# Remove C:/in/ and C:/out/




