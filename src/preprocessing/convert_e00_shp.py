# Batch import from .e00 and convert to shapefiles
# Long filenames not supported by arc: create temporary in & out directories on C drive

import arcpy
from arcpy import env
import os

env.workspace = "C:\in"
fin = "C:\in"
fout = "C:\out"

fcs = arcpy.ListFiles()

for fc in fcs:

    arcpy.ImportFromE00_conversion(fc, fin, fc[0:11])
    arcpy.env.workspace = fc[0:11]
    arcpy.FeatureClassToShapefile_conversion("polygon",fout)
    arcpy.env.workspace = "C:\in"

