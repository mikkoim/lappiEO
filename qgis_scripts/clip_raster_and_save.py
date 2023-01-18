import os
from qgis import processing
from qgis.PyQt.QtCore import QVariant
from qgis.core import QgsVectorLayer

OUTFOLDER = '/run/nvme/job_5369942/tmp/v3_segtrain_35k/'
RASTER_NAME = 'lahtodata_v3'
MASK_NAME = 'segtrain_35k'

def vlayer_from_feature(fet0, layername, crs):
    fieldlist = [f for f in fet0.fields()]
    attributelist = [a for a in fet0.attributes()]
    # Create layer
    v = QgsVectorLayer("Polygon", layername, "memory")
    v.setCrs(crs)
    pr = v.dataProvider()
    pr.addAttributes(fieldlist)
    v.updateFields() 
    # Add feature
    f = QgsFeature()
    f.setGeometry(fet0.geometry())
    f.setAttributes(attributelist)
    pr.addFeature(f)
    v.updateExtents() 
    return v

crs = QgsProject.instance().crs()
#processing.algorithmHelp("gdal:cliprasterbymasklayer")

#rlayer = QgsProject.instance().mapLayersByName('S1_crop')[0]
vlayer = QgsProject.instance().mapLayersByName(MASK_NAME)[0]
N = sum(1 for _ in vlayer.getFeatures())
for i, feature in enumerate(vlayer.getFeatures()):
    layername = str(int(feature['fid']))
    outname = OUTFOLDER + layername + '.tif'
    if os.path.exists(outname):
        continue
        
    v = vlayer_from_feature(feature, layername, crs)
    #QgsProject.instance().addMapLayer(v)

    myResult = processing.run("gdal:cliprasterbymasklayer", 
                {'INPUT': RASTER_NAME,
                 'MASK': v,
                 'SOURCE_CRS': 'EPSG:3067',
                 'TARGET_CRS': 'EPSG:3067',
                 'OUTPUT': outname})
                 
    print(i+1,'/',N,myResult['OUTPUT'])