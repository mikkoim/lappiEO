import os
from qgis import processing
from qgis.PyQt.QtCore import QVariant
from qgis.core import QgsVectorLayer


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