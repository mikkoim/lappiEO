import processing

def table_from_dict(d):
    table = []
    for k in d.keys():
        table.append(k)
        table.append(k)
        table.append(d[k])
        
    return table
    
m = {2: 14, 12: 6, 18: 28, 21: 4, 23: 14, 24: 6, 28: 6, 32: 14, 33: 26, 40: 6, 42: 6, 44: 26, 45: 4, 47: 14, 50: 14, 58: 14, 59: 28, 61: 4, 63: 14, 64: 26, 67: 21}
input = 'D:\mikko\ylalappi\tulkinnat\30km_buf\70c_3x3_set2.tif'
table = table_from_dict(m)

d = { 'DATA_TYPE' : 5, 
'INPUT_RASTER' : input, 
'NODATA_FOR_MISSING' : False, 
'NO_DATA' : -9999, 
'OUTPUT' : 'TEMPORARY_OUTPUT', 
'RANGE_BOUNDARIES' : 2, 
'RASTER_BAND' : 1, 
'TABLE' : table }

processing.runAndLoadResults("qgis:reclassifybytable", d)