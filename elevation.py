import gdal
import numpy as np
import glob
def elevation(lat, lon):
    lats = range( int(lat.min()), int(lat.max())+1)
    lons = range( int(lon.min()), int(lon.max())+1)
    eles=np.zeros_like(lat)
    for lat0 in lats:
        for lon0 in lons:
            if lat0>=0:
                lat_name = 'N%02d'%int(lat0)
            elif lat0<0:
                lat_name = 'S%02d'%int(-lat0)
            else:
                'Wrong lat given, and float is expected!'
            if lon0>=0:
                lon_name = 'E%03d'%int(lon0)
            elif lon0<0:
                lon_name = 'W%03d'%int(-lon0)
            else:
                'Wrong lon given, and float is expected!'
            fname = 'SRTM/'+lat_name+lon_name+ '.hgt'
            if glob.glob(fname)==[]:
                print 'No such file contain elevation data!!!\
                Please download %s from USGS from link %s' %(lat_name+lon_name+ '.hgt', 'http://dds.cr.usgs.gov/srtm/version1/') 
            else:    
                mask = (lat>lat0)&(lat<lat0+1)&(lon>lon0)&(lon<lon0+1)
                g = gdal.Open(fname)
                geo = g.GetGeoTransform()
                l_lon, lon_size,l_lat, lat_size, = geo[0], geo[1],geo[3],geo[5]
                x, y = ((lon-l_lon)/lon_size).astype(int), ((lat-l_lat)/lat_size).astype(int)
                ele = g.ReadAsArray()
                eles[mask] = ele[x[mask],y[mask]]    
    return eles