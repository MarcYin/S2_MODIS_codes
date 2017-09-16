from netCDF4 import Dataset
import numpy as np
from netcdftime import utime
import datetime
import glob

def read_net(year, month, day, lats, lons, dataset=None):
    '''
    datasets should be wv(water vapur), aot (aerosol), tco (Ozone)
    '''
    if dataset =='wv':
        fname = glob.glob('ECMWF/Total_column_water_vapour_%d*.nc'%year)[0]
    elif dataset == 'aot':
        fname = glob.glob('ECMWF/Total_Aerosol_Optical_Depth_at_550nm_%d*.nc'%year)[0]
    elif dataset == 'tco':
        fname = glob.glob('ECMWF/GEMS_Total_column_ozone_%d*.nc'%year)[0]
    else:
        print 'Wrong dataset is given, please see the doc string!'
    net = Dataset(fname, 'r')
    lon, lat, time, val = net.variables.keys()
    lat_ind, lon_ind = ((lats-net[lat][0])/(-0.125)).astype(int),((lons-net[lon][0])/0.125).astype(int)
    
    cdf_time = utime(net[time].units)
    date = cdf_time.num2date(net[time][:])
    time_ind = np.where(date==datetime.datetime(year, month, day, 3,0))[0][0]
    
    return net[val][time_ind, lat_ind, lon_ind]