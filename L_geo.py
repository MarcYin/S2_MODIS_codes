import sys
sys.path.insert(0,'python')
from geo_trans import *
         

def geo_inter(coords, dic, corners):
    
    '''
    a matrix way to convert a lat, lon to the coordinates of an image array, 
    based on bilinear interpolation on matrix form. (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    --------------------------------------------------------------------------------------------------------------
    coords is the coordinates [(lat, lon), (lat, lon)...] needed to transfer
    
    dic is a dictionary of the Upper lfet (UL), UR, LL, LR 's lat and lons
    
    example: dic ={'LL_LAT': 36.35288,
                   'LL_LON': 113.00651,
                   'LR_LAT': 36.41186,
                   'LR_LON': 115.6326,
                   'UL_LAT': 38.51077,
                   'UL_LON': 112.88999,
                   'UR_LAT': 38.57451,
                   'UR_LON': 115.59258}
    
    corners is the (x, y) corresponding to the shape of the area (array)            
    '''
    
    a = np.matrix([[1, dic['LL_LAT'], dic['LL_LON'], dic['LL_LAT']*dic['LL_LON']],
                   [1, dic['UL_LAT'], dic['UL_LON'], dic['UL_LAT']*dic['UL_LON']], 
                   [1, dic['LR_LAT'], dic['LR_LON'], dic['LR_LAT']*dic['LR_LON']],
                   [1, dic['UR_LAT'], dic['UR_LON'], dic['UR_LAT']*dic['UR_LON']]])
    x0, y0 = corners
    coords = np.array(coords)
    convs = np.ones((4,len(coords)))
    convs[1] = coords[:,0]
    convs[2] = coords[:,1] 
    convs[3] = (coords[:,0]* coords[:,1])
    convs = np.matrix(convs)
    
    x = np.matrix([x0,0,x0,0])*((a**-1).T)*convs
    y = np.matrix([0,0,y0,y0])*((a**-1).T)*convs
    return np.array([np.squeeze(np.array(np.round(x).astype('int'))), np.squeeze(np.array((np.round(y).astype('int'))))])

def MSL_geo_trans(lat, lon, dic, corners):
    h, v = mtile_cal(lat, lon)
    wgs_3d = np.array(get_wgs(h,v))
    wgs = wgs_3d[:,:2][:,::-1]
    cors = geo_inter(wgs, dic, corners)
    hinds =np.array([cors[0][(cors[0]>=0)&(cors[0]<corners[0])&(cors[1]>=0)&(cors[1]<corners[1])],
                     cors[1][(cors[0]>=0)&(cors[0]<corners[0])&(cors[1]>=0)&(cors[1]<corners[1])]])
    minds = np.where(((cors[0]>=0)&(cors[0]<corners[0])&(cors[1]>=0)&(cors[1]<corners[1])).reshape((2400,2400)))
   
    return minds, hinds 


def cor_inter(cords,dic, corners):
    '''
    From landsat to lattitude, longtitude 
    
    cords is landsat coordinates
    '''
    
    a = np.matrix([[1, corners[0], 0,0],
                   [1, 0, 0,0],
                   [1, corners[0], corners[1],corners[0]*corners[1]],
                   [1, 0, corners[1],0],])
    cords = np.array(cords)
    convs = np.ones((4,len(cords[0])))
    convs[1] = cords[0]
    convs[2] = cords[1] 
    convs[3] = (cords[0]* cords[1])
    convs = np.matrix(convs)
    
    x = np.matrix([dic['LL_LAT'],dic['UL_LAT'],dic['LR_LAT'],dic['UR_LAT']])*((a**-1).T)*convs
    y = np.matrix([dic['LL_LON'],dic['UL_LON'],dic['LR_LON'],dic['UR_LON']])*((a**-1).T)*convs
    return np.array([np.squeeze(np.array(x)), np.squeeze(np.array(y))])