import os
import sys
sys.path.insert(0,'python')
import numpy.ma as ma
from collections import Counter
import cPickle as pkl
from scipy import optimize
from functools import partial
import scipy.ndimage as ndimage
import numpy as np
import scipy
from scipy import signal
import scipy.stats
from cloud import *
from fastRWpkl import *
import numpy 
from numpy import clip, where
from scipy.ndimage.morphology import *
import xml.etree.cElementTree as ET
import multiprocessing
from get_brdf import *
from sklearn import linear_model
from scipy import ndimage
from L_geo import *
import glob
import cPickle as pkl
from get_wrs import *


directory = os.getcwd()+'/'

keys = 'B02', 'B03','B04','B08','B8A','B11','B12'
bands = [2,3,4,8,13,11,12]

L_bands = [2,3,4,5,6,7]

def gaussian(xwin, ywin, xstd, ystd, angle, norm = True):
    win = max(xwin, ywin)
    winx = win*2**0.5
    winy = win*2**0.5
        
    xgaus = signal.gaussian(winx, xstd)
    ygaus = signal.gaussian(winy, ystd)
    gaus  = np.outer(xgaus, ygaus)
    r_gaus = scipy.ndimage.interpolation.rotate(gaus, angle, reshape=True)
    center = np.array(r_gaus.shape)/2
    cgaus = r_gaus[center[0]-xwin/2: center[0]+xwin/2, center[1]-ywin/2:center[1]+ywin/2]
    if norm:
        return cgaus/cgaus.sum()
    else:
        return cgaus

def cost(G_paras, Ker_size, H_data, H_inds, L_data, L_inds, B_num, val_mask):    
    to_regression =[]  
    xstd,ystd,angle, xs, ys = G_paras
    xwin,ywin = Ker_size
    Lx, Ly = L_inds
    Hx, Hy = H_inds
    xlimit, ylimit = H_data.shape
    #new centers after shifts
    s_Hx, s_Hy = Hx+xs, Hy+ys
    
    in_x = (s_Hx>xwin/2)&(s_Hx<(xlimit-xwin/2))
    in_y = (s_Hy>ywin/2)&(s_Hy<(ylimit-ywin/2))
    val = in_x&in_y
    
    G = gaussian(xwin,ywin,xstd,ystd,angle,False)                              
    ker = G/(G.sum())
    s = signal.fftconvolve(H_data, ker, mode='same')
   
   
    # remove the cloud pixel
    s[~val_mask] = np.nan
   
    Hvals = s[s_Hx.astype(int)[val], s_Hy.astype(int)[val]]
    Lvals = L_data[Lx.astype(int)[val], Ly.astype(int)[val]]*0.001
    mask = (Lvals>0)&(Lvals<1)&(Hvals>0)&(Hvals<1)&(~Lvals.mask)
    
    if sum(mask) ==0:
        print 'Too much cloud again to affect the convolve results'
        return 10000
    else:
        
        dif = Hvals[mask] - Lvals[mask]
        
        inliers = (dif>(np.nanmean(dif)-3*np.nanstd(dif)))&(dif<(np.nanmean(dif)+3*np.nanstd(dif)))
       
        H_in = Hvals[mask][inliers]#y.ravel()
        L_in = Lvals[mask][inliers]#x.ravel()
       
        r = scipy.stats.linregress(H_in, L_in)    
        costs = abs(1-r.rvalue)

        print 'band: ',B_num,'\n','costs:', costs, 'rvalue: ', r.rvalue, 'slop: ', r.slope,'inter',r.intercept, '\n', 'parameters: ', G_paras,'\n'
        if (r.intercept<0) or (r.slope>1):
            costs = costs*1000000000000000.
        return costs

def ransaclin(x,y):
    y, x = y.reshape((len(y),1)), x.reshape((len(x),1))
    model = linear_model.LinearRegression()
    model.fit(x, y)

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),max_trials=10000000)
    model_ransac.fit(x, y)
    inlier_mask = model_ransac.inlier_mask_
    return x[inlier_mask], y[inlier_mask]

def ScaleExtent(data, shape): # used for unifine different array,

    re = int(shape[0]/(data.shape[0]))

    a = np.repeat(np.repeat(data, re, axis = 1), re, axis =0)
    
    if (re*(data.shape[0])-shape[0]) != 0:
        extended = np.zeros(shape)
        extended[:re*(data.shape[0]),:re*(data.shape[0])] = a
        extended[re*(data.shape[0]):,re*(data.shape[0]):] = a[re*(data.shape[0])-shape[0]:, re*(data.shape[0])-shape[0]]
        return extended
    else:
        return a

def get_psf( H_data, H_inds, L_data, L_inds, B_num, val_mask, Ker_size = (120,120)):
    
    p = np.array([30, 400, 7.91598096945, 20, 20])
    psolve = optimize.fmin(cost,p,full_output=1, args=(Ker_size, H_data, H_inds, L_data, L_inds, B_num, val_mask))
    print 'solved b%02d: '%B_num, psolve
    return [B_num,psolve]

def cir_struc(dia=10):
    a = b = r = dia
    n = 2*dia+1

    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = (x*x + y*y) <= r*r

    array = np.zeros((n, n))
    array[mask] = 1
    
    return array

def op(ind,  args=None ):
    fpath, sentm, brdfs, sinds, minds = args
    Sent = gdal_read(bands[ind], fpath)[keys[ind]]
    sent = ScaleExtent(Sent, (10980,10980)) 
    sent[sentm]= np.nanmean(sent[~sentm])
    sent[np.isnan(sent)] = np.nanmean(sent[~sentm])
    
    struct = ndimage.generate_binary_structure(2, 2)
    dia_cloud = ndimage.binary_dilation(sentm, structure=struct, iterations=60).astype(sentm.dtype)
    
    # remove border effects
    mask = np.ones((10980,10980)).astype('bool')
    struct1 = ndimage.generate_binary_structure(2, 2)
    small_mask = ndimage.binary_erosion(mask, structure=struct1, iterations=60)
    
    valid = (~dia_cloud)&small_mask
    
    if ind<4:
        brdfs[ind][brdfs[ind].mask] = np.nan               
        psolve = get_psf(sent,sinds, brdfs[ind]*0.001, minds, bands[ind], valid)

    else:
        brdfs[ind-1][brdfs[ind-1].mask] = np.nan
        psolve = get_psf(sent,sinds, brdfs[ind-1]*0.001, minds, bands[ind], valid)
    return psolve


def L_op(ind, args=None):
    
    Hfile, cloud, brdfs, H_inds, L_inds = args
    B_num = L_bands[ind]
    H_data = gdal.Open(Hfile+'band%d.tif'%B_num).ReadAsArray()*0.0001
    struct = ndimage.generate_binary_structure(2, 2)
    dia_cloud = ndimage.binary_dilation(cloud, structure=struct, iterations=20).astype(cloud.dtype)
    mask = ~(H_data<0).astype('bool')
    small_mask = ndimage.binary_erosion(mask, structure=struct, iterations=20).astype(mask.dtype)
    val_mask = (~dia_cloud)&small_mask
    L_data = brdfs[ind]
    
    psolve = get_psf(H_data, H_inds, L_data, L_inds, B_num, val_mask, Ker_size = (40,40))
    
    return psolve
        
def optimizing(lat, lon, Hdoy, Ldoy, year, Hsat):
    
    if Hsat == 'S':
        
        m = mgrs.MGRS()
        mg_coor = m.toMGRS(lat, lon, MGRSPrecision=4)
        h,v = mtile_cal(lat, lon)
        
        sd = datetime.datetime(year, 1, 1) + datetime.timedelta(Hdoy - 1)
        y,m,d = sd.year, sd.month, sd.day
        
        Hfile = directory + 's_data'+'/%s/%s/%s/%s/%s/%s/0/'%(mg_coor[:2], mg_coor[2], mg_coor[3:5], y,m,d)
        
        Lfile = glob.glob('m_data/MCD43A?.A%d%03d.h%02dv%02d.005.*.hdf'%(y,Ldoy,h,v))
        
        sentm = get_cloud_mask(Hfile)
        doy = '%02d/%02d/%02d'%(int(Hfile.split('/')[-3]), int(Hfile.split('/')[-4]), int(Hfile.split('/')[-5]))
        pos = Hfile.split('/')[-8]+Hfile.split('/')[-7]+Hfile.split('/')[-6]
        if sentm.sum()/(10980.*10980.) <0.15:
            print 'DOY: ', doy,'\n', 'Location: ', pos, 
            print 'Cloud proportion: ', sentm.sum()/(10980.*10980.)
            minds, sinds = get_coords(lat,lon) 

            modis_filenames = gdal.Open(Lfile[0]).GetSubDatasets()
            modisQAs = gdal.Open(Lfile[1]).GetSubDatasets()
            
            mete = readxml('%smetadata.xml'%Hfile)
            sza = np.zeros(7)
            sza[:] = mete['mSz']
            vza = (mete['mVz'])[[1,2,3,7,8,11,12],]
            raa = (mete['mSa']-mete['mVa'])[[1,2,3,7,8,11,12],]

            brdfs, rws = get_rs(modisQAs, modis_filenames, (sza,vza, raa), bands=[2,3,0,1,1,5,6])
            args = Hfile, sentm, brdfs, sinds, minds
            par = partial(op, args=args)
            pool = multiprocessing.Pool(processes = 7)
            retval = pool.map(par, range(7))
            pool.close()
            pool.join()
            parallel_rw_pkl(retval, '%s%spsfs'%(pos, doy), 'w')
            if ret:
                parallel_rw_pkl([m,s], '%s%s%to_regs'%(pos, doy), 'w')

        else:
            print 'Too much cloud, and this tile (doy: %s, lat: %s, lon: %s) is considered as invalid.'%(doy, lat, lon)
    
    elif Hsat == 'L':
        
        
        h,v = mtile_cal(lat, lon)
        pr=get_wrs(lat, lon)
        path, row = pr[0]['path'],pr[0]['row']
        Hfile = directory +'l_data/LC8%03d%03d%d%03dLGN00_sr_'%(path, row, year, Hdoy)
        Lfile = glob.glob('m_data/MCD43A?.A%d%03d.h%02dv%02d.005.*.hdf'%(year,Ldoy,h,v))
        with open(Hfile[:-4]+'_MTL.txt', 'r') as inF:
            for line in inF:
                if 'CLOUD_COVER ' in line:
                    cloud_cover =  float(line.split('= ')[1])
        if cloud_cover<20:
            print 'Less than 20% cloud.'
            b1 = gdal.Open(Hfile+'band1.tif').ReadAsArray()
            corners = b1.shape

            cloud = gdal.Open(Hfile[:-4]+'_cfmask.tif').ReadAsArray()
            cl_mask = (cloud>1)&(cloud<255)

            dic = {}

            with open(Hfile[:-4]+'_MTL.txt', 'r') as inF:
                for line in inF:
                    if ('CORNER_' in line)&('LAT_PRODUCT' in line):
                        dic[line.split(' = ')[0].strip()[7:13]] = float(line.split(' = ')[1])
                    elif ('CORNER_' in line)&('LON_PRODUCT' in line):
                        dic[line.split(' = ')[0].strip()[7:13]] = float(line.split(' = ')[1])
                    elif 'ROLL_ANGLE' in line:
                        vza = float(line.split(' = ')[1])
                    elif 'SUN_AZIMUTH' in line:
                        saa = float(line.split(' = ')[1])
                    elif 'SUN_ELEVATION' in line:
                        sza = float(line.split(' = ')[1])
            with open('Landsat_azm.pkl', 'r') as savefile:
                Landsat_azm = pkl.load(savefile)

            lazm = np.nanmean(Landsat_azm[(Landsat_azm[:,2]==path)&(Landsat_azm[:,3]==row)].squeeze()[:2])
            raa = lazm-saa

            L_inds, H_inds = ML_geo_trans(lat, lon, dic, corners)

            modis_filenames = gdal.Open(Lfile[0]).GetSubDatasets()
            modisQAs = gdal.Open(Lfile[1]).GetSubDatasets()

            tems = np.zeros((3,6))
            tems[0,:] = sza
            tems[1,:] = vza
            tems[2,:] = raa

            brdfs, rws = get_rs(modisQAs, modis_filenames, (tems[0], tems[1], tems[2]), bands=[2,3,0,1,5,6])
            brdfs.mask = brdfs.mask|(rws<1)
            args = Hfile, cl_mask, brdfs, H_inds, L_inds
            
            par = partial(L_op, args=args)
            pool = multiprocessing.Pool(processes = 6)
            retval = pool.map(par, range(6))
            pool.close()
            pool.join()
            parallel_rw_pkl(retval, 'P%sR%s%spsfs'%(pat, row, doy), 'w')
                  
        else:
            print 'Too much cloud: ',cloud_cover
    
    else:
        
        print "Please announce satellite 'L' or 'S'..."
        
        

