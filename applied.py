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

keys = 'B02', 'B03','B04','B08','B8A','B11','B12'
bands = [2,3,4,8,13,11,12]

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

def _psf(sent, sinds, mod, minds,band, psf, dia_cloud, rw):    
    xstd,ystd,angle, xs, ys = psf
    xwin,ywin = 120, 120
    
    to_regression =[]          
    cx = sinds[0]
    cy = sinds[1]
    mx = minds[0]
    my = minds[1]
    
    gaus = gaussian(xwin,ywin,xstd,ystd,angle,False)                              
    ker = gaus/(gaus.sum())

    s = signal.fftconvolve(sent, ker, mode='same')
    #new centers after shifts
    n_cx = cx+xs; n_cy = cy+ys
    # to remove the pixels outside of the borders
    in_x = (n_cx>xwin/2)&(n_cx<(10000-xwin/2))
    in_y = (n_cy>ywin/2)&(n_cy<(10000-ywin/2))
    # to remove the cloud pixel
    #c_x, c_y = np.where(dia_cloud)
    #cf_x = ~np.in1d(n_cx, c_x); cf_y = ~np.in1d(n_cy, c_y)
    vld = in_x&in_y

    indx,indy = np.round((n_cx)[vld]).astype(int), np.round((n_cy)[vld]).astype(int)
    # to remove the cloud pixel
    s[dia_cloud] = np.nan
    vals = s[indx,indy]
    brdf = mod[mx[vld], my[vld]]
    p_rw = rw[mx[vld], my[vld]]
    mask = (brdf>0)&(brdf<1)&(vals>0)&(vals<1)&(~brdf.mask)
    print 'Valid values proportion:', 1.*mask.sum()/mask.size
    if sum(mask) ==0:
        print 'Too much cloud again to affect the convolve results'
        return 10000
    else:
        dif = vals[mask] - brdf[mask]
        inliers = (dif>(np.nanmean(dif)-3*np.nanstd(dif)))&(dif<(np.nanmean(dif)+3*np.nanstd(dif)))
        
        #global vals; global mask; global brdf; global inliers
        #x,y = ransaclin(vals[mask][inliers], brdf[mask][inliers])

        vx = indx[mask][inliers]
        vy = indy[mask][inliers]
        sents = vals[mask][inliers]#y.ravel()
        modiss = brdf[mask][inliers]#x.ravel()
        co_rw = p_rw[mask][inliers]

    
    return [band,vx,vy, sents,modiss, co_rw]

def ransaclin(x,y):
    y, x = y.reshape((len(y),1)), x.reshape((len(x),1))
    model = linear_model.LinearRegression()
    model.fit(x, y)

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
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

def op(ind,  args=None ):
    fpath, sentm, brdfs, sinds, minds, psfs, rws = args
    Sent = gdal_read(bands[ind], fpath)[keys[ind]]
    sent = ScaleExtent(Sent, (10980,10980)) 
    sent[sentm]= np.nanmean(sent[~sentm])
    sent[np.isnan(sent)] = np.nanmean(sent[~sentm])
    struct1 = ndimage.generate_binary_structure(2, 2)
    dia_cloud = ndimage.binary_dilation(sentm, structure=struct1, iterations=5).astype(sentm.dtype)
    if ind<4:
        brdfs[ind][brdfs[ind].mask] = np.nan                
        to_regs = _psf(sent,sinds, brdfs[ind]*0.001, minds, bands[ind], psfs[ind], dia_cloud, rws[ind])

    else:
        brdfs[ind-1][brdfs[ind-1].mask] = np.nan
        to_regs = _psf(sent,sinds, brdfs[ind-1]*0.001, minds, bands[ind], psfs[ind], dia_cloud, rws[ind-1])
    return to_regs
        

        
def applied(lat, lon,fpath, mfile, psfs, pr=1):
    
    sentm = get_cloud_mask(fpath)
    doy = '%02d/%02d/%02d'%(int(fpath.split('/')[-3]), int(fpath.split('/')[-4]), int(fpath.split('/')[-5]))
    pos = fpath.split('/')[-8]+fpath.split('/')[-7]+fpath.split('/')[-6]
    if pr:
        print 'DOY: ', doy,'\n', 'Location: ', pos
        print 'Cloud proportion: ', sentm.sum()/(10980.*10980.)
    
    minds, sinds = get_coords(lat,lon) 
    modis_filenames = gdal.Open(mfile[0]).GetSubDatasets()
    modisQAs = gdal.Open(mfile[1]).GetSubDatasets()
    
    mete = readxml('%smetadata.xml'%fpath)
    sza = np.zeros(7)
    sza[:] = mete['mSz']
    vza = (mete['mVz'])[[1,2,3,7,8,11,12],]
    raa = (mete['mSa']-mete['mVa'])[[1,2,3,7,8,11,12],]

    brdfs, rws = get_rs(modisQAs, modis_filenames, (sza,vza, raa), bands=[2,3,0,1,1,5,6])
    brdfs.mask = brdfs.mask|(rws<1)
    args = fpath, sentm, brdfs, sinds, minds, psfs, rws
    par = partial(op, args=args)
    pool = multiprocessing.Pool(processes = 7)
    retval = pool.map(par, range(7))
    pool.close()
    pool.join()
    return retval
    

   