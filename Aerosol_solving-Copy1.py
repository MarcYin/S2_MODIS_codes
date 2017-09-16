import os
import sys
sys.path.insert(0, 'python')
import numpy as np
from read_net import *
from fastRWpkl import *
import multiprocessing
from functools import partial
import gdal
from L_geo import *
from get_brdf import *
from geo_trans import *
import glob
from scipy import ndimage, signal, optimize
import scipy
from elevation import elevation
from get_wrs import *
import tifffile
import scipy.stats as stats
from mgrspy import mgrs as mg 
directory = os.getcwd()+'/'



def ScaleExtent(data, shape =(10980, 10980)): # used for unifine different array,     
        re = int(shape[0]/(data.shape[0]))
        return np.repeat(np.repeat(data, re, axis = 1), re, axis =0)

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

class Aerosol_retrival(object):
    
    def __init__(self, L8_fname=None, aot_site=None, lat=None, \
                 lon=None, path=None, row=None, L8_month=None, L8_day=None, \
                 S2_fname=None, year=None, S2_month=None, S2_day=None):
        
        self.aot_site = aot_site
        self.year = int(year)
        self.L8_fname = L8_fname
        if L8_month != None:
            self.L8_month = int(L8_month)
            self.L8_day= int(L8_day)
            self.L8_doy = datetime.datetime(self.year, self.L8_month, self.L8_day).timetuple().tm_yday
            self.path = int(path)
            self.row = int(row)
        
        self.S2_fname = S2_fname
        if S2_month != None:
            self.S2_month = int(S2_month)
            self.S2_day= int(S2_day)
            self.S2_doy = datetime.datetime(self.year, self.S2_month, self.S2_day).timetuple().tm_yday
        
        self.lat = float(lat)
        self.lon = float(lon)
        self.h, self.v = mtile_cal(self.lat, self.lon)
        self._alpha = 1.42 #angstrom exponent for continental type aerosols
       
    def L8_aot(self):
        self.wl = np.array([482.04,561.41,654.59,864.67,1608.86,2200.73])/1000
        self.bands = [2,3,4,5,6,7]
        pr=get_wrs(lat, lon)
        self.path, self.row = pr[0]['path'],pr[0]['row']
        self.Hfile = directory +'l_data/LC8%03d%03d%d%03dLGN00_toa_'%(self.path, self.row, self.year, self.doy)
        self.sza, self.saa, self.vza, self.vaa, self.dic, self.corners = read_meta(self.Hfile)
        self.L_inds, self.H_inds = ML_geo_trans(self.lat, self.lon, self.dic, self.corners)
        self.Lx, self.Ly = self.L_inds
        self.Hx, self.Hy = self.H_inds
        angles = np.zeros((3,6))
        angles[0,:] = sza
        angles[1,:] = vza
        angles[2,:] = vaa-saa
        
        if glob.glob(self.Lfile+'_L8_aoi_brdf.pkl')==[]:
            self.brdf, self.qa = get_brdf_six(self.Lfile, (self.sza, self.vza, self.vaa-self.saa), bands=[3,4,1,2,2,6,7], flag=None, Linds= self.L_inds)
            pkl.dump(np.array([self.brdf, self.qa]), open(self.Lfile+'_L8_aoi_brdf.pkl', 'w'))
            
        else:
            self.brdf, self.qa = pkl.load(open(self.Lfile+'_L8_aoi_brdf.pkl', 'r'))
        
        self.cloud = gdal.Open(self.Hfile[:-4]+'_cfmask.tif').ReadAsArray()
        cl_mask = cloud==4 # cloud pixels; strictest way is to set the clear pixels with cloud==0
        struct = ndimage.generate_binary_structure(2, 2)
        dia_cloud = ndimage.binary_dilation(cl_mask, structure=struct, iterations=20).astype(cl_mask.dtype)
        
        shape =  dia_cloud.shape
        xstd,ystd, angle, xs, ys = self.psf[:5]
        self.shx, self.shy = (self.Hx+xs).astype(int), (self.Hy+ys).astype(int)
        self.val = (self.Hx+xs<shape[0])&(self.Hy+ys<shape[1])&(self.Hx+xs>0)&(self.Hy+ys>0)
        self.ker = self.gaussian(xstd,ystd,angle,True)
        retval = parmap(self.L8_get_to_cor, bands, nprocs=len(bands))
        
        self.L8_mask = np.array(retval)[:,1,:].astype(bool)
        self.L8_data = np.array(retval)[:,0,:]
        
        Mcomb_mask = np.all(self.qa==0, axis=0)
        Lcomb_mask = np.all(self.L8_mask, axis = 0)
        
        l8 = self.L8_data.copy()
        br = self.brdf.copy()
        l8[:,(~Lcomb_mask)|(~Mcomb_mask[self.val])]=np.nan
        l8[np.isnan(l8)], br[np.isnan(br)] = -9999999, -9999999
        mas = np.all((br[:,self.val]>0)&(br[:,self.val]<1)&(l8>0)&(l8<1), axis=0)
        self.to_cor = self.shx[self.val][mas], self.shy[self.val][mas],s2[:,mas], br[:,self.val][:,mas]
        
        self.L8_emus = parallel_rw_pkl(None, '6S_emulation_L8_', 'r')
        self.aot = read_net(self.year, self.month,self.day, np.arange(self.lat-2,self.lat+2, 0.125),\
                            np.arange(self.lon-3,self.lon+3, 0.125), dataset='aot')
        self.twv = read_net(self.year, self.month,self.day, np.arange(self.lat-2,self.lat+2, 0.125),\
                            np.arange(self.lon-3,self.lon+3, 0.125), dataset='wv')
        self.tco = read_net(self.year, self.month,self.day, np.arange(self.lat-2,self.lat+2, 0.125),\
                            np.arange(self.lon-3,self.lon+3, 0.125), dataset='tco')
        
        self.w = (np.array(self.wl)/self.wl[0])**(-self._alpha)
        self._lats, self._lons = np.arange(self.lat-2,self.lat+2, 0.125), np.arange(self.lon-3,self.lon+3, 0.125)
        this_lat, this_lon = cor_inter(np.array([self.to_cor[0], self.to_cor[1]]), dic, corners)
        self.eles = elevation(this_lat, this_lon)/1000.
        
        ret = parmap(self._opt1, range(len(self.to_cor[0])), nprocs=10)
        parallel_rw_pkl(ret, 'L8_%s_%s_%s_aot'%(self.path, self.row, self.doy), 'w')
        
        return ret

    
    def _cost1(self, p, args = None):
        #'aot550', 'water', 'ozone'
        aot550, water = p
        TOA_refs, M_refs, angles, ele, ozone = args        
        sz, sa, vz, va = angles        
        Sur_refs = [self.L8_emus[ind][0].predict(np.array([[toa_ref, aot550, water, ozone, \
                                                       np.sin(sz), np.sin(vz), np.cos((sa-va)), \
                                                       ele],]))[0][0] for ind, toa_ref in enumerate(TOA_refs)]
        Sur_refs = np.array(Sur_refs)
        M_refs = np.array(M_refs)
        cost = sum(((Sur_refs-M_refs)**2)*self.w)    
        return cost

    def _opt1(self,ind):

        #sent_refs, modis_refs = np.array([refs[ii][tuple(aoi[ind])] for ii in range(7)]).T

        TOA_refs = self.to_cor[2][:,ind]
        M_refs = self.to_cor[3][:,ind]
        m = mgrs.MGRS()
        pix_lat, pix_lon = cor_inter(np.array([[self.to_cor[0][ind], self.to_cor[1][ind]],]).T, self.dic, self.corners)
        ele = self.eles[ind]
        inx_lat, inx_lon = (np.abs(self._lats-pix_lat)).argmin(),(np.abs(self._lons-pix_lon)).argmin()

        aot0, tcw0, tco0 = self.aot[inx_lat, inx_lon], self.twv[inx_lat, inx_lon]/10., self.tco[inx_lat, inx_lon]
        angles =[i*np.pi/180 for i in [self.sza, self.saa, self.vza, self.vaa]]
        ozone = tco0*46.698
        args = TOA_refs, M_refs , angles, ele, ozone
        p = aot0, tcw0 
        bounds = ((0.,1.),(0.,5.))
        psolve = optimize.fmin_l_bfgs_b(self._cost1,p, iprint=-1, approx_grad=1, args=(args,), bounds=bounds)

        return [self.to_cor[0][ind], self.to_cor[1][ind],psolve]
    
    
    def L8_get_to_cor(self, band):
        fname = self.Hfile + 'band%s.tif'%band
        data = gdal.Open(fname).ReadAsArray()*0.0001
        mask = ~(data<=0).astype('bool')
        struct = ndimage.generate_binary_structure(2, 2)
        small_mask = ndimage.binary_erosion(mask, structure=struct, iterations=20).astype(mask.dtype)
        val_mask = (~self.dia_cloud)&small_mask
        used_mask = val_mask[self.shx[self.val], self.shy[self.val]]
        used_data = signal.fftconvolve(data, self.ker, mode='same')[self.shx[self.val], self.shy[self.val]]
        return used_data, used_mask 
        
        
    def gaussian(self, xstd, ystd, angle, norm = True):
        win = int(round(max(2*1.69*xstd, 3*ystd)))
        winx = win*2**0.5
        winy = win*2**0.5
        xgaus = signal.gaussian(winx, xstd)
        ygaus = signal.gaussian(winy, ystd)
        gaus  = np.outer(xgaus, ygaus)
        r_gaus = ndimage.interpolation.rotate(gaus, angle, reshape=True)
        center = np.array(r_gaus.shape)/2
        cgaus = r_gaus[center[0]-win/2: center[0]+win/2, center[1]-win/2:center[1]+win/2]
        if norm:
            return cgaus/cgaus.sum()
        else:
            return cgaus  
        
        
        
    def S2_aot(self):
        self.wl=0.490,0.560,0.665,0.842,0.865,1.610,2.190
        self.bands = 'B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12'
        m = mgrs.MGRS()
        mg_coor = m.toMGRS(self.lat, self.lon, MGRSPrecision=4)
        self.place = mg_coor[:5]
        
        self.Hfile = os.getcwd()+'/s_data/%s/%s/%s/%d/%d/%d/0/'%(mg_coor[:2], mg_coor[2], mg_coor[3:5], self.year, self.month, self.day)
        self.Lfile = glob.glob('m_data/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'%(self.year,self.doy,self.h,self.v))[0]

        mete = readxml('%smetadata.xml'%self.Hfile)
        self.sza = np.zeros(7)
        self.sza[:] = mete['mSz']
        self.saa = self.sza.copy()
        self.saa[:] = mete['mSa']
        self.vza = (mete['mVz'])[[1,2,3,7,8,11,12],]
        self.vaa = (mete['mVa'])[[1,2,3,7,8,11,12],]

        self.L_inds, self.H_inds = get_coords(self.lat,self.lon)
        self.Lx, self.Ly = self.L_inds
        self.Hx, self.Hy = self.H_inds
        
        if glob.glob(self.Lfile+'_S2_aoi_brdf.pkl')==[]:
            self.brdf, self.qa = get_brdf_six(self.Lfile, (self.sza, self.vza, self.vaa-self.saa), bands=[3,4,1,2,2,6,7], flag=None, Linds= self.L_inds)
            pkl.dump(np.array([self.brdf, self.qa]), open(self.Lfile+'_S2_aoi_brdf.pkl', 'w'))
            
        else:
            self.brdf, self.qa = pkl.load(open(self.Lfile+'_S2_aoi_brdf.pkl', 'r'))
        
        if glob.glob(self.Hfile+'cloud.tif')==[]:
            cl = classification(fhead = self.Hfile, bands = (2,3,4,8,11,12,13), bounds = None)
            cl.Get_cm_p()
            self.cloud = cl.cm.copy()
            tifffile.imsave(self.Hfile+'cloud.tif', self.cloud.astype(int))
            self.H_data = np.repeat(np.repeat(cl.b12, 2, axis=1), 2, axis=0)
            del cl
        else:
            self.cloud = tifffile.imread(self.Hfile+'cloud.tif')
            
        struct = ndimage.generate_binary_structure(2, 2)
        self.dia_cloud = ndimage.binary_dilation(self.cloud.astype(bool), structure=struct, iterations=60).astype(bool)
        
        shape = (10000, 10000)
        xstd,ystd, angle, xs, ys = self.psf[:5]
        self.shx, self.shy = (self.Hx+xs).astype(int), (self.Hy+ys).astype(int)
        self.val = (self.Hx+xs<shape[0])&(self.Hy+ys<shape[1])&(self.Hx+xs>0)&(self.Hy+ys>0)
        self.ker = self.gaussian(xstd,ystd,angle,True)
       
        retval = parmap(self.S2_get_to_cor, self.bands, nprocs=len(self.bands))
        self.S2_mask = np.array(retval)[:,1,:].astype(bool)
        self.S2_data = np.array(retval)[:,0,:]
        
        Mcomb_mask = np.all(self.qa==0, axis=0)
        Scomb_mask = np.all(self.S2_mask, axis = 0)
        
        s2 = self.S2_data.copy()
        br = self.brdf.copy()
        s2[:,(~Scomb_mask)|(~Mcomb_mask[self.val])]=np.nan
        s2[np.isnan(s2)], br[np.isnan(br)] = -9999999, -9999999
        mas = np.all((br[:,self.val]>0)&(br[:,self.val]<1)&(s2>0)&(s2<1), axis=0)
        self.to_cor = self.shx[self.val][mas], self.shy[self.val][mas],s2[:,mas], br[:,self.val][:,mas]
        
        self.S2_emus = pkl.load(open('6S_emulation_S2.pkl', 'r'))
        self.aot = read_net(self.year, self.month,self.day, np.arange(self.lat-2,self.lat+2, 0.125),\
                            np.arange(self.lon-3,self.lon+3, 0.125), dataset='aot')
        self.twv = read_net(self.year, self.month,self.day, np.arange(self.lat-2,self.lat+2, 0.125),\
                            np.arange(self.lon-3,self.lon+3, 0.125), dataset='wv')
        self.tco = read_net(self.year, self.month,self.day, np.arange(self.lat-2,self.lat+2, 0.125),\
                            np.arange(self.lon-3,self.lon+3, 0.125), dataset='tco')
        
        ll, ul, lr, ur = mg.toWgs(u'%s0000000000'%self.S2_fname), mg.toWgs(u'%s0000099999'%self.S2_fname),\
        mg.toWgs(u'%s9999900000'%self.S2_fname), mg.toWgs(u'%s9999999999'%self.S2_fname)
            
        dic = {'LL_LAT': ll[0],
               'LL_LON': ll[1],
               'LR_LAT': lr[0],
               'LR_LON': lr[1],
               'UL_LAT': ul[0],
               'UL_LON': ul[1],
               'UR_LAT': ur[0],
               'UR_LON': ur[1]}
        corners = 10000, 10000
        
        self.w = (np.array(self.wl)/self.wl[0])**(-self._alpha)
        self._lats, self._lons = np.arange(self.lat-2,self.lat+2, 0.125), np.arange(self.lon-3,self.lon+3, 0.125)
        this_lat, this_lon = cor_inter(np.array([self.to_cor[0], self.to_cor[1]]), dic, corners)
        self.eles = elevation(this_lat, this_lon)/1000.
        
        ret = parmap(self._opt2, range(len(self.to_cor[0])), nprocs=10)
        parallel_rw_pkl(ret, 'S2_%s_%s_aot'%(self.place, self.doy), 'w')
        
        return ret

    def S2_get_to_cor(self, band):
        fname = self.Hfile + '%s.jp2'%band
        data = gdal.Open(fname).ReadAsArray()*0.0001
        
        if data.shape[0]<10980:
            data = ScaleExtent(data)
        mask = ~(data<=0).astype('bool')
        struct = ndimage.generate_binary_structure(2, 2)
        small_mask = ndimage.binary_erosion(mask, structure=struct, iterations=60).astype(mask.dtype)
        val_mask = (~self.dia_cloud)&small_mask
        used_mask = val_mask[self.shx[self.val], self.shy[self.val]]
        used_data = signal.fftconvolve(data, self.ker, mode='same')[self.shx[self.val], self.shy[self.val]]

        return used_data, used_mask 


        
    def _cost2(self, p, args = None):
        
        #'aot550', 'water'
        aot550, water = p
        TOA_refs, M_refs, angles, ele, ozone = args        
        sz, sa, vz, va = angles        
        Sur_refs = [self.S2_emus[ind][0].predict(np.array([[toa_ref, aot550, water, ozone, \
                                                            np.sin(sz[ind]), np.sin(vz[ind]), np.cos((sa-va)[ind]), \
                                                            ele],]))[0][0] for ind, toa_ref in enumerate(TOA_refs)]
        Sur_refs = np.array(Sur_refs)
        M_refs = np.array(M_refs)
        cost = sum(((Sur_refs-M_refs)**2)*self.w)    
        return cost


    def _opt2(self,ind):

        #sent_refs, modis_refs = np.array([refs[ii][tuple(aoi[ind])] for ii in range(7)]).T

        TOA_refs = self.to_cor[2][:,ind]
        M_refs = self.to_cor[3][:,ind]
        m = mgrs.MGRS()
        pix_lat, pix_lon = m.toLatLon(self.place+'%04d%04d'%(self.to_cor[0][ind], self.to_cor[1][ind]))
        ele = self.eles[ind]
        inx_lat, inx_lon = (np.abs(self._lats-pix_lat)).argmin(),(np.abs(self._lons-pix_lon)).argmin()
        aot0, tcw0, tco0 = self.aot[inx_lat, inx_lon], self.twv[inx_lat, inx_lon]/10., self.tco[inx_lat, inx_lon]

        angles =[i*np.pi/180 for i in [self.sza, self.saa, self.vza, self.vaa]]

        ozone = tco0*46.698

        args = TOA_refs, M_refs , angles, ele, ozone
        #print args
        p = aot0, tcw0 
        bounds = ((0.,1),(0.,5.))
        psolve = optimize.fmin_l_bfgs_b(self._cost2,p, iprint=-1, approx_grad=1, args=(args,), bounds=bounds)
        return [self.to_cor[0][ind],self.to_cor[1][ind],psolve]
        
        
        
        



