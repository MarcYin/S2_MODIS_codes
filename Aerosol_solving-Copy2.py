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
from smoothn import *
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
                 S2_fname=None, year=None, S2_month=None, S2_day=None, S2_psf=None, L8_psf=None):
        
        self.aot_site = aot_site
        self.year = int(year)
        self.L8_fname = L8_fname
        if L8_month != None:
            self.L8_month = int(L8_month)
            self.L8_day= int(L8_day)
            self.L8_doy = datetime.datetime(self.year, self.L8_month, self.L8_day).timetuple().tm_yday
            self.path = int(path)
            self.row = int(row)
            self.L8_psf =L8_psf
        
        self.S2_fname = S2_fname
        if S2_month != None:
            self.S2_month = int(S2_month)
            self.S2_day= int(S2_day)
            self.S2_doy = datetime.datetime(self.year, self.S2_month, self.S2_day).timetuple().tm_yday
            self.S2_psf = S2_psf
        
        self.lat = float(lat)
        self.lon = float(lon)
        self.h, self.v = mtile_cal(self.lat, self.lon)
        self._alpha = 1.42 #angstrom exponent for continental type aerosols
    
    def read_meta(self, Hfile, path=None, row=None):
    
        with open(Hfile.split('_toa_')[0]+'_MTL.txt', 'r') as inF:
            for line in inF:
                if 'CLOUD_COVER ' in line:
                    cloud_cover =  float(line.split('= ')[1])
        if cloud_cover<20:
            #print 'Less than 20% cloud.'
            b1 = gdal.Open(Hfile+'band1.tif').ReadAsArray()
            corners = b1.shape
            dic = {}
            with open(Hfile.split('_toa_')[0]+'_MTL.txt', 'r') as inF:
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

            vaa = np.nanmean(Landsat_azm[(Landsat_azm[:,2]==path)&(Landsat_azm[:,3]==row)].squeeze()[:2])

            return sza, saa, vza, vaa, dic, corners
        else:
            print 'To much cloud: ', cloud_cover
            return []     
        
       
    def L8_aot(self):
        self.wl = np.array([482.04,561.41,654.59,864.67,1608.86,2200.73])/1000
        self.bands = [2,3,4,5,6,7]
        pr=get_wrs(self.lat, self.lon)
        self.path, self.row = pr[0]['path'],pr[0]['row']
        self.Hfile = directory +'l_data/%s_toa_'%(self.L8_fname)
        self.Lfile = glob.glob('m_data/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'%(self.year, self.L8_doy,self.h,self.v))[0]
        self.sza, self.saa, self.vza, self.vaa, self.dic, self.corners = self.read_meta(self.Hfile, self.path, self.row)
        self.L_inds, self.H_inds = MSL_geo_trans(self.lat, self.lon, self.dic, self.corners)
        self.Lx, self.Ly = self.L_inds
        self.Hx, self.Hy = self.H_inds
        self.angles = np.zeros((3,6))
        self.angles[0,:] = self.sza
        self.angles[1,:] = self.vza
        self.angles[2,:] = self.vaa-self.saa
        
        if glob.glob(self.Lfile+'_L8_aoi_brdf.pkl')==[]:
            self.brdf, self.qa = get_brdf_six(self.Lfile, (self.angles[0], self.angles[1], self.angles[2]),\
                                              bands=[3,4,1,2,6,7], flag=None, Linds= self.L_inds)
            pkl.dump(np.array([self.brdf, self.qa]), open(self.Lfile+'_L8_aoi_brdf.pkl', 'w'))
            
        else:
            self.brdf, self.qa = pkl.load(open(self.Lfile+'_L8_aoi_brdf.pkl', 'r'))
        
        cloud = gdal.Open(self.Hfile[:-5]+'_cfmask.tif').ReadAsArray()
        cl_mask = cloud==4 # cloud pixels; strictest way is to set the clear pixels with cloud==0
        struct = ndimage.generate_binary_structure(2, 2)
        self.dia_cloud = ndimage.binary_dilation(cl_mask, structure=struct, iterations=20).astype(cl_mask.dtype)
        
        shape =  self.dia_cloud.shape
        xstd,ystd, angle, xs, ys = self.L8_psf[:5]
        self.shx, self.shy = (self.Hx+xs).astype(int), (self.Hy+ys).astype(int)
        self.val = (self.Hx+xs<shape[0])&(self.Hy+ys<shape[1])&(self.Hx+xs>0)&(self.Hy+ys>0)
        self.ker = self.gaussian(xstd,ystd,angle,True)
        retval = parmap(self.L8_get_to_cor, self.bands, nprocs=len(self.bands))

        self.L8_mask = np.array(retval)[:,1,:].astype(bool)
        self.L8_data = np.array(retval)[:,0,:]
        
        Mcomb_mask = np.all(self.qa<2, axis=0)
        Lcomb_mask = np.all(self.L8_mask, axis = 0)
        
        l8 = self.L8_data.copy()
        br = self.brdf.copy()
        l8[:,(~Lcomb_mask)|(~Mcomb_mask[self.val])]=np.nan
        l8[np.isnan(l8)], br[np.isnan(br)] = -9999999, -9999999
        mas = np.all((br[:,self.val]>0)&(br[:,self.val]<1)&(l8>0)&(l8<1), axis=0)
        self.to_cor = self.shx[self.val][mas], self.shy[self.val][mas],l8[:,mas], br[:,self.val][:,mas]
        
        dif = aot.to_cor[3]-aot.to_cor[2]
        u,d = dif.mean(axis=1)+ 3*dif.std(axis=1), dif.mean(axis=1)- 3*dif.std(axis=1)
        in_mask = np.all(np.array([(dif[i]>d[i])&(dif[i]<u[i]) for i in range(len(dif))]), axis=0)
        
        self.to_cor = self.shx[self.val][mas][in_mask], self.shy[self.val][mas][in_mask], l8[:,mas][:,in_mask], br[:,self.val][:,mas][:, in_mask]
          
        self.emus = parallel_rw_pkl(None, '6S_emulation_L8_', 'r')

        self.w = (np.array(self.wl))**(-self._alpha)
        self.w = self.w/self.w.sum()
        
        self.patch_pixs = 300.
        patches = []        
        self.inds = []
        indx, indy = self.to_cor[:2]
        for i in np.arange(0, np.ceil(shape[0]/self.patch_pixs)):
            for j in np.arange(0, np.ceil(shape[0]/self.patch_pixs)):
                patch_mask = (indx>i*self.patch_pixs) & (indx<(i+1)*self.patch_pixs)\
                & (indy>j*self.patch_pixs) & (indy<(j+1)*self.patch_pixs)
                patches.append(self._l8_opt(i, j, patch_mask))
                self.inds.append([i,j])
                
        self.inds = np.array(self.inds)
        paras = np.array([i[0] for i in patches])
        cost = np.array([i[1] for i in patches]).reshape(self.inds[:,0].max()+1,self.inds[:,1].max()+1)
        mask = (np.array(paras[:,1]).reshape(cost.shape)==0) | (np.array(paras[:,0]).reshape(cost.shape)==0) | np.isnan(cost)
        w = np.zeros_like(cost)
        w[~mask] = np.abs(1./cost[~mask])
        aot_map = np.zeros_like(cost)
        twv_map = np.zeros_like(cost)
        aot_map[mask] = paras[:,0].reshape(cost.shape)[~mask].mean()
        twv_map[mask] = paras[:,1].reshape(cost.shape)[~mask].mean()
        aot_map[~mask] = paras[:,0].reshape(cost.shape)[~mask]
        twv_map[~mask] = paras[:,1].reshape(cost.shape)[~mask]
        smed_aot = smoothn(aot_map, s=1, W=w**2, isrobust=True)
        smed_twv = smoothn(twv_map, s=1, W=w**2, isrobust=True)
        self.aot_map, self.twv_map = smed_aot[0], smed_wv[0]
        
        return self.aot_map, self.twv_map
       
    def _l8_opt(self, indx,indy, patch_mask):
        
        if np.all(~patch_mask):
            px, py = np.mgrid[indx*self.patch_pixs: (indx+1)*self.patch_pixs, indy*self.patch_pixs: (indy+1)*self.patch_pixs]
            pix_lat, pix_lon = cor_inter(np.array([px.ravel(), py.ravel()]), self.dic, self.corners)
            eles = np.mean(elevation(pix_lat, pix_lon)/1000.)
        else:
            pix_lat, pix_lon = cor_inter(np.array([self.to_cor[0][patch_mask],\
                                                   self.to_cor[1][patch_mask]]), self.dic, self.corners)
            eles = elevation(pix_lat, pix_lon)/1000.
        self.ECWMF_aot = np.mean(read_net(self.year,\
                                       self.L8_month,self.L8_day, np.arange(pix_lat.min(), pix_lat.max()+0.125, 0.125),\
                                       np.arange(pix_lon.min(),pix_lon.max()+0.125, 0.125), dataset='aot'))
        
        self.ECWMF_twv = np.mean(read_net(self.year,\
                                       self.L8_month,self.L8_day, np.arange(pix_lat.min(), pix_lat.max()+0.125, 0.125),\
                                       np.arange(pix_lon.min(),pix_lon.max()+0.125, 0.125), dataset='wv'))
        
        self.ECWMF_tco = np.mean(read_net(self.year,\
                                       self.L8_month,self.L8_day, np.arange(pix_lat.min(), pix_lat.max()+0.125, 0.125),\
                                       np.arange(pix_lon.min(),pix_lon.max()+0.125, 0.125), dataset='tco'))
        angles =[i*np.pi/180 for i in [self.sza, self.saa, self.vza, self.vaa]]
        args = angles, eles, self.ECWMF_tco*46.698, patch_mask
        bounds = ((0.,1.),(0.,5.))
        
        p0 = self.ECWMF_aot, self.ECWMF_twv/10.
        psolve = optimize.fmin_l_bfgs_b(self._cost,p0, approx_grad=0, iprint=1, bounds=bounds,fprime=None,args=(args,))
        
        return psolve
    
    def _cost(self, p, args = None):
        aot550, water = p
        angles, eles, ozone, patch_mask = args        
        sz, sa, vz, va = angles        
        pas = [self.to_cor[2][:,patch_mask], aot550, water, ozone, np.sin(sz), np.sin(vz), np.cos(sa-va), eles]

        pix_num = len(self.to_cor[2][0, patch_mask])

        paras =  np.zeros((len(self.bands), pix_num, 8))
        for i in range(len(pas)):
            if np.array(pas[i]).shape==(7,):
                dat = np.repeat(pas[1], axis=0, repeats= pix_num)
                paras[:,:,i] =  dat
            else:
                paras[:,:,i] =  pas[i]

        J = 0
        J_prime = np.zeros(2)
        for i in range(6):
            fwd, grad = self.emus[i][0].predict(paras[i], do_unc=0)
            # select from set of gradients
            g0,g1 = grad[:,1],grad[:,2]
            residual = np.array(fwd-self.to_cor[3][i, patch_mask])
            J_prime_wrt0_i = self.w[i] * residual * g0 
            J_prime_wrt1_i = self.w[i] * residual * g1
            J_i = 0.5 *self.w[i] * (residual**2)[:].sum()
            J += J_i
            J_prime += np.array([J_prime_wrt0_i,J_prime_wrt1_i])[:,:].sum(axis=1)

        return 1.*J/pix_num , J_prime
    
    
    
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
        
        
        
    def S2_aot(self,):
        self.wl=0.490,0.560,0.665,0.842, 1.610,2.190, 0.865
        self.bands = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A'
        m = mgrs.MGRS()
        mg_coor = m.toMGRS(self.lat, self.lon, MGRSPrecision=4)
        self.place = mg_coor[:5]
        
        self.Hfile = os.getcwd()+'/s_data/%s/%s/%s/%d/%d/%d/0/'%(self.S2_fname[:2], self.S2_fname[2],\
                                                                 self.S2_fname[3:5], self.year, self.S2_month, self.S2_day)
        
        self.Lfile = glob.glob('m_data/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'%(self.year, self.S2_doy,self.h,self.v))[0]

        mete = readxml('%smetadata.xml'%self.Hfile)
        self.sza = np.zeros(7)
        self.sza[:] = mete['mSz']
        self.saa = self.sza.copy()
        self.saa[:] = mete['mSa']
        # sometimes not all of the angles are available
        try:
            self.vza = (mete['mVz'])[[1,2,3,7,11,12,8],]
            self.vaa = (mete['mVa'])[[1,2,3,7,11,12,8],]
        except:
            self.vza = np.repeat(np.nanmean(mete['mVz']), 7)
            self.vaa = np.repeat(np.nanmean(mete['mVa']), 7)

        ll, ul, lr, ur = mg.toWgs(u'%s0000000000'%self.S2_fname), mg.toWgs(u'%s0000099999'%self.S2_fname),\
        mg.toWgs(u'%s9999900000'%self.S2_fname), mg.toWgs(u'%s9999999999'%self.S2_fname)

        self.dic = {'LL_LAT': ll[0],
                   'LL_LON': ll[1],
                   'LR_LAT': lr[0],
                   'LR_LON': lr[1],
                   'UL_LAT': ul[0],
                   'UL_LON': ul[1],
                   'UR_LAT': ur[0],
                   'UR_LON': ur[1]}
        self.corners = 10000, 10000

        #self.L_inds, self.H_inds = get_coords(self.lat,self.lon)
        self.L_inds, self.H_inds = MSL_geo_trans(self.lat, self.lon, self.dic, self.corners)

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
        xstd,ystd, angle, xs, ys = self.S2_psf[:5]
        self.shx, self.shy = (self.Hx+xs).astype(int), (self.Hy+ys).astype(int)
        self.val = (self.Hx+xs<shape[0])&(self.Hy+ys<shape[1])&(self.Hx+xs>0)&(self.Hy+ys>0)
        self.ker = self.gaussian(xstd,ystd,angle,True)
       
        retval = parmap(self.S2_get_to_cor, self.bands, nprocs=len(self.bands))
        self.S2_mask = np.array(retval)[:,1,:].astype(bool)
        self.S2_data = np.array(retval)[:,0,:]
        
        Mcomb_mask = np.all(self.qa<2, axis=0)
        Scomb_mask = np.all(self.S2_mask, axis = 0)
        
        s2 = self.S2_data.copy()
        br = self.brdf.copy()
        s2[:,(~Scomb_mask)|(~Mcomb_mask[self.val])]=np.nan
        s2[np.isnan(s2)], br[np.isnan(br)] = -9999999, -9999999
        mas = np.all((br[:,self.val]>0)&(br[:,self.val]<1)&(s2>0)&(s2<1), axis=0)
        self.to_cor = self.shx[self.val][mas], self.shy[self.val][mas],s2[:,mas], br[:,self.val][:,mas]
        
        dif = aot.to_cor[3]-aot.to_cor[2]
        u,d = dif.mean(axis=1)+ 3*dif.std(axis=1), dif.mean(axis=1)- 3*dif.std(axis=1)
        in_mask = np.all(np.array([(dif[i]>d[i])&(dif[i]<u[i]) for i in range(len(dif))]), axis=0)
        
        self.to_cor = self.shx[self.val][mas][in_mask], self.shy[self.val][mas][in_mask],\
        l8[:,mas][:,in_mask], br[:,self.val][:,mas][:, in_mask]
  
        self.emus = parallel_rw_pkl(None, '6S_emulation_S2_', 'r')
        self.w = (np.array(self.wl))**(-self._alpha)
        self.w = self.w/(self.w.sum())
        
        self.patch_pixs = 500
        patches = []        
        self.inds = []
        indx, indy = self.to_cor[:2]
        for i in range(0, 20):
            for j in range(0, 20):
                patch_mask = (indx>i*self.patch_pixs) & (indx<(i+1)*self.patch_pixs)\
                & (indy>j*self.patch_pixs) & (indy<(j+1)*self.patch_pixs)
                patches.append(self._S2_opt(i, j, patch_mask))
                self.inds.append([i,j])
        self.inds = np.array(self.inds)
        paras = np.array([i[0] for i in patches])
        paras = np.array([i[0] for i in patches])
        cost = np.array([i[1] for i in patches]).reshape(self.inds[:,0].max()+1,self.inds[:,1].max()+1)
        mask = (np.array(paras[:,1]).reshape(cost.shape)==0) | (np.array(paras[:,0]).reshape(cost.shape)==0) | np.isnan(cost)
        w = np.zeros_like(cost)
        w[~mask] = np.abs(1./cost[~mask])
        aot_map = np.zeros_like(cost)
        twv_map = np.zeros_like(cost)
        aot_map[mask] = paras[:,0].reshape(cost.shape)[~mask].mean()
        twv_map[mask] = paras[:,1].reshape(cost.shape)[~mask].mean()
        aot_map[~mask] = paras[:,0].reshape(cost.shape)[~mask]
        twv_map[~mask] = paras[:,1].reshape(cost.shape)[~mask]
        smed_aot = smoothn(aot_map, s=1, W=w**2, isrobust=True)
        smed_twv = smoothn(twv_map, s=1, W=w**2, isrobust=True)
        self.aot_map, self.twv_map = smed_aot[0], smed_wv[0]
        
        tifffile.imsave( self.Hfile+'aot.tiff', self.aot_map)
        tifffile.imsave( self.Hfile+'tcw.tiff', self.twv_map)
        
        return self.aot_map, self.twv_map
    
    def S2_cor(self,save=True):
        self.bands = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A'
        self.Hfile = os.getcwd()+'/s_data/%s/%s/%s/%d/%d/%d/0/'%(self.S2_fname[:2], self.S2_fname[2],\
                                                                 self.S2_fname[3:5], self.year, self.S2_month, self.S2_day)
        self.patch_pixs=500
        mete = readxml('%smetadata.xml'%self.Hfile)
        self.sza = np.zeros(7)
        self.sza[:] = mete['mSz']
        self.saa = self.sza.copy()
        self.saa[:] = mete['mSa']
        # sometimes not all of the angles are available
        try:
            self.vza = (mete['mVz'])[[1,2,3,7,11,12,8],]
            self.vaa = (mete['mVa'])[[1,2,3,7,11,12,8],]
        except:
            self.vza = np.repeat(np.nanmean(mete['mVz']), 7)
            self.vaa = np.repeat(np.nanmean(mete['mVa']), 7)
        
        ll, ul, lr, ur = mg.toWgs(u'%s0000000000'%self.S2_fname), mg.toWgs(u'%s0000099999'%self.S2_fname),\
        mg.toWgs(u'%s9999900000'%self.S2_fname), mg.toWgs(u'%s9999999999'%self.S2_fname)

        self.dic = {'LL_LAT': ll[0],
                   'LL_LON': ll[1],
                   'LR_LAT': lr[0],
                   'LR_LON': lr[1],
                   'UL_LAT': ul[0],
                   'UL_LON': ul[1],
                   'UR_LAT': ur[0],
                   'UR_LON': ur[1]}
        self.corners = 10000, 10000     
        if glob.glob(self.Hfile+'aot.tiff')==[]:
            self.aot_map, self.twv_map = self.S2_aot()
        else:
            self.aot_map = tifffile.imread(self.Hfile+'aot.tiff')
            self.twv_map = tifffile.imread(self.Hfile+'tcw.tiff')
        gridx, gridy = np.mgrid[0:20:1, 0:20:1]

        self.emus = parallel_rw_pkl(None, '6S_emulation_S2_', 'r')
        self.inds = np.array(zip(gridx.ravel(), gridy.ravel()))
        retval = parmap(self.S2_patch_correction, self.inds)
        
        self.S2_cored = np.zeros((7,10000,10000))
        
        for _,ind in enumerate(self.inds):
            self.S2_cored[:,ind[0]*self.patch_pixs:(ind[0]+1)*self.patch_pixs,\
                          ind[1]*self.patch_pixs:(ind[1]+1)*self.patch_pixs] = retval[_][1]
        if save:
            print 'Saving surface reflectance....'
            tifffile.imsave(self.Hfile+'sur_ref.tiff', self.S2_cored)
        return self.S2_cored
    
    def J_func(self, p, sur_ref=None, TOA_ref=None):
        a,b,c = p
        y = a+b*sur_ref/(1-c*sur_ref)
        cost = np.nansum(0.5*(TOA_ref-y)**2)
        return cost

    def S2_patch_correction(self, inds):
        ix,iy=inds
        toa_refs = []
        for _,band in enumerate(self.bands):
            if (_<=3):
                data = gdal.Open(self.Hfile+band+'.jp2').ReadAsArray(iy*self.patch_pixs,\
                                                                     ix*self.patch_pixs,self.patch_pixs, self.patch_pixs)*0.0001
                toa_refs.append(data)
            else:
                data = gdal.Open(self.Hfile+band+'.jp2').ReadAsArray(iy*self.patch_pixs/2,\
                                                                     ix*self.patch_pixs/2,self.patch_pixs/2,\
                                                                     self.patch_pixs/2)*0.0001
                data = ScaleExtent(data, shape=(self.patch_pixs, self.patch_pixs))
                toa_refs.append(data)

        toa_refs = np.array(toa_refs)
        TOA_refs = np.linspace(0.0001, 1,200)
        TOA_refs = np.tile(TOA_refs, 7).reshape((7,200))
        px, py = np.mgrid[ix*self.patch_pixs: (ix+1)*self.patch_pixs, iy*self.patch_pixs: (iy+1)*self.patch_pixs]
        pix_lat, pix_lon = cor_inter(np.array([px.ravel(), py.ravel()]), self.dic, self.corners)
        ele = mean(elevation(pix_lat, pix_lon)/1000.)

        aot = self.aot_map[ix,iy]
        twv = self.twv_map[ix,iy]
        tco = mean(read_net(self.year,\
                            self.S2_month,self.S2_day, np.arange(pix_lat.min(), pix_lat.max()+0.125, 0.125),\
                            np.arange(pix_lon.min(),pix_lon.max()+0.125, 0.125), dataset='tco'))

        pas = [TOA_refs, aot,  twv, tco*46.698, np.sin(self.sza*np.pi/180),\
               np.sin(self.vza*np.pi/180), np.cos((self.saa-self.vaa)*np.pi/180), ele]

        paras =  np.zeros((len(self.bands), 200, 8))
        for i in range(8):
            if np.array(pas[i]).shape==(7,):
                dat = np.repeat(pas[1], axis=0, repeats=200)
                paras[:,:,i] =  dat
            else:
                paras[:,:,i] =  pas[i]

        corrected_train = [self.emus[ind][0].predict(paras[ind]) for ind in range(7)]

        Sur_REF = []
        for i in range(7):
            par = partial(self.J_func, sur_ref=corrected_train[i][0], TOA_ref = TOA_refs[i])
            bounds = [0,0.1], [0.5, 1], [0., 0.5]
            solved = optimize.fmin_l_bfgs_b(par, (0.01,0.9, 0.01), iprint=-1, approx_grad=1, bounds=bounds)
            a,b,c = solved[0]
            cored = (np.array(toa_refs)[i]-a)/(b+(np.array(toa_refs)[i]-a)*c)
            Sur_REF.append(cored)
        Sur_REF = np.array(Sur_REF)
        Sur_REF[Sur_REF<0]=0.0001
        Sur_REF[Sur_REF>1]=1
        return [inds, Sur_REF]
        
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


    def _S2_opt(self, indx,indy, patch_mask):
        
        if np.all(~patch_mask):
            px, py = np.mgrid[indx*self.patch_pixs: (indx+1)*self.patch_pixs, indy*self.patch_pixs: (indy+1)*self.patch_pixs]
            pix_lat, pix_lon = cor_inter(np.array([px.ravel(), py.ravel()]), self.dic, self.corners)
            eles = np.mean(elevation(pix_lat, pix_lon)/1000.)
        else:
            pix_lat, pix_lon = cor_inter(np.array([self.to_cor[0][patch_mask],\
                                                   self.to_cor[1][patch_mask]]), self.dic, self.corners)
            eles = elevation(pix_lat, pix_lon)/1000.
        
        self.ECWMF_aot = np.mean(read_net(self.year,\
                                       self.S2_month,self.S2_day, np.arange(pix_lat.min(), pix_lat.max()+0.125, 0.125),\
                                       np.arange(pix_lon.min(),pix_lon.max()+0.125, 0.125), dataset='aot'))
        
        self.ECWMF_twv = np.mean(read_net(self.year,\
                                       self.S2_month,self.S2_day, np.arange(pix_lat.min(), pix_lat.max()+0.125, 0.125),\
                                       np.arange(pix_lon.min(),pix_lon.max()+0.125, 0.125), dataset='wv'))
        
        self.ECWMF_tco = np.mean(read_net(self.year,\
                                       self.S2_month,self.S2_day, np.arange(pix_lat.min(), pix_lat.max()+0.125, 0.125),\
                                       np.arange(pix_lon.min(),pix_lon.max()+0.125, 0.125), dataset='tco'))

        angles =[i*np.pi/180 for i in [self.sza, self.saa, self.vza, self.vaa]]
        args = angles, eles, self.ECWMF_tco*46.698, patch_mask
        bounds = ((0.,1.),(0.,5.))
        
        p0 = self.ECWMF_aot, self.ECWMF_twv/10.
        psolve = optimize.fmin_l_bfgs_b(self._cost,p0, approx_grad=0, iprint=1, bounds=bounds,fprime=None,args=(args,))
        
        return psolve
