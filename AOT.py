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
from Py6S import *
from getmodis import grab_modis_toa

directory = os.getcwd()+'/'
magic = 0.618034

def ScaleExtent(data, shape =(10980, 10980)): # used for unifine different array,     
        re = int(shape[0]/(data.shape[0]))
        return np.repeat(np.repeat(data, re, axis = 1), re, axis =0)

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))
        
rsr = PredefinedWavelengths.LANDSAT_OLI_B2,PredefinedWavelengths.LANDSAT_OLI_B3,PredefinedWavelengths.LANDSAT_OLI_B4, \
        PredefinedWavelengths.LANDSAT_OLI_B5, PredefinedWavelengths.LANDSAT_OLI_B6, PredefinedWavelengths.LANDSAT_OLI_B7        
def atm(p, RSR=None):
    #print len(p[0])
    # ele in km
    #print  len(p[0])
    aot550, water, ozone, sz, vz, raa , ele= p
    path = '/home/ucfafyi/DATA/Downloads/6SV2.1/sixsV2.1'
    s = SixS(path)
    ss = []
    s = SixS(path)
    s.altitudes.set_target_custom_altitude(ele)
    s.altitudes.set_sensor_satellite_level()
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(GroundReflectance.GreenVegetation)
    s.geometry = Geometry.User()
    s.geometry.solar_a = 0
    s.geometry.solar_z = np.arcsin(sz)*180/np.pi
    s.geometry.view_a = np.arccos(raa)*180/np.pi
    s.geometry.view_z = np.arcsin(vz)*180/np.pi
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.aot550 = aot550
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(water, ozone)
    s.wavelength = Wavelength(RSR)
    #s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(TOA_ref)
    s.run()      
    return s

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
                 S2_fname=None, year=None, S2_month=None, S2_day=None, S2_psf=None, L8_psf=None,
                 modis_dir=None, doy=None, h=None, v=None, mdata='m_data'):
        
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
            self.S2_doy = doy
            self.S2_psf = S2_psf
        if modis_dir is not None:
            self.doy = doy
            self.modis_dir = modis_dir
            self.h=h
            self.v=v
        # where the mcd43 data are    
        self.mdata = mdata
        
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
        self.Lfile = glob.glob('%s/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'%(self.mdata,self.year, self.L8_doy,self.h,self.v))[0]
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
        
        
        
        dif = self.to_cor[3]-self.to_cor[2]
        u,d = dif.mean(axis=1)+ 3*dif.std(axis=1), dif.mean(axis=1)- 3*dif.std(axis=1)
        in_mask = np.all(np.array([(dif[i]>d[i])&(dif[i]<u[i]) for i in range(len(dif))]), axis=0)
        
        self.to_cor = self.shx[self.val][mas][in_mask], self.shy[self.val][mas][in_mask], l8[:,mas][:,in_mask], br[:,self.val][:,mas][:, in_mask]
        self.qa = magic** (self.qa[:,self.val][:,mas][:, in_mask])
        
        
        
        self.emus = parallel_rw_pkl(None, '6S_emulation_L8_', 'r')

        self.w = (np.array(self.wl))**(-self._alpha)
        self.w = self.w/self.w.sum()
        
        self.patch_pixs = 100.
        patches = []        
        self.inds = []
        indx, indy = self.to_cor[:2]
        self.post_uncs = []
        for i in np.arange(0, np.ceil(shape[0]/self.patch_pixs)):
            for j in np.arange(0, np.ceil(shape[1]/self.patch_pixs)):
                patch_mask = (indx>i*self.patch_pixs) & (indx<(i+1)*self.patch_pixs)\
                & (indy>j*self.patch_pixs) & (indy<(j+1)*self.patch_pixs)
                if patch_mask.sum() == 0:
                    patches.append(([0,0,0],0))
                    self.inds.append([i,j])
                else:
                    patches.append(self._l8_opt(i, j, patch_mask))
                    self.inds.append([i,j])
                
        self.inds = np.array(self.inds)
        paras = np.array([i[0] for i in patches])
        cost = np.array([i[1] for i in patches]).reshape(int(self.inds[:,0].max()+1),int(self.inds[:,1].max()+1))
        para_names = 'aot', 'twv', 'tco'
        masks = []
        para_maps = []
        smed_paras = []
        unc_maps = []
        for _ in range(3):
            mask = (np.array(paras[:,_]).reshape(cost.shape)==0) | np.isnan(cost)
            masks.append(mask)
            unc = np.array([np.r_[np.array([i[0], i[1]]), i[2][_]] for i in self.post_uncs])
            unc_map = np.zeros_like(cost)
            unc_map[:] = np.nan
            unc_map[unc[:,0].astype(int), unc[:,1].astype(int)] = unc[:,2]
            unc_maps.append(unc_map)
            w = np.zeros_like(cost)
            w[~mask] = 1./unc_map[~mask]
            para_map = np.zeros_like(cost)
            para_map[mask] = paras[:,_].reshape(cost.shape)[~mask].mean()
            para_map[~mask] = paras[:,_].reshape(cost.shape)[~mask]
            para_maps.append(para_map)
            smed_para = smoothn(para_map,s=0.05, W=w**2, isrobust=True)[0]
            smed_paras.append(smed_para)
            tifffile.imsave( self.Hfile+'%s.tiff'%para_names[_], smed_para)
        
        self.masks, self.para_maps, self.unc_maps = masks, para_maps, unc_maps
        return unc_maps, smed_paras, para_maps
       
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
        args = angles, eles, patch_mask
        
        bounds = [[0.001, 2.1], 
                  [0.1, 6], 
                  [0.25, 0.45],
                  ]
        
        self.pix_num = len(self.to_cor[3][0, patch_mask])
        self.prior_mean =  np.zeros((len(self.bands), self.pix_num, 3))
        #self.prior_mean[:,:,0] = self.to_cor[3][:,patch_mask]
        self.prior_mean[:,:,:3] = np.array([self.ECWMF_aot, self.ECWMF_twv/10., self.ECWMF_tco*46.698])
        p0 =  self.ECWMF_aot, self.ECWMF_twv/10., self.ECWMF_tco*46.698
        psolve = optimize.fmin_l_bfgs_b(self.L8_cal_cost,p0, approx_grad=0, iprint=1, bounds=bounds,fprime=None,args=(args,))
        post_unc = ((1./ np.array([0.5, 0.5, 0.01])**2 + self.inv_obs_unc)**-1)**0.5
        self.post_uncs.append([indx, indy, post_unc ])
        return psolve
    
    def L8_cal_cost(self, p, args=None):
        aot550, water, ozone  = p
        angles, eles,  patch_mask = args        
        sz, sa, vz, va = angles   
        toa_ref = self.to_cor[2][:,patch_mask]
        self.sur_ref = self.to_cor[3][:,patch_mask]
        sur_ref_sigma1 = self.sur_ref*0.05 + 0.005
        mod09_unc = np.array([0.003, 0.004, 0.004, 0.015,  0.010, 0.006])
        sur_ref_sigma2 = np.repeat(mod09_unc[:self.sur_ref.shape[0]], self.sur_ref.shape[1]).reshape(self.sur_ref.shape)
        self.sur_ref_sigma = np.array((sur_ref_sigma1, sur_ref_sigma2)).max(axis=0)/self.qa[:,patch_mask]

        pas = [toa_ref, aot550, water, ozone, np.sin(sz), np.sin(vz), np.cos(sa-va), eles]
        self.paras =  np.zeros((len(self.bands), self.pix_num, 8))
        for i in range(len(pas)):
            #if np.array(pas[i]).shape==(6,):
            #    dat = np.repeat(pas[i], axis=0, repeats= self.pix_num)
            #    self.paras[:,:,i] =  dat
            #else:
            self.paras[:,:,i] =  pas[i]

        prior, dprior = self.prior_cost ()
        lklhood, dlklhood = self.obs_cost ()

        return prior + lklhood, dprior + dlklhood

    def obs_cost_old(self,):
        J = 0
        J_prime = np.zeros(3)
        for i in range(6):
            fwd, grad = self.emus[i][0].predict(self.paras[i], do_unc=0)
            # select from set of gradients
            g1,g2, g3 = grad[:,1],grad[:,2], grad[:,3]
            residual = np.array(fwd-self.sur_ref[i])
            J_prime_wrt1_i = self.w[i] * residual * g1 /self.sur_ref_sigma[i]**2
            J_prime_wrt2_i = self.w[i] * residual * g2 /self.sur_ref_sigma[i]**2
            J_prime_wrt3_i = self.w[i] * residual * g3 /self.sur_ref_sigma[i]**2
            J_i = (0.5 *self.w[i] * (residual**2)[:]/self.sur_ref_sigma[i]**2).sum()
            J += J_i
            J_prime += np.array([J_prime_wrt1_i, J_prime_wrt2_i, J_prime_wrt3_i])[:,:].sum(axis=1)

        return 1.*J/self.pix_num , J_prime
    
    def obs_cost(self,):
        J = 0
        self.inv_obs_unc = np.zeros(3)
        J_prime = np.zeros(3)
        for i in range(6):
            fwd, m_unc, grad = self.emus[i][0].predict(self.paras[i])
            # select from set of gradients
            g1,g2, g3 = grad[:,1],grad[:,2], grad[:,3] 
            com_unc = self.sur_ref_sigma[i]**2 #m_unc**2 + self.sur_ref_sigma[i]**2
            residual = np.array(fwd-self.sur_ref[i])
            J_prime_wrt1_i = self.w[i] * residual * g1 /com_unc
            J_prime_wrt2_i = self.w[i] * residual * g2 /com_unc
            J_prime_wrt3_i = self.w[i] * residual * g3 /com_unc
            J_i = (0.5 *self.w[i] * (residual**2)[:]/com_unc).sum()
            J += J_i
            ind_dev = np.array([J_prime_wrt1_i, J_prime_wrt2_i, J_prime_wrt3_i])[:,:].sum(axis=1)
            J_prime += ind_dev

            self.inv_obs_unc += (ind_dev**2)*((self.w[i] * 1./ com_unc).sum())
        

        return 1.*J, J_prime

    def prior_cost(self,):
        x = self.paras[:,:,1:4]
        self.prior_sigma = np.zeros_like(x)
        #self.prior_sigma[:,:,0] = self.sur_ref_sigma
        self.prior_sigma[:,:,0:3] = np.array([0.5, 0.5, 0.01])
        cost = 0.5*( x - self.prior_mean)**2/self.prior_sigma**2
        dcost = (( x - self.prior_mean)/self.prior_sigma**2).sum(axis=(0,1))

        return cost.sum(), dcost
    
    
    def L8_cor(self,save=True):
        self.bands = [2,3,4,5,6,7]
        self.Hfile = directory +'l_data/%s_toa_'%(self.L8_fname)
        #self.patch_pixs=160
        self.sza, self.saa, self.vza, self.vaa, self.dic, self.corners = self.read_meta(self.Hfile, self.path, self.row)
        if glob.glob(self.Hfile+'aot.tiff')==[]:
            self.aot_map, self.twv_map, self.tco_map = self.L8_aot()
        else:
            self.aot_map = tifffile.imread(self.Hfile+'aot.tiff')
            self.twv_map = tifffile.imread(self.Hfile+'twv.tiff')
            self.tco_map = tifffile.imread(self.Hfile+'tco.tiff')
        
        gridx, gridy = np.mgrid[0:self.aot_map.shape[0]:1, 0:self.aot_map.shape[1]:1]
        self.inds = np.array(zip(gridx.ravel(), gridy.ravel()))
        
        retval = parmap(self.L8_patch_correction, self.inds, nprocs=10)
        
        self.L8_cored = np.zeros((6,self.corners[0],self.corners[1]))
        
        for _,ind in enumerate(self.inds):
            self.L8_cored[:,ind[0]*self.patch_pixs:(ind[0]+1)*self.patch_pixs,\
                          ind[1]*self.patch_pixs:(ind[1]+1)*self.patch_pixs] = retval[_][1]
        if save:
            print 'Saving surface reflectance....'
            for i,band in enumerate(self.bands):
                tifffile.imsave(self.Hfile+'sur_ref_band%d.tiff'%band, self.L8_cored[i])
        return self.L8_cored
    
    def L8_patch_correction(self, inds):
        ix,iy=inds
        toa_refs = []
        #self.Hfile = directory +'l_data/%s_toa_'%(self.L8_fname)
        #self.aot_map = tifffile.imread(self.Hfile+'aot.tiff')
        #self.twv_map = tifffile.imread(self.Hfile+'twv.tiff')
        #self.tco_map = tifffile.imread(self.Hfile+'tco.tiff')
        #self.bands = [2,3,4,5,6,7]
        #self.emus = parallel_rw_pkl(None, '6S_emulation_S2_', 'r')
        #self.patch_pixs=300
        #self.sza, self.saa, self.vza, self.vaa, self.dic, self.corners = self.read_meta(self.Hfile, self.path, self.row)
        for _,band in enumerate(self.bands):

            fname = self.Hfile + 'band%s.tif'%band
            yoff, xoff = min(self.corners[1]-iy*self.patch_pixs, self.patch_pixs), min(self.corners[0]-ix*self.patch_pixs, self.patch_pixs)
            data = gdal.Open(fname).ReadAsArray(iy*self.patch_pixs,ix*self.patch_pixs,\
                                                yoff, xoff)*0.0001
            toa_refs.append(data)
        
        toa_refs = np.array(toa_refs)
        
        toa_refs[toa_refs<0] = np.nan
        px, py = np.mgrid[ix*self.patch_pixs: (ix+1)*self.patch_pixs, iy*self.patch_pixs: (iy+1)*self.patch_pixs]
        pix_lat, pix_lon = cor_inter(np.array([px.ravel(), py.ravel()]), self.dic, self.corners)
        ele = max(mean(elevation(pix_lat, pix_lon)/1000.), 0)
        

        aot = self.aot_map[ix,iy]
        twv = self.twv_map[ix,iy]
        tco = self.tco_map[ix,iy]
        #tco = mean(read_net(self.year,\
        #                    self.L8_month,self.L8_day, np.arange(pix_lat.min(), pix_lat.max()+0.125, 0.125),\
        #                    np.arange(pix_lon.min(),pix_lon.max()+0.125, 0.125), dataset='tco'))

        p = [aot,  twv, tco, np.sin(self.sza*np.pi/180),\
               np.sin(self.vza*np.pi/180), np.cos((self.saa-self.vaa)*np.pi/180), ele]

        Sur_REF = []
        for band in range(6):
            s = atm(p, rsr[band])
            a,b,c = s.outputs.atmospheric_intrinsic_reflectance, s.outputs.transmittance_total_scattering.total, s.outputs.spherical_albedo.total
            cored = (np.array(toa_refs)[band]-a)/(b+(np.array(toa_refs)[band]-a)*c)
            Sur_REF.append(cored)
        Sur_REF = np.array(Sur_REF)
        Sur_REF[Sur_REF<0]=0.0001
        Sur_REF[Sur_REF>1]=1

        return [inds, Sur_REF]
    
    def gaussian(self, xstd, ystd, angle, norm = True):
        win = int(round(max(2*1.69*xstd, 3*ystd)))
        winx = int(round(win*2**0.5))
        winy = int(round(win*2**0.5))
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
    
    def modis_aot(self):
        '''
        Load data and solve the aot problem
        
        
        self.bands = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A'
        m = mgrs.MGRS()
        mg_coor = m.toMGRS(self.lat, self.lon, MGRSPrecision=4)
        self.place = mg_coor[:5]
        
        self.Hfile = os.getcwd()+'/s_data/%s/%s/%s/%d/%d/%d/0/'%(self.S2_fname[:2], self.S2_fname[2],\
                                                                 self.S2_fname[3:5], self.year, self.S2_month, self.S2_day)
        '''
        import pdb;pdb.set_trace()
        self.wl= 0.645 ,  0.8585,  0.469 ,  0.555 ,  1.24  ,  1.64  ,  2.13
        
        # vza,sza,vaa,saa
        modis_toa, modis_angles = grab_modis_toa(year=2006,doy=200,verbose=True,
        mcd43file = '/home/ucfafyi/DATA/S2_MODIS/m_data/MCD43A1.A2016128.h11v04.006.2016180234038.hdf',
        directory_l1b="/data/selene/ucfajlg/Bondville_MODIS/THERMAL")
        
        self.mcd43_f = glob.glob('%s/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'%(self.mdata,self.year, self.doy,self.h,self.v))[0]

        if glob.glob(self.mcd43_f+'_S2_aoi_brdf.pkl')==[]:
            
            self.brdf, self.qa = get_brdf_six(self.mcd43_f, \
                                              (modis_angles[1], modis_angles[0], modis_angles[2]-modis_angles[3]),\
                                              bands=[3,4,1,2,2,6,7], flag=None, Linds= None)
            
            pkl.dump(np.array([self.brdf, self.qa]), open(self.mcd43_f + '_S2_aoi_brdf.pkl', 'w'))   
        else:
            self.brdf, self.qa = pkl.load(open(self.Lfile+'_S2_aoi_brdf.pkl', 'r'))
        '''
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
        '''
        
        
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
        
        dif = self.to_cor[3]-self.to_cor[2]
        u,d = dif.mean(axis=1)+ 3*dif.std(axis=1), dif.mean(axis=1)- 3*dif.std(axis=1)
        in_mask = np.all(np.array([(dif[i]>d[i])&(dif[i]<u[i]) for i in range(len(dif))]), axis=0)
        
        self.to_cor = self.shx[self.val][mas][in_mask], self.shy[self.val][mas][in_mask],\
        s2[:,mas][:,in_mask], br[:,self.val][:,mas][:, in_mask]
        self.qa = magic** (self.qa[:,self.val][:,mas][:, in_mask])
  
        self.emus = parallel_rw_pkl(None, '6S_emulation_S2_', 'r')
    
    
    
        self.w = (np.array(self.wl))**(-self._alpha)
        self.w = self.w/(self.w.sum())
        
        patch_pixel = 240
        patches = 2400/240
        t = 1
        for i in xrange(patches):
            for j in xrange(patches):
                patch_toa =      r[t][:,i*patch_pxiel: (i+1)*patch_pxiel, j*patch_pxiel: (j+1)*patch_pxiel]
                patch_boa = self.brdf[:,i*patch_pxiel: (i+1)*patch_pxiel, j*patch_pxiel: (j+1)*patch_pxiel]
                patch_ang =    angles[:,i*patch_pxiel: (i+1)*patch_pxiel, j*patch_pxiel: (j+1)*patch_pxiel]
                
                
        
        
        
        
        
        
        
        self.patch_pixs = 300
        patches = []        
        self.inds = []
        indx, indy = self.to_cor[:2]
        self.post_uncs = []
        
        
        for i in np.arange(0, np.ceil(shape[0]/self.patch_pixs)):
            for j in np.arange(0, np.ceil(shape[1]/self.patch_pixs)):
                patch_mask = (indx>i*self.patch_pixs) & (indx<(i+1)*self.patch_pixs)\
                & (indy>j*self.patch_pixs) & (indy<(j+1)*self.patch_pixs)
                if patch_mask.sum() == 0:
                    patches.append(([0,0,0],0))
                    self.inds.append([i,j])
                else: 
                    patches.append(self._S2_opt(i, j, patch_mask))
                    self.inds.append([i,j])
                   
        self.inds = np.array(self.inds)
        paras = np.array([i[0] for i in patches])
        cost = np.array([i[1] for i in patches]).reshape(int(self.inds[:,0].max()+1),int(self.inds[:,1].max()+1))
        para_names = 'aot', 'twv', 'tco'
        masks = []
        para_maps = []
        smed_paras = []
        unc_maps = []
        
        for _ in range(3):
            mask = (np.array(paras[:,_]).reshape(cost.shape)==0) | np.isnan(cost)
            masks.append(mask)
            unc = np.array([np.r_[np.array([i[0], i[1]]), i[2][_]] for i in self.post_uncs])
            unc_map = np.zeros_like(cost)
            unc_map[:] = np.nan
            unc_map[unc[:,0].astype(int), unc[:,1].astype(int)] = unc[:,2]
            unc_maps.append(unc_map)
            w = np.zeros_like(cost)
            w[~mask] = 1./unc_map[~mask]
            para_map = np.zeros_like(cost)
            para_map[mask] = paras[:,_].reshape(cost.shape)[~mask].mean()
            para_map[~mask] = paras[:,_].reshape(cost.shape)[~mask]
            para_maps.append(para_map)
            smed_para = smoothn(para_map,s=0.05, W=w**2, isrobust=True)[0]
            smed_paras.append(smed_para)
            tifffile.imsave( self.Hfile+'%s.tiff'%para_names[_], smed_para)
        
        self.masks, self.para_maps, self.unc_maps, self.smed_paras = masks, para_maps, unc_maps, smed_paras
        return unc_maps, smed_paras, para_maps
        
        
        
    
    def S2_aot(self,):
        self.wl=0.490,0.560,0.665,0.842, 1.610,2.190, 0.865
        self.bands = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A'
        m = mgrs.MGRS()
        mg_coor = m.toMGRS(self.lat, self.lon, MGRSPrecision=4)
        self.place = mg_coor[:5]
        
        self.Hfile = os.getcwd()+'/s_data/%s/%s/%s/%d/%d/%d/0/'%(self.S2_fname[:2], self.S2_fname[2],\
                                                                 self.S2_fname[3:5], self.year, self.S2_month, self.S2_day)
        
        self.Lfile = glob.glob('%s/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'%(self.mdata,self.year, self.S2_doy,self.h,self.v))[0]

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
        
        dif = self.to_cor[3]-self.to_cor[2]
        u,d = dif.mean(axis=1)+ 3*dif.std(axis=1), dif.mean(axis=1)- 3*dif.std(axis=1)
        in_mask = np.all(np.array([(dif[i]>d[i])&(dif[i]<u[i]) for i in range(len(dif))]), axis=0)
        
        self.to_cor = self.shx[self.val][mas][in_mask], self.shy[self.val][mas][in_mask],\
        s2[:,mas][:,in_mask], br[:,self.val][:,mas][:, in_mask]
        self.qa = magic** (self.qa[:,self.val][:,mas][:, in_mask])
  
        self.emus = parallel_rw_pkl(None, '6S_emulation_S2_', 'r')
        self.w = (np.array(self.wl))**(-self._alpha)
        self.w = self.w/(self.w.sum())
        
        self.patch_pixs = 300
        patches = []        
        self.inds = []
        indx, indy = self.to_cor[:2]
        self.post_uncs = []
        
        for i in np.arange(0, np.ceil(shape[0]/self.patch_pixs)):
            for j in np.arange(0, np.ceil(shape[1]/self.patch_pixs)):
                patch_mask = (indx>i*self.patch_pixs) & (indx<(i+1)*self.patch_pixs)\
                & (indy>j*self.patch_pixs) & (indy<(j+1)*self.patch_pixs)
                if patch_mask.sum() == 0:
                    patches.append(([0,0,0],0))
                    self.inds.append([i,j])
                else: 
                    patches.append(self._S2_opt(i, j, patch_mask))
                    self.inds.append([i,j])
                   
        self.inds = np.array(self.inds)
        paras = np.array([i[0] for i in patches])
        cost = np.array([i[1] for i in patches]).reshape(int(self.inds[:,0].max()+1),int(self.inds[:,1].max()+1))
        para_names = 'aot', 'twv', 'tco'
        masks = []
        para_maps = []
        smed_paras = []
        unc_maps = []
        
        for _ in range(3):
            mask = (np.array(paras[:,_]).reshape(cost.shape)==0) | np.isnan(cost)
            masks.append(mask)
            unc = np.array([np.r_[np.array([i[0], i[1]]), i[2][_]] for i in self.post_uncs])
            unc_map = np.zeros_like(cost)
            unc_map[:] = np.nan
            unc_map[unc[:,0].astype(int), unc[:,1].astype(int)] = unc[:,2]
            unc_maps.append(unc_map)
            w = np.zeros_like(cost)
            w[~mask] = 1./unc_map[~mask]
            para_map = np.zeros_like(cost)
            para_map[mask] = paras[:,_].reshape(cost.shape)[~mask].mean()
            para_map[~mask] = paras[:,_].reshape(cost.shape)[~mask]
            para_maps.append(para_map)
            smed_para = smoothn(para_map,s=0.05, W=w**2, isrobust=True)[0]
            smed_paras.append(smed_para)
            tifffile.imsave( self.Hfile+'%s.tiff'%para_names[_], smed_para)
        
        self.masks, self.para_maps, self.unc_maps, self.smed_paras = masks, para_maps, unc_maps, smed_paras
        return unc_maps, smed_paras, para_maps

    
    
    
    
    
    def S2_cor(self,save=True):
        self.bands = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A'
        self.Hfile = os.getcwd()+'/s_data/%s/%s/%s/%d/%d/%d/0/'%(self.S2_fname[:2], self.S2_fname[2],\
                                                                 self.S2_fname[3:5], self.year, self.S2_month, self.S2_day)
        #self.patch_pixs=300
        mete = readxml('%smetadata.xml'%self.Hfile)
        self.sza = np.zeros(7)
        self.sza[:] = mete['mSz']
        self.saa = self.sza.copy()
        self.saa[:] = mete['mSa']
        shape = (10000, 10000)
        # sometimes not all of the angles are available
        #try:
        #    self.vza = (mete['mVz'])[[1,2,3,7,11,12,8],]
        #    self.vaa = (mete['mVa'])[[1,2,3,7,11,12,8],]
        #except:
        #    self.vza = np.repeat(np.nanmean(mete['mVz']), 7)
        #    self.vaa = np.repeat(np.nanmean(mete['mVa']), 7)
        
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
        #if glob.glob(self.Hfile+'aot.tiff')==[]:
        #    self.aot_map, self.twv_map = self.S2_aot()
        #else:
        self.aot_map = tifffile.imread(self.Hfile+'aot.tiff')
        self.twv_map = tifffile.imread(self.Hfile+'twv.tiff')
        self.tco_map = tifffile.imread(self.Hfile+'tco.tiff')
        gridx, gridy = np.mgrid[0:np.ceil(shape[0]/self.patch_pixs):1, 0:np.ceil(shape[1]/self.patch_pixs):1]

        self.emus = parallel_rw_pkl(None, '6S_emulation_S2_', 'r')
        self.inds = np.array(zip(gridx.ravel(), gridy.ravel()))
        self.S2_RSRs = pkl.load(open('pkls/S2_RSRs.pkl', 'r'))
        retval = parmap(self.S2_patch_correction, self.inds, nprocs=4)
        
        self.S2_cored = np.zeros((7,10000,10000))
        for _,ind in enumerate(self.inds):
            self.S2_cored[:,int(ind[0]*self.patch_pixs):int((ind[0]+1)*self.patch_pixs),\
                          int(ind[1]*self.patch_pixs):int((ind[1]+1)*self.patch_pixs)] = retval[_][1]
        if save:
            print 'Saving surface reflectance....'
            for i,band in enumerate(self.bands):
                tifffile.imsave(self.Hfile+'sur_ref_band%s.tiff'%band, self.S2_cored[i])
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
        #toa_refs[toa_refs<0] = np.nan
        px, py = np.mgrid[ix*self.patch_pixs: (ix+1)*self.patch_pixs, iy*self.patch_pixs: (iy+1)*self.patch_pixs]
        pix_lat, pix_lon = cor_inter(np.array([px.ravel(), py.ravel()]), self.dic, self.corners)
        ele = max(mean(elevation(pix_lat, pix_lon)/1000.), 0)
        

        aot = self.aot_map[int(ix),int(iy)]
        twv = self.twv_map[int(ix),int(iy)]
        tco = self.tco_map[int(ix),int(iy)]
        #tco = mean(read_net(self.year,\
        #                    self.S2_month,self.S2_day, np.arange(pix_lat.min(), pix_lat.max()+0.125, 0.125),\
        #                    np.arange(pix_lon.min(),pix_lon.max()+0.125, 0.125), dataset='tco'))
        Sur_REF = []
        for band in range(7):
            p = [aot,  twv, tco, np.sin(self.sza[band]*np.pi/180),\
               np.sin(self.vza[band]*np.pi/180), np.cos((self.saa[band]-self.vaa[band])*np.pi/180), ele]
            s = atm(p, self.S2_RSRs[band])
            a,b,c = s.outputs.atmospheric_intrinsic_reflectance,\
            s.outputs.transmittance_total_scattering.total, s.outputs.spherical_albedo.total
            cored = (np.array(toa_refs)[band]-a)/(b+(np.array(toa_refs)[band]-a)*c)
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
        #print patch_mask.shape, self.to_cor[2].shape,self.to_cor[1].shape,self.to_cor[3].shape,self.to_cor[0].shape
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
        args = angles, eles, patch_mask
        bounds = [[0.001, 2.1], 
                  [0.1, 6], 
                  [0.25, 0.45]]
        
        self.pix_num = len(self.to_cor[3][0, patch_mask])
        self.prior_mean =  np.zeros((len(self.bands), self.pix_num, 3))
        #self.prior_mean[:,:,0] = self.to_cor[3][:,patch_mask]
        self.prior_mean[:,:,:3] = np.array([self.ECWMF_aot, self.ECWMF_twv/10., self.ECWMF_tco*46.698])
        p0 =  self.ECWMF_aot, self.ECWMF_twv/10., self.ECWMF_tco*46.698
        psolve = optimize.fmin_l_bfgs_b(self.S2_cal_cost,p0, approx_grad=0, iprint=1, bounds=bounds,fprime=None,args=(args,))
        post_unc = ((1./ np.array([0.5, 0.5, 0.01])**2 + self.inv_obs_unc)**-1)**0.5
        self.post_uncs.append([indx, indy, post_unc ])
        
        return psolve

    def S2_cal_cost(self, p, args=None):
        aot550, water, ozone  = p
        angles, eles,  patch_mask = args        
        sz, sa, vz, va = angles
        
        toa_ref = self.to_cor[2][:,patch_mask]
        self.sur_ref = self.to_cor[3][:,patch_mask]
        sur_ref_sigma1 = self.sur_ref*0.05 + 0.005
        mod09_unc = np.array([0.003, 0.004, 0.004, 0.015,  0.010, 0.006, 0.015])
        sur_ref_sigma2 = np.repeat(mod09_unc, self.sur_ref.shape[1]).reshape(self.sur_ref.shape)
        self.sur_ref_sigma = np.array((sur_ref_sigma1, sur_ref_sigma2)).max(axis=0)/self.qa[:,patch_mask]
        #print self.qa.shape, patch_mask.shape
        pas = [toa_ref, aot550, water, ozone, np.sin(sz), np.sin(vz), np.cos(sa-va), eles]
        self.paras =  np.zeros((len(self.bands), self.pix_num, 8))
        for i in range(len(pas)):
            if (i>3)&(i<7):
                dat = np.repeat(pas[i], axis=0, repeats= self.pix_num).reshape(self.sur_ref.shape)
                self.paras[:,:,i] =  dat
            else:
                self.paras[:,:,i] =  pas[i]

        prior, dprior = self.prior_cost ()
        lklhood, dlklhood = self.obs_cost ()

        return prior + lklhood, dprior + dlklhood   
    
    def obs_cost(self,):
        J = 0
        self.inv_obs_unc = np.zeros(3)
        J_prime = np.zeros(3)
        for i in range(6):
            fwd, m_unc, grad = self.emus[i][0].predict(self.paras[i])
            # select from set of gradients
            g1,g2, g3 = grad[:,1],grad[:,2], grad[:,3] 
            com_unc = self.sur_ref_sigma[i]**2 #m_unc**2 + self.sur_ref_sigma[i]**2
            residual = np.array(fwd-self.sur_ref[i])
            J_prime_wrt1_i = self.w[i] * residual * g1 /com_unc
            J_prime_wrt2_i = self.w[i] * residual * g2 /com_unc
            J_prime_wrt3_i = self.w[i] * residual * g3 /com_unc
            J_i = (0.5 *self.w[i] * (residual**2)[:]/com_unc).sum()
            J += J_i
            ind_dev = np.array([J_prime_wrt1_i, J_prime_wrt2_i, J_prime_wrt3_i])[:,:].sum(axis=1)
            J_prime += ind_dev

            self.inv_obs_unc += (ind_dev**2)*((self.w[i] * 1./ com_unc).sum())
        

        return 1.*J, J_prime

    def prior_cost(self,):
        x = self.paras[:,:,1:4]
        self.prior_sigma = np.zeros_like(x)
        #self.prior_sigma[:,:,0] = self.sur_ref_sigma
        self.prior_sigma[:,:,0:3] = np.array([0.5, 0.5, 0.01])
        cost = 0.5*( x - self.prior_mean)**2/self.prior_sigma**2
        dcost = (( x - self.prior_mean)/self.prior_sigma**2).sum(axis=(0,1))

        return cost.sum(), dcost