import os
import sys
sys.path.insert(0,'python')
import multiprocessing
from functools import partial
import gdal
from L_geo import *
from get_brdf import *
from geo_trans import *
from Dload_Finder import file_finder
import datetime
import glob
from cloud import get_cloud_mask
from classification import *
from scipy import ndimage, signal, optimize
import scipy
from get_wrs import *
import tifffile
directory = os.getcwd()+'/'
import cPickle as pkl
from smoothn import *
import scipy.stats as stats
from lhd import lhd

def create_training_set ( parameters, minvals, maxvals, n_train=200 ):
    """Creates a traning set for a set of parameters specified by 
    ``parameters`` (not actually used, but useful for debugging
    maybe). Parameters are assumed to be uniformly distributed
    between ``minvals`` and ``maxvals``. ``n_train`` input parameter
    sets will be produced, and returned with the actual distributions
    list. The latter is useful to create validation sets.
    Parameters
    -------------
    parameters: list
        A list of parameter names
    minvals: list
        The minimum value of the parameters. Same order as ``parameters``
    maxvals: list
        The maximum value of the parameters. Same order as ``parameters``
    n_train: int
        How many training points to produce
    Returns
    ---------
    The training set and a distributions object that can be used by
    ``create_validation_set``-- Jose:
    https://github.com/jgomezdans/gp_emulator/blob/master/gp_emulator/emulation_helpers.py
    """

    distributions = []
    for i,p in enumerate(parameters):
        distributions.append ( stats.uniform ( loc=minvals[i], \
                            scale=(maxvals[i]-minvals[i] ) ) )
    samples = lhd ( dist=distributions, size=n_train )
    return samples, distributions


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



class PSF_optimization(object):
    def __init__(self, year, month, day, doy, lat, lon, sate):
        self.year = year
        self.doy = doy
        self.month = month
        self.day= day
        self.lat = lat
        self.lon = lon
        self.sate = sate 
        self.Lx = None
        self.LY = None
        self.Hx = None
        self.Hy = None
        self.H_data = None
        self.L_data = None
        self.sza = None
        self.saa = None
        self.vza = None
        self.vaa = None
        self.retval = None
        self.BRDF_16_days =None
        self.composite_brdf = None
        self.base_mask = None
        
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
            return None  

    def update_mask(self, base_map,forward_8_days, backward_8_days, pre_mask, qa=0, start_point=4, thresh=0.005):

        for i in range(8):
            qa_fo_mask = forward_8_days[-(i+1),1,:]<=qa
            qa_ba_mask = backward_8_days[i,1,:]<=qa
            dif = np.abs(forward_8_days[i,0,:] - backward_8_days[-(i+1),0,:])
            dif_mask = (dif/np.abs(forward_8_days[i,0,:])) <thresh
            non_negative_mask = (forward_8_days[i,0,:]>0) & (backward_8_days[-(i+1),0,:]>0)

            i_day_mask = (qa_fo_mask | qa_ba_mask)&dif_mask&non_negative_mask
            current_mask = (~pre_mask)&i_day_mask
            base_map[current_mask] = i+start_point
            pre_mask = pre_mask|current_mask  
        return base_map, pre_mask
    
    def compositing(self, angles, thre=0.005):
        BRDF_16_days = np.array([get_brdf_six(Lfile,angles,bands=(7,), \
                                              flag=None, Linds= self.L_inds) for Lfile in self.Lfiles]).squeeze()

        backward_8_days,mid_day, forward_8_days = BRDF_16_days[:8], BRDF_16_days[8], BRDF_16_days[9:]

        qa_mask_0 = np.all(BRDF_16_days[:,1,:]<=0, axis=0)
        stable_mask_0 =  BRDF_16_days[:,0,:].std(axis=0)<0.005
        filter_mask_0 = qa_mask_0&stable_mask_0

        qa_mask_1 = np.all(BRDF_16_days[:,1,:]<=1, axis=0)
        stable_mask_1 =  BRDF_16_days[:,0,:].std(axis=0)<0.004
        filter_mask_1 = qa_mask_1&stable_mask_1&np.all(BRDF_16_days[:,0,:]>=0, axis=0)


        base_mask = np.zeros(len(self.L_inds[0]))
        current_mask = filter_mask_0
        base_mask[current_mask] = 1
        previous_mask = current_mask

        current_mask = (~previous_mask)&filter_mask_1 #| ((mid_day[1]==0)&stable_mask)
        base_mask[current_mask] = 2
        previous_mask = previous_mask|current_mask

        current_mask = (~previous_mask)&((mid_day[1]==0)&stable_mask_0)
        base_mask[current_mask] = 3
        previous_mask = previous_mask|current_mask

        base_mask, pre_mask = self.update_mask(base_mask,forward_8_days, backward_8_days, previous_mask, qa=0, start_point=4)
        base_mask, _mask = self.update_mask(base_mask, forward_8_days, backward_8_days, pre_mask, qa=1, start_point=12,thresh=0.00025)

        composite_brdf = np.zeros(base_mask.shape)
        composite_brdf[:]=np.nan
        composite_brdf[base_mask==1] = mid_day[0][base_mask==1]
        composite_brdf[base_mask==2] = mid_day[0][base_mask==2]
        composite_brdf[base_mask==3] = mid_day[0][base_mask==3]

        for i in range(8):
            composite_brdf[base_mask==i+4] = forward_8_days[i][0][base_mask==i+4]
        for i in range(8):
            composite_brdf[base_mask==i+12] = backward_8_days[-(i+1)][0][base_mask==i+12]

        return BRDF_16_days, composite_brdf, base_mask
    
    def L8_PSF_optimization(self):
        self.h,self.v = mtile_cal(self.lat, self.lon)
        pr=get_wrs(self.lat, self.lon)
        self.path, self.row = pr[0]['path'],pr[0]['row']
        #self.Hfiles = glob.glob(directory +'l_data/LC8%03d%03d%d*LGN00_sr_band1.tif'%(self.path, self.row, self.year))
        self.Hfile = directory +'l_data/LC8%03d%03d%d%03dLGN00_toa_'%(self.path, self.row, self.year, self.doy)
        #Lfile = glob.glob('m_data/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'%(year,doy,h,v))[0]
        self.Lfiles = [glob.glob('m_data/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'%(self.year,i,self.h,self.v))[0] for i in range(self.doy-8, self.doy+9)]

        if self.read_meta(self.Hfile, self.path, self.row)==None:  
            print 'Too much cloud!!'
        else:
            self.sza, self.saa, self.vza, self.vaa, self.dic, self.corners = self.read_meta(self.Hfile, self.path, self.row)
            self.L_inds, self.H_inds = ML_geo_trans(self.lat, self.lon, self.dic, self.corners)
            self.Lx, self.Ly = self.L_inds
            self.Hx, self.Hy = self.H_inds

            tems = np.zeros((3,6))
            tems[0,:] = self.sza
            tems[1,:] = self.vza
            tems[2,:] = self.vaa - self.saa
            angles = (tems[0][-1], tems[1][-1], tems[2][-1])

            #self.BRDF_16_days, self.composite_brdf, self.base_mask = self.compositing(angles, thre=0.005)
            
            if glob.glob(self.Lfiles[8][:-3]+'L8.16days.pkl')==[]:
                self.BRDF_16_days = np.array([get_brdf_six(Lfile,angles,bands=(7,), \
                                                  flag=None, Linds= self.L_inds) for Lfile in self.Lfiles]).squeeze()
                valid_range = (self.BRDF_16_days[:,0,:]>=0)&(self.BRDF_16_days[:,0,:]<=1)
                magic = 0.618034
                test = self.BRDF_16_days[:,0,:].copy()
                test[~valid_range]=np.nan
                W = magic**self.BRDF_16_days[:,1,:]
                W[self.BRDF_16_days[:,1,:]>1]=0
                #smothed = smoothn(test, axis=0, isrobust=1, W =W, s=1)[0]
                pkl.dump(self.BRDF_16_days, open(self.Lfiles[8][:-3]+'L8.16days.pkl', 'w'))
                #pkl.dump(smothed, open(self.Lfiles[8][:-3]+'L8.16days.smoothed.pkl', 'w'))
            else:
                self.BRDF_16_days = pkl.load(open(self.Lfiles[8][:-3]+'L8.16days.pkl', 'r'))
                #smothed = pkl.load(open(self.Lfiles[8][:-3]+'L8.16days.smoothed.pkl', 'r'))

            cloud = gdal.Open(self.Hfile[:-5]+'_cfmask.tif').ReadAsArray()
            cl_mask = cloud==4 # cloud pixels; strictest way is to set the clear pixels with cloud==0
            struct = ndimage.generate_binary_structure(2, 2)
            dia_cloud = ndimage.binary_dilation(cl_mask, structure=struct, iterations=20).astype(cl_mask.dtype)

            self.H_data = gdal.Open(self.Hfile+'band%d.tif'%7).ReadAsArray()*0.0001
            mask = ~(self.H_data<0).astype('bool')
            small_mask = ndimage.binary_erosion(mask, structure=struct, iterations=20).astype(mask.dtype)
            self.val_mask = (~dia_cloud)&small_mask
            
            
            self.L_data = np.zeros(self.BRDF_16_days[8,0,:].shape[0])
            self.L_data[:] = np.nan
            self.L_data[self.BRDF_16_days[8,1,:]==0] = self.BRDF_16_days[8,0,:][self.BRDF_16_days[8,1,:]==0]
            #args = s, self.L_data, 


            avker = np.ones((40,40))
            navker = avker/avker.sum()
            self.s = signal.fftconvolve(self.H_data, navker, mode='same')
            self.s[~self.val_mask]=np.nan
            
            min_val = [-40,-40]
            max_val = [40,40]
            
            ps, distributions = create_training_set([ 'xs', 'ys'], min_val, max_val, n_train=50)
            solved = parmap(self.op1, ps, nprocs=10)    
            paras, costs = np.array([i[0] for i in solved]),np.array([i[1] for i in solved])
            xs, ys = paras[costs==costs.min()][0]
   
            if costs.min()<0.1:
                min_val = [5,5, -15,xs-5,ys-5]
                max_val = [100,100, 15, xs+5,ys+5]

                self.bounds = [5,100],[5,100],[-15,15],[xs-5,xs+5],[ys-5, ys+5]

                ps, distributions = create_training_set(['xstd', 'ystd', 'ang', 'xs', 'ys'], min_val, max_val, n_train=50)

                #ps = zip(xstd.ravel(), ystd.ravel())
                print 'Start solving...'

                self.solved = parmap(self.op, ps, nprocs=10)

                #costs = np.array([i[1] for i in self.solved])
                #paras = np.array([i[0] for i in self.solved])
                #min_mask = costs==costs.min()
                #min_costs = costs.min()
                #para = paras[min_mask]
                print self.solved

                return self.solved, self.BRDF_16_days 
            else:
                print 'Cost is too large, plese check!', xs, ys, costs.min()
                return [[xs, ys, costs.min()], self.BRDF_16_days ] 
            
            '''
            xs, ys = retval[0]
            self.xs, self.ys = xs, ys
            self.p0 = 13, 32, 4, xs, ys
            self.bounds = [5,100],[5,100],[-15,15],[xs-5,xs+5],[ys-5, ys+5]
            #args = H_data, L_data, val_mask
            #optimize.fmin_l_bfgs_b(cost2, p0, approx_grad=1, iprint=-1, args=args, bounds=bounds)

            ps = [(5, 20, 10, xs, ys),(10, 20, 10, xs, ys), (20, 20, 10, xs, ys), (30, 20, 10, xs, ys),(40, 20, 10, xs, ys),
            (10, 5, 10, xs, ys),(10, 10, 10, xs, ys), (10, 20, 10, xs, ys), (10, 30, 10, xs, ys),(10, 40, 10, xs, ys),
            (10, 20, 0, xs, ys),(10, 20, 5, xs, ys), (10, 20, 10, xs, ys), (10, 30, 15, xs, ys),(10, 30, -10, xs, ys)]
            #retval = optimize.fmin_l_bfgs_b(cost2, p0, approx_grad=1, iprint=-1, args=args, bounds=bounds)
            #par = partial(self._op)
            #pool = multiprocessing.Pool(processes = 15)
            #self.solved = pool.map(op, ps)
            #pool.close()
            #pool.join()

            self.solved = parmap(self.op, ps, nprocs=5)
            
            return self.solved, self.BRDF_16_days, self.L_data, self.base_mask
            '''
        
    
    def S2_PSF_optimization(self):
        self.h,self.v = mtile_cal(self.lat, self.lon)
        m = mgrs.MGRS()
        mg_coor = m.toMGRS(self.lat, self.lon, MGRSPrecision=4)
        self.place = mg_coor[:5]
        #self.Hfiles = glob.glob(directory +'l_data/LC8%03d%03d%d*LGN00_sr_band1.tif'%(self.path, self.row, self.year))
        self.Hfile = os.getcwd()+'/s_data/%s/%s/%s/%d/%d/%d/0/'%(mg_coor[:2], mg_coor[2], mg_coor[3:5], self.year, self.month, self.day)
        #Lfile = glob.glob('m_data/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'%(year,doy,h,v))[0]
        self.doy = datetime .datetime(self.year, self.month, self.day).timetuple().tm_yday
        self.Lfiles = [glob.glob('m_data/MCD43A1.A%d%03d.h%02dv%02d.006.*.hdf'%(self.year,i,self.h,self.v))[0] for i in range(self.doy-8, self.doy+9)]
        
        if glob.glob(self.Hfile+'cloud.tif')==[]:
            cl = classification(fhead = self.Hfile, bands = (2,3,4,8,11,12,13), bounds = None)
            cl.Get_cm_p()
            self.cloud = cl.cm.copy()
            tifffile.imsave(self.Hfile+'cloud.tif', self.cloud.astype(int))
            self.H_data = np.repeat(np.repeat(cl.b12, 2, axis=1), 2, axis=0)
            del cl
        else:
            b12 = gdal.Open(self.Hfile+'B12.jp2').ReadAsArray()*0.0001
            self.H_data = np.repeat(np.repeat(b12, 2, axis=1), 2, axis=0)
            self.cloud = tifffile.imread(self.Hfile+'cloud.tif').astype(bool)
        cloud_cover = 1.*self.cloud.sum()/self.cloud.size
        cloud_cover = 1.*self.cloud.sum()/self.cloud.size
        if cloud_cover > 0.2:  
            print 'Too much cloud, cloud proportion: %.03f !!'%cloud_cover
        else:
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

            angles = (self.sza[-1], self.vza[-1], (self.vaa - self.saa)[-1])
            
            if glob.glob(self.Lfiles[8][:-3]+'L8.16days.pkl')==[]:
                self.BRDF_16_days = np.array([get_brdf_six(Lfile,angles,bands=(7,), \
                                                  flag=None, Linds= self.L_inds) for Lfile in self.Lfiles]).squeeze()
                valid_range = (self.BRDF_16_days[:,0,:]>=0)&(self.BRDF_16_days[:,0,:]<=1)
                magic = 0.618034
                test = self.BRDF_16_days[:,0,:].copy()
                test[~valid_range]=np.nan
                W = magic**self.BRDF_16_days[:,1,:]
                W[self.BRDF_16_days[:,1,:]>1]=0
                #smothed = smoothn(test, axis=0, isrobust=1, W=W, s=1)[0]
                pkl.dump(self.BRDF_16_days, open(self.Lfiles[8][:-3]+'L8.16days.pkl', 'w'))
                #pkl.dump(smothed, open(self.Lfiles[8][:-3]+'L8.16days.smoothed.pkl', 'w'))
            
            else:
                self.BRDF_16_days = pkl.load(open(self.Lfiles[8][:-3]+'L8.16days.pkl', 'r'))
                #smothed = pkl.load(open(self.Lfiles[8][:-3]+'L8.16days.smoothed.pkl', 'r'))   
            
            struct = ndimage.generate_binary_structure(2, 2)
            dia_cloud = ndimage.binary_dilation(self.cloud, structure=struct, iterations=60).astype(self.cloud.dtype)

            mask = ~(self.H_data<=0).astype('bool')
            small_mask = ndimage.binary_erosion(mask, structure=struct, iterations=60).astype(mask.dtype)
            self.val_mask = (~dia_cloud)&small_mask

            self.L_data = np.zeros(self.BRDF_16_days[8,0,:].shape[0])
            self.L_data[:] = np.nan
            self.L_data[self.BRDF_16_days[8,1,:]==0] = self.BRDF_16_days[8,0,:][self.BRDF_16_days[8,1,:]==0]
            #args = s, self.L_data, 
            
            avker = np.ones((120,120))
            navker = avker/avker.sum()
            self.s = signal.fftconvolve(self.H_data, navker, mode='same')
            self.s[~self.val_mask]=np.nan

            min_val = [-100,-100]
            max_val = [100,100]
            
            ps, distributions = create_training_set([ 'xs', 'ys'], min_val, max_val, n_train=50)
            solved = parmap(self.op1, ps, nprocs=10)    
            paras, costs = np.array([i[0] for i in solved]),np.array([i[1] for i in solved])
            xs, ys = paras[costs==costs.min()][0]
            
            if costs.min()<0.1:
                min_val = [5,5, -15,xs-5,ys-5]
                max_val = [100,100, 15, xs+5,ys+5]

                self.bounds = [5,100],[5,100],[-15,15],[xs-5,xs+5],[ys-5, ys+5]

                ps, distributions = create_training_set(['xstd', 'ystd', 'ang', 'xs', 'ys'], min_val, max_val, n_train=50)

                #ps = zip(xstd.ravel(), ystd.ravel())
                print 'Start solving...'

                self.solved = parmap(self.op, ps, nprocs=10)

                #costs = np.array([i[1] for i in self.solved])
                #paras = np.array([i[0] for i in self.solved])
                #min_mask = costs==costs.min()
                #min_costs = costs.min()
                #para = paras[min_mask]
                print self.solved
                return self.solved, self.BRDF_16_days 
            else:
                print 'Cost is too large, plese check!', xs, ys, costs.min()
                return [[xs, ys, costs.min()], self.BRDF_16_days ]
                
            '''
            xs, ys = retval[0]
            self.xs, self.ys = xs, ys
            self.p0 = 13, 32, 4, xs, ys
            self.bounds = [5,100],[5,100],[-15,15],[xs-5,xs+5],[ys-5, ys+5]
            #args = H_data, L_data, val_mask
            #optimize.fmin_l_bfgs_b(cost2, p0, approx_grad=1, iprint=-1, args=args, bounds=bounds)

            ps = [(5, 20, 10, xs, ys),(10, 20, 10, xs, ys), (20, 20, 10, xs, ys), (30, 20, 10, xs, ys),(40, 20, 10, xs, ys),
            (10, 5, 10, xs, ys),(10, 10, 10, xs, ys), (10, 20, 10, xs, ys), (10, 30, 10, xs, ys),(10, 40, 10, xs, ys),
            (10, 20, 0, xs, ys),(10, 20, 5, xs, ys), (10, 20, 10, xs, ys), (10, 30, 15, xs, ys),(10, 30, -10, xs, ys)]
            #retval = optimize.fmin_l_bfgs_b(cost2, p0, approx_grad=1, iprint=-1, args=args, bounds=bounds)
            #par = partial(self._op)
            #pool = multiprocessing.Pool(processes = 15)
            #self.solved = pool.map(op, ps)
            #pool.close()
            #pool.join()
            
            self.solved = parmap(self.op, ps, nprocs=5)

            return self.solved, self.BRDF_16_days, self.L_data, self.base_mask
            '''
                        
    def _cost1(self, shifts):
        xs, ys = shifts 
        val = (self.Hx+xs<min(self.s.shape[0], 10000))&(self.Hy+ys<min(self.s.shape[1], 10000))
        shx, shy = (self.Hx+xs).astype(int), (self.Hy+ys).astype(int)
        Lvals, Hvals = self.L_data[val], self.s[shx[val], shy[val]]
        Lvals[np.isnan(Lvals)],Hvals[np.isnan(Hvals)]=-9999999, -9999999
        mas = (Lvals>0)&(Lvals<1)&(Hvals>0)&(Hvals<1)
        try:
            r = scipy.stats.linregress(Lvals[mas], Hvals[mas])
            costs = abs(1-r.rvalue)
        except:
            costs = 100000000000
        return costs 

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

    def _cost2(self, para):
        xstd,ystd,angle, xs, ys = para 
        G = self.gaussian(xstd,ystd,angle,True)                              
        ss = signal.fftconvolve(self.H_data, G, mode='same')
        # remove the cloud pixel
        ss[~self.val_mask]=np.nan
        val = (self.Hx+xs<self.H_data.shape[0])&(self.Hy+ys<self.H_data.shape[1])
        shx, shy = (self.Hx+xs).astype(int), (self.Hy+ys).astype(int)
        Lvals, Hvals = self.L_data[val], ss[shx[val], shy[val]]
        Lvals[np.isnan(Lvals)],Hvals[np.isnan(Hvals)]=-9999999, -9999999
        mas = (Lvals>0)&(Lvals<1)&(Hvals>0)&(Hvals<1)
        try:
            r = scipy.stats.linregress(Lvals[mas], Hvals[mas])
            costs = abs(1-r.rvalue)
        except:
            costs = 100000000000
        return costs 
    
    def op1(self, p0):
        #p0 =  ps[ind
        #args = self.H_data, self.L_data, self.val_mas
        return optimize.fmin(self._cost1, p0, full_output=1, maxiter=100, maxfun=150)

    def op(self, p0):
        #p0 =  ps[ind]
        #args = self.H_data, self.L_data, self.val_mask
        return optimize.fmin_l_bfgs_b(self._cost2, p0, approx_grad=1, iprint=-1, bounds=self.bounds,maxiter=100, maxfun=150)

    