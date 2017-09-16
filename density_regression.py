import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib import gridspec,cm, colors
from scipy.stats import gaussian_kde
import scipy
import multiprocessing
from functools import partial
def cal_density(ind, mods=None, sens=None):
    xy = np.vstack([mods[ind], sens[ind]])
    z = gaussian_kde(xy)(xy)
    return z

def density_regression(modis_refs, sent_refs, cmap = cm.get_cmap('YlGnBu'),\
                 titles=None, xlabel="MOD reflectance", ylabel='SEN reflectance',\
                 three_sigma=0, figsize=(24,12), rows=2, columns=4):
    plot_config()
    fig = plt.figure()
    gs = gridspec.GridSpec(rows, columns)  # generate a grid space
    fig = plt.figure(figsize=figsize)
    
    par = partial(cal_density, mods=modis_refs, sens=sent_refs)
    pool = multiprocessing.Pool(processes = 7)
    zs = pool.map(par, range(len(modis_refs)))
    pool.close()
    pool.join()
    
    for i in range(len(modis_refs)):
        ax = fig.add_subplot(gs[i])
        mod = modis_refs[i]
        sen = sent_refs[i]
        if three_sigma==1:

            dis = mod-sen
            std = np.std(dis)
            mean = np.mean(dis)
            inl = (dis > mean-3*std)&(dis < mean+3*std)
            mod = mod[inl]
            sen = sen[inl]
        else:
            pass

        mval = np.nanmax([mod, sen])
        fit = np.polyfit(mod, sen,1)
        fit_fn = np.poly1d(fit)
        #xy = np.vstack([mod, sen])
        #z = gaussian_kde(xy)(xy)
        ax.scatter(mod, sen, c=zs[i], s=4, edgecolor='',norm=colors.LogNorm(vmin=zs[i].min(), vmax=zs[i].max()*1.2), cmap = cmap,
                  rasterized=True)
        ax.plot([0,1],[0.,1], '--',linewidth=0.5)
        ax.plot(np.arange(0,1,0.1), fit_fn(np.arange(0,1,0.1)), '--', color='grey')
        slope,inter, rval, pval, std = scipy.stats.linregress(mod, sen)
        ax.set_title('%s'%titles[i])
        ax.text(mval*(4./6.),mval*(1.5/6.),'a = %.03f'%(slope)+r"${\times}$"+'b + '+ '%.03f \n'%(inter)+ r"${r^2}$"+': %.03f' %(rval), 
            )
        ax.set_xlim(0,mval)
        ax.set_ylim(0,mval)
        ax.set_yticks(np.arange(0,mval+0.1,mval/5.))
        ax.set_xticks(np.arange(0,mval+0.1,mval/5.))
        if i==4:
            ax.set_xlabel ( xlabel )
            ax.set_ylabel ( ylabel )
        
    plt.tight_layout()