import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib import gridspec,cm, colors
from scipy.stats import gaussian_kde

def plot_config ():
    """Update the MPL configuration"""
    config_json='''{
            "lines.linewidth": 2.0,
            "axes.edgecolor": "#bcbcbc",
            "patch.linewidth": 0.5,
            "legend.fancybox": true,
            "axes.color_cycle": [
                "#FC8D62",
                "#66C2A5",
                "#8DA0CB",
                "#E78AC3",
                "#A6D854",
                "#FFD92F",
                "#E5C494",
                "#B3B3B3"
            ],
            "axes.facecolor": "w",
            "axes.labelsize": "large",
            "axes.grid": false,
            "patch.edgecolor": "#eeeeee",
            "axes.titlesize": "x-large",
            "svg.embed_char_paths": "path",
            "xtick.direction" : "out",
            "ytick.direction" : "out",
            "xtick.color": "#262626",
            "ytick.color": "#262626",
            "axes.edgecolor": "#262626",
            "axes.labelcolor": "#262626",
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12
            
    }
    '''
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['xtick.minor.size'] = 10
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['ytick.minor.size'] = 10
    plt.rcParams['ytick.minor.width'] = 0.5
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']

    s = json.loads ( config_json )
    plt.rcParams.update(s)
    plt.rcParams["axes.formatter.limits"] = [-5,5]
    

def pretty_axes( ax ):
    """This function takes an axis object ``ax``, and makes it purrty.
    Namely, it removes top and left axis & puts the ticks at the
    bottom and the left"""

    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(True)  
    ax.spines["right"].set_visible(False)              
    ax.spines["left"].set_visible(True)  

    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    loc = plt.MaxNLocator( 6 )
    ax.yaxis.set_major_locator( loc )
    ax.xaxis.set_major_locator( loc )
    

    ax.tick_params(axis="both", which="both", bottom="on", top="off",  
            labelbottom="on", left="on", right="off", labelleft="on")
def linear_plots(modis_refs, sent_refs):
    plot_config()
    fig = plt.figure()
    gs = gridspec.GridSpec(2,4)  # generate a grid space
    fig = plt.figure(figsize=(24,12))
    names = ('SB2_MB3,SB3_MB4,SB4_MB1,SB8_MB2,SB8A_MB2,SB11_MB6,SB12_MB7').split(',')
    cmap = cm.get_cmap('YlGnBu')
    psfsolve = []
    for i in range(len(modis_refs)):
        ax = fig.add_subplot(gs[i])
        #data = np.array(to_regs[4+i])


        mod = modis_refs[i]
        sen = sent_refs[i]

        #dis = mod-sen
        #std = np.std(dis)
        #mean = np.mean(dis)
        #inl = (dis > mean-3*std)&(dis < mean+3*std)
        #s = mod[inl]
        #m = sen[inl]

        mval = np.nanmax([mod, sen])
        fit = np.polyfit(mod, sen,1)
        fit_fn = np.poly1d(fit)
        xy = np.vstack([mod, sen])
        z = gaussian_kde(xy)(xy)
        ax.scatter(mod, sen, c=z, s=4, edgecolor='',norm=colors.LogNorm(vmin=z.min(), vmax=z.max()*1.2), cmap = cmap)
        ax.plot([0,1],[0.,1], '--',linewidth=0.5)
        ax.plot(np.arange(0,1,0.1), fit_fn(np.arange(0,1,0.1)), '--', color='grey')
        slope,inter, rval, pval, std = r = scipy.stats.linregress(mod, sen)
        ax.set_title('%s'%names[i])
        ax.text(mval*(4./6.),mval*(1.5/6.),'slope: %02f \nintercept: %02f \ncorrelation: %02f \nstderr: %02f'%(slope,inter, rval, std), 
            )
        pretty_axes(ax)
        ax.set_xlim(0,mval)
        ax.set_ylim(0,mval)
        ax.set_yticks(np.arange(0,mval+0.1,mval/5.))
        ax.set_xticks(np.arange(0,mval+0.1,mval/5.))
        if i==4:
            ax.set_xlabel ( "MOD reflectance")
            ax.set_ylabel ( "SEN reflectance")
        psfsolve.append([slope,inter])
    plt.tight_layout()