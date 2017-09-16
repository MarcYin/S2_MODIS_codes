import numpy as np
def read_aeronet(place, doy, year):
    import glob
    files = glob.glob('aeronets/%d*_%s*'%(year-2000, place))
    if files ==[]:
        print 'no such file: ', 'aeronets/%d*_%s*'%(year-2000, place)
        return None
    else:
        fname = files[0]
        data = []
        with open(fname, 'r') as inF:
            for line in inF:
                if 'Date(dd-mm-yyyy)' in line:
                    title = np.array((line.split(',')[2:20]))
                if 'Level_1.5' in line:
                    data.append(line.split(',')[2:20])
        vals = np.array(data)
        ski_m = vals==np.array('N/A', dtype='|S10')
        vals[ski_m] = np.nan
        vals = vals.astype(float)
        col_m = np.all(np.isnan(vals), axis=0)
        
        vals[:,~col_m]
        aots = np.vstack((title[~col_m], vals[:,~col_m]))
        jdays = np.floor(aots[1:,0].astype(float)).astype(int)
        d_m = jdays==doy
        aot = aots[1:,][d_m]
        return title[~col_m], aot.astype('float')