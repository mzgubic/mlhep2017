import numpy as np
import pandas as pd

bits = 32

def var_to_type(var):
    if bits == 32:
        return np.float32

    if bits == 16:
        return np.float16

tag = 'ElecPathAdded'
whichvars = ['e_dR', 'e_dT', 'e_dZ']

#for which in ['train', 'test']:
for which in ['train']:
    
    # start adding the columns
    print 'loading original'
    original = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_'+which+'_extended_float32.hdf')

    print 'loading new'
    print tag
    new = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_'+which+'_'+tag+'.hdf', usecols=whichvars)
    new.set_index('index', inplace=True)

    vars2 = whichvars[:]
    for var in vars2:
        print 'adding', var

        print 
        print original.head()
        print original.columns
        print new[var].head()
        print 
        original[var] = new[var].astype(var_to_type(var))
    
    print 
    print original.head()
    print 
    
    original.to_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_'+which+'_LVL2_float'+str(bits)+'.hdf', 'features')
