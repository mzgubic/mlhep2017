import numpy as np
import pandas as pd

bits = 32

def var_to_type(var):
    if bits == 32:
        return np.float32

    if bits == 16:
        return np.float16

tags = ['MihaAdded', 'TrackingAdded3']
whichvarses = [ ['index', 'grid_value', 'smoothgrid_value'],
                ['index'] + ['min_sep', 'cone1', 'cone2'] + ['cone'+var+'_'+str(i)+updown for var in ['1', '2'] for i in range(1,5) for updown in ['up', 'down']] + ['d'+var+str(i)+updown for var in ['T', 'R'] for i in range(1,5) for updown in ['up', 'down']]
              ]

#for which in ['train', 'test']:
for which in ['train']:
    
    # start adding the columns
    print 'loading original'
    original = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_'+which+'.hdf')
    original['X'] = original['X'].astype(np.float32)
    original['Y'] = original['Y'].astype(np.float32)
    original['Z'] = original['Z'].astype(np.float32)
    original['TX'] = original['TX'].astype(np.float32)
    original['TY'] = original['TY'].astype(np.float32)
    original['chi2'] = original['chi2'].astype(np.float32)
    original['brick_number'] = original['brick_number'].astype(np.int8)

    for tag, whichvars in zip(tags, whichvarses):
        print tag
        print 'loading new'
        new = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_'+which+'_'+tag+'.hdf', usecols=whichvars)
        new.set_index('index', inplace=True)
    
        vars2 = whichvars[:]
        vars2.remove('index')
        for var in vars2:
            print 'adding', var
            original[var] = new[var].astype(var_to_type(var))
    
    print 
    print original.head()
    print 
    
    original.to_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_'+which+'_extended_float'+str(bits)+'.hdf', 'features')
    #original.to_csv('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_'+which+'_extended.csv')
