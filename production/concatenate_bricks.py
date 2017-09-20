import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# first construct the new file containing only the new variable and the index

#tag = 'TrackingAdded'
#whichvars = ['index'] + ['d'+var+str(i)+updown for var in ['T', 'R'] for i in range(1,5) for updown in ['up', 'down']]

#tag = 'TrackingAdded3'
#whichvars = ['index'] + ['min_sep', 'cone1', 'cone2'] + ['cone'+var+'_'+str(i)+updown for var in ['1', '2'] for i in range(1,5) for updown in ['up', 'down']] + ['d'+var+str(i)+updown for var in ['T', 'R'] for i in range(1,5) for updown in ['up', 'down']]

#tag = 'MihaAdded'
#whichvars = ['index', 'grid_value', 'smoothgrid_value']

tag = 'ElecPathAdded'
whichvars = ['index', 'e_dR', 'e_dT', 'e_dZ']




# and do the concatenation
new_train = pd.read_csv('/data/atlas/atlasdata/zgubic/MLHEP17/'+tag+'/DS_2_train_brick1_'+tag+'.csv')[whichvars]
new_test = pd.read_csv('/data/atlas/atlasdata/zgubic/MLHEP17/'+tag+'/DS_2_test_brick1_'+tag+'.csv')[whichvars]

for ib in range(2, 101):
    print ib
    train = pd.read_csv('/data/atlas/atlasdata/zgubic/MLHEP17/'+tag+'/DS_2_train_brick'+str(ib)+'_'+tag+'.csv')[whichvars]
    test = pd.read_csv('/data/atlas/atlasdata/zgubic/MLHEP17/'+tag+'/DS_2_test_brick'+str(ib)+'_'+tag+'.csv')[whichvars]
    new_train = pd.concat([new_train, train])
    new_test = pd.concat([new_test, test])


new_train.to_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_train_'+tag+'.hdf', tag)
new_test.to_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_test_'+tag+'.hdf', tag)
