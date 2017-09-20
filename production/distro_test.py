import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

#ds_train = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_train_ElecPathAdded.hdf')
#ds_train = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_train_extended_float32.hdf')
#ds_train = pd.read_csv('/data/atlas/atlasdata/zgubic/MLHEP17/ElecPathAdded/DS_2_train_brick59_ElecPathAdded.csv')
ds_train = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_train_LVL2_float32.hdf')

print ds_train.head()
print ds_train.columns

print 'loaded'
sig = ds_train[ds_train.signal == 1]
bkg = ds_train[ds_train.signal == 0]


for var in ['grid_value', 'smoothgrid_value', 'min_sep', 'cone1', 'cone2'] + ['d'+var+str(i)+updown for var in ['T', 'R'] for i in range(1,5) for updown in ['up', 'down']] + ['cone'+var+'_'+str(i)+updown for var in ['1', '2'] for i in range(1,5) for updown in ['up', 'down']]:
#for var in ['e_dR', 'e_dT', 'e_dZ']:
#for var in ['min_sep', 'cone1', 'cone2']:
    fig, ax = plt.subplots(1, figsize=(15, 8))
    print var
    ax.hist(bkg[var], alpha=0.5, bins=100, normed=True)
    ax.hist(sig[var], alpha=0.5, bins=100, normed=True)
    ax.set_title(var)
    fig.savefig('figures/'+var+'.pdf')
    os.system('evince figures/'+var+'.pdf')
