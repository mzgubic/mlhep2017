import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

#ds_train = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_train_LVL2_float32.hdf')
#ds_train = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_train_extended_float32.hdf')
#ds_train = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_test_TrackingAdded3.hdf')
#ds_train = pd.read_csv('/data/atlas/atlasdata/zgubic/MLHEP17/ElecPathAdded/DS_2_train_brick59_ElecPathAdded.csv')
ds_train = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_train_ElecPathAdded.hdf')


print 'loaded'

print ds_train.columns

#for var in ['min_sep', 'cone1', 'cone2']:
for var in ['e_dR', 'e_dT', 'e_dZ']:
    fig, ax = plt.subplots(1, figsize=(15, 8))
    print var
    ax.hist(ds_train[var], alpha=0.5, bins=100, normed=True)
    ax.set_title(var)
    fig.savefig('figures/'+var+'.pdf')
