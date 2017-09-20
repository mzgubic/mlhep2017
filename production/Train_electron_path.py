print 'loading modules'
import numpy as np
import pandas as pd
import time
import pickle
import os
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

##############################
print  'loading the datasets'
###############################

# import the training set
D_train = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_train_extended_float32.hdf')

# and construct the X and Y values
Y_train = D_train['signal']

X_train = D_train.drop('index', axis=1).drop('event_id', axis=1).drop('signal', axis=1).drop('brick_number', axis=1)

###############################
print 'train on the dataset'
###############################

n_est = 128
model = RandomForestClassifier(n_estimators=n_est, max_depth=3, n_jobs=8)

t1= time.time()
model.fit(X_train, Y_train)
print 'n_est:', n_est
print 'time taken:', time.time() - t1

###############################
print 'saving the model'
###############################

filename = 'level1_model.pkl'
os.system('rm '+filename)
pickle.dump(model, open(filename, 'wb'))

