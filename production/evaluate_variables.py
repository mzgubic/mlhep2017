import numpy as np
import pandas as pd
import time
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# import the dataset
ds_train = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_train_extended_float16.hdf')

Y_train = ds_train[ds_train.brick_number < 60]['signal']
Y_test1 = ds_train[(ds_train.brick_number > 59) & (ds_train.brick_number < 80)]['signal']
Y_test2 = ds_train[(ds_train.brick_number > 79)]['signal']

X_full = ds_train.copy()
ds_train = None
X_full = X_full.drop('index', axis=1).drop('event_id', axis=1).drop('signal', axis=1)
print X_full.columns

# select the training and test samples
X_train = X_full[X_full.brick_number < 60].drop('brick_number', axis=1)
X_test1 = X_full[(X_full.brick_number > 59) & (X_full.brick_number < 80)].drop('brick_number', axis=1)
X_test2 = X_full[(X_full.brick_number > 79)].drop('brick_number', axis=1)

# do the training
n_est = 8
model = GradientBoostingClassifier(n_estimators=n_est, max_depth=4, loss='exponential')
t1= time.time()
model.fit(X_train, Y_train)

# and prediction
pred_train = model.predict_proba(X_train)[:,1]
pred_test1 = model.predict_proba(X_test1)[:,1]
pred_test2 = model.predict_proba(X_test2)[:,1]
print 'n_est:', n_est
print 'time taken:', time.time() - t1
print 'train score:', roc_auc_score(Y_train, pred_train)
print 'test1 score:', roc_auc_score(Y_test1, pred_test1)
print 'test2 score:', roc_auc_score(Y_test2, pred_test2)
