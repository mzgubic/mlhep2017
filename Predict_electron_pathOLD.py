print 'loading modules'
import pickle
import argparse
import numpy as np
import pandas as pd
import time
import os
import scipy.optimize as spo
from sklearn.metrics import roc_auc_score
from sklearn.metrics import silhouette_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# parse them args
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--brick',help='which brick n', default=1)
parser.add_argument('-t', '--train', help='training or test set', action='store_true', default=False)
args = parser.parse_args()

class electron:
    def __init__(self, x1, y1, z1, tx, ty):
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.tx = tx
        self.ty = ty
        
    def get_x_pos(self, z):
        x_pos = self.x1 + z*np.sin(self.tx)
        return x_pos
    
    def get_y_pos(self, z):
        y_pos = self.y1 + z*np.sin(self.ty)
        return y_pos
    
    def get_distance_from(self,x,y,z):
        dx = x - self.get_x_pos(z)
        dy = y - self.get_y_pos(z)
        return (dx**2 + dy**2)**0.5
    
    def get_dTX(self, tx):
        return tx - self.tx
    
    def get_dTY(self, ty):
        return ty - self.ty

def linear(x, intercept, slope):
    return intercept + x*slope

def write_out(output):
    # write out the results
    if args.train:
        output.to_csv('/data/atlas/atlasdata/zgubic/MLHEP17/ElecPathAdded/DS_2_train_brick'+str(args.brick)+'_ElecPathAdded.csv')
    else:
        output.to_csv('/data/atlas/atlasdata/zgubic/MLHEP17/ElecPathAdded/DS_2_test_brick'+str(args.brick)+'_ElecPathAdded.csv')
    exit()

###############################
print  'loading the datasets'
###############################

# import the training set and choose what the test set is
if args.train:
    D_train = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_train_extended_float32.hdf')
    D_test = D_train[D_train.brick_number == np.int8(args.brick)]
else:
    D_test  = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_test_extended_float32.hdf')
    D_test = D_test[D_test.brick_number == np.int8(args.brick)]

if args.train:
    X_test = D_test.drop('index', axis=1).drop('event_id', axis=1).drop('signal', axis=1).drop('brick_number', axis=1)
else:
    X_test = D_test.drop('index', axis=1).drop('brick_number', axis=1)

##############################
print 'determine the number of showers'
###############################

filename = 'level1_model.pkl'
model = pickle.load(open(filename, 'rb'))
pred_test = model.predict_proba(X_test)[:,1]

# predict tracks belonging to the shower:
D_test['prediction'] = pred_test
showers = D_test[D_test.prediction > 0.30]

# in case of no showers, save these values
print 'showers.shape', showers.shape
if showers.shape[0] < 2:
    print 'outputting early'
    output = D_test[['index', 'brick_number']]
    output['e_dR'] = 40000*np.ones(len(D_test['X']))
    output['e_dT'] = 0.5*np.ones(len(D_test['X']))
    output['e_dZ'] = 30000*np.ones(len(D_test['X']))
    if args.train:
        output['signal'] = D_test['signal']
    write_out(output)
    exit()

# now guess the number of showers
shower_coords = showers[['X', 'Y']]
shower_pred = list(range(5))
shower_pred[0] = np.zeros(showers.shape[0])
sil_scores = np.array(range(5), dtype=float)

# compute the kmeans and save the silhouette score
max_clusters = min(6, showers.shape[0])
for n_clus in range(2, max_clusters):
    print 'n_clust', n_clus
    kmeans = KMeans(n_clusters=n_clus)
    shower_pred[n_clus-1] = kmeans.fit_predict(shower_coords)
    silhouette_avg = silhouette_score(shower_coords, shower_pred[n_clus-1])
    sil_scores[n_clus-1] = silhouette_avg

    print "Silhouette score avg is:", sil_scores[n_clus-1]

# determine the number of clusters
pred_n_showers = np.argmax(sil_scores)+1
x_sd = np.std(showers['X'])
y_sd = np.std(showers['Y'])
sdev = np.sqrt(x_sd**2+y_sd**2)

# correct for very small number of tracks:
if showers.shape[0] < 80:
    pred_n_showers = 1

# correct for the 2/1 cluster case
if pred_n_showers == 2 and sdev < 4500:
    pred_n_showers = 1

# let us know the outcome
print 
print sil_scores
print 'x_sd', x_sd
print 'y_sd', y_sd
print 'standard deviation is:', sdev
print
print '###############################'
print 'Predicted number of clusters is:', pred_n_showers
print '###############################'

shower_coords['shower_id'] = shower_pred[pred_n_showers-1]
shower_coords['Z'] = showers['Z']

###############################
print 'fit the electron tracks'
###############################

electrons = []
for sh_i in range(pred_n_showers):

    print 'computing shower', sh_i
    
    # define and plot the shower 
    current_shower = shower_coords[shower_coords.shower_id == sh_i]
    print 'showers', shower_coords
    print 'current shower', current_shower
    
    # fit a line for electron position
    #print 'Z'
    print current_shower['Z']
    #print 'Y'
    print current_shower['Y']
    popt, pcov = spo.curve_fit(linear, current_shower['Z'], current_shower['Y'], p0=[np.mean(current_shower['Y']), 0])
    y0 = popt[0]
    yk = popt[1]
    popt, pcov = spo.curve_fit(linear, current_shower['Z'], current_shower['X'], p0=[np.mean(current_shower['X']), 0])
    x0 = popt[0]
    xk = popt[1]
    zmin = np.min(current_shower['Z'])
    electrons.append(electron(x0, y0, zmin, xk, yk))

###########################
print 'Now take all the tracks in the set and compute the distance to the electron path'
###########################
tracks_xs = D_test['X']
tracks_ys = D_test['Y']
tracks_zs = D_test['Z']

# compute the distance of each track to each electron
distances = np.zeros(shape=(len(tracks_xs), len(electrons)))
for i, elec in enumerate(electrons):
    distances[:,i] = elec.get_distance_from(tracks_xs, tracks_ys, tracks_zs)
print 'distances', distances

# get the index of the closest electron for each track
closest_e = distances.argsort()[:, 0]
print closest_e

# compute the dTheta from the closest electorn track, and dZ (distance along Z from the start of the shower)
dR = np.zeros(len(tracks_xs))
dT = np.zeros(len(tracks_xs))
dZ = np.zeros(len(tracks_xs))

for tr_i in range(len(D_test)):
    e_i = closest_e[tr_i]
    dTX = electrons[e_i].get_dTX(D_test.iloc[tr_i]['TX'])
    dTY = electrons[e_i].get_dTY(D_test.iloc[tr_i]['TY'])
    dT[tr_i] = np.sqrt(dTX**2 + dTY**2)
    dZ[tr_i] = electrons[e_i].z1 - D_test.iloc[tr_i]['Z']

# and save the distance to the closest electron track
dR = np.min(distances, axis=1)


###########################
print 'and finally add to the dataset'
###########################

output = D_test[['index', 'brick_number']]
output['e_dR'] = dR
output['e_dT'] = dT
output['e_dZ'] = dZ

if args.train:
    output['signal'] = D_test['signal']

write_out(output)
