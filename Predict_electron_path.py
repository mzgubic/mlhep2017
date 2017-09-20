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


class Brick:
    def __init__(self, brick_nb, which):
        self.brick_nb = brick_nb
        self.which = which
        self.pred_n_showers = 1
        self.max_score = -1

        self.load_dataset()
        self.load_model()
        self.model_predict()
        self.predict_nb_showers()
        self.fit_electron_tracks()
        self.compute_variables()

    def say(self, what):
        print 
        print '#####################################'
        print '# '+str(what)
        print '#####################################'
        print 

    def load_dataset(self):
        self.say('loading dataset')

        if self.which == 'train':
            Data = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_train_extended_float32.hdf')
            self.D_test = Data[Data.brick_number == np.int8(self.brick_nb)]
            self.X_test = self.D_test.drop('index', axis=1).drop('event_id', axis=1).drop('signal', axis=1).drop('brick_number', axis=1)

        if self.which == 'test':
            Data = pd.read_hdf('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_test_extended_float32.hdf')
            self.D_test = Data[Data.brick_number == np.int8(args.brick)]
            self.X_test = self.D_test.drop('index', axis=1).drop('brick_number', axis=1)

        self.n_tracks = len(self.D_test['X'])
        
    def load_model(self):
        self.say('loading model')
        filename = 'level1_model.pkl'
        self.model = pickle.load(open(filename, 'rb'))

    def model_predict(self, pred_threshold=0.30):
        self.say('making predictions')
        self.predictions = self.model.predict_proba(self.X_test)[:,1]
        self.D_test['prediction'] = self.predictions
        self.showers = self.D_test[self.D_test.prediction > pred_threshold]
        self.shower_coords = self.showers[['X', 'Y']]

    def predict_nb_showers(self):
        self.say('determine the number of showers')

        self.n_shower_tracks = self.showers.shape[0]
        
        # if nb of predicted tracks is too low, get out
        if self.n_shower_tracks < 2:
            self.write_out(self.get_dummy_output())

        # loop over possible numbers of showers, and score each one
        self.shower_sets = [ShowerSet(i, self.n_shower_tracks) for i in range(1,6)]
        for shower_set in self.shower_sets:
            shower_set.label(self.shower_coords)
            if shower_set.silh_score > self.max_score:
                self.max_score = shower_set.silh_score
                self.set_pred_n_showers(shower_set.n_showers)

        # and loop to check all requirements
        self.all_good = False
        while not self.all_good:
            all_good = True
            all_good = all_good and self.check_std_dev()
            all_good = all_good and self.check_nb_tracks()
            all_good = all_good and self.check_label_counts()
            self.all_good = all_good

        if not self.pred_n_showers == self.best_shower_set.n_showers:
            self.say('ERROR: Wrong shower set selected')

        print 'Predicted number of clusters is:', self.pred_n_showers

        # store results for fitting
        self.shower_coords['shower_id'] = self.best_shower_set.labels
        self.shower_coords['Z'] = self.showers['Z']

    def fit_electron_tracks(self):
        self.say('fit the electron tracks')

        self.electrons = []
        for sh_i in range(self.pred_n_showers):
            print 'computing shower', sh_i
            current_shower = self.shower_coords[self.shower_coords.shower_id == sh_i]

            # do the fitting
            popt, pcov = spo.curve_fit(linear, current_shower['Z'], current_shower['Y'], p0=[np.mean(current_shower['Y']), 0])
            y0 = popt[0]
            yk = popt[1]
            popt, pcov = spo.curve_fit(linear, current_shower['Z'], current_shower['X'], p0=[np.mean(current_shower['X']), 0])
            x0 = popt[0]
            xk = popt[1]

            # create electron
            zmin = np.min(current_shower['Z'])
            self.electrons.append(electron(x0, y0, zmin, xk, yk))

    def compute_variables(self):

        self.say('Now take all the tracks in the set and compute the distance to the electron path')

        tracks_xs = self.D_test['X']
        tracks_ys = self.D_test['Y']
        tracks_zs = self.D_test['Z']
        
        # compute the distance of each track to each electron
        distances = np.zeros(shape=(self.n_tracks, self.pred_n_showers))
        for i, elec in enumerate(self.electrons):
            distances[:,i] = elec.get_distance_from(tracks_xs, tracks_ys, tracks_zs)
        
        # get the index of the closest electron for each track
        closest_e = distances.argsort()[:, 0]
        
        # compute the dTheta from the closest electorn track, and dZ (distance along Z from the start of the shower)
        dR = np.zeros(self.n_tracks)
        dT = np.zeros(self.n_tracks)
        dZ = np.zeros(self.n_tracks)
        
        for tr_i in range(self.n_tracks):
            e_i = closest_e[tr_i]
            dTX = self.electrons[e_i].get_dTX(self.D_test.iloc[tr_i]['TX'])
            dTY = self.electrons[e_i].get_dTY(self.D_test.iloc[tr_i]['TY'])
            dT[tr_i] = np.sqrt(dTX**2 + dTY**2)
            dZ[tr_i] = self.electrons[e_i].z1 - self.D_test.iloc[tr_i]['Z']
        
        # and save the distance to the closest electron track
        dR = np.min(distances, axis=1)

        # and finally add to the dataset
        output = self.D_test[['index', 'brick_number']]
        output['e_dR'] = dR
        output['e_dT'] = dT
        output['e_dZ'] = dZ
        
        if self.which == 'train':
            output['signal'] = self.D_test['signal']
        
        self.write_out(output)

    def select_set_with_n(self, N):
        self.best_shower_set = self.shower_sets[N-1]

    def set_pred_n_showers(self, N):
        self.pred_n_showers = N
        self.select_set_with_n(N)

    def check_nb_tracks(self):
        if self.n_tracks < 80:
            self.set_pred_n_showers(1)
            return False
        else:
            return True
                
    def check_std_dev(self):
        x_sd = np.std(self.showers['X'])
        y_sd = np.std(self.showers['Y'])
        sdev = np.sqrt(x_sd**2+y_sd**2)
        if self.pred_n_showers == 2 and sdev < 4500:
            self.set_pred_n_showers(1)
            return False
        else:
            return True

    def check_label_counts(self):
        print 'pred showers', self.pred_n_showers, self.best_shower_set.n_showers
        print 'second', self.best_shower_set.label_counts
        if (not self.pred_n_showers == 1) and any(np.array(self.best_shower_set.label_counts) < 3):
            self.set_pred_n_showers(self.best_shower_set.n_showers - 1)
            return False
        else:
            return True

    def get_dummy_output(self):
        output = self.D_test[['index', 'brick_number']]
        output['e_dR'] = 40000*np.ones(self.n_tracks)
        output['e_dT'] = 0.5*np.ones(self.n_tracks)
        output['e_dZ'] = 30000*np.ones(self.n_tracks)
        if args.train:
            output['signal'] = D_test['signal']
        return output

    def write_out(self, output):
        if self.which == 'train':
            output.to_csv('/data/atlas/atlasdata/zgubic/MLHEP17/ElecPathAdded/DS_2_train_brick'+str(self.brick_nb)+'_ElecPathAdded.csv')
        else:
            output.to_csv('/data/atlas/atlasdata/zgubic/MLHEP17/ElecPathAdded/DS_2_test_brick'+str(self.brick_nb)+'_ElecPathAdded.csv')
        exit()

class ShowerSet:
    def __init__(self, n_showers, n_tracks):
        self.n_showers = n_showers
        self.n_tracks = n_tracks
        self.labels = np.zeros(self.n_tracks)
        self.silh_score = -1
        self.kmeans = KMeans(n_clusters=n_showers)
        
    def label(self, coords):
        if self.n_showers > 1:
            self.labels = self.kmeans.fit_predict(coords)
            self.silh_score = silhouette_score(coords, self.labels)

        # check for the number of each labels
        self.label_counts = []
        for label in range(0, self.n_showers):
            self.label_counts.append(np.sum(self.labels == label))
        print self.label_counts

which = 'test'
if args.train:
    which = 'train'

brick = Brick(args.brick, which)




