import numpy as np
import itertools
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--brick',help='which brick n', default=1)
parser.add_argument('-t', '--train', help='training or test set', action='store_true', default=False)
args = parser.parse_args()

if args.train:
    ds = pd.read_csv('/data/atlas/atlasdata/zgubic/MLHEP17/RawBricks/DS_2_train_brick'+args.brick+'.csv')
else:
    ds = pd.read_csv('/data/atlas/atlasdata/zgubic/MLHEP17/RawBricks/DS_2_test_brick'+args.brick+'.csv')

class Brick:
    def __init__(self):
        self.tracks = ds
        print 'n_tracks = ', len(self.tracks)
        self.zlayers = np.arange(57)
        self.dz = 1293
        self.R1 = 2000.0
        self.R2 = 5000.0
        self.nsteps = 4
        
    def z_to_layer(self, z):
        return int(z/self.dz)
    
    def do_tracking(self):
        
        # set the columns
        self.tracks['min_sep'] = -1*np.ones(len(self.tracks))
        self.tracks['cone1'] = -1*np.ones(len(self.tracks))
        self.tracks['cone2'] = -1*np.ones(len(self.tracks))

        for i in range(1,self.nsteps+1):
            self.tracks['dR'+str(i)+'up'] = -999*np.ones(len(self.tracks))
            self.tracks['dR'+str(i)+'down'] = -999*np.ones(len(self.tracks))
            self.tracks['dT'+str(i)+'up'] = -1*np.ones(len(self.tracks))
            self.tracks['dT'+str(i)+'down'] = -1*np.ones(len(self.tracks))
            self.tracks['cone1_'+str(i)+'up'] = -1*np.ones(len(self.tracks))
            self.tracks['cone1_'+str(i)+'down'] = -1*np.ones(len(self.tracks))
            self.tracks['cone2_'+str(i)+'up'] = -1*np.ones(len(self.tracks))
            self.tracks['cone2_'+str(i)+'down'] = -1*np.ones(len(self.tracks))

        # loop over all tracks
        for tr_id in self.tracks.index:

            zlayer = self.z_to_layer(self.tracks.loc[tr_id]['Z'])
            
            # predict location of next/previous track in the shower
            x0 = self.tracks.loc[tr_id]['X']
            y0 = self.tracks.loc[tr_id]['Y']
            z0 = self.tracks.loc[tr_id]['Z']
            tx = self.tracks.loc[tr_id]['TX']
            ty = self.tracks.loc[tr_id]['TY']
            
            # upstream
            xNup = [x0 - i*self.dz*np.tan(tx) for i in range(1, self.nsteps+1)]
            yNup = [y0 - i*self.dz*np.tan(ty) for i in range(1, self.nsteps+1)]
            # downstream
            xNdown = [x0 + i*self.dz*np.tan(tx) for i in range(1, self.nsteps+1)]
            yNdown = [y0 + i*self.dz*np.tan(ty) for i in range(1, self.nsteps+1)]
            
            # compute distances to all the tracks: first in the same zlayer, then up and downstream
            xs_0 = np.array(self.tracks[self.tracks.Z == z0]['X'])
            ys_0 = np.array(self.tracks[self.tracks.Z == z0]['Y'])
            dists = np.sqrt((xs_0-x0)**2 + (ys_0-y0)**2)
            self.tracks.set_value(tr_id, 'min_sep', np.partition(dists, 1)[1])
            self.tracks.set_value(tr_id, 'cone1', np.sum(dists < self.R1))
            self.tracks.set_value(tr_id, 'cone2', np.sum(dists < self.R2))

            for nstep in range(1, self.nsteps+1):
                # upstream
                if zlayer > nstep-1:
                    # get the minimum distance
                    xs_up = np.array(self.tracks[self.tracks.Z == z0-nstep*self.dz]['X'])
                    ys_up = np.array(self.tracks[self.tracks.Z == z0-nstep*self.dz]['Y'])
                    dists_up = np.sqrt((xs_up-xNup[nstep-1])**2 + (ys_up-yNup[nstep-1])**2)
                    self.tracks.set_value(tr_id, 'dR'+str(nstep)+'up', np.min(dists_up))
                    
                    # get the difference in angle from that minimum distance track
                    min_ind = np.argmin(dists_up)
                    tx_up = self.tracks[self.tracks.Z == z0-nstep*self.dz].iloc[min_ind]['TX']
                    ty_up = self.tracks[self.tracks.Z == z0-nstep*self.dz].iloc[min_ind]['TY']
                    self.tracks.set_value(tr_id, 'dT'+str(nstep)+'up', np.sqrt((tx-tx_up)**2 + (ty-ty_up)**2))

                    # get the number of tracks inside the cylinder of radius R1, R2
                    N1up = np.sum(dists_up < self.R1)
                    self.tracks.set_value(tr_id, 'cone1_'+str(nstep)+'up', N1up)
                    N2up = np.sum(dists_up < self.R2)
                    self.tracks.set_value(tr_id, 'cone2_'+str(nstep)+'up', N2up)
                    
                # downstream
                if zlayer < 57 - nstep:
                    xs_down = np.array(self.tracks[self.tracks.Z == z0+nstep*self.dz]['X'])
                    ys_down = np.array(self.tracks[self.tracks.Z == z0+nstep*self.dz]['Y'])
                    dists_down = np.sqrt((xs_down-xNdown[nstep-1])**2 + (ys_down-yNdown[nstep-1])**2)
                    self.tracks.set_value(tr_id, 'dR'+str(nstep)+'down', np.min(dists_down))
                    
                    # get the difference in angle from that minimum distance track
                    min_ind = np.argmin(dists_down)
                    tx_down = self.tracks[self.tracks.Z == z0+nstep*self.dz].iloc[min_ind]['TX']
                    ty_down = self.tracks[self.tracks.Z == z0+nstep*self.dz].iloc[min_ind]['TY']
                    self.tracks.set_value(tr_id, 'dT'+str(nstep)+'down', np.sqrt((tx-tx_down)**2 + (ty-ty_down)**2))

                    # get the number of tracks inside the cylinder of radius R1, R2
                    N1down = np.sum(dists_down < self.R1+self.dz*0.3*nstep)
                    self.tracks.set_value(tr_id, 'cone1_'+str(nstep)+'down', N1down)
                    N2down = np.sum(dists_down < self.R2+self.dz*0.3*nstep)
                    self.tracks.set_value(tr_id, 'cone2_'+str(nstep)+'down', N2down)

br2 = Brick()
br2.do_tracking()

if args.train:
    br2.tracks.to_csv('/data/atlas/atlasdata/zgubic/MLHEP17/TrackingAdded3/DS_2_train_brick'+args.brick+'_TrackingAdded3.csv')
else:
    br2.tracks.to_csv('/data/atlas/atlasdata/zgubic/MLHEP17/TrackingAdded3/DS_2_test_brick'+args.brick+'_TrackingAdded3.csv')

