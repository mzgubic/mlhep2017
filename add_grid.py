import numpy as np
import itertools
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--brick',help='which brick n', default=1)
parser.add_argument('-t', '--train', help='training or test set', action='store_true', default=False)
parser.add_argument('-n', '--ntracks',help='number of tracks to runon', default=10)
args = parser.parse_args()


n_tracks = args.ntracks
if args.train:
    ds = pd.read_csv('/data/atlas/atlasdata/zgubic/MLHEP17/RawBricks/DS_2_train_brick'+args.brick+'.csv')
else:
    ds = pd.read_csv('/data/atlas/atlasdata/zgubic/MLHEP17/RawBricks/DS_2_test_brick'+args.brick+'.csv')


class Brick:
    def __init__(self, ID):
        self.ID = ID
        #print ID
        #print ds.brick_number
        self.tracks = ds#[ds.brick_number==ID]
        print 'n_tracks = ', len(self.tracks)
        self.grid = np.zeros(shape=(10,10,10))
        self.xmin = np.min(self.tracks['X'])
        self.ymin = np.min(self.tracks['Y'])
        self.zmin = np.min(self.tracks['Z'])
        self.xrange = (np.max(self.tracks['X']) - self.xmin)/10. + 1
        self.yrange = (np.max(self.tracks['Y']) - self.ymin)/10. + 1
        self.zrange = (np.max(self.tracks['Z']) - self.zmin)/10. + 1
        self.has_grid = False
        
    def fill_grid(self):
        # prevent redoing it
        if self.has_grid:
            return None
        
        # decorate tracks with the index of the cell
        self.tracks['ix'] = np.zeros(len(self.tracks), dtype=int)
        self.tracks['iy'] = np.zeros(len(self.tracks), dtype=int)
        self.tracks['iz'] = np.zeros(len(self.tracks), dtype=int)
        
        for tr_id in self.tracks.index:#[:n_tracks]:
            ind = self.get_grid_index(tr_id)
            self.grid[ind] += 1
            self.tracks.set_value(tr_id, 'ix', ind[0])
            self.tracks.set_value(tr_id, 'iy', ind[1])
            self.tracks.set_value(tr_id, 'iz', ind[2])
        
        self.grid = self.grid/np.mean(self.grid)
        self.has_grid = True
    
    def get_grid_index(self, track_id):
        #print 'min', np.min(self.tracks['X'])
        #print 'max', np.max(self.tracks['X'])
        #print np.max(self.tracks['X']) - np.min(self.tracks['X'])
        x = self.tracks[self.tracks.index == track_id]['X']
        y = self.tracks[self.tracks.index == track_id]['Y']
        z = self.tracks[self.tracks.index == track_id]['Z']
        xind = int((x-self.xmin)/self.xrange)
        yind = int((y-self.ymin)/self.yrange)
        zind = int((z-self.zmin)/self.zrange)

        return (xind, yind, zind)
    
    def decorate_tracks(self):
        if not self.has_grid:
            self.fill_grid()
        
        self.smooth_grid()
        
        self.tracks['grid_value'] = np.zeros(len(self.tracks))
        self.tracks['smoothgrid_value'] = np.zeros(len(self.tracks))

        for tr_id in self.tracks.index:#[:n_tracks]:
            ix = self.tracks.get_value(tr_id, 'ix')
            iy = self.tracks.get_value(tr_id, 'iy')
            iz = self.tracks.get_value(tr_id, 'iz')
            self.tracks.set_value(tr_id, 'grid_value', self.grid[(ix, iy, iz)])
            self.tracks.set_value(tr_id, 'smoothgrid_value', self.smooth[(ix, iy, iz)])

    def smooth_grid(self):
        self.smooth = self.grid.copy()
        
        a = [range(self.grid.shape[0]), range(self.grid.shape[1]), range(self.grid.shape[2])]
        indices = list(itertools.product(*a))

        for inds, dimension, pm in itertools.product(indices, [0,1,2], [-1, 1]):
            #print
            #print inds, dimension, pm
            try:
                newinds = list(inds)
                newinds[dimension] += pm
                newinds = tuple(newinds)
                if -1 in newinds:
                    continue
                #print 'successfull:', newinds
                self.smooth[newinds] += self.grid[inds]/2.
            except IndexError:
                continue
        
br2 = Brick(args.brick)
br2.fill_grid()
br2.decorate_tracks()

if args.train:
    br2.tracks.to_csv('/data/atlas/atlasdata/zgubic/MLHEP17/MihaAdded/DS_2_train_brick'+args.brick+'_MihaAdded.csv')
else:
    br2.tracks.to_csv('/data/atlas/atlasdata/zgubic/MLHEP17/MihaAdded/DS_2_test_brick'+args.brick+'_MihaAdded.csv')

