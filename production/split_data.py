import numpy as np
import pandas as pd

for which in ['test', 'train']:
    ds = pd.read_csv('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_'+which+'.csv')
    for i in range(1,101):
        brick = ds[ds.brick_number == i]
        print brick.head()
        brick.to_csv('/data/atlas/atlasdata/zgubic/MLHEP17/DS_2_'+which+'_brick'+str(i)+'.csv', header=True)

