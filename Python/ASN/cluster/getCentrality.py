import os
from pathlib import Path
file_path = Path(os.path.dirname(os.path.realpath(__file__)))
root = file_path.parent
os.chdir(root)

import numpy as np 
from utils import *
from analysis.GraphTheory import *
import pickle

if __name__ == '__main__':
    Connectivity = connectivity__('100nw_261junctions.mat')
    sim1 = runSim(Connectivity, biasType = 'DC', onAmp = 1, T=10, 
                contactMode = 'farthest')
    out = dict(junctionCent = get_junction_centrality(sim1, 0),
                wireCent = get_wire_centrality(sim1, 0, mode='betweenness'))

    with open('data/centralityTest.pkl', 'wb') as handle:
        pickle.dump(out, handle, protocol = pickle.HIGHEST_PROTOCOL)