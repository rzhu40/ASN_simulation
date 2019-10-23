import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import *
from scipy.io import loadmat, savemat
from tqdm import tqdm

def Alon_test(Connectivity, pairing):
    # sep_list = np.arange(5, 500, 1)
    sep_list = np.arange(5,500,5)
    out_dict = dict()
    for this_sep in sep_list:
        signal1 = np.ones(1200)*0.005
        signal1[0:200] = 2
        signal1[200+this_sep:400+this_sep] = 2
        signal2 = np.ones(1200)*0.005
        signal2[200:400] = 2
        signal2[400+this_sep:600+this_sep] = 2

        sim1 = defaultSimulation(Connectivity = Connectivity, 
                                contactMode = 'preSet', electrodes= pairing, 
                                dt= 1e-2, T=12, 
                                biasType='Custom', cutsomSignal=signal1,
                                findFirst = False, disable_tqdm=True)

        sim2 = defaultSimulation(Connectivity = Connectivity, 
                                contactMode = 'preSet', electrodes= pairing, 
                                dt= 1e-2, T=12, 
                                biasType='Custom', cutsomSignal=signal2,
                                findFirst = False, disable_tqdm=True)
        sub_dict_name = 'sep_' + str(this_sep)

        out_dict[sub_dict_name] = dict(seperation= this_sep,
                                    pairing= pairing,
                                    IDrain1= sim1.electrodeCurrent[:,1],
                                    IDrain2= sim2.electrodeCurrent[:,1],
                                    VSource1= signal1,
                                    VSource2= signal2)
        
    return out_dict

if __name__ == '__main__':
    import time
    start = time.time()
    pairing_list = loadmat('connectivity_data/ElecPos.mat')['elecPos'] - 1
    pairing_list[0,:] = np.array([4,17])
    pairing_list[80,:] = np.array([22, 80])
    pairing_list[81,:] = np.array([99, 81])
    
    adjMat = loadmat('connectivity_data/alon100nw.mat')['adjmat'].todense()
    G = nx.from_numpy_array(adjMat)
    Connectivity = connectivity__(graph = G)
    for this_pairing in tqdm(pairing_list):
        out = Alon_test(Connectivity, this_pairing)
        filename = 'data/source_' + str(this_pairing[0]) + '_drain_' + str(this_pairing[1]) + '.mat'
        savemat(filename, out)
    end = time.time()
    
    print(f'100x2x2 Sims used {end - start} seconds!')