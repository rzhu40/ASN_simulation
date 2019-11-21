import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
import time
from multiprocessing import Pool, Queue
import pickle

import logging
logging.basicConfig(filename = 'log/multi_test.log',level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def inputPacker(Connectivity, 
                junctionMode = 'binary', collapse = False,
                contactMode='farthest', electrodes=None,
                dt=1e-3, T=10, 
                biasType = 'DC',
                onTime=0, offTime=5,
                onAmp=1.1, offAmp=0.005,
                f = 1, customSignal = None,
                lite_mode = False, save_steps = 1,
                findFirst = True,
                disable_tqdm = False):

    return list(locals().values())

if __name__ == "__main__":
    logging.info(f'{4} cores are being used.')
    Connectivity = connectivity__('100nw_261junctions.mat')
    logging.info('starting simulation.')
    Amps = np.arange(0,200)
    initList = []
    time1 = time.time()
    from tqdm import tqdm
    for i in Amps:
        initList.append(inputPacker(Connectivity, onAmp = i, dt = 0.01, T = 2.8, contactMode='farthest', disable_tqdm = True, findFirst= False))
    with Pool(processes=4) as pool:  
        sim = list(tqdm(pool.istarmap(defaultSimulation, initList), total = len(initList), desc = 'Running Simulation'))
    # sims = pool.starmap(defaultSimulation, initList)
    time2 = time.time()
    logging.info(f'{len(initList)} simulations runs for {time2 - time1} seconds.')
