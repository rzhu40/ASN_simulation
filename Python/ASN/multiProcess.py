import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
import time
from multiprocessing import Pool, set_start_method
import pickle

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

if __name__ == "__main__":
    logging.info(f'{4} cores are being used.')
    Connectivity = connectivity__('100nw_261junctions.mat')
    logging.info('starting simulation.')
    Amps = np.arange(1,200)
    calcList = []
    time1 = time.time()
    from tqdm import tqdm
    for i in Amps:
        calcList.append(inputPacker(defaultSimulation, Connectivity, onAmp = i, dt = 0.01, T = 2.8, contactMode='farthest', disable_tqdm = True, findFirst= False))
    
    set_start_method('spawn')
    with Pool(processes=4) as pool:  
        sim = list(tqdm(pool.istarmap(defaultSimulation, calcList), total = len(calcList), desc = 'Running Simulation'))
    # sims = pool.starmap(defaultSimulation, initList)
    time2 = time.time()
    logging.info(f'{len(calcList)} simulations runs for {time2 - time1} seconds.')
