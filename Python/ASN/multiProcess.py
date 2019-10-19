import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
import time

from multiprocessing import Pool


def inputPacker(Connectivity, 
                contactMode='farthest', electrodes=None,
                dt= 1e-3, T=10, 
                biasType = 'DC',
                onTime=0, offTime=5,
                onAmp=1.1, offAmp=0.005,
                f = 1):

    packer = [Connectivity, contactMode, electrodes, dt, T, biasType, onTime, offTime, onAmp, offAmp, f]
    return packer
    
if __name__ == "__main__":
    pool = Pool(processes=8) 
    # Connectivity = connectivity__(
    #                     filename = '2016-09-08-155153_asn_nw_00100_nj_00261_seed_042_avl_100.00_disp_10.00.mat')
    
    Connectivity = connectivity__(
                            filename = '2016-09-08-155044_asn_nw_00700_nj_14533_seed_042_avl_100.00_disp_10.00.mat')

    Amps = np.arange(0,60*2*8)
    initList = []
    time1 = time.time()
    for i in Amps:
        initList.append(inputPacker(Connectivity, onAmp = i, dt = 0.01, T = 2.8, contactMode='Alon'))        

    sims = pool.starmap(defaultSimulation, initList)
    time2 = time.time()
    print(time2-time1)

    # ss = []
    # for i in Amps:
    #     ss.append(defaultSimulation(Connectivity, onAmp = i))
    # time3 = time.time()
    # print(time3-time2)

