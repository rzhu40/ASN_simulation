from utils import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from analysis import *

plt.style.use('classic')
from analysis.GraphTheory import *

from tqdm import tqdm
from multiprocessing import Pool
from scipy.io import loadmat, savemat


if __name__ == '__main__':
    adjMat = np.array(loadmat('connectivity_data/alon100nw.mat')['adjmat'].todense())
    dist = 4
    pairingList = getCertainDistPairing(adjMat, dist, 100)

    initList = [inputPacker(defaultSimulation, connectivity__(adjMat = adjMat), 
                       contactMode = 'preSet', electrodes = pairingList[i],
                       biasType = 'DC', T = 2, onAmp = 1.9, findFirst = False, disable_tqdm = True) 
            for i in range(len(pairingList))]

    with Pool(processes=4) as pool:   
        simList = list(tqdm(pool.istarmap(defaultSimulation, initList), total = len(initList), desc = f'Simulating with {pool._processes} processors.'))
        comm_calcList = [inputPacker(getSubGraphComm, simList[i], len(simList[i].TimeVector)-1) for i in range(len(simList))]
        commResult = list(tqdm(pool.istarmap(getSubGraphComm, comm_calcList), total = len(comm_calcList), desc = f'Calculating comm with {pool._processes} processors.'))

    commList = np.array([commResult[i][simList[i].sources[0]][simList[i].drains[0]] for i in range(len(simList))])
    currList = [simList[i].electrodeCurrent[-1,1] for i in range(len(simList))]

    maxI = 1e-4
    reachList = np.zeros(len(simList))
    for i in range(len(simList)):
        try:
            reachList[i] = min(np.where(simList[i].electrodeCurrent[:,1] > maxI)[0])*0.001
        except:
            reachList[i] = -1
    
    outDict = dict(commList = commList,
                currList = currList,
                reachList = reachList)
    filename = f'data/dist{dist}_comm_curr'
    savemat(filename, outDict)
    # plt.loglog(currList, commList,  '.')
    # plt.xlabel('current')
    # plt.ylabel('comm')
    # plt.show()

    # plt.figure()
    # plt.plot(commList, reachList, 'x')
    # plt.xscale('log')