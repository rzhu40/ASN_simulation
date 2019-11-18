import numpy as np
import matplotlib.pyplot as plt
from jpype import *

from utils import *
from dataStruct import *
from analysis.InfoTheory import calc_TE, calc_networkTE
from analysis.mkg import mkg_generator
from draw.draw_graph import draw_graph
import pickle
plt.style.use('classic')

def calc_network(Network, dt_sampling = 1e-1, N = 1e3, t_start=10, calculator = 'kraskov', return_sampling = False):
    dt_euler = Network.TimeVector[1] - Network.TimeVector[0]
    sample_start = int(t_start/dt_euler)
    sample_end = sample_start + int(N*dt_sampling/dt_euler)
    sampling = np.arange(sample_start, sample_end, int(dt_sampling/dt_euler))
    if sampling[-1] > Network.TimeVector.size:
        return None
    
    wireVoltage = Network.wireVoltage
    E = Network.numOfJunctions
    TE = np.zeros((sampling.size, E))
    edgeList = Network.connectivity.edge_list
    mean_direction = np.sign(np.mean(Network.filamentState, axis=0))
    for i in tqdm(range(len(edgeList)), desc = 'Calculating TE ', disable = True):
        if mean_direction[i] >= 0:
            wire1, wire2 = edgeList[i,:]
        else:
            wire2, wire1 = edgeList[i,:]
        TE[:,i] = calc_TE(wireVoltage[sampling, wire1], wireVoltage[sampling, wire2], calculator = calculator, calc_type = 'local')
#         TE[:,i] = calc_TE(wireVoltage[sampling, wire2], wireVoltage[sampling, wire1], calculator = 'gaussian', calc_type = 'local')
    if return_sampling:
        return TE, sampling
    else:
        return TE


if __name__ == '__main__': 
    sim1 = defaultSimulation(connectivity__(filename = '100nw_261junctions'), 
                            lite_mode = True,
                            T = 2010, dt = 1e-3, offTime = 10000000, 
                            biasType = 'MKG', onAmp = 1, f = 1,
                            contactMode = 'preSet', electrodes = [72,29], 
                            junctionMode = 'binary', findFirst = False)

    dt_sample1 = [1e-3, 2e-3, 4e-3, 5e-3, 1e-2, 2e-2, 4e-2, 5e-2]
    # dt_sample1 = [1e-3, 2e-3]
    dt_sample2 = [1e-1, 2e-1, 4e-1]

    sim2 = network__()
    sim2.TimeVector = np.arange(0, 20010, 0.001)
    sim2.numOfJunctions = 261
    sim2.connectivity = sim1.connectivity

    with open('data/longSim_V.pkl', 'rb') as handle:
        sim2.wireVoltage = pickle.load(handle)

    with open('data/longSim_lambda.pkl', 'rb') as handle:
        sim2.filamentState = pickle.load(handle)
    
    TE20000 = []
    TE40000 = []

    for i, this_dt in enumerate(dt_sample1):
        TE20000.append(calc_network(sim1, this_dt, N = 20000, t_start = 10, calculator = 'kraskov'))
        TE40000.append(calc_network(sim2, this_dt, N = 40000, t_start = 10, calculator = 'kraskov'))
    
    with open('data/TE_N20000.pkl', 'wb') as handle:
        pickle.dump(TE20000, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open('data/TE_N40000.pkl', 'wb') as handle:
        pickle.dump(TE40000, handle, protocol = pickle.HIGHEST_PROTOCOL)