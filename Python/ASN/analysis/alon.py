import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import *
from scipy.io import loadmat, savemat
from tqdm import tqdm

def simulateAlon(simulationOptions, connectivity, junctionState, disable_tqdm = False, maxI = 1e-4):

    if simulationOptions.contactMode == 'farthest':
        simulationOptions.electrodes = get_farthest_pairing(connectivity.adj_matrix)
    elif simulationOptions.contactMode == 'Alon':
        simulationOptions.electrodes = get_boundary_pairing(connectivity, 28)

    stimulusChecker = len(simulationOptions.electrodes) - len(simulationOptions.stimulus)
    if stimulusChecker > 0:
        for i in range(stimulusChecker):
            tempStimulus = stimulus__(biasType = 'Drain', 
            TimeVector = simulationOptions.TimeVector)
            simulationOptions.stimulus.append(tempStimulus)

    niterations = simulationOptions.NumOfIterations
    electrodes = simulationOptions.electrodes
    numOfElectrodes = len(electrodes)
    E = connectivity.numOfJunctions
    V = connectivity.numOfWires

    edgeList = connectivity.edge_list
    rhs = np.zeros(V+numOfElectrodes)

    import dataStruct 
    Network = dataStruct.network__()
    Network.filamentState = np.zeros((niterations, E))
    Network.junctionVoltage = np.zeros((niterations, E))
    Network.junctionResistance = np.zeros((niterations, E))
    Network.junctionSwitch = np.zeros((niterations, E), dtype = bool)
    Network.wireVoltage = np.zeros((niterations, V))
    Network.electrodeCurrent = np.zeros((niterations, numOfElectrodes))
    
    Network.sources = []
    Network.drains = []
    stop_step = niterations
    
    for i in range(numOfElectrodes):
        if np.mean(simulationOptions.stimulus[i].signal) != 0:
            Network.sources.append(electrodes[i])
        else:
            Network.drains.append(electrodes[i])
    if len(Network.drains) == 0:
        Network.drains.append(electrodes[1])

    for this_time in tqdm(range(niterations), desc='Running Simulation ', disable = disable_tqdm):
        junctionState.updateResistance()
        junctionConductance = 1/junctionState.resistance
        
        Gmat = np.zeros((V,V))
        Gmat[edgeList[:,0], edgeList[:,1]] = junctionConductance
        Gmat[edgeList[:,1], edgeList[:,0]] = junctionConductance
        Gmat = np.diag(np.sum(Gmat,axis=0)) - Gmat

        lhs = np.zeros((V+numOfElectrodes, V+numOfElectrodes))
        lhs[0:V,0:V] = Gmat
        for i in range(numOfElectrodes):
            this_elec = electrodes[i],
            lhs[V+i, this_elec] = 1
            lhs[this_elec, V+i] = 1
            rhs[V+i] = simulationOptions.stimulus[i].signal[this_time]
        
        # from scipy.sparse import csc_matrix
        # from scipy.sparse.linalg import spsolve
        # LHS = csc_matrix(lhs)
        # RHS = csc_matrix(rhs.reshape(V+numOfElectrodes,1))
        # sol = spsolve(LHS,RHS)

        sol = np.linalg.solve(lhs,rhs)
        wireVoltage = sol[0:V]
        junctionState.voltage = wireVoltage[edgeList[:,0]] - wireVoltage[edgeList[:,1]]
        junctionState.updateJunctionState(simulationOptions.dt)

        Network.wireVoltage[this_time,:] = wireVoltage
        Network.electrodeCurrent[this_time,:] = sol[V:]
        Network.filamentState[this_time,:] = junctionState.filamentState
        Network.junctionVoltage[this_time,:] = junctionState.voltage
        Network.junctionResistance[this_time,:] = junctionState.resistance
        Network.junctionSwitch[this_time,:] = junctionState.OnOrOff

        if abs(sol[V+1]) > maxI:
            stop_step = this_time
            break
    
    Network.TimeVector = simulationOptions.TimeVector[0:stop_step]
    Network.numOfWires = V
    Network.numOfJunctions = E
    Network.adjMat = connectivity.adj_matrix
    Network.graph = nx.from_numpy_array(connectivity.adj_matrix)
    Network.shortestPaths = [p for p in nx.all_shortest_paths(Network.graph, 
                                                        source=Network.sources[0], 
                                                        target=Network.drains[0])]
    Network.electrodes = simulationOptions.electrodes
    Network.criticalFlux = junctionState.critialFlux
    Network.stimulus = [simulationOptions.stimulus[i] for i in range(numOfElectrodes)]
    # Network.junctionList = np.add(connectivity.edge_list, 1).T
    Network.connectivity = connectivity
    # Network.TimeVector = simulationOptions.TimeVector
    return Network

def defaultAlon(Connectivity, junctionMode = 'binary', 
                    contactMode='farthest', electrodes=None,
                    dt= 1e-3, T=10, 
                    biasType = 'DC',
                    onTime=0, offTime=5,
                    onAmp=1.1, offAmp=0.005,
                    f = 1, cutsomSignal = None,
                    findFirst = True,
                    disable_tqdm = False):

    SimulationOptions = simulation_options__(dt = dt, T = T,
                                            contactMode = contactMode,
                                            electrodes = electrodes)
    
    SimulationOptions.stimulus = []                                         
    tempStimulus = stimulus__(biasType = biasType, 
                            TimeVector = SimulationOptions.TimeVector, 
                            onTime = onTime, offTime = offTime,
                            onAmp = onAmp, offAmp = offAmp,
                            f= f, customSignal= cutsomSignal)
    SimulationOptions.stimulus.append(tempStimulus)

    JunctionState = junctionState__(Connectivity.numOfJunctions, mode = junctionMode)

    this_realization = simulateAlon(SimulationOptions, Connectivity, JunctionState, disable_tqdm)
    
    if findFirst:
        from analysis.GraphTheory import findCurrent

        try:
            activation = findCurrent(this_realization, 1)
            print(f'First current path {activation[0][0]} formed at time = {activation[1][0]} s.')
        except:
            print('Unfortunately, no current path is formed in simulation time.')

    return this_realization

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

def Alon_test2(Connectivity, pairing):
    sim = defaultAlon(Connectivity = Connectivity, 
                                contactMode = 'preSet', electrodes= pairing, 
                                dt= 1e-2, T=2, 
                                biasType='Pulse', f = 5,
                                onAmp = 1.5,
                                findFirst = False, disable_tqdm=True)
    endCurrent = sim.electrodeCurrent[-1,1]

    return endCurrent, sim.TimeVector[-1]

def classifier(endTime, endCurrent):
    emln = np.zeros(len(endTime))
    for i in range(len(endTime)):
        if endTime[i] < 0.6:
            emln[i] = 0
        elif endTime[i] < 1.4:
            emln[i] = 1
        elif endTime[i] < 1.99:
            emln[i] = 2
        elif endCurrent[i] > 1e-4:
            emln[i] = 2
        else:
            emln[i] = 3
    return emln

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

    endTime = np.zeros(len(pairing_list))
    endCurrent = np.zeros(len(pairing_list))
    count = 0
    for this_pairing in tqdm(pairing_list):
        endCurrent[count], endTime[count] = Alon_test2(Connectivity, this_pairing)
        count += 1
    emln = classifier(endTime, endCurrent)
    # for this_pairing in tqdm(pairing_list):
    #     out = Alon_test(Connectivity, this_pairing)
    #     filename = 'data/source_' + str(this_pairing[0]) + '_drain_' + str(this_pairing[1]) + '.mat'
    #     savemat(filename, out)
    end = time.time()
    
    print(f'100x2x2 Sims used {end - start} seconds!')

