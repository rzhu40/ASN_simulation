import numpy as np
import scipy.io as sio
import networkx as nx
import time

from tqdm import tqdm
from copy import deepcopy

class simulation_options__:
    def __init__(self, dt=1e-3, T=1e1,
                contactMode = 'farthest',
                interfaceElectrodes = None):
        self.dt = dt
        self.T = T
        self.TimeVector = np.arange(0, T, dt)
        self.NumOfIterations = self.TimeVector.size
        self.contactMode = contactMode
        if contactMode == 'preSet':
            self.interfaceElectrodes = interfaceElectrodes
            self.electrodes = [i-1 for i in interfaceElectrodes]

class connectivity__:
    def __init__(self, filename=None, wires_dict=None, graph=None):
        if not filename == None:
            fullpath = 'connectivity/connectivity_data/' + filename
            matfile = sio.loadmat(fullpath, squeeze_me=True, struct_as_record=False)
            for key in matfile.keys():
                if key[0:2] != '__':
                    setattr(self, key, matfile[key])
            self.numOfJunctions = self.number_of_junctions
            self.numOfWires = self.number_of_wires
        elif not wires_dict == None:
            matfile = wires_dict
            for key in matfile.keys():
                if key[0:2] != '__':
                    setattr(self, key, matfile[key])
            self.numOfJunctions = self.number_of_junctions
            self.numOfWires = self.number_of_wires
        elif not graph == None:
            self.adj_matrix = nx.adjacency_matrix(graph).todense()
            self.numOfWires = np.size(self.adj_matrix[:,0])
            self.numOfJunctions = int(np.sum(self.adj_matrix)/2)
            self.edge_list = np.array(np.where(np.triu(self.adj_matrix) == 1)).T


class junctionState__:
    def __init__(self, NumOfJunctions, setVoltage=1e-2, resetVoltage=1e-3,
                criticalFlux=1e-1, maxFlux=1.5e-1):
        self.type = 'Atomic_Switch'
        self.voltage = np.zeros(NumOfJunctions)
        self.resistance = np.zeros(NumOfJunctions)
        self.onResistance = np.ones(NumOfJunctions)*1e4
        self.offResistance = np.ones(NumOfJunctions)*1e7

        self.filamentState = np.zeros(NumOfJunctions)
        self.OnOrOff = np.full(NumOfJunctions, False, dtype=bool)
        self.setVoltage = setVoltage
        self.resetVoltage = resetVoltage
        self.critialFlux = criticalFlux
        self.maxFlux = maxFlux

    def updateResistance(self):
        self.OnOrOff = abs(self.filamentState) >= self.critialFlux
        self.resistance = self.offResistance + \
                            (self.onResistance-self.offResistance)*self.OnOrOff

    def updateJunctionState(self, dt):
        self.filamentState = self.filamentState + \
                            (abs(self.voltage) > self.setVoltage) *\
                            (abs(self.voltage) - self.setVoltage) *\
                            np.sign(self.voltage) * dt
        self.filamentState = self.filamentState - \
                            (abs(self.voltage) < self.resetVoltage) *\
                            (self.resetVoltage - abs(self.voltage)) *\
                            np.sign(self.filamentState) * dt * 10    
        maxPosition = np.where(abs(self.filamentState) > self.maxFlux)[0]
        self.filamentState[maxPosition] = np.sign(self.filamentState[maxPosition]) * \
                                            self.maxFlux

class stimulus__:
    def __init__(self, biasType='DC',
                TimeVector=np.arange(0, 1e1, 1e-3), 
                onTime=1, offTime=2,
                onAmp=1.1, offAmp=0.005):
        if biasType == 'Drain':
            self.signal = np.zeros(TimeVector.size)
        elif biasType == 'DC':
            onIndex = np.where((TimeVector >= onTime) & (TimeVector <= offTime))
            self.signal = np.zeros(TimeVector.size)
            self.signal = np.ones(TimeVector.size) * offAmp
            self.signal[onIndex] = onAmp

def get_farthest_pairing(adjMat):
    distMat = np.zeros(adjMat.shape)
    G = nx.from_numpy_array(adjMat)
    for i in range(adjMat[:,0].size):
        for j in range(i+1, adjMat[:,0].size):
            distMat[i,j] = nx.shortest_path_length(G, i, j)
    distMat = distMat + distMat.T
    farthest = np.array(np.where(distMat == np.max(distMat)))[:,1]
    return farthest

def get_boundary_pairing(connectivity, numOfPairs=5):
    centerX = connectivity.xc
    electrodes = np.zeros(2*numOfPairs)
    electrodes[0:numOfPairs] = np.argsort(centerX)[0:numOfPairs]
    electrodes[numOfPairs:] = np.argsort(centerX)[-numOfPairs:]
    return electrodes.astype(int)

def simulateNetworkPlus(simulationOptions, 
                        connectivity, junctionState):

    if simulationOptions.contactMode == 'farthest':
        simulationOptions.electrodes = get_farthest_pairing(connectivity.adj_matrix)
        simulationOptions.interfaceElectrodes = [i+1 for i in simulationOptions.electrodes]
    elif simulationOptions.contactMode == 'Alon':
        simulationOptions.electrodes = get_boundary_pairing(connectivity)
        simulationOptions.interfaceElectrodes = [i+1 for i in simulationOptions.electrodes]

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
    adjMat = connectivity.adj_matrix
    rhs = np.zeros(V+numOfElectrodes)

    import dataStruct 
    Network = dataStruct.network__()
    Network.filamentState = np.zeros((niterations, E))
    Network.junctionVoltage = np.zeros((niterations, E))
    Network.junctionResistance = np.zeros((niterations, E))
    Network.junctionSwitch = np.zeros((niterations, E), dtype = bool)
    Network.networkCurrent = np.zeros(niterations)
    Network.wireVoltage = np.zeros((niterations, V))
    Network.electrodeCurrent = np.zeros((niterations, numOfElectrodes))

    Network.sources = []
    Network.drains = []
    for i in range(numOfElectrodes):
        if np.mean(simulationOptions.stimulus[i].signal) != 0:
            Network.sources.append(electrodes[i]+1)
        else:
            Network.drains.append(electrodes[i]+1)

    for this_time in tqdm(range(niterations), desc='Running Simulation '):
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

    Network.numOfWires = V
    Network.numOfJunctions = E
    Network.adjMat = connectivity.adj_matrix
    Network.graph = nx.from_numpy_array(connectivity.adj_matrix)
    Network.shortestPaths = [p for p in nx.all_shortest_paths(Network.graph, 
                                                        source=Network.sources[0]-1, 
                                                        target=Network.drains[0]-1)]
    Network.shortestPaths = np.add(Network.shortestPaths, 1)
    Network.contactWires = simulationOptions.interfaceElectrodes
    Network.criticalFlux = junctionState.critialFlux
    Network.stimulus = [simulationOptions.stimulus[i] for i in range(numOfElectrodes)]
    Network.junctionList = np.add(connectivity.edge_list, 1).T
    Network.connectivity = connectivity
    Network.TimeVector = simulationOptions.TimeVector

    return Network