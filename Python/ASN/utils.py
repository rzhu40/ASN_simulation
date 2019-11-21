import numpy as np
import scipy.io as sio
import networkx as nx
import time

from tqdm import tqdm
from draw import *

class simulation_options__:
    def __init__(self, dt=1e-3, T=1e1,
                contactMode = 'farthest',
                electrodes = None):
        self.dt = dt
        self.T = T
        self.TimeVector = np.arange(0, T, dt)
        self.NumOfIterations = self.TimeVector.size
        self.contactMode = contactMode
        if contactMode == 'preSet':
            self.electrodes = electrodes
        self.stimulus = []

class connectivity__:
    def __init__(self, filename=None, wires_dict=None, graph=None, adjMat = None):
        if filename != None:
            fullpath = 'connectivity_data/' + filename
            matfile = sio.loadmat(fullpath, squeeze_me=True, struct_as_record=False)
            for key in matfile.keys():
                if key[0:2] != '__':
                    setattr(self, key, matfile[key])
            self.numOfJunctions = self.number_of_junctions
            self.numOfWires = self.number_of_wires

        elif wires_dict != None:
            matfile = wires_dict
            for key in matfile.keys():
                if key[0:2] != '__':
                    setattr(self, key, matfile[key])
            self.numOfJunctions = self.number_of_junctions
            self.numOfWires = self.number_of_wires

        elif graph != None:
            self.adj_matrix = np.array(nx.adjacency_matrix(graph).todense())
            self.numOfWires = np.size(self.adj_matrix[:,0])
            self.numOfJunctions = int(np.sum(self.adj_matrix)/2)
            self.edge_list = np.array(np.where(np.triu(self.adj_matrix) == 1)).T

        elif adjMat.all() != None:
            self.adj_matrix = np.array(adjMat)
            self.numOfWires = np.size(adjMat[:,0])
            self.numOfJunctions = int(np.sum(adjMat)/2)
            self.edge_list = np.array(np.where(np.triu(adjMat) == 1)).T

class junctionState__:
    def __init__(self, numOfJunctions, 
                mode = 'binary', collapse = False,
                setVoltage=1e-2, resetVoltage=1e-3,
                onResistance=1e4, offResistance=1e7,
                criticalFlux=1e-1, maxFlux=1.5e-1):
        self.mode = mode
        self.collapse = collapse
        self.voltage = np.zeros(numOfJunctions)
        self.conductance = np.zeros(numOfJunctions)
        self.onConductance = np.ones(numOfJunctions)/onResistance
        self.offConductance = np.ones(numOfJunctions)/offResistance

        self.filamentState = np.zeros(numOfJunctions)
        self.OnOrOff = np.full(numOfJunctions, False, dtype=bool)
        self.setVoltage = setVoltage
        self.resetVoltage = resetVoltage
        self.critialFlux = criticalFlux
        self.maxFlux = maxFlux

    def updateConductance(self):
        self.OnOrOff = abs(self.filamentState) >= self.critialFlux
        if self.mode == 'binary':
            self.conductance = self.offConductance + \
                            (self.onConductance-self.offConductance)*self.OnOrOff
                            
        elif self.mode == 'tunneling':
            phi = 0.81
            C0 = 10.19
            J1 = 0.0000471307
            A = 0.17
            d = (0.1 - abs(self.filamentState))*50
            d[d<0] = 0 
            tun = 2/A * d**2 / phi**0.5 * np.exp(C0*phi**2 * d)/J1
            self.conductance = 1/(tun + 1/self.onConductance[0]) + self.offConductance[0]

    def updateJunctionState(self, dt):
        # last_sign = np.sign(self.filamentState)
        wasOpen = abs(self.filamentState) >= self.critialFlux

        self.filamentState = self.filamentState + \
                            (abs(self.voltage) > self.setVoltage) *\
                            (abs(self.voltage) - self.setVoltage) *\
                            np.sign(self.voltage) * dt
        self.filamentState = self.filamentState - \
                            (self.resetVoltage > abs(self.voltage)) *\
                            (self.resetVoltage - abs(self.voltage)) *\
                            np.sign(self.filamentState) * dt * 10    
                            
        # this_sign = np.sign(self.filamentState)
        # change = abs(this_sign - last_sign)
        # self.filamentState[np.where(change == 2)[0]] = 0

        self.filamentState[abs(self.filamentState) > self.maxFlux] = \
                    np.sign(self.filamentState[abs(self.filamentState) > self.maxFlux]) * self.maxFlux

        if self.collapse:
            justClosed = wasOpen & (abs(self.filamentState) < self.critialFlux)
            self.filamentState[justClosed] = self.filamentState[np.where(justClosed)[0]] / 10

class stimulus__:
    def __init__(self, biasType='DC',
                TimeVector=np.arange(0, 1e1, 1e-3), 
                onTime=1, offTime=2,
                onAmp=1.1, offAmp=0.005,
                f = 1, shift = 0, 
                customSignal = None):

        period = 1/f
        offIndex = np.where((TimeVector<=onTime) + (TimeVector>=offTime))
        if biasType == 'Drain':
            self.signal = np.zeros(TimeVector.size)
        elif biasType == 'DC':
            onIndex = np.where((TimeVector >= onTime) & (TimeVector <= offTime))
            self.signal = np.ones(TimeVector.size) * offAmp
            self.signal[onIndex] = onAmp
        elif biasType == 'AC':
            self.signal = onAmp*np.sin(2*np.pi*f*TimeVector)
        elif biasType == 'Square':
            self.signal = onAmp * (-np.sign(TimeVector % period - period/2))
        elif biasType == 'Triangular':
            self.signal = 4*onAmp/period * abs((TimeVector-period/4) % period - period/2) - onAmp
        elif biasType == 'Sawtooth':
            self.signal = onAmp/period * (TimeVector % period)
        elif biasType == 'Pulse':
            self.signal= (onAmp-offAmp)*((TimeVector % period) < period/2) + np.ones(TimeVector.size)*offAmp
        elif biasType == 'MKG':
            from analysis.mkg import mkg_generator
            mkg_period = int(1/f/(TimeVector[1]-TimeVector[0]))
            self.signal = mkg_generator(TimeVector.size, dt = 100/mkg_period)*onAmp
        elif biasType == 'Custom': 
            if len(customSignal) >= TimeVector.size:
                self.signal = np.array(customSignal[0:TimeVector.size])
            else:
                print(f'Signal length is shorter to simulation length. Current TimeVector length is {TimeVector.size}!')
                from sys import exit
                exit()
        else:
            # self.signal = np.ones(TimeVector.size) * offAmp
            print('Stimulus type error. Options are: ')
            print('Drain | DC | AC | Square | Triangular | Sawtooth | Pulse | MKG | Custom')
            from sys import exit
            exit()

        if shift != 0:
            self.signal += shift

        if (biasType != 'Custom') and (biasType != 'Drain'):
            self.signal[offIndex] = offAmp
            
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

def simulateNetwork(simulationOptions, connectivity, junctionState, lite_mode = False, disable_tqdm = False, save_steps = 1):

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
    Network.junctionSwitch = np.zeros((niterations, E), dtype = bool)
    if lite_mode:
        Network.filamentState = np.zeros((int(niterations/save_steps), E))
        Network.wireVoltage = np.zeros((int(niterations/save_steps), V))
        Network.TimeVector = np.zeros(int(niterations/save_steps))
    else:        
        Network.filamentState = np.zeros((niterations, E))
        Network.junctionVoltage = np.zeros((niterations, E))
        # Network.junctionResistance = np.zeros((niterations, E))
        Network.junctionConductance = np.zeros((niterations, E))
        Network.junctionSwitch = np.zeros((niterations, E), dtype = bool)
        Network.wireVoltage = np.zeros((niterations, V))
        Network.electrodeCurrent = np.zeros((niterations, numOfElectrodes))

    Network.sources = []
    Network.drains = []
    for i in range(numOfElectrodes):
        if np.mean(simulationOptions.stimulus[i].signal) != 0:
            Network.sources.append(electrodes[i])
        else:
            Network.drains.append(electrodes[i])
    if len(Network.drains) == 0:
        Network.drains.append(electrodes[1])

    for this_time in tqdm(range(niterations), desc='Running Simulation ', disable = disable_tqdm):
        junctionState.updateConductance()
        junctionConductance = junctionState.conductance
        
        Gmat = np.zeros((V,V))
        Gmat[edgeList[:,0], edgeList[:,1]] = junctionConductance
        Gmat[edgeList[:,1], edgeList[:,0]] = junctionConductance
        Gmat = np.diag(np.sum(Gmat,axis=0)) - Gmat

        lhs = np.zeros((V+numOfElectrodes, V+numOfElectrodes))
        lhs[0:V,0:V] = Gmat
        for i in range(numOfElectrodes):
            this_elec = electrodes[i]
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

        if lite_mode:
            if this_time%save_steps == 0:
                Network.wireVoltage[this_time//save_steps,:] = wireVoltage
                Network.filamentState[this_time//save_steps,:] = junctionState.filamentState
                Network.TimeVector[this_time//save_steps] = simulationOptions.TimeVector[this_time]
        else:
            Network.wireVoltage[this_time,:] = wireVoltage
            Network.electrodeCurrent[this_time,:] = sol[V:]
            Network.filamentState[this_time,:] = junctionState.filamentState
            Network.junctionVoltage[this_time,:] = junctionState.voltage
            Network.junctionConductance[this_time,:] = junctionState.conductance
            Network.junctionSwitch[this_time,:] = junctionState.OnOrOff

    Network.numOfWires = V
    Network.numOfJunctions = E
    # Network.adjMat = connectivity.adj_matrix
    # Network.graph = nx.from_numpy_array(connectivity.adj_matrix)
    # if nx.has_path(Network.graph, Network.sources[0], Network.drains[0]):
    #     Network.shortestPaths = [p for p in nx.all_shortest_paths(Network.graph, 
    #                                                         source=Network.sources[0], 
    #                                                         target=Network.drains[0])]
    Network.electrodes = simulationOptions.electrodes
    Network.criticalFlux = junctionState.critialFlux
    Network.stimulus = [simulationOptions.stimulus[i] for i in range(numOfElectrodes)]
    Network.connectivity = connectivity
    if not lite_mode:
        Network.junctionResistance = 1/Network.junctionConductance
        Network.TimeVector = simulationOptions.TimeVector
    return Network

def defaultSimulation(Connectivity, 
                    junctionMode = 'binary', collapse = False,
                    contactMode='farthest', electrodes=None,
                    dt= 1e-3, T=10, 
                    biasType = 'DC',
                    onTime=0, offTime=5,
                    onAmp=1.1, offAmp=0.005,
                    f = 1, customSignal = None,
                    lite_mode = False, save_steps = 1,
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
                            f= f, customSignal= customSignal)
    SimulationOptions.stimulus.append(tempStimulus)

    JunctionState = junctionState__(Connectivity.numOfJunctions, mode = junctionMode, collapse = collapse)

    this_realization = simulateNetwork(SimulationOptions, Connectivity, JunctionState, lite_mode, disable_tqdm, save_steps)
    
    if findFirst:
        from analysis.GraphTheory import findCurrent

        try:
            activation = findCurrent(this_realization, 1)
            print(f'First current path {activation[0][0]} formed at time = {activation[1][0]} s.')
        except:
            print('Unfortunately, no current path is formed in simulation time.')

    return this_realization

def generate_network(numOfWires = 100, dispersion=100, mean_length = 100, this_seed = 42, iterations=0, max_iters = 10):
    import wires
    wires_dict = wires.generate_wires_distribution(number_of_wires = numOfWires,
                                            wire_av_length = mean_length,
                                            wire_dispersion = 20,
                                            gennorm_shape = 3,
                                            centroid_dispersion = dispersion,
                                            Lx = 5e2,
                                            Ly = 5e2,
                                            this_seed = this_seed)

    wires_dict = wires.detect_junctions(wires_dict)
    wires_dict = wires.generate_graph(wires_dict)
    if wires.check_connectedness(wires_dict):
        return wires_dict
    elif iterations < max_iters: 
        generate_network(numOfWires, dispersion, mean_length, this_seed+1, iterations+1, max_iters)
    else:
        return None

def findJunctionIndex(connectivity, wire1, wire2):
    edgeList = connectivity.edge_list
    index = np.where((edgeList[:,0] == wire1) & (edgeList[:,1] == wire2))[0]
    if len(index) == 0:
        index = np.where((edgeList[:,0] == wire2) & (edgeList[:,1] == wire1))[0]
    if len(index) == 0:
        print(f'Unfortunately, no junction between nanowire {wire1} and {wire2}')
        return None
    else:
        return index[0]

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

import multiprocessing.pool as mpp
def istarmap(self, func, iterable, chunksize=1):
    """
    starmap-version of imap
    """
    
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap

def check_memory():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print(f'Current Memory usage is {process.memory_info().rss/1e6} MB.')