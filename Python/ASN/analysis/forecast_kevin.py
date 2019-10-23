import numpy as np
from utils import *

def expander(some, scale = 10):
    output = np.zeros(scale*len(some))
    count = 0
    for i in range(len(some)):
        output[count:count+scale] = some[i]
        count += scale
    return output

def pre_activation(sim_options, connectivity, junctionState):
    SimulationOptions = simulation_options__(dt = sim_options.dt, T = 1,
                                            contactMode = sim_options.contactMode,
                                            electrodes = sim_options.electrodes)

    tempStimulus = stimulus__(biasType = 'DC', 
        TimeVector = SimulationOptions.TimeVector, 
        onTime = 0, offTime = 20,
        onAmp = 10, offAmp = 0.005)
    SimulationOptions.stimulus.append(tempStimulus)
    
    pre = simulateNetwork(SimulationOptions, connectivity, junctionState)
    return pre

def getWeight(wireVoltage, stimulus, non_elec):
    training_length, V = wireVoltage.shape
    num_non_elec = len(non_elec)
    lhs = np.zeros((training_length-2, 2*num_non_elec+2))
    rhs = stimulus[2:training_length]

    lhs[:,0] = np.ones(training_length-2)
    lhs[:,1] = stimulus[1:training_length-1]
    lhs[:,2:num_non_elec+2] = wireVoltage[1:training_length-1,non_elec]
    lhs[:,num_non_elec+2:] = wireVoltage[0:training_length-2,non_elec]
    # weight = np.linalg.lstsq(lhs, rhs)[0].reshape(2*num_non_elec+2, 1)
    
    from scipy.optimize import lsq_linear
    weight = lsq_linear(lhs, rhs, (-10,10))['x']
    return weight

def forecast_kev(simulationOptions, connectivity, junctionState, training_ratio = 0.5, exclude_electrode = False, pre_activate = True):
    if simulationOptions.contactMode == 'farthest':
        simulationOptions.electrodes = get_farthest_pairing(connectivity.adj_matrix)
    elif simulationOptions.contactMode == 'Alon':
        simulationOptions.electrodes = get_boundary_pairing(connectivity, 28)

    niterations = simulationOptions.NumOfIterations
    training_length = int(niterations * training_ratio)
    electrodes = simulationOptions.electrodes
    numOfElectrodes = len(electrodes)
    E = connectivity.numOfJunctions
    V = connectivity.numOfWires    
    if exclude_electrode:
        non_elec = np.setdiff1d(np.arange(V), electrodes)
    else:
        non_elec = np.arange(V)
    num_non_elec = len(non_elec)

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
    for i in range(numOfElectrodes):
        if np.mean(simulationOptions.stimulus[i].signal) != 0:
            Network.sources.append(electrodes[i])
        else:
            Network.drains.append(electrodes[i])
    if len(Network.drains) == 0:
        Network.drains.append(electrodes[1])
    if pre_activate:
        pre = pre_activation(simulationOptions, connectivity, junctionState)

    for this_time in tqdm(range(training_length), desc='Training weight vector ❤ Big Kev 112%'):
        junctionState.updateResistance()
        junctionConductance = 1/junctionState.resistance
        
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

        Network.wireVoltage[this_time,:] = wireVoltage
        Network.electrodeCurrent[this_time,:] = sol[V:]
        Network.filamentState[this_time,:] = junctionState.filamentState
        Network.junctionVoltage[this_time,:] = junctionState.voltage
        Network.junctionResistance[this_time,:] = junctionState.resistance
        Network.junctionSwitch[this_time,:] = junctionState.OnOrOff

    weight = getWeight(Network.wireVoltage[:training_length,:], simulationOptions.stimulus[0].signal[:training_length], non_elec)
    # import matplotlib.pyplot as plt
    # plt.plot(weight)
    # plt.show()
    forecast = np.zeros(niterations)
    forecast[:training_length] = simulationOptions.stimulus[0].signal[:training_length]

    for this_time in tqdm(range(training_length, niterations), desc='Forecasting '):
        junctionState.updateResistance()
        junctionConductance = 1/junctionState.resistance
        
        Gmat = np.zeros((V,V))
        Gmat[edgeList[:,0], edgeList[:,1]] = junctionConductance
        Gmat[edgeList[:,1], edgeList[:,0]] = junctionConductance
        Gmat = np.diag(np.sum(Gmat, axis=0)) - Gmat

        lhs = np.zeros((V+numOfElectrodes, V+numOfElectrodes))
        lhs[0:V,0:V] = Gmat

        last_V2 = Network.wireVoltage[this_time-2,non_elec]
        last_V = Network.wireVoltage[this_time-1,non_elec]
        hist = np.zeros(2*num_non_elec+2)
        hist[0] = 1
        hist[1] = forecast[this_time-1]
        hist[2:num_non_elec+2] = last_V
        hist[num_non_elec+2:] = last_V2
        forecast[this_time] = np.dot(hist, weight)

        for i in range(numOfElectrodes):
            this_elec = electrodes[i]
            lhs[V+i, this_elec] = 1
            lhs[this_elec, V+i] = 1
            if i == 0:
                rhs[V+i] = forecast[this_time]
        
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
                                                        source=Network.sources[0], 
                                                        target=Network.drains[0])]

    Network.electrodes = simulationOptions.electrodes
    Network.criticalFlux = junctionState.critialFlux
    Network.stimulus = [simulationOptions.stimulus[i] for i in range(numOfElectrodes)]
    # Network.junctionList = np.add(connectivity.edge_list, 1).T
    Network.connectivity = connectivity
    Network.TimeVector = simulationOptions.TimeVector
    Network.forecast = forecast
    return Network, weight

if __name__ == '__main__':
    from analysis.mkg import mkg_generator
    SimulationOptions = simulation_options__(dt = 1e-3, T = 10,
                                            contactMode = 'farthest',
                                            electrodes = [72,29])

    Connectivity = connectivity__(
        filename = '2016-09-08-155153_asn_nw_00100_nj_00261_seed_042_avl_100.00_disp_10.00.mat')

    # Connectivity = connectivity__(
    #     filename = '2016-09-08-155044_asn_nw_00700_nj_14533_seed_042_avl_100.00_disp_10.00.mat')

    tempStimulus = stimulus__(biasType = 'AC', 
            TimeVector = SimulationOptions.TimeVector, 
            onTime = 0, offTime = 20,
            onAmp = 5, offAmp = 0.005)

    # signal = mkg_generator(10000)
    # signal = (np.sin(SimulationOptions.TimeVector))**2 + 2*(np.cos(SimulationOptions.TimeVector))**3

    # tempStimulus = stimulus__(biasType = 'Custom', TimeVector = SimulationOptions.TimeVector, customSignal = signal)

    # tempStimulus.signal = expander(tempStimulus.signal,2)
    SimulationOptions.stimulus.append(tempStimulus)

    tempStimulus = stimulus__(biasType = 'Drain', 
            TimeVector = SimulationOptions.TimeVector)
    SimulationOptions.stimulus.append(tempStimulus)

    JunctionState = junctionState__(Connectivity.numOfJunctions, mode = 'tunneling')

    sim1, weight = forecast_kev(SimulationOptions, Connectivity, JunctionState, 0.8, exclude_electrode = False, pre_activate = True)