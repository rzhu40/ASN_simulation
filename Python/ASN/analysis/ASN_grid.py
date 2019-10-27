import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import *

def generate_ASN_grid(numOfNetworks, wires_per_network):
    import wires
    adj_list = []
    graph_list = []
    for i in range(numOfNetworks):
        wires_dict = wires.generate_wires_distribution(number_of_wires = wires_per_network,
                                                    wire_av_length = 150,
                                                    wire_dispersion = 50,
                                                    gennorm_shape = 3,
                                                    centroid_dispersion = 60,
                                                    Lx = 3e2,
                                                    Ly = 3e2,
                                                    this_seed = i*5)

        wires_dict = wires.detect_junctions(wires_dict)
        wires_dict = wires.generate_graph(wires_dict)
        while not wires.check_connectedness(wires_dict):
            wires_dict = wires.generate_wires_distribution(number_of_wires = wires_per_network,
                                                    wire_av_length = 150,
                                                    wire_dispersion = 50,
                                                    gennorm_shape = 3,
                                                    centroid_dispersion = 60,
                                                    Lx = 3e2,
                                                    Ly = 3e2,
                                                    this_seed = np.random.randint(1000))
            wires_dict = wires.detect_junctions(wires_dict)
            wires_dict = wires.generate_graph(wires_dict)
        adj_list.append(wires_dict['adj_matrix'])
        graph_list.append(wires_dict['G'])
    
    num_wires_list = [len(graph_list[i]) for i in range(len(graph_list))]
    total_wires = sum(num_wires_list)
    bigMat = np.zeros((total_wires+2, total_wires+2))
    inList = np.zeros(len(adj_list))
    outList = np.zeros(len(adj_list))

    node_count = 0
    for i in range(len(adj_list)):
        bigMat[node_count:node_count+num_wires_list[i], node_count:node_count+num_wires_list[i]] = adj_list[i]
        inList[i], outList[i] = get_farthest_pairing(adj_list[i]) + node_count
        node_count += num_wires_list[i]

    inList = inList.astype(int)
    outList = outList.astype(int)
    bigMat[total_wires, inList] = 1
    bigMat[inList, total_wires] = 1
    bigMat[total_wires+1, outList] = 1
    bigMat[outList, total_wires+1] = 1

    return bigMat

def getWeight(measure, stimulus, steps = 1, WLS = False, target = None):
    training_length, E = measure.shape
    lhs = np.zeros((training_length-steps, E*steps+2))
    rhs = stimulus[steps:training_length]
    lhs[:,0] = 1
    lhs[:,1] = stimulus[0:training_length-steps]
    for i in range(steps):
        lhs[:,i*E+2:(i+1)*E+2] = measure[steps-i:training_length-i,:]
    if WLS:
        diff = abs(stimulus - target)[steps:training_length]
        w = np.zeros(len(diff))
        w[np.where(diff == 0)[0]] = 1
        w[np.where(diff != 0)[0]] = 1/diff[np.where(diff != 0)[0]]
        lhs = np.dot(np.diag(w), lhs)
        rhs = np.dot(np.diag(w), rhs)
    weight = np.linalg.lstsq(lhs,rhs)[0].reshape(E*steps+2, 1)
    return weight

def forecast(simulationOptions, connectivity, junctionState, 
            training_ratio = 0.5, steps = 1,
            measure_type = 'conductance', forecast_on = False,
            update_weight = False, update_stepsize = 1):
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

    for this_time in tqdm(range(training_length), desc='Training weight vector '):
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
    
    junctionList = np.where(connectivity.adj_matrix[Network.drains[0],:] == 1)[0]

    if measure_type == 'conductance':
        stimulus_packer = np.array([simulationOptions.stimulus[0].signal[:training_length]]*len(junctionList)).T
        measure = Network.junctionVoltage[:training_length,junctionList]/Network.junctionResistance[:training_length,junctionList]/stimulus_packer
    elif measure_type == 'filament':
        measure = Network.filamentState[:training_length,junctionList]
    elif measure_type == 'current':
        measure = Network.junctionVoltage[:training_length,junctionList]/Network.junctionResistance[:training_length,junctionList]

    weight = getWeight(measure, simulationOptions.stimulus[0].signal[:training_length], steps)
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
        paras = len(junctionList)
        hist = np.zeros(paras*steps+2)
        hist[0] = 1
        hist[1] = forecast[this_time-1]
        for i in range(steps):
            hist[i*paras+2:(i+1)*paras+2] = measure[this_time-1-i]
        
        forecast[this_time] = np.dot(hist, weight)

        for i in range(numOfElectrodes):
            this_elec = electrodes[i]
            lhs[V+i, this_elec] = 1
            lhs[this_elec, V+i] = 1
            if i == 0:
                if forecast_on:
                    rhs[V+i] = forecast[this_time]
                else:
                    rhs[V+i] = simulationOptions.stimulus[i].signal[this_time]

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
        if measure_type == 'conductance':
            this_measure = Network.junctionVoltage[this_time,junctionList]/Network.junctionResistance[this_time,junctionList]/simulationOptions.stimulus[0].signal[this_time]
        elif measure_type == 'filament':
            this_measure = Network.filamentState[this_time, junctionList]
        elif measure_type == 'current':
            this_measure = Network.junctionVoltage[this_time,junctionList]/Network.junctionResistance[this_time,junctionList]
        measure = np.vstack((measure, this_measure))
        
        if update_weight & (this_time % update_stepsize == 0):
            weight = getWeight(measure[:this_time,:],simulationOptions.stimulus[0].signal[:this_time], steps)

    
    Network.numOfWires = V
    Network.numOfJunctions = E
    Network.adjMat = connectivity.adj_matrix
    Network.graph = nx.from_numpy_array(connectivity.adj_matrix)
    Network.shortestPaths = [p for p in nx.all_shortest_paths(Network.graph, 
                                                        source=Network.sources[0], 
                                                        target=Network.drains[0])]

    Network.electrodes = simulationOptions.electrodes
    Network.criticalFlux = junctionState.critialFlux
    Network.stimulus = simulationOptions.stimulus
    Network.connectivity = connectivity
    Network.TimeVector = simulationOptions.TimeVector
    Network.forecast = forecast
    return Network, weight, measure