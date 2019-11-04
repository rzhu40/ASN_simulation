import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    if training_length-steps < E*steps+2:
        from sys import exit
        exit()
    lhs = np.zeros((training_length-steps, E*steps+2))
    rhs = stimulus[steps:training_length]
    lhs[:,0] = 1
    # lhs[:,1] = stimulus[0:training_length-steps]
    # lhs[:,1] = stimulus[steps-1:training_length-1]
    for i in range(steps):
        lhs[:,i*E+2:(i+1)*E+2] = measure[steps-i-1:training_length-i-1,:]
    if WLS:
        diff = abs(stimulus - target)[steps:training_length]
        w = np.zeros(len(diff))
        w[np.where(diff == 0)[0]] = 1
        w[np.where(diff != 0)[0]] = 1/diff[np.where(diff != 0)[0]]
        lhs = np.dot(np.diag(w), lhs)
        rhs = np.dot(np.diag(w), rhs)
        
    weight = np.linalg.lstsq(lhs,rhs)[0].reshape(E*steps+2, 1)
    return weight

def pre_activation(sim_options, connectivity, junctionState):
    SimulationOptions = simulation_options__(dt = sim_options.dt, T = 5,
                                            contactMode = sim_options.contactMode,
                                            electrodes = sim_options.electrodes)

    # tempStimulus = stimulus__(biasType = 'DC', 
    #     TimeVector = SimulationOptions.TimeVector, 
    #     onTime = 0, offTime = 20,
    #     onAmp = 10, offAmp = 0.005)
    # SimulationOptions.stimulus.append(tempStimulus)
    SimulationOptions.stimulus = sim_options.stimulus
    
    pre = simulateNetwork(SimulationOptions, connectivity, junctionState)
    return pre

def forecast(simulationOptions, connectivity, junctionState, 
            training_ratio = 0.5, steps = 1,
            measure_type = 'conductance',
            pre_activate = False,
            forecast_on = False, 
            cheat_on = False, cheat_period = 75, cheat_steps = 25,
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
    
    if pre_activate:
        pre = pre_activation(simulationOptions, connectivity, junctionState)

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
    
    # junctionList = np.where(connectivity.adj_matrix[Network.drains[0],:] == 1)[0]
    # junctionList = [0, 0, 0, 0]
    outList = np.where(connectivity.adj_matrix[Network.drains[0],:] == 1)[0]
    junctionList = [findJunctionIndex(connectivity, Network.drains[0], i) for i in outList]

    if measure_type == 'conductance':
        stimulus_packer = np.array([simulationOptions.stimulus[0].signal[:training_length]]*len(junctionList)).T
        measure = Network.junctionVoltage[:training_length,junctionList]/Network.junctionResistance[:training_length,junctionList]/stimulus_packer
    elif measure_type == 'filament':
        measure = Network.filamentState[:training_length,junctionList]
    elif measure_type == 'current':
        measure = Network.junctionVoltage[:training_length,junctionList]/Network.junctionResistance[:training_length,junctionList]
    elif measure_type == 'voltage':
        measure = Network.wireVoltage[:training_length,outList]

    weight = getWeight(measure, simulationOptions.stimulus[0].signal[:training_length], steps)
    predict = np.zeros(niterations)
    predict[:training_length] = simulationOptions.stimulus[0].signal[:training_length]
    hist_list = np.zeros(20*steps+2)
    
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
        # hist[1] = forecast[this_time-1]
        hist[1] = 0

        for i in range(steps):
            hist[i*paras+2:(i+1)*paras+2] = measure[this_time-1-i,:]
        hist_list = np.vstack((hist_list, hist))
        predict[this_time] = np.dot(hist, weight)

        for i in range(numOfElectrodes):
            this_elec = electrodes[i]
            lhs[V+i, this_elec] = 1
            lhs[this_elec, V+i] = 1
            if i == 0:
                if forecast_on:
                    if cheat_on & (this_time % cheat_period > cheat_period-cheat_steps-1):
                        predict[this_time] = simulationOptions.stimulus[i].signal[this_time]
                    rhs[V+i] = predict[this_time]
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
            this_measure = Network.junctionVoltage[this_time,junctionList]/Network.junctionResistance[this_time,junctionList]/predict[this_time]
        elif measure_type == 'filament':
            this_measure = Network.filamentState[this_time,junctionList]
        elif measure_type == 'current':
            this_measure = Network.junctionVoltage[this_time,junctionList]/Network.junctionResistance[this_time,junctionList]
        elif measure_type == 'voltage':
            this_measure = Network.wireVoltage[this_time,outList]
        measure = np.vstack((measure, this_measure))
        
        if update_weight & (this_time % update_stepsize == 0):
            weight = getWeight(measure[:this_time,:],simulationOptions.stimulus[0].signal[:this_time], steps)
            # weight = getWeight(measure[:this_time,:],forecast[:this_time], steps)

    
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
    Network.forecast = predict
    Network.outList = outList
    Network.junctionList = junctionList
    Network.weight = weight
    Network.hist_list = hist_list
    return Network, weight, measure

def plotForecastPanel(Network):
    fig = plt.figure(tight_layout=True, figsize = (25,10))
    gs = gridspec.GridSpec(2, 5)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(Network.stimulus[0].signal)
    ax.plot(Network.forecast)

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(range(len(Network.weight)), Network.weight, marker = 'x')
    ax.scatter([1], Network.weight[1], color = 'r')
    ax.set_title('weight')

    ax = fig.add_subplot(gs[1, 1])
    for i in Network.outList:
        plt.plot(Network.wireVoltage[:,i])
    ax.set_title('voltage')

    ax = fig.add_subplot(gs[1, 2])
    for i in Network.junctionList:
        plt.plot(Network.junctionVoltage[:,i]/Network.junctionResistance[:,i])
    ax.set_title('current')

    ax = fig.add_subplot(gs[1, 3])
    for i in Network.junctionList:
        plt.plot(Network.junctionVoltage[:,i]/Network.junctionResistance[:,i]/Network.stimulus[0].signal)
    ax.set_title('conductance')

    ax = fig.add_subplot(gs[1, 4])
    totalOn = np.sum(Network.junctionSwitch, axis = 1)
    ax.plot(totalOn)
    ax.set_title('# of ON switches')
    return None
                 
if __name__ == '__main__':
    haha = connectivity__(filename = '200nw_1213junctions.mat')
    outList = [26, 90, 58, 113, 161, 96, 125, 38, 89, 63, 198, 45, 55, 162, 122, 137, 97, 61, 17, 190]
    inList = [51, 80, 92, 126, 184, 139, 32, 72, 116, 23]
    theMat = np.zeros((202, 202))
    theMat[0:200, 0:200] = haha.adj_matrix
    theMat[200, inList] = 1
    theMat[inList, 200] = 1
    theMat[201, outList] = 1
    theMat[outList, 201] = 1
    
    theGraph = nx.from_numpy_array(theMat)
    Connectivity = connectivity__(graph = theGraph)
    junctionList = [findJunctionIndex(Connectivity, 201, i) for i in outList]
    
    SimulationOptions = simulation_options__(dt = 1e-2, T = 20,
                                        contactMode = 'preSet',
                                        electrodes = [200, 201])

# tempStimulus = stimulus__(TimeVector = SimulationOptions.TimeVector, biasType = 'AC', onAmp = 1, onTime = 0, offTime = 20, f = 1.5)
# tempStimulus.signal = tempStimulus.signal*3 - 2.95
    from analysis.mkg import mkg_generator
    signal = mkg_generator(2000, tau = 18, a = 0.2, b = 0.1, dt = 1)*2 - 0.49
    # signal = mkg_generator(2000, tau = 17, a = 1, b = 0.1, dt = 1)*3 - 2.95
    tempStimulus = stimulus__(biasType = 'Custom', TimeVector = SimulationOptions.TimeVector, customSignal = signal)
    
    SimulationOptions.stimulus.append(tempStimulus)
    tempStimulus = stimulus__(biasType = 'Drain', 
            TimeVector = SimulationOptions.TimeVector)
    SimulationOptions.stimulus.append(tempStimulus)
    JunctionState = junctionState__(Connectivity.numOfJunctions, mode = 'binary', collapse = True)
    
#    sim1, weight1, measure1 = forecast(SimulationOptions, Connectivity, JunctionState,
#                                    0.5, 50, forecast_on = False, measure_type = 'current', pre_activate = False, cheat = False, update_weight = False, update_stepsize = 1)
    
    sim2, weight2, measure2 = forecast(SimulationOptions, Connectivity, JunctionState,
                                    0.5, 50, forecast_on = True, measure_type = 'current', pre_activate = False, cheat = False, update_weight = False, update_stepsize = 1)
    plotForecastPanel(sim2)