import numpy as np
from utils import *

def junctionLyapunov(network, epsilon = 1e-4, 
                    dt = 1e-3, T = 1, f = 1,
                    biasType = 'AC', onAmp = 4):

    SimulationOptions = simulation_options__(dt = dt, T = T,
                                        contactMode = 'preSet',
                                        electrodes = network.electrodes)

    Connectivity = network.connectivity

    tempStimulus = stimulus__(biasType = biasType, 
        TimeVector = SimulationOptions.TimeVector, 
        onTime = 0, offTime = T, f = f,
        onAmp = onAmp, offAmp = 0.005)
    SimulationOptions.stimulus.append(tempStimulus)

    tempStimulus = stimulus__(biasType = 'Drain', 
            TimeVector = SimulationOptions.TimeVector)
    SimulationOptions.stimulus.append(tempStimulus)

    JunctionState = junctionState__(Connectivity.numOfJunctions, mode = 'tunneling')

    normalSim = simulateNetwork(SimulationOptions, Connectivity, JunctionState)
    normal_lambda = normalSim.filamentState
    
    E = Connectivity.numOfJunctions
    V = Connectivity.numOfWires

    niterations = SimulationOptions.NumOfIterations
    electrodes = SimulationOptions.electrodes
    numOfElectrodes = len(electrodes)
    edgeList = Connectivity.edge_list
    rhs = np.zeros(V+numOfElectrodes)

    Lyapunov = np.zeros((niterations, E))
    avgLyapunov = np.zeros(E)

    for this_junction in tqdm(range(E), desc = 'Running Lyapunov'):
        # perturb initial condition of one junction's filament state
        temp_junctionState = junctionState__(Connectivity.numOfJunctions, mode = 'tunneling')
        temp_junctionState.filamentState[this_junction] = epsilon
        
        for this_time in range(niterations):
            temp_junctionState.updateResistance()
            junctionConductance = 1/temp_junctionState.resistance

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
                rhs[V+i] = SimulationOptions.stimulus[i].signal[this_time]

            # from scipy.sparse import csc_matrix
            # from scipy.sparse.linalg import spsolve
            # LHS = csc_matrix(lhs)
            # RHS = csc_matrix(rhs.reshape(V+numOfElectrodes,1))
            # sol = spsolve(LHS,RHS)

            sol = np.linalg.solve(lhs,rhs)
            wireVoltage = sol[0:V]
            temp_junctionState.voltage = wireVoltage[edgeList[:,0]] - wireVoltage[edgeList[:,1]]
            temp_junctionState.updateJunctionState(SimulationOptions.dt)

            delta_lambda = temp_junctionState.filamentState - normal_lambda[this_time,:]
            norm_delta = np.linalg.norm(delta_lambda)

            if norm_delta == 0:
                Lyapunov[this_time:,this_junction] = -np.Inf
                break
            
            temp_junctionState.filamentState = normal_lambda[this_time,:] + epsilon/norm_delta * delta_lambda
            # Lyapunov[this_time,this_junction] = np.log(np.max(abs(delta_lambda/epsilon)))
            Lyapunov[this_time, this_junction] = np.log(norm_delta/epsilon)
        
        nonInf = np.where(1-np.isinf(Lyapunov[:,this_junction]))[0]
        avgLyapunov[this_junction] = np.mean(Lyapunov[nonInf,this_junction])
    
    network.Lyapunov = avgLyapunov
    return avgLyapunov

def wireLyapunov(network, epsilon = 1e-4, 
                    dt = 1e-3, T = 1, f = 1,
                    biasType = 'AC', onAmp = 4):

    SimulationOptions = simulation_options__(dt = dt, T = T,
                                        contactMode = 'preSet',
                                        electrodes = network.electrodes)

    Connectivity = network.connectivity

    tempStimulus = stimulus__(biasType = biasType, 
        TimeVector = SimulationOptions.TimeVector, 
        onTime = 0, offTime = T, f = f,
        onAmp = onAmp, offAmp = 0.005)
    SimulationOptions.stimulus.append(tempStimulus)

    tempStimulus = stimulus__(biasType = 'Drain', 
            TimeVector = SimulationOptions.TimeVector)
    SimulationOptions.stimulus.append(tempStimulus)

    JunctionState = junctionState__(Connectivity.numOfJunctions, mode = 'tunneling')

    normalSim = simulateNetwork(SimulationOptions, Connectivity, JunctionState)
    normal_wireVoltage = normalSim.wireVoltage
    
    E = Connectivity.numOfJunctions
    V = Connectivity.numOfWires

    niterations = SimulationOptions.NumOfIterations
    electrodes = SimulationOptions.electrodes
    numOfElectrodes = len(electrodes)
    edgeList = Connectivity.edge_list
    rhs = np.zeros(V+numOfElectrodes)

    Lyapunov = np.zeros((niterations, V))
    avgLyapunov = np.zeros(V)

    for this_wire in tqdm(range(V), desc = 'Running Lyapunov'):
        # perturb initial condition of one wire's voltage
        temp_junctionState = junctionState__(Connectivity.numOfJunctions, mode = 'tunneling')
        init_wireVoltage = normal_wireVoltage[0,:]
        init_wireVoltage[this_wire] = init_wireVoltage[this_wire] * (1+epsilon)
        temp_junctionState.voltage = init_wireVoltage[edgeList[:,0]] - init_wireVoltage[edgeList[:,1]]

        for this_time in range(niterations):
            temp_junctionState.updateResistance()
            junctionConductance = 1/temp_junctionState.resistance

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
                rhs[V+i] = SimulationOptions.stimulus[i].signal[this_time]

            # from scipy.sparse import csc_matrix
            # from scipy.sparse.linalg import spsolve
            # LHS = csc_matrix(lhs)
            # RHS = csc_matrix(rhs.reshape(V+numOfElectrodes,1))
            # sol = spsolve(LHS,RHS)

            sol = np.linalg.solve(lhs,rhs)
            wireVoltage = sol[0:V]
            delta_wireV = wireVoltage - normal_wireVoltage[this_time,:]
            norm_delta = np.linalg.norm(delta_wireV)
            if norm_delta == 0:
                Lyapunov[this_time:,this_wire] = -np.Inf
                break
            wireVoltage = normal_wireVoltage[this_time,:] + epsilon/norm_delta * delta_wireV

            temp_junctionState.voltage = wireVoltage[edgeList[:,0]] - wireVoltage[edgeList[:,1]]
            temp_junctionState.updateJunctionState(SimulationOptions.dt)
            
            # Lyapunov[this_time,this_junction] = np.log(np.max(abs(delta_lambda/epsilon)))
            Lyapunov[this_time, this_wire] = np.log(norm_delta/(epsilon*init_wireVoltage[this_wire]))
        
        # nonInf = np.where(1-np.isinf(Lyapunov[:,this_wire]))[0]
        # avgLyapunov[this_wire] = np.mean(Lyapunov[nonInf,this_wire])
    avgLyapunov = np.mean(Lyapunov)
    network.Lyapunov = avgLyapunov
    return Lyapunov