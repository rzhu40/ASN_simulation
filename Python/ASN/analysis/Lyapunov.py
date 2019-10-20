import numpy as np
from utils import *

def junctionLyapunov(network, epsilon = 1e-4, 
                    dt = 1e-3, T = 1,
                    biasType = 'AC', onAmp = 4):

    SimulationOptions = simulation_options__(dt = dt, T = T,
                                        contactMode = 'preSet',
                                        electrodes = network.electrodes)

    Connectivity = network.connectivity

    tempStimulus = stimulus__(biasType = biasType, 
        TimeVector = SimulationOptions.TimeVector, 
        onTime = 0, offTime = T,
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