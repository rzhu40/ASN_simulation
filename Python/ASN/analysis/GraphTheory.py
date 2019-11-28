import numpy as np
import networkx as nx
from itertools import islice

# def generateGrid()
def getOnGraph(network, this_TimeStamp = 0, isDirected = True):
    if isDirected:
        edgeList = network.connectivity.edge_list
        diMat = np.zeros(network.connectivity.adj_matrix.shape)
        diMat[edgeList[:,0], edgeList[:,1]] = np.sign(network.junctionVoltage[this_TimeStamp,:]\
                                                *network.junctionSwitch[this_TimeStamp,:])
        diMat = diMat-diMat.T
        diMat[diMat<0] = 0
        onGraph = nx.from_numpy_array(diMat, create_using=nx.DiGraph())
        # onGraph = nx.DiGraph()
        # onGraph.add_nodes_from(range(network.numOfWires))
        # junctionCurrent = network.junctionVoltage[this_TimeStamp,:]*network.junctionConductance[this_TimeStamp,:]
        # this_direction = np.sign(junctionCurrent*network.junctionSwitch[this_TimeStamp,:])
        # for i in range(network.numOfJunctions):
        #     if this_direction[i] == 1:
        #         onGraph.add_edge(edgeList[i,0], edgeList[i,1])
        #     elif this_direction[i] == -1:
        #         onGraph.add_edge(edgeList[i,1], edgeList[i,0])
    else:
        edgeList = network.connectivity.edge_list
        adjMat = np.zeros(network.connectivity.adj_matrix.shape)
        adjMat[edgeList[:,0], edgeList[:,1]] = network.junctionSwitch[this_TimeStamp,:]
        adjMat = adjMat + adjMat.T
        onGraph = nx.from_numpy_array(adjMat)
    return onGraph

def getDiGraph(network, this_TimeStamp = 0):
    edgeList = network.connectivity.edge_list
    diMat = np.zeros(network.connectivity.adj_matrix.shape)
    diMat[edgeList[:,0], edgeList[:,1]] = np.sign(network.junctionVoltage[this_TimeStamp,:])
    diMat = diMat-diMat.T
    diMat[diMat<0] = 0
    return nx.from_numpy_array(diMat, create_using=nx.DiGraph())

def getInDegree(network, this_TimeStamp = 0):
    diGraph = getDiGraph(network, this_TimeStamp)
    return np.array(list(dict(diGraph.in_degree(range(len(diGraph)))).values()))

def getOutDegree(network, this_TimeStamp = 0):
    diGraph = getDiGraph(network, this_TimeStamp)
    return np.array(list(dict(diGraph.out_degree(range(len(diGraph)))).values()))

def findCurrent(network, numToFind = 1):
    source = network.sources[0]
    drain = network.drains[0]
    numFound = 0
    PathList = []
    foundTime = []
    onGraph = getOnGraph(network, 0)
    last_direction = np.zeros(network.numOfJunctions)
    for this_time in range(1,network.TimeVector.size):
        this_direction = np.sign(network.junctionVoltage[this_time,:]*network.junctionConductance[this_time,:]*network.junctionSwitch[this_time,:])
        flag = this_direction == last_direction
        changed_pos = np.where(flag == False)[0]
        if changed_pos.size == 0:
            continue
        else:
            onGraph = getOnGraph(network, this_time)
        
        pathFormed = nx.has_path(onGraph, source, drain)
        if pathFormed:
            tempPaths = [i for i in nx.all_simple_paths(onGraph, source, drain)]
            if len(tempPaths) > numFound:
                for i in tempPaths:
                    if i not in PathList:
                        PathList.append(i)
                        foundTime.append(np.round(network.TimeVector[this_time], 3))
                        numFound+=1
        
        if numFound >= numToFind:
            break
    
    if numFound < numToFind:
        print(f'Unfortunately, only {numFound} current paths found in simulation time.')

    # if numFound == 0:
    #     return None
    return PathList, foundTime

def getSubGraphComm(network, this_TimeStamp = 0):
    onGraph = getOnGraph(network, this_TimeStamp, False)
    components = [i for i in nx.connected_components(onGraph)]
    giant_component = components[np.argmax([len(i) for i in nx.connected_components(onGraph)])]
    nodes = list(giant_component)
    commMat = np.zeros((network.numOfWires, network.numOfWires))
    subComm = nx.communicability(onGraph.subgraph(giant_component))
    for i in nodes:
        for j in nodes:
            commMat[i,j] = subComm[i][j]
    return commMat

def wireDistanceToSource(network):
    V = network.numOfWires
    G = network.graph
    distance = np.zeros(V)
    source = network.sources[0]
    for i in range(V):
        distance[i] = nx.shortest_path_length(G, source, i)
    return distance

def wireDistanceToPath(network, path):
    V = network.numOfWires
    G = network.graph
    distance = np.zeros(V)
    
    for i in range(V):
        if i in path:
            distance[i] = 0
        else:
            min_dist = 1e5
            for this_node in path:
                temp_dist = nx.shortest_path_length(G, source=i, target=this_node)
                min_dist = min(min_dist, temp_dist)
            distance[i] = min_dist
    return distance

def get_junction_centrality(network, this_TimeStamp=0):
    edgeList = network.connectivity.edge_list
    conMat = np.zeros((network.numOfWires, network.numOfWires))
    conMat[edgeList[:,0], edgeList[:,1]] = network.junctionConductance[this_TimeStamp,:]
    conG = nx.from_numpy_array(conMat)
    
    return np.array(list(nx.edge_current_flow_betweenness_centrality_subset(conG, network.sources, network.drains, weight = 'weight').values()))

def get_wire_centrality(network, this_TimeStamp=0, mode = 'betweenness'):
    edgeList = network.connectivity.edge_list
    conMat = np.zeros((network.numOfWires, network.numOfWires))
    conMat[edgeList[:,0], edgeList[:,1]] = network.junctionConductance[this_TimeStamp,:]
    conG = nx.from_numpy_array(conMat)
    if mode == 'betweenness':
        return np.array(list(nx.current_flow_betweenness_centrality_subset(conG, network.sources, network.drains, weight = 'weight').values()))
    elif mode == 'closeness':
        return np.array(list(nx.current_flow_closeness_centrality(conG, weight = 'weight').values()))

def getCommMat(network):
    adjMat = network.connectivity.adj_matrix
    G = nx.from_numpy_array(adjMat)
    comm = nx.communicability(G)
    commMat = np.array([comm[i][j] for i in range(len(G)) for j in range(len(G))]).reshape(len(G),len(G))
    return commMat

def extendLaplacian(network, this_TimeStamp=0, extend_pos = []):
    N = network.numOfWires
    edgeList = network.connectivity.edge_list
    pos = np.append(network.electrodes, extend_pos).astype(int)
    Gmat = np.zeros((N,N))
    Gmat[edgeList[:,0], edgeList[:,1]] = network.junctionConductance[this_TimeStamp,:]
    Gmat[edgeList[:,1], edgeList[:,0]] = network.junctionConductance[this_TimeStamp,:]
    Gmat = np.diag(np.sum(Gmat,axis=0)) - Gmat
    
    L = np.zeros((N+len(pos),N+len(pos)))
    L[:N, :N] = Gmat
    for i, this_elec in enumerate(pos):
        L[N+i, this_elec] = 1
        L[this_elec, N+i] = 1
    return L
def junctionDistanceToSource(network):
    edgeList = network.connectivity.edge_list
    E = network.numOfJunctions
    G = network.graph
    distance = np.zeros(E)
    source = network.sources[0]

    for i in range(E):
        node1, node2 = edgeList[i,:]
        dist1 = nx.shortest_path_length(G, source, node1)
        dist2 = nx.shortest_path_length(G, source, node2)
        distance[i] = min(dist1,dist2)+1
    return distance

def junctionDistanceToPath(network, path):
    edgeList = network.connectivity.edge_list
    E = network.numOfJunctions
    G = network.graph
    distance = np.zeros(E)  
    
    for i in range(E):
        touching = edgeList[i,:]
        flag = np.intersect1d(path, touching)
        if len(flag) == 2:
            distance[i] = 0
        elif len(flag) == 1:
            distance[i] = 1
        else: 
            node1, node2 = touching
            dist = 1000
            for this_node in path:
                dist1 = nx.shortest_path_length(G, node1, this_node)
                dist2 = nx.shortest_path_length(G, node2, this_node)
                dist = min(dist, dist1, dist2)
            distance[i] = dist+1
    return distance

def graphVoltageDistribution(network, plot_type='nanowire',**kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.style.use('classic')
    if 'TimeStamp' in kwargs:
            this_TimeStamp = kwargs['TimeStamp']
    elif 'time' in kwargs:
        if kwargs['time'] in network.TimeVector:
            this_TimeStamp = np.where(network.TimeVector == kwargs['time'])[0][0]
        elif (kwargs['time'] < min(network.TimeVector)) or (kwargs['time'] > max(network.TimeVector)):
            print('Input time exceeds simulation period.')
            this_TimeStamp = np.argmin(abs(network.TimeVector - kwargs['time']))
        else:
            this_TimeStamp = np.argmin(abs(network.TimeVector - kwargs['time']))
    else:
        this_TimeStamp = 0
    
    if 'figsize' in kwargs:
        figsize = kwargs['figsize']
    else:
        figsize = None
        
    finder = findCurrent(network,1)
    if len(finder[0]) != 0:
        mainPath = finder[0][0]
    else:
        mainPath = network.shortestPaths[0]
        print('Using graphcial shortest path as main path.')
    
    wireToSource = wireDistanceToSource(network)
    junctionToSource = junctionDistanceToSource(network)
    wireToPath = wireDistanceToPath(network, mainPath)
    junctionToPath = junctionDistanceToPath(network, mainPath)
    
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)
    if plot_type == 'nanowire':
        ax.plot_trisurf(wireToSource, wireToPath, network.wireVoltage[this_TimeStamp,:],
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('distance to source')
        ax.set_ylabel('distance to main path')
        ax.set_zlabel('Nanowire voltage')
    elif plot_type == 'junction':
        ax.plot_trisurf(junctionToSource, junctionToPath, abs(network.junctionVoltage[this_TimeStamp,:]),
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('distance to source')
        ax.set_ylabel('distance to main path')
        ax.set_zlabel('Junction voltage')
    else:
        print('Plot type error! Either nanowire or junction.')

def getCorrelation(network, this_TimeStamp = 0, perturbation_rate = 0.1):
    N = network.numOfWires
    corrMat = np.zeros(network.connectivity.adj_matrix.shape)
    for i in range(network.numOfWires):
        count = 0
        if i in network.electrodes:
            tempL = extendLaplacian(network, this_TimeStamp)
            rhs = np.zeros(tempL.shape[0])
            rhs[N:] = np.array([i.signal[this_TimeStamp] for i in network.stimulus])
            if network.stimulus[count].signal[this_TimeStamp] == 0:
                rhs[N+count] = perturbation_rate
            else:
                rhs[N+count] = network.stimulus[count].signal[this_TimeStamp]*(1+perturbation_rate)
            count += 1
        else:
            tempL = extendLaplacian(network, this_TimeStamp, i)
            rhs = np.zeros(tempL.shape[0])
            rhs[N:-1] = np.array([i.signal[this_TimeStamp] for i in network.stimulus])
            rhs[-1] = network.wireVoltage[this_TimeStamp, i]*(1+perturbation_rate)
            
        newDistribution = np.linalg.solve(tempL, rhs)[:N]
        if network.wireVoltage[this_TimeStamp, i] != 0:
            corrMat[i,:] = abs((newDistribution - network.wireVoltage[this_TimeStamp, :])/network.wireVoltage[this_TimeStamp, :])/ \
                            abs(perturbation_rate)
    return corrMat

def getNodeInfluence(connectivity, nodeIdx, onAmp = 2, perturbeRate = 0.05):
    N = connectivity.numOfWires
    others = np.setdiff1d(range(N), nodeIdx)
    custom = np.zeros(5000)
    perturbeTime = 4000
    custom[:perturbeTime] = 2
    custom[perturbeTime:] = 2.2
    calcList = [inputPacker(runSimulation, connectivity, T = 5, 
                        contactMode = 'preSet', electrodes = [nodeIdx, i], 
                        biasType = 'Custom', customSignal = custom,
                        findFirst = False, disable_tqdm = True, lite_mode = True) 
            for i in others]
    
    with Pool(4) as pool: 
        simList = list(tqdm(pool.istarmap(runSim, calcList), total = N-1, desc = '🚴.......🚓'))
        
    S1List = [sim.wireVoltage[perturbeTime,others] for sim in simList]
    S2List = [sim.wireVoltage[-1,others] for sim in simList]
    influence = np.zeros((N-1,N))
    
    for i in range(N-1):
        influence[i,others] = np.nan_to_num(abs(((S2List[i] - S1List[i])/S1List[i])/(0.2/2)))
        
    return np.mean(influence, axis = 0)