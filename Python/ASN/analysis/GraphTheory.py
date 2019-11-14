import numpy as np
import networkx as nx
from itertools import islice

# def generateGrid()
def getOnGraph(network, this_TimeStamp = 0, isDirected = True):
    if isDirected:
        edgeList = network.connectivity.edge_list
        source = network.sources[0]
        drain = network.drains[0]
        onGraph = nx.DiGraph()
        onGraph.add_nodes_from(range(network.numOfWires))
        junctionCurrent = network.junctionVoltage[this_TimeStamp,:]*network.junctionConductance[this_TimeStamp,:]
        this_direction = np.sign(junctionCurrent*network.junctionSwitch[this_TimeStamp,:])
        for i in range(network.numOfJunctions):
            if this_direction[i] == 1:
                onGraph.add_edge(edgeList[i,0], edgeList[i,1])
            elif this_direction[i] == -1:
                onGraph.add_edge(edgeList[i,1], edgeList[i,0])
    else:
        edgeList = network.connectivity.edge_list
        adjMat = np.zeros((network.numOfWires, network.numOfWires))
        adjMat[edgeList[:,0], edgeList[:,1]] = network.junctionSwitch[this_TimeStamp,:]
        adjMat[edgeList[:,1], edgeList[:,0]] = network.junctionSwitch[this_TimeStamp,:]
        onGraph = nx.from_numpy_array(adjMat)
    return onGraph

def findCurrent(network, numToFind = 1):
    edgeList = network.connectivity.edge_list
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

def wireDistanceToSource(network):
    V = network.numOfWires
    G = network.graph
    distance = np.zeros(V)
    source = network.sources[0]
    for i in range(V):
        distance[i] = nx.shortest_path_length(G, source, i)
    return distance

def wireDistanceToPath(network, path):
    edgeList = network.connectivity.edge_list
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