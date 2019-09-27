import numpy as np
import networkx as nx
def random_generator(N,E):
    adjMat = np.zeros((N,N))
    offNode = np.arange(0,N)
    onNode =  np.random.choice(offNode, 1)
    offNode = np.setdiff1d(offNode, onNode)
    edgeCount = 0
    while onNode.size < N:
        this_node = int(np.random.choice(onNode, 1)[0])
        new_node = int(np.random.choice(offNode, 1)[0])
        adjMat[this_node,new_node] = 1
        adjMat[new_node,this_node] = 1
        onNode = np.append(onNode, new_node)
        offNode = np.setdiff1d(offNode, new_node)
        edgeCount += 1

    while edgeCount < E:
        node1, node2 = np.random.choice(onNode, 2).astype(int)
        if (node1 != node2)&(adjMat[node1,node2] == 0):
            adjMat[node1,node2] = 1
            adjMat[node2,node1] = 1
            edgeCount+=1
        else:
            continue
    return adjMat

def more_tri_generator(N,E):
    adjMat = np.zeros((N,N))
    offNode = np.arange(0,N)
    onNode =  np.random.choice(offNode, 1)
    offNode = np.setdiff1d(offNode, onNode)
    edgeCount = 0
    while onNode.size < N:
        this_node = int(np.random.choice(onNode, 1)[0])
        new_node = int(np.random.choice(offNode, 1)[0])
        adjMat[this_node,new_node] = 1
        adjMat[new_node,this_node] = 1
        onNode = np.append(onNode, new_node)
        offNode = np.setdiff1d(offNode, new_node)
        edgeCount += 1

    while edgeCount < E:
        this_node = np.random.randint(N)
        connected_nodes = np.where(adjMat[this_node,:]!=0)[0]
        if connected_nodes.size == 1:
            node1 = this_node
            node2 = np.random.randint(N)
            if (node1 != node2)&(adjMat[node1,node2]!=1):
                adjMat[node1, node2] = 1
                adjMat[node2, node1] = 1
                edgeCount += 1
        else:
            node1, node2 = np.random.choice(connected_nodes,2).astype(int)
            if (node1 != node2)&(adjMat[node1, node2]!=1):
                adjMat[node1, node2] = 1
                adjMat[node2, node1] = 1
                edgeCount += 1
    return adjMat

def bin_generator(N,E):
    nBins = 10
    bins = np.random.multinomial(N, [1/nBins]*nBins, size=1)
    endIndex = np.cumsum(bins)
    binList = np.split(np.arange(N), endIndex)[0:-1]
    graph = nx.empty_graph(N)
    edgeCount = 0
    for this_bin in binList:
        while not nx.is_connected(nx.subgraph(graph, this_bin)):
            node1, node2 = np.random.choice(this_bin, 2, replace = False)
            graph.add_edge(node1, node2)
    #         graph.add_edge(node2, node3)
    #         graph.add_edge(node1, node3)
            edgeCount += 1

    while edgeCount < E:
        binId1, binId2 = np.random.choice(np.arange(0,nBins), 2, replace = False)
        bin1 = binList[binId1]
        bin2 = binList[binId2]
        node1 = np.random.choice(bin1, 1)[0]
        node2 = np.random.choice(bin2, 1)[0]
        graph.add_edge(node1, node2)
        edgeCount += 1
    return graph