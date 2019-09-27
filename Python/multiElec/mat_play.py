import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

N = 10
E = 20
adjMat = np.zeros((N,N))

adjMat = np.zeros((N,N))
numOfNodes = 1
numOfEdges = 0
nodeOn = np.array([])
nodeRemaining = np.arange(0,N)
nodeOn = np.append(nodeOn, np.random.randint(0,N)).astype(int)
nodeRemaining = np.setdiff1d(nodeRemaining, nodeOn)

while numOfNodes < N:
    this_node = np.random.choice(nodeOn, 1)[0]
    new_node  = np.random.choice(nodeRemaining, 1)[0]
    adjMat[this_node, new_node] = 1
    nodeOn = np.append(nodeOn, new_node)
    nodeRemaining = np.setdiff1d(nodeRemaining, [new_node]) 
    numOfNodes += 1
    numOfEdges += 1
    
while numOfEdges < E:
    node1 = np.random.choice(nodeOn, 1)[0]
    node2 = np.random.choice(nodeOn, 1)[0]
    if (node1 != node2)&(adjMat[node1, node2] != 1):
        adjMat[node1, node2] = 1
        numOfEdges += 1
    else:
        continue

adjMat = adjMat + adjMat.T
