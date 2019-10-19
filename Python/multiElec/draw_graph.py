import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(network, **kwargs):
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

    G = network.graph
    pos = nx.layout.kamada_kawai_layout(G)   
    this_current = network.junctionVoltage[this_TimeStamp,:]/network.junctionResistance[this_TimeStamp,:]
    edgeList = network.connectivity.edge_list
    sources = [i-1 for i in network.sources]
    drains = [i-1 for i in network.drains]
    graphView = nx.DiGraph()
    graphView.add_nodes_from(range(network.numOfWires))

    for i in range(network.numOfJunctions):
        if network.junctionSwitch[this_TimeStamp, i]:
            if this_current[i]>0:
                graphView.add_edge(edgeList[i,0], edgeList[i,1], weight = abs(this_current[i]))
            else:
                graphView.add_edge(edgeList[i,1], edgeList[i,0], weight = abs(this_current[i]))
    diEdgeList = np.array(graphView.edges)
    edge_colors = [graphView[diEdgeList[i,0]][diEdgeList[i,1]]['weight'] for i in range(len(graphView.edges))]
    node_colors = ['#1f78b4']*len(graphView)
    for i in sources:
        node_colors[i] = 'g'
    for i in drains:
        node_colors[i] = 'r'
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(10,10))
    
    tempPaths = [i for i in nx.all_simple_paths(graphView, network.sources[0]-1, network.drains[0]-1)]
    pathFormed = len(tempPaths) > 0
    if pathFormed:
        nx.draw_networkx(graphView, pos,
                         node_size=350,
                         node_color=node_colors,
                         edge_color=edge_colors,
                         edge_cmap=plt.cm.Reds, 
                         font_color='w',
                         width=2)
    else:
        nx.draw_networkx(graphView, pos,
                         node_size=350,
                         node_color=node_colors,
                         edge_color=edge_colors,
                         edge_cmap=plt.cm.Blues,
                         font_color='w',
                         width=2)
    nx.draw_networkx_edges(G, pos, width = 1, alpha = 0.15)
    # ax = plt.gca()
    ax.set_axis_off()
    ax.set_title(f'Network Current Flow at t = {network.TimeVector[this_TimeStamp]}')
    # import matplotlib as mpl
    # pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    # pc.set_array(edge_colors)
    # plt.colorbar(pc)
    return ax