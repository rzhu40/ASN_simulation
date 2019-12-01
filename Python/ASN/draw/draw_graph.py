import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

def draw_graph(network, ax = None, figsize=(10,10), edge_mode = 'current', colorbar=False, **kwargs):
    TimeVector = network.TimeVector
    if 'TimeStamp' in kwargs:
        this_TimeStamp = kwargs['TimeStamp']
    elif 'time' in kwargs:
        if kwargs['time'] in TimeVector:
            this_TimeStamp = np.where(TimeVector == kwargs['time'])[0][0]
        elif (kwargs['time'] < min(TimeVector)) or (kwargs['time'] > max(TimeVector)):
            print('Input time exceeds simulation period.')
            this_TimeStamp = np.argmin(abs(TimeVector - kwargs['time']))
        else:
            this_TimeStamp = np.argmin(abs(TimeVector - kwargs['time']))
    else:
        this_TimeStamp = 0

    G = nx.from_numpy_array(network.connectivity.adj_matrix)
    pos = nx.layout.kamada_kawai_layout(G)   
    edgeList = network.connectivity.edge_list
    sources = network.sources
    drains = network.drains
    this_switch = network.junctionSwitch[this_TimeStamp, :]
    
    if edge_mode == 'current':
        graphView = nx.DiGraph()
        graphView.add_nodes_from(range(network.numOfWires))
        edge_weight = network.junctionVoltage[this_TimeStamp,:]*network.junctionConductance[this_TimeStamp,:]
        onPos = np.sign(edge_weight * this_switch) > 0
        onNeg = np.sign(edge_weight * this_switch) < 0

        zipPos = [(edgeList[i,0], edgeList[i,1], abs(edge_weight[i])) for i in np.where(onPos)[0]]
        zipNeg = [(edgeList[i,1], edgeList[i,0], abs(edge_weight[i]) )for i in np.where(onNeg)[0]]
        graphView.add_weighted_edges_from(zipPos, width = 2, style = 'solid')
        graphView.add_weighted_edges_from(zipNeg, width = 2, style = 'solid')
    else:
        if edge_mode == 'filament': 
            edge_weight = network.filamentState[this_TimeStamp,:]
        elif edge_mode == 'voltage':
            edge_weight = network.junctionVoltage[this_TimeStamp,:]
        elif edge_mode == 'custom':
            edge_weight = kwargs['edge_weight']

        graphView = nx.empty_graph(network.numOfWires)
        zipOn = [(edgeList[i,0], edgeList[i,1], abs(edge_weight[i])) for i in np.where(this_switch)[0]]
        zipOff = [(edgeList[i,0], edgeList[i,1], abs(edge_weight[i])) for i in np.where(1-this_switch)[0]]
        graphView.add_weighted_edges_from(zipOn, width = 2, style='solid')
        graphView.add_weighted_edges_from(zipOff, width = 1, style='dashed')

    from analysis.GraphTheory import getOnGraph
    tempGraph = getOnGraph(network, this_TimeStamp=this_TimeStamp)
    pathFormed = nx.has_path(tempGraph, sources[0], drains[0])
    if pathFormed:
        cmap = plt.cm.Reds
    else:
        cmap = plt.cm.Blues

    edge_colors = [graphView[u][v]['weight'] for u,v in graphView.edges]
    widths = [graphView[u][v]['width'] for u,v in graphView.edges]
    styles = [graphView[u][v]['style'] for u,v in graphView.edges]

    node_colors = ['#1f78b4']*len(graphView)
    for i in sources:
        node_colors[i] = 'g'
    for i in drains:
        node_colors[i] = 'r'

    from utils import useMyRC
    useMyRC()

    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        if len(edge_colors)==0:
            cmap_max = 1
        else:
            cmap_max = np.max(edge_colors)
    else:
        if len(edge_colors)==0:
            cmap_max = 1
        elif edge_mode == 'current':
            cmap_max = np.max(abs(network.junctionVoltage * network.junctionConductance))
        elif edge_mode == 'voltage':
            cmap_max = np.max(abs(network.junctionVoltage))
        elif edge_mode == 'filament':
            cmap_max = np.max(network.filamentState)
        else:
            cmap_max = np.max(abs(edge_colors))
    cmap_min = 0

    nx.draw_networkx(graphView, pos,
                    node_size=350,
                    node_color=node_colors,
                    edge_color=edge_colors,
                    edge_cmap=cmap,
                    edge_vmin=cmap_min,
                    edge_vmax=cmap_max,
                    font_color='w',
                    width=widths,
                    style = styles,
                    ax=ax)

    if edge_mode == 'current':
        nx.draw_networkx_edges(G, pos, width = 1, alpha = 0.15, ax=ax)

    ax.set_facecolor((0.8,0.8,0.8))
    fig.set_facecolor((0.8,0.8,0.8))
    if edge_mode == 'custom':
        try:
            ax.set_title(f'Network at t = {np.round(TimeVector[this_TimeStamp],3)}')
        except:
            ax.set_title(f'Network at t = {np.round(TimeVector[this_TimeStamp],3)}')
    else:
        ax.set_title(f'Network {edge_mode} at t = {np.round(TimeVector[this_TimeStamp],3)}')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    if colorbar:        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=cmap_min, vmax=cmap_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax,
                            fraction = 0.05, label=edge_mode)

    return ax

if __name__ == '__main__':
    from utils import *
    # Connectivity = connectivity__('100nw_261junctions.mat')
    # sim1 = runSim(Connectivity, T = 5, contactMode='farthest', biasType = 'DC', findFirst=False)
    # from analysis.GraphTheory import get_junction_centrality
    # cent = get_junction_centrality(sim1, 4000)

    # draw_graph(sim1, time = 4, edge_mode ='custom', edge_weight = cent)
    # plt.show()
    wires = generateNetwork(20, 50, 100)
    Connectivity = connectivity__(wires_dict = wires)
    sim1 = runSim(Connectivity, T = 5, contactMode = 'farthest', biasType = 'DC', onAmp = 2)
    draw_graph(sim1, time = 0)