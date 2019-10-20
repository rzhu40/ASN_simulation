import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

def draw_graph(network, ax = None, figsize=(10,10), edge_mode = 'current', colorbar=False,**kwargs):
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
    edgeList = network.connectivity.edge_list
    sources = network.sources
    drains = network.drains

    this_switch = network.junctionSwitch[this_TimeStamp, :]

    if edge_mode == 'current':
        graphView = nx.DiGraph()
        graphView.add_nodes_from(range(network.numOfWires))
        this_current = network.junctionVoltage[this_TimeStamp,:]/network.junctionResistance[this_TimeStamp,:]
        for i in range(network.numOfJunctions):
            if this_switch[i]:
                if this_current[i]>0:
                    graphView.add_edge(edgeList[i,0], edgeList[i,1], weight = abs(this_current[i]), width = 2, style = 'solid')
                else:
                    graphView.add_edge(edgeList[i,1], edgeList[i,0], weight = abs(this_current[i]), width = 2, style = 'solid')

    elif edge_mode == 'voltage':
        graphView = nx.empty_graph(network.numOfWires)
        this_voltage = network.junctionVoltage[this_TimeStamp,:]
        for i in range(network.numOfJunctions):
            if this_switch[i]:
                graphView.add_edge(edgeList[i,0], edgeList[i,1], weight = abs(this_voltage[i]), width = 2, style='solid')
            else:
                graphView.add_edge(edgeList[i,0], edgeList[i,1], weight = abs(this_voltage[i]), width = 1, style='dashed')
    
    elif edge_mode == 'filament':
        graphView = nx.empty_graph(network.numOfWires)
        this_filament = network.filamentState[this_TimeStamp,:]
        for i in range(network.numOfJunctions):
            if this_switch[i]:
                graphView.add_edge(edgeList[i,0], edgeList[i,1], weight = abs(this_filament[i]), width = 2, style='solid')
            else:
                graphView.add_edge(edgeList[i,0], edgeList[i,1], weight = abs(this_filament[i]), width = 1, style='dashed')

    elif edge_mode == 'Lyapunov':
        graphView = nx.empty_graph(network.numOfWires)
        if not hasattr(network, 'Lyapunov'):
            print('Lyapunov not calculated')
            network.Lyapunov = np.zeros(network.numOfJunctions)
        for i in range(network.numOfJunctions):
            if this_switch[i]:
                graphView.add_edge(edgeList[i,0], edgeList[i,1], weight = network.Lyapunov[i], width = 2, style='solid')
            else:
                graphView.add_edge(edgeList[i,0], edgeList[i,1], weight = network.Lyapunov[i], width = 1, style='dashed')

    
    from analysis.GraphTheory import onGraph
    tempGraph = onGraph(network, this_TimeStamp=this_TimeStamp)
    tempPaths = [i for i in nx.all_simple_paths(tempGraph, sources[0], drains[0])]
    pathFormed = len(tempPaths) > 0

    # diEdgeList = np.array(list(graphView.edges))
    # edge_colors = [graphView[diEdgeList[i,0]][diEdgeList[i,1]]['weight'] for i in range(len(graphView.edges))]
    # widths = this_switch+1

    edge_colors = [graphView[u][v]['weight'] for u,v in graphView.edges]
    widths = [graphView[u][v]['width'] for u,v in graphView.edges]
    styles = [graphView[u][v]['style'] for u,v in graphView.edges]

    node_colors = ['#1f78b4']*len(graphView)
    for i in sources:
        node_colors[i] = 'g'
    for i in drains:
        node_colors[i] = 'r'
    plt.style.use('classic')

    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
    

    if pathFormed:
        cmap = plt.cm.Reds
    else:
        cmap = plt.cm.Blues
    
    cmap_min = 0
    if len(edge_colors)==0:
        cmap_max = 1
    elif edge_mode == 'current':
        cmap_max = max(edge_colors)
    elif edge_mode == 'voltage':
        cmap_max = np.max(abs(network.junctionVoltage))
    elif edge_mode == 'filament':
        cmap_max = np.max(network.filamentState)
    elif edge_mode == 'Lyapunov':
        cmap_min = np.min(network.Lyapunov)
        cmap_max = np.max(network.Lyapunov)

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
    ax.set_title('Network '+edge_mode+f' at t = {np.round(network.TimeVector[this_TimeStamp],3)}')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    if colorbar:
        fig.set_size_inches((figsize[0]*1.2, figsize[1]))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=cmap_min, vmax=cmap_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax,
                            fraction = 0.05, label=edge_mode)

    return ax

if __name__ == '__main__':
    import os 
    os.chdir(r'C:\Users\rzhu\Documents\PhD\ASN_simulation\Python\ASN')
    from utils import *
    sim1 = defaultSimulation(connectivity__(filename = '2016-09-08-155153_asn_nw_00100_nj_00261_seed_042_avl_100.00_disp_10.00.mat'),
                        contactMode='preSet', electrodes=[72,29],
                        biasType='DC', offTime = 5, onAmp = 1.1, T = 1)
    draw_graph(sim1, time = 0.9, edge_mode ='voltage')
    plt.show()