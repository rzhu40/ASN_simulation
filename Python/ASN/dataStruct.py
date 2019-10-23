import numpy as np
import networkx as nx

from tqdm import tqdm
from copy import deepcopy
from draw.draw_plotly import draw_plotly

class network__:
    pass

    def findJunction(self, wire1, wire2):
        edgeList = self.connectivity.edge_list
        index = np.where((edgeList[:,0] == wire1) & (edgeList[:,1] == wire2))[0]
        if len(index) == 0:
            index = np.where((edgeList[:,0] == wire2) & (edgeList[:,1] == wire1))[0]
        if len(index) == 0:
            print(f'Unfortunately, no junction between nanowire {wire1} and {wire2}')
            return None
        else:
            return index[0]

    def allocateData(self):
        print('Allocating data to obejcts....')
        self.gridSize = [self.connectivity.length_x, self.connectivity.length_y]
        self.junctionCurrent = self.junctionVoltage/self.junctionResistance
        # self.networkConductance = 1/self.networkResistance        
        self.isOnCurrentPath()
        # self.getWireVoltage()

        print('.')
        self.Junctions = [junction__(self, i) for i in range(self.numOfJunctions)]
        print('.')
        self.Nanowires = [nanowire__(self, i) for i in range(self.numOfWires)]
        print('.')
        self.JunctionsInd = np.sort([self.Junctions[i].index for i in range(len(self.Junctions))]).astype(int)
        self.WiresInd = np.sort([self.Nanowires[i].index for i in range(len(self.Nanowires))]).astype(int)
        
        print('Done!')

    def isOnCurrentPath(self):
        self.onCurrentPath = np.zeros((self.TimeVector.size, self.numOfJunctions), dtype = bool)
        edgeList = self.connectivity.edge_list
        onMat = np.zeros((self.numOfWires, self.numOfWires))
        onMat[edgeList[:,0], edgeList[:,1]] = self.junctionSwitch[0,:]
        onMat[edgeList[:,1], edgeList[:,0]] = self.junctionSwitch[0,:]
        onGraph = nx.from_numpy_array(onMat)
        # onGraph = nx.empty_graph(self.numOfWires)

        for this_time in range(1, self.TimeVector.size):
            flag = self.junctionSwitch[this_time,:] == self.junctionSwitch[this_time-1,:]
            changed_pos = np.where(flag == False)[0]
            if changed_pos.size == 0:
                self.onCurrentPath[this_time,:] = self.onCurrentPath[this_time-1,:]
                continue
            for i in changed_pos:
                if self.junctionSwitch[this_time,i] == 1:
                    onGraph.add_edge(edgeList[i,0], edgeList[i,1])
                else:
                    onGraph.remove_edge(edgeList[i,0], edgeList[i,1])

            component = nx.node_connected_component(onGraph, self.sources[0])
            if self.drains[0] in component:
                for i in range(self.numOfJunctions):
                    if (self.junctionSwitch[this_time,i])&(edgeList[i,0] in component)&(edgeList[i,1] in component):
                        self.onCurrentPath[this_time,i] = True

    def draw(self, **kwargs):
        if not hasattr(self, 'Junctions'):
            self.allocateData()
            
        Lx, Ly = self.gridSize

        if 'TimeStamp' in kwargs:
            this_TimeStamp = kwargs['TimeStamp']
        elif 'time' in kwargs:
            if kwargs['time'] in self.TimeVector:
                this_TimeStamp = np.where(self.TimeVector == kwargs['time'])[0][0]
            elif (kwargs['time'] < min(self.TimeVector)) or (kwargs['time'] > max(self.TimeVector)):
                print('Input time exceeds simulation period.')
                this_TimeStamp = np.argmin(abs(self.TimeVector - kwargs['time']))
            else:
                this_TimeStamp = np.argmin(abs(self.TimeVector - kwargs['time']))
        else:
            this_TimeStamp = 0

        if 'JunctionsToObserve' in kwargs:
            JunctionsToObserve = kwargs['JunctionsToObserve']
        else:
            JunctionsToObserve = []

        if 'PathHighlight' in kwargs:
            PathHighlight = kwargs['PathHighlight']
        else:
            PathHighlight = None
        
        fig = draw_plotly(self, PathHighlight = PathHighlight, 
                        JunctionsToObserve = JunctionsToObserve, 
                        TimeStamp = this_TimeStamp)
        return fig

class junction__:
    def __init__(self, network, index):
        self.index = index
        self.position = np.array([network.connectivity.xi[index], network.connectivity.yi[index]])
        self.filamentState = network.filamentState[:,index]
        self.switch = network.junctionSwitch[:,index]
        self.resistance = network.junctionResistance[:,index]
        self.voltage = network.junctionVoltage[:,index]
        self.current = network.junctionCurrent[:,index]
        self.contactWires = network.connectivity.edge_list[index,:]
        self.onCurrentPath = network.onCurrentPath[:,index]
        self.conductance = 1/self.resistance
        
        # self.Node1Voltage = network.nodeVoltage[:,self.ContactNodes[0]-1]
        # self.Node2Voltage = network.nodeVoltage[:,self.ContactNodes[1]-1]

class nanowire__:
    def __init__(self, network, index):
        self.index = index
        self.centerPosition = np.array([network.connectivity.xc[index], network.connectivity.yc[index]])
        self.wireEnds =  np.array([network.connectivity.xa[index], network.connectivity.ya[index],
                                    network.connectivity.xb[index], network.connectivity.yb[index]])
        self.voltage = network.wireVoltage[:,index]
        self.adj = network.adjMat[:,index]
        self.contactWires = np.where(self.adj == 1)[0]
        self.contactJunctions = np.where(network.connectivity.edge_list[:,1] == index)
        self.contactJunctions = np.append(self.contactJunctions, np.where(network.connectivity.edge_list[:,0] == index))
        self.onPathChecker = np.sum([network.Junctions[i].onCurrentPath for i in self.contactJunctions], axis = 0)

        """
        This part determines the current on this nanowire
        """

        xa, ya, xb, yb = self.wireEnds
        xc, yc = self.centerPosition
        
        xi, yi = network.connectivity.xi, network.connectivity.yi

        self.sectionCenterX = np.array([])
        self.sectionCenterY = np.array([])
        self.sectionCurrentX = np.array([])
        self.sectionCurrentY = np.array([])

        self.wireAngle = np.arctan2(yb-ya, xb-xa) % np.pi
        self.wireAngle = np.where(self.wireAngle <= np.pi/2, self.wireAngle, self.wireAngle - np.pi)

        if xa != xb:
            sortedInd = np.argsort([network.Junctions[i].position[0] for i in self.contactJunctions])
        else:
            sortedInd = np.argsort([network.Junctions[i].position[1] for i in self.contactJunctions])

        sortedContactJunctions = self.contactJunctions[sortedInd]
        sortedContactWires = self.contactWires[sortedInd]

        direction = ((sortedContactWires < index) - 0.5) * 2

        self.wireCurrents = np.zeros((network.TimeVector.size, self.contactJunctions.size-1))
        for i in range(network.TimeVector.size):
            self.wireCurrents[i,:] = np.cumsum(network.junctionCurrent[i,sortedContactJunctions[0:-1]]*direction[0:-1])

        # self.sectionCenterX = np.append(self.sectionCenterX, np.mean([xi[np.add(sortedContactJunctions[0:-1], -1)], xi[np.add(sortedContactJunctions[1:], -1)]], axis = 0))
        # self.sectionCenterY = np.append(self.sectionCenterY, np.mean([yi[np.add(sortedContactJunctions[0:-1], -1)], yi[np.add(sortedContactJunctions[1:], -1)]], axis = 0))
        self.sectionCenterX = np.mean([xi[sortedContactJunctions[0:-1]], xi[sortedContactJunctions[1:]]], axis = 0)
        self.sectionCenterY = np.mean([yi[sortedContactJunctions[0:-1]], yi[sortedContactJunctions[1:]]], axis = 0)
        self.sectionCurrentX = np.cos(self.wireAngle) * self.wireCurrents
        self.sectionCurrentY = np.sin(self.wireAngle) * self.wireCurrents

        self.isElectrode = 0
        if (index in network.sources) or (index in network.drains):
            if index in network.sources:
                self.isElectrode = 1
            else:
                self.isElectrode = -1

            if xa != xb:
                if xa < xb:
                    self.contactEnd = [xb, yb]
                else:
                    self.contactEnd = [xa, ya]
            else:
                if ya < yb:
                    self.contactEnd = [xb, yb]
                else:
                    self.contactEnd = [xa, ya]

            self.totalCurrent = np.sum(network.junctionCurrent[:,sortedContactJunctions]*direction, axis = 1).reshape(network.TimeVector.size, 1)
            self.sectionCenterX = np.append(self.sectionCenterX, np.mean([xi[sortedContactJunctions[-1]], self.contactEnd[0]]))
            self.sectionCenterY = np.append(self.sectionCenterY, np.mean([yi[sortedContactJunctions[-1]], self.contactEnd[1]]))
            self.sectionCurrentX = np.append(self.sectionCurrentX, np.cos(self.wireAngle) * self.totalCurrent, axis = 1)
            self.sectionCurrentY = np.append(self.sectionCurrentY, np.sin(self.wireAngle) * self.totalCurrent, axis = 1)


class loop__:
    def __init__(self, network, mat1d, **kwargs):
        self.gridSize = network.gridSize
        self.TimeVector = network.TimeVector
        self.mat1d = mat1d

        if len(mat1d) == network.numOfJunctions:
            self.onLoopJunctionsInd = np.where(mat1d != 0)[0].astype(int)
            self.Junctions = [network.Junctions[i] for i in self.onLoopJunctionsInd]
            self.onLoopJunctionsNum = len(self.Junctions)

            self.onLoopWiresInd = []
            for this_junction in self.Junctions:
                for i in this_junction.contactWires:
                    if not i in self.onLoopWiresInd:
                        self.onLoopWiresInd = np.append(self.onLoopWiresInd, i)
            
            self.onLoopWiresInd = self.onLoopWiresInd.astype(int)
            self.Nanowires = [network.Nanowires[i] for i in self.onLoopWiresInd]
            self.onLoopWiresNum = len(self.Nanowires)


        # when the input is a row of adjacency matrix.
        # make diagonal of adjacency matrix 1.
        elif len(mat1d) == network.numOfWires:
            self.onLoopWiresInd = np.where(mat1d != 0)[0].astype(int)
            self.Nanowires = [network.Nanowires[i] for i in self.onLoopWiresInd]
            self.onLoopWiresNum = len(self.Nanowires)

            self.onLoopJunctionsInd = []
            for this_wire in self.Nanowires:
                for i in this_wire.contactJunctions:
                    if not i in self.onLoopJunctionsInd:
                        self.onLoopJunctionsInd = np.append(self.onLoopJunctionsInd, i)

            self.onLoopJunctionsInd = self.onLoopJunctionsInd.astype(int)
            self.Junctions = [network.Junctions[i] for i in self.onLoopJunctionsInd]
            self.onLoopJunctionsNum = len(self.Junctions)

        self.JunctionsInd = np.sort([self.Junctions[i].index for i in range(len(self.Junctions))]).astype(int)
        self.WiresInd = np.sort(self.onLoopWiresInd).astype(int)
    
    def draw(self, **kwargs):
        Lx, Ly = self.gridSize

        if 'TimeStamp' in kwargs:
            this_TimeStamp = kwargs['TimeStamp']
        elif 'time' in kwargs:
            if kwargs['time'] in self.TimeVector:
                this_TimeStamp = np.where(self.TimeVector == kwargs['time'])[0][0]
            elif (kwargs['time'] < min(self.TimeVector)) or (kwargs['time'] > max(self.TimeVector)):
                print('Input time exceeds simulation period.')
                this_TimeStamp = np.argmin(abs(self.TimeVector - kwargs['time']))
            else:
                this_TimeStamp = np.argmin(abs(self.TimeVector - kwargs['time']))
        else:
            this_TimeStamp = 0

        if 'JunctionsToObserve' in kwargs:
            JunctionsToObserve = kwargs['JunctionsToObserve']
        else:
            JunctionsToObserve = []

        if 'PathHighlight' in kwargs:
            PathHighlight = kwargs['PathHighlight']
        else:
            PathHighlight = []
        
        fig = draw.draw(self, PathHighlight = PathHighlight, 
                        JunctionsToObserve = JunctionsToObserve, 
                        TimeStamp = this_TimeStamp)
        return fig

class subnetwork_KVL__:

    def __init__(self, network, CenterJunctions = [], **kwargs):
        self.gridSize = network.gridSize
        self.TimeVector = network.TimeVector
        self.CenterJunctionsInd = CenterJunctions

        self.Loops = []
        self.Junctions = []
        self.Nanowires = []
        self.KVL_LoopsInd = np.array([])
        self.JunctionsInd = np.array([])
        self.WiresInd = np.array([])

        for this_junctionInd in self.CenterJunctionsInd:
            tempLoopsInd = np.where(network.KVL[:,this_junctionInd-1] != 0)[0]
            for i in tempLoopsInd:
                if i not in self.KVL_LoopsInd:
                    self.KVL_LoopsInd = np.append(self.KVL_LoopsInd, i)

        self.KVL_LoopsInd = np.sort(self.KVL_LoopsInd.astype(int))
        self.Loops.extend([loop__(network, network.KVL[i,:]) for i in self.KVL_LoopsInd])

        for this_loop in self.Loops:
            for this_wire in this_loop.Nanowires:
                if not this_wire in self.Nanowires:
                    self.Nanowires.append(this_wire)
                    self.WiresInd = np.append(self.WiresInd, this_wire.index)

            for this_junction in this_loop.Junctions:
                if not this_junction in self.Junctions:
                    self.Junctions.append(this_junction)
                    self.JunctionsInd = np.append(self.JunctionsInd, this_junction.index)
        
        self.WiresInd = np.sort(self.WiresInd).astype(int)
        self.JunctionsInd = np.sort(self.JunctionsInd).astype(int)

    def draw(self, **kwargs):
        Lx, Ly = self.gridSize

        if 'TimeStamp' in kwargs:
            this_TimeStamp = kwargs['TimeStamp']
        elif 'time' in kwargs:
            if kwargs['time'] in self.TimeVector:
                this_TimeStamp = np.where(self.TimeVector == kwargs['time'])[0][0]
            elif (kwargs['time'] < min(self.TimeVector)) or (kwargs['time'] > max(self.TimeVector)):
                print('Input time exceeds simulation period.')
                this_TimeStamp = np.argmin(abs(self.TimeVector - kwargs['time']))
            else:
                this_TimeStamp = np.argmin(abs(self.TimeVector - kwargs['time']))
        else:
            this_TimeStamp = 0

        if 'JunctionsToObserve' in kwargs:
            JunctionsToObserve = kwargs['JunctionsToObserve']
        else:
            JunctionsToObserve = []

        if 'PathHighlight' in kwargs:
            PathHighlight = kwargs['PathHighlight']
        else:
            PathHighlight = []
        
        fig = draw.draw(self, PathHighlight = PathHighlight, 
                        JunctionsToObserve = JunctionsToObserve, 
                        TimeStamp = this_TimeStamp)
        return fig