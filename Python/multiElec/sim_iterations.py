import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
from utils import *
from dataStruct import *
from draw_mpl import *
from connectivity import wires
import logging
import pickle
"""
All interface indexing, like interfaceContactWires, use the index convention of MatLab,
which starts from 1.
"""

sparse_ranger = range(20, 100, 10)
realization_list = []


for this_sparse in sparse_ranger:
    logging.info(f"Simulation #{sparse_ranger.index(this_sparse)+1}/{len(sparse_ranger)}")
    wires_dict = wires.generate_wires_distribution(number_of_wires = 100,
                                             wire_av_length = 100,
                                             wire_dispersion = 10,
                                             gennorm_shape = 3,
                                             centroid_dispersion = this_sparse,
                                             Lx = 3e2,
                                             Ly = 3e2);

    # Get junctions list and their positions
    
    wires_dict = wires.detect_junctions(wires_dict)
    wires_dict = wires.generate_graph(wires_dict)
    if not wires.check_connectedness(wires_dict):
        logging.warning(f"This network is not connected, will move on to next one! Current dispersion is {wires_dict['centroid_dispersion']}.")
        continue
    Connectivity = connectivity__(wires_dict = wires_dict)
    
    SimulationOptions = simulation_options__(dt = 1e-3, T = 10,
                                        contactMode = 'farthest')
    
    SimulationOptions.stimulus = []
    tempStimulus = stimulus__(biasType = 'DC', 
            TimeVector = SimulationOptions.TimeVector, 
            onTime = 0, offTime = 10,
            onAmp = 0.5, offAmp = 0.005)
    SimulationOptions.stimulus.append(tempStimulus)

    tempStimulus = stimulus__(biasType = 'Drain', 
            TimeVector = SimulationOptions.TimeVector)
    SimulationOptions.stimulus.append(tempStimulus)


    JunctionState = junctionState__(Connectivity.numOfJunctions)

    this_realization = simulateNetworkPlus(SimulationOptions, Connectivity, JunctionState)
    realization_list.append(this_realization)

filename = 'data/sparsity0923.pkl'
logging.info(f"Saving data to {filename}.")

with open(filename, 'wb') as output:
    pickle.dump(realization_list, output, pickle.HIGHEST_PROTOCOL)

logging.info("Simulation finished.")