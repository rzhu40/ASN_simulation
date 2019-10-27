import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
os.chdir('..')
from utils import *
from dataStruct import *
import wires
import logging
import pickle

wires_dict = wires.generate_wires_distribution(number_of_wires = 100,
                                            wire_av_length = 100,
                                            wire_dispersion = 10,
                                            gennorm_shape = 3,
                                            centroid_dispersion = 150,
                                            Lx = 3e2,
                                            Ly = 3e2)

# Get junctions list and their positions

wires_dict = wires.detect_junctions(wires_dict)
wires_dict = wires.generate_graph(wires_dict)
if not wires.check_connectedness(wires_dict):
    logging.warning(f"This network is not connected. Simulation Stopped! Current dispersion is {wires_dict['centroid_dispersion']}.")
    sys.exit()
Connectivity = connectivity__(wires_dict = wires_dict)

SimulationOptions = simulation_options__(dt = 1e-3, T = 10,
                                        contactMode = 'farthest')

tempStimulus = stimulus__(biasType = 'DC', 
        TimeVector = SimulationOptions.TimeVector, 
        onTime = 0, offTime = 10,
        onAmp = 0.5, offAmp = 0.005)
SimulationOptions.stimulus.append(tempStimulus)

tempStimulus = stimulus__(biasType = 'Drain', 
        TimeVector = SimulationOptions.TimeVector)
SimulationOptions.stimulus.append(tempStimulus)


JunctionState = junctionState__(Connectivity.numOfJunctions)

this_realization = simulateNetwork(SimulationOptions, Connectivity, JunctionState)