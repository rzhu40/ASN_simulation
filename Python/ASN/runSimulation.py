import numpy as np
import matplotlib.pyplot as plt
from utils import *


SimulationOptions = simulation_options__(dt = 1e-3, T = 100,
                                        contactMode = 'preSet',
                                        electrodes = [72,29])
                                        
Connectivity = connectivity__(
                        filename = '2016-09-08-155153_asn_nw_00100_nj_00261_seed_042_avl_100.00_disp_10.00.mat')

# Connectivity = connectivity__(
#     filename = '2016-09-08-155044_asn_nw_00700_nj_14533_seed_042_avl_100.00_disp_10.00.mat')


tempStimulus = stimulus__(biasType = 'DC', 
        TimeVector = SimulationOptions.TimeVector, 
        onTime = 0, offTime = 1,
        onAmp = 1.1, offAmp = 0.005)
SimulationOptions.stimulus.append(tempStimulus)

tempStimulus = stimulus__(biasType = 'Drain', 
        TimeVector = SimulationOptions.TimeVector)
SimulationOptions.stimulus.append(tempStimulus)

# tempStimulus = stimulus__(biasType = 'DC', 
#         TimeVector = SimulationOptions.TimeVector, 
#         onTime = 0, offTime = 1,
#         onAmp = 1.4, offAmp = 0.005)
# SimulationOptions.stimulus.append(tempStimulus)

# tempStimulus = stimulus__(biasType = 'Drain', 
#         TimeVector = SimulationOptions.TimeVector)
# SimulationOptions.stimulus.append(tempStimulus)

JunctionState = junctionState__(Connectivity.numOfJunctions)

this_realization = simulateNetwork(SimulationOptions, Connectivity, JunctionState)

# this_realization.draw(time=0.9)