import numpy as np
import matplotlib.pyplot as plt
from jpype import *
from tqdm import tqdm
from utils import istarmap, inputPacker
import warnings
import os
from pathlib import Path
from multiprocessing import Pool
warnings.filterwarnings("ignore",category=UserWarning)

def readFloatsFile(filename):
	"Read a 2D array of floats from a given file"
	with open(filename) as f:
		# Space separate numbers, one time step per line, each column is a variable
		array = []
		for line in f:
			# read all lines
			if (line.startswith("%") or line.startswith("#")):
				# Assume this is a comment line
				continue
			if (len(line.split()) == 0):
				# Line is empty
				continue
			array.append([float(x) for x in line.split()])
		return array

def calc_AIS(data, k = 1, tau = 1, 
            calculator='kraskov', calc_type='average'):

    file_path = Path(os.path.dirname(os.path.realpath(__file__)))
    root = file_path.parent
    jarLocation = os.path.join(root, 'JIDT', 'infodynamics.jar')

    # jarLocation = r"C:\Users\rzhu\Documents\PhD\ASN_simulation\Python\ASN\JIDT\infodynamics.jar"
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
    
    if calculator == 'kraskov':
        calcAISClass = JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
        calcAIS = calcAISClass()
        calcAIS.setProperty("NOISE_LEVEL_TO_ADD", "0")
        # calcAIS.setProperty("NORM_TYPE", "EUCLIDEAN")
        calcAIS.setProperty("ALG_NUM", "1")
    elif calculator == 'gaussian':
        calcAISClass = JPackage("infodynamics.measures.continuous.gaussian").ActiveInfoStorageCalculatorGaussian
        calcAIS = calcAISClass()
    
    calcAIS.setProperty("k_HISTORY", str(k))
    calcAIS.setProperty("TAU", str(tau))
    calcAIS.setProperty("AUTO_EMBED_K_SEARCH_MAX", "2")
    calcAIS.setProperty("AUTO_EMBED_TAU_SEARCH_MAX", "2")
    calcAIS.setProperty("BIAS_CORRECTION", "true")


    calcAIS.initialise()
    calcAIS.setObservations(data)

    if calc_type == 'local':
        AISvalue = calcAIS.computeLocalOfPreviousObservations()[:]
    elif calc_type == 'average':
        AISvalue = calcAIS.computeAverageLocalOfObservations()
    else:
        print('Calculation Type not right!')
        return None
    return AISvalue

def calc_TE(source, destination, auto_embed = True,
            k_hist = 1, k_tau = 1, l_hist = 1, l_tau = 1,
            calculator='kraskov', calc_type='average'):

    file_path = Path(os.path.dirname(os.path.realpath(__file__)))
    root = file_path.parent
    jarLocation = os.path.join(root, 'JIDT', 'infodynamics.jar')

    # jarLocation = r"C:\Users\rzhu\Documents\PhD\ASN_simulation\Python\ASN\JIDT\infodynamics.jar"
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

    if calculator == 'kraskov':
        calcTEClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
        calcTE = calcTEClass()
        # calcTE.setProperty("NOISE_LEVEL_TO_ADD", "0")
        calcTE.setProperty("ALG_NUM", "1")

    elif calculator == 'gaussian':
        calcTEClass = JPackage("infodynamics.measures.continuous.gaussian").TransferEntropyCalculatorGaussian
        calcTE = calcTEClass()

    if auto_embed:
        calcTE.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS_DEST_ONLY")
        calcTE.setProperty("AUTO_EMBED_K_SEARCH_MAX", "10")
        calcTE.setProperty("AUTO_EMBED_TAU_SEARCH_MAX", "3")
        # print(calcTE.getProperty('k_HISTORY'))
        # print(calcTE.getProperty('k_TAU'))

    else:
        calcTE.setProperty("k_HISTORY", str(k_hist))
        calcTE.setProperty("k_TAU", str(k_tau))
        calcTE.setProperty("l_HISTORY", str(l_hist))
        calcTE.setProperty("l_TAU", str(l_tau))

    calcTE.initialise()
    calcTE.setObservations(source, destination)

    if calc_type == 'local':
        TEvalue = calcTE.computeLocalOfPreviousObservations()[:]
    elif calc_type == 'average':
        TEvalue = calcTE.computeAverageLocalOfObservations()
    else:
        print('Calculation Type not right')
        return None
    return TEvalue

def calc_networkTE(Network, average = False, samples_per_sec = 10,average_mode = 'time'):
    if not hasattr(Network, 'sampling'):
        dt = Network.TimeVector[1] - Network.TimeVector[0]
        sampling = np.arange(0,Network.TimeVector.size, int(1/dt/samples_per_sec))
        sampling = sampling
        Network.sampling = sampling
    else:
        sampling = Network.sampling
        
    wireVoltage = Network.wireVoltage
    E = Network.numOfJunctions
    TE = np.zeros((sampling.size, E))
    edgeList = Network.connectivity.edge_list
    mean_direction = np.sign(np.mean(Network.filamentState, axis=0))
    for i in tqdm(range(len(edgeList)), desc = 'Calculating TE '):
        if mean_direction[i] >= 0:
            wire1, wire2 = edgeList[i,:]
        else:
            wire2, wire1 = edgeList[i,:]
        TE[:,i] = calc_TE(wireVoltage[sampling, wire1], wireVoltage[sampling, wire2], calculator = 'gaussian', calc_type = 'local')
        #     TE[:,i] = calc_TE(wireVoltage[sampling, wire2], wireVoltage[sampling, wire1], calculator = 'gaussian', calc_type = 'local')
    
    Network.TE = TE

    if average:
        if average_mode == 'time':
            return np.mean(TE, axis = 0)
        elif average_mode == 'junction':
            return np.mean(TE, axis = 1)
    else:
        return TE

def calc_network(Network, dt_sampling = 1e-1, N = 1e3, t_start=10, calculator = 'kraskov', 
                 return_sampling = False, disable_tqdm = False):
    dt_euler = Network.TimeVector[1] - Network.TimeVector[0]
    sample_start = int(t_start/dt_euler)
    sample_end = sample_start + int(N*dt_sampling/dt_euler)
    sampling = np.arange(sample_start, sample_end, int(dt_sampling/dt_euler))
    if sampling[-1] > Network.TimeVector.size:
        return None
    
    wireVoltage = Network.wireVoltage
    E = Network.numOfJunctions
    TE = np.zeros((sampling.size, E))
    edgeList = Network.connectivity.edge_list
    mean_direction = np.sign(np.mean(Network.filamentState, axis=0))
    for i in tqdm(range(len(edgeList)), desc = 'Calculating TE ', disable = disable_tqdm):
        if mean_direction[i] >= 0:
            wire1, wire2 = edgeList[i,:]
        else:
            wire2, wire1 = edgeList[i,:]
        TE[:,i] = calc_TE(wireVoltage[sampling, wire1], wireVoltage[sampling, wire2], calculator = calculator, calc_type = 'local')
#         TE[:,i] = calc_TE(wireVoltage[sampling, wire2], wireVoltage[sampling, wire1], calculator = 'gaussian', calc_type = 'local')
    if return_sampling:
        return TE, sampling
    else:
        return TE

def TE_multi(Network, dt_sampling = 1e-1, N = 1e3, t_start=10, calculator = 'kraskov', return_sampling = False, disable_tqdm = False):
    dt_euler = Network.TimeVector[1] - Network.TimeVector[0]
    sample_start = int(t_start/dt_euler)
    sample_end = sample_start + int(N*dt_sampling/dt_euler)
    sampling = np.arange(sample_start, sample_end, int(dt_sampling/dt_euler))
    if sampling[-1] > Network.TimeVector.size:
        print('Simulation length not enough for sampling.')
        return None
    
    wireVoltage = Network.wireVoltage
    E = Network.numOfJunctions
    TE = np.zeros((sampling.size, E))
    edgeList = Network.connectivity.edge_list
    mean_direction = np.sign(np.mean(Network.filamentState, axis=0))
    calcList = []
    for i in range(len(edgeList)):
        if mean_direction[i] >= 0:
            wire1, wire2 = edgeList[i,:]
        else:
            wire2, wire1 = edgeList[i,:]
        calcList.append(inputPacker(calc_TE, wireVoltage[sampling, wire1], wireVoltage[sampling, wire2], calculator = calculator, calc_type='local'))

    with Pool(processes=4) as pool:    
        result = list(tqdm(pool.istarmap(calc_TE, calcList), total = len(calcList), desc = f'Calculating TE with {pool._processes} processors.', disable=disable_tqdm))
    return np.array(result).T