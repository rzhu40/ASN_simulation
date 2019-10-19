import numpy as np
import matplotlib.pyplot as plt
from jpype import *

import warnings
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

def AIS_play(data, k = 1, tau = 1, 
            calculator='kraskov', calc_type='average'):
    jarLocation = r"C:\Users\rzhu\Documents\PhD\JIDT\infodynamics.jar"
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

def TE_play(source, destination, 
            k_hist = 1, k_tau = 1, l_hist = 1, l_tau = 1, 
            calculator='kraskov', calc_type='average'):

    jarLocation = r"C:\Users\rzhu\Documents\PhD\JIDT\infodynamics.jar"
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

    if calculator == 'kraskov':
        calcTEClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
        calcTE = calcTEClass()
        calcTE.setProperty("NOISE_LEVEL_TO_ADD", "0")
        calcTE.setProperty("ALG_NUM", "1")
    elif calculator == 'gaussian':
        calcTEClass = JPackage("infodynamics.measures.continuous.gaussian").TransferEntropyCalculatorGaussian
        calcTE = calcTEClass()

    calcTE.setProperty("k_HISTORY", str(k_hist))
    calcTE.setProperty("k_TAU", str(k_tau))
    calcTE.setProperty("l_HISTORY", str(l_hist))
    calcTE.setProperty("l_TAU", str(l_tau))

    # calcTE.setProperty("NORM_TYPE", "EUCLIDEAN")

    calcTE.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS")
    calcTE.setProperty("AUTO_EMBED_K_SEARCH_MAX", "2")
    calcTE.setProperty("AUTO_EMBED_TAU_SEARCH_MAX", "2")

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