import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from jpype import *
import time

from utils import *
from analysis.InfoTheory import calc_network, TE_multi
from analysis.mkg import mkg_generator
from draw.draw_graph import draw_graph
import pickle
plt.style.use('classic')
from multiprocessing import Pool
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


if __name__ == '__main__':
    sim1 = defaultSimulation(connectivity__(filename = '100nw_261junctions.mat'), T = 12, onAmp = 4, biasType = 'Sawtooth',findFirst = False)
    logging.info('Simulation Finished, starting multi-calculating with 4 cores')
    TE1 = TE_multi(sim1, calculator = 'gaussian', N = 1e3, dt_sampling = 1e-3, t_start = 1)
    logging.info('multi finished, starting series calculation')
    TE2 = calc_network(sim1, calculator = 'gaussian', N = 1e3, dt_sampling = 1e-3, t_start = 1)
    logging.info('haha')