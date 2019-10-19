import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
plt.style.use('classic')
import logging

import os
os.chdir('..')
import wires

def wires_tester(number_of_wires, avg_length, centroid_dispersion, out_mode='number_of_junctions', attempt_limit=5):
    counter = 0
    out = -1
    while (out == -1)&(counter<attempt_limit):
        temp_seed = np.random.randint(2000)
        temp_distribution = wires.generate_wires_distribution(number_of_wires = number_of_wires,
                                                            wire_av_length = avg_length,
                                                            wire_dispersion = 20,
                                                            gennorm_shape = 3,
                                                            centroid_dispersion = centroid_dispersion,
                                                            Lx = 2e2,
                                                            Ly = 2e2,
                                                            this_seed = temp_seed)
        counter += 1
        temp_distribution = wires.detect_junctions(temp_distribution)
        temp_distribution = wires.generate_graph(temp_distribution)
        if wires.check_connectedness(temp_distribution):
            logging.info(f"{temp_distribution['number_of_wires']} nanowire network is connected with seed {temp_seed}, average length of {temp_distribution['avg_length']}, dispersion of {temp_distribution['centroid_dispersion']}.")
            temp_distribution['clustering'] = nx.average_clustering(temp_distribution['G'])
            temp_distribution['avg_path_length'] = nx.average_shortest_path_length(temp_distribution['G'])
            out = temp_distribution[out_mode]
        else:    
            logging.warning(f"{temp_distribution['number_of_wires']} nanowire network is NOT connected. Current seed is {temp_seed}, average length is {temp_distribution['avg_length']}, dispersion is {temp_distribution['centroid_dispersion']}.")
    return out

def draw_dist(wires_dict):
    fig, ax = plt.subplots()
    fig.set_size_inches(5,5)

    Lx = wires_dict['length_x']
    Ly = wires_dict['length_y']

    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0,0), Lx, Ly, color=(1.0, 0.918, 0.0), alpha=0.77))  
    ax = wires.draw_wires(ax, wires_dict)
    ax.axis([-.1*Lx,1.1*Lx,-.1*Lx,1.1*Lx]) 
    plt.show()
    return ax