import wires

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


# Generate the network
wires_dict = wires.generate_wires_distribution(number_of_wires = 200,
                                         wire_av_length = 15,
                                         wire_dispersion = 20,
                                         gennorm_shape = 3,
                                         centroid_dispersion=700.0,
                                         this_seed = 42,
                                         Lx = 3e3,
                                         Ly = 3e3);

# Get junctions list and their positions
wires_dict = wires.detect_junctions(wires_dict)

# Genreate graph object and adjacency matrix
wires_dict = wires.generate_graph(wires_dict)