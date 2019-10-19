import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import is_notebook
from draw.draw_graph import draw_graph
# if is_notebook():
#     from draw.draw_graph import draw_graph
# else:
#     from draw_graph import draw_graph

import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 1e8

class graph_movie:
    def __init__(self, network, start=0, end=1, dt=0.01, interval = 100, figsize=(10,10), edge_mode='current'):
        plt.style.use('classic')
        self.network = network
        self.start = start
        self.end = end
        self.dt = dt
        self.interval = interval
        self.edge_mode = edge_mode
        self.fig, self.ax = plt.subplots(figsize=figsize)
        if is_notebook():
            plt.close()
        self.fig.set_facecolor((0.8,0.8,0.8))
        self.ax.set_facecolor((0.8,0.8,0.8))
        self.ax.set_title(f'Network Current Flow at t = {start}')
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.compile()

    def compile(self):
        self.movie = animation.FuncAnimation(self.fig, self.update, np.arange(self.start, self.end, self.dt), interval=self.interval)
        return self.movie                                                                                    

    def update(self, i):
        self.ax.clear()
        self.ax = draw_graph(network=self.network, ax=self.ax, 
                            edge_mode = self.edge_mode,
                            time = i)
        return self.ax

    def show(self):
        from IPython.display import HTML
        html = HTML(self.movie.to_jshtml())
        plt.close()
        return html
    
    def save(self, filename=None):
        if filename == None:
            from time import gmtime, strftime
            curr = strftime("%Y%m%d_%H%M", gmtime())
            filename = r'movie/random_movie_' + curr + '.mp4'
        else: 
            filename = r'movie/' + filename
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
        self.movie.save(filename, writer = writer)
        if is_notebook():
            plt.close()

if __name__ == '__main__':
    # import os 
    # os.chdir(r'C:\Users\rzhu\Documents\PhD\ASN_simulation\Python\ASN')
    from utils import *

    sim1 = defaultSimulation(connectivity__(filename = '2016-09-08-155153_asn_nw_00100_nj_00261_seed_042_avl_100.00_disp_10.00.mat'),
                        contactMode='preSet', electrodes=[72,29],
                        biasType='DC', offTime = 5, onAmp = 1.1, T = 1)
    movie1 = graph_movie(sim1, start=0.8, end=1, dt=0.01)

    plt.show()
