import networkx as nx
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import config

G = nx.cycle_graph(sp.symbols('a:d'))
pos = nx.circular_layout(G)

cliques = nx.find_cliques(G)


fig, axs = plt.subplots(2, 2)
for clique, ax in zip(cliques, np.ravel(axs)):
    # ax.axis('off')
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, nodelist=[n for n in G if n in clique], node_color='r', ax=ax)
plt.show()
