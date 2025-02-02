import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
from typing import Literal

def visualize_signal_over_graph(G,
                                signal,
                                pos,
                                ax=None,
                                cmap: str = 'viridis',
                                nodesize: float = 65,
                                edgewidth: float = 0.85):
    '''
    Visualizes a signal over a graph using a specified colormap.
    '''
    if ax is None:
        ax = plt.gca()
    norm = plt.Normalize(vmin=signal.min(), vmax=signal.max())
    node_colors = plt.cm.get_cmap(cmap)(norm(signal))

    nx.draw(G,
            pos,
            node_color=node_colors,
            with_labels=False,
            edgecolors='black',
            node_size=nodesize,
            width=edgewidth,
            edge_color='#C6C6C6',
            ax=ax)

    ax.axis('off')

    return norm, cmap