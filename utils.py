import time
import datetime
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


'''
Custom 'print' function
    statements      : Print statements
    end             : End token

    return          : None
'''
def debug_print(statements=[], end='\n'):
    ct = datetime.datetime.now()
    print('[', str(ct)[:19], '] ', sep='', end='')
    for statement in statements:
        print(statement, end=' ')
    print(end=end)

def time_function(func, params=[], name=''):
    start_time = time.time()
    func(*params)
    end_time = time.time()

    print(f'Time [{name}]:', round(end_time - start_time, 4), 's')


# fix this
def visualize_network(network, paths=[], show_weight=False):
    G = nx.Graph(network.adjacency_matrix)
    G_ = G.copy()
    G_weight = nx.Graph()

    pos = network.neuron_positions

    # Set neuron colors
    input_neuron_color = (0.1, 0.35, 0.95)
    output_neuron_color = (0.9, 0.7, 0.2)
    hidden_neuron_color = (0.2, 0.2, 0.2)
    neuron_colors = [hidden_neuron_color for _ in range(len(G.nodes))]
    neuron_colors = [input_neuron_color if i in network.input_neuron_indices else neuron_colors[i] for i in range(len(G.nodes))]
    neuron_colors = [output_neuron_color if i in network.output_neuron_indices else neuron_colors[i] for i in range(len(G.nodes))]

    # Set neuron sizes
    neuron_sizes = [15 for _ in range(len(G.nodes))]
    neuron_sizes = [30 if i in network.input_neuron_indices + network.output_neuron_indices else neuron_sizes[i] for i in range(len(G.nodes))]

    edge_attributes = {}
    for path in paths:
        path_edge_attributes = {edge: {'color': 'skyblue', 'width': 0.1} for edge in zip(path[:-1], path[1:])}
        edge_attributes = {**edge_attributes, **path_edge_attributes}

    min_weight_color = np.array([0, 0, 0])
    max_weight_color = np.array([0.6, 0.75, 0.9])
    if show_weight:
        max_weight = 1
        for neuron in network.neurons: 
            for weight in neuron.weights: max_weight = max(max_weight, np.abs(np.sum(weight)))

        for i, neuron in enumerate(network.neurons):
            for weight, next in zip(neuron.weights, neuron.next):
                if not next: continue
                next_i = network.neurons.index(next)
                weight_mag = np.abs(np.sum(weight)) / max_weight
                G_weight.add_edge(i, next_i, weight=weight_mag)
                edge_attributes[(i, next_i)] = {'color': min_weight_color + 0.6 * max_weight_color * weight_mag, 'weight':3}
                G[i][next_i]['weight'] = 0.1

    nx.set_edge_attributes(G, edge_attributes)
                
    print(edge_attributes)

    max_weight = np.max([np.sum(neuron.weights) for neuron in network.neurons])

    nx.draw(
        G, 
        pos, 
        with_labels=False, 
        font_weight='bold', 
        node_size=neuron_sizes,
        node_color=neuron_colors, 
        font_color='black', 
        font_size=8, 
        edge_color=(0.4, 0.4, 0.4), 
        width=[np.sum(neuron.weights) / max_weight for neuron in network.neurons]
    )
    # nx.draw_networkx_edges(G_, pos)

    # Create legend
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=input_neuron_color, markersize=8, label='Input Neurons'),
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=output_neuron_color, markersize=8, label='Output Neurons'),
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=hidden_neuron_color, markersize=6, label='Hidden Neurons')]

    plt.legend(handles=legend_elements, loc='upper right')

    plt.show()