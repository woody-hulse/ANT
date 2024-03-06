import time
import datetime
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import imageio
import tracemalloc
import seaborn

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


def plot_graph(network, title='', save=False, save_directory='graph_images/'):
    G = nx.Graph()

    for i, node in enumerate(network.neurons):
        G.add_node(i)

    # Set neuron colors
    input_neuron_color = (0.1, 0.35, 0.95)
    output_neuron_color = (0.9, 0.7, 0.2)
    hidden_neuron_color = (0.2, 0.2, 0.2)
    neuron_colors = [hidden_neuron_color for _ in range(len(G.nodes))]
    neuron_colors = [input_neuron_color if i in network.input_neuron_indices else neuron_colors[i] for i in range(len(G.nodes))]
    neuron_colors = [output_neuron_color if i in network.output_neuron_indices else neuron_colors[i] for i in range(len(G.nodes))]

    # Set neuron sizes
    neuron_sizes = [0 for _ in range(len(G.nodes))]
    neuron_sizes = [30 if i in network.input_neuron_indices + network.output_neuron_indices else neuron_sizes[i] for i in range(len(G.nodes))]
    
    rows, cols = np.where(network.adjacency_matrix > 0)
    edges = zip(rows.tolist(), cols.tolist())
    for edge in edges:
        J = network.neurons[edge[1]].inputs
        out_index = network.neurons[edge[1]].prev.index(network.neurons[edge[0]])
        weight = np.sum(J[out_index])
        G.add_edge(edge[0], edge[1], weight=weight)
    
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    nx.draw(G, pos=network.neuron_positions, with_labels=False, node_color=neuron_colors, node_size=neuron_sizes, edge_color=weights, width=1)
    
    plt.title(title)
    if save:
        plt.savefig(save_directory + title)
        plt.close()
    else: plt.show()



def convert_files_to_gif(directory, name):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory, filename)
            images.append(imageio.imread(file_path))

    with imageio.get_writer(f'{name}', mode='I') as writer:
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image = imageio.imread(directory + filename)
                writer.append_data(image)

    # imageio.mimsave(os.path.join(directory, name), images, fps=1)
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def arr(array):
    return np.array(array)