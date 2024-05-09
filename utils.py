import time
import datetime
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import csv
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


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def weight_function(network, edge):
    weights = network.neurons[edge[1]].weights
    out_index = network.neurons[edge[1]].prev.index(network.neurons[edge[0]])
    weight = np.sum(np.abs(weights[out_index]))
    return weight

def plot_graph_simple(adjacency_matrix, input_neurons=[], output_neurons=[]):
    num_neurons = adjacency_matrix.shape[0]
    G = nx.Graph(adjacency_matrix)
    plt.figure(figsize=(6, 3))

    neuron_sizes = [3 for _ in range(len(G.nodes))]
    neuron_sizes = [20 if i in input_neurons + output_neurons else neuron_sizes[i] for i in range(len(G.nodes))]

    input_neuron_color = '#883ab5'
    output_neuron_color = '#3ab1b5'
    hidden_neuron_color = (0.4, 0.4, 0.4)
    neuron_colors = [hidden_neuron_color for _ in range(len(G.nodes))]
    neuron_colors = [input_neuron_color if i in input_neurons else neuron_colors[i] for i in range(len(G.nodes))]
    neuron_colors = [output_neuron_color if i in output_neurons else neuron_colors[i] for i in range(len(G.nodes))]

    pos = nx.spring_layout(G, k=1/np.sqrt(num_neurons), iterations=20)
    edge_color = '#b6b6b6'
    width = 0.2
    nx.draw(G, pos=pos, with_labels=False, node_color=neuron_colors, node_size=neuron_sizes, edge_color=edge_color, width=width, edgecolors='#444444')

    legend_elements = [
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=input_neuron_color, markersize=8, label='Input (sensory) neurons'),
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=output_neuron_color, markersize=8, label='Output (motor) neurons'),
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=hidden_neuron_color, markersize=4, label='Hidden (inter) neurons')]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.show()



def plot_graph(network, title='', spring=False, save=False, save_directory='graph_images/'):
    G = nx.Graph()
    plt.figure(figsize=(6, 3))

    for i, node in enumerate(network.neurons):
        G.add_node(i)

    # Set neuron colors
    input_neuron_color = '#883ab5'
    output_neuron_color = '#3ab1b5'
    hidden_neuron_color = (0.4, 0.4, 0.4)
    neuron_colors = [hidden_neuron_color for _ in range(len(G.nodes))]
    neuron_colors = [input_neuron_color if i in network.input_neuron_indices else neuron_colors[i] for i in range(len(G.nodes))]
    neuron_colors = [output_neuron_color if i in network.output_neuron_indices else neuron_colors[i] for i in range(len(G.nodes))]
    # neuron_colors = [critic_neuron_color if i in network.critic_neuron_indices else neuron_colors[i] for i in range(len(G.nodes))]

    # Set neuron sizes
    neuron_sizes = [3 for _ in range(len(G.nodes))]
    neuron_sizes = [20 if i in network.input_neuron_indices + network.output_neuron_indices else neuron_sizes[i] for i in range(len(G.nodes))]
    # neuron_sizes = [3 if i in network.critic_neuron_indices else neuron_sizes[i] for i in range(len(G.nodes))]
    
    rows, cols = np.where(network.adjacency_matrix > 0)
    weights = np.zeros(len(rows.tolist()))
    for i, edge in enumerate(zip(rows.tolist(), cols.tolist())):
        weights[i] = weight_function(network, edge)
    weights /= np.max(weights) + 1e-5
    weights += 0.01

    for i, edge in enumerate(zip(rows.tolist(), cols.tolist())):
        G.add_edge(edge[0], edge[1], weight=weights[i])
    
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    if spring: 
        pos = nx.spring_layout(G, k=6/np.sqrt(network.num_neurons), pos=network.neuron_positions, iterations=2)
        network.neuron_positions = pos
    else: pos = network.neuron_positions
    edge_color = '#b6b6b6'
    # edge_color = weights
    width = 0.2
    # width = weights
    nx.draw(G, pos=pos, with_labels=False, node_color=neuron_colors, node_size=neuron_sizes, edge_color=edge_color, width=width, edgecolors='#444444')
    
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=input_neuron_color, markersize=8, label='Input neurons'),
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=output_neuron_color, markersize=8, label='Output neurons'),
        matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=hidden_neuron_color, markersize=4, label='Hidden neurons')]

    plt.legend(handles=legend_elements, loc='upper right')

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


def visualize_episode(network, env, name, time=500):
    # debug_print(['Visualizing episode'])

    state = env.reset()

    image_frames = []
    for t in tqdm(range(time)):
        logits = network.forward(state)
        action = np.argmax(logits)
        state, reward, done = env.step(action)
        
        image_array = env.env.render()
        image_frame = Image.fromarray(image_array)
        image_frames.append(image_frame)
        
        if done: break

    image_frames[0].save(name + '.gif', 
                        save_all = True, 
                        duration = 20,
                        loop = 0,
                        append_images = image_frames[1:])
    

def visualize_evolution(network, gif=False, mutation_args=None, time=10, spring=True):
    if not mutation_args:
        mutation_args = {
            'neuron_mutation_rate': 1,
            'edge_mutation_rate': 0.05,
            'weight_mutation_rate': 1e-4
        }
    if gif: save=True
    else: save=False
    for i in tqdm(range(time)):
        network.mutate(**mutation_args)
        # network.print_info()
        plot_graph(network, title=f'e{i}', spring=spring, save=save, save_directory='graph_evolution/')
    
    if gif: convert_files_to_gif('graph_evolution/', f'{network.name}_evolution.gif')


def load_csv_to_adjacency_matrix(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    num_nodes = len(data)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            if row_idx == col_idx: continue
            if not value == '': adjacency_matrix[row_idx, col_idx] = 1

    return adjacency_matrix


if __name__ == '__main__':

    adjacency_matrix = load_csv_to_adjacency_matrix('../celegans_adjacency.csv')
    input_neurons = [i for i in range(36)]
    output_neurons = [i for i in range(81, 91)]
    plot_graph_simple(adjacency_matrix, input_neurons=input_neurons, output_neurons=output_neurons)


    '''
    # Sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(8, 6), dpi=300)  # High-resolution setting
    plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)  # Simple color and line width

    
    plt.title('Template Line Plot', fontsize=14)
    plt.xlabel('X Axis', fontsize=12)
    plt.ylabel('Y Axis', fontsize=12)

    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('line_plot.png')
    '''