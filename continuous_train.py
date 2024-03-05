from continuous_network import ContinuousNetwork, MSE, SGD, ADAM, RMSProp

from utils import *
from datasets import *


def train(network, X, Y, epochs=10):
    debug_print([f'Training {network.name}'])
    loss_layer = MSE()
    for epoch in range(epochs):
        total_loss = 0
        train_size = len(X)
        for i in tqdm(range(train_size), desc=f'                 Epoch {epoch + 1} | '):
            x, y = X[i], Y[i]
            y_hat = network.forward_pass(x)
            total_loss += loss_layer(y, y_hat)
            network.backward_pass(loss_layer, y, y_hat, alpha=5e-6)
        debug_print([f'Epoch {epoch + 1} | loss : {total_loss / train_size}'])


def continuous_train(network, X, Y, time=1000, metrics=True, gif=False):
    tracemalloc.start()

    loss_layer = MSE()

    pbar = tqdm(range(time))

    for t in pbar:
        x, y = X[t], Y[t]

        if gif:
            plot_graph(network, title=f't{t}', save=True, save_directory='graph_images/')

        y_hat = network.forward_pass(x, decay=0)
        network.backward_pass(loss_layer, y, y_hat, decay=0, update_metrics=metrics)
        if metrics: pbar.set_description(f't={t+1:05}; ' + network.get_metrics_string())
        else: pbar.set_description(f't={t+1:05}')
    
    if metrics:
        network.plot_metrics()

    if gif:
        convert_files_to_gif(directory='graph_images/', name='network_weights.gif')

    current, peak = tracemalloc.get_traced_memory()
    debug_print([f'Current memory usage is {current / 1024**2:.2f}MB; Peak was {peak / 1024**2:.2f}MB'])
    tracemalloc.stop()

def main():
    network = ContinuousNetwork(
    num_neurons         = 128,
    edge_probability    = 1.3,
    num_input_neurons   = 1,
    num_output_neurons  = 1
    )
    # visualize_network(network)
    # visualize_network(network, show_weight=True)

    network.print_info()
    sgd = SGD(alpha=1e-4)
    adam = ADAM(alpha=1e-4, beta1=0.9, beta2=0.99, reg=0)
    rmsprop = RMSProp(alpha=3e-5, beta=0.99, reg=0)
    network.set_optimizer(rmsprop)

    time = 10000
    X, Y = ripple_sinusoidal_pulse(time, n=20)
    continuous_train(network, X, Y, time=time, metrics=True, gif=False)


if __name__ == '__main__':
    main()