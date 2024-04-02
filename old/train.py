from old.network import Network, MeanSquaredError

from utils import *

import tensorflow as tf


def train(network, X, Y, epochs=10, max_depth=5):
    debug_print([f'Training {network.name}'])
    loss_layer = MeanSquaredError()
    for epoch in range(epochs):
        total_loss = 0
        train_size = len(X)
        for i in tqdm(range(train_size), desc=f'                 Epoch {epoch + 1} | '):
            x, y = X[i], Y[i]
            y_hat = network.forward_pass(x)
            total_loss += loss_layer(y, y_hat)
            network.backward_pass(loss_layer, y, y_hat, alpha=5e-6)
        debug_print([f'Epoch {epoch + 1} | loss : {total_loss / train_size}'])


def main():

    max_depth = 5

    network = Network(
    num_neurons         = 16,
    edge_probability    = 1,
    num_input_neurons   = 1,
    num_output_neurons  = 1
    )
    visualize_network(network)
    # visualize_network(network, show_weight=True)

    network.print_info()
    network.compile(max_depth=max_depth)

    dummy_X = np.array([[np.random.randint(0, 10) for _ in range(1)] for _ in range(1000)])
    dummy_Y = np.array([np.sum(x) for x in dummy_X])

    train(network, dummy_X, dummy_Y, epochs=100)

    '''
    example_dense_network = tf.keras.Sequential([
        tf.keras.layers.Dense(50),
        tf.keras.layers.Dense(1)
    ])
    example_dense_network.compile(
        optimizer='sgd', loss='mse'
    )
    example_dense_network(dummy_X)
    example_dense_network.summary()

    example_dense_network.fit(dummy_X, dummy_Y, batch_size=1, epochs=10)
    '''


if __name__ == '__main__':
    main()