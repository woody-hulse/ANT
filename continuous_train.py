from continuous_network import ContinuousNetwork, MeanSquaredError, SGD, ADAM, RMSProp

from utils import *


def train(network, X, Y, epochs=10):
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


def continuous_train(network, time=1000, plot=True):
    loss_layer = MeanSquaredError()
    energies = []
    gradient_energies = []
    weight_energies = []
    predictions = []
    losses = []

    pbar = tqdm(range(time))

    for t in pbar:
        x = np.array([int(t % 2 == 0)])
        y = np.array([10])

        y_hat = network.forward_pass(x, decay=0)
        loss = loss_layer(y, y_hat)
        internal_energy = min(network.compute_internal_energy(), 1000)
        gradient_energy = min(network.compute_gradient_energy(), 1000)
        weight_energy = min(network.compute_weight_energy(), 1000)
        prediction = max(min(y_hat, 1000), -100)
        output_grad = np.sum([np.sum(np.abs(output.input_J)) for output in network.output_neurons])
        
        energies.append(internal_energy)
        gradient_energies.append(gradient_energy)
        weight_energies.append(weight_energy)
        predictions.append(prediction)
        losses.append(min(loss, 1000))

        pbar.set_description(f"t={t:05} loss={loss:.3f} energy={internal_energy:.3f} grad_energy={gradient_energy:.3f} pred={prediction[0]:.3f} output_grad={output_grad:.3f}")
        
        network.backward_pass(loss_layer, y, y_hat, decay=0, t=t+1)
    
    if plot:
        plt.plot(predictions, label='Network predictions')
        # plt.plot(energies, label='Network output energies')
        plt.plot(gradient_energies, label='Network gradient energies')
        # plt.plot(weight_energies, label='Network weight energies')
        plt.plot(losses, label='Network loss (MSE)')
        plt.plot([0 for _ in range(time)], linestyle=':')
        plt.plot()
        plt.legend()
        plt.show()

def main():
    network = ContinuousNetwork(
    num_neurons         = 32,
    edge_probability    = 1,
    num_input_neurons   = 1,
    num_output_neurons  = 1
    )
    # visualize_network(network)
    # visualize_network(network, show_weight=True)

    network.print_info()

    dummy_X = np.array([[np.random.randint(0, 10) for _ in range(1)] for _ in range(1000)])
    dummy_Y = np.array([np.sum(x) for x in dummy_X])

    # train(network, dummy_X, dummy_Y, epochs=100)

    sgd = SGD(alpha=1e-4)
    adam = ADAM(alpha=1e-4, beta1=0.9, beta2=0.99, reg=0.001)
    rmsprop = RMSProp(alpha=4e-4, beta=0.99, reg=0)

    network.set_optimizer(rmsprop)
    continuous_train(network, time=2000, plot=True)


if __name__ == '__main__':
    main()