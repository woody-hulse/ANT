from utils import *

'''
File of toy datasets for continuous training
 * indicates models can perform sufficiently well at task
'''


'''
Fit a cosine wave given a sine wave *
    Simple function approximation
    1 in, 1 out
'''
def sinusoidal(time):
    X, Y = [arr([np.sin(t / 100)]) for t in range(time)], [arr([np.cos(t / 100)]) for t in range(time)]
    return X, Y


'''
Fit a cosine wave given nothing *
    Simple function approximation
    1 in, 1 out
'''
def sinusoidal_no_input(time):
    X, Y = [arr([1]) for t in range(time)], [arr([np.cos(t / 100)]) for t in range(time)]
    return X, Y


'''
Create a pulse every n timesteps
    Simple memory
    1 in, 1 out
'''
def pulse(time, n=3):
    X, Y = [arr([1]) for t in range(time)], [arr([int(t % n == 0)]) for t in range(time)]
    return X, Y


'''
Create a ripple pulse every n timesteps
    Simple memory
    1 in, 1 out
'''
def ripple_pulse(time, n=10):
    X, Y = [arr([1]) for t in range(time)], [arr(pow(-1, t % n) / (t % n + 1)) for t in range(time)]
    return X, Y


'''
Create a sinusoidal ripple pulse every n timesteps *
    Simple memory
    1 in, 1 out
'''
def ripple_sinusoidal_pulse(time, n=20):
    X, Y = [arr([1]) for t in range(time)], [arr(np.sin((t % n) / n * 10) / (t % n + 1)) for t in range(time)]
    return X, Y


'''
Create a wave that is the difference between two other waves
    Simple computation
    2 in, 1 out
'''
def subtract_sinusoidal(time):
    # params

    p1, p2 = np.random.normal() * 0.01, np.random.normal() * 0.01
    v1, v2 = np.random.normal() * 0.1, np.random.normal() * 0.1
    a1, a2 = np.random.normal(), np.random.normal()

    X = []
    for _ in range(time):
        a1 += np.random.normal(-v1 * 10) * 0.001
        a2 += np.random.normal(-v2 * 10) * 0.001
        
        v1 += np.random.normal(np.random.normal(-v1 * 100)) * 0.01
        v2 += np.random.normal(np.random.normal(-v2 * 100)) * 0.01

        p1 += v1
        p2 += v2

        X.append(arr([p1, p2]))
    
    Y = [arr(x1 - x2) for x1, x2 in X]

    # plt.plot(X)
    # plt.plot(Y)
    # plt.show()

    return X, Y



'''
Add n random bits
    Simple computation
    n in, 1 out
'''
def bits(time, n=2):
    X = [arr([np.random.randint(0, 2), np.random.randint(0, 2)]) for t in range(time + 1)]
    Y = [sum(X[t + 1]) for t in range(time + 1)]
    return X[1:], Y[:-1]


def binary(time):
    X = np.array([[1] for _ in range(time)])
    Y = np.array([[t % 2] for t in range(time)])
    return X, Y