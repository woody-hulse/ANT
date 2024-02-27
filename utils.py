import time


def time_function(func, params=[], name=''):
    start_time = time.time()
    func(*params)
    end_time = time.time()

    print(f'Time [{name}]:', round(end_time - start_time, 4), 's')
