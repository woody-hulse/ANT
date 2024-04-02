'''
Superclass of differentiable objects
'''
class Diff():
    def __init__(self):
        pass

    def dydx(self):
        raise Exception('Input gradient not defined')
    
    def dydw(self):
        raise Exception('Weight gradient not defined')
    
    def dydb(self):
        raise Exception('Bias gradient not defined')