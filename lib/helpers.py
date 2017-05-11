import numpy as np

def activationFunction(netinput, T):
    y = 1 / (1 + np.exp(-netinput/T))
    return y
