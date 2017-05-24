import numpy as np
import setting

def activationFunction(netinput, T):
    y = 1 / (1 + np.exp(-netinput/T))
    return y


def updateActivation(M,inputArray, updatedArray): # same  as  update forward activation but might change later on
    net = np.zeros(len(updatedArray))
    for i in xrange(len(updatedArray)):
        net[i] =  np.dot( inputArray.T,  M[i] )  # take transpose of input units
        '''Using logarithmic function '''
        #inputArray[i] = activationFunction(net[i], 0.5)
        '''Using function from article:'''
        if net[i] >= 0:
            updatedArray[i] = updatedArray[i] + setting.lamb * ((setting.aMax -updatedArray[i]) * net[i] - setting.decay * (updatedArray[i]-setting.rest))
        else:
            updatedArray[i] = updatedArray[i] + setting.lamb * ((updatedArray[i] - setting.aMin) * net[i] - setting.decay * (updatedArray[i] - setting.rest))

        #clamp units to max or min
        if updatedArray[i] > setting.aMax:
            updatedArray[i] = setting.aMax
        elif updatedArray[i] < setting.aMin:
            updatedArray[i] = setting.aMin


    return updatedArray



def updateRecurrentActivation(M,inputArray, updatedArray): # same  as  update forward activation but might change later on

    for i in xrange(len(updatedArray)):
        net = 0
        for j in xrange(len(inputArray)): # much faster when taking transpose
            net =  net + M[i][j] * inputArray[j] # implement this as the transpose of M in the original function

        '''Using logarithmic function '''
        #inputArray[i] = activationFunction(net[i], 0.5)
        '''Using function from article:'''
        if net >= 0:
            updatedArray[i] = updatedArray[i] + setting.lamb * (
            (setting.aMax - updatedArray[i]) * net - setting.decay * (updatedArray[i] - setting.rest))
        else:
            updatedArray[i] = updatedArray[i] + setting.lamb * (
            (updatedArray[i] - setting.aMin) * net - setting.decay * (updatedArray[i] - setting.rest))

        # clamp units to max or min
        if updatedArray[i] > setting.aMax:
            updatedArray[i] = setting.aMax
        elif updatedArray[i] < setting.aMin:
            updatedArray[i] = setting.aMin
    return updatedArray

