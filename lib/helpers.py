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



def updateRecurrentActivation(M,inputArray, outputArray): # same  as  update forward activation but might change later on
    net = np.zeros(len(outputArray))
    for i in xrange(len(outputArray)):
        for i in xrange(len(inputArray)):
            net[i] =  np.dot( inputArray.T,  M[i] )  # take transpose of input units
            '''Using logarithmic function '''
            #inputArray[i] = activationFunction(net[i], 0.5)
            '''Using function from article:'''
            if net[i] >= 0:
                inputArray[i] =inputArray[i] + setting.lamb * ((setting.aMax -inputArray[i]) * net[i] - setting.decay * (inputArray[i]-setting.rest))
            else:
                inputArray[i] =inputArray[i] + setting.lamb * ((inputArray[i] - setting.aMin) * net[i] - setting.decay * (inputArray[i] - setting.rest))

            #clamp units to max or min
            if inputArray[i] > setting.aMax:
                inputArray[i] = setting.aMax
            elif inputArray[i] < setting.aMin:
                inputArray[i] = setting.aMin


    return outputArray