import numpy as np
import setting

def activationFunction(netinput, T):
    y = 1 / (1 + np.exp(-netinput/T))
    return y

def linearFunction(net,m):
    output = m * net
    return output
def updateActivation(M,inputArray, updatedArray): # same  as  update forward activation but might change later on
    #net = np.zeros(len(updatedArray))


    for i in xrange(len(updatedArray)):

        net =  np.dot( inputArray.T,  M[i] )  # take transpose of input units


        '''Using sigmoid function '''
        updatedArray[i] = activationFunction(net, 1)

        #linear

        '''Using function from article:'''
        """if net[i] >= 0:
            updatedArray[i] = updatedArray[i] + setting.lamb * ((setting.aMax -updatedArray[i]) * net[i] - setting.decay * (updatedArray[i]-setting.rest))
        else:
            updatedArray[i] = updatedArray[i] + setting.lamb * ((updatedArray[i] - setting.aMin) * net[i] - setting.decay * (updatedArray[i] - setting.rest))

        #clamp units to max or min
        if updatedArray[i] > setting.aMax:
            updatedArray[i] = setting.aMax
        elif updatedArray[i] < setting.aMin:
            updatedArray[i] = setting.aMin
        """


    return updatedArray


def updateActivationLinear(M, inputArray, updatedArray):
    for i in xrange(len(updatedArray)):
        net =  np.dot( inputArray.T,  M[i] )  # take transpose of input units

        updatedArray[i] = linearFunction(net,1)
    return updatedArray
def updateRecurrentActivation(M,inputArray, updatedArray):

    for i in xrange(len(updatedArray)):
        net = 0
        for j in xrange(len(inputArray)): # much faster when taking transpose
            net =  net + M[j][i] * inputArray[j] # implement this as the transpose of M in the original function

        '''Using sigmoid  function '''
        updatedArray[i] = activationFunction(net, 1)

        '''Using function from article:'''
        '''if net >= 0:
            updatedArray[i] = updatedArray[i] + setting.lamb * ((setting.aMax - updatedArray[i]) * net - setting.decay * (updatedArray[i] - setting.rest))
        else:
            updatedArray[i] = updatedArray[i] + setting.lamb * ((updatedArray[i] - setting.aMin) * net - setting.decay * (updatedArray[i] - setting.rest))

        # clamp units to max or min
        if updatedArray[i] > setting.aMax:
            updatedArray[i] = setting.aMax
        elif updatedArray[i] < setting.aMin:
            updatedArray[i] = setting.aMin
        '''
    return updatedArray

def collectCrossProducts(firstArray, secondArray):
    crossProducts = np.zeros(len(firstArray),len(secondArray))
    for i in xrange(len(firstArray)):
        for j in xrange(len(secondArray)):
            crossProducts[i][j] = firstArray[i] * secondArray[j]


