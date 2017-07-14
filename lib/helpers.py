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

    activityChangeTotal = 0
    for i in xrange(len(updatedArray)):

        net =  np.dot( inputArray.T,  M[:,i] )  # take transpose of input units

        '''Using sigmoid function '''
        activityChange = setting.lamb * (-updatedArray[i] + activationFunction(net, 1))
        updatedArray[i] = updatedArray[i] + activityChange

        if setting.debugWeights == 1:
            activityChangeTotal = activityChangeTotal + np.square(activityChange)

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

    if setting.debugWeights == 1:
        print "Total change of Activity in forward update: ", activityChangeTotal
    return updatedArray


def updateActivationLinear(M, inputArray, updatedArray):
    activityChangeTotal = 0
    for i in xrange(len(updatedArray)):
        net =  np.dot(inputArray.T,  M[:,i])  # take transpose of input units

        activityChange = setting.lamb * (-updatedArray[i] + linearFunction(net, 1))
        updatedArray[i] = updatedArray[i] + activityChange

        if setting.debugWeights == 1:
            activityChangeTotal = activityChangeTotal + np.square(activityChange)


    if setting.debugWeights == 1:
        print "Total squared change of Activity in forward linear activation: ", activityChangeTotal
    return updatedArray

def updateRecurrentActivation(M, inputArray, updatedArray):

    activityChangeTotal = 0

    for i in xrange(len(updatedArray)):

        net = np.dot(inputArray.T, M[i,:].T)

        '''Using sigmoid  function '''
        activityChange = setting.lamb * (-updatedArray[i] + activationFunction(net, 1))

        updatedArray[i] = updatedArray[i] + activityChange

        if setting.debugWeights == 1:
            activityChangeTotal = activityChangeTotal + np.square(activityChange)

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
    if setting.debugWeights == 1:
        print "Total change of Activity in recurrent update: ", activityChangeTotal, "\n"
    return updatedArray

def collectCrossProducts(firstArray, secondArray):
    crossProducts = np.zeros(len(firstArray),len(secondArray))
    for i in xrange(len(firstArray)):
        for j in xrange(len(secondArray)):
            crossProducts[i][j] = firstArray[i] * secondArray[j]


def calcError(arrOutputUnits, label):
    target = np.zeros(len(arrOutputUnits)) - 1
    target[label] = 1
    error = np.sum(np.square(arrOutputUnits - target))
    return error