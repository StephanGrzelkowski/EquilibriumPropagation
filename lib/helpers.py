import numpy as np
import setting

def activationFunction(net, T):
    y = 1 / (1 + np.exp(-net/T))
    return y


def updateActivation(M1, M2, inputUnits, hiddenUnits, outputUnits): # same  as  update forward activation but might change later on

    net1 = np.dot(inputUnits, M1)
    net2 = np.dot(M2, outputUnits)
    updatedArray = hiddenUnits + (setting.lamb * (-hiddenUnits + activationFunction((net1 + net2), 1)))
    maxDelta = np.max(np.square(updatedArray - hiddenUnits))
    return updatedArray, maxDelta


def updateActivationLinear(M, hiddenUnits, outputUnits):
    net = np.dot(hiddenUnits, M)
    updatedArray = outputUnits + (setting.lamb * (-outputUnits + net))
    maxDelta =  np.max(np.square(updatedArray - outputUnits))
    return updatedArray, maxDelta


def calcErrorDiff(arrOutputUnits, label):
    target = np.zeros(len(arrOutputUnits)) #- 1
    target[label] = 1
    negTarget = np.ones(len(arrOutputUnits))
    negTarget[label] = 0
    posTarget = np.zeros(len(arrOutputUnits))
    posTarget[label] = 1
    error = np.square(arrOutputUnits - target)
    errorNeg = (arrOutputUnits >= -1) * negTarget * error
    errorPos = (arrOutputUnits <= 1) * posTarget * error
    errorTot = np.sum(errorNeg + errorPos)
    return errorTot