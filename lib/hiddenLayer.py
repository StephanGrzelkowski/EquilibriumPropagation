import numpy as np
import helpers



def buildHiddenLayer(nHiddenUnits):
    arrHiddenUnits = np.ones(nHiddenUnits)
    return arrHiddenUnits

def updateForwardActivation(M1,arrInputUnits, arrHiddenUnits):
    net = np.zeros(len(arrHiddenUnits))
    for i in xrange(len(arrHiddenUnits)):
        net[i] = np.dot( arrInputUnits.T * M1[i] )  # take transpose of input units
        arrHiddenUnits[i] = helpers.activationFunction(net[i], 0.5)
    return arrHiddenUnits

def updateRecurrentActivation(M2,arrOutputUnits, arrHiddenUnits): # same  as  update forward activation but might change later on
    net = np.zeros(len(arrHiddenUnits))
    for i in xrange(len(arrHiddenUnits)):
        net[i] =  np.dot( arrOutputUnits.T * M2[i] )  # take transpose of input units
        arrHiddenUnits[i] = helpers.activationFunction(net[i], 0.5)



