import numpy as np
import helpers
def buildOutputLayer():
    arrOutput = np.zeros(10)

    return arrOutput


def updateRecurrentActivation(M2,arrHiddenUnits, arrOutputUnits): # same  as  update forward activation but might change later on
    net = np.zeros(len(arrOutputUnits))
    for i in xrange(len(arrOutputUnits)):
        net[i] =  np.dot( arrHiddenUnits.T * M2[i] )  # take transpose of input units
        arrHiddenUnits[i] = helpers.activationFunction(net[i], 0.5)
    return arrOutputUnits