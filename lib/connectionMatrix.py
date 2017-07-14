import numpy as np
import helpers
import setting
def resetConnections(arrInputUnits, arrHiddenUnits, arrOutputUnits):
    W1 = np.matrix(setting.varWeights * np.random.randn(len(arrInputUnits),len(arrHiddenUnits)))
    W2 = np.matrix(setting.varWeights * np.random.randn(len(arrHiddenUnits),len(arrOutputUnits)))

    return W1, W2


def updateWeights(M, inputArray, outputArray, phase):
    U = np.matrix(np.zeros([len(inputArray), len(outputArray)]))

    for i in xrange(len(inputArray)):
        for j in xrange(len(outputArray)):
            cross = inputArray[i] * outputArray[j]
            if cross != 0:
                U[i,j] = (phase * setting.epsilon * cross)

    return U


