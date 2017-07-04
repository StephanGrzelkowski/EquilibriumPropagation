import numpy as np
import helpers
import setting
def resetConnections(arrInputUnits, arrHiddenUnits, arrOutputUnits):
    W1 = setting.varWeights * np.random.randn(len(arrHiddenUnits),len(arrInputUnits))
    W2 = setting.varWeights * np.random.randn(len(arrOutputUnits), len(arrHiddenUnits))

    return W1, W2


def updateWeights(M, inputArray, outputArray, phase):

    for i in xrange(len(inputArray)):
        for j in xrange(len(outputArray)):
            cross = inputArray[i] * outputArray[j]
            if cross != 0:
                M[j][i] = M[j][i]  + (phase * setting.epsilon * cross)

    return M


