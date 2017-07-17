import numpy as np
import helpers
import setting
def resetConnections(arrInputUnits, arrHiddenUnits, arrOutputUnits):
    W1 = setting.varWeights * np.random.randn(len(arrInputUnits),len(arrHiddenUnits))
    W2 = setting.varWeights * np.random.randn(len(arrHiddenUnits),len(arrOutputUnits))
    return W1, W2


def updateWeights(M, inputArray, outputArray, phase):
    cross = np.dot(inputArray[:,  None], outputArray[None, :])
    U = (phase * setting.epsilon * cross)

    return U


