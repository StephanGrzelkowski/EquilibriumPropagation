import numpy as np
import helpers
def resetConnections(arrInputUnits, arrHiddenUnits, arrOutputUnits):
    W1 = np.ones([len(arrInputUnits), len(arrHiddenUnits)])
    W2 = np.ones([len(arrHiddenUnits), len(arrOutputUnits)])

    return W1, W2




