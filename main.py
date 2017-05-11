
import lib.inputUnits
import lib.hiddenLayer
import lib.connectionMatrix
import lib.outputLayer

import numpy as np

'''Settings: '''
#number of hidden Unit
nHiddenUnits = 200
iterations = 30


'''Building Framework'''
# load in MNIST Data
train_set, valid_set, test_set = lib.inputUnits.loadDataset()
arrInputUnits = lib.inputUnits.buildInputUnits(len(train_set[0][0])) # later just make the inputUnits the nth array of
arrHiddenUnits = lib.hiddenLayer.buildHiddenLayer(nHiddenUnits)
arrOutputUnits = lib.outputLayer.buildOutputLayer()
M1, M2 = lib.connectionMatrix.resetConnections(arrInputUnits, arrHiddenUnits, arrOutputUnits)

for i in xrange(iterations):
    arrHiddenUnits = lib.hiddenLayer.updateForwardActivation(M1,arrInputUnits,arrHiddenUnits)
    arrOutputUnits = lib.outputLayer.updateActivation(M2,)


print M1