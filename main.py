import lib.helpers
import lib.inputUnits
import lib.hiddenLayer
import lib.connectionMatrix
import lib.outputLayer
import setting
import numpy as np

'''Settings: '''
#number of hidden Unit



'''Building Framework'''
# load in MNIST Data
train_set, valid_set, test_set = lib.inputUnits.loadDataset()
arrLabels = train_set[1]

#build layers:
arrInputUnits = train_set[0][0] #lib.inputUnits.buildInputUnits(len(train_set[0][0])) # later just make the inputUnits the nth array of
arrHiddenUnits = lib.hiddenLayer.buildHiddenLayer(setting.nHiddenUnits)
arrOutputUnits = lib.outputLayer.buildOutputLayer()

#build connection matrices
M1, M2 = lib.connectionMatrix.resetConnections(arrInputUnits, arrHiddenUnits, arrOutputUnits)


#free phase
phase = 1
for i in xrange(setting.settlingIterations):
    #update forward activation
    arrHiddenUnits = lib.helpers.updateActivation(M1,arrInputUnits,arrHiddenUnits)
    arrOutputUnits = lib.helpers.updateActivation(M2,arrHiddenUnits, arrOutputUnits)
    #update recurrent activation

    #update connection matrix every settling step
    M1 = lib.connectionMatrix.updateWeights(M1, arrInputUnits, arrHiddenUnits, phase )
    M2 = lib.connectionMatrix.updateWeights(M2, arrHiddenUnits, arrOutputUnits, phase )
print "Settling free phase done"
#clamped phase
phase = -1
arrOutputUnits = np.zeros(10)
arrOutputUnits[arrLabels[0]] = 1

for i in xrange(setting.settlingIterations):
    #update foward activation
    arrHiddenUnits = lib.helpers.updateActivation(M1,arrInputUnits,arrHiddenUnits)

    #update recurrent activation
    arrHiddenUnits = lib.helpers.updateRecurrentActivation(M1, arrOutputUnits, arrHiddenUnits)
    #do this after settling iterations
    M1 = lib.connectionMatrix.updateWeights(M1, arrInputUnits, arrHiddenUnits, phase)

    M2 = lib.connectionMatrix.updateWeights(M2, arrHiddenUnits, arrOutputUnits, phase)
print "settling clamped phase done"



'''Finished recording '''