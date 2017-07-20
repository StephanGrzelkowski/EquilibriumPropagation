import lib.helpers
import lib.inputUnits
import lib.connectionMatrix
import setting
import numpy as np
from sklearn.datasets import load_iris

debug = 0

'''Building Framework'''
#Load Iris
iris = load_iris()
arrPerm = np.random.permutation(150)
arrTrainImages = arrPerm[0:setting.nTrainImages]

arrLabels = iris.target
train_set = iris.data
#build connection matrices
M1 = setting.varWeights * np.random.randn(setting.nInputUnits,setting.nHiddenUnits)
M2 = setting.varWeights * np.random.randn(setting.nHiddenUnits,setting.nOutputUnits)

#Training phase
for k in xrange(setting.batchIterations):

    print "BATCH NUMBER: #",k
    for n in xrange(setting.nTrainImages):
        print "\nTRAINING ON IMAGE:", n,". Label: ", arrLabels[arrTrainImages[n]]

#Free phase
        #initialize layers
        arrInputUnits = train_set[arrTrainImages[n]] #lib.inputUnits.buildInputUnits(len(train_set[0][0])) # later just make the inputUnits the nth array of
        arrHiddenUnits = np.zeros(setting.nHiddenUnits) - setting.rest
        arrOutputUnits = np.zeros(setting.nOutputUnits) - setting.rest

        #update activation of layers

        maxDeltaHid = 1
        maxDeltaOut = 1
        steps = 0
        while (maxDeltaHid > setting.delta) & (maxDeltaOut > setting.delta):
            steps += 1
            arrHiddenUnits, maxDeltaHid = lib.helpers.updateActivation(M1, M2, arrInputUnits, arrHiddenUnits, arrOutputUnits)
            arrOutputUnits, maxDeltaOut = lib.helpers.updateActivationLinear(M2, arrHiddenUnits, arrOutputUnits)

        errorDiff = lib.helpers.calcErrorDiff(arrOutputUnits, arrLabels[arrTrainImages[n]])
        print "Squared Error off target: ", errorDiff
        print "\nSettling free phase done in", steps, "settling iterations"
        print arrOutputUnits

        #Calculate weight change for free phase
        phase = -1
        U1 = lib.connectionMatrix.updateWeights(M1, arrInputUnits, arrHiddenUnits, phase)
        U2 = lib.connectionMatrix.updateWeights(M2, arrHiddenUnits, arrOutputUnits, phase)

#Clamped phase
        arrOutputUnits = np.zeros(setting.nOutputUnits) - 1
        arrOutputUnits[arrLabels[arrTrainImages[n]]] = 1

        maxDeltaHid = 1
        steps = 0
        while maxDeltaHid > setting.delta:
            #update activation
            steps += 1
            arrHiddenUnits, maxDeltaHid = lib.helpers.updateActivation(M1, M2, arrInputUnits, arrHiddenUnits, arrOutputUnits)

        print "Maximal update for Unit:", maxDeltaHid
        print "settling clamped phase done in :", steps, "settling iterations"

        print "\nUpdating the weights"
        #weight update after clamped phase
        phase = 1
        U1 = U1 + lib.connectionMatrix.updateWeights(M1, arrInputUnits, arrHiddenUnits, phase)
        U2 = U2 + lib.connectionMatrix.updateWeights(M2, arrHiddenUnits, arrOutputUnits, phase)

        print "Squared weight update M1: ", np.sum(np.square(U1))
        print "Squared weight update M2: ", np.sum(np.square(U2))
        M1 = M1 + U1
        M2 = M2 + U2

print "\nTraining phase done!\n"

#Test Phase

nCorrect = 0.0
arrTestImages = arrPerm[setting.nTrainImages + 1 : len(arrPerm)]

if setting.debugEql == 1:
    arrTestImages = arrTrainImages
for n in xrange(setting.nTestImages): # test for 10 images
    # get a test image
    arrHiddenUnits = np.zeros(setting.nHiddenUnits) - setting.rest
    arrOutputUnits = np.zeros(setting.nOutputUnits) - setting.rest
    if setting.debugEql == 1:
        arrInputUnits = train_set[arrTestImages[n]]
    else:
        arrInputUnits = train_set[arrTestImages[n]]

    maxDeltaHid = 1
    maxDeltaOut = 1
    steps = 0
    while (maxDeltaHid > setting.delta) & (maxDeltaOut > setting.delta):
        # update forward activation
        arrHiddenUnits, maxDeltaHid = lib.helpers.updateActivation(M1, M2,  arrInputUnits, arrHiddenUnits, arrOutputUnits)
        arrOutputUnits, maxDeltaOut = lib.helpers.updateActivationLinear(M2, arrHiddenUnits, arrOutputUnits)

        if setting.debugEql:
            errorDiff = lib.helpers.calcErrorDiff(arrOutputUnits, arrLabels[arrTrainImages[n]])
        else:
            errorDiff = lib.helpers.calcErrorDiff(arrOutputUnits, arrLabels[arrTestImages[n]])
        print "Squared Error off target: ", errorDiff
        steps += 1
    print "\nResponse to Image", n, arrOutputUnits
    if setting.debugEql == 1:
        if np.argmax(arrOutputUnits) == arrLabels[arrTestImages[n]]:
            nCorrect += 1.0
        print "Max acitivity unit (train Image set): ", np.argmax(arrOutputUnits), ". Correct label: ", arrLabels[arrTestImages[n]], "; \n"
    elif np.argmax(arrOutputUnits) == arrLabels[arrTestImages[n]]:
        nCorrect += 1.0
    print "Max acitivity unit: ", np.argmax(arrOutputUnits), ". Correct label: ", arrLabels[arrTestImages[n]], "; \n"

accuracy = (nCorrect / (setting.nTestImages)) * 100

print "\nAccuracy is at : ", accuracy, "%"


