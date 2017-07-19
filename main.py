import lib.helpers
import lib.inputUnits
import lib.connectionMatrix
import setting
import numpy as np

debug = 0

'''Building Framework'''
# load in MNIST Data
train_set, valid_set, test_set = lib.inputUnits.loadDataset()
arrLabels = train_set[1]
arrTrainImages = np.random.permutation(len(train_set[0])-1)
arrTrainImages = arrTrainImages[0 : setting.nTrainImages]

#build connection matrices
M1 = setting.varWeights * np.random.randn(len(train_set[0][0]),setting.nHiddenUnits)
M2 = setting.varWeights * np.random.randn(setting.nHiddenUnits,10)

#Training phase
for k in xrange(setting.batchIterations):

    print "BATCH NUMBER: #",k
    for n in xrange(setting.nTrainImages):
        print "\nTRAINING ON IMAGE:", n,". Label: ", arrLabels[arrTrainImages[n]]

#Free phase
        #initialize layers
        arrInputUnits = train_set[0][arrTrainImages[n]] #lib.inputUnits.buildInputUnits(len(train_set[0][0])) # later just make the inputUnits the nth array of
        arrHiddenUnits = np.zeros(setting.nHiddenUnits) - setting.rest
        arrOutputUnits = np.zeros(10) - setting.rest

        #update activation of layers
        #for i in xrange(setting.settlingIterations):
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
        arrOutputUnits = np.zeros(10) - 1
        arrOutputUnits[arrLabels[arrTrainImages[n]]] = 1


        #for i in xrange(setting.settlingIterations):
        maxDeltaHid = 1
        setps = 0
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
arrTestImages = np.random.permutation(len(test_set[0])-1)
arrTestImages = arrTestImages[0:setting.nTestImages]
if setting.debugEql == 1:
    arrTestImages = arrTrainImages
for n in xrange(setting.nTestImages): # test for 10 images
    # get a test image
    arrHiddenUnits = np.zeros(setting.nHiddenUnits) - setting.rest
    arrOutputUnits = np.zeros(10) - setting.rest
    if setting.debugEql == 1:
        arrInputUnits = train_set[0][arrTestImages[n]]
    else:
        arrInputUnits = test_set[0][arrTestImages[n]]

    #for i in xrange(setting.settlingIterationsTest):
    maxDeltaHid = 1
    maxDeltaOut = 1
    setps = 0
    while (maxDeltaHid > setting.delta) & (maxDeltaOut > setting.delta):
        # update forward activation
        arrHiddenUnits, maxDeltaHid = lib.helpers.updateActivation(M1, M2,  arrInputUnits, arrHiddenUnits, arrOutputUnits)
        arrOutputUnits, maxDeltaOut = lib.helpers.updateActivationLinear(M2, arrHiddenUnits, arrOutputUnits)

        if setting.debugEql:
            errorDiff = lib.helpers.calcErrorDiff(arrOutputUnits, arrLabels[arrTrainImages[n]])
        else:
            errorDiff = lib.helpers.calcErrorDiff(arrOutputUnits, test_set[1][arrTestImages[n]])
        print "Squared Error off target: ", errorDiff
        steps += 1
    print "\nResponse to Image", n, arrOutputUnits
    if setting.debugEql == 1:
        if np.argmax(arrOutputUnits) == arrLabels[arrTestImages[n]]:
            nCorrect += 1.0
        print "Max acitivity unit (train Image set): ", np.argmax(arrOutputUnits), ". Correct label: ", arrLabels[arrTestImages[n]], "; \n"
    elif np.argmax(arrOutputUnits) == test_set[1][arrTestImages[n]]:
        nCorrect += 1.0
    print "Max acitivity unit: ", np.argmax(arrOutputUnits), ". Correct label: ", test_set[1][arrTestImages[n]], "; \n"

accuracy = (nCorrect / (setting.nTestImages)) * 100

print "\nAccuracy is at : ", accuracy, "%"


