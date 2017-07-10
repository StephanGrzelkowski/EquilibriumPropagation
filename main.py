import lib.helpers
import lib.inputUnits
import lib.hiddenLayer
import lib.connectionMatrix
import lib.outputLayer
import setting
import numpy as np

debug = 0

'''Building Framework'''
# load in MNIST Data
train_set, valid_set, test_set = lib.inputUnits.loadDataset()
arrLabels = train_set[1]
arrTrainImages = np.random.randint(0, len(train_set[0])-1, setting.nTrainImages)
#build layers:

#build connection matrices

'''Training phase'''
for n in xrange(setting.nTrainImages): #10 images to train initially
    print "\nTRAINING ON IMAGE: ", n

    """FREE PHASE"""
    arrInputUnits = train_set[0][arrTrainImages[n]] #lib.inputUnits.buildInputUnits(len(train_set[0][0])) # later just make the inputUnits the nth array of
    arrHiddenUnits = lib.hiddenLayer.buildHiddenLayer(setting.nHiddenUnits)
    arrOutputUnits = lib.outputLayer.buildOutputLayer()
    if n == 0:
        M1, M2 = lib.connectionMatrix.resetConnections(arrInputUnits, arrHiddenUnits, arrOutputUnits)


    for i in xrange(setting.settlingIterations):
        #update forward activation
        arrHiddenUnits = lib.helpers.updateActivation(M1,arrInputUnits,arrHiddenUnits)
        arrOutputUnits = lib.helpers.updateActivationLinear(M2,arrHiddenUnits, arrOutputUnits)
        error = lib.helpers.calcError(arrOutputUnits, arrLabels[arrTrainImages[n]])
        print "Squared error: ", error
        """" DEBUGGING """
        if debug == 1:
            print "\nIteration of forward activity #", i
            print "Average activity of the Hidden Units: ", np.mean(arrHiddenUnits)
            print "Standard deviation of the Hidden Units: ", np.std(arrHiddenUnits)
            print "Average Activity of the output Units: ", np.mean(arrOutputUnits)
            print "Standard deviation of the Output Units: ", np.std(arrOutputUnits)
        #update recurrent activation
        arrHiddenUnits = lib.helpers.updateRecurrentActivation(M2, arrOutputUnits, arrHiddenUnits)
        """ DEBUGGING """
        if debug == 1:
            print "\nIteration of recurrent activity #", i
            print "Average activity of the Hidden Units: ", np.mean(arrHiddenUnits)
            print "Standard deviation of the Hidden Units: ", np.std(arrHiddenUnits)



    print "\nSettling free phase done\n"

    print "\nCaluculating weight change of clamped phase."

    #reset weight change


    phase = -1

    U1 = lib.connectionMatrix.updateWeights(M1, arrInputUnits, arrHiddenUnits, phase)
    U2 = lib.connectionMatrix.updateWeights(M2, arrHiddenUnits, arrOutputUnits, phase)


    """DEBUGGING"""
    if debug == 1:
        print "\nWeight update after clamped phase"
        print "Updating connection Matrix 1. Average Activity:", np.mean(M1)
        print "Average Standard dev: ", np.std(M1)
        print "Updating connection Matrix 2. Average Activity:", np.mean(M2)
        print "Average Standard dev: ", np.std(M2)





    """CLAMPED PHASE"""

    arrOutputUnits = np.zeros(10) - 1
    arrOutputUnits[arrLabels[arrTrainImages[n]]] = 1

    for i in xrange(setting.settlingIterations):
        #update foward activation
        arrHiddenUnits = lib.helpers.updateActivation(M1,arrInputUnits,arrHiddenUnits)
        if debug == 1:
            print "\nIteration of forward activity #", i
            print "Average activity of the Hidden Units: ", np.mean(arrHiddenUnits)
            print "Standard deviation of the Hidden Units: ", np.std(arrHiddenUnits)

        #update recurrent activation
        arrHiddenUnits = lib.helpers.updateRecurrentActivation(M2, arrOutputUnits, arrHiddenUnits)
        if debug == 1:
            print "\nIteration of recurrent activity #", i
            print "Average activity of the Hidden Units: ", np.mean(arrHiddenUnits)
            print "Standard deviation of the Hidden Units: ", np.std(arrHiddenUnits)

    print "settling clamped phase done"

    #update weights after each phase

    phase = 1
    #U1 =  lib.connectionMatrix.updateWeights(M1, arrInputUnits, arrHiddenUnits, phase) - U1
    #U2 =  lib.connectionMatrix.updateWeights(M2, arrHiddenUnits, arrOutputUnits, phase) - U2

    print "\nUpdating the weights"
    U1 = U1 + lib.connectionMatrix.updateWeights(M1, arrInputUnits, arrHiddenUnits, phase)
    U2 = U2 + lib.connectionMatrix.updateWeights(M2, arrHiddenUnits, arrOutputUnits, phase)

    M1 = M1 + U1
    M2 = M2 + U2


    if debug == 1:
        print "\nWeight update after clamped phase"
        print "Updating connection Matrix 1. Average weight:", np.mean(M1)
        print "Standard deviation: ", np.std(M1)
        print "Updating connection Matrix 2. Average weight:", np.mean(M2)
        print "Standard dev: ", np.std(M2)

print "\nTraining phase done!\n"


'''Test the training scheme'''
nCorrect = 0.0
nTestImages = 10
arrTestImages = np.random.randint(0, len(test_set[0])-1, setting.nTestImages)
for n in xrange(setting.nTestImages): # test for 10 images
    # get a test image
    arrHiddenUnits = lib.hiddenLayer.buildHiddenLayer(setting.nHiddenUnits) #do these get set randomly again or do the reamin the same
    arrOutputUnits = lib.outputLayer.buildOutputLayer()
    arrInputUnits = test_set[0][arrTestImages[n]]
    for i in xrange(setting.settlingIterationsTest):
        # update forward activation
        arrHiddenUnits = lib.helpers.updateActivation(M1, arrInputUnits, arrHiddenUnits)
        arrOutputUnits = lib.helpers.updateActivationLinear(M2, arrHiddenUnits, arrOutputUnits)
        error = lib.helpers.calcError(arrOutputUnits, arrLabels[arrTrainImages[n]])
        print "Squared error: ", error

        if debug == 1:
            print "\nIteration of forward activity #", i
            print "Average activity of the Hidden Units: ", np.mean(arrHiddenUnits)
            print "Standard deviation of the Hidden Units: ", np.std(arrHiddenUnits)
            print "Average Activity of the output Units: ", np.mean(arrOutputUnits)
            print "Standard deviation of the Output Units: ", np.std(arrOutputUnits)

        # update recurrent activation
        arrHiddenUnits = lib.helpers.updateRecurrentActivation(M2, arrOutputUnits, arrHiddenUnits)

        if debug == 1:
            print "\nIteration of recurrent activity #", i
            print "Average activity of the Hidden Units: ", np.mean(arrHiddenUnits)
            print "Standard deviation of the Hidden Units: ", np.std(arrHiddenUnits)

    print "\nResponse to Image", n, arrOutputUnits
    if np.argmax(arrOutputUnits) == test_set[1][arrTestImages[n]]:
        nCorrect += 1.0
    print "Max acitivity unit: ", np.argmax(arrOutputUnits), ". Correct label: ", test_set[1][arrTestImages[n]], "; \n"

accuracy = (nCorrect / (setting.nTestImages)) * 100

print "\nAccuracy is at : ", accuracy, "%"


