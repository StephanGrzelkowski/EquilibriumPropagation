import lib.helpers
import lib.inputUnits
import lib.hiddenLayer
import lib.connectionMatrix
import lib.outputLayer
import setting
import numpy as np

debug = 1

'''Settings: '''
#number of hidden Unit



'''Building Framework'''
# load in MNIST Data
train_set, valid_set, test_set = lib.inputUnits.loadDataset()
arrLabels = train_set[1]

#build layers:

#build connection matrices

'''Training phase'''
for n in xrange(10): #10 images to train initially
    print "\nTRAINING ON IMAGE: ", n

    """FREE PHASE"""
    arrInputUnits = train_set[0][n] #lib.inputUnits.buildInputUnits(len(train_set[0][0])) # later just make the inputUnits the nth array of
    arrHiddenUnits = lib.hiddenLayer.buildHiddenLayer(setting.nHiddenUnits)
    arrOutputUnits = lib.outputLayer.buildOutputLayer()
    if n == 0:
        M1, M2 = lib.connectionMatrix.resetConnections(arrInputUnits, arrHiddenUnits, arrOutputUnits)

    phase = -1
    for i in xrange(setting.settlingIterations):
        #update forward activation
        arrHiddenUnits = lib.helpers.updateActivation(M1,arrInputUnits,arrHiddenUnits)
        arrOutputUnits = lib.helpers.updateActivationLinear(M2,arrHiddenUnits, arrOutputUnits)


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

    #update weights after each phase
    M1 = lib.connectionMatrix.updateWeights(M1, arrInputUnits, arrHiddenUnits, phase)
    M2 = lib.connectionMatrix.updateWeights(M2, arrHiddenUnits, arrOutputUnits, phase)


    """DEBUGGING"""
    if debug == 1:
        print "\nWeight update after clamped phase"
        print "Updating connection Matrix 1. Average Activity:", np.mean(M1)
        print "Average Standard dev: ", np.std(M1)
        print "Updating connection Matrix 2. Average Activity:", np.mean(M2)
        print "Average Standard dev: ", np.std(M2)





    """CLAMPED PHASE"""
    phase = 1
    arrOutputUnits = np.zeros(10) - 1
    arrOutputUnits[arrLabels[n]] = 1

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
    M1 = lib.connectionMatrix.updateWeights(M1, arrInputUnits, arrHiddenUnits, phase)
    M2 = lib.connectionMatrix.updateWeights(M2, arrHiddenUnits, arrOutputUnits, phase)
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
for n in xrange(nTestImages): # test for 10 images
    # get a test image
    arrHiddenUnits = lib.hiddenLayer.buildHiddenLayer(setting.nHiddenUnits) #do these get set randomly again or do the reamin the same
    arrOutputUnits = lib.outputLayer.buildOutputLayer()
    arrInputUnits = test_set[0][n]
    for i in xrange(setting.settlingIterationsTest):
        # update forward activation
        arrHiddenUnits = lib.helpers.updateActivation(M1, arrInputUnits, arrHiddenUnits)
        arrOutputUnits = lib.helpers.updateActivationLinear(M2, arrHiddenUnits, arrOutputUnits)
        # update recurrent activation
        arrHiddenUnits = lib.helpers.updateRecurrentActivation(M2, arrOutputUnits, arrHiddenUnits)


    print "\nImage", n, arrOutputUnits
    if np.argmax(arrOutputUnits) == test_set[1][n]:
        nCorrect += 1.0
    print "Max acitivty unit: ", np.argmax(arrOutputUnits), ". Correct label: ", test_set[1][n]

accuracy = (nCorrect / (nTestImages)) * 100

print "\nAccuracy is at : ", accuracy, "%"


