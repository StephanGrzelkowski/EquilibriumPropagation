debugActivityUpdate = 0
debugWeights = 0 #some debug prints to the console regarding the update rules of weight changes
debugEql = 1

nTrainImages = 10
nTestImages = 10
nHiddenUnits = 200  #number of hidden Units per Hidden layer

settlingIterations = 30 #number of settling operations during the training phase
settlingIterationsTest = 30 #number of settling operations during the test phase
batchIterations = 10

rest = 0 #activation value for initialization
#decay = 0.0001
lamb = 0.01  #
aMax = 1 #maximal activity
aMin = -1 #Minimal activity
epsilon = 0.001 #weight update step size

varWeights = 0.25 #standard variance of the weight initialization