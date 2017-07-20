debugActivityUpdate = 0
debugWeights = 0 #some debug prints to the console regarding the update rules of weight changes
debugEql = 0

nTrainImages = 90           #number of training images
nTestImages = 30            #number of Test images
nHiddenUnits = 10           #number of hidden Units per Hidden layer
nOutputUnits = 3            #Number of output Units
nInputUnits = 4             #number of input units

settlingIterations = 30     #number of settling operations during the training phase
settlingIterationsTest = 30 #number of settling operations during the test phase
batchIterations = 100

rest = 0                    #activation value for initialization
#decay = 0.0001
lamb = 0.5  #
aMax = 1                    #maximal activity
aMin = -1                   #Minimal activity
epsilon = 0.1               #weight update step size
delta = 0.0001              #allowed change in settling iterations before equilibrium is reached

varWeights = 0.25           #standard variance of the weight initialization