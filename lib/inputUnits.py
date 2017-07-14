#get MNIST dataest as input
import cPickle
import gzip
import numpy as np

def loadDataset():

    # load MNISTdataset

    f = gzip.open('mnist.pkl.gz', 'rb')

    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return train_set, valid_set, test_set

