import numpy as np
from mnist import MNIST
from numpy.core.defchararray import array
from numpy.core.fromnumeric import size

class AMLP(object):

    INPUTNODE_NUM = 784
    HIDDENODE_NUM = 40
    OUTPUTNODE_NUM = 10

    def __init__(self, nRate, bRate, iter):
        self.learningRate = nRate 
        self.biasValue = bRate
        self.iteration = iter

    def takeData(self):
        mnist = MNIST('.\dataset\MNIST')
        x_train, y_train = mnist.load_training() #60000 samples
        x_test, y_test = mnist.load_testing()    #10000 samples

        self.trainInput = np.asarray(x_train).astype(np.float32)
        self.trainOutput = np.asarray(y_train).astype(np.int32)
        self.testInput = np.asarray(x_test).astype(np.float32)
        self.testOutput = np.asarray(y_test).astype(np.int32)
    
    def createWeights(self):
        self.layerOneWeights = np.random.rand(self.INPUTNODE_NUM,self.HIDDENODE_NUM)
        self.layerTwoWeights = np.random.rand(self.HIDDENODE_NUM,self.OUTPUTNODE_NUM)
        self.biasWeights = np.random.rand(2,1)

    def sigmoidFunction(self, value):
        return 1/(1 + np.exp(-value))
    
    
    def trainNN(self):
        print(1)
