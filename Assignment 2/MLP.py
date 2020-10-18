import numpy as np
from mnist import MNIST
import random

from numpy.core.fromnumeric import size

class MLP(object):

    INPUTNODE_NUM = 784
    HIDDENODE_NUM = 397
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
        rgen = np.random.RandomState(1)
        self.layerOneWeights = rgen.normal(loc=0.0, scale=0.01, size=0 + self.INPUTNODE_NUM * self.HIDDENODE_NUM) # Initializing all weights from input layer to hidden layer
        self.layerOneWeights = np.reshape(self.layerOneWeights, (-1, self.HIDDENODE_NUM))
        self.layerTwoWeights = rgen.normal(loc=0.0, scale=0.01, size=0 + self.HIDDENODE_NUM * self.OUTPUTNODE_NUM) # Initializing all weights from hidden layer to output layer
        self.layerTwoWeights = np.reshape(self.layerTwoWeights, (-1, self.OUTPUTNODE_NUM))
        self.biasWeights = rgen.normal(loc=0.0, scale=0.01, size=2)

    def sigmoidFunction(self, value):
        return 1/(1 + np.exp(-value))
    
    
    def trainNN(self):
        error = []
        output = []
        hiddenNodes = []
        activationFunction = 0

        #for x in range(np.size(self.trainInput)):

        for counter in range(self.HIDDENODE_NUM):
            for i in range(np.size(self.trainInput[0])):
                activationFunction += self.trainInput[0][i] * self.layerOneWeights[i][counter]
            hiddenNodes.append(self.sigmoidFunction(activationFunction) + self.biasWeights[0]*1)
            activationFunction = 0
    
        for counter in range(self.OUTPUTNODE_NUM):
            for i in range(np.size(hiddenNodes)):
                activationFunction += hiddenNodes[i] * self.layerTwoWeights[i][counter]
            output.append(self.sigmoidFunction(activationFunction) + self.biasWeights[1]*1)
            activationFunction = 0

        print(output.index(max(output)))
        print(self.trainOutput[0])
        for i in range(self.OUTPUTNODE_NUM):
            if(i != self.trainOutput[0]):
                error.append( (0 - output[i]) )
            else:
                error.append( (1 - output[i]) )
        print(output)
        print(error)
        
        self.adjustWeight(output,error,hiddenNodes)

    def adjustWeight(self, output, error, hiddenNodes):
        counter = 0
        for i in range(np.size(hiddenNodes)):
            for j in range(np.size(output)):
                self.layerTwoWeights[i][j] += self.learningRate * error[j] * output[j] * ( 1 - output[j] ) * hiddenNodes[i]
                counter+=1
        print(counter)