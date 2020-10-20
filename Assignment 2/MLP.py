import numpy as np
from mnist import MNIST
import random
import math

from numpy.core.fromnumeric import size
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=5)
from matplotlib import pyplot as plt

class MLP(object):

    INPUTNODE_NUM = 784
    HIDDENODE_NUM = 50
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
        self.trainInput = self.trainInput / np.linalg.norm(self.trainInput)
        self.trainOutput = np.asarray(y_train).astype(np.int32)
        self.testInput = np.asarray(x_test).astype(np.float32)
        self.testOutput = np.asarray(y_test).astype(np.int32)
    
    def createWeights(self):
        rgen = np.random.RandomState(1)
        self.layerOneWeights = np.random.rand(self.INPUTNODE_NUM,self.HIDDENODE_NUM)
        self.layerTwoWeights = np.random.rand(self.HIDDENODE_NUM,self.OUTPUTNODE_NUM)
        self.biasWeights = rgen.normal(loc=0.0, scale=0.01, size=2)

    def sigmoidFunction(self, value):
        return 1/(1 + math.exp(-value))
        
    def normalizeData1(self, value):
        return ((value-max(value)) / (max(value)-min(value)) - 0.5 ) *2
    
    def normalizeData2(self, value):
        return value / np.sqrt(np.sum(value**2))
    
    def trainNN(self):
        for x in range(5):
            error = []
            output = []
            hiddenNodes = []
            activationFunction = 0
            q = []

            #self.trainInput[x] =  self.trainInput[x] / np.sqrt(np.sum(self.trainInput[x]**2))
            #self.trainInput[x] = self.normalizeData2(self.trainInput[x])
            #print(self.trainInput[x])
            a = self.layerOneWeights * self.trainInput[x].reshape((self.INPUTNODE_NUM,1))

            
            hiddenNodes = [sum(column) for column in zip(*a)]
            print('Hidden node 1 is : ', hiddenNodes)
            hiddenNodes = [self.sigmoidFunction(element + self.biasWeights[0]*1) for element in hiddenNodes]
            for element in hiddenNodes:
                q.append(self.sigmoidFunction(element + self.biasWeights[0]*1))
            print('Hidden node 2 is : ', hiddenNodes)
            print('Q is ', q)
            hiddenNodes = np.array(hiddenNodes)
            
            a = self.layerTwoWeights * hiddenNodes.reshape(self.HIDDENODE_NUM,1)
            output = [sum(column) for column in zip(*a)]
            output = [self.sigmoidFunction(element + self.biasWeights[1]*1) for element in output]
            print('Output : ', output)


            #print('Actual Output : ', output.index(max(output)))
            #print('Desired Output : ',self.trainOutput[x])
            
            if(output.index(max(output)) != self.trainOutput[x]):
                for i in range(self.OUTPUTNODE_NUM):
                    if(i != self.trainOutput[x]):
                        error.append( (0 - output[i]) )
                    else:
                        error.append( (1 - output[i]) )
                #print(error)
                self.adjustWeight(output,error,hiddenNodes)
            print(x)

    def adjustWeight(self, output, error, hiddenNodes):
        for i in range(np.size(hiddenNodes)):
            propogation = 0
            for j in range(np.size(output)):
                change = self.learningRate * error[j] * output[j] * ( 1 - output[j] ) * hiddenNodes[i]
                #print('Change in weight is : ', change)
                self.layerTwoWeights[i][j] += change
                propogation += change
            for h in range(np.size(self.trainInput[i])):
                self.layerOneWeights[h][i] += propogation * (1 - hiddenNodes[i]) * self.trainInput[i][h]
            #print(change)

    def testNN(self):
        self.correct = 0
        self.incorrect = 0

        for x in range(300):
            activationFunction = 0
            hiddenNodes = []
            output = []

            a = self.layerOneWeights * self.testInput[x].reshape((784,1))
            hiddenNodes = [sum(column) for column in zip(*a)]
            hiddenNodes = [self.sigmoidFunction(element) for element in hiddenNodes]
            hiddenNodes = np.array(hiddenNodes)
        
            for counter in range(self.OUTPUTNODE_NUM):
                for i in range(np.size(hiddenNodes)):
                    activationFunction += hiddenNodes[i] * self.layerTwoWeights[i][counter]
                output.append(self.sigmoidFunction(activationFunction) + self.biasWeights[1]*1)
                activationFunction = 0
            
            if(output.index(max(output)) == self.trainOutput[x]):
                self.correct += 1
            else:
                self.incorrect += 1
        
        print("Total correctness : ", self.correct)
        print("Total incorrectness : ", self.incorrect)