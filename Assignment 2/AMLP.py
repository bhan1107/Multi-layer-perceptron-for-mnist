import numpy as np
from mnist import MNIST
from numpy.core.defchararray import array
from numpy.core.fromnumeric import size
from numpy.lib.histograms import histogramdd
from sklearn.preprocessing import normalize
import math

np.set_printoptions(precision=5)


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
        #self.trainInput = self.trainInput / np.linalg.norm(self.trainInput) # Normalize MNIST data for sigmoid function
        self.trainInput = self.trainInput / 256
        self.testInput = np.asarray(x_test).astype(np.float32)
        #self.testInput = self.testInput / np.linalg.norm(self.testInput)
        self.testInput = self.testInput / 256

        self.trainOutput = np.asarray(y_train).astype(np.int32)
        self.testOutput = np.asarray(y_test).astype(np.int32)
    
    def createWeights(self):
        #self.layerOneWeights = np.random.rand(self.INPUTNODE_NUM,self.HIDDENODE_NUM)
        #self.layerTwoWeights = np.random.rand(self.HIDDENODE_NUM,self.OUTPUTNODE_NUM)
        self.inputToHiddenWeight = np.random.randn(self.INPUTNODE_NUM,self.HIDDENODE_NUM) / np.sqrt(self.INPUTNODE_NUM/2)
        self.hiddenToOutputWeight = np.random.randn(self.HIDDENODE_NUM,self.HIDDENODE_NUM) / np.sqrt(self.HIDDENODE_NUM/2)
        self.biasWeights = np.random.rand(2,1)

    def sigmoidFunction(self, value):
        return 1.0 / (1.0 + np.exp(-value))
    
    def transfer_derivative(self, output):
	    return output * (1.0 - output)
    
    def BP(self):
        for x in range(100):
            #hiddenLayer = np.asarray(self.layerOneWeights * self.trainInput[x].reshape(self.INPUTNODE_NUM,1))
            hiddenLayer = self.trainInput[x].dot(self.layerOneWeights)
            hiddenLayer = np.asarray([self.sigmoidFunction(np.sum(column)) for column in hiddenLayer.T])

            #output = np.asarray(self.layerTwoWeights * hiddenLayer.reshape(40,1))
            output = hiddenLayer.dot(self.layerTwoWeights)
            #print(output)
            output = [self.sigmoidFunction(np.sum(column)) for column in output.T]

            target = np.zeros(10)
            target[self.trainOutput[x]] = 1.0 # Represents desired output array [0,1,0,0,0,0,0,...]
            Error = [ (a - b)*(a - b) for a,b in zip(target,output) ]
            print('Actual output : ', sum(Error))

            #### Adjust weight ####
            for h in range(len(hiddenLayer)):
                propogate = 0
                for i in range(len(output)):
                    ErrorOverOutput = (-1) * (target[i] - output[i])
                    OutputOverNet = self.sigmoidFunction(output[i]) * (1 - self.sigmoidFunction(output[i]))
                    total = ErrorOverOutput * OutputOverNet * hiddenLayer[h]
                    self.layerTwoWeights[h][i] = self.layerTwoWeights[h][i] - (self.learningRate * total)
                    propogate += total
                for j in range(len(self.trainInput[x])):
                    self.layerOneWeights[j][h] = self.layerOneWeights[j][h] + (self.learningRate * propogate * ( hiddenLayer[h] * (1 - hiddenLayer[h]) ) * self.trainInput[x][j])
            #print(x)
        #self.testNN()

    
    def testNN(self):
        correct = 0
        incorrect = 0
        for x in range(3):
            #hidden = np.asarray(self.layerOneWeights * self.testInput[x].reshape(self.INPUTNODE_NUM,1))
            hidden = self.testInput[x].dot(self.layerOneWeights)
            #hidden = np.asarray([self.sigmoidFunction(np.sum(column)) for column in hidden.T])

            #output = np.asarray(self.layerTwoWeights * hidden.reshape(40,1))
            output = hidden.dot(self.layerTwoWeights)
            #output = [self.sigmoidFunction(np.sum(column)) for column in output.T]
            print(output)
            #print(type(output))

            target = np.argmax(output)
            print('Actual Output : ', target)
            #print('Desired Output : ',self.testOutput[x])

            #if(target == self.testOutput[x]):
            #    correct += 1
            #else:
            #    incorrect +=1
        
        print('Total correct : ', correct)
        print('Total incorrect : ', incorrect)
