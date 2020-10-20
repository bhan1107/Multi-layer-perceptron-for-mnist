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
        self.trainInput = self.trainInput / np.linalg.norm(self.trainInput) # Normalize MNIST data for sigmoid function
        self.testInput = np.asarray(x_test).astype(np.float32)
        self.testInput = self.testInput / np.linalg.norm(self.testInput)

        self.trainOutput = np.asarray(y_train).astype(np.int32)
        self.testOutput = np.asarray(y_test).astype(np.int32)
    
    def createWeights(self):
        self.layerOneWeights = np.random.rand(self.INPUTNODE_NUM,self.HIDDENODE_NUM)
        self.layerTwoWeights = np.random.rand(self.HIDDENODE_NUM,self.OUTPUTNODE_NUM)
        self.biasWeights = np.random.rand(2,1)

    def sigmoidFunction(self, value):
        #return 1/(1 + np.exp(-value))
        return 1.0 / (1.0 + np.exp(-value))

    def normalizeData1(self, value):
        return ((value-max(value)) / (max(value)-min(value)) - 0.5 ) *2
    
    def transfer_derivative(self, output):
	    return output * (1.0 - output)
    
    def trainNN(self):
        for x in range(10):
            output = []

            dataWeightProduct = self.layerOneWeights * self.trainInput[x].reshape(self.INPUTNODE_NUM, 1)
            hiddenNodes = [sum(column) for column in zip(*dataWeightProduct)]
            hiddenNodes = np.asarray([self.sigmoidFunction(node + self.biasWeights[0]*self.biasValue) for node in hiddenNodes])

            tempArray = hiddenNodes * self.layerTwoWeights
            output = [sum(column)-2 for column in zip(*tempArray)]
            output = [np.asscalar(self.sigmoidFunction(node + self.biasWeights[1]*self.biasValue)) for node in output]

            self.adjustWeights(output, hiddenNodes, self.trainOutput[x], self.trainInput[x])
        #print(output)


    
    def adjustWeights(self, output, hiddenNodes, outputVal, input):
        error = np.zeros(10)
        error[outputVal] = 1.0 # Represents 
        #error = error - output * self.transfer_derivative(output)
        error = 0
        propogation = 0

        for h in range(len(hiddenNodes)):
            for j in range(len(output)):
                weightChange = pow(error[j] - output[j])
                #weightChange = error[j] * output[j] * (1 - output[j]) * hiddenNodes[h]
                self.layerTwoWeights[h][j] += self.learningRate * weightChange
                propogation += weightChange
                #print(weightChange)
            for i in range(len(input)):
                self.layerOneWeights[i][h] += self.learningRate *  propogation * hiddenNodes[h] * (1 - hiddenNodes[h]) * input[i]
            #print('The propogation is : ', propogation)
            propogation = 0

        #print(error)

    def BP(self):
        for x in range(100):
            hiddenLayer = np.asarray(self.layerOneWeights * self.trainInput[x].reshape(self.INPUTNODE_NUM,1))
            hiddenLayer = np.asarray([self.sigmoidFunction(sum(column)) for column in hiddenLayer.T])
            #print(hiddenLayer)

            output = np.asarray(self.layerTwoWeights * hiddenLayer.reshape(40,1))
            output = [self.sigmoidFunction(sum(column)) for column in output.T]
            print(output)

            target = np.zeros(10)
            target[self.trainOutput[x]] = 1.0 # Represents 
            Error = [ (a - b)*(a - b) for a,b in zip(target,output) ]

            #### Adjust weight
            for h in range(len(hiddenLayer)):
                propogate = 0
                for i in range(len(output)):
                    ErrorOverOutput = (-1) * (target[i] - output[i])
                    OutputOverNet = output[i] * (1 - output[i])
                    total = ErrorOverOutput * OutputOverNet * self.layerTwoWeights[h][i]
                    self.layerTwoWeights[h][i] -= self.learningRate * total
                    propogate += total
                for j in range(len(self.trainInput[x])):
                    self.layerOneWeights[j][h] -= self.learningRate * propogate * ( hiddenLayer[h] * (1 - hiddenLayer[h]) ) * self.trainInput[x][j]
            print(x)

    
    def testNN(self):
        correct = 0
        incorrect = 0
        for x in range(300):
            hiddenLayer = np.asarray(self.layerOneWeights * self.testInput[x].reshape(self.INPUTNODE_NUM,1))
            hiddenLayer = np.asarray([self.sigmoidFunction(sum(column)) for column in hiddenLayer.T])

            output = np.asarray(self.layerTwoWeights * hiddenLayer.reshape(40,1))
            output = [self.sigmoidFunction(sum(column)) for column in output.T]

            target = output.index(max(output))
            print('Target : ', target)
            print('Actual Output : ',self.testOutput[x])

            if(target == self.testOutput[x]):
                correct += 1
            else:
                incorrect +=1
        
        print('Total correct : ', correct)
        print('Total incorrect : ', incorrect)


