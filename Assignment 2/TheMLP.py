import numpy as np
from tensorflow.keras.datasets import mnist
import math

class Neural_Network(object):
    def __init__(self, learningRate, inputSize, hiddenSize, outputSize):
        # Set parameters
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.learningRate = learningRate
        # Create weights
        self.inputToHiddenWeight = np.random.randn(self.inputSize,self.hiddenSize) / np.sqrt(self.inputSize/2)
        self.hiddenToOutputWeight = np.random.randn(self.hiddenSize,self.outputSize) / np.sqrt(self.hiddenSize/2)
        self.biasWeightih = np.random.randn(self.hiddenSize).reshape(1,self.hiddenSize)
        self.biasWeightho = np.random.randn(self.outputSize).reshape(1,self.outputSize)
        # Check
        self.correct = 0
        self.incorrect = 0

    def sigmoidFunction(self, value):
        return 1/(1 + np.exp(-value))
    
    def feedForward(self,input, desired):
        self.hiddenLayerNet = np.dot(input,self.inputToHiddenWeight) + self.biasWeightih# wixi + b for input to hidden layer
        self.hiddenLayerOut = np.asarray([self.sigmoidFunction(neuron) for neuron in self.hiddenLayerNet]) # sigmoid(wixi + b) for input to hidden
        self.outputLayerNet = np.dot(self.hiddenLayerOut, self.hiddenToOutputWeight) + self.biasWeightho
        self.outputLayerOut = np.asarray([self.sigmoidFunction(neuron) for neuron in self.outputLayerNet])

        self.hiddenLayerOut = self.hiddenLayerOut.reshape(1,64)

        if(desired != None):
            if(np.argmax(self.outputLayerOut) == desired):
                self.correct += 1
            else:
                self.incorrect += 1
    
    def backPropagation(self, inputLayer, desired):
        self.desired_output = np.zeros(10)
        self.desired_output[desired] = 1.0 # Represents desired output array [0,1,0,0,0,0,0,...]

        output_error = ( self.outputLayerOut - self.desired_output ) * self.outputLayerOut * ( 1 - self.outputLayerOut ).reshape(1,10)
        self.hiddenToOutputWeight = self.hiddenToOutputWeight - np.dot(self.hiddenLayerOut.T,output_error) * self.learningRate
        self.biasWeightho -= self.learningRate * output_error

        hidden_error = ( np.dot(output_error,self.hiddenToOutputWeight.T) * self.hiddenLayerOut * (1 - self.hiddenLayerOut) ).reshape(1,64)
        inputLayer = inputLayer.reshape(1,784)
        self.inputToHiddenWeight = self.inputToHiddenWeight - self.learningRate * np.dot(inputLayer.T, hidden_error)
        self.biasWeightih -= self.learningRate * hidden_error
    
    def Mean_Squared_Error(self,d,y):
        return np.mean((d - y)**2)
    
    def train(self, epoch):
        # Import = data
        (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
        trainInput  = np.asarray(X_train).astype(np.float32)
        trainOutput = np.asarray(Y_train).astype(np.int)
        trainInput  = trainInput / 255 # Normalize data

        for num in range(epoch):
            for i in range(len(trainInput)):
                nn.feedForward(trainInput[i].flatten(), None)
                nn.backPropagation(np.asarray(trainInput[i].flatten()),trainOutput[i])

                if i%400 == 0:
                    print('MSE : ',self.Mean_Squared_Error(self.desired_output, self.outputLayerOut))
    
    def test(self):
        (X_train,Y_train),(X_test,Y_test) = mnist.load_data()

        testInput  = np.asarray(X_test).astype(np.float32)
        testOutput = np.asarray(Y_test).astype(np.int)
        testInput  = testInput / 255 # Normalize data

        for i in range(len(testInput)):
            nn.feedForward(testInput[i].flatten(), testOutput[i])

        print('Correct : ', nn.correct)
        print('Incorrect : ', nn.incorrect)
        print('Percentage ', (nn.correct / len(testInput)) * 100, '%')

if __name__ == "__main__":

    nn = Neural_Network(0.07,784,64,10)
    nn.train(1)
    nn.test()
