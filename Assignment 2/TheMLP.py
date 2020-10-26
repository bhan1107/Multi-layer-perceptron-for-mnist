from tensorflow.keras.datasets import mnist
import numpy as np
from operator import truediv

class Neural_Network(object):
    def __init__(self, learningRate, inputSize, hiddenSize, outputSize, momentum): # Initialize neural network
        # Set parameters
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.learningRate = learningRate
        self.momentum = momentum
        # Create weights
        self.inputToHiddenWeight = np.random.randn(self.inputSize,self.hiddenSize) / np.sqrt(self.inputSize/2) # Matrix that represents weights between input layer and hidden layer 
        self.hiddenToOutputWeight = np.random.randn(self.hiddenSize,self.outputSize) / np.sqrt(self.hiddenSize/2)
        self.biasWeightih = np.random.randn(self.hiddenSize).reshape(1,self.hiddenSize) # Matrix for bias weights
        self.biasWeightho = np.random.randn(self.outputSize).reshape(1,self.outputSize)
        self.weightDelta = np.zeros(self.outputSize) # To hold delta value for momentuum calculation 
        self.weightDelta2 = np.zeros(self.hiddenSize)
        # Variables for checking performance and confusion matrix
        self.correct = 0 # Performance check
        self.incorrect = 0
        self.train_outputs = [] # For confusion matrix
        self.test_outputs = [] # For confusion matrix
        self.MSE = 1

    def sigmoidFunction(self, value): # sigmoid function
        return 1/(1 + np.exp(-value))
    
    def feedForward(self,input, desired): # For feedforward, calculation of output
        self.hiddenLayerNet = np.dot(input,self.inputToHiddenWeight) + self.biasWeightih # wixi + b for input to hidden layer
        self.hiddenLayerOut = np.asarray([self.sigmoidFunction(neuron) for neuron in self.hiddenLayerNet]) # sigmoid(wixi + b) for input to hidden
        self.outputLayerNet = np.dot(self.hiddenLayerOut, self.hiddenToOutputWeight) + self.biasWeightho # wixi + b for hidden to output layer
        self.outputLayerOut = np.asarray([self.sigmoidFunction(neuron) for neuron in self.outputLayerNet]) # sigmoid(wixi + b) for hidden to output layer

        self.hiddenLayerOut = self.hiddenLayerOut.reshape(1,64)
        self.train_outputs.append(np.argmax(self.outputLayerOut)) # For confusion matrix 

        if(desired != None): # Performance check
            self.test_outputs.append(np.argmax(self.outputLayerOut))
            if(np.argmax(self.outputLayerOut) == desired): # If classification is correct += 1
                self.correct += 1
            else:
                self.incorrect += 1
    
    def backPropagation(self, inputLayer, desired): # For back propagation learnign
        self.desired_output = np.zeros(10)
        self.desired_output[desired] = 1.0 # Represents desired output in one-hot encoding format [0,1,0,0,0,0,0,0,0,0]

        output_error = ( self.outputLayerOut - self.desired_output ) * self.outputLayerOut * ( 1 - self.outputLayerOut ).reshape(1,10) # Calculation of output layer delta
        self.hiddenToOutputWeight = self.hiddenToOutputWeight - ( np.dot(self.hiddenLayerOut.T,output_error) * self.learningRate + ( self.weightDelta * self.momentum ) ) # calculation of change in hiden layer to output layer weight + moementum
        self.biasWeightho -= self.learningRate * output_error # Update on bias weight

        hidden_error = ( np.dot(output_error,self.hiddenToOutputWeight.T) * self.hiddenLayerOut * (1 - self.hiddenLayerOut) ).reshape(1,64) # Calculation of hidden layer delta
        inputLayer = inputLayer.reshape(1,784)
        self.inputToHiddenWeight = self.inputToHiddenWeight - ( self.learningRate * np.dot(inputLayer.T, hidden_error) + ( self.weightDelta2 * self.momentum ) ) # calculation of change in input layer to hidden layer weight + moementum
        self.biasWeightih -= self.learningRate * hidden_error # Update on bias weight

        self.weightDelta  = output_error # For momentum
        self.weightDelta2 = hidden_error # For momentum

    
    def Mean_Squared_Error(self,d,y): # For MSE calculation
        self.MSE = np.mean((d - y)**2)
        return self.MSE
    
    def train(self, epoch, X_train,Y_train): # Begin training
        trainInput  = np.asarray(X_train).astype(np.float32)
        trainOutput = np.asarray(Y_train).astype(np.int)
        trainInput  = trainInput / 255 # Normalize data

        for num in range(epoch) : # Runs for this many epoch
            self.train_outputs = [] # For confusion matrix
            self.test_outputs = [] # For confusion matrix
            for i in range(len(trainInput)): # Run training length of input data for mnist, 60000 times
                nn.feedForward(trainInput[i].flatten(), None)
                nn.backPropagation(np.asarray(trainInput[i].flatten()),trainOutput[i])

                if i%400 == 0:
                    if self.MSE < 0.0000005 : # If MSE is lower than threshold then stop trraining
                        break
                    print('MSE : ',self.Mean_Squared_Error(self.desired_output, self.outputLayerOut)) # Print MSE

     
    def test(self, X_test,Y_test): # Test neural network's performance
        testInput  = np.asarray(X_test).astype(np.float32)
        testOutput = np.asarray(Y_test).astype(np.int)
        testInput  = testInput / 255 # Normalize data

        for i in range(len(testInput)):
            nn.feedForward(testInput[i].flatten(), testOutput[i])

        print('Correct   : ', nn.correct)
        print('Incorrect : ', nn.incorrect)
        print('Accuracy  : ', (nn.correct / len(testInput)) * 100, '%')
    
if __name__ == "__main__":
    # Without printing confusion matrix
    (X_train,Y_train),(X_test,Y_test) = mnist.load_data() # Load data
    nn = Neural_Network(0.07,784,64,10, 0.11) # Initialize with learning rate, input layer size, hidden layer size, output layer size and momentum value
    nn.train(5,X_train,Y_train) # Train neural network with given number of epoch
    nn.test(X_test,Y_test) # Test and print result of test data set


    # With printing confusion matrix
    '''(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
    nn = Neural_Network(0.07,784,64,10, 0.11)
    nn.train(5,X_train,Y_train)
    con = tf.math.confusion_matrix(Y_train, nn.train_outputs)
    np.savetxt("train.csv", con, delimiter=",", fmt='%1.0f')
    print(con)
    tp = np.diag(con)
    prec = list(map(truediv, tp, np.sum(con, axis=0)))
    rec = list(map(truediv, tp, np.sum(con, axis=1)))
    print('Precision       : ',np.around(np.asarray(prec),4))
    print('Recall          : ',np.around(np.asarray(rec),4))
    print('Total Precision : ',np.sum(np.asarray(prec) / 10))
    print('Total Recall    : ',np.sum(np.asarray(rec) / 10))

    nn.test(X_test,Y_test)
    con = tf.math.confusion_matrix(Y_test, nn.test_outputs)
    np.savetxt("test.csv", con, delimiter=",", fmt='%1.0f')
    print(con)
    tp = np.diag(con)
    prec = list(map(truediv, tp, np.sum(con, axis=0)))
    rec = list(map(truediv, tp, np.sum(con, axis=1)))
    print('Precision : ',np.around(np.asarray(prec),4))
    print('Recall    : ',np.around(np.asarray(rec),4))
    print('Precision avg : ',np.sum(np.asarray(prec) / 10))
    print('Recall    avg : ',np.sum(np.asarray(rec) / 10))'''