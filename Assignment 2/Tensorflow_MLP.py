
from tensorflow import keras
import numpy as np

class Keras_MLP(object):
    def __init__(self):
        mnist = keras.datasets.mnist # input data
        (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
        self.trainInput  = np.asarray(X_train).astype(np.float32)
        self.trainOutput = np.asarray(Y_train).astype(np.int)
        self.trainInput  = self.trainInput / 255 # Normalize data
        self.testInput  = np.asarray(X_test).astype(np.float32)
        self.testOutput = np.asarray(Y_test).astype(np.int)
        self.testInput  = self.testInput / 255 # Normalize data

    def createNN(self, function_name, hidden_size, output_size, learning_rate, momentum_rate): # Creating neural network with parameters
        self.model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(hidden_size, activation=function_name), # hidden node size and which activation functions to use
        keras.layers.Dense(output_size, activation=function_name) ]) # output node size and which activation functions to use

        custom_optimizer = keras.optimizers.SGD(lr=learning_rate, momentum = momentum_rate) # learning rate and momentum rate
        self.model.compile(optimizer=custom_optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy']) # Compile neural network

    def trainNN(self, epoch):
        self.model.fit(self.trainInput, self.trainOutput, epochs = epoch) # Repeat train epoch amout 

    def testNN(self, model_name):
        test_loss, test_acc = self.model.evaluate(self.testInput, self.testOutput, verbose=2) # Evaluate neural network's performance
        print(model_name , '- Test Result : ', np.around(test_acc*100, 2), '%') # Print neural network's accuracy

if __name__ == "__main__":
    nn = Keras_MLP() # Model 2A
    nn.createNN('sigmoid', 64, 10, 0.07, 0.11)
    nn.trainNN(5)
    nn.testNN('Model 2A')

    kp = Keras_MLP() # Model 2B
    kp.createNN('sigmoid', 100, 10, 0.09, 0.11)
    kp.trainNN(10)
    kp.testNN('Model 2B')