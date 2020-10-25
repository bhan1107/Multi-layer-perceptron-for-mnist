
import tensorflow as tf
from tensorflow import keras
import numpy as np
from operator import truediv

class Keras_MLP(object):
    def __init__(self):
        mnist = keras.datasets.mnist
        (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
        self.trainInput  = np.asarray(X_train).astype(np.float32)
        self.trainOutput = np.asarray(Y_train).astype(np.int)
        self.trainInput  = self.trainInput / 255 # Normalize data
        self.testInput  = np.asarray(X_test).astype(np.float32)
        self.testOutput = np.asarray(Y_test).astype(np.int)
        self.testInput  = self.testInput / 255 # Normalize data
    def createNN(self):
        self.model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid') ])

        #custom_optimizer = keras.optimizers.Adam(lr=0.01)
        custom_optimizer = keras.optimizers.SGD(lr=0.09, momentum = 0.11)
        self.model.compile(optimizer=custom_optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    def trainNN(self):
        self.model.fit(self.trainInput, self.trainOutput, epochs = 10)
        pred = self.model.predict(self.trainInput)
        con = tf.math.confusion_matrix(self.trainOutput, np.argmax(pred,axis=1))
        print('Test : ',con)
        np.savetxt("train.csv", con, delimiter=",", fmt='%1.0f')
        tp = np.diag(con)
        prec = list(map(truediv, tp, np.sum(con, axis=0)))
        rec = list(map(truediv, tp, np.sum(con, axis=1)))
        print('Precision : ',np.around(np.asarray(prec),4))
        print('Recall    : ',np.around(np.asarray(rec),4))
        print('Precision avg : ',np.sum(np.asarray(prec) / 10))
        print('Recall    avg : ',np.sum(np.asarray(rec) / 10))



    def testNN(self):
        test_loss, test_acc = self.model.evaluate(self.testInput, self.testOutput, verbose=2)
        print('Test Result : ', test_acc)
        pred = self.model.predict(self.testInput)
        con = tf.math.confusion_matrix(self.testOutput, np.argmax(pred,axis=1))
        print(con)
        np.savetxt("test.csv", con, delimiter=",", fmt='%1.0f')
        tp = np.diag(con)
        prec = list(map(truediv, tp, np.sum(con, axis=0)))
        rec = list(map(truediv, tp, np.sum(con, axis=1)))
        print('Precision : ',np.around(np.asarray(prec),4))
        print('Recall    : ',np.around(np.asarray(rec),4))
        print('Precision avg : ',np.sum(np.asarray(prec) / 10))
        print('Recall    avg : ',np.sum(np.asarray(rec) / 10))


if __name__ == "__main__":
    kp = Keras_MLP()
    kp.createNN()
    kp.trainNN()
    kp.testNN()
