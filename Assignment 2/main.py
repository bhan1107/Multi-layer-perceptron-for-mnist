from MLP import MLP
from AMLP import AMLP
from MLPP import MLPP


if __name__ == "__main__":
    '''mlp = MLP(0.01,1,3)
    mlp.takeData()
    mlp.createWeights()
    mlp.trainNN()
    mlp.testNN()'''

    mlp = MLPP(0.1,1,3)
    mlp.takeData()
    mlp.createWeights()
    mlp.BP()
    mlp.testNN()
    #mlp.trainNN()
    #mlp.testNN()