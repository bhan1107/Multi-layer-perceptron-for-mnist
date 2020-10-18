from MLP import MLP

if __name__ == "__main__":
    mlp = MLP(0.01,1,3)
    mlp.takeData()
    mlp.createWeights()
    mlp.trainNN()
    mlp.testNN()