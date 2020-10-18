from MLP import MLP

if __name__ == "__main__":
    mlp = MLP(1,2,3)
    mlp.takeData()
    mlp.createWeights()
    mlp.trainNN()