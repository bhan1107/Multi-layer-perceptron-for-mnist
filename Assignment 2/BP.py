import numpy as np

def sigmoidFunction(value):
        return 1.0 / (1.0 + np.exp(-value))

def BP():
    np.set_printoptions(precision=5)
    input = np.array([0.05, 0.1])
    weightOne = np.array([[0.15,0.25],[0.2,0.3]])
    weightTwo = np.array([[0.35,0.45],[0.4,0.5]])
    target = [0.01,0.99]

    hiddenLayer = np.asarray(weightOne * input.reshape(len(input),1))
    hiddenLayer = [sigmoidFunction(sum(column)) for column in hiddenLayer.T]

    output = np.asarray(weightTwo * hiddenLayer)
    output = [sigmoidFunction(sum(column)) for column in output.T]
    #print(output)

    Error = [ 0.5 * (a - b)*(a - b) for a,b in zip(target,output)]
    #print(sum(Error))

    #### Adjust weight
    for h in range(len(hiddenLayer)):
        propogate = 0
        for i in range(len(output)):
            ErrorOverOutput = (-1) * (target[i] - output[i])
            OutputOverNet = output[i] * (1 - output[i])
            NetOverWeight = hiddenLayer[h]
            #total = ErrorOverOutput * OutputOverNet * NetOverWeight
            total = ErrorOverOutput * OutputOverNet * weightTwo[h][i]
            weightTwo[h][i] -= 0.5 * total
            propogate += total
            #print(total)
        for j in range(len(input)):
            #print(propogate)
            #print(hiddenLayer[j] * (1 - hiddenLayer[j]))
            #print(hiddenLayer[j])
            weightOne[j][h] -= 0.5 * propogate * ( hiddenLayer[h] * (1 - hiddenLayer[h]) ) * input[j]
    print(weightTwo)
    print(weightOne)
    

if __name__ == "__main__":
    BP()