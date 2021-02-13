#!/usr/local/bin/python3
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_der(x):
    x = sigmoid(x);
    return x * (1 - x)

def printActivations(activations):
    for l in range(len(activations)):
        for j in range(len(activations[l])):
            print(activations[l][j])

def Z(activations, weights, bias, l, j):
    z = 0.0
    for k in range(len(activations[l - 1])):
        z += weights[l][j][k] * activations[l - 1][k] + bias[l][j]
    return z

def forward(activations, weights, bias):
    for l in range(1, len(activations)):
        for j in range(len(activations[l])):
            activations[l][j] = sigmoid(Z(activations, weights, bias, l, j))
#    return activations
'''
def Zsigmoid(activations, weights, bias, l):
    array = activations[l].copy()

    for j in range(len(array)):
        array[j] = sigmoid_der(Z(activations, weights, bias, l, j))
    return array

def NablaCost(activations, expectation, l):
    array = activations[l].copy()

    for j in range(len(array)):
        array[j] -= expectation[j]
    return array
'''
def Error(activations, expectation, weights, bias):
    errors = [[0 for l in range(len(activations[i]))] for i in range(len(activations))]
    errors_temp = errors.copy()

    l = len(errors) - 1
# compute last layer
#    nabla = NablaCost(activations, expectation, l)
#    sig_z = Zsigmoid(activations, weights, bias, l)
    cycle = 0

    for j in range(len(errors[l])):
        if cycle == 0:
            errors[l][j] = (activations[l][j] - expectation[j]) * sigmoid_der(Z(activations, weights, bias, l, j))
        else:
            errors[l][j] += (activations[l][j] - expectation[j]) * sigmoid_der(Z(activations, weights, bias, l, j))
# backward propagation of the error
    l -= 1

    while l != 0:
        for j in range(len(errors[l])):
            errors_temp[l][j] = 0.0
            sig = sigmoid_der(Z(activations, weights, bias, l, j))

        #    matrix = errors[l + 1].copy()
            for a in range(len(weights[l + 1])):
                delta_cost = 0.0
                for b in range(len(weights[l + 1][a])):
                    delta_cost += errors_temp[l + 1][a] * weights[l + 1][a][b]
                errors_temp[l][j] += delta_cost * sig

            if cycle == 0:
                errors[l][j] = errors_temp[l][j]
            else:
                errors[l][j] += errors_temp[l][j]
        l -= 1
    return errors

def backward(activations, errors, weights, bias):
    for l in range(1, len(activations)):
        for j in range(len(activations[l])):
            bias[l][j] -= errors[l][j]
            for k in range(len(weights[l][j])):
                weights[l][j][k] = weights[l][j][k] - errors[l][j] * activations[l - 1][k]

# n is the total number of training examples
def Cost(activations, expectation, n): # make it an Average c= 1/n * c(x)
    l = len(activations) - 1
    sum = 0.0

    for k in range(len(activations[l])):
        sum += math.pow(abs(expectation[k] - activations[l][k]), 2)
    return (1 / (2 * n)) * sum


'''
activations = [[0.7, 0.7, 0.3], [0, 0, 0, 0], [0, 0]]
weights = [[[]],[[0.3, 0.7, 0.7], [0.3, 0.7, 0.6], [0.3, 0.7, 0.4], [0.3, 0.7, 0.8]], [[0.3, 0.7, 0.2, 0.8], [0.3, 0.7, 0.2, 0.7]]]
bias = [[],[0.4, 0.2, 0.4, 0.2], [0.4, 0.2]] # -3 -> 3
expectation = [1.0, 1.0]
'''
activations = [[0.2],[0.3], [0]]
weights = [[[]],[[0.3]], [[0.3]]]
bias = [[],[0.2, 0.2], [0.2]]
expectation = [0.0]

forward(activations, weights, bias)
#printActivations(activations)
print(activations[len(activations) - 1])
#print("")
#print(errors)
print("-----------")

for x in range(10):
    errors = Error(activations, expectation, weights, bias)
#    print(errors)
    backward(activations, errors, weights, bias)
    forward(activations, weights, bias)
    #printActivations(activations)
    print(activations[len(activations) - 1])

#printActivations(activations)
#print(Cost(activations, expectation, 1))
