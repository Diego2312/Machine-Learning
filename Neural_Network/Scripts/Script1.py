import numpy as np

#Neural Network implementation


#Start by defining training examples

X = np.array([30, 35, 32])
X = X.reshape(-1, 1)


#NN L layers, (4, 3, 1)

layers = 4
n0 = X.shape #Layer 0 is the input layer
n1 = 4
n2 = 3
n3 = 1 #Last layer is the output layer

