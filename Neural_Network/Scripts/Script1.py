import numpy as np

# Neural Network implementation

# Define Training examples

# Inputs
X = np.array([[0.5, 0.64, 0.4],
              [0.63, 0.39, 0.5],
              [0.52, 0.38, 0.43]])
X = X.T  # Shape (n0, m)

# Outputs
Y = np.array([0.7, 0.3, 0.48]).reshape(1, -1)  # Shape (n3, m)

# Number of examples
m = X.shape[1]

# NN Layers
n0 = X.shape[0]
n1 = 4 #vnodes
n2 = 3
n3 = 1

# Initialize weights and biases using He initialization
W1 = np.random.randn(n1, n0) * np.sqrt(2 / n0)
W2 = np.random.randn(n2, n1) * np.sqrt(2 / n1)
W3 = np.random.randn(n3, n2) * np.sqrt(2 / n2)

# Bias initialized normally since no problem starting all zeroes
B1 = np.zeros((n1, 1))
B2 = np.zeros((n2, 1))
B3 = np.zeros((n3, 1))

# Activation functions and their derivatives
def sigmoid(z): #USeful for output layer, so output is between 0 and 1.
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z): #More efficient training in the hidden layers
    return np.maximum(0, z)

def relu_derivative(a):
    return np.where(a > 0, 1, 0)

#Hyperparameters

epoch = 500  # Number of training rounds
alpha = 0.1  # Learning rate


#Model training

for i in range(epoch):

    # Forward propagation

    #Layer 1
    Z1 = np.dot(W1, X) + B1
    A1 = relu(Z1)

    #Layer 2
    Z2 = np.dot(W2, A1) + B2
    A2 = relu(Z2)

    #Layer 3
    Z3 = np.dot(W3, A2) + B3
    A3 = sigmoid(Z3)

    # Compute loss to report current epoch loss
    loss = -np.mean(Y * np.log(A3) + (1 - Y) * np.log(1 - A3))
    if i % 10 == 0:  # Print loss every 10 epochs
        print(f"Epoch {i+1}, Loss: {loss:.4f}")

    # Back propagation

    #Layer 3
    dA3 = A3 - Y  # Loss derivative with respect to A3 (output)
    dZ3 = dA3 * sigmoid_derivative(A3)
    dW3 = (1/m) * np.dot(dZ3, A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    #Layer 2
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * relu_derivative(A2)
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    #Layer 1
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    # Update weights and biases
    W3 -= alpha * dW3
    W2 -= alpha * dW2
    W1 -= alpha * dW1

    B3 -= alpha * db3
    B2 -= alpha * db2
    B1 -= alpha * db1