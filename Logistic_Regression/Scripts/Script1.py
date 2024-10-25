import numpy as np

#Implementing a Logistic Regression Model

#Dataset

# Feature matrix X (2 features, 5 samples)
X = np.array([[0.5, 1.0, 1.5, 2.0, 2.5],
              [1.0, 2.0, 3.0, 4.0, 5.0]])

#Number of examples in data set
data_set_length = X.shape[1]

# Output vector Y (binary outputs for 5 samples)
Y = np.array([0, 0, 1, 1, 1])

#Initialize the weight vector and scalar b
W = np.zeros((2, 1))
B = 0

#Define the sigmoid function to be used
def sigmoid(z):
    return 1/(1 + (np.exp(-z)))

#Define number of training iterations
epoch  = 1000

#Define learning rate
alpha = 0.01

#Model Training

def model_train(X, Y, W, B, data_set_length, alpha,  epoch):

    for i in range(epoch):

        #Compute the vector Z containing the linear function (true predicted value) z for every example
        Z = np.dot(W.transpose(), X) + B

        #Compute the vector A containing the sigmoid predicted value ([0,1]) for every example
        A = sigmoid(Z)

        #Compute the vector dZ containing the computed derivatives of z in respect to cost function for every example.
        dZ = A - Y

        #Compute the vector dw containing the mean on each derivative wi in respect to the cost function for every example
        dW = (1/data_set_length) * np.dot(X, dZ.transpose())

        #Compute the scalar dB representing the mean computed derivatives of b in respect to the cost function for every example
        dB = (1/data_set_length) * np.sum(dZ)

        #Update W and update B
        W = W - (alpha * dW)
        B = B - (alpha * dB)

        #Print progress as model is trained
        if i % 200 == 0:
            print(f"Current W {W} \n")
            print(f"Current B {B}\n \n")





model_train(X, Y, W, B, data_set_length, alpha,  epoch)

