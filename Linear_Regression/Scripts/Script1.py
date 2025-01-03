import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load datasets
df_it_pop = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Data-Analysis\Human-Population-Analysis\Datasets\Italy_population.csv")
df_it_lf = pd.read_csv(r"C:\Users\Owner\ACSAI\Extra\Data-Analysis\Human-Population-Analysis\Datasets\Italy_life_expectancy.csv")


# Define functions for gradient descent
def update_w_b(time, temps, w, b, alpha):
    dw = 0.0
    db = 0.0
    N = len(time)

    for i in range(N):
        dw += -2 * time[i] * (temps[i] - ((time[i] * w) + b))
        db += -2 * (temps[i] - ((time[i] * w) + b))

    w = w - (1 / float(N)) * dw * alpha
    b = b - (1 / float(N)) * db * alpha

    return w, b


def train(time, temps, w, b, alpha, epochs):
    for e in range(epochs):
        w, b = update_w_b(time, temps, w, b, alpha)
        if e % 400 == 0:
            print("epoch: ", e, "loss: ", avg_loss(time, temps, w, b))
    return w, b


def avg_loss(time, temps, w, b):
    N = len(time)
    total_error = 0.0
    for i in range(N):
        total_error += (temps[i] - ((time[i] * w) + b)) ** 2
    return total_error / float(N)


def predict(w, b, time):
    return w * time + b


# Train model (Population as a function of life expectancy)

# Normalize the Age
age_min = df_it_lf["Age"].min()
age_max = df_it_lf["Age"].max()
df_it_lf["age_norm"] = df_it_lf["Age"].apply(lambda x: (x - age_min) / (age_max - age_min))

# Set training parameters
w = 0.5
b = 0.3
alpha = 0.002  #Learning rate

x = df_it_lf["age_norm"].values #Convert series into values
y = df_it_pop["Population"].values

# Train the model
w, b = train(x, y, w, b, alpha, 10000) #train for 10001 epochs

# Test prediction
to_predict = 70
pred_norm = (to_predict - age_min) / (age_max - age_min)
predicted_population = predict(w, b, pred_norm)


#Example model with sklearn library

model = LinearRegression()
x_shape = x.reshape(-1, 1)
model.fit(x_shape, y)
predicted = model.predict(x_shape)


# Plot

plt.figure(figsize=(8,8))

# Function with trained parameters
x_norm = df_it_lf["age_norm"].values
y_pred = w * x_norm + b #The best fit line must be created using the normalized values since the model was trained on normalized values

# Plot my best fit line
x_actual = df_it_lf["Age"].values #Corresponding, non normalized age values
plt.plot(x_actual, y_pred, label="Manual") #We plot the predicted values for each actual age, eventhough the predicted values are calculated from the normalized ages

#Plot sklearn best-fit line
plt.plot(x_actual, predicted, label="Sci-kit learn", color="green")

# Scatter plot of actual data
plt.scatter(df_it_lf["Age"], df_it_pop["Population"], color="red", label="Actual data")

# Plot details
plt.title("Sci-kit learn vs Manual Linear Regression (ITA life expectancy and population)")
plt.xlabel("Age")
plt.ylabel("Population (10 million)")
plt.legend()

#plt.savefig(r"C:\Users\Owner\ACSAI\Extra\Machine-Learning\Linear_Regression\plots\Sci-kit_vs_Manual.png")

plt.show()


