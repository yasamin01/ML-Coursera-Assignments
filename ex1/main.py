import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# WARMUPEXERCISE

print('Running warmUpExercise ... ')
print('5x5 Identity Matrix: ')
def warmUpExercise(X):
    return np.eye(X)

A = warmUpExercise(5)
print(A)

input('Program paused. Press enter to continue.')

# PLOTTING THE DATA

print('Plotting Data ...')
data = pd.read_csv('ex1data1.csv')
X = data.iloc[:, 0]
y = data.iloc[:, 1]
m = len(y) #number of training examples


plt.scatter(X, y, marker='X', color='red')
plt.title('Scatter plot of training data')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

input('Program paused. Press enter to continue.\n')

# Cost and Gradient descent

X = X[:, np.newaxis]
y = y[:, np.newaxis]
X = np.hstack((np.ones((m, 1)), X)) #Add a column of ones to x

theta = np.zeros([2, 1])

iterations = 1500
alpha = 0.01

print('Testing the cost function ...')

def computeCost(X, y, theta):
    h = np.dot(X, theta) - y
    J = np.sum(np.power(h, 2) / (2*m))
    return J

J = computeCost(X, y, theta)
print('With theta = [0 ; 0]')
print('Cost computed =', J)

theta2 = np.array([[-1],[2]])
J = computeCost(X, y, theta2)
print('With theta = [-1 ; 2]')
print('Cost computed =', J)

input('Program paused. Press enter to continue.\n')

print('Running Gradient Descent ...')

def gradientDescent(X, y, theta, alpha, iterations):
    for num in range(iterations):
        Xj = np.dot(X, theta)-y
        Xj = np.dot(np.transpose(X), Xj)
        theta = theta - (alpha/m) * Xj
    return theta

theta = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent:')
print(theta)

J = computeCost(X, y, theta)
print(J)

# Plot the linear fit

plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], np.dot(X, theta))
plt.title('Training data with linear regression fit')
plt.show()

# Predict values for population sizes of 35,000 and 70,000

predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of\n ', predict1*10000)
predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of\n', predict2*10000)

input('Program paused. Press enter to continue.\n')

# Visualizing J(theta_0, theta_1)

print('Visualizing J(theta_0, theta_1)...')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

#Fill out J vals

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = computeCost(X, y, t)

J_vals = np.transpose(J_vals)


# Linear regression with multiple variables

print('Loading data ...')
data = pd.read_csv('ex1data2.csv')
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]
m = len(y)

print('First 10 examples from the dataset: ')
print(data.head(10))

input('Program paused. Press enter to continue.')

# Normalizing Features

print('Normalizing Features ...')
def featureNormalize(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


X_norm, mu, sigma = featureNormalize(X)

X = np.hstack((np.ones((m, 1)), X_norm))

def computeCostMulti(X, y, theta):
    h = np.dot(X, theta) - y
    J = np.sum(np.power(h, 2) / (2*m))
    return J

print('Running gradient descent ...')
alpha = 0.01
num_iters = 400

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    for num in range(num_iters):
        Xj = np.dot(X, theta)-y
        Xj = np.dot(np.transpose(X), Xj)
        theta = theta - (alpha/m) * Xj
    return theta

theta = np.zeros(3)
theta = gradientDescentMulti(X, y, theta, alpha, num_iters)
J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
print('Theta computed from gradient descent: ')
print(theta)

J = computeCostMulti(X, y, theta)
print(J)

# Plot the convergence graph

plt.plot(J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost J')
plt.title('Convergence of gradient descent with an appropriate learning rate')
plt.show()

# Estimate the price of a 1650 sq-ft, 3 br house

predict = featureNormalize(np.array([1650, 3]))[0]
predict = np.append(np.ones(1), predict)
price = np.dot(predict, theta)
print('Predicted price of a 1650 sq-ft, 3 br house ...\n', price)

input('Program paused. Press enter to continue.\n')

# Normal Equations

data = pd.read_csv('ex1data2.csv')
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]
m = len(y)

X = np.hstack((np.ones((m, 1)), X))

def normalEqn(X, y):
    return np.dot((np.linalg.inv(np.dot(np.transpose(X), X))), np.dot(np.transpose(X), y))

theta = normalEqn(X, y)

print('Solving with normal equations...')
print('Theta computed from the normal equations:\n', theta)

# Estimate the price of a 1650 sq-ft, 3 br house

predict = np.array([1, 1650, 3])
price = np.dot(predict, theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations)...\n', price)