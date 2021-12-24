import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Logistic Regression

# Load Data
from matplotlib import pyplot

data = pd.read_csv('ex2data1.csv')
X = data.iloc[:, :2]
y = data.iloc[:, 2]

# Plotting

"""print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

pos = np.where(y == 1)
neg = np.where(y == 0)
plt.scatter(X[pos, 0], X[pos, 1], marker='+')
plt.scatter(X[neg, 0], X[neg, 1], marker='o')
plt.title('Scatter plot of training data')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not admitted'])
plt.show()

input('\nProgram paused. Press enter to continue.\n')"""


# sigmoid function

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


g1 = sigmoid(0)
g2 = sigmoid(20)
g3 = sigmoid(-1)
A = np.array([[1, -5], [0, 9]])
g4 = sigmoid(A)
print(g1, ',', g2, ',', g3, ',', g4)


# COSTFUNCTION Compute cost and gradient for logistic regression

def costFunction(theta, X, y):
    m = len(y)
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(np.dot(X, theta))
    temp = y * np.transpose(np.log(h))
    temp2 = (1 - y) * np.transpose(np.log(1 - sigmoid(np.dot(X, theta))))
    J = (1 / m) * (-temp - temp2).sum()
    grad = (1 / m) * np.dot(np.transpose(X), (h - y))
    return J, grad


(m, n) = X.shape
X = np.hstack((np.ones((m, 1)), X))
initial_theta = np.zeros(n + 1)
cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros):\n', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

input('\nProgram paused. Press enter to continue.\n')

# Compute and display cost and gradient with non-zero theta

test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y)

print('\nCost at test theta: \n', cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('\nProgram paused. Press enter to continue.\n')

# Optimizing using fminunc

theta = opt.fmin_tnc(func=costFunction, x0=initial_theta, args=(X, y))
optimal_theta = theta[0]
cost, grad = costFunction(optimal_theta, X, y)
print('Cost at theta found by fminunc: \n', cost)
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(theta[0])
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

input('\nProgram paused. Press enter to continue.\n')

# Plotting Decision Boundary

input('\nProgram paused. Press enter to continue.\n')


# Prediction and Accuracies

def predict(theta, X):
    h = sigmoid(np.dot(X, theta))
    p = h >= 0.5
    return p


m = np.size((X, 1))
p = np.zeros((m, 1))
prob = sigmoid(np.dot([1, 45, 85], optimal_theta))
p = predict(optimal_theta, X)
print('For a student with scores 45 and 85, we predict an admission probability of \n', prob)
print('Expected value: 0.775 +/- 0.002\n\n')
print('Train Accuracy:\n', np.mean(p == y) * 100)
print('Expected accuracy (approx): 89.0\n')
print('\n')

input('\nProgram paused. Press enter to continue.\n')

# Regularized logistic regression

# Load Data
data = pd.read_csv('ex2data2.csv')
X = data.iloc[:, :2]
y = data.iloc[:, 2]

# Plotting

"""pos = y == 1
neg = y == 0
plt.scatter(X[pos, 0], X[pos, 1], marker='+')
plt.scatter(X[neg, 0], X[neg, 1], marker='o')
plt.title('Scatter plot of training data')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'])
plt.show()

input('\nProgram paused. Press enter to continue.\n')"""


# Feature mapping

def mapFeature(X1, X2):
    degree = 6
    out = np.ones((X1.shape[0], sum(range(degree + 2))))
    x = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out[:, x] = np.power(X1, i - j) * np.power(X2, j)
            x += 1

    return out


X1 = X.iloc[:, 0]
X2 = X.iloc[:, 1]
X = mapFeature(X1, X2)

# Compute and display initial cost and gradient for regularized logistic regression

def costFunctionReg(theta, X, y, lambda_):
    m = len(y)
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(np.dot(X, theta))
    temp = y * np.transpose(np.log(h))
    temp2 = (1 - y) * np.transpose(np.log(1 - sigmoid(np.dot(X, theta))))
    temp3 = (float(lambda_) / (2 * m)) * np.power(theta[1:theta.shape[0]], 2).sum()
    J = (1 / m) * (-temp - temp2).sum() + temp3
    grad = (1. / m) * np.dot(sigmoid(np.dot(X, theta)).T - y, X).T + (float(lambda_) / m) * theta
    grad_ = (1. / m) * np.dot(sigmoid(np.dot(X, theta)).T - y, X).T
    grad[0] = grad_[0]
    return J, grad


initial_theta = np.zeros(X.shape[1])
lambda_ = 1
cost, grad = costFunctionReg(initial_theta, X, y, lambda_)
print('Cost at initial theta (zeros):\n', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print(grad[:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

input('\nProgram paused. Press enter to continue.\n')

test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10):\n', cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(grad[:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

input('\nProgram paused. Press enter to continue.\n')

# Regularization and Accuracies

theta = opt.fmin_tnc(func=costFunctionReg, x0=initial_theta, args=(X, y, lambda_))
optimal_theta = theta[0]

# Plot Decision Boundary

"""if X.shape[1] <= 3:
    plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
    plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])
    p = plt.plot(plot_x, plot_y)
    plt.legend(['y = 1', 'y = 0', 'Decision Boundary'])
    plt.axis([30, 100, 30, 100])
else:
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = np.dot(mapFeature(np.array([u[i]]), np.array([v[j]])), theta)
    z = np.transpose(z)

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title('lambda = ', lambda_)"""

# Compute accuracy on our training set

p = predict(optimal_theta, X)
print('Train Accuracy:\n', np.mean(p == y) * 100)
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')
