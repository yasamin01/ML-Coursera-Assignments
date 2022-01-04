import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import optimize as opt

# Loading and Visualizing Data

print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex5data1.mat')
X = data['X'].reshape(-1)
y = data['y'].reshape(-1)
Xval = data['Xval'].reshape(-1)
yval = data['yval'].reshape(-1)
Xtest = data['Xtest'].reshape(-1)
ytest = data['ytest'].reshape(-1)
m = X.shape[0]

# Plot training data

plt.plot(X, y, 'rx', markersize=10, markeredgewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

input('Program paused. Press enter to continue.\n')


# Regularized Linear Regression Cost And Gradient


def linearRegCostFunction(X, y, theta, lambda_):
    m = len(y)
    h = np.matmul(X, theta) - y
    J = 1 / (2 * m) * np.matmul(h, h)
    J += lambda_ / (2 * m) * np.matmul(theta[1:], theta[1:])
    grad = 1 / m * np.matmul(X.transpose(), np.matmul(X, theta) - y)
    grad[1:] += lambda_ / m * theta[1:]
    return J, grad


theta = np.array([1, 1])
J, grad = linearRegCostFunction(np.column_stack([np.ones(m), X]), y, theta, 1)

print('Cost at theta = [1 ; 1]:\n(this value should be about 303.993192)\n', J)

input('Program paused. Press enter to continue.\n')

print('Gradient at theta = [1 ; 1]:\n(this value should be about [-15.303016; 598.250744])\n', '[',grad[0], grad[1],']')

input('Program paused. Press enter to continue.\n')

# Train Linear Regression


def trainLinearReg(X, y, lambda_):
    initial_theta = np.zeros(X.shape[1])
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)[0]
    gradFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)[1]

    options = {'maxiter': 200, 'disp': True}
    res = opt.minimize(costFunction, initial_theta, method='CG', jac=gradFunction, options=options)
    theta = res.x
    return theta


lambda_ = 0
theta = trainLinearReg(np.column_stack([np.ones(m),  X]), y, lambda_)

# Plot fit over the data

plt.plot(X, y, 'rx', markersize=10, markeredgewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, np.matmul(np.column_stack([np.ones(m), X]), theta), '--', linewidth=2)
plt.show()

input('Program paused. Press enter to continue.\n')

# Learning Curve for Linear Regression


def learningCurve(X, y, Xval, yval, lambda_):
    m = X.shape[0]
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    for i in range(m):
        theta = trainLinearReg(X[:i + 1], y[:i + 1], lambda_)
        error_train[i] = linearRegCostFunction(X[:i + 1], y[:i + 1], theta, 0)[0]
        error_val[i] = linearRegCostFunction(Xval, yval, theta, 0)[0]
    return error_train, error_val


lambda_ = 0
error_train, error_val = learningCurve(np.column_stack([np.ones(m), X]), y, np.column_stack([np.ones_like(Xval), Xval]), yval, lambda_)

plt.plot(np.arange(m), error_train)
plt.plot(np.arange(m), error_val)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])

print('# Training Examples\tTrain Error\tCross Validation Error\n')

for i in range(m):
    print(i, error_train[i], error_val[i])

input('Program paused. Press enter to continue.\n')

# Feature Mapping for Polynomial Regression


def polyFeatures(X, p):
    X_poly = np.zeros((X.shape[0], p))
    for i in range(p):
        X_poly[:, i] = X ** (i + 1)
    return X_poly


def featureNormalize(X):
    mu = X.mean(axis=0)
    X_norm = X - mu
    sigma = X_norm.std(axis=0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


p = 8
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.insert(X_poly, 0, 1, axis=1)

X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.insert(X_poly_test, 0, 1, axis=1)

X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.insert(X_poly_val, 0, 1, axis=1)

print('Normalized Training Example 1:')
print(X_poly[0, :])

input('\nProgram paused. Press enter to continue.\n')

# Learning Curve for Polynomial Regression


def plotFit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 15, max_x + 25.05, 0.05)
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma
    X_poly = np.insert(X_poly, 0, 1, axis=1)
    plt.plot(x, np.matmul(X_poly, theta), '--', linewidth=2)


lambda_ = 0
theta = trainLinearReg(X_poly, y, lambda_)

# Plot training data and fit

plt.figure()
plt.plot(X, y, 'rx', markersize=10, markeredgewidth=1.5)
plotFit(min(X), max(X), mu, sigma, theta, p);
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title(f'Polynomial Regression Fit ($\lambda$ = {lambda_:g})')
plt.show()

plt.figure()
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
plt.plot(np.arange(m), error_train)
plt.plot(np.arange(m), error_val)

plt.title(f'Polynomial Regression Learning Curve ($\lambda$ = {lambda_:g})')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend(['Train', 'Cross Validation'])

print(f'Polynomial Regression (lambda = {lambda_:g})\n')
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print(f'  \t{i}\t\t{error_train[i]:f}\t{error_val[i]:f}')

input('Program paused. Press enter to continue.\n')

# Validation for Selecting Lambda


def validationCurve(X, y, Xval, yval):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train = []
    error_val = []
    for lambda_ in lambda_vec:
        theta = trainLinearReg(X, y, lambda_)
        error_train.append(linearRegCostFunction(X, y, theta, 0)[0])
        error_val.append(linearRegCostFunction(Xval, yval, theta, 0)[0])

    return lambda_vec, error_train, error_val


lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

plt.plot(lambda_vec, error_train)
plt.plot(lambda_vec, error_val)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('$\lambda$')
plt.ylabel('Error')

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(f' {lambda_vec[i]:f}\t{error_train[i]:f}\t{error_val[i]:f}')


plt.title(f'Polynomial Regression Learning Curve ($\lambda$ = {lambda_:g})')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend(['Train', 'Cross Validation'])
plt.show()
