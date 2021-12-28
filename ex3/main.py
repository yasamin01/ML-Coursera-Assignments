import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import optimize as opt

# Multi-class Classification


# Loading and Visualizing Data

input_layer_size = 400
num_labels = 10
print('Loading and Visualizing Data ...\n')
data = sio.loadmat('ex3data1.mat')
X = data['X']
y = data['y'].flatten()
m = X.shape[0]


def displayData(X, example_width=None):
    if example_width is None:
        example_width = int(round(np.sqrt(X.shape[1])))

    colormap = 'gray'
    m, n = X.shape
    example_height = int(n / example_width)

    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    pad = 1
    display_array = -np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))

    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            max_val = max(abs(X[curr_ex, :]))
            y = pad + j * (example_height + pad)
            x = pad + i * (example_width + pad)
            z = X[curr_ex, :].reshape((example_height, example_width), order='F') / max_val
            display_array[y:y + example_height, x:x + example_width] = z
            curr_ex += 1
        if curr_ex > m:
            break

    h = plt.imshow(display_array, vmin=-1, vmax=1, cmap=colormap)
    plt.axis('off')
    plt.show()
    return h, display_array


ran_indices = np.random.permutation(m)
sel = X[ran_indices[:100], :]
displayData(sel)

input('Program paused. Press enter to continue.\n')


# Vectorize Logistic Regression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lrCostFunction(theta, X, y, lambda_t):
    m = len(y)
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(np.dot(X, theta))
    one = y * np.transpose(np.log(h))
    two = (1 - y) * np.transpose(np.log(1 - h))
    reg = (lambda_t / (2 * m)) * np.power(theta[1:theta.shape[0]], 2).sum()
    J = -(1 / m) * (one + two).sum() + reg
    grad = (1 / m) * np.dot(h.T - y, X).T + (lambda_t / m) * theta
    grad_no = (1 / m) * np.dot(h.T - y, X).T
    grad[0] = grad_no[0]
    return J, grad


print('\nTesting lrCostFunction() with regularization')
theta_t = np.array([-2, -1, 1, 2])
X_t = np.c_[np.ones(5), np.arange(1, 16).reshape((3, 5)).T / 10]
y_t = np.array([1, 0, 1, 0, 1]) >= 0.5
lambda_t = 3
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('\nCost:\n', J)
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(grad)
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

input('Program paused. Press enter to continue.\n')


# Vectorizing regularized logistic regression

# One-vs-All Training

def oneVsAll(X, y, num_labels, lambda_t):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.insert(X, 0, 1, axis=1)
    initial_theta = np.zeros(n + 1)
    fun = lambda theta, y: lrCostFunction(theta, X, y, lambda_t)[0]
    jac = lambda theta, y: lrCostFunction(theta, X, y, lambda_t)[1]
    options = {'disp': True, 'maxiter': 400}

    for c in range(num_labels):
        # Run fmincg to obtain the optimal theta
        args = ((y == c + 1).astype(np.int),)
        res = opt.minimize(fun, initial_theta, args=args, method='CG', jac=jac, options=options)
        all_theta[c, :] = res.x
    return all_theta


print('\nTraining One-vs-All Logistic Regression...\n')
lambda_t = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_t)

input('Program paused. Press enter to continue.\n')


# Predict for One-Vs-All

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    X = np.c_[np.ones((m, 1)), X]
    prob = sigmoid(np.matmul(X, all_theta.transpose()))
    p = prob.argmax(axis=1) + 1
    return p


pred = predictOneVsAll(all_theta, X)

print('\nTraining Set Accuracy: \n', np.mean(np.double(pred == y)) * 100)

input('Program paused. Press enter to continue.\n')

# Neural Networks


# Loading and Visualizing Data

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

print('Loading and Visualizing Data ...\n')
data = sio.loadmat('ex3data1.mat')
X = data['X']
y = data['y'].flatten()
m = X.shape[0]

sel = np.random.permutation(m)
sel = sel[:100]

displayData(X[sel, :])

input('Program paused. Press enter to continue.\n')

# Loading Parameters

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
data = sio.loadmat('ex3weights.mat')
Theta1 = data['Theta1']
Theta2 = data['Theta2']
print(Theta1.shape, Theta2.shape)

input('Program paused. Press enter to continue.\n')

# Implement Predict

def predict(Theta1, Theta2, X):
    if X.ndim == 1:
        X = np.reshape(X, (-1, X.shape[0]))
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.zeros((m, 1))
    X = np.c_[np.ones((m, 1)), X]
    a2 = sigmoid(np.dot(X, Theta1.T))
    a2 = np.c_[np.ones((a2.shape[0], 1)), a2]
    a3 = sigmoid(np.dot(a2, Theta2.T))
    p = np.argmax(a3, axis=1) + 1
    return p


pred = predict(Theta1, Theta2, X)

print('\nTraining Set Accuracy:\n', np.mean(np.double(pred == y)) * 100)

input('Program paused. Press enter to continue.\n')

rp = np.random.permutation(m)

for i in range(m):
    print('\nDisplaying Example Image\n')
    displayData(X[rp[i], :].reshape((-1, input_layer_size)))

    pred = predict(Theta1, Theta2, X[rp[i], :].reshape((-1, input_layer_size)))
    print('\nNeural Network Prediction:\n', pred, np.mod(pred, 10))

    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break
