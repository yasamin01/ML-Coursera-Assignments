import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import optimize as opt

# Loading and Visualizing Data

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

print('Loading and Visualizing Data ...\n')
data = sio.loadmat('ex4data1.mat')
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


sel = np.random.permutation(m)
sel = sel[:100]

displayData(X[sel, :])

input('Program paused. Press enter to continue.\n')

# Loading Parameters

print('\nLoading Saved Neural Network Parameters ...\n')
data = sio.loadmat('ex4weights.mat')
Theta1 = data['Theta1']
Theta2 = data['Theta2']
nn_params = np.concatenate([Theta1.reshape(-1), Theta2.reshape(-1)])


# Compute Cost (Feedforward)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))
    m = X.shape[0]
    J = 0
    X = np.c_[np.ones((m, 1)), X]
    a2 = sigmoid(np.matmul(X, Theta1.T))
    a2 = np.c_[np.ones((a2.shape[0], 1)), a2]
    a3 = sigmoid(np.matmul(a2, Theta2.T))
    y_one = np.zeros_like(a3)
    for i in range(m):
        y_one[i, y[i] - 1] = 1

    ones = np.ones_like(a3)
    A = np.matmul(y_one.T, np.log(a3)) + np.matmul((ones - y_one).T, np.log(ones - a3))
    J = -1 / m * A.trace()
    J += lambda_ / (2 * m) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))

    delta3 = a3 - y_one
    delta2 = np.matmul(delta3, Theta2[:, 1:]) * sigmoidGradient(np.matmul(X, Theta1.T))
    Theta2_grad = np.matmul(a2.T, delta3).T
    Theta1_grad = np.matmul(X.T, delta2).T

    Theta1_grad[:, 1:] += lambda_ * Theta1[:, 1:]
    Theta2_grad[:, 1:] += lambda_ * Theta2[:, 1:]
    Theta1_grad /= m
    Theta2_grad /= m
    grad = np.concatenate([Theta1_grad.reshape(-1), Theta2_grad.reshape(-1)])

    return J, grad


print('\nFeedforward Using Neural Network ...\n')

lambda_ = 0

J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights): \n(this value should be about 0.287629)\n', J)

input('\nProgram paused. Press enter to continue.\n')

# Implement Regularization

print('\nChecking Cost Function (w/ Regularization) ... \n')

lambda_ = 1

J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights):\n(this value should be about 0.383770)\n', J)

input('Program paused. Press enter to continue.\n')

# Sigmoid Gradient

print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
print(g)
print('\n\n')

input('Program paused. Press enter to continue.\n')


# Initializing Parameters


def randInitializeWeights(L_in, L_out):
    epsilon_init = np.sqrt(6 / (L_in + L_out))
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W


print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

initial_nn_params = np.concatenate([initial_Theta1.reshape(-1), initial_Theta2.reshape(-1)])


# Implement Backpropagation


def debugInitializeWeights(fan_out, fan_in):
    W = np.sin(np.arange(1, fan_out * (1 + fan_in) + 1)).reshape((fan_out, 1 + fan_in)) / 10
    return W


def computeNumericalGradient(J, theta):
    numgrad = np.zeros_like(theta).reshape(-1)
    perturb = np.zeros_like(theta).reshape(-1)
    e = 1e-4
    for p in range(theta.size):
        perturb[p] = e
        loss1, _ = J(theta - perturb.reshape(theta.shape))
        loss2, _ = J(theta + perturb.reshape(theta.shape))
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad.reshape(theta.shape)


def checkNNGradients(lambda_=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.arange(1, m + 1) % num_labels
    nn_params = np.concatenate([Theta1.reshape(-1), Theta2.reshape(-1)])
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)
    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)
    print(np.column_stack([numgrad, grad]))
    print('The above two columns you get should be very similar.\n'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          f'\nRelative Difference: {diff:g}')


print('\nChecking Backpropagation... \n')

checkNNGradients()

input('\nProgram paused. Press enter to continue.\n')

# Implement Regularization

print('\nChecking Backpropagation (w/ Regularization) ... \n')

lambda_ = 3
checkNNGradients(lambda_)

debug_J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)

print(
    '\n\nCost at (fixed) debugging parameters (w/ lambda): \n(for lambda = 3, this value should be about 0.576051)\n\n',
    debug_J[0])

input('Program paused. Press enter to continue.\n')

# Training NN

print('\nTraining Neural Network... \n')

options = {'disp': True, 'maxiter': 400}

lambda_ = 1

fun = lambda nn_params: nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)[0]
jac = lambda nn_params: nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)[1]

res = opt.minimize(fun, initial_nn_params, method='CG', jac=jac, options=options)
nn_params = res.x
cost = res.fun

Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

input('Program paused. Press enter to continue.\n')


# Visualize Weights

print('\nVisualizing Neural Network... \n')

displayData(Theta1[:, 1:])

input('\nProgram paused. Press enter to continue.\n')


# Implement Predict


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    X = np.insert(X, 0, 1, axis=1)
    h1 = sigmoid(np.matmul(X, Theta1.transpose()))
    h1 = np.insert(h1, 0, 1, axis=1)
    h2 = sigmoid(np.matmul(h1, Theta2.transpose()))
    p = h2.argmax(axis=1) + 1
    return p


pred = predict(Theta1, Theta2, X)

print('\nTraining Set Accuracy:\n', np.mean(np.double(pred == y)) * 100)
