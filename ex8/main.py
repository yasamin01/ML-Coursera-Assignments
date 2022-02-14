import numpy as np
from matplotlib import pyplot as plt
from scipy import io as sio


# Load Example Dataset

print('Visualizing example dataset for outlier detection.\n')

data = sio.loadmat('ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval'].reshape(-1)

#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bx')
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

input('Program paused. Press enter to continue.\n')

# Estimate the dataset statistics

print('Visualizing Gaussian fit.\n')


def estimateGaussian(X):
    m, n = X.shape
    mu = X.mean(axis=0)
    sigma2 = ((X - mu) ** 2).mean(axis=0)
    return mu, sigma2


#  Estimate my and sigma2
mu, sigma2 = estimateGaussian(X)


def multivariateGaussian(X, mu, Sigma2):
    k = len(mu)

    if len(Sigma2.shape) == 1:
        Sigma2 = np.diag(Sigma2)

    X = X - mu.reshape(1, -1)
    p = (2 * np.pi) ** (- k / 2) * np.linalg.det(Sigma2) ** (-0.5) * np.exp(
        -0.5 * np.diag(np.matmul(np.matmul(X, np.linalg.pinv(Sigma2)), X.transpose())))
    return p


#  Returns the density of the multivariate normal at each data point (row) of X
p = multivariateGaussian(X, mu, sigma2)

#  Visualize the fit


def visualizeFit(X, mu, sigma2):
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariateGaussian(np.column_stack([X1.reshape(-1), X2.reshape(-1)]), mu, sigma2)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')
    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, 10.0 ** np.arange(-20, 1, 3))


#  Visualize the fit
visualizeFit(X,  mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

input('Program paused. Press enter to continue.\n')

# Find Outliers


def selectThreshold(yval, pval):
    bestEpsilon = np.nan
    bestF1 = 0

    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval) + stepsize, max(pval) + stepsize, stepsize):
        predictions = (pval < epsilon).astype(int)
        precision = ((predictions == 1) & (yval == 1)).sum() / (predictions == 1).sum()
        recall = ((predictions == 1) & (yval == 1)).sum() / (yval == 1).sum()
        F1 = 2 * precision * recall / (precision + recall)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1


pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)
print(f'Best epsilon found using cross-validation: {epsilon:e}')
print(f'Best F1 on Cross Validation Set:  {F1:f}')
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)\n')

#  Find the outliers in the training set and plot the
outliers = p < epsilon

#  Draw a red circle around those outliers
plt.figure()
visualizeFit(X,  mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', linewidth=2, markersize=10)
plt.show()

input('Program paused. Press enter to continue.\n')

# Multidimensional Outliers

data = sio.loadmat('ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval'].reshape(-1)

#  Apply the same steps to the larger dataset
mu, sigma2 = estimateGaussian(X)

#  Apply the same steps to the larger dataset
p = multivariateGaussian(X, mu, sigma2)

#  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2)

#  Find the best threshold
epsilon, F1 = selectThreshold(yval, pval)

print(f'Best epsilon found using cross-validation: {epsilon:e}')
print(f'Best F1 on Cross Validation Set:  {F1:f}')
print('   (you should see a value epsilon of about 1.38e-18)')
print('   (you should see a Best F1 value of 0.615385)')
print(f'# Outliers found: {(p < epsilon).sum()}')

input('Program paused. Press enter to continue.\n')

# Loading movie ratings dataset

print('Loading movie ratings dataset.\n')

#  Load data
data = sio.loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i

#  From the matrix, we can compute statistics like average rating.
print(f'Average rating for movie 1 (Toy Story): {Y[0, R[0, :].astype(bool)].mean():f} / 5\n')

#  We can "visualize" the ratings matrix by plotting it with imagesc
plt.imshow(Y)
plt.ylabel('Movies')
plt.xlabel('Users')
plt.show()

input('\nProgram paused. Press enter to continue.\n')

# Collaborative Filtering Cost Function

data = sio.loadmat('ex8_movieParams.mat')
X = data['X']
Theta = data['Theta']
num_users = np.asscalar(data['num_users'])
num_movies = np.asscalar(data['num_movies'])
num_features = np.asscalar(data['num_features'])

#  Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

#  Evaluate cost function


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_):
    X = params[:num_movies * num_features].reshape((num_movies, num_features))
    Theta = params[num_movies * num_features:].reshape((num_users, num_features))

    h = np.matmul(X, Theta.transpose()) - Y
    J = 1 / 2 * (h ** 2 * R).sum()

    X_grad = np.matmul(h * R, Theta)
    Theta_grad = np.matmul((h * R).transpose(), X)

    J += lambda_ / 2 * ((Theta ** 2).sum() + (X ** 2).sum())

    X_grad += lambda_ * X
    Theta_grad += lambda_ * Theta

    grad = np.concatenate([X_grad.reshape(-1), Theta_grad.reshape(-1)])
    return J, grad


J, grad = cofiCostFunc(np.concatenate([X.reshape(-1), Theta.reshape(-1)]), Y, R, num_users, num_movies, num_features, 0)

print(f'Cost at loaded parameters: {J:f} '
      '\n(this value should be about 22.22)')

input('\nProgram paused. Press enter to continue.\n')

# Collaborative Filtering Gradient


def computeNumericalGradient(J, theta):
    numgrad = np.zeros_like(theta).reshape(-1)
    perturb = np.zeros_like(theta).reshape(-1)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = e
        loss1, _ = J(theta - perturb.reshape(theta.shape))
        loss2, _ = J(theta + perturb.reshape(theta.shape))
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad.reshape(theta.shape)


def checkCostFunction(lambda_ = 0):
    ## Create small problem
    X_t = np.random.random((4, 3))
    Theta_t = np.random.random((5, 3))

    # Zap out most entries
    Y = np.matmul(X_t, Theta_t.transpose())
    Y[np.random.random(Y.shape) > 0.5] = 0
    R = np.zeros_like(Y)
    R[Y != 0] = 1

    ## Run Gradient Checking
    X = np.random.randn(*X_t.shape)
    Theta = np.random.randn(*Theta_t.shape)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    numgrad = computeNumericalGradient(
        lambda t: cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lambda_),
        np.concatenate([X.reshape(-1), Theta.reshape(-1)]))

    cost, grad = cofiCostFunc(np.concatenate([X.reshape(-1), Theta.reshape(-1)]),
        Y, R, num_users, num_movies, num_features, lambda_)

    print(np.column_stack([numgrad, grad]))
    print(['The above two columns you get should be very similar.\n'
           '(Left-Your Numerical Gradient, Right-Analytical Gradient)'])

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your cost function implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          f'\nRelative Difference: {diff:g}')


print('Checking Gradients (without regularization) ... ')

#  Check gradients by running checkNNGradients
checkCostFunction()

input('\nProgram paused. Press enter to continue.\n')

# Collaborative Filtering Cost Regularization

J, grad = cofiCostFunc(np.concatenate([X.reshape(-1), Theta.reshape(-1)]), Y, R, num_users, num_movies, num_features, 1.5)

print(f'Cost at loaded parameters (lambda = 1.5): {J:f} '
      '\n(this value should be about 31.34)')

input('\nProgram paused. Press enter to continue.\n')

# Collaborative Filtering Gradient Regularization

print('Checking Gradients (with regularization) ... ')

#  Check gradients by running checkNNGradients
checkCostFunction(1.5)

input('\nProgram paused. Press enter to continue.\n')

# Entering ratings for a new user


def loadMovieList():
    ## Read the fixed movieulary list
    with open('movie_ids.txt', encoding='ISO-8859-1') as fid:
        # Store all movies in cell array movie{}
        n = 1682  # Total number of movies

        movieList = []
        for i in range(n):
            # Read line
            line = fid.readline()
            # Word Index (can ignore since it will be = i)
            idx, movieName = line.split(' ', maxsplit=1)
            # Actual Word
            movieList.append(movieName.strip())

    return movieList


movieList = loadMovieList()

#  Initialize my ratings
my_ratings = np.zeros(1682)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

print('New user ratings:')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Rated {my_ratings[i]:g} for {movieList[i]}')


# Loading movie ratings dataset

print('Loading movie ratings dataset.\n\n')

#  Load data
data = sio.loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i

#  Add our own ratings to the data matrix
Y = np.column_stack([my_ratings, Y])
R = np.column_stack([(my_ratings != 0).astype(int), R])

input('\nProgram paused. Press enter to continue.\n')

#  Normalize Ratings

def normalizeRatings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros_like(Y)
    for i in range(m):
        idx = R[i] == 1
        Ymean[i] = Y[i, idx].mean()
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean


Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.concatenate([X.reshape(-1), Theta.reshape(-1)])

# Set options for fmincg
options = {'disp': True, 'maxiter': None}

# Set Regularization
from scipy import optimize as opt
lambda_ = 10
res = opt.minimize(lambda t: cofiCostFunc(t, Ynorm, R, num_users, num_movies, num_features, lambda_)[0], initial_parameters,
                   method='CG', jac=lambda t: cofiCostFunc(t, Ynorm, R, num_users, num_movies, num_features, lambda_)[1], options=options)
theta = res.x

# Unfold the returned theta back into U and W
X = theta[:num_movies * num_features].reshape((num_movies, num_features))
Theta = theta[num_movies * num_features:].reshape((num_users, num_features))

print('Recommender system learning completed.')

# Recommendation for you

p = np.matmul(X, Theta.transpose())
my_predictions = p[:, 0] + Ymean

movieList = loadMovieList()

ix = my_predictions.argsort()[::-1]
print('Top recommendations for you:')
for i in range(10):
    j = ix[i]
    print(f'Predicting rating {my_predictions[j]:.1f} for movie {movieList[j]}')

print('\n\nOriginal ratings provided:')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Rated {my_ratings[i]:g} for {movieList[i]}')
