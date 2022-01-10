import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm
from nltk.stem.porter import PorterStemmer
import re

# Loading and Visualizing Data

print('Loading and Visualizing Data ...\n')
data = sio.loadmat('ex6data1.mat')
X = data['X']
y = data['y'].reshape(-1)


def plotData(X, y):
    pos = y == 1
    neg = y == 0
    plt.figure()
    plt.plot(X[pos, 0], X[pos, 1], 'k+', markeredgewidth=1, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)


plotData(X, y)

input('Program paused. Press enter to continue.\n')

# Training Linear SVM


def linearKernel(x1, x2):
    sim = np.matmul(x1, x2.transpose())  # dot product
    return sim


def visualizeBoundaryLinear(X, y, model):
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    xp, yp = np.meshgrid(xp, yp)
    P = model.decision_function(np.column_stack([xp.reshape(-1), yp.reshape(-1)])).reshape(xp.shape)
    plotData(X, y)
    plt.contour(xp, yp, P, '-b', levels=[0])
    plt.show()


print('\nTraining Linear SVM ...\n')
C = 1000
clf = svm.SVC(C=C, kernel=linearKernel, tol=1e-3, max_iter=-1)
model = clf.fit(X, y)
visualizeBoundaryLinear(X, y, model)

input('Program paused. Press enter to continue.\n')

# Implementing Gaussian Kernel


def gaussianKernel(x1, x2, sigma):
    m = x1.shape[0]
    n = x2.shape[0]
    sim = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d = x1[i] - x2[j]
            sim[i, j] = np.exp(-np.matmul(d, d) / (2 * sigma ** 2))

    return sim


print('Evaluating the Gaussian Kernel ...')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = np.asscalar(gaussianKernel(x1.reshape(1, -1), x2.reshape(1, -1), sigma))

print(f'Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {sigma:g} :'f'\n\t{sim:f}\n(for sigma = 2, this value should be about 0.324652)')


# Visualizing Dataset 2

print('Loading and Visualizing Data ...\n')
data = sio.loadmat('ex6data2.mat')
X = data['X']
y = data['y'].reshape(-1)

plotData(X, y)

input('Program paused. Press enter to continue.\n')

# Training SVM with RBF Kernel (Dataset 2)

print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

data = sio.loadmat('ex6data2.mat')
X = data['X']
y = data['y'].reshape(-1)


def visualizeBoundary(X, y, model, *args):
    plotData(X, y)
    x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = model.predict(np.column_stack([X1.reshape(-1), X2.reshape(-1)])).reshape(X1.shape)

    plt.contour(X1, X2, vals, 'b', levels=[0])
    plt.show()


C = 1
sigma = 0.1
import warnings
warnings.filterwarnings('ignore')
clf = svm.SVC(C=C, kernel=lambda x1, x2: gaussianKernel(x1, x2, sigma))
model = clf.fit(X, y)
visualizeBoundary(X, y, model)

# Visualizing Dataset 3

print('Loading and Visualizing Data ...')
data = sio.loadmat('ex6data3.mat')
X = data['X']
y = data['y'].reshape(-1)

plotData(X, y)

# Training SVM with RBF Kernel (Dataset 3)

data = sio.loadmat('ex6data3.mat')
X = data['X']
y = data['y'].reshape(-1)
Xval = data['Xval']
yval = data['yval'].reshape(-1)


def dataset3Params(X, y, Xval, yval):
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    max_accu = 0

    from sklearn import svm
    clf = svm.SVC()
    for C in values:
        clf.C = C
        for sigma in values:
            clf.kernel = lambda x1, x2: gaussianKernel(x1, x2, sigma)
            clf.fit(X, y)
            pred = clf.predict(Xval)
            accu = (pred == yval).mean()
            if accu > max_accu:
                max_accu = accu
                best_C = C
                best_sigma = sigma

    return best_C, best_sigma


C, sigma = dataset3Params(X, y, Xval, yval)

clf = svm.SVC(C=C, kernel=lambda x1, x2: gaussianKernel(x1, x2, sigma))
model = clf.fit(X, y)
visualizeBoundary(X, y, model)

input('Program paused. Press enter to continue.\n')


# Email Preprocessing

np.set_printoptions(precision=6)


def readFile(filename):
    with open(filename, 'r') as fid:
        file_content = fid.read()
    return file_content


def getVocabList():
    with open('vocab.txt') as fid:
        vocabList = re.findall(r'\d+\t(\w+)', fid.read())
    return vocabList


def processEmail(email_contents):
    # Load Vocabulary
    vocabList = getVocabList()

    # Init return value
    word_indices = []

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub(r'<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub(r'[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub(r'[$]+', 'dollar', email_contents)

    # Output the email to screen as well
    print('\n==== Processed Email ====\n')

    # Process file
    l = 0

    p = PorterStemmer()
    while email_contents:

        # Tokenize and also get rid of any punctuation
        str, email_contents = re.split(r'[ @\$/#\.-:&\*\+=\[\]\?!\(\)\{\},\'">_<;%\r\n]+', email_contents, maxsplit=1)

        # Remove any non alphanumeric characters
        str = re.sub(r'[^a-zA-Z0-9]', '', str)

        # Stem the word
        # (the porterStemmer sometimes has issues, so we use a try catch block)
        try:
            str = p.stem(str.strip())
        except:
            str = ''

        # Skip the word if it is too short
        if not str:
            continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        if str in vocabList:
            word_indices.append(vocabList.index(str))

        # Print to screen, ensuring that the output lines are not too long
        if (l + len(str) + 1) > 78:
            print()
            l = 0
        print(str, end=' ')
        l = l + len(str) + 1

    # Print footer
    print('\n\n=========================\n')
    return word_indices


print('\nPreprocessing sample email (emailSample1.txt)\n')

file_contents = readFile('emailSample1.txt')
word_indices = processEmail(file_contents)

print('Word Indices: \n')
print(f' {word_indices}')
print('\n\n')

input('Program paused. Press enter to continue.\n')

# Feature Extraction


def emailFeatures(word_indices):
    # Total number of words in the dictionary
    n = 1899
    x = np.zeros(n)
    for i in word_indices:
        x[i] = 1
    return x


print('\nExtracting features from sample email (emailSample1.txt)\n')

file_contents = readFile('emailSample1.txt')
word_indices = processEmail(file_contents)
features = emailFeatures(word_indices)

print(f'Length of feature vector:\n {len(features)}')
print(f'Number of non-zero entries:\n {sum(features > 0)}')

input('Program paused. Press enter to continue.\n')

# Train Linear SVM for Spam Classification

data = sio.loadmat('spamTrain.mat')
X = data['X']
y = data['y'].reshape(-1)

print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')

C = 0.1
model = svm.SVC(C=C, kernel='linear')
model.fit(X, y)

p = model.predict(X)

print('Training Accuracy: %f\n', np.mean(np.double(p == y)) * 100)

# Test Spam Classification

data = sio.loadmat('spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest'].reshape(-1)

print('\nEvaluating the trained Linear SVM on a test set ...\n')

p = model.predict(Xtest)

print('Test Accuracy: %f\n', np.mean(np.double(p == ytest)) * 100)

# Top Predictors of Spam

weight = np.sort(model.coef_[0])[::-1]
idx = model.coef_[0].argsort()[::-1]
vocabList = getVocabList()

print('\nTop predictors of spam: \n')
for i in range(15):
    print(f' {vocabList[idx[i]]} ({weight[i]:f}) ')

print('\n\n')
input('\nProgram paused. Press enter to continue.\n')

# Try Your Own Emails

filenames = ['spamSample1.txt', 'spamSample2.txt', 'emailSample1.txt', 'emailSample2.txt']

for filename in filenames:
    file_contents = readFile(filename)
    word_indices = processEmail(file_contents)
    x = emailFeatures(word_indices)
    p = model.predict(x.reshape(1, -1))

    print(f'Processed {filename}\n\nSpam Classification: {p}')
    print('(1 indicates spam, 0 indicates not spam)')
